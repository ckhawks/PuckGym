"""
Puck RL Environment

Gymnasium environment that communicates with the Puck Unity game via shared memory.
The shared memory struct MUST match the C# RLBridge.SharedState exactly.
"""

import struct
import mmap
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class SharedMemoryConfig:
    """Configuration for shared memory communication. Must match C# exactly."""

    MEMORY_NAME = "PuckRL"
    MEMORY_SIZE = 4096

    # Struct format (must match C# SharedState with Pack=1)
    # < = little-endian
    # B = unsigned byte (uint8)
    # f = float (float32)
    FORMAT = "<" + "".join([
        "4B",   # obs_ready, action_ready, done, reset_flag
        "f",    # reward
        # Observation floats (16 total - matches PuckCapture v3)
        "7f",   # skater: x, z, y(height), vel_x, vel_z, vel_y, rotation
        "6f",   # puck: x, z, y(height), vel_x, vel_z, vel_y
        "3f",   # stick: x, z, y(height)
        # Extra state (not part of 16-obs)
        "4f",   # our_goal_x, our_goal_z, their_goal_x, their_goal_z
        "3f",   # game_time, our_score, their_score
        # Action floats and bytes
        "5f",   # move_x, move_y, aim_x, aim_y, blade_angle
        "3B",   # jump, crouch, boost
        # Config
        "f",    # time_scale (0 = use default)
    ])

    SIZE = struct.calcsize(FORMAT)

    # Indices into unpacked tuple
    OBS_READY = 0
    ACTION_READY = 1
    DONE = 2
    RESET_FLAG = 3
    REWARD = 4

    # Observation indices (16 floats, indices 5-20)
    # Skater: x, z, y, vel_x, vel_z, vel_y, rotation (7)    indices 5-11
    # Puck: x, z, y, vel_x, vel_z, vel_y (6)                indices 12-17
    # Stick: x, z, y (3)                                     indices 18-20
    # Goals: our_x, our_z, their_x, their_z (4)             indices 21-24 (not used in obs)
    # Game: time, our_score, their_score (3)                 indices 25-27 (not used in obs)
    OBS_START = 5
    OBS_COUNT = 16  # 16 observation values (matches PuckCapture v3)
    OBS_END = OBS_START + OBS_COUNT  # = 21

    # Action indices (after 4 bytes + 1 reward + 16 obs + 7 extra = 28 elements)
    MOVE_X = 28
    MOVE_Y = 29
    AIM_X = 30
    AIM_Y = 31
    BLADE_ANGLE = 32
    JUMP = 33
    CROUCH = 34
    BOOST = 35
    TIME_SCALE = 36


class PuckEnv(gym.Env):
    """
    Gymnasium environment for Puck RL training.

    Observation space: 16 floats (matches PuckCapture v3 format)
        [0] skater_x
        [1] skater_z (forward axis in Unity)
        [2] skater_y (height in Unity)
        [3] skater_vel_x
        [4] skater_vel_z (forward velocity)
        [5] skater_vel_y (vertical velocity)
        [6] skater_rotation
        [7] puck_x
        [8] puck_z (forward axis)
        [9] puck_y (height)
        [10] puck_vel_x
        [11] puck_vel_z (forward velocity)
        [12] puck_vel_y (vertical velocity)
        [13] stick_x
        [14] stick_z (forward axis)
        [15] stick_y (height)

    Action space: Box(8,)
        [0] move_x: -1 to 1 (turn left/right)
        [1] move_y: -1 to 1 (forward/back)
        [2] aim_x: -1 to 1 (stick vertical angle)
        [3] aim_y: -1 to 1 (stick horizontal angle)
        [4] blade_angle: -1 to 1 (blade rotation)
        [5] jump: 0 to 1 (thresholded at 0.5)
        [6] crouch: 0 to 1 (thresholded at 0.5)
        [7] boost: 0 to 1 (thresholded at 0.5)
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        normalize_obs: bool = True,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.timeout_seconds = timeout_seconds
        self.normalize_obs = normalize_obs
        self.render_mode = render_mode

        self.cfg = SharedMemoryConfig()
        self._mm: Optional[mmap.mmap] = None
        self._connected = False

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.OBS_COUNT,),
            dtype=np.float32
        )

        # Continuous action space
        # move_x, move_y, aim_x, aim_y, blade_angle are -1 to 1
        # jump, crouch, boost are 0 to 1 (will threshold at 0.5)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # Stats tracking
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._total_episodes = 0

        # NaN detection (physics explosion)
        self._nan_detected = False
        self._nan_count = 0

        # Time scale control (0 = let game use its own setting)
        self._time_scale = 0.0

    def _connect(self) -> bool:
        """Connect to shared memory. Returns True if successful."""
        if self._connected:
            return True

        try:
            # On Windows, use tagname parameter
            self._mm = mmap.mmap(-1, self.cfg.MEMORY_SIZE, tagname=self.cfg.MEMORY_NAME)
            self._connected = True
            print(f"[PuckEnv] Connected to shared memory '{self.cfg.MEMORY_NAME}'")
            print(f"[PuckEnv] Struct size: {self.cfg.SIZE} bytes")
            return True
        except Exception as e:
            print(f"[PuckEnv] Failed to connect to shared memory: {e}")
            print("[PuckEnv] Make sure the Unity game is running with the RL mod!")
            return False

    def _read_state(self) -> Tuple:
        """Read and unpack the shared state."""
        self._mm.seek(0)
        data = self._mm.read(self.cfg.SIZE)
        return struct.unpack(self.cfg.FORMAT, data)

    def _write_state(self, state: list):
        """Pack and write the shared state."""
        self._mm.seek(0)
        self._mm.write(struct.pack(self.cfg.FORMAT, *state))

    def _wait_for_obs(self) -> Optional[Tuple]:
        """Wait for C# to signal observation ready. Returns state or None on timeout."""
        start_time = time.perf_counter()
        spin_count = 0

        while True:
            state = self._read_state()
            if state[self.cfg.OBS_READY] == 1:
                return state

            spin_count += 1
            elapsed = time.perf_counter() - start_time

            if elapsed > self.timeout_seconds:
                return None

            # Small sleep every 1000 spins to not burn CPU
            if spin_count % 1000 == 0:
                time.sleep(0.0001)  # 0.1ms

    def _send_action(self, action: np.ndarray):
        """Send action to C# side."""
        state = list(self._read_state())

        # Set action values
        state[self.cfg.MOVE_X] = float(np.clip(action[0], -1, 1))
        state[self.cfg.MOVE_Y] = float(np.clip(action[1], -1, 1))
        state[self.cfg.AIM_X] = float(np.clip(action[2], -1, 1))
        state[self.cfg.AIM_Y] = float(np.clip(action[3], -1, 1))
        state[self.cfg.BLADE_ANGLE] = float(np.clip(action[4], -1, 1))
        state[self.cfg.JUMP] = 1 if action[5] > 0.5 else 0
        state[self.cfg.CROUCH] = 1 if action[6] > 0.5 else 0
        state[self.cfg.BOOST] = 1 if action[7] > 0.5 else 0

        # Set time scale
        state[self.cfg.TIME_SCALE] = self._time_scale

        # Signal action ready
        state[self.cfg.ACTION_READY] = 1

        self._write_state(state)

    def _signal_reset(self):
        """Signal C# to reset the episode."""
        state = list(self._read_state())
        state[self.cfg.RESET_FLAG] = 1
        self._write_state(state)

    def _extract_obs(self, state: Tuple) -> np.ndarray:
        """Extract observation array from state tuple."""
        obs = np.array(
            state[self.cfg.OBS_START:self.cfg.OBS_END],
            dtype=np.float32
        )

        # Check for NaN/Inf values (physics explosion in Unity)
        if np.any(~np.isfinite(obs)):
            print(f"[PuckEnv] WARNING: NaN/Inf detected in observation! Replacing with zeros.")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            self._nan_detected = True

        if self.normalize_obs:
            obs = self._normalize_observation(obs)

        # Clip to reasonable range after normalization
        obs = np.clip(obs, -10.0, 10.0)

        return obs

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations to roughly [-1, 1] range.
        Game field is ±50 units (world limit from SynchronizedObjectManager).

        16-obs layout:
        0: skater_x, 1: skater_z, 2: skater_y (height)
        3: skater_vel_x, 4: skater_vel_z, 5: skater_vel_y
        6: skater_rotation
        7: puck_x, 8: puck_z, 9: puck_y (height)
        10: puck_vel_x, 11: puck_vel_z, 12: puck_vel_y
        13: stick_x, 14: stick_z, 15: stick_y (height)
        """
        normalized = obs.copy()

        # Positions: world limit is ±50 units
        # Indices 0,1 (skater x,z), 7,8 (puck x,z), 13,14 (stick x,z)
        horiz_position_indices = [0, 1, 7, 8, 13, 14]
        for i in horiz_position_indices:
            normalized[i] = obs[i] / 50.0

        # Heights: typically 0-5 units
        # Indices 2 (skater y), 9 (puck y), 15 (stick y)
        height_indices = [2, 9, 15]
        for i in height_indices:
            normalized[i] = obs[i] / 5.0

        # Horizontal velocities: assume max ~20 units/sec
        # Indices 3,4 (skater vel x,z), 10,11 (puck vel x,z)
        horiz_vel_indices = [3, 4, 10, 11]
        for i in horiz_vel_indices:
            normalized[i] = obs[i] / 20.0

        # Vertical velocities: typically smaller
        # Indices 5 (skater vel y), 12 (puck vel y)
        vert_vel_indices = [5, 12]
        for i in vert_vel_indices:
            normalized[i] = obs[i] / 10.0

        # Rotation (index 6): already in radians, normalize to [-1, 1]
        normalized[6] = obs[6] / np.pi

        return normalized

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        # Connect if not already
        if not self._connect():
            raise RuntimeError("Could not connect to shared memory. Is the game running?")

        # Signal reset to Unity
        self._signal_reset()

        # Wait for first observation
        state = self._wait_for_obs()
        if state is None:
            raise TimeoutError("Timeout waiting for Unity after reset")

        # Send neutral action to complete handshake
        neutral_action = np.zeros(8, dtype=np.float32)
        self._send_action(neutral_action)

        # Reset stats
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._total_episodes += 1
        self._nan_detected = False  # Clear NaN flag for new episode

        obs = self._extract_obs(state)
        info = {"episode_num": self._total_episodes}

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Returns:
            observation, reward, terminated, truncated, info
        """
        if not self._connected:
            raise RuntimeError("Not connected to shared memory")

        # Send action
        self._send_action(action)

        # Wait for next observation
        state = self._wait_for_obs()
        if state is None:
            raise TimeoutError("Timeout waiting for Unity during step")

        # Extract data
        obs = self._extract_obs(state)
        reward = float(state[self.cfg.REWARD])
        terminated = state[self.cfg.DONE] == 1
        truncated = False

        # Check reward for NaN/Inf
        if not np.isfinite(reward):
            print(f"[PuckEnv] WARNING: NaN/Inf reward detected! Setting to 0.")
            reward = 0.0
            self._nan_detected = True

        # Force episode end if NaN was detected (physics exploded)
        if self._nan_detected:
            self._nan_count += 1
            print(f"[PuckEnv] Forcing episode termination due to NaN (total: {self._nan_count})")
            terminated = True

        # Update stats
        self._episode_steps += 1
        self._episode_reward += reward

        info = {
            "episode_steps": self._episode_steps,
            "episode_reward": self._episode_reward,
            "nan_detected": self._nan_detected,
        }

        if terminated:
            print(f"[PuckEnv] Episode {self._total_episodes} ended: "
                  f"{self._episode_steps} steps, reward: {self._episode_reward:.2f}")

        return obs, reward, terminated, truncated, info

    def set_time_scale(self, scale: float):
        """Set the game time scale. 0 = let game use its own setting."""
        self._time_scale = scale

    def close(self):
        """Clean up resources."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        self._connected = False
        print("[PuckEnv] Closed")


# =============================================================================
# TEST / DEBUG
# =============================================================================

def test_connection():
    """Test if we can connect to the shared memory."""
    print("Testing connection to Puck...")
    print("Make sure the game is running with the RL mod!")

    env = PuckEnv(timeout_seconds=10.0)

    try:
        obs, info = env.reset()
        print(f"Connected! Initial observation shape: {obs.shape}")
        print(f"Observation: {obs}")

        # Do a few random steps
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: reward={reward:.3f}, done={terminated}")

            if terminated:
                obs, info = env.reset()
                print("Episode reset!")

        print("Connection test successful!")

    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        env.close()


if __name__ == "__main__":
    test_connection()
