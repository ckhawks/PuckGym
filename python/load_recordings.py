"""
Load recorded demonstrations from PuckCapture binary files.

File format (per file):
- Header: 52 bytes
- Steps: N × 92 bytes each

Usage:
    from load_recordings import load_all_recordings
    obs, actions, rewards, metadata = load_all_recordings("path/to/recordings")
"""

import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class EpisodeMetadata:
    """Metadata for a single episode."""
    filename: str
    format_version: int
    team: int           # 0=Blue, 1=Red
    handedness: int     # 0=Right, 1=Left
    outcome: int        # 0=Goal, 1=Timeout, 2=Manual
    step_count: int
    duration: float
    timestamp: int      # Unix ms
    steam_id: str
    username: str


# Format version 3: 76 bytes header, 16 observations
HEADER_FORMAT_V3 = "<BBBBIfq24s32s"  # 76 bytes (same as v2)
HEADER_SIZE_V3 = struct.calcsize(HEADER_FORMAT_V3)
STEP_FORMAT_V3 = "<16f8ff"  # 16 obs + 8 action + 1 reward = 25 floats = 100 bytes
STEP_SIZE_V3 = struct.calcsize(STEP_FORMAT_V3)

# Format version 2: 76 bytes header with steam_id and username (legacy)
HEADER_FORMAT_V2 = "<BBBBIfq24s32s"  # 76 bytes
HEADER_SIZE_V2 = struct.calcsize(HEADER_FORMAT_V2)
STEP_FORMAT_V2 = "<14f8ff"  # 14 obs + 8 action + 1 reward = 23 floats = 92 bytes
STEP_SIZE_V2 = struct.calcsize(STEP_FORMAT_V2)

# Format version 1: 52 bytes header (legacy)
HEADER_FORMAT_V1 = "<BBBBIfq32s"  # 52 bytes
HEADER_SIZE_V1 = struct.calcsize(HEADER_FORMAT_V1)
STEP_FORMAT_V1 = "<14f8ff"  # 14 obs
STEP_SIZE_V1 = struct.calcsize(STEP_FORMAT_V1)


def load_episode(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, EpisodeMetadata]:
    """
    Load a single episode from a .bin file.

    Returns:
        obs: (N, 16) float32 array of observations (v3) or (N, 14) for older versions
        actions: (N, 8) float32 array of actions
        rewards: (N,) float32 array of rewards
        metadata: EpisodeMetadata object
    """
    with open(filepath, "rb") as f:
        # Peek at format version to determine header size
        version_byte = f.read(1)
        f.seek(0)
        format_version = version_byte[0]

        if format_version >= 3:
            # Version 3: 76 byte header, 16 observations
            header_data = f.read(HEADER_SIZE_V3)
            header = struct.unpack(HEADER_FORMAT_V3, header_data)
            steam_id = header[7].rstrip(b'\x00').decode('ascii', errors='ignore')
            username = header[8].rstrip(b'\x00').decode('ascii', errors='ignore')
            step_format = STEP_FORMAT_V3
            step_size = STEP_SIZE_V3
            num_obs = 16
        elif format_version >= 2:
            # Version 2: 76 byte header with steam_id and username, 14 observations
            header_data = f.read(HEADER_SIZE_V2)
            header = struct.unpack(HEADER_FORMAT_V2, header_data)
            steam_id = header[7].rstrip(b'\x00').decode('ascii', errors='ignore')
            username = header[8].rstrip(b'\x00').decode('ascii', errors='ignore')
            step_format = STEP_FORMAT_V2
            step_size = STEP_SIZE_V2
            num_obs = 14
        else:
            # Version 1: 52 byte header with player_id
            header_data = f.read(HEADER_SIZE_V1)
            header = struct.unpack(HEADER_FORMAT_V1, header_data)
            player_id = header[7].rstrip(b'\x00').decode('ascii', errors='ignore')
            steam_id = player_id  # Use player_id as steam_id for v1
            username = player_id
            step_format = STEP_FORMAT_V1
            step_size = STEP_SIZE_V1
            num_obs = 14

        team = header[1]
        handedness = header[2]
        outcome = header[3]
        step_count = header[4]
        duration = header[5]
        timestamp = header[6]

        metadata = EpisodeMetadata(
            filename=filepath.name,
            format_version=format_version,
            team=team,
            handedness=handedness,
            outcome=outcome,
            step_count=step_count,
            duration=duration,
            timestamp=timestamp,
            steam_id=steam_id,
            username=username
        )

        # Read steps
        obs_list = []
        action_list = []
        reward_list = []

        for _ in range(step_count):
            step_data = f.read(step_size)
            if len(step_data) < step_size:
                print(f"Warning: truncated file {filepath.name}")
                break

            step = struct.unpack(step_format, step_data)
            obs_list.append(step[:num_obs])
            action_list.append(step[num_obs:num_obs+8])
            reward_list.append(step[num_obs+8])

        obs = np.array(obs_list, dtype=np.float32)
        actions = np.array(action_list, dtype=np.float32)
        rewards = np.array(reward_list, dtype=np.float32)

        return obs, actions, rewards, metadata


def load_all_recordings(
    folder: str,
    filter_outcome: Optional[int] = 0,  # Default: only successful (goal)
    filter_team: Optional[int] = 0,      # Default: only Blue team
    filter_handedness: Optional[int] = 0 # Default: only Right-handed
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[EpisodeMetadata]]:
    """
    Load all recordings from a folder.

    Args:
        folder: Path to recordings folder
        filter_outcome: Only include episodes with this outcome (0=Goal, 1=Timeout, 2=Manual, None=all)
        filter_team: Only include episodes with this team (0=Blue, 1=Red, None=all)
        filter_handedness: Only include episodes with this handedness (0=Right, 1=Left, None=all)

    Returns:
        obs: (total_steps, 14) float32 array
        actions: (total_steps, 8) float32 array
        rewards: (total_steps,) float32 array
        metadata: List of EpisodeMetadata for included episodes
    """
    folder = Path(folder)
    files = list(folder.glob("*.bin"))

    print(f"Found {len(files)} recording files")

    all_obs = []
    all_actions = []
    all_rewards = []
    all_metadata = []

    skipped_outcome = 0
    skipped_team = 0
    skipped_handedness = 0

    for filepath in sorted(files):
        try:
            obs, actions, rewards, meta = load_episode(filepath)

            # Apply filters
            if filter_outcome is not None and meta.outcome != filter_outcome:
                skipped_outcome += 1
                continue

            if filter_team is not None and meta.team != filter_team:
                skipped_team += 1
                continue

            if filter_handedness is not None and meta.handedness != filter_handedness:
                skipped_handedness += 1
                continue

            all_obs.append(obs)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_metadata.append(meta)

        except Exception as e:
            print(f"Error loading {filepath.name}: {e}")

    if skipped_outcome > 0:
        print(f"Skipped {skipped_outcome} episodes (wrong outcome)")
    if skipped_team > 0:
        print(f"Skipped {skipped_team} episodes (wrong team)")
    if skipped_handedness > 0:
        print(f"Skipped {skipped_handedness} episodes (wrong handedness)")

    if len(all_obs) == 0:
        print("No valid recordings found!")
        return np.array([]), np.array([]), np.array([]), []

    # Concatenate all episodes
    obs = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)

    print(f"Loaded {len(all_metadata)} episodes, {len(obs)} total steps")

    return obs, actions, rewards, all_metadata


def print_summary(metadata: List[EpisodeMetadata]):
    """Print summary statistics for loaded recordings."""
    if not metadata:
        print("No recordings loaded.")
        return

    total_steps = sum(m.step_count for m in metadata)
    total_duration = sum(m.duration for m in metadata)
    players = {}  # steam_id -> username
    for m in metadata:
        players[m.steam_id] = m.username
    outcomes = {0: "Goal", 1: "Timeout", 2: "Manual"}

    print("\n=== Recording Summary ===")
    print(f"Episodes: {len(metadata)}")
    print(f"Total steps: {total_steps}")
    print(f"Total duration: {total_duration / 60:.1f} minutes")
    print(f"Unique players: {len(players)}")
    for steam_id, username in players.items():
        print(f"  - {username} ({steam_id})")

    # Outcome breakdown
    outcome_counts = {}
    for m in metadata:
        outcome_counts[m.outcome] = outcome_counts.get(m.outcome, 0) + 1

    print("\nOutcomes:")
    for outcome, count in sorted(outcome_counts.items()):
        print(f"  {outcomes.get(outcome, 'Unknown')}: {count}")


def downsample(obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, factor: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample data by taking every Nth step.
    Useful if you recorded at 50Hz but want to train at 10Hz.
    """
    return obs[::factor], actions[::factor], rewards[::factor]


def compute_relative_features(obs: np.ndarray, goal_z: float = 26.0) -> np.ndarray:
    """
    Compute relative/derived features from absolute observations.

    Input obs layout (16 floats, already normalized):
        0: skater_x, 1: skater_z, 2: skater_y
        3: skater_vel_x, 4: skater_vel_z, 5: skater_vel_y
        6: skater_rotation (normalized by /π)
        7: puck_x, 8: puck_z, 9: puck_y
        10: puck_vel_x, 11: puck_vel_z, 12: puck_vel_y
        13: stick_x, 14: stick_z, 15: stick_y

    Output: original 16 + 9 new = 25 features

    New features (indices 16-24):
        16: puck_rel_x (puck x relative to player)
        17: puck_rel_z (puck z relative to player)
        18: goal_rel_x (enemy goal x relative to player)
        19: goal_rel_z (enemy goal z relative to player)
        20: angle_to_puck (angle to puck relative to player facing, normalized)
        21: distance_to_puck (normalized)
        22: stick_to_puck_x (stick to puck delta x)
        23: stick_to_puck_z (stick to puck delta z)
        24: stick_to_puck_y (stick to puck delta y - for fine control)

    Args:
        obs: (N, 16) normalized observations
        goal_z: Z position of enemy goal in world units (default 26 for red goal)

    Returns:
        (N, 25) augmented observations
    """
    N = obs.shape[0]
    new_features = np.zeros((N, 9), dtype=np.float32)

    # Denormalization constants (to compute in world space, then renormalize)
    POS_NORM = 50.0
    HEIGHT_NORM = 5.0

    # Extract positions (still normalized)
    player_x = obs[:, 0]
    player_z = obs[:, 1]
    player_rot = obs[:, 6] * np.pi  # Denormalize rotation

    puck_x = obs[:, 7]
    puck_z = obs[:, 8]
    puck_y = obs[:, 9]

    stick_x = obs[:, 13]
    stick_z = obs[:, 14]
    stick_y = obs[:, 15]

    # Puck relative to player (already normalized since both are /50)
    puck_rel_x = puck_x - player_x
    puck_rel_z = puck_z - player_z
    new_features[:, 0] = puck_rel_x
    new_features[:, 1] = puck_rel_z

    # Goal relative to player (goal is at z=26 in world, normalized = 26/50 = 0.52)
    goal_x_norm = 0.0  # Goal is at x=0
    goal_z_norm = goal_z / POS_NORM
    new_features[:, 2] = goal_x_norm - player_x
    new_features[:, 3] = goal_z_norm - player_z

    # Angle to puck relative to player facing
    # atan2 gives angle in world frame, subtract player rotation
    angle_to_puck_world = np.arctan2(puck_rel_x * POS_NORM, puck_rel_z * POS_NORM)
    angle_to_puck_rel = angle_to_puck_world - player_rot
    # Normalize to [-1, 1] by dividing by π
    new_features[:, 4] = np.clip(angle_to_puck_rel / np.pi, -1, 1)

    # Distance to puck (in normalized space, then scale reasonably)
    # Max distance on rink is ~60 units, normalized ~1.2, so this stays in reasonable range
    distance = np.sqrt(puck_rel_x**2 + puck_rel_z**2)
    new_features[:, 5] = distance

    # Stick to puck delta (for fine control)
    new_features[:, 6] = puck_x - stick_x
    new_features[:, 7] = puck_z - stick_z
    new_features[:, 8] = puck_y - stick_y  # Y matters for hitting airborne puck

    # Concatenate original + new features
    return np.concatenate([obs, new_features], axis=1)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and inspect recorded demonstrations")
    parser.add_argument("folder", help="Path to recordings folder")
    parser.add_argument("--all-outcomes", action="store_true", help="Include all outcomes (not just goals)")
    parser.add_argument("--all-teams", action="store_true", help="Include all teams")
    parser.add_argument("--all-handedness", action="store_true", help="Include all handedness")
    args = parser.parse_args()

    obs, actions, rewards, metadata = load_all_recordings(
        args.folder,
        filter_outcome=None if args.all_outcomes else 0,
        filter_team=None if args.all_teams else 0,
        filter_handedness=None if args.all_handedness else 0
    )

    print_summary(metadata)

    if len(obs) > 0:
        print(f"\nData shapes:")
        print(f"  Observations: {obs.shape}")
        print(f"  Actions: {actions.shape}")
        print(f"  Rewards: {rewards.shape}")

        print(f"\nObservation stats:")
        print(f"  Min: {obs.min(axis=0)}")
        print(f"  Max: {obs.max(axis=0)}")
        print(f"  Mean: {obs.mean(axis=0)}")

        print(f"\nAction stats:")
        print(f"  Min: {actions.min(axis=0)}")
        print(f"  Max: {actions.max(axis=0)}")
        print(f"  Mean: {actions.mean(axis=0)}")
