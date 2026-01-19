"""
Puck RL Training Script

Uses Stable-Baselines3 PPO to train an agent to play Puck.
"""

import os
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from puck_env import PuckEnv


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_lengths = []

    def _on_step(self) -> bool:
        # Log episode stats when available
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if "r" in ep_info:
                self.logger.record("rollout/ep_reward_raw", ep_info["r"])
            if "l" in ep_info:
                self.logger.record("rollout/ep_length_raw", ep_info["l"])
        return True


class RewardLoggingCallback(BaseCallback):
    """
    Callback to print training progress to console.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._episode_count = 0
        self._total_reward = 0.0

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Get recent episode info
            if len(self.model.ep_info_buffer) > 0:
                recent_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                recent_lengths = [ep["l"] for ep in self.model.ep_info_buffer]

                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                avg_length = np.mean(recent_lengths) if recent_lengths else 0

                print(f"\n[Step {self.num_timesteps}] "
                      f"Episodes: {len(self.model.ep_info_buffer)}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.0f}")

        return True


class TimeScaleCallback(BaseCallback):
    """
    Callback to periodically slow down training for viewers.
    Runs at 1x speed for slow_duration seconds every cycle_minutes minutes.
    """

    def __init__(
        self,
        env,
        fast_scale: float = 30.0,
        slow_scale: float = 1.0,
        cycle_minutes: float = 10.0,
        slow_duration: float = 30.0,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.env = env
        self.fast_scale = fast_scale
        self.slow_scale = slow_scale
        self.cycle_seconds = cycle_minutes * 60
        self.slow_duration = slow_duration
        self._start_time = None
        self._is_slow = False

    def _on_training_start(self):
        import time
        self._start_time = time.time()
        self._set_time_scale(self.fast_scale)
        print(f"[TimeScale] Starting at {self.fast_scale}x, "
              f"will slow to {self.slow_scale}x for {self.slow_duration}s "
              f"every {self.cycle_seconds/60:.0f} minutes")

    def _on_step(self) -> bool:
        import time
        elapsed = time.time() - self._start_time
        cycle_position = elapsed % self.cycle_seconds

        # Check if we should be in slow mode
        should_be_slow = cycle_position < self.slow_duration

        if should_be_slow and not self._is_slow:
            self._set_time_scale(self.slow_scale)
            self._is_slow = True
            print(f"\n[TimeScale] Slowing to {self.slow_scale}x for viewers...")
        elif not should_be_slow and self._is_slow:
            self._set_time_scale(self.fast_scale)
            self._is_slow = False
            print(f"\n[TimeScale] Speeding up to {self.fast_scale}x...")

        return True

    def _set_time_scale(self, scale: float):
        """Set time scale on the underlying env."""
        # Unwrap to get to the base PuckEnv
        env = self.env
        while hasattr(env, 'envs'):
            env = env.envs[0]
        while hasattr(env, 'env'):
            env = env.env

        if hasattr(env, 'set_time_scale'):
            env.set_time_scale(scale)


def make_env(normalize_obs: bool = True, timeout: float = 5.0):
    """Create and wrap the environment."""
    def _init():
        env = PuckEnv(
            timeout_seconds=timeout,
            normalize_obs=normalize_obs,
        )
        env = Monitor(env)  # Wrap for logging
        return env
    return _init


def train(args):
    """Main training function."""

    # Check if resuming from existing run
    resuming = args.resume is not None

    if resuming:
        output_dir = Path(args.resume)
        if not output_dir.exists():
            raise ValueError(f"Resume directory not found: {output_dir}")
        model_path = output_dir / "final_model.zip"
        vec_norm_path = output_dir / "vec_normalize.pkl"
        if not model_path.exists():
            raise ValueError(f"Model not found: {model_path}")
    else:
        # Create new output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"puck_{args.algo}_{timestamp}"
        output_dir = Path(args.output_dir) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PUCK RL TRAINING {'(RESUMING)' if resuming else ''}")
    print(f"{'='*60}")
    print(f"  Algorithm: {args.algo}")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # Create environment
    print("Creating environment...")
    print("Make sure the Puck game is running with the RL mod!")

    env = DummyVecEnv([make_env(normalize_obs=True, timeout=args.timeout)])

    # Load or create VecNormalize wrapper
    if resuming and vec_norm_path.exists():
        print(f"Loading VecNormalize from: {vec_norm_path}")
        env = VecNormalize.load(str(vec_norm_path), env)
    elif args.vec_normalize:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

    # Load or create model
    if resuming:
        print(f"Loading model from: {model_path}")
        model = PPO.load(
            str(model_path),
            env=env,
            tensorboard_log=str(output_dir / "tensorboard"),
            device=args.device,
        )
    elif args.algo == "PPO":
        print(f"Creating new {args.algo} model...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            verbose=1,
            tensorboard_log=str(output_dir / "tensorboard"),
            device=args.device,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=args.policy_arch,  # Policy network
                    vf=args.value_arch,   # Value network
                )
            )
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Setup callbacks
    callbacks = []

    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="puck_model",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Tensorboard callback
    callbacks.append(TensorboardCallback())

    # Console logging callback
    callbacks.append(RewardLoggingCallback(log_freq=1000))

    # Time scale callback for viewer-friendly training
    if not args.no_time_scale:
        callbacks.append(TimeScaleCallback(
            env=env,
            fast_scale=args.fast_scale,
            slow_scale=args.slow_scale,
            cycle_minutes=args.slow_cycle,
            slow_duration=args.slow_duration,
        ))

    # Train!
    print(f"\nStarting training for {args.total_timesteps:,} timesteps...")
    print("Press Ctrl+C to stop early (model will be saved)\n")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    # Save final model
    final_model_path = output_dir / "final_model"
    model.save(str(final_model_path))
    print(f"\nModel saved to: {final_model_path}")

    # Save VecNormalize stats if used (check if env is VecNormalize)
    if isinstance(env, VecNormalize):
        vec_norm_path = output_dir / "vec_normalize.pkl"
        env.save(str(vec_norm_path))
        print(f"VecNormalize stats saved to: {vec_norm_path}")

    env.close()
    print("\nTraining complete!")

    return str(final_model_path)


def play(model_path: str, vec_norm_path: str = None, episodes: int = 5):
    """Play using a trained model (for testing)."""

    print(f"\nLoading model from: {model_path}")

    # Create env
    env = DummyVecEnv([make_env(normalize_obs=True, timeout=10.0)])

    # Load VecNormalize if available
    if vec_norm_path and os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False  # Don't update stats during play
        env.norm_reward = False
        print(f"Loaded VecNormalize from: {vec_norm_path}")

    # Load model
    model = PPO.load(model_path, env=env)

    print(f"\nPlaying {episodes} episodes...")

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1

        print(f"Episode {ep + 1}: {steps} steps, reward: {total_reward:.2f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train RL agent to play Puck")

    # Mode
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "play"],
                        help="Mode: train or play")

    # For play mode
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model for play mode")
    parser.add_argument("--vec-norm", type=str, default=None,
                        help="Path to VecNormalize stats for play mode")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to play")

    # For resuming training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to run directory to resume (e.g., ./runs/puck_PPO_20260118_211005)")

    # Training settings
    parser.add_argument("--algo", type=str, default="PPO",
                        choices=["PPO"],
                        help="RL Algorithm")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--output-dir", type=str, default="./runs",
                        help="Output directory for models and logs")
    parser.add_argument("--save-freq", type=int, default=10_000,
                        help="Save model every N steps")

    # Environment settings
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Timeout in seconds waiting for game")
    parser.add_argument("--vec-normalize", action="store_true",
                        help="Use VecNormalize wrapper")

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Max gradient norm")

    # Network architecture
    parser.add_argument("--policy-arch", type=int, nargs="+", default=[256, 256],
                        help="Policy network architecture")
    parser.add_argument("--value-arch", type=int, nargs="+", default=[256, 256],
                        help="Value network architecture")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda")

    # Time scale for viewer-friendly training
    parser.add_argument("--fast-scale", type=float, default=30.0,
                        help="Fast time scale during training (default 30x)")
    parser.add_argument("--slow-scale", type=float, default=1.0,
                        help="Slow time scale for viewer periods (default 1x)")
    parser.add_argument("--slow-cycle", type=float, default=10.0,
                        help="Minutes between slow periods (default 10)")
    parser.add_argument("--slow-duration", type=float, default=30.0,
                        help="Seconds to run at slow speed (default 30)")
    parser.add_argument("--no-time-scale", action="store_true",
                        help="Disable automatic time scaling")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "play":
        if args.model is None:
            print("Error: --model required for play mode")
            return
        play(args.model, args.vec_norm, args.episodes)


if __name__ == "__main__":
    main()
