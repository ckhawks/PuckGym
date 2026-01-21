"""
Viewer script for watching RL training in real-time.

Connects to game instance 0 (viewer mode):
- Runs at 1x speed (watchable)
- Uses deterministic actions (best policy, no exploration)
- Periodically reloads the latest model checkpoint
- Monitors training directory for new checkpoints
"""

import os
import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from puck_env import PuckEnv


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the most recent checkpoint file in the directory."""
    if not checkpoint_dir.exists():
        return None

    # Look for checkpoint files
    checkpoints = list(checkpoint_dir.glob("puck_model_*_steps.zip"))

    if not checkpoints:
        return None

    # Sort by modification time, return most recent
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def find_model_path(run_dir: Path) -> Optional[Path]:
    """Find a model to load from the run directory."""
    # Priority order:
    # 1. Latest checkpoint in checkpoints/ directory
    # 2. final_model.zip (completed training)
    # 3. bc_ppo_model.zip (BC pretrained model)

    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists():
        latest = find_latest_checkpoint(checkpoint_dir)
        if latest:
            return latest

    final_model = run_dir / "final_model.zip"
    if final_model.exists():
        return final_model

    bc_model = run_dir / "bc_ppo_model.zip"
    if bc_model.exists():
        return bc_model

    return None


def make_env(instance_id: int = 0, timeout: float = 10.0):
    """Create viewer environment (always instance 0)."""
    def _init():
        env = PuckEnv(
            instance_id=instance_id,
            timeout_seconds=timeout,
            normalize_obs=True,
        )
        # Set time scale to 1x for viewer
        env.set_time_scale(1.0)
        env = Monitor(env)
        return env
    return _init


def viewer(args):
    """Main viewer loop."""

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return

    print(f"\n{'='*60}")
    print(f"  PUCK RL VIEWER (Instance 0)")
    print(f"{'='*60}")
    print(f"  Run directory: {run_dir}")
    print(f"  Reload interval: {args.reload_interval}s")
    print(f"  Max episode steps: {args.max_steps}")
    print(f"{'='*60}\n")

    # Create environment (instance 0 for viewer)
    print("Creating viewer environment (instance 0)...")
    print("Make sure game instance 0 is running!")

    env = DummyVecEnv([make_env(instance_id=0, timeout=args.timeout)])

    # Check for VecNormalize stats
    vec_norm_path = run_dir / "vec_normalize.pkl"
    if vec_norm_path.exists():
        print(f"Loading VecNormalize from: {vec_norm_path}")
        env = VecNormalize.load(str(vec_norm_path), env)
        env.training = False  # Don't update stats during viewing
        env.norm_reward = False
    else:
        print("No VecNormalize stats found (will try to use model without them)")

    # Initial model load
    model_path = find_model_path(run_dir)
    if model_path is None:
        print(f"Error: No model found in {run_dir}")
        print("Looking for: checkpoints/puck_model_*_steps.zip, final_model.zip, or bc_ppo_model.zip")
        env.close()
        return

    print(f"\nLoading initial model: {model_path.name}")
    model = PPO.load(str(model_path), env=env)
    last_model_path = model_path
    last_reload_time = time.time()

    print("\n" + "="*60)
    print("VIEWER RUNNING - Press Ctrl+C to stop")
    print("="*60)
    print(f"Running deterministic policy at 1x speed...")
    print(f"Model will reload every {args.reload_interval}s if updated\n")

    episode_num = 0

    try:
        while True:
            # Check if we should reload the model
            current_time = time.time()
            if current_time - last_reload_time >= args.reload_interval:
                # Look for updated checkpoint
                latest_path = find_model_path(run_dir)
                if latest_path and latest_path != last_model_path:
                    print(f"\n[{time.strftime('%H:%M:%S')}] New checkpoint detected: {latest_path.name}")
                    try:
                        model = PPO.load(str(latest_path), env=env)
                        last_model_path = latest_path
                        print(f"Model reloaded successfully!")
                    except Exception as e:
                        print(f"Failed to reload model: {e}")
                        print("Continuing with current model...")

                last_reload_time = current_time

            # Run one episode
            episode_num += 1
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done and steps < args.max_steps:
                # Deterministic action (best policy)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                steps += 1

            reason = "goal/done" if done else "max steps"
            timestamp = time.strftime('%H:%M:%S')
            print(f"[{timestamp}] Episode {episode_num}: {steps} steps, "
                  f"reward: {total_reward:.2f} ({reason})")

    except KeyboardInterrupt:
        print("\n\nViewer stopped by user")
    finally:
        env.close()
        print("Viewer closed")


def main():
    parser = argparse.ArgumentParser(description="Watch RL training in real-time")

    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to training run directory (e.g., ./runs/puck_PPO_20260118_211005)")
    parser.add_argument("--reload-interval", type=float, default=30.0,
                        help="Seconds between checking for new checkpoints (default: 30)")
    parser.add_argument("--max-steps", type=int, default=2400,
                        help="Max steps per episode (default 2400 = ~2 min at 1x speed)")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Timeout waiting for game connection")

    args = parser.parse_args()
    viewer(args)


if __name__ == "__main__":
    main()
