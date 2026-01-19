"""
Behavior Cloning Training for Puck

Trains a policy to imitate human demonstrations using the imitation library.
Can be used standalone or as a pretrained policy for RL fine-tuning.

Usage:
    python train_bc.py path/to/recordings
    python train_bc.py path/to/recordings --epochs 100 --batch-size 64
    python train_bc.py path/to/recordings --output ./models/bc_policy
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces

from imitation.algorithms import bc
from imitation.data import types

from load_recordings import load_all_recordings, print_summary


# Observation layout in v3 format (16 floats)
# 0: skater_x, 1: skater_z, 2: skater_y (height)
# 3: skater_vel_x, 4: skater_vel_z, 5: skater_vel_y
# 6: skater_rotation
# 7: puck_x, 8: puck_z, 9: puck_y (height)
# 10: puck_vel_x, 11: puck_vel_z, 12: puck_vel_y
# 13: stick_x, 14: stick_z, 15: stick_y (height)

OBS_DIM = 16
ACT_DIM = 8


def create_transitions(
    obs: np.ndarray,
    actions: np.ndarray,
    episode_lengths: list,
) -> types.Transitions:
    """
    Convert loaded recordings to imitation library Transitions format.

    Args:
        obs: (N, 16) observations
        actions: (N, 8) actions
        episode_lengths: list of episode lengths

    Returns:
        imitation.data.types.Transitions
    """

    # Create done flags from episode lengths
    dones = np.zeros(len(obs), dtype=bool)
    idx = 0
    for length in episode_lengths:
        if idx + length <= len(dones):
            dones[idx + length - 1] = True
        idx += length

    # Next observations (shifted by 1, last one doesn't matter)
    next_obs = np.roll(obs, -1, axis=0)
    next_obs[-1] = obs[-1]  # Last step's next_obs is itself

    # Infos (empty dicts)
    infos = np.array([{}] * len(obs))

    return types.Transitions(
        obs=obs.astype(np.float32),
        acts=actions.astype(np.float32),
        next_obs=next_obs.astype(np.float32),
        dones=dones,
        infos=infos
    )


def create_dummy_env():
    """
    Create a dummy environment for policy initialization.
    Must match the observation/action space of recordings (16 obs, 8 actions).
    """
    import gymnasium as gym

    class DummyPuckEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(OBS_DIM,), dtype=np.float32
            )
            # Continuous actions: move, aim are -1 to 1, buttons are 0 to 1
            self.action_space = spaces.Box(
                low=np.array([-1, -1, -1, -1, -1, 0, 0, 0], dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
                dtype=np.float32
            )

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(OBS_DIM, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(OBS_DIM, dtype=np.float32), 0.0, True, False, {}

    return DummyVecEnv([lambda: DummyPuckEnv()])


def train_bc(
    recordings_folder: str,
    output_dir: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    device: str = "auto"
):
    """
    Train behavior cloning policy on recorded demonstrations.

    Args:
        recordings_folder: Path to folder containing .bin recordings
        output_dir: Where to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Training device (auto, cpu, cuda)
    """
    print("\n" + "=" * 60)
    print("  BEHAVIOR CLONING TRAINING")
    print("=" * 60)

    # Load recordings (only goal episodes, blue team, right-handed)
    print(f"\nLoading recordings from: {recordings_folder}")
    obs, actions, rewards, metadata = load_all_recordings(
        recordings_folder,
        filter_outcome=0,      # Only goals
        filter_team=0,         # Only blue team
        filter_handedness=0    # Only right-handed
    )

    if len(obs) == 0:
        print("No valid recordings found!")
        return None

    print_summary(metadata)

    print(f"\nData shapes:")
    print(f"  Observations: {obs.shape}")
    print(f"  Actions: {actions.shape}")

    # Get episode lengths for creating done flags
    episode_lengths = [m.step_count for m in metadata]

    # Create transitions
    print(f"\nCreating transitions...")
    transitions = create_transitions(obs, actions, episode_lengths)

    print(f"  Observation dim: {OBS_DIM}")
    print(f"  Total transitions: {len(transitions.obs)}")

    # Create dummy environment for policy initialization
    venv = create_dummy_env()

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("./runs") / f"bc_{timestamp}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    # Create BC trainer
    print(f"\nInitializing BC trainer...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")

    # Create policy with custom architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],  # Policy network
            vf=[256, 256],  # Value network (not used in BC but required)
        )
    )

    rng = np.random.default_rng(seed=42)

    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=transitions,
        batch_size=batch_size,
        optimizer_kwargs={"lr": learning_rate},
        device=device,
        rng=rng,
        policy=ActorCriticPolicy(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            lr_schedule=lambda _: learning_rate,
            **policy_kwargs
        ),
    )

    # Train
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 40)

    bc_trainer.train(n_epochs=epochs)

    print("-" * 40)
    print("Training complete!")

    # Save policy
    policy_path = output_dir / "bc_policy.pt"
    torch.save(bc_trainer.policy.state_dict(), policy_path)
    print(f"\nPolicy saved to: {policy_path}")

    # Also save as a PPO model for compatibility with train.py
    # This allows using the BC policy as a starting point for RL
    print("\nCreating SB3-compatible PPO model...")
    ppo_model = PPO(
        policy="MlpPolicy",
        env=venv,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0
    )
    # Copy weights from BC policy to PPO policy
    ppo_model.policy.load_state_dict(bc_trainer.policy.state_dict())

    ppo_path = output_dir / "bc_ppo_model"
    ppo_model.save(str(ppo_path))
    print(f"PPO model saved to: {ppo_path}.zip")

    # Save training info
    info_path = output_dir / "training_info.txt"
    with open(info_path, "w") as f:
        f.write(f"Behavior Cloning Training\n")
        f.write(f"========================\n\n")
        f.write(f"Recordings: {recordings_folder}\n")
        f.write(f"Episodes: {len(metadata)}\n")
        f.write(f"Total steps: {len(obs)}\n")
        f.write(f"Observation dim: {OBS_DIM}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Device: {device}\n")
        f.write(f"\nPlayers:\n")
        players = {}
        for m in metadata:
            players[m.steam_id] = m.username
        for steam_id, username in players.items():
            f.write(f"  - {username} ({steam_id})\n")

    print(f"Training info saved to: {info_path}")

    return str(ppo_path) + ".zip"


def evaluate_bc(model_path: str, recordings_folder: str):
    """
    Evaluate BC policy on held-out demonstrations.
    Computes mean squared error between predicted and actual actions.
    """
    print(f"\nEvaluating BC policy: {model_path}")

    # Load recordings
    obs, actions, rewards, metadata = load_all_recordings(
        recordings_folder,
        filter_outcome=0,
        filter_team=0,
        filter_handedness=0
    )

    if len(obs) == 0:
        print("No recordings to evaluate on!")
        return

    # Load model
    venv = create_dummy_env()
    model = PPO.load(model_path, env=venv)

    # Get predictions
    obs_tensor = torch.FloatTensor(obs).to(model.device)
    with torch.no_grad():
        predicted_actions, _, _ = model.policy(obs_tensor)
        predicted_actions = predicted_actions.cpu().numpy()

    # Compute errors
    mse = np.mean((predicted_actions - actions) ** 2)
    mae = np.mean(np.abs(predicted_actions - actions))

    # Per-action errors
    action_names = ["move_x", "move_y", "aim_x", "aim_y", "blade", "jump", "crouch", "boost"]

    print(f"\nEvaluation Results:")
    print(f"  Total steps: {len(obs)}")
    print(f"  Overall MSE: {mse:.6f}")
    print(f"  Overall MAE: {mae:.6f}")
    print(f"\n  Per-action MAE:")
    for i, name in enumerate(action_names):
        action_mae = np.mean(np.abs(predicted_actions[:, i] - actions[:, i]))
        print(f"    {name}: {action_mae:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train behavior cloning policy")

    parser.add_argument("recordings", help="Path to recordings folder")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Training device")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate model after training")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Only evaluate existing model (path to .zip)")

    args = parser.parse_args()

    if args.eval_only:
        evaluate_bc(args.eval_only, args.recordings)
        return

    model_path = train_bc(
        recordings_folder=args.recordings,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )

    if args.eval and model_path:
        evaluate_bc(model_path, args.recordings)


if __name__ == "__main__":
    main()
