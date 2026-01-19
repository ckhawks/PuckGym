# PuckGym - Reinforcement Learning for Puck

Train an AI to play [Puck](https://store.steampowered.com/app/2513900/Puck/) using reinforcement learning with PPO.

## How It Works

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   Puck Game (Unity)              Python Trainer          │
│   ─────────────────              ──────────────          │
│                                                          │
│   ┌─────────────┐   Shared Memory   ┌────────────┐      │
│   │ RLBridge    │◄─────────────────►│ PuckEnv    │      │
│   │ RLController│  observations     │ (Gym)      │      │
│   └─────────────┘  actions/rewards  └─────┬──────┘      │
│         │                                 │              │
│         ▼                                 ▼              │
│   ┌─────────────┐                  ┌────────────┐       │
│   │ Puck Game   │                  │ PPO Agent  │       │
│   │ Physics     │                  │ (SB3)      │       │
│   └─────────────┘                  └────────────┘       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Setup

### 1. Build the C# Mod

```bash
cd PuckGym
dotnet build PuckGym.csproj
```

The DLL will be copied to your Puck plugins folder automatically.

### 2. Install Python Dependencies

```bash
cd python
pip install -r requirements.txt
```

## Training

### Start a New Training Run

1. Launch Puck and start a local game (create a private match)
2. Press **F9** to enable training mode
3. Run training:

```bash
cd python
python train.py --total-timesteps 1000000 --vec-normalize
```

### Resume Training from Checkpoint

```bash
python train.py --resume ./runs/puck_PPO_YYYYMMDD_HHMMSS --total-timesteps 5000000
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--total-timesteps` | 1,000,000 | Total training steps |
| `--resume` | none | Path to run directory to resume from |
| `--vec-normalize` | off | Enable observation/reward normalization (recommended) |
| `--learning-rate` | 3e-4 | Learning rate |
| `--n-steps` | 2048 | Steps per rollout |
| `--batch-size` | 64 | Minibatch size |
| `--device` | auto | Device: auto, cpu, cuda |
| `--policy-arch` | 256 256 | Policy network architecture |
| `--value-arch` | 256 256 | Value network architecture |

### Monitor with TensorBoard

```bash
tensorboard --logdir ./python/runs
```

Key metrics:
- `rollout/ep_rew_mean` - Average episode reward (should increase over time)
- `rollout/ep_len_mean` - Average episode length

## Playing Back a Trained Model

To demonstrate what the agent learned:

1. Launch Puck and start a local game
2. Press **F9** to enable training mode
3. Run:

```bash
cd python
python train.py --mode play --model ./runs/YOUR_RUN/final_model.zip --vec-norm ./runs/YOUR_RUN/vec_normalize.pkl --episodes 10
```

4. Watch the agent play!

You can adjust time scale with F10/F11 while watching.

## In-Game Keybinds

| Key | Action |
|-----|--------|
| **F8** | Debug: teleport puck into goal (test goal detection) |
| **F9** | Toggle training mode |
| **F10** | Increase time scale (+10x) |
| **F11** | Decrease time scale (-10x) |
| **F12** | Reset time scale to 1x |

## Reward Function

| Event | Reward |
|-------|--------|
| Goal scored | +100.0 |
| Goal conceded | -100.0 |
| Approach puck | +0.02 per unit closer |
| Facing puck | +0.005 x dot product |
| Puck toward goal | +0.01 per unit closer |
| Puck in front of goal (<15 units) | +0.002 per step |
| Puck in offensive zone | +0.001 per step |
| Shot toward goal | +0.05 x (shot quality x speed) |
| Stick touches puck | +0.1 (once per contact) |
| Time penalty | -0.001 per step |

## Episode Structure

- Episodes last **2 minutes** max, then reset
- Episodes also end when a **goal is scored**
- On reset: player and puck positions are randomized
- Game clock is set to 60 minutes to prevent period endings

## Action Space

| Action | Range | Description |
|--------|-------|-------------|
| move_x | -1 to 1 | Turn left/right |
| move_y | -1 to 1 | Skate forward/back |
| aim_x | -1 to 1 | Stick vertical angle |
| aim_y | -1 to 1 | Stick horizontal angle |
| blade_angle | -1 to 1 | Blade rotation (scroll wheel) |
| jump | 0 to 1 | Jump (threshold 0.5) |
| crouch | 0 to 1 | Slide/crouch (threshold 0.5) |
| boost | 0 to 1 | Sprint (threshold 0.5) |

## Observation Space

14 normalized floats:
- Skater position (x, z)
- Skater velocity (x, z)
- Skater rotation (radians)
- Puck position (x, z)
- Puck height (y)
- Puck velocity (x, z)
- Puck vertical velocity (y)
- Stick blade position (x, z, height)

(Goals, time, scores not included - agent learns direction from rewards)

## Project Structure

```
puck-ai-training/
├── PuckGym/
│   └── src/
│       ├── Plugin.cs         # BepInEx plugin entry point
│       ├── RLController.cs   # Main RL loop, rewards, observations
│       ├── RLBridge.cs       # Shared memory communication
│       └── RLKeybinds.cs     # Keybinds and Harmony patches
├── python/
│   ├── puck_env.py          # Gymnasium environment
│   ├── train.py             # Training script
│   └── requirements.txt
├── CLAUDE.md                # Development notes
└── README.md
```

## Troubleshooting

**"Could not connect to shared memory"**
- Make sure Puck is running
- Press F9 to enable training mode
- Check the game log at `C:\Program Files (x86)\Steam\steamapps\common\Puck\Logs\Puck.log`

**Timeout errors**
- Game might be in a menu or loading screen
- Make sure you're in an active game (not lobby)
- Try increasing timeout: `--timeout 30`

**Agent not learning (flat reward curve)**
- Make sure episodes are completing (check ep_len_mean)
- Use `--vec-normalize` for better reward scaling
- Train longer - 1M+ steps minimum to see improvement
- Check that the agent is actually controlling the player (watch the game)

**Game stuck in FaceOff at 0 seconds**
- This was a bug with the 60-minute clock being set during FaceOff
- Update to latest mod version

**Build can't copy DLL**
- Close the game before building
- The DLL is locked while the game is running
