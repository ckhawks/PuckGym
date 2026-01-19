# Imitation Learning Plan

## Overview

Pivot from pure RL to behavior cloning (BC). Record human demonstrations of the task:
- Puck and player spawn at random positions
- Player skates to puck, gains control, scores on goal

## Standardization Requirements

All demonstrators MUST use these settings:

| Setting | Value | Reason |
|---------|-------|--------|
| **Team** | Blue | Attack positive Z goal |
| **Handedness** | Right | Stick pivot is mirrored for left-handed |
| **Position** | Any except Goalie | Goalie has different mechanics |

These will be captured in episode metadata for validation.

## Recording Specification

### Capture Frequency

**50Hz** (every physics tick)

- Captures fast stick movements for shots
- A 150ms shot flick = 7-8 samples (good fidelity)
- Can downsample in Python for training if needed
- Storage is negligible (~92KB per 20s episode)

### Per-Step Data (92 bytes)

```
Offset  Size    Field
0       56      14 floats - observations (normalized)
56      32      8 floats - actions (human input)
88      4       1 float - reward
```

**Observations (14 floats):**
| Index | Field | Normalization |
|-------|-------|---------------|
| 0 | skater_x | /30 |
| 1 | skater_y | /30 |
| 2 | skater_vel_x | /20 |
| 3 | skater_vel_y | /20 |
| 4 | skater_rotation | /π |
| 5 | puck_x | /30 |
| 6 | puck_y | /30 |
| 7 | puck_height | /2 |
| 8 | puck_vel_x | /20 |
| 9 | puck_vel_y | /20 |
| 10 | puck_vel_height | /10 |
| 11 | stick_x | /30 |
| 12 | stick_y | /30 |
| 13 | stick_height | /2 |

**Actions (8 floats):**
| Index | Field | Range |
|-------|-------|-------|
| 0 | move_x | [-1, 1] |
| 1 | move_y | [-1, 1] |
| 2 | aim_x | [-1, 1] |
| 3 | aim_y | [-1, 1] |
| 4 | blade_angle | [-1, 1] |
| 5 | jump | [0, 1] |
| 6 | crouch | [0, 1] |
| 7 | boost | [0, 1] |

### Episode File Format

**Header (written at end, prepended to file):**
```
Offset  Size    Field
0       1       Format version (uint8) = 1
1       1       Team (uint8) - Blue=0, Red=1
2       1       Handedness (uint8) - Right=0, Left=1
3       1       Outcome (uint8) - Goal=0, Timeout=1, Manual=2
4       4       Step count (uint32)
8       4       Episode duration seconds (float)
12      8       Timestamp UTC ms (int64)
20      32      Player ID (32 bytes, null-padded string)
52      ...     Step data (step_count × 92 bytes)
```

**File naming:**
```
recordings/
  {player_id}_{timestamp}_{outcome}.bin

Examples:
  recordings/steve_20240119_143052_goal.bin
  recordings/steve_20240119_143112_timeout.bin
```

## Keybinds

| Key | Action |
|-----|--------|
| **F7** | Toggle recording (primary) |
| **Numpad 0** | Toggle recording (alt 1) |
| **Insert** | Toggle recording (alt 2) |
| **F8** | Manual episode reset (discard current) |

## Config File

Location: `BepInEx/config/PuckCapture.cfg` (BepInEx standard)

```ini
[Recording]
# Your unique identifier (no spaces)
PlayerID = anonymous

# These are auto-detected but can be overridden for validation
# Team = Blue
# Handedness = Right
```

## Episode Flow

1. Player presses F7 to start recording mode
2. Mod spawns puck at random position (away from goals)
3. Mod spawns player at random position (away from goals and puck)
4. Recording starts automatically
5. Player plays normally - skate to puck, score
6. Episode ends on:
   - **Goal scored** → Save as `_goal.bin` ✓
   - **Timeout (120s)** → Save as `_timeout.bin`
   - **F8 pressed** → Save as `_manual.bin` (or discard?)
7. Auto-reset: new random positions, recording continues
8. F7 again to stop recording mode

## Data Requirements

| Quality | Episodes | Gameplay Time | Notes |
|---------|----------|---------------|-------|
| Minimum | 50-100 | ~30 min | Might work for simple behavior |
| Solid | 200-500 | 1.5-3 hours | Good baseline |
| Robust | 1000+ | 6+ hours | Best results |

With 3-4 players contributing 1-2 hours each = 500+ demonstrations easily.

**Important:** Only successful episodes (goal scored) should be used for training. Failed attempts add noise.

## Distribution Plan

1. Build release DLL of PuckCapture mod
2. Create README with:
   - Installation instructions
   - Required settings (Team, Handedness)
   - How to record
   - How to submit recordings
3. Players install mod, record sessions, zip `recordings/` folder
4. Collect and merge all recordings
5. Train BC model on combined dataset

## Project Structure

```
puck-ai-training/
├── PuckGym/              # RL training mod (existing)
│   └── src/
├── PuckCapture/          # Recording mod (new)
│   ├── src/
│   │   ├── Plugin.cs
│   │   ├── RecordingManager.cs
│   │   ├── CaptureKeybinds.cs
│   │   ├── EpisodeRecorder.cs
│   │   └── HumanInputCapture.cs
│   └── PuckCapture.csproj
├── python/
│   ├── puck_env.py       # Gymnasium env (existing)
│   ├── train.py          # RL training (existing)
│   ├── load_recordings.py    # Load .bin files (new)
│   ├── train_bc.py           # BC training (new)
│   └── requirements.txt
└── recordings/           # Collected demonstrations
```

## Python Training Pipeline

```python
# load_recordings.py
def load_recordings(path):
    """Load all .bin files, return obs/action arrays."""
    # Filter to only _goal.bin files
    # Validate team=Blue, handedness=Right
    # Return: obs (N, 14), actions (N, 8), episode_ids (N,)

# train_bc.py
def train_bc(obs, actions):
    """Train policy with supervised learning."""
    # Simple MLP: obs (14) → actions (8)
    # MSE loss for continuous actions
    # Can use PyTorch or stable-baselines3's BC
```

## Technical Notes

### Capturing Human Input

The mod needs to read what the human is actually doing, not what the RL controller would do:

```csharp
// Read actual human inputs from PlayerInput
float moveX = playerInput.MoveInput.ClientValue.x;
float moveY = playerInput.MoveInput.ClientValue.y;
float aimX = /* normalize stick angle to [-1,1] */;
float aimY = /* normalize stick angle to [-1,1] */;
// etc.
```

### Goal Detection

Reuse the existing `GoalTrigger` Harmony patch from PuckGym, but instead of giving reward, mark episode as successful.

### Avoiding Goal Spawns

Reuse the spawn logic from PuckGym - 7 unit exclusion zone around goals to prevent physics issues.
