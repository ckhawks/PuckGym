# PuckGym - RL Training for Puck

## Quick Reference

**Game Log:** `C:\Program Files (x86)\Steam\steamapps\common\Puck\Logs\Puck.log`
**Decompiled Source:** `C:\PuckModdingTools\PuckDecompiled202Project\Puck\`
**Reference Mod:** `C:\Projects\Puck Plugins\ToasterCameras\`

## Architecture

```
C# Mod (PuckGym)          Shared Memory "PuckRL"         Python (stable-baselines3)
    RLController  ───────────────────────────────────►  PuckEnv (Gymnasium)
    - GatherObservation()     obs_ready flag                - step()
    - ApplyAction()           action_ready flag             - reset()
    - CalculateReward()       91 byte struct                - PPO training
```

## Key Game Classes

| Class | Singleton Access | Purpose |
|-------|------------------|---------|
| `PlayerManager` | `NetworkBehaviourSingleton<PlayerManager>.Instance` | `.GetLocalPlayer()`, `.GetPlayers()` |
| `PuckManager` | `NetworkBehaviourSingleton<PuckManager>.Instance` | `.GetPuck()`, `.GetPucks()` |
| `GameManager` | `NetworkBehaviourSingleton<GameManager>.Instance` | `.GameState.Value`, `.Phase`, `.Server_SetPhase()` |
| `PhysicsManager` | `MonoBehaviourSingleton<PhysicsManager>.Instance` | `.TickRate` (default 50, game runs 360) |

## Player Structure

```
Player
├── .PlayerBody (PlayerBodyV2) → .Rigidbody, .transform
├── .PlayerInput → .MoveInput, .StickRaycastOriginAngleInput, .SprintInput, etc.
├── .Stick → .IsTouchingPuck, blade position
├── .StickPositioner → controls stick via raycast angles
├── .Team.Value → PlayerTeam.Blue or PlayerTeam.Red
└── .IsLocalPlayer
```

## Input System

**Movement:** `playerInput.MoveInput.ClientValue = new Vector2(turnX, forwardY)`
- X: -1 (left) to 1 (right) turn
- Y: -1 (back) to 1 (forward)

**Stick Control:** `playerInput.StickRaycastOriginAngleInput.ClientValue = new Vector2(angleX, angleY)`
- X (vertical): -25° to 80° (default)
- Y (horizontal): -92.5° to 92.5°
- Moving stick fast = shooting (physics-based)

**Other Inputs:** `.SlideInput`, `.SprintInput`, `.TrackInput`, `.LookInput`, `.StopInput` (all bool)

## Game Phases (GamePhase enum)

`None` → `Warmup` (11 pucks) → `FaceOff` → `Playing` (1 puck) → `BlueScore`/`RedScore` → `Replay` → `PeriodOver` → `GameOver`

**Change phase:** `GameManager.Instance.Server_SetPhase(GamePhase.FaceOff)`

## Keybind Pattern (from ToasterCameras)

```csharp
// Setup
var action = new InputAction(binding: "<Keyboard>/f9");
action.Enable();

// Check in Harmony patch on PlayerInput.Update
[HarmonyPatch(typeof(PlayerInput), "Update")]
public static class Patch {
    [HarmonyPostfix]
    public static void Postfix() {
        if (action.WasPressedThisFrame()) { /* do thing */ }
    }
}
```

Check chat focus before processing: `UIChat.Instance` + reflection for `isFocused` field.

## Observation Space (25 floats = 16 base + 9 computed)

Base observations match PuckCapture v3 recording format. All positions normalized by /50 (world limit), velocities by /20, heights by /5, rotation by /π.

**Base Observations (from shared memory / recordings):**

| Index | Field | Description |
|-------|-------|-------------|
| 0 | skater_x | Player horizontal position |
| 1 | skater_z | Player forward position |
| 2 | skater_y | Player height |
| 3 | skater_vel_x | Player horizontal velocity |
| 4 | skater_vel_z | Player forward velocity |
| 5 | skater_vel_y | Player vertical velocity |
| 6 | skater_rotation | Player facing direction (normalized) |
| 7 | puck_x | Puck horizontal position |
| 8 | puck_z | Puck forward position |
| 9 | puck_y | Puck height |
| 10 | puck_vel_x | Puck horizontal velocity |
| 11 | puck_vel_z | Puck forward velocity |
| 12 | puck_vel_y | Puck vertical velocity |
| 13 | stick_x | Stick blade horizontal position |
| 14 | stick_z | Stick blade forward position |
| 15 | stick_y | Stick blade height |

**Computed Relative Features (egocentric, helps generalization):**

| Index | Field | Description |
|-------|-------|-------------|
| 16 | puck_rel_x | Puck X relative to player |
| 17 | puck_rel_z | Puck Z relative to player |
| 18 | goal_rel_x | Enemy goal X relative to player |
| 19 | goal_rel_z | Enemy goal Z relative to player |
| 20 | angle_to_puck | Angle to puck relative to player facing |
| 21 | distance_to_puck | Distance to puck (normalized) |
| 22 | stick_to_puck_x | Stick to puck delta X (for fine control) |
| 23 | stick_to_puck_z | Stick to puck delta Z |
| 24 | stick_to_puck_y | Stick to puck delta Y (for airborne puck) |

## Action Space (8 floats)

| Index | Field | Range | Description |
|-------|-------|-------|-------------|
| 0 | move_x | -1 to 1 | Turn left/right |
| 1 | move_y | -1 to 1 | Forward/back |
| 2 | aim_x | -1 to 1 | Stick vertical angle |
| 3 | aim_y | -1 to 1 | Stick horizontal angle |
| 4 | blade_angle | -1 to 1 | Blade rotation |
| 5 | jump | 0 to 1 | Jump (threshold 0.5) |
| 6 | crouch | 0 to 1 | Crouch/slide (threshold 0.5) |
| 7 | boost | 0 to 1 | Sprint (threshold 0.5) |

## Current Keybinds

- **F9** - Toggle training
- **F10** - Increase time scale (+0.5x, max 5x)
- **F11** - Decrease time scale (-0.5x, min 0.25x)
- **F12** - Reset time scale

## Reward Function

| Event | Reward |
|-------|--------|
| Score goal | +10.0 |
| Concede goal | -10.0 |
| Approach puck | +0.02 × Δdistance |
| Puck toward goal | +0.01 × Δdistance |
| Stick-puck contact | +0.5 (once per contact) |
| Time penalty | -0.0005/step |

## Training Commands

```bash
# Install Python deps
cd python && pip install -r requirements.txt

# === Behavior Cloning (from human demos) ===
# Train BC policy on recordings
python train_bc.py path/to/recordings --epochs 100

# Evaluate BC policy
python train_bc.py path/to/recordings --eval-only ./runs/bc_xxx/bc_ppo_model.zip

# === RL Training (PPO) ===
# Train from scratch
python train.py --total-timesteps 1000000

# With options
python train.py --learning-rate 1e-4 --device cuda --vec-normalize

# Resume from BC pretrained model
python train.py --resume ./runs/bc_xxx

# Test trained model
python train.py --mode play --model ./runs/<name>/final_model

# === Recording Analysis ===
python analyze_recordings.py path/to/recordings --verbose
```

## PuckCapture (Recording Demonstrations)

Keybinds:
- **F7 / Numpad0 / Insert** - Toggle recording
- **F8** - Reset episode

Requirements to start recording:
- Must be on Blue team
- Must be Attacker (not Goalie)
- Must be right-handed

Recordings saved to: `Plugins/PuckCapture/recordings/` as `.bin` files (v3 format).

## Notes

- Game normally runs at 360 tick/fps
- Time.timeScale up to 5x is safe
- Warmup = 11 pucks (good for learning contact)
- FaceOff → Playing = 1 puck (realistic training)
- Mod runs on client with local server for prototyping
- Eventually migrate to dedicated server for faster training
- **BC Training**: Need 50-100 demos minimum, 200-500 for good results
