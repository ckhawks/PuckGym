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

## Shared Memory Struct (must match C# and Python exactly)

```
Offset  Type    Field
0       byte    obs_ready
1       byte    action_ready
2       byte    done
3       byte    reset_flag
4       float   reward
8-72    float   observations (16 floats: skater x/y/vel/rot, puck x/y/vel, goals x/y, time, scores)
72-88   float   actions (4 floats: move_x, move_y, aim_x, aim_y)
88-91   byte    buttons (shoot, pass, boost)
```

Total: 91 bytes

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

# Train
python train.py --total-timesteps 1000000

# With options
python train.py --learning-rate 1e-4 --device cuda --vec-normalize

# Test trained model
python train.py --mode play --model ./runs/<name>/final_model
```

## Notes

- Game normally runs at 360 tick/fps
- Time.timeScale up to 5x is safe
- Warmup = 11 pucks (good for learning contact)
- FaceOff → Playing = 1 puck (realistic training)
- Mod runs on client with local server for prototyping
- Eventually migrate to dedicated server for faster training
