# Multi-Instance Training Plan

## Overview

Run multiple Puck instances simultaneously:
- Instance 0: Viewer mode (1x speed, deterministic, shows "best" model)
- Instances 1-N: Training mode (high speed, stochastic, learning)

## C# Changes (PuckGym)

### 1. RLController.cs
- Add `_instanceId` field (default 0)
- Add `SetInstanceId(int id)` method
- Show instance ID in chat when F9 pressed: `"Training started on instance {id}"`

### 2. RLBridge.cs
- Change `MEMORY_NAME` from `"PuckRL"` to `$"PuckRL_{instanceId}"`
- Pass instance ID to `Initialize(int instanceId)`

### 3. Chat command handler (new or add to existing)
- `/rl <id>` - sets instance ID before starting training
- Example: `/rl 0` for viewer, `/rl 1`, `/rl 2` for training instances

## Python Changes

### 4. puck_env.py
- Add `instance_id` parameter to `PuckEnv.__init__()`
- Update `SharedMemoryConfig` to use `f"PuckRL_{instance_id}"`

### 5. train.py
- Add `--num-envs` argument
- Add `--instance-offset` argument (default 1, so training uses 1,2,3... and 0 is for viewer)
- Use `SubprocVecEnv` with multiple `PuckEnv` instances

### 6. viewer.py (new script)
- Connects to instance 0
- Loads latest checkpoint from training run
- Runs deterministic inference at 1x speed
- Periodically reloads model (every N episodes or minutes)
- Sets time scale to 1x on the game

## Usage Flow

```bash
# Terminal 1: Start viewer script
python viewer.py --run-dir ./runs/puck_PPO_xxx

# Terminal 2: Start training (uses instances 1, 2, 3)
python train.py --num-envs 3 --instance-offset 1

# In each game instance (launch 4 copies of Puck):
#   Game 0: /rl 0  then F9 (viewer - will run at 1x, deterministic)
#   Game 1: /rl 1  then F9 (training)
#   Game 2: /rl 2  then F9 (training)
#   Game 3: /rl 3  then F9 (training)
```

## Architecture Diagram

```
Game Instance 0 (Viewer)     Game Instances 1-3 (Training)
    1x speed, deterministic       30x speed, stochastic
           |                              |
    SharedMem "PuckRL_0"          SharedMem "PuckRL_1", "PuckRL_2", "PuckRL_3"
           |                              |
    viewer.py (inference)         train.py (SubprocVecEnv)
           |                              |
    Loads latest checkpoint <---- Saves checkpoints periodically
```

## Implementation Order

1. C# instance ID support (RLController + RLBridge + chat command)
2. Python env instance ID (puck_env.py)
3. Multi-env training (train.py with SubprocVecEnv)
4. Viewer script (viewer.py)
5. Testing with 2 instances first, then scale up

## Notes

- Start with 2 instances to test, scale up based on PC performance
- Each game instance uses ~1-2GB RAM + GPU resources
- Training instances can run at 30x+ speed
- Viewer instance stays at 1x for watchability
- Viewer reloads model periodically to show training progress
