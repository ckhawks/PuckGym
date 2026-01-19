# Training Specification

## Observation Space

14 normalized floats sent each step:

| Index | Field | Raw Range | Normalization |
|-------|-------|-----------|---------------|
| 0 | skater_x | ~[-25, 25] | /30 |
| 1 | skater_y | ~[-30, 30] | /30 |
| 2 | skater_vel_x | ~[-20, 20] | /20 |
| 3 | skater_vel_y | ~[-20, 20] | /20 |
| 4 | skater_rotation | [-π, π] | /π |
| 5 | puck_x | ~[-25, 25] | /30 |
| 6 | puck_y | ~[-30, 30] | /30 |
| 7 | puck_height | [0, ~3] | /2 |
| 8 | puck_vel_x | ~[-20, 20] | /20 |
| 9 | puck_vel_y | ~[-20, 20] | /20 |
| 10 | puck_vel_height | ~[-10, 10] | /10 |
| 11 | stick_x | ~[-25, 25] | /30 |
| 12 | stick_y | ~[-30, 30] | /30 |
| 13 | stick_height | [0, ~3] | /2 |

**Coordinate system:** X = left/right, Y = forward/back (Unity Z), Height = vertical (Unity Y)

**Not included:** Goal positions (static), game time, scores (agent learns direction from rewards)

## Action Space

8 continuous values:

| Index | Field | Range | Description |
|-------|-------|-------|-------------|
| 0 | move_x | [-1, 1] | Turn left/right |
| 1 | move_y | [-1, 1] | Skate forward/back |
| 2 | aim_x | [-1, 1] | Stick vertical angle (-25° to 80°) |
| 3 | aim_y | [-1, 1] | Stick horizontal angle (-92.5° to 92.5°) |
| 4 | blade_angle | [-1, 1] | Blade rotation (-4 to 4) |
| 5 | jump | [0, 1] | Jump (>0.5 = pressed) |
| 6 | crouch | [0, 1] | Slide/crouch (>0.5 = pressed) |
| 7 | boost | [0, 1] | Sprint (>0.5 = pressed) |

**Action repeat:** 5 physics ticks per decision (~10 Hz at 50 tick physics)

## Reward Function

### Sparse Rewards (via GoalTrigger patch)

| Event | Reward |
|-------|--------|
| Goal scored | +100.0 |
| Goal conceded | -100.0 |

### Dense Rewards (per step)

| Event | Reward | Notes |
|-------|--------|-------|
| Approach puck | +0.02 × Δdist | Closer = positive |
| Puck toward goal | +0.02 × Δdist | Puck moving toward their goal |
| Puck in scoring position | +0.002 | In front of goal, <15 units away |
| Puck in offensive zone | +0.001 | Puck on their half |
| Shot toward goal | +0.05 × quality × speed | `quality` = dot(puck_vel, to_goal), `speed` capped at 1.0 |
| Stick-puck contact | +0.1 | Once per contact (not per frame) |
| Time penalty | -0.001 | Encourages urgency |

### Reward Scaling

Assuming 60-second episodes at 10 Hz = 600 steps:
- Max shaping rewards: ~5-10 per episode
- Goal scored: +100 (dominates when it happens)
- Goal conceded: -100

## Episode Structure

- **Max duration:** 120 seconds (2 minutes)
- **Ends on:** Goal scored, goal conceded, or time limit
- **On reset:** Puck and player teleported to random positions (no phase change)
- **Game clock:** Set to 60 minutes to prevent period transitions

## Shared Memory Protocol

Binary struct, 111 bytes, little-endian:

```
[0-3]    4 bytes   control flags (obs_ready, action_ready, done, reset_flag)
[4-7]    float     reward
[8-91]   21 floats observations (skater 5, puck 6, stick 3, goals 4, game 3)
[92-107] 4 floats  actions (move_x, move_y, aim_x, aim_y)
[108-110] 3 bytes  buttons (jump, crouch, boost)
```

**Handshake:**
1. C# writes observation, sets `obs_ready=1`
2. Python reads observation, writes action, sets `action_ready=1`
3. C# reads action, clears flags, applies action
4. Repeat
