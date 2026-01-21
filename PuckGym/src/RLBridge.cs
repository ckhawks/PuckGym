using System;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Threading;
using UnityEngine;

namespace PuckGym;

/// <summary>
/// Shared memory bridge for RL training.
/// Communicates with Python training script via memory-mapped files.
/// </summary>
public class RLBridge : IDisposable
{
    // =============================================================
    // SHARED STATE STRUCT - MUST MATCH PYTHON EXACTLY
    // =============================================================
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct SharedState
    {
        // Control flags
        public byte obs_ready;       // C# sets to 1 when observation written
        public byte action_ready;    // Python sets to 1 when action written
        public byte done;            // 1 = episode ended
        public byte reset_flag;      // Python sets to 1 to request reset

        // Reward for this step
        public float reward;

        // === OBSERVATION DATA (16 floats - matches PuckCapture v3) ===
        // Skater (player we're controlling)
        public float skater_x;           // 0
        public float skater_z;           // 1  (forward axis, was "y")
        public float skater_y;           // 2  (height)
        public float skater_vel_x;       // 3
        public float skater_vel_z;       // 4  (forward velocity, was "y")
        public float skater_vel_y;       // 5  (vertical velocity)
        public float skater_rotation;    // 6

        // Puck
        public float puck_x;             // 7
        public float puck_z;             // 8  (forward, was "y")
        public float puck_y;             // 9  (height, was "height")
        public float puck_vel_x;         // 10
        public float puck_vel_z;         // 11 (forward, was "y")
        public float puck_vel_y;         // 12 (vertical, was "height")

        // Stick blade position
        public float stick_x;            // 13
        public float stick_z;            // 14 (forward, was "y")
        public float stick_y;            // 15 (height, was "height")

        // === EXTRA STATE (not part of 16-obs, for debug/reward) ===
        // Goals
        public float our_goal_x;
        public float our_goal_z;
        public float their_goal_x;
        public float their_goal_z;

        // Game state
        public float game_time;
        public float our_score;
        public float their_score;

        // === ACTION DATA (Python writes these) ===
        public float move_x;         // -1 to 1 (turn left/right)
        public float move_y;         // -1 to 1 (forward/back)
        public float aim_x;          // -1 to 1 (stick vertical angle)
        public float aim_y;          // -1 to 1 (stick horizontal angle)
        public float blade_angle;    // -1 to 1 (blade rotation, maps to -4 to 4)
        public byte jump;            // 0 or 1 (spacebar)
        public byte crouch;          // 0 or 1 (ctrl)
        public byte boost;           // 0 or 1 (shift)

        // === CONFIG (Python writes) ===
        public float time_scale;     // Requested time scale (0 = use default)
    }

    // =============================================================
    // CONFIGURATION
    // =============================================================
    private const string SHARED_MEMORY_NAME_BASE = "PuckRL";
    private const int SHARED_MEMORY_SIZE = 4096;
    private const int SPINLOCK_YIELD_INTERVAL = 100;
    private const int TIMEOUT_SPINS = 10_000_000;

    // =============================================================
    // STATE
    // =============================================================
    private int _instanceId;
    private string _memoryName;
    private MemoryMappedFile _mmf;
    private MemoryMappedViewAccessor _accessor;
    private SharedState _state;
    private bool _initialized;
    private bool _disposed;

    // Episode tracking
    private int _episodeSteps;
    private float _episodeReward;
    private int _totalEpisodes;

    public bool IsInitialized => _initialized;
    public int EpisodeSteps => _episodeSteps;
    public float EpisodeReward => _episodeReward;
    public int TotalEpisodes => _totalEpisodes;

    // =============================================================
    // SINGLETON
    // =============================================================
    private static RLBridge _instance;
    public static RLBridge Instance => _instance ??= new RLBridge();

    private RLBridge() { }

    // =============================================================
    // INITIALIZATION
    // =============================================================
    public bool Initialize(int instanceId = 0)
    {
        if (_initialized) return true;

        try
        {
            _instanceId = instanceId;
            _memoryName = $"{SHARED_MEMORY_NAME_BASE}_{_instanceId}";

            _mmf = MemoryMappedFile.CreateOrOpen(_memoryName, SHARED_MEMORY_SIZE);
            _accessor = _mmf.CreateViewAccessor();

            _state = new SharedState();
            WriteState();

            _initialized = true;
            _episodeSteps = 0;
            _episodeReward = 0;
            _totalEpisodes = 0;

            Plugin.Log($"RLBridge initialized. Instance: {_instanceId}, Memory: '{_memoryName}', Size: {Marshal.SizeOf<SharedState>()} bytes");
            return true;
        }
        catch (Exception ex)
        {
            Plugin.LogError($"RLBridge init failed: {ex.Message}");
            return false;
        }
    }

    // =============================================================
    // CHECK FOR RESET REQUEST (non-blocking)
    // =============================================================
    public bool CheckResetRequested()
    {
        if (!_initialized) return false;

        ReadState();
        if (_state.reset_flag == 1)
        {
            _state.reset_flag = 0;
            _episodeSteps = 0;
            _episodeReward = 0;
            _totalEpisodes++;
            WriteState();
            return true;
        }
        return false;
    }

    // =============================================================
    // SEND OBSERVATION (non-blocking)
    // =============================================================
    public void SendObservation(
        Vector3 skaterPos, Vector3 skaterVel, float skaterRot,
        Vector3 puckPos, Vector3 puckVel,
        Vector3 stickPos,
        Vector3 ourGoal, Vector3 theirGoal,
        float gameTime, int ourScore, int theirScore,
        float reward, bool done)
    {
        if (!_initialized) return;

        // Skater (obs 0-6)
        _state.skater_x = skaterPos.x;
        _state.skater_z = skaterPos.z;     // Forward axis
        _state.skater_y = skaterPos.y;     // Height
        _state.skater_vel_x = skaterVel.x;
        _state.skater_vel_z = skaterVel.z; // Forward velocity
        _state.skater_vel_y = skaterVel.y; // Vertical velocity
        _state.skater_rotation = skaterRot;

        // Puck (obs 7-12)
        _state.puck_x = puckPos.x;
        _state.puck_z = puckPos.z;         // Forward axis
        _state.puck_y = puckPos.y;         // Height
        _state.puck_vel_x = puckVel.x;
        _state.puck_vel_z = puckVel.z;     // Forward velocity
        _state.puck_vel_y = puckVel.y;     // Vertical velocity

        // Stick (obs 13-15)
        _state.stick_x = stickPos.x;
        _state.stick_z = stickPos.z;       // Forward axis
        _state.stick_y = stickPos.y;       // Height

        // Extra state (not part of 16-obs)
        _state.our_goal_x = ourGoal.x;
        _state.our_goal_z = ourGoal.z;
        _state.their_goal_x = theirGoal.x;
        _state.their_goal_z = theirGoal.z;

        _state.game_time = gameTime;
        _state.our_score = ourScore;
        _state.their_score = theirScore;

        _state.reward = reward;
        _state.done = done ? (byte)1 : (byte)0;

        _state.obs_ready = 1;
        _state.action_ready = 0;
        WriteState();

        _episodeSteps++;
        _episodeReward += reward;

        if (done)
        {
            Plugin.Log($"Episode {_totalEpisodes} ended: {_episodeSteps} steps, reward: {_episodeReward:F2}");
        }
    }

    // =============================================================
    // WAIT FOR ACTION (blocking)
    // =============================================================
    public RLAction WaitForAction()
    {
        if (!_initialized) return RLAction.Neutral;

        int spins = 0;
        while (spins < TIMEOUT_SPINS)
        {
            ReadState();
            if (_state.action_ready == 1)
            {
                _state.obs_ready = 0;
                _state.action_ready = 0;
                WriteState();

                // Apply time scale if Python requested one
                if (_state.time_scale > 0)
                {
                    UnityEngine.Time.timeScale = _state.time_scale;
                }

                return new RLAction
                {
                    MoveX = _state.move_x,
                    MoveY = _state.move_y,
                    AimX = _state.aim_x,
                    AimY = _state.aim_y,
                    BladeAngle = _state.blade_angle,
                    Jump = _state.jump == 1,
                    Crouch = _state.crouch == 1,
                    Boost = _state.boost == 1
                };
            }

            spins++;
            if (spins % SPINLOCK_YIELD_INTERVAL == 0)
                Thread.SpinWait(10);
        }

        Plugin.LogError("Timeout waiting for Python action!");
        return RLAction.Neutral;
    }

    // =============================================================
    // HELPERS
    // =============================================================
    private void ReadState()
    {
        _accessor.Read(0, out _state);
    }

    private void WriteState()
    {
        _accessor.Write(0, ref _state);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _accessor?.Dispose();
        _mmf?.Dispose();
        _initialized = false;

        Plugin.Log("RLBridge disposed.");
    }
}

// =============================================================
// ACTION STRUCT
// =============================================================
public struct RLAction
{
    public float MoveX;      // -1 to 1 (turn left/right)
    public float MoveY;      // -1 to 1 (forward/back)
    public float AimX;       // -1 to 1 (stick vertical angle)
    public float AimY;       // -1 to 1 (stick horizontal angle)
    public float BladeAngle; // -1 to 1 (blade rotation)
    public bool Jump;        // spacebar
    public bool Crouch;      // ctrl
    public bool Boost;       // shift

    public static RLAction Neutral => new RLAction
    {
        MoveX = 0, MoveY = 0, AimX = 0, AimY = 0, BladeAngle = 0,
        Jump = false, Crouch = false, Boost = false
    };
}
