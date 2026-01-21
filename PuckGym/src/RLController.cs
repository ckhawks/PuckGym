using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Netcode;
using UnityEngine;

namespace PuckGym;

/// <summary>
/// Main controller for RL training.
/// Hooks into Puck's game systems to gather observations and apply actions.
/// </summary>
public class RLController : MonoBehaviour
{
    // =============================================================
    // CONFIGURATION
    // =============================================================
    private bool _trainingEnabled = false;

    // Curriculum learning - enable stick control to interact with puck
    private const bool ENABLE_STICK_CONTROL = true;

    // Max episode length in seconds (reset to keep training fresh)
    private const float MAX_EPISODE_SECONDS = 120f;
    private float _episodeTimer = 0f;

    // Action repeat - only communicate with Python every N physics ticks
    // Higher = less noise, faster training, but less precise control
    private const int ACTION_REPEAT = 5;  // ~10Hz if physics is 50Hz
    private int _tickCounter = 0;
    private float _accumulatedReward = 0f;
    private RLAction _currentAction = RLAction.Neutral;

    // Time scale control
    private float _timeScale = 1f;
    private const float MIN_TIME_SCALE = 0.25f;
    private const float MAX_TIME_SCALE = 100f;
    private const float TIME_SCALE_STEP = 10f;

    // Stick angle limits (from PlayerInput defaults)
    // X = vertical angle (up/down), Y = horizontal angle (left/right)
    private Vector2 _minStickAngle = new Vector2(-25f, -92.5f);
    private Vector2 _maxStickAngle = new Vector2(80f, 92.5f);

    // Track if we're waiting for game to reset
    private bool _waitingForPlayableState = false;

    // Goal event tracking (set by GoalTrigger patch)
    private float _pendingGoalReward = 0f;
    private bool _goalScoredThisFrame = false;

    public bool IsTrainingEnabled => _trainingEnabled;

    // =============================================================
    // CACHED REFERENCES
    // =============================================================
    private Player _localPlayer;
    private PlayerInput _playerInput;
    private PlayerBodyV2 _playerBody;
    private Goal[] _goals;

    // =============================================================
    // REWARD TRACKING
    // =============================================================
    private float _prevDistToPuck;
    private float _prevPuckDistToGoal;
    private int _prevOurScore;
    private int _prevTheirScore;
    private Vector3 _prevPuckPos;
    private bool _hadPuckContact;

    // =============================================================
    // SINGLETON
    // =============================================================
    private static RLController _instance;
    public static RLController Instance => _instance;

    public static RLController Create()
    {
        if (_instance != null) return _instance;

        var go = new GameObject("RLController");
        DontDestroyOnLoad(go);
        _instance = go.AddComponent<RLController>();
        return _instance;
    }

    // =============================================================
    // LIFECYCLE
    // =============================================================
    void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(gameObject);
            return;
        }
        _instance = this;
    }

    void Start()
    {
        Plugin.Log("RLController started. Press F9 to toggle training.");
        FindGoals();

        // Register game state change handler to keep clock at 60 minutes during training
        try
        {
            MonoBehaviourSingleton<EventManager>.Instance?.AddEventListener(
                "Event_OnGameStateChanged", OnGameStateChanged);
        }
        catch (Exception e)
        {
            Plugin.LogError($"Failed to register game state listener: {e.Message}");
        }
    }

    void OnDestroy()
    {
        // Unregister event listener
        try
        {
            MonoBehaviourSingleton<EventManager>.Instance?.RemoveEventListener(
                "Event_OnGameStateChanged", OnGameStateChanged);
        }
        catch { }

        if (_instance == this)
            _instance = null;
    }

    /// <summary>
    /// Called when game state changes. Sets clock to 60 minutes during Playing phase to avoid intermissions.
    /// </summary>
    private void OnGameStateChanged(Dictionary<string, object> eventData)
    {
        if (!_trainingEnabled) return;

        try
        {
            var gameManager = NetworkBehaviourSingleton<GameManager>.Instance;
            if (gameManager == null) return;

            var currentState = gameManager.GameState.Value;

            // Only set 60-minute clock during Playing phase
            // Don't interfere with FaceOff countdown or other phase timers
            if (currentState.Phase == GamePhase.Playing)
            {
                gameManager.Server_UpdateGameState(
                    currentState.Phase,
                    3600,  // 60 minutes
                    currentState.Period,
                    currentState.BlueScore,
                    currentState.RedScore
                );

                Plugin.Log("Set game clock to 60 minutes");
            }
        }
        catch (Exception e)
        {
            Plugin.LogError($"Failed to set game time: {e.Message}");
        }
    }

    // =============================================================
    // PUBLIC METHODS (called from RLKeybinds)
    // =============================================================
    public void ToggleTraining()
    {
        _trainingEnabled = !_trainingEnabled;
        Plugin.Log($"RL Training: {(_trainingEnabled ? "ENABLED" : "DISABLED")}");

        if (_trainingEnabled)
        {
            if (!RLBridge.Instance.IsInitialized)
                RLBridge.Instance.Initialize();

            RefreshPlayerReferences();
            FindGoals();

            // Randomize player and puck positions
            DespawnAllPucks();
            SpawnPuckAtRandomPosition();
            RandomizePlayerPosition();
            ResetTrackingState();
            Plugin.Log("Training started - randomized player and puck positions");
        }
        else
        {
            // Reset time scale when disabling training
            ResetTimeScale();
        }
    }

    public void IncreaseTimeScale()
    {
        _timeScale = Mathf.Min(_timeScale + TIME_SCALE_STEP, MAX_TIME_SCALE);
        Time.timeScale = _timeScale;
        Plugin.Log($"Time scale: {_timeScale:F1}x");
    }

    public void DecreaseTimeScale()
    {
        _timeScale = Mathf.Max(_timeScale - TIME_SCALE_STEP, MIN_TIME_SCALE);
        Time.timeScale = _timeScale;
        Plugin.Log($"Time scale: {_timeScale:F1}x");
    }

    public void ResetTimeScale()
    {
        _timeScale = 1f;
        Time.timeScale = _timeScale;
        Plugin.Log($"Time scale reset to {_timeScale:F1}x");
    }

    /// <summary>
    /// Debug method to test goal detection - launches puck into enemy goal.
    /// </summary>
    public void DebugScoreGoal()
    {
        if (!_trainingEnabled)
        {
            Plugin.Log("DebugScoreGoal: Training not enabled");
            return;
        }

        try
        {
            var puck = GetNearestPuck();
            if (puck == null)
            {
                Plugin.Log("DebugScoreGoal: No puck found");
                return;
            }

            // Get enemy goal position
            var (_, theirGoal) = GetGoalPositions();

            // Position puck in front of goal and shoot it in
            float offsetZ = theirGoal.z > 0 ? -3f : 3f;  // 3 units in front of goal
            Vector3 startPos = new Vector3(theirGoal.x, 0.1f, theirGoal.z + offsetZ);
            Vector3 velocity = new Vector3(0, 0, theirGoal.z > 0 ? 20f : -20f);  // Shoot toward goal

            puck.Rigidbody.position = startPos;
            puck.Rigidbody.linearVelocity = velocity;

            Plugin.Log($"DebugScoreGoal: Launched puck from {startPos} toward goal at {velocity}");
        }
        catch (Exception e)
        {
            Plugin.LogError($"DebugScoreGoal failed: {e.Message}");
        }
    }

    /// <summary>
    /// Called by GoalTrigger patch when puck enters a goal during training.
    /// </summary>
    public void OnGoalScored(PlayerTeam goalTeam, Puck puck)
    {
        if (!_trainingEnabled) return;

        PlayerTeam ourTeam = GetPlayerTeam();

        // If puck entered the other team's goal, WE scored
        // If puck entered our team's goal, THEY scored
        if (goalTeam != ourTeam)
        {
            // We scored!
            _pendingGoalReward = 100.0f;
            Plugin.Log("GOAL SCORED! +100 reward");
        }
        else
        {
            // They scored on us
            _pendingGoalReward = -100.0f;
            Plugin.Log("GOAL CONCEDED! -100 reward");
        }

        _goalScoredThisFrame = true;

        // Update game score display to track goals during training
        UpdateScoreDisplay(goalTeam);
    }

    /// <summary>
    /// Update the game's score display to track training goals.
    /// Goal scored on Red = Blue scores, goal scored on Blue = Red scores.
    /// </summary>
    private void UpdateScoreDisplay(PlayerTeam goalTeam)
    {
        try
        {
            var gameManager = NetworkBehaviourSingleton<GameManager>.Instance;
            if (gameManager == null) return;

            var currentState = gameManager.GameState.Value;
            int blueScore = currentState.BlueScore;
            int redScore = currentState.RedScore;

            // Goal scored ON a team means the other team scores
            if (goalTeam == PlayerTeam.Red)
                blueScore++;
            else
                redScore++;

            gameManager.Server_UpdateGameState(
                currentState.Phase,
                currentState.Time,
                currentState.Period,
                blueScore,
                redScore
            );
        }
        catch (Exception e)
        {
            Plugin.LogError($"Failed to update score: {e.Message}");
        }
    }

    // =============================================================
    // UPDATE LOOP
    // =============================================================
    void Update()
    {
        // Refresh player references if we lose them
        if (_trainingEnabled && (_localPlayer == null || _playerInput == null))
        {
            RefreshPlayerReferences();
        }
    }

    void FixedUpdate()
    {
        if (!_trainingEnabled || !RLBridge.Instance.IsInitialized) return;
        if (_localPlayer == null || _playerInput == null) return;

        // Check for reset request from Python
        if (RLBridge.Instance.CheckResetRequested())
        {
            RequestGameReset();
        }

        // If waiting for game to return to playable state, check and skip
        if (_waitingForPlayableState)
        {
            if (IsGamePlayable())
            {
                _waitingForPlayableState = false;

                // Refresh references - game may have respawned player
                RefreshPlayerReferences();

                ResetTrackingState();
                RandomizePuckPosition();
                RandomizePlayerPosition();
                Plugin.Log("Game now playable, randomized positions, resuming training");
            }
            else
            {
                // Still waiting - don't run RL loop
                return;
            }
        }

        // Don't run RL during non-playable phases
        if (!IsGamePlayable())
        {
            return;
        }

        // Track episode time
        _episodeTimer += Time.fixedDeltaTime;

        // Always apply current action (keeps player moving between decision points)
        ApplyAction(_currentAction);

        // Gather observation for reward calculation
        var obs = GatherObservation();

        // Calculate and accumulate reward
        float reward = CalculateReward(obs);
        _accumulatedReward += reward;

        // Check if goal was scored this frame (set by GoalTrigger patch)
        bool goalScored = _goalScoredThisFrame;
        if (goalScored)
        {
            _accumulatedReward += _pendingGoalReward;
            _goalScoredThisFrame = false;
            _pendingGoalReward = 0f;
        }

        // Check if episode done
        bool timeExpired = _episodeTimer >= MAX_EPISODE_SECONDS;
        bool done = goalScored || timeExpired;

        if (timeExpired && !goalScored)
        {
            Plugin.Log($"Episode time limit ({MAX_EPISODE_SECONDS}s) reached, resetting");
        }

        // Increment tick counter
        _tickCounter++;

        // Only communicate with Python every ACTION_REPEAT ticks (or if done)
        if (_tickCounter >= ACTION_REPEAT || done)
        {
            // Send observation with accumulated reward
            RLBridge.Instance.SendObservation(
                obs.SkaterPos, obs.SkaterVel, obs.SkaterRotation,
                obs.PuckPos, obs.PuckVel,
                obs.StickPos,
                obs.OurGoal, obs.TheirGoal,
                obs.GameTime, obs.OurScore, obs.TheirScore,
                _accumulatedReward, done
            );

            // Wait for action from Python (blocking)
            _currentAction = RLBridge.Instance.WaitForAction();

            // Reset counters
            _tickCounter = 0;
            _accumulatedReward = 0f;
        }

        // Update tracking state
        UpdatePreviousState(obs);

        // Handle episode end
        if (done)
        {
            // Respawn puck and player for new episode (no phase change needed)
            RandomizePuckPosition();
            RandomizePlayerPosition();
            ResetTrackingState();
        }
    }

    /// <summary>
    /// Check if game is in a state where we can actually play
    /// </summary>
    private bool IsGamePlayable()
    {
        if (NetworkBehaviourSingleton<GameManager>.Instance == null)
            return false;

        var phase = NetworkBehaviourSingleton<GameManager>.Instance.Phase;

        // Only run during actual gameplay phases
        return phase == GamePhase.Playing ||
               phase == GamePhase.Warmup ||
               phase == GamePhase.FaceOff;
    }

    /// <summary>
    /// Request the game to reset to a playable state
    /// </summary>
    private void RequestGameReset()
    {
        try
        {
            if (NetworkBehaviourSingleton<GameManager>.Instance != null)
            {
                // Go directly to Playing to skip faceoff countdown
                NetworkBehaviourSingleton<GameManager>.Instance.Server_SetPhase(GamePhase.Playing);
                Plugin.Log("Set Playing phase (skipped faceoff)");
            }
        }
        catch (Exception e)
        {
            Plugin.LogError($"Failed to set phase: {e.Message}");
        }

        // Randomize positions immediately
        RandomizePuckPosition();
        RandomizePlayerPosition();
        ResetTrackingState();
    }

    /// <summary>
    /// Reset internal tracking state for new episode
    /// </summary>
    private void ResetTrackingState()
    {
        _prevDistToPuck = 0;
        _prevPuckDistToGoal = 0;
        _prevOurScore = 0;
        _prevTheirScore = 0;
        _hadPuckContact = false;

        // Reset goal tracking
        _goalScoredThisFrame = false;
        _pendingGoalReward = 0f;

        // Reset action repeat state
        _tickCounter = 0;
        _accumulatedReward = 0f;
        _currentAction = RLAction.Neutral;

        // Reset episode timer
        _episodeTimer = 0f;

        if (_playerInput != null)
        {
            _playerInput.ResetInputs(false);
        }
    }

    /// <summary>
    /// Despawn all pucks on the field
    /// </summary>
    private void DespawnAllPucks()
    {
        try
        {
            var puckManager = NetworkBehaviourSingleton<PuckManager>.Instance;
            if (puckManager == null) return;

            puckManager.Server_DespawnPucks(false);
        }
        catch (Exception e)
        {
            Plugin.LogError($"Failed to despawn pucks: {e.Message}");
        }
    }

    /// <summary>
    /// Spawn a single puck at a random position on the rink, away from the player and goals
    /// </summary>
    private void SpawnPuckAtRandomPosition()
    {
        try
        {
            var puckManager = NetworkBehaviourSingleton<PuckManager>.Instance;
            var levelManager = NetworkBehaviourSingleton<LevelManager>.Instance;
            if (puckManager == null) return;

            Vector3 spawnPos;
            Vector3 playerPos = _playerBody != null ? _playerBody.transform.position : Vector3.zero;
            float minDistFromPlayer = 8f; // Don't spawn puck within 8 units of player
            float minDistFromGoal = 7f;   // Don't spawn puck within 7 units of goals (physics issues)

            // Get goal positions
            var (ourGoal, theirGoal) = GetGoalPositions();

            if (levelManager != null)
            {
                // Use actual ice bounds with margin for walls and corners
                Bounds ice = levelManager.IceBounds;
                float margin = 4f; // Stay away from walls
                float cornerCut = 6f; // Extra margin in corners

                // Generate random position within bounds
                float halfX = ice.extents.x - margin;
                float halfZ = ice.extents.z - margin;

                // Keep trying until we get a valid position (not in corner, not near player, not near goals)
                int attempts = 0;
                do
                {
                    float randomX = UnityEngine.Random.Range(-halfX, halfX);
                    float randomZ = UnityEngine.Random.Range(-halfZ, halfZ);
                    spawnPos = new Vector3(ice.center.x + randomX, 0.1f, ice.center.z + randomZ);
                    attempts++;

                    // Check if in corner region (both X and Z near edges)
                    bool inCorner = Mathf.Abs(randomX) > (halfX - cornerCut) &&
                                    Mathf.Abs(randomZ) > (halfZ - cornerCut);

                    // Check distance from player
                    float distToPlayer = Vector3.Distance(
                        new Vector3(spawnPos.x, 0, spawnPos.z),
                        new Vector3(playerPos.x, 0, playerPos.z)
                    );
                    bool tooCloseToPlayer = distToPlayer < minDistFromPlayer;

                    // Check distance from goals (prevent physics explosions)
                    float distToOurGoal = Vector3.Distance(
                        new Vector3(spawnPos.x, 0, spawnPos.z),
                        new Vector3(ourGoal.x, 0, ourGoal.z)
                    );
                    float distToTheirGoal = Vector3.Distance(
                        new Vector3(spawnPos.x, 0, spawnPos.z),
                        new Vector3(theirGoal.x, 0, theirGoal.z)
                    );
                    bool tooCloseToGoal = distToOurGoal < minDistFromGoal || distToTheirGoal < minDistFromGoal;

                    if ((!inCorner && !tooCloseToPlayer && !tooCloseToGoal) || attempts > 30) break;
                } while (true);
            }
            else
            {
                // Fallback to hardcoded values
                int attempts = 0;
                do
                {
                    float randomX = UnityEngine.Random.Range(-12f, 12f);
                    float randomZ = UnityEngine.Random.Range(-20f, 20f);
                    spawnPos = new Vector3(randomX, 0.1f, randomZ);
                    attempts++;

                    float distToPlayer = Vector3.Distance(
                        new Vector3(spawnPos.x, 0, spawnPos.z),
                        new Vector3(playerPos.x, 0, playerPos.z)
                    );

                    // Check distance from goals
                    float distToOurGoal = Vector3.Distance(
                        new Vector3(spawnPos.x, 0, spawnPos.z),
                        new Vector3(ourGoal.x, 0, ourGoal.z)
                    );
                    float distToTheirGoal = Vector3.Distance(
                        new Vector3(spawnPos.x, 0, spawnPos.z),
                        new Vector3(theirGoal.x, 0, theirGoal.z)
                    );
                    bool tooCloseToGoal = distToOurGoal < minDistFromGoal || distToTheirGoal < minDistFromGoal;

                    if ((distToPlayer >= minDistFromPlayer && !tooCloseToGoal) || attempts > 30) break;
                } while (true);
            }

            puckManager.Server_SpawnPuck(spawnPos, Quaternion.identity, Vector3.zero, false);
        }
        catch (Exception e)
        {
            Plugin.LogError($"Failed to spawn puck: {e.Message}");
        }
    }

    /// <summary>
    /// Respawn the puck at a random position (despawn current, spawn new)
    /// </summary>
    private void RandomizePuckPosition()
    {
        DespawnAllPucks();
        SpawnPuckAtRandomPosition();
    }

    /// <summary>
    /// Teleport the player to a random position on the rink, away from goals
    /// </summary>
    private void RandomizePlayerPosition()
    {
        try
        {
            if (_playerBody == null) return;

            var levelManager = NetworkBehaviourSingleton<LevelManager>.Instance;
            Vector3 spawnPos;
            float randomRotation;
            float minDistFromGoal = 7f; // Don't spawn player within 7 units of goals

            // Get goal positions
            var (ourGoal, theirGoal) = GetGoalPositions();

            if (levelManager != null)
            {
                Bounds ice = levelManager.IceBounds;
                float margin = 5f;
                float cornerCut = 7f;

                float halfX = ice.extents.x - margin;
                float halfZ = ice.extents.z - margin;

                int attempts = 0;
                do
                {
                    float randomX = UnityEngine.Random.Range(-halfX, halfX);
                    float randomZ = UnityEngine.Random.Range(-halfZ, halfZ);
                    spawnPos = new Vector3(ice.center.x + randomX, 0.1f, ice.center.z + randomZ);
                    attempts++;

                    bool inCorner = Mathf.Abs(randomX) > (halfX - cornerCut) &&
                                    Mathf.Abs(randomZ) > (halfZ - cornerCut);

                    // Check distance from goals
                    float distToOurGoal = Vector3.Distance(
                        new Vector3(spawnPos.x, 0, spawnPos.z),
                        new Vector3(ourGoal.x, 0, ourGoal.z)
                    );
                    float distToTheirGoal = Vector3.Distance(
                        new Vector3(spawnPos.x, 0, spawnPos.z),
                        new Vector3(theirGoal.x, 0, theirGoal.z)
                    );
                    bool tooCloseToGoal = distToOurGoal < minDistFromGoal || distToTheirGoal < minDistFromGoal;

                    if ((!inCorner && !tooCloseToGoal) || attempts > 20) break;
                } while (true);
            }
            else
            {
                int attempts = 0;
                do
                {
                    float randomX = UnityEngine.Random.Range(-10f, 10f);
                    float randomZ = UnityEngine.Random.Range(-18f, 18f);
                    spawnPos = new Vector3(randomX, 0.1f, randomZ);
                    attempts++;

                    // Check distance from goals (fallback positions)
                    float distToOurGoal = Vector3.Distance(
                        new Vector3(spawnPos.x, 0, spawnPos.z),
                        new Vector3(ourGoal.x, 0, ourGoal.z)
                    );
                    float distToTheirGoal = Vector3.Distance(
                        new Vector3(spawnPos.x, 0, spawnPos.z),
                        new Vector3(theirGoal.x, 0, theirGoal.z)
                    );
                    bool tooCloseToGoal = distToOurGoal < minDistFromGoal || distToTheirGoal < minDistFromGoal;

                    if (!tooCloseToGoal || attempts > 20) break;
                } while (true);
            }

            // Random facing direction
            randomRotation = UnityEngine.Random.Range(0f, 360f);

            _playerBody.Server_Teleport(spawnPos, Quaternion.Euler(0, randomRotation, 0));
        }
        catch (Exception e)
        {
            Plugin.LogError($"Failed to randomize player: {e.Message}");
        }
    }

    // =============================================================
    // REFERENCE MANAGEMENT
    // =============================================================
    private void RefreshPlayerReferences()
    {
        try
        {
            // Get PlayerManager singleton
            if (NetworkBehaviourSingleton<PlayerManager>.Instance == null)
            {
                return;
            }

            _localPlayer = NetworkBehaviourSingleton<PlayerManager>.Instance.GetLocalPlayer();

            if (_localPlayer != null)
            {
                _playerInput = _localPlayer.PlayerInput;
                _playerBody = _localPlayer.PlayerBody;

                // Get stick angle limits from PlayerInput if available
                if (_playerInput != null)
                {
                    // These are the limits the game uses
                    _minStickAngle = _playerInput.MinimumStickRaycastOriginAngle;
                    _maxStickAngle = _playerInput.MaximumStickRaycastOriginAngle;
                }

                Plugin.Log($"Got local player: {_localPlayer.name}");
            }
        }
        catch (Exception e)
        {
            // Silently fail - game may not be fully loaded yet
            Plugin.LogError($"RefreshPlayerReferences: {e.Message}");
        }
    }

    private void FindGoals()
    {
        _goals = FindObjectsOfType<Goal>();
        if (_goals.Length > 0)
        {
            Plugin.Log($"Found {_goals.Length} goals");
        }
    }

    // =============================================================
    // OBSERVATION GATHERING
    // =============================================================
    private GameObservation GatherObservation()
    {
        var obs = new GameObservation();

        // Player body position and velocity
        if (_playerBody != null && _playerBody.Rigidbody != null)
        {
            obs.SkaterPos = _playerBody.transform.position;
            obs.SkaterVel = _playerBody.Rigidbody.linearVelocity;
            obs.SkaterRotation = _playerBody.transform.eulerAngles.y * Mathf.Deg2Rad;
        }

        // Puck position and velocity
        Puck puck = GetNearestPuck();
        if (puck != null)
        {
            obs.PuckPos = puck.transform.position;
            obs.PuckVel = puck.Rigidbody.linearVelocity;
        }

        // Stick blade position
        if (_localPlayer != null && _localPlayer.Stick != null)
        {
            obs.StickPos = _localPlayer.Stick.BladeHandlePosition;
        }

        // Goal positions
        var (ourGoal, theirGoal) = GetGoalPositions();
        obs.OurGoal = ourGoal;
        obs.TheirGoal = theirGoal;

        // Game state
        if (NetworkBehaviourSingleton<GameManager>.Instance != null)
        {
            var gameState = NetworkBehaviourSingleton<GameManager>.Instance.GameState.Value;
            obs.GameTime = gameState.Time;
            obs.OurScore = GetOurScore(gameState);
            obs.TheirScore = GetTheirScore(gameState);
        }

        return obs;
    }

    private Puck GetNearestPuck()
    {
        try
        {
            if (NetworkBehaviourSingleton<PuckManager>.Instance == null)
                return null;

            var pucks = NetworkBehaviourSingleton<PuckManager>.Instance.GetPucks(false);
            if (pucks == null || pucks.Count == 0)
                return null;

            // In warmup there are multiple pucks - get nearest one
            if (_playerBody == null)
                return pucks.FirstOrDefault();

            Vector3 playerPos = _playerBody.transform.position;
            return pucks
                .Where(p => p != null)
                .OrderBy(p => Vector3.Distance(p.transform.position, playerPos))
                .FirstOrDefault();
        }
        catch
        {
            return null;
        }
    }

    private (Vector3 ourGoal, Vector3 theirGoal) GetGoalPositions()
    {
        // Default positions if we can't find goals
        Vector3 blueGoal = new Vector3(0, 0, -26f);  // Typical blue goal Z
        Vector3 redGoal = new Vector3(0, 0, 26f);   // Typical red goal Z

        if (_goals != null && _goals.Length >= 2)
        {
            foreach (var goal in _goals)
            {
                if (goal == null) continue;
                // Goals are sorted by their Z position typically
                if (goal.transform.position.z < 0)
                    blueGoal = goal.transform.position;
                else
                    redGoal = goal.transform.position;
            }
        }

        // Determine which is "our" goal based on player team
        PlayerTeam ourTeam = GetPlayerTeam();
        if (ourTeam == PlayerTeam.Blue)
        {
            return (blueGoal, redGoal);  // We defend blue, attack red
        }
        else
        {
            return (redGoal, blueGoal);  // We defend red, attack blue
        }
    }

    private PlayerTeam GetPlayerTeam()
    {
        if (_localPlayer != null)
        {
            return _localPlayer.Team.Value;
        }
        return PlayerTeam.Blue; // Default
    }

    private int GetOurScore(GameState gameState)
    {
        return GetPlayerTeam() == PlayerTeam.Blue ? gameState.BlueScore : gameState.RedScore;
    }

    private int GetTheirScore(GameState gameState)
    {
        return GetPlayerTeam() == PlayerTeam.Blue ? gameState.RedScore : gameState.BlueScore;
    }

    // =============================================================
    // REWARD FUNCTION
    // =============================================================
    private float CalculateReward(GameObservation obs)
    {
        float reward = 0f;

        // NOTE: Goal scored/conceded rewards are now handled by OnGoalScored()
        // which is called from the GoalTrigger Harmony patch

        // === DENSE REWARDS (shaping) ===

        // Distance to puck (encourage getting close)
        float distToPuck = Vector3.Distance(
            new Vector3(obs.SkaterPos.x, 0, obs.SkaterPos.z),
            new Vector3(obs.PuckPos.x, 0, obs.PuckPos.z)
        );

        // Reward for getting closer to puck
        if (_prevDistToPuck > 0)
        {
            float puckApproach = _prevDistToPuck - distToPuck;
            reward += puckApproach * 0.02f;
        }

        // Puck distance to their goal (reward puck moving toward goal)
        float puckToGoal = Vector3.Distance(
            new Vector3(obs.PuckPos.x, 0, obs.PuckPos.z),
            new Vector3(obs.TheirGoal.x, 0, obs.TheirGoal.z)
        );
        float goalZ = obs.TheirGoal.z;
        float puckZ = obs.PuckPos.z;

        if (_prevPuckDistToGoal > 0)
        {
            float goalApproach = _prevPuckDistToGoal - puckToGoal;
            reward += goalApproach * 0.02f;
        }

        // Reward for puck being IN FRONT of goal (not behind it)
        bool puckInFrontOfGoal;
        if (goalZ > 0)
        {
            puckInFrontOfGoal = puckZ < goalZ && puckZ > 0;
        }
        else
        {
            puckInFrontOfGoal = puckZ > goalZ && puckZ < 0;
        }

        // Small bonus for puck in front of goal and close (reduced from 0.01)
        if (puckInFrontOfGoal && puckToGoal < 15f)
        {
            reward += 0.002f;
        }

        // Small bonus for puck in offensive zone (reduced from 0.005)
        bool puckInOffensiveZone = (goalZ > 0 && puckZ > 0) || (goalZ < 0 && puckZ < 0);
        if (puckInOffensiveZone)
        {
            reward += 0.001f;
        }

        // Reward for puck moving fast TOWARD the goal (encourages shooting)
        Vector3 puckVel = new Vector3(obs.PuckVel.x, 0, obs.PuckVel.z);
        float puckSpeed = puckVel.magnitude;
        if (puckSpeed > 1f)  // Only if puck is actually moving
        {
            Vector3 toGoal = new Vector3(obs.TheirGoal.x - obs.PuckPos.x, 0, obs.TheirGoal.z - obs.PuckPos.z).normalized;
            float shotQuality = Vector3.Dot(puckVel.normalized, toGoal);  // 1 = toward goal, -1 = away
            if (shotQuality > 0)
            {
                // Reward scales with speed and direction toward goal
                reward += shotQuality * Mathf.Min(puckSpeed / 20f, 1f) * 0.05f;
            }
        }

        // Reward for touching puck (hitting it with stick)
        // Keep this small to avoid farming taps instead of actually playing
        if (_localPlayer != null && _localPlayer.Stick != null)
        {
            Puck puck = GetNearestPuck();
            if (puck != null && puck.IsTouchingStick && puck.TouchingStick == _localPlayer.Stick)
            {
                if (!_hadPuckContact)
                {
                    reward += 0.1f; // Small bonus for making contact
                    _hadPuckContact = true;
                }
            }
            else
            {
                _hadPuckContact = false;
            }
        }

        // Small time penalty to encourage urgency
        reward -= 0.001f;

        return reward;
    }

    private void UpdatePreviousState(GameObservation obs)
    {
        _prevDistToPuck = Vector3.Distance(
            new Vector3(obs.SkaterPos.x, 0, obs.SkaterPos.z),
            new Vector3(obs.PuckPos.x, 0, obs.PuckPos.z)
        );

        _prevPuckDistToGoal = Vector3.Distance(
            new Vector3(obs.PuckPos.x, 0, obs.PuckPos.z),
            new Vector3(obs.TheirGoal.x, 0, obs.TheirGoal.z)
        );

        _prevOurScore = obs.OurScore;
        _prevTheirScore = obs.TheirScore;
        _prevPuckPos = obs.PuckPos;
    }

    // =============================================================
    // EPISODE END CHECK
    // =============================================================
    private bool CheckEpisodeDone()
    {
        if (NetworkBehaviourSingleton<GameManager>.Instance == null)
            return false;

        var phase = NetworkBehaviourSingleton<GameManager>.Instance.Phase;

        // Episode ends on goal, game over, or period over
        return phase == GamePhase.BlueScore ||
               phase == GamePhase.RedScore ||
               phase == GamePhase.GameOver ||
               phase == GamePhase.PeriodOver;
    }

    // =============================================================
    // ACTION APPLICATION
    // =============================================================
    private void ApplyAction(RLAction action)
    {
        if (_playerInput == null) return;

        // Movement: MoveX = turn left/right (-1 to 1), MoveY = forward/back (-1 to 1)
        _playerInput.MoveInput.ClientValue = new Vector2(action.MoveX, action.MoveY);

        // Stick control
        if (ENABLE_STICK_CONTROL)
        {
            // Stick position (mouse aim equivalent)
            float stickAngleX = Mathf.Lerp(_minStickAngle.x, _maxStickAngle.x, (action.AimX + 1f) / 2f);
            float stickAngleY = Mathf.Lerp(_minStickAngle.y, _maxStickAngle.y, (action.AimY + 1f) / 2f);
            _playerInput.StickRaycastOriginAngleInput.ClientValue = new Vector2(stickAngleX, stickAngleY);

            // Blade rotation (scroll wheel equivalent) - maps -1..1 to -4..4
            sbyte bladeAngle = (sbyte)Mathf.RoundToInt(action.BladeAngle * 4f);
            _playerInput.BladeAngleInput.ClientValue = bladeAngle;
        }

        // Jump (JumpInput expects byte)
        _playerInput.JumpInput.ClientValue = (byte)(action.Jump ? 1 : 0);

        // Crouch/Slide
        _playerInput.SlideInput.ClientValue = action.Crouch;

        // Boost/Sprint
        _playerInput.SprintInput.ClientValue = action.Boost;
    }

}

// =============================================================
// OBSERVATION STRUCT
// =============================================================
public struct GameObservation
{
    public Vector3 SkaterPos;
    public Vector3 SkaterVel;
    public float SkaterRotation;
    public Vector3 PuckPos;
    public Vector3 PuckVel;
    public Vector3 StickPos;
    public Vector3 OurGoal;
    public Vector3 TheirGoal;
    public float GameTime;
    public int OurScore;
    public int TheirScore;
}
