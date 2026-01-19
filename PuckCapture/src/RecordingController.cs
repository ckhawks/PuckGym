using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Netcode;
using UnityEngine;
using UnityEngine.Rendering;

namespace PuckCapture;

/// <summary>
/// Main controller for recording human demonstrations.
/// Captures observations and human inputs at 50Hz.
/// </summary>
public class RecordingController : MonoBehaviour
{
    // =============================================================
    // CONFIGURATION
    // =============================================================
    private bool _recordingEnabled = false;
    private const float MAX_EPISODE_SECONDS = 120f;
    private const float CAPTURE_RATE_HZ = 50f;
    private const float MIN_DIST_FROM_GOAL = 7f;
    private const float MIN_DIST_FROM_PLAYER = 8f;

    private float _episodeTimer = 0f;
    private float _captureTimer = 0f;
    private float _captureInterval => 1f / CAPTURE_RATE_HZ;

    public bool IsRecording => _recordingEnabled;

    // =============================================================
    // CACHED REFERENCES
    // =============================================================
    private Player _localPlayer;
    private PlayerInput _playerInput;
    private PlayerBodyV2 _playerBody;
    private Goal[] _goals;

    // X mark indicator at blue goal
    private static AssetBundle _xmarkAssetBundle;
    private static GameObject _xmarkPrefab;
    private GameObject _xmarkInstance;

    // Player identity
    private string _steamId = "unknown";
    private string _username = "unknown";

    // Stick angle limits for normalization
    private Vector2 _minStickAngle = new Vector2(-25f, -92.5f);
    private Vector2 _maxStickAngle = new Vector2(80f, 92.5f);

    // =============================================================
    // EPISODE DATA
    // =============================================================
    private EpisodeRecorder _currentEpisode;
    private int _totalEpisodes = 0;
    private int _successfulEpisodes = 0;

    // Total recording time (across all episodes)
    private float _totalRecordingTime = 0f;

    // =============================================================
    // SINGLETON
    // =============================================================
    private static RecordingController _instance;
    public static RecordingController Instance => _instance;

    public static RecordingController Create()
    {
        if (_instance != null) return _instance;

        var go = new GameObject("RecordingController");
        DontDestroyOnLoad(go);
        _instance = go.AddComponent<RecordingController>();
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
        Plugin.Log("RecordingController started. Press F7/Numpad0/Insert to toggle recording.");
        FindGoals();
        LoadXMarkPrefab();
    }

    void OnDestroy()
    {
        // Save any in-progress episode
        if (_currentEpisode != null)
        {
            _currentEpisode.Discard();
            _currentEpisode = null;
        }

        // Clean up X mark
        DespawnXMark();

        if (_instance == this)
            _instance = null;
    }

    // =============================================================
    // PUBLIC METHODS
    // =============================================================
    public void ToggleRecording()
    {
        _recordingEnabled = !_recordingEnabled;

        if (_recordingEnabled)
        {
            RefreshPlayerReferences();
            FindGoals();

            if (!ValidateSettings())
            {
                _recordingEnabled = false;
                return;
            }

            // Initialize recording time at 1 second (0 can cause phase issues)
            _totalRecordingTime = 1f;

            // Start first episode
            StartNewEpisode();
            SpawnXMark();
            Plugin.ChatMessage($"<color=#00FF00><b>Recording STARTED</b></color> - {_username}");
            Plugin.ChatMessage("<color=#AAAAAA>Score goals to record demonstrations!</color>");
            Plugin.Log($"Steam ID: {_steamId}, Recordings: {CaptureConfig.RecordingsFolder}");
        }
        else
        {
            // Discard current episode if recording disabled
            if (_currentEpisode != null)
            {
                _currentEpisode.Discard();
                _currentEpisode = null;
            }
            DespawnXMark();
            Plugin.ChatMessage($"<color=#FF6666><b>Recording STOPPED</b></color> - {_successfulEpisodes} goals recorded");
        }
    }

    public void ResetEpisode()
    {
        if (!_recordingEnabled) return;

        Plugin.ChatMessage("<color=#FFFF00>Episode reset</color>");

        // Save current episode as manual reset
        if (_currentEpisode != null)
        {
            _currentEpisode.Save(EpisodeOutcome.Manual);
            _totalEpisodes++;
        }

        StartNewEpisode();
    }

    public void OnGoalScored(PlayerTeam goalTeam, Puck puck)
    {
        if (!_recordingEnabled || _currentEpisode == null) return;

        PlayerTeam ourTeam = GetPlayerTeam();

        // Check if WE scored (puck entered their goal)
        if (goalTeam != ourTeam)
        {
            _currentEpisode.Save(EpisodeOutcome.Goal);
            _totalEpisodes++;
            _successfulEpisodes++;
            Plugin.ChatMessage($"<color=#FFD700><b>GOAL!</b></color> Demo #{_successfulEpisodes} saved");
        }
        else
        {
            _currentEpisode.Save(EpisodeOutcome.Timeout);
            _totalEpisodes++;
            Plugin.Log("Own goal - episode saved as timeout");
        }

        StartNewEpisode();
    }

    // =============================================================
    // UPDATE LOOP
    // =============================================================
    void Update()
    {
        if (_recordingEnabled && (_localPlayer == null || _playerInput == null))
        {
            RefreshPlayerReferences();
        }

        // Update clock every frame to overpower game's clock updates
        if (_recordingEnabled && IsGamePlayable())
        {
            UpdateGameClock();
        }
    }

    void FixedUpdate()
    {
        if (!_recordingEnabled || _currentEpisode == null) return;
        if (_localPlayer == null || _playerInput == null) return;

        // Check if game is playable
        if (!IsGamePlayable()) return;

        // Track episode time and total recording time
        _episodeTimer += Time.fixedDeltaTime;
        _totalRecordingTime += Time.fixedDeltaTime;
        _captureTimer += Time.fixedDeltaTime;

        // Check timeout
        if (_episodeTimer >= MAX_EPISODE_SECONDS)
        {
            Plugin.ChatMessage("<color=#FFA500>Timeout - resetting</color>");
            _currentEpisode.Save(EpisodeOutcome.Timeout);
            _totalEpisodes++;
            StartNewEpisode();
            return;
        }

        // Capture at fixed rate
        if (_captureTimer >= _captureInterval)
        {
            _captureTimer -= _captureInterval;
            CaptureStep();
        }
    }

    // =============================================================
    // CAPTURE LOGIC
    // =============================================================
    private void CaptureStep()
    {
        // Gather observation
        float[] obs = GatherObservation();

        // Gather human action
        float[] action = GatherHumanAction();

        // Calculate reward (for potential weighted BC)
        float reward = CalculateReward(obs);

        // Record step
        _currentEpisode.AddStep(obs, action, reward);
    }

    private float[] GatherObservation()
    {
        // World position limit is ±50 units (from SynchronizedObjectManager encoding)
        const float POS_LIMIT = 50f;
        const float VEL_LIMIT = 20f;
        const float HEIGHT_LIMIT = 5f;

        float[] obs = new float[16];

        // Player body position, velocity, and rotation
        if (_playerBody != null && _playerBody.Rigidbody != null)
        {
            Vector3 pos = _playerBody.transform.position;
            Vector3 vel = _playerBody.Rigidbody.linearVelocity;
            float rot = _playerBody.transform.eulerAngles.y * Mathf.Deg2Rad;

            obs[0] = pos.x / POS_LIMIT;         // skater_x
            obs[1] = pos.z / POS_LIMIT;         // skater_z (forward/back)
            obs[2] = pos.y / HEIGHT_LIMIT;      // skater_y (height)
            obs[3] = vel.x / VEL_LIMIT;         // skater_vel_x
            obs[4] = vel.z / VEL_LIMIT;         // skater_vel_z
            obs[5] = vel.y / VEL_LIMIT;         // skater_vel_y (vertical)
            obs[6] = rot / Mathf.PI;            // skater_rotation
        }

        // Puck position and velocity
        Puck puck = GetNearestPuck();
        if (puck != null)
        {
            Vector3 pos = puck.transform.position;
            Vector3 vel = puck.Rigidbody.linearVelocity;

            obs[7] = pos.x / POS_LIMIT;         // puck_x
            obs[8] = pos.z / POS_LIMIT;         // puck_z
            obs[9] = pos.y / HEIGHT_LIMIT;      // puck_y (height)
            obs[10] = vel.x / VEL_LIMIT;        // puck_vel_x
            obs[11] = vel.z / VEL_LIMIT;        // puck_vel_z
            obs[12] = vel.y / VEL_LIMIT;        // puck_vel_y
        }

        // Stick blade position
        if (_localPlayer != null && _localPlayer.Stick != null)
        {
            Vector3 stickPos = _localPlayer.Stick.BladeHandlePosition;
            obs[13] = stickPos.x / POS_LIMIT;   // stick_x
            obs[14] = stickPos.z / POS_LIMIT;   // stick_z
            obs[15] = stickPos.y / HEIGHT_LIMIT;// stick_y (height)
        }

        return obs;
    }

    private float[] GatherHumanAction()
    {
        float[] action = new float[8];

        if (_playerInput == null) return action;

        // Movement input
        Vector2 move = _playerInput.MoveInput.ClientValue;
        action[0] = move.x;  // move_x (-1 to 1)
        action[1] = move.y;  // move_y (-1 to 1)

        // Stick aim - normalize from angle range to [-1, 1]
        Vector2 stickAngle = _playerInput.StickRaycastOriginAngleInput.ClientValue;
        action[2] = NormalizeAngle(stickAngle.x, _minStickAngle.x, _maxStickAngle.x);  // aim_x
        action[3] = NormalizeAngle(stickAngle.y, _minStickAngle.y, _maxStickAngle.y);  // aim_y

        // Blade angle - normalize from [-4, 4] to [-1, 1]
        sbyte bladeAngle = _playerInput.BladeAngleInput.ClientValue;
        action[4] = bladeAngle / 4f;  // blade_angle

        // Buttons - as floats [0, 1]
        // Use InputManager for jump since JumpInput is a counter, not a state
        bool jumpHeld = MonoBehaviourSingleton<InputManager>.Instance?.JumpAction?.IsPressed() ?? false;
        action[5] = jumpHeld ? 1f : 0f;                                 // jump
        action[6] = _playerInput.SlideInput.ClientValue ? 1f : 0f;      // crouch
        action[7] = _playerInput.SprintInput.ClientValue ? 1f : 0f;     // boost

        return action;
    }

    private float NormalizeAngle(float angle, float min, float max)
    {
        // Map angle from [min, max] to [-1, 1]
        return (angle - min) / (max - min) * 2f - 1f;
    }

    private float CalculateReward(float[] obs)
    {
        // Simple reward: negative distance to puck (closer = higher reward)
        // This can be used for weighted BC
        float skaterX = obs[0] * 30f;
        float skaterZ = obs[1] * 30f;
        float puckX = obs[5] * 30f;
        float puckZ = obs[6] * 30f;

        float dist = Mathf.Sqrt((skaterX - puckX) * (skaterX - puckX) + (skaterZ - puckZ) * (skaterZ - puckZ));
        return -dist * 0.01f;  // Small negative reward based on distance
    }

    /// <summary>
    /// Update the game clock to show total recording time (counting up).
    /// Also maintains the goal count in BlueScore and shows "RECORDING" as phase.
    /// Directly sets UI labels to bypass the game's clock system.
    /// </summary>
    private void UpdateGameClock()
    {
        try
        {
            // Keep the game in Playing phase so player can move
            var gameManager = NetworkBehaviourSingleton<GameManager>.Instance;
            if (gameManager != null)
            {
                gameManager.Server_SetPhase(GamePhase.Playing);
            }

            // Directly set UI labels
            if (UIGameState.Instance != null)
            {
                UIGameState.Instance.SetGameTime(_totalRecordingTime);
                UIGameState.Instance.SetBlueTeamScore(_successfulEpisodes);
                UIGameState.Instance.SetRedTeamScore(0);
                UIGameState.Instance.SetGamePhase("<color=#FF0000>RECORDING</color>");
            }
        }
        catch
        {
            // Silently fail - this runs every frame
        }
    }

    // =============================================================
    // EPISODE MANAGEMENT
    // =============================================================
    private void StartNewEpisode()
    {
        // Get player settings for metadata
        byte team = (byte)(GetPlayerTeam() == PlayerTeam.Blue ? 0 : 1);
        byte handedness = GetHandedness();

        _currentEpisode = new EpisodeRecorder(_steamId, _username, team, handedness);

        // Ensure we're in Playing phase so the player can move freely
        try
        {
            var gameManager = NetworkBehaviourSingleton<GameManager>.Instance;
            if (gameManager != null)
            {
                gameManager.Server_SetPhase(GamePhase.Playing);
            }
        }
        catch { }

        // Randomize positions
        DespawnAllPucks();
        SpawnPuckAtRandomPosition();
        RandomizePlayerPosition();

        _episodeTimer = 0f;
        _captureTimer = 0f;

        Plugin.Log($"New episode started (#{_totalEpisodes + 1})");
    }

    // =============================================================
    // VALIDATION
    // =============================================================
    private bool ValidateSettings()
    {
        // Check team
        PlayerTeam team = GetPlayerTeam();
        if (team != PlayerTeam.Blue)
        {
            Plugin.ChatMessage($"<color=#FF0000><b>ERROR:</b> Wrong team! You are {team}, need BLUE</color>");
            Plugin.ChatMessage("<color=#FF6666>Switch to Blue team and try again</color>");
            return false;
        }

        // Check role - must be Attacker, not Goalie
        if (_localPlayer != null && _localPlayer.Role.Value != PlayerRole.Attacker)
        {
            Plugin.ChatMessage($"<color=#FF0000><b>ERROR:</b> Wrong role! You are {_localPlayer.Role.Value}, need ATTACKER</color>");
            Plugin.ChatMessage("<color=#FF6666>Switch to Attacker role and try again</color>");
            return false;
        }

        // Check handedness
        if (_localPlayer != null && _localPlayer.Handedness.Value != PlayerHandedness.Right)
        {
            Plugin.ChatMessage($"<color=#FF0000><b>ERROR:</b> Wrong handedness! Need RIGHT-HANDED</color>");
            Plugin.ChatMessage("<color=#FF6666>Change to right-handed in settings and try again</color>");
            return false;
        }

        return true;
    }

    private byte GetHandedness()
    {
        if (_localPlayer != null)
        {
            // 0 = Right, 1 = Left
            return (byte)(_localPlayer.Handedness.Value == PlayerHandedness.Right ? 0 : 1);
        }
        return 0; // Default to right
    }

    // =============================================================
    // SPAWN LOGIC (copied from PuckGym)
    // =============================================================
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

    private void SpawnPuckAtRandomPosition()
    {
        try
        {
            var puckManager = NetworkBehaviourSingleton<PuckManager>.Instance;
            var levelManager = NetworkBehaviourSingleton<LevelManager>.Instance;
            if (puckManager == null) return;

            Vector3 spawnPos;
            Vector3 playerPos = _playerBody != null ? _playerBody.transform.position : Vector3.zero;
            var (ourGoal, theirGoal) = GetGoalPositions();

            if (levelManager != null)
            {
                Bounds ice = levelManager.IceBounds;
                float margin = 4f;
                float cornerCut = 6f;
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

                    float distToPlayer = Vector3.Distance(
                        new Vector3(spawnPos.x, 0, spawnPos.z),
                        new Vector3(playerPos.x, 0, playerPos.z)
                    );
                    bool tooCloseToPlayer = distToPlayer < MIN_DIST_FROM_PLAYER;

                    float distToOurGoal = Vector3.Distance(
                        new Vector3(spawnPos.x, 0, spawnPos.z),
                        new Vector3(ourGoal.x, 0, ourGoal.z)
                    );
                    float distToTheirGoal = Vector3.Distance(
                        new Vector3(spawnPos.x, 0, spawnPos.z),
                        new Vector3(theirGoal.x, 0, theirGoal.z)
                    );
                    bool tooCloseToGoal = distToOurGoal < MIN_DIST_FROM_GOAL || distToTheirGoal < MIN_DIST_FROM_GOAL;

                    if ((!inCorner && !tooCloseToPlayer && !tooCloseToGoal) || attempts > 30) break;
                } while (true);
            }
            else
            {
                spawnPos = new Vector3(
                    UnityEngine.Random.Range(-12f, 12f),
                    0.1f,
                    UnityEngine.Random.Range(-15f, 15f)
                );
            }

            puckManager.Server_SpawnPuck(spawnPos, Quaternion.identity, Vector3.zero, false);
        }
        catch (Exception e)
        {
            Plugin.LogError($"Failed to spawn puck: {e.Message}");
        }
    }

    private void RandomizePlayerPosition()
    {
        try
        {
            if (_playerBody == null) return;

            var levelManager = NetworkBehaviourSingleton<LevelManager>.Instance;
            var (ourGoal, theirGoal) = GetGoalPositions();
            Vector3 spawnPos;

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

                    float distToOurGoal = Vector3.Distance(
                        new Vector3(spawnPos.x, 0, spawnPos.z),
                        new Vector3(ourGoal.x, 0, ourGoal.z)
                    );
                    float distToTheirGoal = Vector3.Distance(
                        new Vector3(spawnPos.x, 0, spawnPos.z),
                        new Vector3(theirGoal.x, 0, theirGoal.z)
                    );
                    bool tooCloseToGoal = distToOurGoal < MIN_DIST_FROM_GOAL || distToTheirGoal < MIN_DIST_FROM_GOAL;

                    if ((!inCorner && !tooCloseToGoal) || attempts > 20) break;
                } while (true);
            }
            else
            {
                spawnPos = new Vector3(
                    UnityEngine.Random.Range(-10f, 10f),
                    0.1f,
                    UnityEngine.Random.Range(-15f, 15f)
                );
            }

            float randomRotation = UnityEngine.Random.Range(0f, 360f);
            _playerBody.Server_Teleport(spawnPos, Quaternion.Euler(0, randomRotation, 0));
        }
        catch (Exception e)
        {
            Plugin.LogError($"Failed to randomize player: {e.Message}");
        }
    }

    // =============================================================
    // HELPERS
    // =============================================================
    private void RefreshPlayerReferences()
    {
        try
        {
            if (NetworkBehaviourSingleton<PlayerManager>.Instance == null) return;

            _localPlayer = NetworkBehaviourSingleton<PlayerManager>.Instance.GetLocalPlayer();

            if (_localPlayer != null)
            {
                _playerInput = _localPlayer.PlayerInput;
                _playerBody = _localPlayer.PlayerBody;

                if (_playerInput != null)
                {
                    _minStickAngle = _playerInput.MinimumStickRaycastOriginAngle;
                    _maxStickAngle = _playerInput.MaximumStickRaycastOriginAngle;
                }

                // Get player identity
                try
                {
                    _steamId = _localPlayer.SteamId.Value.ToString();
                    _username = _localPlayer.Username.Value.ToString();
                }
                catch
                {
                    // Keep defaults if not available
                }
            }
        }
        catch (Exception e)
        {
            Plugin.LogError($"RefreshPlayerReferences: {e.Message}");
        }
    }

    private void FindGoals()
    {
        _goals = FindObjectsOfType<Goal>();
    }

    private bool IsGamePlayable()
    {
        if (NetworkBehaviourSingleton<GameManager>.Instance == null) return false;
        var phase = NetworkBehaviourSingleton<GameManager>.Instance.Phase;
        return phase == GamePhase.Playing || phase == GamePhase.Warmup || phase == GamePhase.FaceOff;
    }

    private PlayerTeam GetPlayerTeam()
    {
        if (_localPlayer != null) return _localPlayer.Team.Value;
        return PlayerTeam.Blue;
    }

    private (Vector3 ourGoal, Vector3 theirGoal) GetGoalPositions()
    {
        Vector3 blueGoal = new Vector3(0, 0, -26f);
        Vector3 redGoal = new Vector3(0, 0, 26f);

        if (_goals != null && _goals.Length >= 2)
        {
            foreach (var goal in _goals)
            {
                if (goal == null) continue;
                if (goal.transform.position.z < 0)
                    blueGoal = goal.transform.position;
                else
                    redGoal = goal.transform.position;
            }
        }

        PlayerTeam ourTeam = GetPlayerTeam();
        if (ourTeam == PlayerTeam.Blue)
            return (blueGoal, redGoal);
        else
            return (redGoal, blueGoal);
    }

    private Puck GetNearestPuck()
    {
        try
        {
            if (NetworkBehaviourSingleton<PuckManager>.Instance == null) return null;
            var pucks = NetworkBehaviourSingleton<PuckManager>.Instance.GetPucks(false);
            if (pucks == null || pucks.Count == 0) return null;

            if (_playerBody == null) return pucks.FirstOrDefault();

            Vector3 playerPos = _playerBody.transform.position;
            return pucks.Where(p => p != null)
                        .OrderBy(p => Vector3.Distance(p.transform.position, playerPos))
                        .FirstOrDefault();
        }
        catch
        {
            return null;
        }
    }

    // =============================================================
    // X MARK INDICATOR
    // =============================================================
    private void LoadXMarkPrefab()
    {
        if (_xmarkPrefab != null) return;

        try
        {
            if (_xmarkAssetBundle == null)
            {
                _xmarkAssetBundle = PrefabHelper.LoadAssetBundle("xmark");
            }

            if (_xmarkAssetBundle != null)
            {
                _xmarkPrefab = PrefabHelper.LoadPrefab(_xmarkAssetBundle, "assets/toaster's rink/xmark.fbx");
            }
        }
        catch (Exception e)
        {
            Plugin.LogError($"Failed to load X mark prefab: {e.Message}");
        }
    }

    private void SpawnXMark()
    {
        if (_xmarkPrefab == null)
        {
            LoadXMarkPrefab();
            if (_xmarkPrefab == null) return;
        }

        try
        {
            // Spawn the X mark
            _xmarkInstance = UnityEngine.Object.Instantiate(_xmarkPrefab);

            // Fix shaders for URP
            MeshRenderer[] renderers = _xmarkInstance.GetComponentsInChildren<MeshRenderer>();
            foreach (MeshRenderer renderer in renderers)
            {
                foreach (Material mat in renderer.sharedMaterials)
                {
                    if (mat != null)
                    {
                        mat.shader = Shader.Find("Universal Render Pipeline/Lit");
                    }
                }
            }

            // Position at blue goal (Blue's own goal - don't score here)
            Vector3 blueGoalPos = new Vector3(0, 0, -26f);

            // Try to get actual goal position using Team field
            if (_goals != null)
            {
                foreach (var goal in _goals)
                {
                    if (goal == null) continue;

                    // Use reflection to get the Team field
                    var teamField = typeof(Goal).GetField("Team", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
                    if (teamField != null)
                    {
                        PlayerTeam goalTeam = (PlayerTeam)teamField.GetValue(goal);
                        if (goalTeam == PlayerTeam.Blue)
                        {
                            blueGoalPos = goal.transform.position;
                            Plugin.Log($"Found blue goal at {blueGoalPos}");
                            break;
                        }
                    }
                }
            }

            // Position in front of the goal
            _xmarkInstance.transform.position = new Vector3(blueGoalPos.x, 1f, blueGoalPos.z - 2f);
            _xmarkInstance.transform.rotation = Quaternion.Euler(0, 90f, 0);
            _xmarkInstance.transform.localScale = new Vector3(38f, 38f, 38f);

            Plugin.Log("X mark spawned at blue goal (don't score here)");
        }
        catch (Exception e)
        {
            Plugin.LogError($"Failed to spawn X mark: {e.Message}");
        }
    }

    private void DespawnXMark()
    {
        if (_xmarkInstance != null)
        {
            UnityEngine.Object.Destroy(_xmarkInstance);
            _xmarkInstance = null;
            Plugin.Log("X mark despawned");
        }
    }
}
