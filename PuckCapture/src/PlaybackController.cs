using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using HarmonyLib;
using Unity.Netcode;
using UnityEngine;
using UnityEngine.InputSystem;

namespace PuckCapture;

/// <summary>
/// Plays back recorded demonstrations in-game.
/// Loads .bin files and applies recorded actions to the player.
/// </summary>
public class PlaybackController : MonoBehaviour
{
    // =============================================================
    // CONFIGURATION
    // =============================================================
    private const string RECORDINGS_FOLDER = @"C:\Projects\puck-ai-training\recordings";
    private const float PLAYBACK_RATE_HZ = 50f;
    private float PlaybackInterval => 1f / PLAYBACK_RATE_HZ;

    // =============================================================
    // STATE
    // =============================================================
    private bool _isPlaying = false;
    private List<RecordingFile> _playlist = new List<RecordingFile>();
    private int _currentRecordingIndex = 0;
    private RecordingData _currentRecording = null;
    private int _currentStep = 0;
    private float _playbackTimer = 0f;

    // Current step data to apply (set in Update, applied in Harmony postfix)
    private float[] _currentActions = null;
    private float[] _currentObservations = null;

    public bool IsPlaying => _isPlaying;
    public float[] CurrentActions => _currentActions;

    // =============================================================
    // CACHED REFERENCES
    // =============================================================
    private Player _localPlayer;
    private PlayerInput _playerInput;
    private PlayerBodyV2 _playerBody;

    // Stick angle limits for denormalization
    private Vector2 _minStickAngle = new Vector2(-25f, -92.5f);
    private Vector2 _maxStickAngle = new Vector2(80f, 92.5f);

    // =============================================================
    // SINGLETON
    // =============================================================
    private static PlaybackController _instance;
    public static PlaybackController Instance => _instance;

    public static PlaybackController Create()
    {
        if (_instance != null) return _instance;

        var go = new GameObject("PlaybackController");
        DontDestroyOnLoad(go);
        _instance = go.AddComponent<PlaybackController>();
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

    void OnDestroy()
    {
        _isPlaying = false;
        if (_instance == this)
            _instance = null;
    }

    // =============================================================
    // PUBLIC API
    // =============================================================

    /// <summary>
    /// Start playback from the specified recording file.
    /// Will continue playing subsequent goal recordings chronologically.
    /// </summary>
    public void StartPlayback(string startingFilename = null)
    {
        if (_isPlaying)
        {
            Plugin.ChatMessage("<color=#FF6666>Playback already running. Use /stopplayback first.</color>");
            return;
        }

        // Refresh player references
        RefreshPlayerReferences();
        if (_localPlayer == null || _playerInput == null)
        {
            Plugin.ChatMessage("<color=#FF0000>Cannot start playback - no local player found</color>");
            return;
        }

        // Build playlist of goal recordings
        BuildPlaylist();

        if (_playlist.Count == 0)
        {
            Plugin.ChatMessage("<color=#FF0000>No goal recordings found!</color>");
            return;
        }

        // Find starting index
        _currentRecordingIndex = 0;
        if (!string.IsNullOrEmpty(startingFilename))
        {
            int foundIndex = _playlist.FindIndex(r =>
                r.Filename.Equals(startingFilename, StringComparison.OrdinalIgnoreCase) ||
                r.Filename.Contains(startingFilename));

            if (foundIndex >= 0)
            {
                _currentRecordingIndex = foundIndex;
                Plugin.ChatMessage($"<color=#00FF00>Starting from: {_playlist[foundIndex].Filename}</color>");
            }
            else
            {
                Plugin.ChatMessage($"<color=#FFA500>Recording '{startingFilename}' not found, starting from first</color>");
            }
        }

        Plugin.ChatMessage($"<color=#00FF00><b>PLAYBACK STARTED</b></color> - {_playlist.Count} goal recordings");
        Plugin.ChatMessage($"<color=#AAAAAA>Starting at #{_currentRecordingIndex + 1}: {_playlist[_currentRecordingIndex].Filename}</color>");
        Plugin.ChatMessage("<color=#AAAAAA>Use /stopplayback or press Escape to stop</color>");

        _isPlaying = true;
        LoadCurrentRecording();
    }

    // We no longer disable hardware input - instead we override values in FixedUpdate/LateUpdate
    // This allows chat and other UI to still work

    /// <summary>
    /// Stop playback.
    /// </summary>
    public void StopPlayback()
    {
        if (!_isPlaying) return;

        _isPlaying = false;
        _currentRecording = null;
        _currentStep = 0;
        _playbackTimer = 0f;
        _currentActions = null;
        _currentObservations = null;

        Plugin.ChatMessage("<color=#FF6666><b>PLAYBACK STOPPED</b></color>");
    }

    /// <summary>
    /// Skip to next recording.
    /// </summary>
    public void NextRecording()
    {
        if (!_isPlaying) return;

        _currentRecordingIndex++;
        if (_currentRecordingIndex >= _playlist.Count)
        {
            Plugin.ChatMessage("<color=#FFA500>End of playlist reached</color>");
            StopPlayback();
            return;
        }

        LoadCurrentRecording();
    }

    // =============================================================
    // UPDATE LOOP
    // =============================================================
    void Update()
    {
        // Check for Escape key to stop playback
        if (_isPlaying && Keyboard.current != null && Keyboard.current.escapeKey.wasPressedThisFrame)
        {
            StopPlayback();
            return;
        }

        if (!_isPlaying || _currentRecording == null)
        {
            _currentActions = null;
            _currentObservations = null;
            return;
        }

        // Refresh references if needed
        if (_localPlayer == null || _playerInput == null)
        {
            RefreshPlayerReferences();
            if (_localPlayer == null) return;
        }

        _playbackTimer += Time.deltaTime;

        // Advance step at playback rate
        while (_playbackTimer >= PlaybackInterval && _currentStep < _currentRecording.Steps.Count)
        {
            _playbackTimer -= PlaybackInterval;
            _currentStep++;
        }

        // Store current step data to be applied
        if (_currentStep < _currentRecording.Steps.Count)
        {
            _currentActions = _currentRecording.Steps[_currentStep].Actions;
            _currentObservations = _currentRecording.Steps[_currentStep].Observations;
        }
        else
        {
            _currentActions = null;
            _currentObservations = null;
        }

        // Check if recording finished
        if (_currentStep >= _currentRecording.Steps.Count)
        {
            Plugin.ChatMessage($"<color=#FFD700>Recording complete: {_playlist[_currentRecordingIndex].Filename}</color>");

            // Move to next recording
            _currentRecordingIndex++;
            if (_currentRecordingIndex >= _playlist.Count)
            {
                Plugin.ChatMessage("<color=#00FF00>Playlist complete!</color>");
                StopPlayback();
                return;
            }

            LoadCurrentRecording();
        }
    }

    void LateUpdate()
    {
        // Apply inputs in LateUpdate to override any input processing that happened in Update
        if (_isPlaying && _currentActions != null && _playerInput != null)
        {
            ApplyActions(_currentActions);
        }
    }

    void FixedUpdate()
    {
        // Also apply in FixedUpdate for physics-based input processing
        if (_isPlaying && _currentActions != null && _playerInput != null)
        {
            ApplyActions(_currentActions);

            // Sync positions during physics step
            if (_currentObservations != null && _currentObservations.Length >= 12)
            {
                SyncPuckPosition(_currentObservations);
                SyncPlayerPosition(_currentObservations);
            }
        }
    }

    /// <summary>
    /// Apply the current playback actions to player input and sync puck position.
    /// Called from Harmony postfix AFTER the game reads hardware input.
    /// </summary>
    public void ApplyCurrentActions()
    {
        if (_currentActions == null || _playerInput == null) return;
        ApplyActions(_currentActions);

        // Teleport puck to recorded position
        if (_currentObservations != null && _currentObservations.Length >= 12)
        {
            SyncPuckPosition(_currentObservations);
        }
    }

    /// <summary>
    /// Teleport the puck to match recorded observation.
    /// Observations are normalized: positions /50, heights /5, velocities /20
    /// </summary>
    private void SyncPuckPosition(float[] obs)
    {
        try
        {
            var puckManager = NetworkBehaviourSingleton<PuckManager>.Instance;
            if (puckManager == null) return;

            var pucks = puckManager.GetPucks(false);
            if (pucks == null || pucks.Count == 0) return;

            var puck = pucks[0];
            if (puck == null || puck.Rigidbody == null) return;

            // Denormalize position (obs 7,8,9 = puck x,z,y)
            float puckX = obs[7] * 50f;
            float puckZ = obs[8] * 50f;
            float puckY = obs[9] * 5f;

            // Denormalize velocity (obs 10,11,12 = puck vel x,z,y)
            float velX = obs[10] * 20f;
            float velZ = obs[11] * 20f;
            float velY = obs[12] * 20f;

            // Teleport puck
            puck.transform.position = new Vector3(puckX, puckY, puckZ);
            puck.Rigidbody.linearVelocity = new Vector3(velX, velY, velZ);
        }
        catch
        {
            // Silently fail - runs every frame
        }
    }

    /// <summary>
    /// Teleport the player to match recorded observation.
    /// Observations: 0,1,2 = pos x,z,y; 3,4,5 = vel x,z,y; 6 = rotation
    /// </summary>
    private void SyncPlayerPosition(float[] obs)
    {
        try
        {
            if (_playerBody == null || _playerBody.Rigidbody == null) return;

            // Denormalize position (obs 0,1,2 = skater x,z,y)
            float posX = obs[0] * 50f;
            float posZ = obs[1] * 50f;
            float posY = obs[2] * 5f;

            // Denormalize velocity (obs 3,4,5 = skater vel x,z,y)
            float velX = obs[3] * 20f;
            float velZ = obs[4] * 20f;
            float velY = obs[5] * 20f;

            // Denormalize rotation (obs 6 = rotation, normalized by /π)
            float rotationRad = obs[6] * Mathf.PI;
            float rotationDeg = rotationRad * Mathf.Rad2Deg;

            // Teleport player
            _playerBody.transform.position = new Vector3(posX, posY, posZ);
            _playerBody.transform.rotation = Quaternion.Euler(0, rotationDeg, 0);
            _playerBody.Rigidbody.linearVelocity = new Vector3(velX, velY, velZ);
        }
        catch
        {
            // Silently fail - runs every frame
        }
    }

    // =============================================================
    // RECORDING LOADING
    // =============================================================
    private void BuildPlaylist()
    {
        _playlist.Clear();

        if (!Directory.Exists(RECORDINGS_FOLDER))
        {
            Plugin.LogError($"Recordings folder not found: {RECORDINGS_FOLDER}");
            return;
        }

        var files = Directory.GetFiles(RECORDINGS_FOLDER, "*.bin");
        Plugin.Log($"Found {files.Length} .bin files in {RECORDINGS_FOLDER}");

        foreach (var filepath in files)
        {
            try
            {
                var header = ReadHeader(filepath);

                // Only include goal recordings (outcome = 0)
                if (header.Outcome == 0)
                {
                    _playlist.Add(new RecordingFile
                    {
                        Filepath = filepath,
                        Filename = Path.GetFileName(filepath),
                        Header = header
                    });
                }
            }
            catch (Exception e)
            {
                Plugin.LogError($"Failed to read header of {filepath}: {e.Message}");
            }
        }

        // Sort chronologically by filename (timestamp is at start of filename)
        _playlist = _playlist.OrderBy(r => r.Filename).ToList();

        Plugin.Log($"Playlist built: {_playlist.Count} goal recordings");
    }

    private void LoadCurrentRecording()
    {
        if (_currentRecordingIndex >= _playlist.Count) return;

        var recordingFile = _playlist[_currentRecordingIndex];

        try
        {
            _currentRecording = LoadRecording(recordingFile.Filepath);
            _currentStep = 0;
            _playbackTimer = 0f;

            Plugin.ChatMessage($"<color=#00FFFF>Playing #{_currentRecordingIndex + 1}/{_playlist.Count}: {recordingFile.Header.Username} ({_currentRecording.Steps.Count} steps)</color>");

            // Teleport player to starting position from first observation
            if (_currentRecording.Steps.Count > 0 && _playerBody != null)
            {
                var firstStep = _currentRecording.Steps[0];
                Vector3 startPos = new Vector3(
                    firstStep.Observations[0] * 50f,  // skater_x (denormalize)
                    0.1f,
                    firstStep.Observations[1] * 50f   // skater_z (denormalize)
                );
                float startRot = firstStep.Observations[6] * Mathf.PI * Mathf.Rad2Deg;  // skater_rotation

                _playerBody.Server_Teleport(startPos, Quaternion.Euler(0, startRot, 0));

                // Spawn puck at recorded position
                SpawnPuckAtPosition(new Vector3(
                    firstStep.Observations[7] * 50f,   // puck_x
                    0.1f,
                    firstStep.Observations[8] * 50f    // puck_z
                ));
            }

            // Ensure game is in Playing phase
            try
            {
                var gameManager = NetworkBehaviourSingleton<GameManager>.Instance;
                if (gameManager != null)
                {
                    gameManager.Server_SetPhase(GamePhase.Playing);
                }
            }
            catch { }
        }
        catch (Exception e)
        {
            Plugin.LogError($"Failed to load recording: {e.Message}");
            NextRecording();
        }
    }

    private RecordingHeader ReadHeader(string filepath)
    {
        using (var stream = new FileStream(filepath, FileMode.Open, FileAccess.Read))
        using (var reader = new BinaryReader(stream))
        {
            byte version = reader.ReadByte();
            byte team = reader.ReadByte();
            byte handedness = reader.ReadByte();
            byte outcome = reader.ReadByte();
            uint stepCount = reader.ReadUInt32();
            float duration = reader.ReadSingle();
            long timestamp = reader.ReadInt64();

            string steamId = "unknown";
            string username = "unknown";

            if (version >= 2)
            {
                byte[] steamIdBytes = reader.ReadBytes(24);
                byte[] usernameBytes = reader.ReadBytes(32);
                steamId = System.Text.Encoding.ASCII.GetString(steamIdBytes).TrimEnd('\0');
                username = System.Text.Encoding.ASCII.GetString(usernameBytes).TrimEnd('\0');
            }

            return new RecordingHeader
            {
                Version = version,
                Team = team,
                Handedness = handedness,
                Outcome = outcome,
                StepCount = stepCount,
                Duration = duration,
                Timestamp = timestamp,
                SteamId = steamId,
                Username = username
            };
        }
    }

    private RecordingData LoadRecording(string filepath)
    {
        var recording = new RecordingData();

        using (var stream = new FileStream(filepath, FileMode.Open, FileAccess.Read))
        using (var reader = new BinaryReader(stream))
        {
            // Read header
            byte version = reader.ReadByte();
            reader.ReadByte(); // team
            reader.ReadByte(); // handedness
            reader.ReadByte(); // outcome
            uint stepCount = reader.ReadUInt32();
            reader.ReadSingle(); // duration
            reader.ReadInt64(); // timestamp

            int numObs = 14;
            int headerSize = 52;

            if (version >= 2)
            {
                reader.ReadBytes(24); // steam_id
                reader.ReadBytes(32); // username
                headerSize = 76;
            }

            if (version >= 3)
            {
                numObs = 16;
            }

            // Read steps
            for (int i = 0; i < stepCount; i++)
            {
                var step = new RecordingStep();

                // Read observations
                step.Observations = new float[numObs];
                for (int j = 0; j < numObs; j++)
                {
                    step.Observations[j] = reader.ReadSingle();
                }

                // Read actions (8 floats)
                step.Actions = new float[8];
                for (int j = 0; j < 8; j++)
                {
                    step.Actions[j] = reader.ReadSingle();
                }

                // Read reward
                step.Reward = reader.ReadSingle();

                recording.Steps.Add(step);
            }
        }

        return recording;
    }

    // =============================================================
    // ACTION APPLICATION
    // =============================================================
    private void ApplyActions(float[] actions)
    {
        if (_playerInput == null || actions == null || actions.Length < 8) return;

        // Movement input (already in [-1, 1])
        _playerInput.MoveInput.ClientValue = new Vector2(actions[0], actions[1]);

        // Stick aim - denormalize from [-1, 1] to angle range
        float aimX = DenormalizeAngle(actions[2], _minStickAngle.x, _maxStickAngle.x);
        float aimY = DenormalizeAngle(actions[3], _minStickAngle.y, _maxStickAngle.y);
        _playerInput.StickRaycastOriginAngleInput.ClientValue = new Vector2(aimX, aimY);

        // Blade angle - denormalize from [-1, 1] to [-4, 4]
        sbyte bladeAngle = (sbyte)Mathf.Clamp(Mathf.RoundToInt(actions[4] * 4f), -4, 4);
        _playerInput.BladeAngleInput.ClientValue = bladeAngle;

        // Buttons
        // Jump - need to trigger jump action
        if (actions[5] > 0.5f)
        {
            // Increment jump counter to trigger jump
            _playerInput.JumpInput.ClientValue++;
        }

        // Crouch/Slide
        _playerInput.SlideInput.ClientValue = actions[6] > 0.5f;

        // Sprint/Boost
        _playerInput.SprintInput.ClientValue = actions[7] > 0.5f;
    }

    private float DenormalizeAngle(float normalized, float min, float max)
    {
        // Map from [-1, 1] to [min, max]
        return (normalized + 1f) / 2f * (max - min) + min;
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
            }
        }
        catch (Exception e)
        {
            Plugin.LogError($"RefreshPlayerReferences: {e.Message}");
        }
    }

    private void SpawnPuckAtPosition(Vector3 position)
    {
        try
        {
            var puckManager = NetworkBehaviourSingleton<PuckManager>.Instance;
            if (puckManager == null) return;

            puckManager.Server_DespawnPucks(false);
            puckManager.Server_SpawnPuck(position, Quaternion.identity, Vector3.zero, false);
        }
        catch (Exception e)
        {
            Plugin.LogError($"Failed to spawn puck: {e.Message}");
        }
    }

    // =============================================================
    // DATA STRUCTURES
    // =============================================================
    private class RecordingFile
    {
        public string Filepath;
        public string Filename;
        public RecordingHeader Header;
    }

    private class RecordingHeader
    {
        public byte Version;
        public byte Team;
        public byte Handedness;
        public byte Outcome;
        public uint StepCount;
        public float Duration;
        public long Timestamp;
        public string SteamId;
        public string Username;
    }

    private class RecordingData
    {
        public List<RecordingStep> Steps = new List<RecordingStep>();
    }

    private class RecordingStep
    {
        public float[] Observations;
        public float[] Actions;
        public float Reward;
    }
}
