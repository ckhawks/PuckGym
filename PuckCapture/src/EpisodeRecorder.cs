using System;
using System.Collections.Generic;
using System.IO;

namespace PuckCapture;

/// <summary>
/// Episode outcome for file naming and metadata.
/// </summary>
public enum EpisodeOutcome : byte
{
    Goal = 0,
    Timeout = 1,
    Manual = 2
}

/// <summary>
/// Records a single episode to a binary file.
///
/// File format (version 3):
/// [Header - 76 bytes]
///   0:  uint8   format_version (3)
///   1:  uint8   team (0=Blue, 1=Red)
///   2:  uint8   handedness (0=Right, 1=Left)
///   3:  uint8   outcome (0=Goal, 1=Timeout, 2=Manual)
///   4:  uint32  step_count
///   8:  float   duration_seconds
///   12: int64   timestamp_utc_ms
///   20: char[24] steam_id (null-padded)
///   44: char[32] username (null-padded)
///
/// [Step data - 100 bytes per step]
///   0:  float[16] observations (player pos/vel/rot, puck pos/vel, stick pos)
///   64: float[8]  actions (move, aim, blade, buttons)
///   96: float     reward
///
/// Observations (16 floats, all normalized to ~[-1,1]):
///   0-2:   skater x, z, y (pos/50, pos/50, height/5)
///   3-5:   skater vel x, z, y (/20)
///   6:     skater rotation (/PI)
///   7-9:   puck x, z, y
///   10-12: puck vel x, z, y
///   13-15: stick x, z, y
/// </summary>
public class EpisodeRecorder
{
    private const byte FORMAT_VERSION = 3;
    private const int HEADER_SIZE = 76;
    private const int STEP_SIZE = 100;  // 16 + 8 + 1 = 25 floats * 4 bytes

    private string _steamId;
    private string _username;
    private byte _team;
    private byte _handedness;
    private DateTime _startTime;

    private List<float[]> _observations = new List<float[]>();
    private List<float[]> _actions = new List<float[]>();
    private List<float> _rewards = new List<float>();

    public int StepCount => _observations.Count;

    public EpisodeRecorder(string steamId, string username, byte team, byte handedness)
    {
        _steamId = steamId ?? "unknown";
        _username = username ?? "unknown";
        _team = team;
        _handedness = handedness;
        _startTime = DateTime.UtcNow;
    }

    public void AddStep(float[] observation, float[] action, float reward)
    {
        _observations.Add(observation);
        _actions.Add(action);
        _rewards.Add(reward);
    }

    public void Save(EpisodeOutcome outcome)
    {
        if (_observations.Count == 0)
        {
            Plugin.Log("Episode has no steps, skipping save.");
            return;
        }

        try
        {
            string filename = GenerateFilename(outcome);
            string filepath = Path.Combine(CaptureConfig.RecordingsFolder, filename);

            using (var stream = new FileStream(filepath, FileMode.Create))
            using (var writer = new BinaryWriter(stream))
            {
                // Write header
                WriteHeader(writer, outcome);

                // Write step data
                for (int i = 0; i < _observations.Count; i++)
                {
                    WriteStep(writer, _observations[i], _actions[i], _rewards[i]);
                }
            }

            float duration = (float)(DateTime.UtcNow - _startTime).TotalSeconds;
            Plugin.Log($"Saved episode: {filename} ({_observations.Count} steps, {duration:F1}s)");
        }
        catch (Exception e)
        {
            Plugin.LogError($"Failed to save episode: {e.Message}");
        }
    }

    public void Discard()
    {
        _observations.Clear();
        _actions.Clear();
        _rewards.Clear();
        Plugin.Log("Episode discarded.");
    }

    private void WriteHeader(BinaryWriter writer, EpisodeOutcome outcome)
    {
        float duration = (float)(DateTime.UtcNow - _startTime).TotalSeconds;
        long timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

        writer.Write(FORMAT_VERSION);           // 1 byte
        writer.Write(_team);                    // 1 byte
        writer.Write(_handedness);              // 1 byte
        writer.Write((byte)outcome);            // 1 byte
        writer.Write((uint)_observations.Count);// 4 bytes
        writer.Write(duration);                 // 4 bytes
        writer.Write(timestamp);                // 8 bytes

        // Steam ID - 24 bytes, null-padded
        byte[] steamIdBytes = new byte[24];
        var steamIdChars = System.Text.Encoding.ASCII.GetBytes(_steamId);
        int steamIdLen = Math.Min(steamIdChars.Length, 23);
        Array.Copy(steamIdChars, steamIdBytes, steamIdLen);
        writer.Write(steamIdBytes);             // 24 bytes

        // Username - 32 bytes, null-padded
        byte[] usernameBytes = new byte[32];
        var usernameChars = System.Text.Encoding.ASCII.GetBytes(_username);
        int usernameLen = Math.Min(usernameChars.Length, 31);
        Array.Copy(usernameChars, usernameBytes, usernameLen);
        writer.Write(usernameBytes);            // 32 bytes

        // Total: 76 bytes
    }

    private void WriteStep(BinaryWriter writer, float[] obs, float[] action, float reward)
    {
        // Observations: 16 floats
        for (int i = 0; i < 16; i++)
        {
            writer.Write(i < obs.Length ? obs[i] : 0f);
        }

        // Actions: 8 floats
        for (int i = 0; i < 8; i++)
        {
            writer.Write(i < action.Length ? action[i] : 0f);
        }

        // Reward: 1 float
        writer.Write(reward);
    }

    private string GenerateFilename(EpisodeOutcome outcome)
    {
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
        string outcomeStr = outcome.ToString().ToLower();
        string safeUsername = SanitizeFilename(_username);
        return $"{timestamp}_{_steamId}_{safeUsername}_{outcomeStr}.bin";
    }

    private string SanitizeFilename(string name)
    {
        // Remove characters that are invalid in filenames
        var invalid = Path.GetInvalidFileNameChars();
        foreach (char c in invalid)
        {
            name = name.Replace(c, '_');
        }
        // Limit length
        if (name.Length > 20)
            name = name.Substring(0, 20);
        return name;
    }
}
