using System;
using System.IO;
using UnityEngine;

namespace PuckCapture;

/// <summary>
/// Configuration for the capture mod.
/// </summary>
public static class CaptureConfig
{
    private static string RecordingsPath => Path.Combine(Application.dataPath, "..", "Plugins", "PuckCapture", "recordings");

    public static string RecordingsFolder => RecordingsPath;

    public static void Load()
    {
        try
        {
            // Ensure recordings directory exists
            if (!Directory.Exists(RecordingsPath))
            {
                Directory.CreateDirectory(RecordingsPath);
                Plugin.Log($"Created recordings folder: {RecordingsPath}");
            }
        }
        catch (Exception e)
        {
            Plugin.LogError($"Failed to create recordings folder: {e.Message}");
        }
    }
}
