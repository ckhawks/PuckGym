using System;
using System.Linq;
using HarmonyLib;
using UnityEngine;
using UnityEngine.Rendering;

namespace PuckCapture;

public class Plugin : IPuckMod
{
    public static string MOD_NAME = "PuckCapture";
    public static string MOD_VERSION = "1.0.0";
    public static string MOD_GUID = "pw.stellaric.toaster.puckcapture";

    static readonly Harmony harmony = new Harmony(MOD_GUID);

    public bool OnEnable()
    {
        Plugin.Log($"Enabling...");
        try
        {
            if (IsDedicatedServer())
            {
                Plugin.Log("Environment: dedicated server.");
                Plugin.Log($"This mod is designed to be only used only on clients!");
            }
            else
            {
                Plugin.Log("Environment: client.");
                harmony.PatchAll();
                LogAllPatchedMethods();

                // Initialize recording components
                CaptureConfig.Load();
                CaptureKeybinds.Initialize();
                RecordingController.Create();

                Plugin.Log("Recording components initialized.");
                Plugin.Log("  F7 / Numpad0 / Insert = Toggle recording");
                Plugin.Log("  F8 = Reset episode (discard current)");
                Plugin.Log($"  Recordings folder: {CaptureConfig.RecordingsFolder}");
            }

            Plugin.Log($"Enabled!");
            return true;
        }
        catch (Exception e)
        {
            Plugin.LogError($"Failed to Enable: {e.Message}!");
            return false;
        }
    }

    public bool OnDisable()
    {
        try
        {
            Plugin.Log($"Disabling...");
            harmony.UnpatchSelf();

            // Cleanup recording components
            CaptureKeybinds.Cleanup();
            if (RecordingController.Instance != null)
                UnityEngine.Object.Destroy(RecordingController.Instance.gameObject);

            Plugin.Log($"Disabled! Goodbye!");
            return true;
        }
        catch (Exception e)
        {
            Plugin.LogError($"Failed to disable: {e.Message}!");
            return false;
        }
    }

    public static bool IsDedicatedServer()
    {
        return SystemInfo.graphicsDeviceType == GraphicsDeviceType.Null;
    }

    public static void LogAllPatchedMethods()
    {
        var allPatchedMethods = harmony.GetPatchedMethods();
        var pluginId = harmony.Id;

        var mine = allPatchedMethods
            .Select(m => new { method = m, info = Harmony.GetPatchInfo(m) })
            .Where(x =>
                x.info.Prefixes.Any(p => p.owner == pluginId) ||
                x.info.Postfixes.Any(p => p.owner == pluginId) ||
                x.info.Transpilers.Any(p => p.owner == pluginId) ||
                x.info.Finalizers.Any(p => p.owner == pluginId)
            )
            .Select(x => x.method);

        foreach (var m in mine)
            Plugin.Log($" - {m.DeclaringType.FullName}.{m.Name}");
    }

    public static void Log(string message)
    {
        Debug.Log($"[{MOD_NAME}] {message}");
    }

    public static void LogError(string message)
    {
        Debug.LogError($"[{MOD_NAME}] {message}");
    }

    /// <summary>
    /// Display a message in the game chat for the user to see.
    /// </summary>
    public static void ChatMessage(string message)
    {
        try
        {
            if (UIChat.Instance != null)
            {
                UIChat.Instance.AddChatMessage($"[PuckCapture] {message}");
            }
        }
        catch
        {
            // Silently fail if chat not available
        }
        // Also log to console
        Log(message);
    }
}
