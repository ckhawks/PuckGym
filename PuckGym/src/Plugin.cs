using System;
using System.Linq;
using HarmonyLib;
using UnityEngine;
using UnityEngine.Rendering;

namespace PuckGym;

public class Plugin : IPuckMod
{
    public static string MOD_NAME = "PuckGym";
    public static string MOD_VERSION = "1.0.0";
    public static string MOD_GUID = "pw.stellaric.toaster.puckgym";

    static readonly Harmony harmony = new Harmony(MOD_GUID);
    
    public static ModSettings modSettings;
    
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
                modSettings = ModSettings.Load();
                modSettings.Save();

                // Initialize RL training components
                RLKeybinds.Initialize();
                RLBridge.Instance.Initialize();
                RLController.Create();
                Plugin.Log("RL components initialized.");
                Plugin.Log("  F9  = Toggle training");
                Plugin.Log("  F10 = Increase time scale");
                Plugin.Log("  F11 = Decrease time scale");
                Plugin.Log("  F12 = Reset time scale");
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

            // Cleanup RL components
            RLKeybinds.Cleanup();
            RLBridge.Instance?.Dispose();
            if (RLController.Instance != null)
                UnityEngine.Object.Destroy(RLController.Instance.gameObject);

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
        var pluginId  = harmony.Id;

        var mine = allPatchedMethods
            .Select(m => new { method = m, info = Harmony.GetPatchInfo(m) })
            .Where(x =>
                // could be prefix, postfix, transpiler or finalizer
                x.info.Prefixes.  Any(p => p.owner == pluginId) ||
                x.info.Postfixes. Any(p => p.owner == pluginId) ||
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
}