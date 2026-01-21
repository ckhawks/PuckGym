using System.Collections.Generic;
using System.Reflection;
using HarmonyLib;
using Unity.Netcode;
using UnityEngine;
using UnityEngine.InputSystem;

namespace PuckCapture;

/// <summary>
/// Keybind handling for capture mod.
/// </summary>
public static class CaptureKeybinds
{
    // Reflection to check if chat is focused
    private static readonly FieldInfo _isFocusedField = typeof(UIComponent<UIChat>)
        .GetField("isFocused", BindingFlags.Instance | BindingFlags.NonPublic);

    // Input actions - multiple keys for toggle recording
    public static InputAction toggleRecordingAction1;  // F7
    public static InputAction toggleRecordingAction2;  // Numpad 0
    public static InputAction toggleRecordingAction3;  // Insert
    public static InputAction resetEpisodeAction;      // F8

    public static void Initialize()
    {
        // Toggle recording - three keybinds
        toggleRecordingAction1 = new InputAction(name: "capture_toggle_1", binding: "<Keyboard>/f7");
        toggleRecordingAction1.Enable();
        Plugin.Log("Registered keybind 'F7' for Toggle Recording");

        toggleRecordingAction2 = new InputAction(name: "capture_toggle_2", binding: "<Keyboard>/numpad0");
        toggleRecordingAction2.Enable();
        Plugin.Log("Registered keybind 'Numpad0' for Toggle Recording");

        toggleRecordingAction3 = new InputAction(name: "capture_toggle_3", binding: "<Keyboard>/insert");
        toggleRecordingAction3.Enable();
        Plugin.Log("Registered keybind 'Insert' for Toggle Recording");

        // Reset episode
        resetEpisodeAction = new InputAction(name: "capture_reset", binding: "<Keyboard>/f8");
        resetEpisodeAction.Enable();
        Plugin.Log("Registered keybind 'F8' for Reset Episode");
    }

    public static void Cleanup()
    {
        toggleRecordingAction1?.Disable();
        toggleRecordingAction2?.Disable();
        toggleRecordingAction3?.Disable();
        resetEpisodeAction?.Disable();
    }

    private static bool IsChatFocused()
    {
        if (_isFocusedField == null || UIChat.Instance == null)
            return false;

        return (bool)_isFocusedField.GetValue(UIChat.Instance);
    }

    /// <summary>
    /// Harmony patch to check keybinds during PlayerInput.Update
    /// </summary>
    [HarmonyPatch(typeof(PlayerInput), "Update")]
    public static class PlayerInputUpdatePatch
    {
        [HarmonyPostfix]
        public static void Postfix(PlayerInput __instance)
        {
            // Apply playback inputs AFTER game reads hardware input
            // This ensures our values override the hardware values
            if (PlaybackController.Instance != null && PlaybackController.Instance.IsPlaying)
            {
                PlaybackController.Instance.ApplyCurrentActions();
                return; // Don't process keybinds during playback
            }

            // Don't process keybinds if chat is focused
            if (IsChatFocused()) return;

            // Toggle recording (any of the three keys)
            if ((toggleRecordingAction1 != null && toggleRecordingAction1.WasPressedThisFrame()) ||
                (toggleRecordingAction2 != null && toggleRecordingAction2.WasPressedThisFrame()) ||
                (toggleRecordingAction3 != null && toggleRecordingAction3.WasPressedThisFrame()))
            {
                RecordingController.Instance?.ToggleRecording();
            }

            // Reset episode
            if (resetEpisodeAction != null && resetEpisodeAction.WasPressedThisFrame())
            {
                RecordingController.Instance?.ResetEpisode();
            }
        }
    }

    /// <summary>
    /// Harmony patch to intercept goal triggers during recording.
    /// Uses Prefix to detect goals and block original behavior.
    /// </summary>
    [HarmonyPatch(typeof(GoalTrigger), "OnTriggerEnter")]
    public static class GoalTriggerPatch
    {
        private static readonly FieldInfo _goalField = typeof(GoalTrigger)
            .GetField("goal", BindingFlags.Instance | BindingFlags.NonPublic);

        private static readonly FieldInfo _teamField = typeof(Goal)
            .GetField("Team", BindingFlags.Instance | BindingFlags.NonPublic);

        [HarmonyPrefix]
        public static bool Prefix(GoalTrigger __instance, Collider collider)
        {
            // Only intercept when recording - let normal behavior through otherwise
            if (RecordingController.Instance == null || !RecordingController.Instance.IsRecording)
                return true;

            // Check if it's a puck
            Puck puck = collider.GetComponentInParent<Puck>();
            if (puck == null)
                return true;

            // Get the Goal from the GoalTrigger
            Goal goal = _goalField?.GetValue(__instance) as Goal;
            if (goal == null)
                return true;

            // Get the team whose goal this is
            PlayerTeam goalTeam = PlayerTeam.Blue;
            if (_teamField != null)
            {
                goalTeam = (PlayerTeam)_teamField.GetValue(goal);
            }

            // Notify the RecordingController
            RecordingController.Instance.OnGoalScored(goalTeam, puck);

            // Block original method to prevent all goal effects
            return false;
        }
    }

    /// <summary>
    /// Block game state UI changes during recording so our "RECORDING" text stays.
    /// </summary>
    [HarmonyPatch(typeof(UIGameStateController), "Event_OnGameStateChanged")]
    public static class BlockGameStateUIPatch
    {
        [HarmonyPrefix]
        public static bool Prefix()
        {
            // Return false to skip the original method when recording
            if (RecordingController.Instance != null && RecordingController.Instance.IsRecording)
                return false;
            return true;
        }
    }
}
