using System.Reflection;
using HarmonyLib;
using Unity.Netcode;
using UnityEngine;
using UnityEngine.InputSystem;

namespace PuckGym;

/// <summary>
/// Keybind handling for RL training, using the same pattern as other Puck mods.
/// </summary>
public static class RLKeybinds
{
    // Reflection to check if chat is focused
    private static readonly FieldInfo _isFocusedField = typeof(UIComponent<UIChat>)
        .GetField("isFocused", BindingFlags.Instance | BindingFlags.NonPublic);

    // Input actions
    public static InputAction toggleTrainingAction;
    public static InputAction increaseTimeScaleAction;
    public static InputAction decreaseTimeScaleAction;
    public static InputAction resetTimeScaleAction;
    public static InputAction debugScoreGoalAction;

    // Default keybinds (can be made configurable via ModSettings later)
    private const string TOGGLE_TRAINING_KEY = "<Keyboard>/f9";
    private const string INCREASE_TIMESCALE_KEY = "<Keyboard>/f10";
    private const string DECREASE_TIMESCALE_KEY = "<Keyboard>/f11";
    private const string RESET_TIMESCALE_KEY = "<Keyboard>/f12";
    private const string DEBUG_SCORE_GOAL_KEY = "<Keyboard>/f8";

    public static void Initialize()
    {
        // Toggle RL training
        toggleTrainingAction = new InputAction(name: "rl_toggle_training", binding: TOGGLE_TRAINING_KEY);
        toggleTrainingAction.Enable();
        Plugin.Log($"Registered keybind '{TOGGLE_TRAINING_KEY}' for Toggle RL Training");

        // Time scale controls
        increaseTimeScaleAction = new InputAction(name: "rl_increase_timescale", binding: INCREASE_TIMESCALE_KEY);
        increaseTimeScaleAction.Enable();
        Plugin.Log($"Registered keybind '{INCREASE_TIMESCALE_KEY}' for Increase Time Scale");

        decreaseTimeScaleAction = new InputAction(name: "rl_decrease_timescale", binding: DECREASE_TIMESCALE_KEY);
        decreaseTimeScaleAction.Enable();
        Plugin.Log($"Registered keybind '{DECREASE_TIMESCALE_KEY}' for Decrease Time Scale");

        resetTimeScaleAction = new InputAction(name: "rl_reset_timescale", binding: RESET_TIMESCALE_KEY);
        resetTimeScaleAction.Enable();
        Plugin.Log($"Registered keybind '{RESET_TIMESCALE_KEY}' for Reset Time Scale");

        // Debug: score goal
        debugScoreGoalAction = new InputAction(name: "rl_debug_score", binding: DEBUG_SCORE_GOAL_KEY);
        debugScoreGoalAction.Enable();
        Plugin.Log($"Registered keybind '{DEBUG_SCORE_GOAL_KEY}' for Debug Score Goal");
    }

    public static void Cleanup()
    {
        toggleTrainingAction?.Disable();
        increaseTimeScaleAction?.Disable();
        decreaseTimeScaleAction?.Disable();
        resetTimeScaleAction?.Disable();
        debugScoreGoalAction?.Disable();
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
            // Don't process keybinds if chat is focused
            if (IsChatFocused()) return;

            // Toggle training
            if (toggleTrainingAction != null && toggleTrainingAction.WasPressedThisFrame())
            {
                RLController.Instance?.ToggleTraining();
            }

            // Time scale controls
            if (increaseTimeScaleAction != null && increaseTimeScaleAction.WasPressedThisFrame())
            {
                RLController.Instance?.IncreaseTimeScale();
            }

            if (decreaseTimeScaleAction != null && decreaseTimeScaleAction.WasPressedThisFrame())
            {
                RLController.Instance?.DecreaseTimeScale();
            }

            if (resetTimeScaleAction != null && resetTimeScaleAction.WasPressedThisFrame())
            {
                RLController.Instance?.ResetTimeScale();
            }

            // Debug: teleport puck into goal to test goal detection
            if (debugScoreGoalAction != null && debugScoreGoalAction.WasPressedThisFrame())
            {
                RLController.Instance?.DebugScoreGoal();
            }
        }
    }

    /// <summary>
    /// Harmony patch to prevent normal input from overwriting RL actions.
    /// When training is enabled, we skip the movement input reading from keyboard.
    /// </summary>
    [HarmonyPatch(typeof(PlayerInput), "UpdateInputs")]
    public static class PlayerInputUpdateInputsPatch
    {
        [HarmonyPrefix]
        public static bool Prefix(PlayerInput __instance)
        {
            // If training is enabled and this is the local player, skip normal input processing
            if (RLController.Instance != null &&
                RLController.Instance.IsTrainingEnabled &&
                __instance.IsOwner)
            {
                // Return false to skip the original method entirely
                // The RL controller will set inputs directly
                return false;
            }

            // Return true to run the original method normally
            return true;
        }
    }

    /// <summary>
    /// Harmony patch to prevent game from spawning pucks during training.
    /// We control puck spawning ourselves for curriculum learning.
    /// </summary>
    [HarmonyPatch(typeof(PuckManager), "Server_SpawnPucksForPhase")]
    public static class PuckManagerSpawnPatch
    {
        [HarmonyPrefix]
        public static bool Prefix(GamePhase phase)
        {
            // Skip game's puck spawning when training is enabled
            if (RLController.Instance != null &&
                RLController.Instance.IsTrainingEnabled)
            {
                Plugin.Log($"Blocked game puck spawn for phase: {phase}");
                return false;
            }

            return true;
        }
    }

    /// <summary>
    /// Harmony patch to intercept goal triggers during training.
    /// When training is enabled, we handle goals ourselves without changing game phase.
    /// </summary>
    [HarmonyPatch(typeof(GoalTrigger), "OnTriggerEnter")]
    public static class GoalTriggerPatch
    {
        // Reflection to access the private 'goal' field on GoalTrigger
        private static readonly FieldInfo _goalField = typeof(GoalTrigger)
            .GetField("goal", BindingFlags.Instance | BindingFlags.NonPublic);

        // Reflection to access the private 'Team' field on Goal
        private static readonly FieldInfo _teamField = typeof(Goal)
            .GetField("Team", BindingFlags.Instance | BindingFlags.NonPublic);

        [HarmonyPrefix]
        public static bool Prefix(GoalTrigger __instance, Collider collider)
        {
            // If training is not enabled, run the original method
            if (RLController.Instance == null || !RLController.Instance.IsTrainingEnabled)
            {
                return true;
            }

            // Only process on server
            if (!NetworkManager.Singleton.IsServer)
            {
                return false; // Skip original, but don't process
            }

            // Check if it's a puck
            Puck puck = collider.GetComponentInParent<Puck>();
            if (puck == null)
            {
                return false; // Not a puck, skip
            }

            // Get the Goal from the GoalTrigger
            Goal goal = _goalField?.GetValue(__instance) as Goal;
            if (goal == null)
            {
                Plugin.LogError("GoalTriggerPatch: Could not get Goal from GoalTrigger");
                return false;
            }

            // Get the team whose goal this is (the team that defends this goal)
            PlayerTeam goalTeam = PlayerTeam.Blue;
            if (_teamField != null)
            {
                goalTeam = (PlayerTeam)_teamField.GetValue(goal);
            }

            // Notify the RLController
            RLController.Instance.OnGoalScored(goalTeam, puck);

            // Skip the original method to prevent phase change
            return false;
        }
    }
}
