using System;
using HarmonyLib;

namespace PuckCapture;

/// <summary>
/// Handles chat commands for playback control.
/// Commands:
///   /playback [filename] - Start playback, optionally from a specific recording
///   /stopplayback - Stop playback
///   /nextplayback - Skip to next recording
/// </summary>
[HarmonyPatch(typeof(UIChat), nameof(UIChat.Client_SendClientChatMessage))]
public class ChatCommandsPatch
{
    [HarmonyPrefix]
    public static bool Prefix(UIChat __instance, string message, bool useTeamChat)
    {
        if (string.IsNullOrEmpty(message)) return true;

        string trimmed = message.Trim();

        // /playback [optional filename]
        if (trimmed.StartsWith("/playback", StringComparison.OrdinalIgnoreCase))
        {
            string filename = null;

            // Extract filename if provided
            if (trimmed.Length > "/playback".Length)
            {
                filename = trimmed.Substring("/playback".Length).Trim();
            }

            // Ensure PlaybackController exists
            if (PlaybackController.Instance == null)
            {
                PlaybackController.Create();
            }

            PlaybackController.Instance.StartPlayback(filename);

            return false; // Don't send as chat message
        }

        // /stopplayback
        if (trimmed.Equals("/stopplayback", StringComparison.OrdinalIgnoreCase))
        {
            if (PlaybackController.Instance != null)
            {
                PlaybackController.Instance.StopPlayback();
            }
            else
            {
                Plugin.ChatMessage("<color=#FF6666>Playback not running</color>");
            }

            return false;
        }

        // /nextplayback - skip to next recording
        if (trimmed.Equals("/nextplayback", StringComparison.OrdinalIgnoreCase) ||
            trimmed.Equals("/skipplayback", StringComparison.OrdinalIgnoreCase))
        {
            if (PlaybackController.Instance != null && PlaybackController.Instance.IsPlaying)
            {
                PlaybackController.Instance.NextRecording();
            }
            else
            {
                Plugin.ChatMessage("<color=#FF6666>Playback not running</color>");
            }

            return false;
        }

        // /listrecordings - list available recordings
        if (trimmed.Equals("/listrecordings", StringComparison.OrdinalIgnoreCase))
        {
            ListRecordings(__instance);
            return false;
        }

        // Not a playback command, let it through
        return true;
    }

    private static void ListRecordings(UIChat chat)
    {
        try
        {
            string folder = @"C:\Projects\puck-ai-training\recordings";
            if (!System.IO.Directory.Exists(folder))
            {
                chat.AddChatMessage($"[PuckCapture] <color=#FF0000>Folder not found: {folder}</color>");
                return;
            }

            var files = System.IO.Directory.GetFiles(folder, "*_goal.bin");
            chat.AddChatMessage($"[PuckCapture] <color=#00FFFF>Found {files.Length} goal recordings:</color>");

            // Show last 10
            int start = Math.Max(0, files.Length - 10);
            for (int i = start; i < files.Length; i++)
            {
                string name = System.IO.Path.GetFileName(files[i]);
                chat.AddChatMessage($"[PuckCapture] <color=#AAAAAA>  {name}</color>");
            }

            if (files.Length > 10)
            {
                chat.AddChatMessage($"[PuckCapture] <color=#AAAAAA>  ... and {files.Length - 10} more</color>");
            }
        }
        catch (Exception e)
        {
            chat.AddChatMessage($"[PuckCapture] <color=#FF0000>Error: {e.Message}</color>");
        }
    }
}
