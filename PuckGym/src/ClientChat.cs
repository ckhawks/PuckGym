// ClientChat.cs

using System;
using System.Linq;
using HarmonyLib;
using UnityEngine;

namespace PuckGym;

public static class ClientChat
{
    
    [HarmonyPatch(typeof(UIChat), nameof(UIChat.Client_SendClientChatMessage))]
    private class PatchUIChatClientSendClientChatMessage
    {
        [HarmonyPrefix]
        private static bool Prefix(UIChat __instance, string message)
        {
            // Plugin.Log($"Patch: UIChat.Client_SendClientChatMessage (Prefix) was called.");
            string[] messageParts = message.Split(' ');

            if (messageParts[0].Equals("/test", StringComparison.OrdinalIgnoreCase))
            {
                __instance.AddChatMessage($"testo!");

                return false;
            }

            // /rl <id> - Set RL instance ID for multi-instance training
            if (messageParts[0].Equals("/rl", StringComparison.OrdinalIgnoreCase))
            {
                if (messageParts.Length < 2)
                {
                    __instance.AddChatMessage("Usage: /rl <id>");
                    __instance.AddChatMessage($"Current instance ID: {RLController.Instance?.InstanceId ?? 0}");
                    return false;
                }

                if (int.TryParse(messageParts[1], out int instanceId))
                {
                    if (instanceId < 0)
                    {
                        __instance.AddChatMessage("Instance ID must be >= 0");
                        return false;
                    }

                    if (RLController.Instance != null)
                    {
                        RLController.Instance.SetInstanceId(instanceId);
                        __instance.AddChatMessage($"Instance ID set to: {instanceId}");
                        __instance.AddChatMessage("Press F9 to start training on this instance");
                    }
                    else
                    {
                        __instance.AddChatMessage("RLController not initialized");
                    }
                }
                else
                {
                    __instance.AddChatMessage($"Invalid instance ID: {messageParts[1]}");
                }

                return false;
            }

            return true;
        }
    }
}