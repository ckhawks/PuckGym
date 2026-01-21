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
            
            return true;
        }
    }
}