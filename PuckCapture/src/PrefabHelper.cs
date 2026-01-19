using System;
using System.IO;
using System.Reflection;
using UnityEngine;

namespace PuckCapture;

public static class PrefabHelper
{
    public static AssetBundle LoadAssetBundle(string assetBundleName)
    {
        try
        {
            string fullPath = Path.Combine(
                Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location),
                assetBundleName
            );

            if (!File.Exists(fullPath))
            {
                Plugin.LogError($"AssetBundle not found at: {fullPath}");
                return null;
            }

            AssetBundle loadedAssetBundle = AssetBundle.LoadFromFile(fullPath);
            if (loadedAssetBundle == null)
            {
                Plugin.LogError("Failed to load AssetBundle.");
            }
            else
            {
                Plugin.Log("AssetBundle loaded successfully.");
            }

            return loadedAssetBundle;
        }
        catch (Exception ex)
        {
            Plugin.LogError($"Error loading AssetBundle: {ex.Message}");
            return null;
        }
    }

    public static GameObject LoadPrefab(AssetBundle assetBundle, string assetPath)
    {
        try
        {
            GameObject loadedObject = assetBundle.LoadAsset<GameObject>(assetPath);
            if (loadedObject == null)
            {
                Plugin.LogError($"Prefab '{assetPath}' not found in AssetBundle.");
                return null;
            }

            return loadedObject;
        }
        catch (Exception ex)
        {
            Plugin.LogError($"Error loading prefab: {ex.Message}");
            return null;
        }
    }
}
