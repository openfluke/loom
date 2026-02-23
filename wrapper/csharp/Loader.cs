using System;
using System.Diagnostics;

namespace Welvet;

/// <summary>
/// Explicit loader class to ensure initialization happens.
/// Call Loader.Initialize() at app startup before using any Welvet APIs.
/// </summary>
public static class Loader
{
    private static bool _initialized = false;
    private static readonly object _lock = new object();

    /// <summary>
    /// Explicitly initialize the Welvet library.
    /// Safe to call multiple times - will only initialize once.
    /// </summary>
    public static void Initialize()
    {
        lock (_lock)
        {
            if (_initialized)
            {
                Debug.WriteLine("[Welvet.Loader] Already initialized");
                return;
            }

            Debug.WriteLine("[Welvet.Loader] ========================================");
            Debug.WriteLine("[Welvet.Loader] Welvet Explicit Initialization Starting");
            Debug.WriteLine("[Welvet.Loader] ========================================");

            try
            {
                // Trigger the NativeMethods static constructor
                var _ = typeof(NativeMethods);
                Debug.WriteLine("[Welvet.Loader] ✅ NativeMethods type loaded");

                // Try to actually call GetVersion to verify library loads
                try
                {
                    string version = Network.GetVersion();
                    Debug.WriteLine($"[Welvet.Loader] ✅ LOOM Version: {version}");
                    Debug.WriteLine($"[Welvet.Loader] ✅ Initialization SUCCESSFUL");
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[Welvet.Loader] ⚠️ GetVersion failed: {ex.GetType().Name}: {ex.Message}");
                    throw;
                }

                _initialized = true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[Welvet.Loader] ❌ Initialization FAILED");
                Debug.WriteLine($"[Welvet.Loader] Exception: {ex.GetType().Name}");
                Debug.WriteLine($"[Welvet.Loader] Message: {ex.Message}");
                Debug.WriteLine($"[Welvet.Loader] StackTrace: {ex.StackTrace}");
                throw;
            }
            finally
            {
                Debug.WriteLine("[Welvet.Loader] ========================================");
            }
        }
    }

    /// <summary>
    /// Check if Welvet has been initialized.
    /// </summary>
    public static bool IsInitialized => _initialized;
}
