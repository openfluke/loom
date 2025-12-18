using System;
using System.Runtime.InteropServices;
using System.Reflection;
using System.IO;
using System.Diagnostics;

namespace Welvet;

/// <summary>
/// P/Invoke declarations for LOOM C-ABI functions.
/// This class wraps the native libloom library (libloom.so/dylib/dll).
/// </summary>
internal static class NativeMethods
{
    // Use simple name - let the resolver handle platform specifics
    private const string LibName = "libloom";

    static NativeMethods()
    {
        // Register a custom DLL import resolver to help locate the native library
        NativeLibrary.SetDllImportResolver(typeof(NativeMethods).Assembly, DllImportResolver);

        // Print debug information
        Debug.WriteLine("=== Welvet Native Library Loader ===");
        Debug.WriteLine($"OS: {RuntimeInformation.OSDescription}");
        Debug.WriteLine($"OS Platform: Windows={RuntimeInformation.IsOSPlatform(OSPlatform.Windows)}, Linux={RuntimeInformation.IsOSPlatform(OSPlatform.Linux)}, macOS={RuntimeInformation.IsOSPlatform(OSPlatform.OSX)}");
        Debug.WriteLine($"Architecture: {RuntimeInformation.ProcessArchitecture}");
        Debug.WriteLine($"Runtime ID: {RuntimeInformation.RuntimeIdentifier}");
        Debug.WriteLine($"Framework: {RuntimeInformation.FrameworkDescription}");
        Debug.WriteLine($"Expected library: {GetLibraryFileName()}");
        Debug.WriteLine($"Custom RID: {GetCustomRuntimeIdentifier()}");
        Debug.WriteLine($"Standard RID: {GetRuntimeIdentifier()}");
        Debug.WriteLine($"AppContext.BaseDirectory: {AppContext.BaseDirectory}");

        var assemblyLocation = typeof(NativeMethods).Assembly.Location;
        if (!string.IsNullOrEmpty(assemblyLocation))
        {
            var assemblyDir = Path.GetDirectoryName(assemblyLocation);
            Debug.WriteLine($"Assembly directory: {assemblyDir}");
        }

        // On Windows, add the base directory to the DLL search path
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            try
            {
                var baseDir = AppContext.BaseDirectory;
                SetDllDirectory(baseDir);
                Debug.WriteLine($"✅ Set DLL search directory to: {baseDir}");

                // Also try AddDllDirectory (Windows 8+)
                var handle = AddDllDirectory(baseDir);
                if (handle != IntPtr.Zero)
                {
                    Debug.WriteLine($"✅ Added DLL directory: {baseDir}");
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"⚠️ Failed to set DLL directory: {ex.Message}");
            }
        }

        Debug.WriteLine("====================================");
    }

    // Windows API to add directory to DLL search path
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SetDllDirectory(string lpPathName);

    // Windows API to add directory to DLL search path (Windows 8+)
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr AddDllDirectory(string lpPathName);

    private static IntPtr DllImportResolver(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        Debug.WriteLine($"\n[DllImportResolver] Resolving: {libraryName}");

        // Only handle our library (accept both "libloom" and platform-specific names)
        if (libraryName != "libloom" &&
            libraryName != "libloom.dll" &&
            libraryName != "libloom.so" &&
            libraryName != "libloom.dylib")
        {
            Debug.WriteLine($"[DllImportResolver] Not our library, skipping");
            return IntPtr.Zero;
        }

        string libFileName = GetLibraryFileName();
        string assemblyLocation = assembly.Location;

        Debug.WriteLine($"[DllImportResolver] Looking for: {libFileName}");
        Debug.WriteLine($"[DllImportResolver] Assembly location: {assemblyLocation}");

        // Prefer AppContext.BaseDirectory when available (more reliable in single-file / MAUI)
        string baseDir = AppContext.BaseDirectory;
        if (!string.IsNullOrEmpty(baseDir))
        {
            // Try BaseDirectory direct
            string baseDirect = Path.Combine(baseDir, libFileName);
            Debug.WriteLine($"[DllImportResolver] Try 0 - AppContext.BaseDirectory: {baseDirect}");
            Debug.WriteLine($"[DllImportResolver] File exists: {File.Exists(baseDirect)}");
            if (File.Exists(baseDirect))
            {
                try
                {
                    if (NativeLibrary.TryLoad(baseDirect, out IntPtr baseHandle))
                    {
                        Debug.WriteLine($"[DllImportResolver] ✅ SUCCESS! Loaded from BaseDirectory");
                        return baseHandle;
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[DllImportResolver] ⚠️ Exception loading from BaseDirectory: {ex.Message}");
                }
            }
        }

        if (!string.IsNullOrEmpty(assemblyLocation))
        {
            string assemblyDir = Path.GetDirectoryName(assemblyLocation)!;

            // Try 1: Check if it's directly in the output directory (from .targets file)
            string directPath = Path.Combine(assemblyDir, libFileName);
            Debug.WriteLine($"[DllImportResolver] Try 1 - Direct: {directPath}");
            Debug.WriteLine($"[DllImportResolver] File exists: {File.Exists(directPath)}");
            if (File.Exists(directPath))
            {
                Debug.WriteLine($"[DllImportResolver] Attempting to load from direct path...");
                try
                {
                    if (NativeLibrary.TryLoad(directPath, out IntPtr handle))
                    {
                        Debug.WriteLine($"[DllImportResolver] ✅ SUCCESS! Loaded from direct path");
                        return handle;
                    }
                    Debug.WriteLine($"[DllImportResolver] ❌ Failed to load from direct path");
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[DllImportResolver] ⚠️ Exception loading from direct path: {ex.Message}");
                }
            }

            // Try 2: Check our custom runtime paths (windows_x86_64, linux_x86_64, etc.)
            string customRid = GetCustomRuntimeIdentifier();
            string customPath = Path.Combine(assemblyDir, "runtimes", customRid, libFileName);
            Debug.WriteLine($"[DllImportResolver] Try 2 - Custom RID: {customPath}");
            Debug.WriteLine($"[DllImportResolver] File exists: {File.Exists(customPath)}");
            if (File.Exists(customPath))
            {
                Debug.WriteLine($"[DllImportResolver] Attempting to load from custom RID path...");
                try
                {
                    if (NativeLibrary.TryLoad(customPath, out IntPtr handle))
                    {
                        Debug.WriteLine($"[DllImportResolver] ✅ SUCCESS! Loaded from custom RID path");
                        return handle;
                    }
                    Debug.WriteLine($"[DllImportResolver] ❌ Failed to load from custom RID path");
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[DllImportResolver] ⚠️ Exception loading from custom RID path: {ex.Message}");
                }
            }

            // Try 3: Check standard .NET runtime paths (win-x64, linux-x64, etc.)
            string standardRid = GetRuntimeIdentifier();
            string standardPath = Path.Combine(assemblyDir, "runtimes", standardRid, "native", libFileName);
            Debug.WriteLine($"[DllImportResolver] Try 3 - Standard RID: {standardPath}");
            Debug.WriteLine($"[DllImportResolver] File exists: {File.Exists(standardPath)}");
            if (File.Exists(standardPath))
            {
                Debug.WriteLine($"[DllImportResolver] Attempting to load from standard RID path...");
                try
                {
                    if (NativeLibrary.TryLoad(standardPath, out IntPtr handle))
                    {
                        Debug.WriteLine($"[DllImportResolver] ✅ SUCCESS! Loaded from standard RID path");
                        return handle;
                    }
                    Debug.WriteLine($"[DllImportResolver] ❌ Failed to load from standard RID path");
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[DllImportResolver] ⚠️ Exception loading from standard RID path: {ex.Message}");
                }
            }

            // Try 4: List what's actually in the directory
            Debug.WriteLine($"[DllImportResolver] Listing files in assembly directory:");
            try
            {
                var files = Directory.GetFiles(assemblyDir, "*loom*", SearchOption.TopDirectoryOnly);
                foreach (var file in files)
                {
                    Debug.WriteLine($"  - {Path.GetFileName(file)}");
                }

                var runtimesDir = Path.Combine(assemblyDir, "runtimes");
                if (Directory.Exists(runtimesDir))
                {
                    Debug.WriteLine($"[DllImportResolver] Runtimes directory exists: {runtimesDir}");
                    var subdirs = Directory.GetDirectories(runtimesDir);
                    foreach (var subdir in subdirs)
                    {
                        Debug.WriteLine($"  - {Path.GetFileName(subdir)}/");
                        var libFiles = Directory.GetFiles(subdir, "*loom*", SearchOption.AllDirectories);
                        foreach (var libFile in libFiles)
                        {
                            Debug.WriteLine($"    - {libFile.Replace(runtimesDir, "runtimes")}");
                        }
                    }
                }
                else
                {
                    Debug.WriteLine($"[DllImportResolver] Runtimes directory does not exist");
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[DllImportResolver] Error listing files: {ex.Message}");
            }
        }

        // Fallback: try default search
        Debug.WriteLine($"[DllImportResolver] Try 4 - Default search...");
        if (NativeLibrary.TryLoad(libraryName, assembly, searchPath, out IntPtr defaultHandle))
        {
            Debug.WriteLine($"[DllImportResolver] ✅ SUCCESS! Loaded via default search");
            return defaultHandle;
        }

        Debug.WriteLine($"[DllImportResolver] ❌ ALL ATTEMPTS FAILED");
        return IntPtr.Zero;
    }

    private static string GetCustomRuntimeIdentifier()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return "windows_x86_64";
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            if (RuntimeInformation.RuntimeIdentifier.Contains("android"))
                return "android_arm64";
            return "linux_x86_64";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            return "macos_arm64";

        return "unknown";
    }

    private static string GetRuntimeIdentifier()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.X64 => "win-x64",
                Architecture.X86 => "win-x86",
                Architecture.Arm64 => "win-arm64",
                _ => "win-x64"
            };
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            // Check for Android
            if (RuntimeInformation.RuntimeIdentifier.Contains("android"))
            {
                return RuntimeInformation.ProcessArchitecture switch
                {
                    Architecture.Arm64 => "android-arm64",
                    Architecture.Arm => "android-arm",
                    Architecture.X64 => "android-x64",
                    _ => "android-arm64"
                };
            }

            return RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.X64 => "linux-x64",
                Architecture.Arm64 => "linux-arm64",
                Architecture.Arm => "linux-arm",
                _ => "linux-x64"
            };
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            return RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.Arm64 => "osx-arm64",
                Architecture.X64 => "osx-x64",
                _ => "osx-arm64"
            };
        }

        return "unknown";
    }

    private static string GetLibraryFileName()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return "libloom.dll";
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            return "libloom.dylib";
        else
            return "libloom.so";
    }

    /// <summary>
    /// Creates a new neural network and returns JSON with handle.
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr Loom_NewNetwork(
        int inputSize,
        int gridRows,
        int gridCols,
        int layersPerCell,
        [MarshalAs(UnmanagedType.I1)] bool useGpu);

    /// <summary>
    /// Calls a method on a network handle using the reflection API.
    /// Returns JSON array of return values.
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr Loom_Call(
        long handle,
        [MarshalAs(UnmanagedType.LPStr)] string methodName,
        [MarshalAs(UnmanagedType.LPStr)] string argsJson);

    /// <summary>
    /// Frees a network handle.
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void Loom_Free(long handle);

    /// <summary>
    /// Frees a C string returned by the library.
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void Loom_FreeCString(IntPtr cstr);

    /// <summary>
    /// Gets the LOOM library version string.
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr Loom_GetVersion();

    /// <summary>
    /// Creates a dense layer configuration (legacy API).
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr Loom_InitDenseLayer(
        int inputSize,
        int outputSize,
        int activation);

    /// <summary>
    /// Generic layer initialization via registry system.
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr Loom_CallLayerInit(
        [MarshalAs(UnmanagedType.LPStr)] string functionName,
        [MarshalAs(UnmanagedType.LPStr)] string paramsJson);

    /// <summary>
    /// Lists all available layer initialization functions.
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr Loom_ListLayerInitFunctions();

    /// <summary>
    /// Sets a layer in the network at specific grid position.
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr Loom_SetLayer(
        long handle,
        int gridRow,
        int gridCol,
        int layerIndex,
        [MarshalAs(UnmanagedType.LPStr)] string layerConfigJson);

    /// <summary>
    /// Gets network information as JSON.
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr Loom_GetInfo(long handle);

    /// <summary>
    /// Loads a complete model from JSON string (structure + weights).
    /// Returns JSON with handle.
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr Loom_LoadModel(
        [MarshalAs(UnmanagedType.LPStr)] string modelJson,
        [MarshalAs(UnmanagedType.LPStr)] string modelId);

    /// <summary>
    /// Saves a model to JSON string (structure + weights).
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr Loom_SaveModel(
        long handle,
        [MarshalAs(UnmanagedType.LPStr)] string modelId);

    // ====================================================================
    // New Simple API (global network instance)
    // ====================================================================

    /// <summary>
    /// Creates a network from JSON configuration (new simple API).
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr CreateLoomNetwork(
        [MarshalAs(UnmanagedType.LPStr)] string jsonConfig);

    /// <summary>
    /// Forward pass with float array input (new simple API).
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr LoomForward(
        [MarshalAs(UnmanagedType.LPArray)] float[] inputs,
        int length);

    /// <summary>
    /// Backward pass with gradients (new simple API).
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr LoomBackward(
        [MarshalAs(UnmanagedType.LPArray)] float[] gradients,
        int length);

    /// <summary>
    /// Update weights with learning rate (new simple API).
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void LoomUpdateWeights(float learningRate);

    /// <summary>
    /// Train network with batches and config JSON (new simple API).
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr LoomTrain(
        [MarshalAs(UnmanagedType.LPStr)] string batchesJSON,
        [MarshalAs(UnmanagedType.LPStr)] string configJSON);

    /// <summary>
    /// Save model to JSON string (new simple API).
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr LoomSaveModel(
        [MarshalAs(UnmanagedType.LPStr)] string modelID);

    /// <summary>
    /// Load model from JSON string (new simple API).
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr LoomLoadModel(
        [MarshalAs(UnmanagedType.LPStr)] string jsonString,
        [MarshalAs(UnmanagedType.LPStr)] string modelID);

    /// <summary>
    /// Get network information JSON (new simple API).
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr LoomGetNetworkInfo();

    /// <summary>
    /// Evaluate network with inputs and expected outputs (new simple API).
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr LoomEvaluateNetwork(
        [MarshalAs(UnmanagedType.LPStr)] string inputsJSON,
        [MarshalAs(UnmanagedType.LPStr)] string expectedOutputsJSON);

    /// <summary>
    /// Free a C string returned by LOOM (new simple API).
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void FreeLoomString(IntPtr str);

    // ====================================================================
    // Stepping API (Fine-grained control)
    // ====================================================================

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern long LoomInitStepState(int inputSize);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void LoomSetInput(long handle, [MarshalAs(UnmanagedType.LPArray)] float[] input, int length);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern long LoomStepForward(long handle);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr LoomGetOutput(long handle);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr LoomStepBackward(long handle, [MarshalAs(UnmanagedType.LPArray)] float[] gradients, int length);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void LoomApplyGradients(float learningRate);

    [DllImport(LibName, EntryPoint = "LoomApplyGradientsAdamW", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void LoomApplyGradientsAdamW(float learningRate, float beta1, float beta2, float weightDecay);

    [DllImport(LibName, EntryPoint = "LoomApplyGradientsRMSprop", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void LoomApplyGradientsRMSprop(float learningRate, float alpha, float epsilon, float momentum);

    [DllImport(LibName, EntryPoint = "LoomApplyGradientsSGDMomentum", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void LoomApplyGradientsSGDMomentum(float learningRate, float momentum, float dampening, int nesterov);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void LoomFreeStepState(long handle);

    // ====================================================================
    // Transformer Inference API
    // ====================================================================

    /// <summary>
    /// Loads a tokenizer from JSON bytes.
    /// Returns JSON with {"success": true, "vocab_size": ..., ...}
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr LoadTokenizerFromBytes(
        IntPtr data,
        int dataLen);

    /// <summary>
    /// Loads a transformer model from config and weights bytes.
    /// Returns JSON with {"success": true, "num_layers": ..., ...}
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr LoadTransformerFromBytes(
        IntPtr configData,
        int configLen,
        IntPtr weightsData,
        int weightsLen);

    /// <summary>
    /// Encodes text to token IDs.
    /// Returns JSON with {"success": true, "ids": [...]}
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr EncodeText(
        [MarshalAs(UnmanagedType.LPStr)] string text,
        [MarshalAs(UnmanagedType.I1)] bool addSpecialTokens);

    /// <summary>
    /// Decodes token IDs to text.
    /// Returns JSON with {"success": true, "text": "..."}
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr DecodeTokens(
        [MarshalAs(UnmanagedType.LPStr)] string idsJson,
        [MarshalAs(UnmanagedType.I1)] bool skipSpecialTokens);

    /// <summary>
    /// Generates text from prompt (blocking, all tokens at once).
    /// Returns JSON with {"success": true, "generated_text": "..."}
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr GenerateText(
        [MarshalAs(UnmanagedType.LPStr)] string prompt,
        int maxTokens,
        float temperature);

    /// <summary>
    /// Generates next token given current token sequence (for streaming).
    /// Returns JSON with {"success": true, "token": ..., "is_eos": ...}
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr GenerateNextToken(
        [MarshalAs(UnmanagedType.LPStr)] string idsJson,
        float temperature);

    /// <summary>
    /// Helper to marshal IntPtr to managed string and free native memory.
    /// </summary>
    internal static string PtrToStringAndFree(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero)
            return string.Empty;

        try
        {
            return Marshal.PtrToStringAnsi(ptr) ?? string.Empty;
        }
        finally
        {
            FreeLoomString(ptr);
        }
    }

    // ====================================================================
    // TweenState API (Neural Tweening)
    // ====================================================================

    /// <summary>
    /// Creates a new TweenState for neural tweening.
    /// </summary>
    /// <param name="useChainRule">If 1, use chain rule (TweenChain mode)</param>
    /// <returns>TweenState handle, or -1 on error</returns>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern long LoomCreateTweenState(int useChainRule);

    /// <summary>
    /// Applies one tween step.
    /// </summary>
    /// <returns>Gap value (distance to target)</returns>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern float LoomTweenStep(
        long handle,
        [MarshalAs(UnmanagedType.LPArray)] float[] input,
        int inputLen,
        int targetClass,
        int outputSize,
        float learningRate);

    /// <summary>
    /// Frees a TweenState.
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void LoomFreeTweenState(long handle);

    // ====================================================================
    // AdaptationTracker API (Benchmark Task Switching)
    // ====================================================================

    /// <summary>
    /// Creates a new AdaptationTracker.
    /// </summary>
    /// <param name="windowDurationMs">Duration of each accuracy window in milliseconds</param>
    /// <param name="totalDurationMs">Total test duration in milliseconds</param>
    /// <returns>Tracker handle, or -1 on error</returns>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern long LoomCreateAdaptationTracker(int windowDurationMs, int totalDurationMs);

    /// <summary>
    /// Sets model information for the tracker.
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern void LoomTrackerSetModelInfo(
        long handle,
        [MarshalAs(UnmanagedType.LPStr)] string modelName,
        [MarshalAs(UnmanagedType.LPStr)] string modeName);

    /// <summary>
    /// Schedules a task change at a specific offset from start.
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern void LoomTrackerScheduleTaskChange(
        long handle,
        int atOffsetMs,
        int taskId,
        [MarshalAs(UnmanagedType.LPStr)] string taskName);

    /// <summary>
    /// Starts the tracker with an initial task.
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern void LoomTrackerStart(
        long handle,
        [MarshalAs(UnmanagedType.LPStr)] string taskName,
        int taskId);

    /// <summary>
    /// Records an output and whether it was correct.
    /// </summary>
    /// <returns>Previous task ID (for detecting task changes)</returns>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int LoomTrackerRecordOutput(long handle, int isCorrect);

    /// <summary>
    /// Gets the current task ID.
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int LoomTrackerGetCurrentTask(long handle);

    /// <summary>
    /// Finalizes tracking and returns results as JSON.
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr LoomTrackerFinalize(long handle);

    /// <summary>
    /// Frees an AdaptationTracker.
    /// </summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void LoomFreeTracker(long handle);
}
