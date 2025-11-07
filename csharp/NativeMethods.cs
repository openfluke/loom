using System;
using System.Runtime.InteropServices;
using System.Reflection;
using System.IO;

namespace Welvet;

/// <summary>
/// P/Invoke declarations for LOOM C-ABI functions.
/// This class wraps the native libloom library (libloom.so/dylib/dll).
/// </summary>
internal static class NativeMethods
{
    // On Windows, .NET requires the .dll extension in the library name
    // On Linux/macOS, it automatically adds .so/.dylib
#if WINDOWS
    private const string LibName = "libloom.dll";
#else
    private const string LibName = "libloom";
#endif

    static NativeMethods()
    {
        // Register a custom DLL import resolver to help locate the native library
        NativeLibrary.SetDllImportResolver(typeof(NativeMethods).Assembly, DllImportResolver);
    }

    private static IntPtr DllImportResolver(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        // Only handle our library
        if (libraryName != LibName && libraryName != "libloom")
            return IntPtr.Zero;

        // Get the runtime identifier
        string rid = GetRuntimeIdentifier();
        string libFileName = GetLibraryFileName();
        
        // Try to load from runtimes/<rid>/native/
        string assemblyLocation = assembly.Location;
        if (!string.IsNullOrEmpty(assemblyLocation))
        {
            string assemblyDir = Path.GetDirectoryName(assemblyLocation)!;
            string runtimePath = Path.Combine(assemblyDir, "runtimes", rid, "native", libFileName);
            
            if (File.Exists(runtimePath))
            {
                if (NativeLibrary.TryLoad(runtimePath, out IntPtr handle))
                    return handle;
            }
        }

        // Fallback: try default search
        if (NativeLibrary.TryLoad(libraryName, assembly, searchPath, out IntPtr defaultHandle))
            return defaultHandle;

        return IntPtr.Zero;
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
            Loom_FreeCString(ptr);
        }
    }
}
