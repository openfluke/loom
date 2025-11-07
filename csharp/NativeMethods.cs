using System;
using System.Runtime.InteropServices;

namespace Welvet;

/// <summary>
/// P/Invoke declarations for LOOM C-ABI functions.
/// This class wraps the native libloom library (libloom.so/dylib/dll).
/// </summary>
internal static class NativeMethods
{
    private const string LibName = "libloom";

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
