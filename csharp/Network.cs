using System;
using System.Text.Json;
using System.Collections.Generic;
using System.Linq;

namespace Welvet;

/// <summary>
/// LOOM Neural Network - High-level .NET API for neural network operations.
/// Wraps the native LOOM C-ABI library with managed, idiomatic C# interface.
/// </summary>
public class Network : IDisposable
{
    private long _handle;
    private bool _disposed;

    /// <summary>
    /// Gets the native handle for this network.
    /// </summary>
    public long Handle => _handle;

    /// <summary>
    /// Private constructor - use factory methods to create networks.
    /// </summary>
    private Network(long handle)
    {
        _handle = handle;
    }

    /// <summary>
    /// Creates a new neural network with grid-based architecture.
    /// </summary>
    /// <param name="inputSize">Size of input layer</param>
    /// <param name="gridRows">Number of rows in grid (default: 2)</param>
    /// <param name="gridCols">Number of columns in grid (default: 2)</param>
    /// <param name="layersPerCell">Number of layers per grid cell (default: 3)</param>
    /// <param name="useGpu">Enable GPU acceleration (default: false)</param>
    /// <returns>New Network instance</returns>
    public static Network Create(
        int inputSize,
        int gridRows = 2,
        int gridCols = 2,
        int layersPerCell = 3,
        bool useGpu = false)
    {
        var responsePtr = NativeMethods.Loom_NewNetwork(
            inputSize, gridRows, gridCols, layersPerCell, useGpu);

        var responseJson = NativeMethods.PtrToStringAndFree(responsePtr);
        if (string.IsNullOrEmpty(responseJson))
            throw new Exception("Failed to create network");

        using var doc = JsonDocument.Parse(responseJson);
        var root = doc.RootElement;

        if (root.TryGetProperty("error", out var errorProp))
            throw new Exception($"Failed to create network: {errorProp.GetString()}");

        if (!root.TryGetProperty("handle", out var handleProp))
            throw new Exception("Network created but no handle returned");

        return new Network(handleProp.GetInt64());
    }

    /// <summary>
    /// Loads a complete model from JSON string (ONE LINE - structure + weights all at once!).
    /// This is the easy way - no manual layer setup needed!
    /// </summary>
    /// <param name="modelJson">JSON string containing complete model</param>
    /// <param name="modelId">Optional model identifier (default: "loaded_model")</param>
    /// <returns>Fully configured Network with all weights loaded</returns>
    /// <example>
    /// <code>
    /// // Load model from file
    /// string modelJson = File.ReadAllText("model.json");
    /// var network = Network.LoadFromString(modelJson, "my_model");
    /// 
    /// // Use immediately - no setup needed!
    /// float[] output = network.Forward(input);
    /// </code>
    /// </example>
    public static Network LoadFromString(string modelJson, string modelId = "loaded_model")
    {
        var responsePtr = NativeMethods.Loom_LoadModel(modelJson, modelId);
        var responseJson = NativeMethods.PtrToStringAndFree(responsePtr);

        if (string.IsNullOrEmpty(responseJson))
            throw new Exception("Failed to load model");

        using var doc = JsonDocument.Parse(responseJson);
        var root = doc.RootElement;

        if (root.TryGetProperty("error", out var errorProp))
            throw new Exception($"Failed to load model: {errorProp.GetString()}");

        if (!root.TryGetProperty("handle", out var handleProp))
            throw new Exception("Model loaded but no handle returned");

        return new Network(handleProp.GetInt64());
    }

    /// <summary>
    /// Saves the model to JSON string (structure + weights).
    /// </summary>
    /// <param name="modelId">Model identifier (default: "saved_model")</param>
    /// <returns>JSON string containing complete model</returns>
    /// <example>
    /// <code>
    /// string modelJson = network.SaveToString("my_model");
    /// File.WriteAllText("model.json", modelJson);
    /// </code>
    /// </example>
    public string SaveToString(string modelId = "saved_model")
    {
        ThrowIfDisposed();

        var responsePtr = NativeMethods.Loom_SaveModel(_handle, modelId);
        var responseJson = NativeMethods.PtrToStringAndFree(responsePtr);

        if (string.IsNullOrEmpty(responseJson))
            throw new Exception("Failed to save model");

        // Check if it's an error wrapped in JSON
        try
        {
            using var doc = JsonDocument.Parse(responseJson);
            var root = doc.RootElement;

            if (root.ValueKind == JsonValueKind.Object &&
                root.TryGetProperty("error", out var errorProp))
            {
                throw new Exception($"Failed to save model: {errorProp.GetString()}");
            }
        }
        catch (JsonException)
        {
            // Not JSON, probably the raw model string - this is fine
        }

        return responseJson;
    }

    /// <summary>
    /// Performs forward pass through the network.
    /// </summary>
    /// <param name="input">Input vector</param>
    /// <returns>Output vector</returns>
    public float[] Forward(float[] input)
    {
        ThrowIfDisposed();
        var result = JsonCall("ForwardCPU", input);

        if (result.Count == 0)
            throw new Exception("No output from forward pass");

        // First element is the output array
        return result[0].Deserialize<float[]>() 
            ?? throw new Exception("Failed to deserialize output");
    }

    /// <summary>
    /// Performs backward pass (backpropagation).
    /// </summary>
    /// <param name="gradOutput">Gradient of output</param>
    /// <returns>Gradient of input</returns>
    public float[] Backward(float[] gradOutput)
    {
        ThrowIfDisposed();
        var result = JsonCall("BackwardCPU", gradOutput);

        if (result.Count == 0)
            return Array.Empty<float>();

        return result[0].Deserialize<float[]>() ?? Array.Empty<float>();
    }

    /// <summary>
    /// Updates network weights using computed gradients.
    /// </summary>
    /// <param name="learningRate">Learning rate for weight updates</param>
    public void UpdateWeights(float learningRate)
    {
        ThrowIfDisposed();
        JsonCall("UpdateWeights", learningRate);
    }

    /// <summary>
    /// Zeros out all gradients in the network.
    /// </summary>
    public void ZeroGradients()
    {
        ThrowIfDisposed();
        JsonCall("ZeroGradients");
    }

    /// <summary>
    /// Gets network information as a dictionary.
    /// </summary>
    /// <returns>Dictionary with network metadata</returns>
    public Dictionary<string, object> GetInfo()
    {
        ThrowIfDisposed();

        var responsePtr = NativeMethods.Loom_GetInfo(_handle);
        var responseJson = NativeMethods.PtrToStringAndFree(responsePtr);

        if (string.IsNullOrEmpty(responseJson))
            throw new Exception("Failed to get network info");

        return JsonSerializer.Deserialize<Dictionary<string, object>>(responseJson)
            ?? new Dictionary<string, object>();
    }

    /// <summary>
    /// Gets the LOOM library version string.
    /// </summary>
    /// <returns>Version string (e.g., "0.0.2")</returns>
    public static string GetVersion()
    {
        var ptr = NativeMethods.Loom_GetVersion();
        return NativeMethods.PtrToStringAndFree(ptr);
    }

    /// <summary>
    /// Calls a method on the network using the reflection API.
    /// </summary>
    private List<JsonElement> JsonCall(string methodName, params object[] args)
    {
        var argsJson = JsonSerializer.Serialize(args);
        var responsePtr = NativeMethods.Loom_Call(_handle, methodName, argsJson);
        var responseJson = NativeMethods.PtrToStringAndFree(responsePtr);

        if (string.IsNullOrEmpty(responseJson))
            throw new Exception($"No response from {methodName}");

        using var doc = JsonDocument.Parse(responseJson);
        var root = doc.RootElement;

        // Check for error response
        if (root.ValueKind == JsonValueKind.Object &&
            root.TryGetProperty("error", out var errorProp))
        {
            throw new Exception($"C library error: {errorProp.GetString()}");
        }

        // Response should be an array of return values
        if (root.ValueKind != JsonValueKind.Array)
            throw new Exception($"Unexpected response format from {methodName}");

        var results = new List<JsonElement>();
        foreach (var element in root.EnumerateArray())
        {
            results.Add(element.Clone());
        }

        return results;
    }

    /// <summary>
    /// Throws if the network has been disposed.
    /// </summary>
    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(Network));
    }

    /// <summary>
    /// Disposes the network and frees native resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        if (_handle != 0)
        {
            NativeMethods.Loom_Free(_handle);
            _handle = 0;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Finalizer to ensure native resources are freed.
    /// </summary>
    ~Network()
    {
        Dispose();
    }
}
