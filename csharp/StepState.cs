using System;
using System.Text.Json;
using System.Collections.Generic;

namespace Welvet;

/// <summary>
/// Manages the state for fine-grained stepping execution of the network.
/// Useful for RNNs, LSTMs, and other stateful architectures where you need
/// control over the execution loop.
/// </summary>
public class StepState : IDisposable
{
    private long _handle;
    private bool _disposed;

    /// <summary>
    /// Gets the native handle for this step state.
    /// </summary>
    public long Handle => _handle;

    /// <summary>
    /// Initialize a new stepping state.
    /// </summary>
    /// <param name="inputSize">Size of the input vector</param>
    public StepState(int inputSize)
    {
        _handle = NativeMethods.LoomInitStepState(inputSize);
        if (_handle < 0)
        {
            throw new Exception("Failed to initialize step state");
        }
    }

    /// <summary>
    /// Set the input data for the current step.
    /// </summary>
    /// <param name="input">Input vector</param>
    public void SetInput(float[] input)
    {
        ThrowIfDisposed();
        NativeMethods.LoomSetInput(_handle, input, input.Length);
    }

    /// <summary>
    /// Execute forward pass for one step.
    /// </summary>
    /// <returns>Duration in nanoseconds</returns>
    public long StepForward()
    {
        ThrowIfDisposed();
        return NativeMethods.LoomStepForward(_handle);
    }

    /// <summary>
    /// Get the output of the network for the current step.
    /// </summary>
    /// <returns>Output vector</returns>
    public float[] GetOutput()
    {
        ThrowIfDisposed();
        var responsePtr = NativeMethods.LoomGetOutput(_handle);
        var responseJson = NativeMethods.PtrToStringAndFree(responsePtr);

        if (string.IsNullOrEmpty(responseJson))
            throw new Exception("Failed to get output");

        try
        {
            // The C API returns a JSON array of floats directly for GetOutput
            // Or it might return an object with "error"
            using var doc = JsonDocument.Parse(responseJson);
            var root = doc.RootElement;

            if (root.ValueKind == JsonValueKind.Object && root.TryGetProperty("error", out var errorProp))
            {
                throw new Exception($"Failed to get output: {errorProp.GetString()}");
            }

            if (root.ValueKind == JsonValueKind.Array)
            {
                return JsonSerializer.Deserialize<float[]>(responseJson) ?? Array.Empty<float>();
            }
            
            return Array.Empty<float>();
        }
        catch (JsonException)
        {
            throw new Exception($"Invalid JSON response: {responseJson}");
        }
    }

    /// <summary>
    /// Execute backward pass for one step.
    /// </summary>
    /// <param name="gradients">Gradient vector from the next layer/step</param>
    /// <returns>Dictionary containing 'grad_input' and 'duration'</returns>
    public Dictionary<string, object> StepBackward(float[] gradients)
    {
        ThrowIfDisposed();
        var responsePtr = NativeMethods.LoomStepBackward(_handle, gradients, gradients.Length);
        var responseJson = NativeMethods.PtrToStringAndFree(responsePtr);

        if (string.IsNullOrEmpty(responseJson))
            throw new Exception("Failed to step backward");

        try
        {
            using var doc = JsonDocument.Parse(responseJson);
            var root = doc.RootElement;

            if (root.ValueKind == JsonValueKind.Object && root.TryGetProperty("error", out var errorProp))
            {
                throw new Exception($"Failed to step backward: {errorProp.GetString()}");
            }

            return JsonSerializer.Deserialize<Dictionary<string, object>>(responseJson) 
                ?? new Dictionary<string, object>();
        }
        catch (JsonException)
        {
            throw new Exception($"Invalid JSON response: {responseJson}");
        }
    }

    /// <summary>
    /// Throws if the state has been disposed.
    /// </summary>
    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(StepState));
    }

    /// <summary>
    /// Disposes the step state and frees native resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        if (_handle >= 0)
        {
            NativeMethods.LoomFreeStepState(_handle);
            _handle = -1;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Finalizer to ensure native resources are freed.
    /// </summary>
    ~StepState()
    {
        Dispose();
    }
}
