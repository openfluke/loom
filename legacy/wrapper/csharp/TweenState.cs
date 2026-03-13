using System;

namespace Welvet;

/// <summary>
/// Manages tween state for neural tweening - enables networks to adapt 
/// to changing goals in real-time without full backpropagation.
/// </summary>
public class TweenState : IDisposable
{
    private long _handle;
    private bool _disposed;

    /// <summary>
    /// Gets the native handle for this tween state.
    /// </summary>
    public long Handle => _handle;

    /// <summary>
    /// Creates a new TweenState for neural tweening.
    /// </summary>
    /// <param name="useChainRule">If true, use chain rule (TweenChain mode)</param>
    public TweenState(bool useChainRule = false)
    {
        _handle = NativeMethods.LoomCreateTweenState(useChainRule ? 1 : 0);
        if (_handle < 0)
        {
            throw new Exception("Failed to create TweenState - ensure network is created first");
        }
    }

    /// <summary>
    /// Applies one tween step to adapt the network toward the target.
    /// </summary>
    /// <param name="input">Input vector</param>
    /// <param name="targetClass">Target class index (0-based)</param>
    /// <param name="outputSize">Number of output classes</param>
    /// <param name="learningRate">Learning rate for weight updates</param>
    /// <returns>Gap value (distance to target)</returns>
    public float Step(float[] input, int targetClass, int outputSize, float learningRate = 0.02f)
    {
        ThrowIfDisposed();
        return NativeMethods.LoomTweenStep(_handle, input, input.Length, targetClass, outputSize, learningRate);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(TweenState));
    }

    /// <summary>
    /// Disposes the tween state and frees native resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        if (_handle >= 0)
        {
            NativeMethods.LoomFreeTweenState(_handle);
            _handle = -1;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~TweenState()
    {
        Dispose();
    }
}
