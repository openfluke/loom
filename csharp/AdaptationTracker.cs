using System;
using System.Text.Json;

namespace Welvet;

/// <summary>
/// Tracks accuracy across task changes for benchmarking real-time adaptation.
/// The tracker divides the test duration into windows and records accuracy
/// in each window. It also supports scheduling task changes at specific times.
/// </summary>
public class AdaptationTracker : IDisposable
{
    private long _handle;
    private bool _disposed;

    /// <summary>
    /// Gets the native handle for this tracker.
    /// </summary>
    public long Handle => _handle;

    /// <summary>
    /// Creates a new AdaptationTracker.
    /// </summary>
    /// <param name="windowDurationMs">Duration of each accuracy window in milliseconds</param>
    /// <param name="totalDurationMs">Total test duration in milliseconds</param>
    public AdaptationTracker(int windowDurationMs = 1000, int totalDurationMs = 10000)
    {
        _handle = NativeMethods.LoomCreateAdaptationTracker(windowDurationMs, totalDurationMs);
        if (_handle < 0)
        {
            throw new Exception("Failed to create AdaptationTracker");
        }
    }

    /// <summary>
    /// Sets model information for the tracker.
    /// </summary>
    /// <param name="modelName">Name of the model (e.g., "Dense-5L")</param>
    /// <param name="modeName">Training mode name (e.g., "StepTweenChain")</param>
    public void SetModelInfo(string modelName, string modeName)
    {
        ThrowIfDisposed();
        NativeMethods.LoomTrackerSetModelInfo(_handle, modelName, modeName);
    }

    /// <summary>
    /// Schedules a task change at a specific offset from start.
    /// </summary>
    /// <param name="atOffsetMs">Milliseconds from start when task should change</param>
    /// <param name="taskId">New task ID</param>
    /// <param name="taskName">New task name</param>
    public void ScheduleTaskChange(int atOffsetMs, int taskId, string taskName)
    {
        ThrowIfDisposed();
        NativeMethods.LoomTrackerScheduleTaskChange(_handle, atOffsetMs, taskId, taskName);
    }

    /// <summary>
    /// Starts the tracker with an initial task.
    /// </summary>
    /// <param name="taskName">Initial task name</param>
    /// <param name="taskId">Initial task ID</param>
    public void Start(string taskName, int taskId = 0)
    {
        ThrowIfDisposed();
        NativeMethods.LoomTrackerStart(_handle, taskName, taskId);
    }

    /// <summary>
    /// Records an output and whether it was correct.
    /// </summary>
    /// <param name="isCorrect">Whether the network output was correct</param>
    /// <returns>Previous task ID (for detecting task changes)</returns>
    public int RecordOutput(bool isCorrect)
    {
        ThrowIfDisposed();
        return NativeMethods.LoomTrackerRecordOutput(_handle, isCorrect ? 1 : 0);
    }

    /// <summary>
    /// Gets the current task ID.
    /// </summary>
    /// <returns>Current task ID</returns>
    public int GetCurrentTask()
    {
        ThrowIfDisposed();
        return NativeMethods.LoomTrackerGetCurrentTask(_handle);
    }

    /// <summary>
    /// Finalizes tracking and get results.
    /// </summary>
    /// <returns>Results as a JsonDocument containing avg_accuracy, total_outputs, window_accuracies, etc.</returns>
    public JsonDocument GetResults()
    {
        ThrowIfDisposed();
        var responsePtr = NativeMethods.LoomTrackerFinalize(_handle);
        var responseJson = NativeMethods.PtrToStringAndFree(responsePtr);

        if (string.IsNullOrEmpty(responseJson))
            throw new Exception("Failed to finalize tracker");

        return JsonDocument.Parse(responseJson);
    }

    /// <summary>
    /// Finalizes tracking and returns results as a parsed object.
    /// </summary>
    /// <returns>AdaptationResult with parsed fields</returns>
    public AdaptationResult FinalizeResult()
    {
        using var doc = GetResults();
        var root = doc.RootElement;

        return new AdaptationResult
        {
            ModelName = root.TryGetProperty("model_name", out var mn) ? mn.GetString() ?? "" : "",
            ModeName = root.TryGetProperty("mode_name", out var mode) ? mode.GetString() ?? "" : "",
            AvgAccuracy = root.TryGetProperty("avg_accuracy", out var acc) ? acc.GetDouble() : 0,
            TotalOutputs = root.TryGetProperty("total_outputs", out var total) ? total.GetInt32() : 0
        };
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AdaptationTracker));
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        if (_handle >= 0)
        {
            NativeMethods.LoomFreeTracker(_handle);
            _handle = -1;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~AdaptationTracker()
    {
        Dispose();
    }
}

/// <summary>
/// Results from AdaptationTracker.FinalizeResult()
/// </summary>
public class AdaptationResult
{
    public string ModelName { get; set; } = "";
    public string ModeName { get; set; } = "";
    public double AvgAccuracy { get; set; }
    public int TotalOutputs { get; set; }
}
