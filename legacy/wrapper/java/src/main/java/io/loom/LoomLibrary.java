package io.loom;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

/**
 * JNA interface mapping every exported symbol from the Loom C ABI (libloom).
 *
 * <p>JNA resolves {@code libloom} to:
 * <ul>
 *   <li>Windows  → {@code libloom.dll}
 *   <li>Linux    → {@code libloom.so}
 *   <li>macOS    → {@code libloom.dylib}
 * </ul>
 *
 * <p>All {@code char*} return values that must be freed by the caller are
 * typed as {@code Pointer} to prevent JNA from auto-releasing them before
 * we can call {@link #FreeLoomString(Pointer)}.  Use
 * {@link LoomNetwork#ptrToString(Pointer)} to convert safely.
 */
public interface LoomLibrary extends Library {

    /** Singleton instance loaded from the native library. */
    LoomLibrary INSTANCE = Native.load("libloom", LoomLibrary.class);

    // -------------------------------------------------------------------------
    // Simple / Global network API
    // -------------------------------------------------------------------------

    /** Create a network from a JSON config string. Returns status JSON. */
    Pointer CreateLoomNetwork(String jsonConfig);

    /** Release the global network and any GPU resources. */
    void FreeLoomNetwork();

    /** Free a {@code char*} string allocated by the library. */
    void FreeLoomString(Pointer str);

    /** Forward pass. {@code inputs} is a {@code float[length]} array. */
    Pointer LoomForward(float[] inputs, int length);

    /** Backward pass. {@code gradients} is a {@code float[length]} array. */
    Pointer LoomBackward(float[] gradients, int length);

    /** SGD weight update. */
    void LoomUpdateWeights(float learningRate);

    /** Batch training. Both arguments are JSON strings. */
    Pointer LoomTrain(String batchesJSON, String configJSON);

    /** Standard training with 2-D inputs/targets JSON and config JSON. */
    Pointer LoomTrainStandard(String inputsJSON, String targetsJSON, String configJSON);

    /** Label-based training. Labels are integer indices. */
    Pointer LoomTrainLabels(String inputsJSON, String labelsJSON, String configJSON);

    /** Save current model to JSON string. */
    Pointer LoomSaveModel(String modelID);

    /** Load model from JSON string. */
    Pointer LoomLoadModel(String jsonString, String modelID);

    /** Get basic network metadata as JSON. */
    Pointer LoomGetNetworkInfo();

    /** Evaluate accuracy. Returns metrics JSON. */
    Pointer LoomEvaluateNetwork(String inputsJSON, String expectedOutputsJSON);

    /** Sync GPU state (no-op on CPU-only builds). */
    void LoomSyncGPU();

    // -------------------------------------------------------------------------
    // Optimizer shortcuts
    // -------------------------------------------------------------------------

    void LoomApplyGradients(float learningRate);

    void LoomApplyGradientsAdamW(float learningRate, float beta1, float beta2, float weightDecay);

    void LoomApplyGradientsRMSprop(float learningRate, float alpha, float epsilon, float momentum);

    void LoomApplyGradientsSGDMomentum(float learningRate, float momentum, float dampening, int nesterov);

    // -------------------------------------------------------------------------
    // StepState API (fine-grained control)
    // -------------------------------------------------------------------------

    long LoomInitStepState(int inputSize);

    void LoomSetInput(long handle, float[] input, int length);

    /** Returns elapsed nanoseconds. */
    long LoomStepForward(long handle);

    Pointer LoomGetOutput(long handle);

    Pointer LoomStepBackward(long handle, float[] gradients, int length);

    void LoomFreeStepState(long handle);

    // -------------------------------------------------------------------------
    // TweenState API
    // -------------------------------------------------------------------------

    long LoomCreateTweenState(int useChainRule);

    /** Returns gap (distance to target class). */
    float LoomTweenStep(long handle, float[] input, int inputLen,
                        int targetClass, int outputSize, float learningRate);

    void LoomFreeTweenState(long handle);

    // -------------------------------------------------------------------------
    // AdaptationTracker API
    // -------------------------------------------------------------------------

    long LoomCreateAdaptationTracker(int windowDurationMs, int totalDurationMs);

    void LoomTrackerSetModelInfo(long handle, String modelName, String modeName);

    void LoomTrackerScheduleTaskChange(long handle, int atOffsetMs, int taskID, String taskName);

    void LoomTrackerStart(long handle, String taskName, int taskID);

    /** Returns previous task ID. */
    int LoomTrackerRecordOutput(long handle, int isCorrect);

    int LoomTrackerGetCurrentTask(long handle);

    Pointer LoomTrackerFinalize(long handle);

    void LoomFreeTracker(long handle);

    // -------------------------------------------------------------------------
    // Learning-rate Schedulers
    // -------------------------------------------------------------------------

    long LoomCreateConstantScheduler(float baseLR);

    long LoomCreateLinearDecayScheduler(float startLR, float endLR, int totalSteps);

    long LoomCreateCosineScheduler(float startLR, float minLR, int totalSteps);

    float LoomSchedulerGetLR(long handle, int step);

    Pointer LoomSchedulerName(long handle);

    void LoomFreeScheduler(long handle);

    // -------------------------------------------------------------------------
    // Statistical Tools
    // -------------------------------------------------------------------------

    Pointer LoomKMeansCluster(String dataJSON, int k, int maxIter);

    float LoomSilhouetteScore(String dataJSON, String assignmentsJSON);

    Pointer LoomComputeCorrelation(String dataJSON);

    // -------------------------------------------------------------------------
    // Network Grafting
    // -------------------------------------------------------------------------

    long LoomCreateNetworkForGraft(String jsonConfig);

    Pointer LoomGraftNetworks(String networkIDsJSON, String combineMode);

    void LoomFreeGraftNetwork(long handle);
}
