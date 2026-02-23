/// Idiomatic Dart API for the Loom inference engine.
///
/// Uses [LoomNetwork] as the main entry point. All JSON marshalling is
/// handled internally — callers work with plain Dart [List] and [Map] values.
library loom_api;

import 'dart:convert';
import 'dart:ffi';
import 'package:ffi/ffi.dart';

import 'bindings.dart' as _b;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert a native `char*` result to a Dart [String] and free the memory.
String _str(Pointer<Utf8> ptr) {
  if (ptr == nullptr) return '';
  final s = ptr.toDartString();
  _b.freeLoomString(ptr);
  return s;
}

/// Encode [obj] to a JSON UTF-8 native string. Caller must [calloc.free] it.
Pointer<Utf8> _enc(Object obj) => jsonEncode(obj).toNativeUtf8();

/// Allocate a native float array from a Dart [List<double>].
/// Caller must [calloc.free] the returned pointer.
Pointer<Float> _floats(List<double> data) {
  final ptr = calloc<Float>(data.length);
  for (var i = 0; i < data.length; i++) {
    ptr[i] = data[i];
  }
  return ptr;
}

/// Parse a JSON string and check for an `error` key.
Map<String, dynamic> _parse(String json) {
  final m = jsonDecode(json) as Map<String, dynamic>;
  if (m.containsKey('error')) throw LoomException(m['error'] as String);
  return m;
}

// ---------------------------------------------------------------------------
// Exception
// ---------------------------------------------------------------------------

/// Thrown when the native library returns an error JSON.
class LoomException implements Exception {
  final String message;
  const LoomException(this.message);
  @override
  String toString() => 'LoomException: $message';
}

// ---------------------------------------------------------------------------
// LoomNetwork
// ---------------------------------------------------------------------------

/// High-level wrapper around the Loom global-network API.
///
/// ```dart
/// final net = LoomNetwork.create({
///   'grid_rows': 2, 'grid_cols': 2, 'layers_per_cell': 3,
///   'input_size': 2,
/// });
/// final out = net.forward([0.0, 1.0]);
/// net.dispose();
/// ```
class LoomNetwork {
  LoomNetwork._();

  // ---- Lifecycle -----------------------------------------------------------

  /// Create a network from a JSON config map.
  ///
  /// Typical keys: `grid_rows`, `grid_cols`, `layers_per_cell`, `input_size`,
  /// `batch_size`, `use_gpu`.
  factory LoomNetwork.create(Map<String, dynamic> config) {
    final cfg = _enc(config);
    final res = _str(_b.createLoomNetwork(cfg));
    calloc.free(cfg);
    _parse(res); // throws on error
    return LoomNetwork._();
  }

  /// Load a previously saved model JSON.
  factory LoomNetwork.load(String modelJson, {String modelId = 'model'}) {
    final js = _enc(modelJson);
    final id = modelId.toNativeUtf8();
    final res = _str(_b.loomLoadModel(js, id));
    calloc.free(js);
    calloc.free(id);
    _parse(res);
    return LoomNetwork._();
  }

  /// Release the global network and any GPU resources.
  void dispose() => _b.freeLoomNetwork();

  // ---- Inference -----------------------------------------------------------

  /// Run a single forward pass. Returns the output vector.
  List<double> forward(List<double> inputs) {
    final ptr = _floats(inputs);
    final res = _str(_b.loomForward(ptr, inputs.length));
    calloc.free(ptr);
    final decoded = jsonDecode(res);
    if (decoded is List) return decoded.cast<double>();
    throw LoomException('Unexpected forward response: $res');
  }

  /// Run a backward pass with gradient vector.
  void backward(List<double> gradients) {
    final ptr = _floats(gradients);
    final res = _str(_b.loomBackward(ptr, gradients.length));
    calloc.free(ptr);
    _parse(res);
  }

  /// Update weights using plain SGD.
  void updateWeights(double learningRate) =>
      _b.loomUpdateWeights(learningRate);

  /// Apply AdamW gradients.
  void applyAdamW({
    required double learningRate,
    double beta1 = 0.9,
    double beta2 = 0.999,
    double weightDecay = 0.01,
  }) =>
      _b.loomApplyGradientsAdamW(learningRate, beta1, beta2, weightDecay);

  // ---- Training ------------------------------------------------------------

  /// Train with a list of [TrainingBatch] maps and a config map.
  Map<String, dynamic> train(
      List<Map<String, dynamic>> batches, Map<String, dynamic> config) {
    final b = _enc(batches);
    final c = _enc(config);
    final res = _str(_b.loomTrain(b, c));
    calloc.free(b);
    calloc.free(c);
    return _parse(res);
  }

  /// Train with 2-D inputs / targets arrays.
  Map<String, dynamic> trainStandard(
    List<List<double>> inputs,
    List<List<double>> targets,
    Map<String, dynamic> config,
  ) {
    final i = _enc(inputs);
    final t = _enc(targets);
    final c = _enc(config);
    final res = _str(_b.loomTrainStandard(i, t, c));
    calloc.free(i);
    calloc.free(t);
    calloc.free(c);
    return _parse(res);
  }

  // ---- Persistence ---------------------------------------------------------

  /// Serialise the current network to a JSON string.
  String save({String modelId = 'model'}) {
    final id = modelId.toNativeUtf8();
    final res = _str(_b.loomSaveModel(id));
    calloc.free(id);
    return res;
  }

  // ---- Info / Evaluation ---------------------------------------------------

  /// Return basic network metadata (grid_rows, grid_cols, etc.).
  Map<String, dynamic> info() => _parse(_str(_b.loomGetNetworkInfo()));

  /// Evaluate accuracy. [expectedOutputs] is a 1-D array of expected values.
  Map<String, dynamic> evaluate(
      List<List<double>> inputs, List<double> expectedOutputs) {
    final i = _enc(inputs);
    final e = _enc(expectedOutputs);
    final res = _str(_b.loomEvaluateNetwork(i, e));
    calloc.free(i);
    calloc.free(e);
    return _parse(res);
  }

  /// Sync GPU state (no-op on CPU-only builds).
  void syncGPU() => _b.loomSyncGPU();
}

// ---------------------------------------------------------------------------
// StepState
// ---------------------------------------------------------------------------

/// Fine-grained step-by-step execution handle.
class LoomStepState {
  final int _handle;

  LoomStepState._(this._handle);

  factory LoomStepState.create(int inputSize) {
    final h = _b.loomInitStepState(inputSize);
    if (h < 0) throw const LoomException('Failed to init StepState — no network?');
    return LoomStepState._(h);
  }

  void setInput(List<double> input) {
    final ptr = _floats(input);
    _b.loomSetInput(_handle, ptr, input.length);
    calloc.free(ptr);
  }

  /// Returns elapsed nanoseconds.
  int stepForward() => _b.loomStepForward(_handle);

  List<double> getOutput() {
    final res = _str(_b.loomGetOutput(_handle));
    final decoded = jsonDecode(res);
    if (decoded is List) return decoded.cast<double>();
    throw LoomException('Unexpected getOutput response: $res');
  }

  Map<String, dynamic> stepBackward(List<double> gradients) {
    final ptr = _floats(gradients);
    final res = _str(_b.loomStepBackward(_handle, ptr, gradients.length));
    calloc.free(ptr);
    return _parse(res);
  }

  void dispose() => _b.loomFreeStepState(_handle);
}

// ---------------------------------------------------------------------------
// TweenState
// ---------------------------------------------------------------------------

/// Neural tweening handle for online adaptation.
class LoomTweenState {
  final int _handle;

  LoomTweenState._(this._handle);

  factory LoomTweenState.create({bool useChainRule = false}) {
    final h = _b.loomCreateTweenState(useChainRule ? 1 : 0);
    if (h < 0) throw const LoomException('Failed to create TweenState — no network?');
    return LoomTweenState._(h);
  }

  /// Returns gap (distance to target).
  double step({
    required List<double> input,
    required int targetClass,
    required int outputSize,
    required double learningRate,
  }) {
    final ptr = _floats(input);
    final gap = _b.loomTweenStep(_handle, ptr, input.length, targetClass, outputSize, learningRate);
    calloc.free(ptr);
    return gap;
  }

  void dispose() => _b.loomFreeTweenState(_handle);
}

// ---------------------------------------------------------------------------
// AdaptationTracker
// ---------------------------------------------------------------------------

/// Benchmark task-switching tracker.
class LoomAdaptationTracker {
  final int _handle;

  LoomAdaptationTracker._(this._handle);

  factory LoomAdaptationTracker.create({
    required int windowMs,
    required int totalMs,
  }) {
    final h = _b.loomCreateAdaptationTracker(windowMs, totalMs);
    return LoomAdaptationTracker._(h);
  }

  void setModelInfo(String modelName, String modeName) {
    final m = modelName.toNativeUtf8();
    final n = modeName.toNativeUtf8();
    _b.loomTrackerSetModelInfo(_handle, m, n);
    calloc.free(m);
    calloc.free(n);
  }

  void start(String taskName, int taskId) {
    final t = taskName.toNativeUtf8();
    _b.loomTrackerStart(_handle, t, taskId);
    calloc.free(t);
  }

  /// Returns previous task ID.
  int recordOutput(bool isCorrect) =>
      _b.loomTrackerRecordOutput(_handle, isCorrect ? 1 : 0);

  int get currentTask => _b.loomTrackerGetCurrentTask(_handle);

  Map<String, dynamic> finalize() =>
      _parse(_str(_b.loomTrackerFinalize(_handle)));

  void dispose() => _b.loomFreeTracker(_handle);
}

// ---------------------------------------------------------------------------
// LR Schedulers
// ---------------------------------------------------------------------------

/// Learning-rate scheduler backed by a native handle.
class LoomScheduler {
  final int _handle;

  LoomScheduler._(this._handle);

  factory LoomScheduler.constant(double lr) =>
      LoomScheduler._(_b.loomCreateConstantScheduler(lr));

  factory LoomScheduler.linearDecay(double start, double end, int steps) =>
      LoomScheduler._(_b.loomCreateLinearDecayScheduler(start, end, steps));

  factory LoomScheduler.cosine(double start, double min, int steps) =>
      LoomScheduler._(_b.loomCreateCosineScheduler(start, min, steps));

  double getLR(int step) => _b.loomSchedulerGetLR(_handle, step);

  void dispose() => _b.loomFreeScheduler(_handle);
}
