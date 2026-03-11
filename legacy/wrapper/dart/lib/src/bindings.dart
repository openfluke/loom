// ignore_for_file: non_constant_identifier_names, camel_case_types

/// Low-level dart:ffi bindings for the Loom C ABI (libloom).
///
/// All exported C functions are declared here as [DynamicLibrary] lookups.
/// Prefer [loom_api.dart] for the idiomatic Dart API.
library loom_bindings;

import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

// ---------------------------------------------------------------------------
// Native type aliases
// ---------------------------------------------------------------------------
typedef _VoidFunc = Void Function();
typedef _CharPFunc = Pointer<Utf8> Function();

// ---------------------------------------------------------------------------
// Library loader
// ---------------------------------------------------------------------------

/// Loads the native libloom shared library for the current platform.
DynamicLibrary _openLib() {
  if (Platform.isWindows) return DynamicLibrary.open('libloom.dll');
  if (Platform.isMacOS) return DynamicLibrary.open('libloom.dylib');
  return DynamicLibrary.open('libloom.so');
}

/// Singleton access to the native library. Throws if the library is not found.
final DynamicLibrary LoomLib = _openLib();

// ---------------------------------------------------------------------------
// Helper: look up a symbol, return null if absent
// ---------------------------------------------------------------------------
T? _sym<T extends Function>(String name, {required T Function(Pointer<NativeFunction<dynamic>>) cast}) {
  // ignore symbol-not-found errors gracefully
  try {
    final ptr = LoomLib.lookup<NativeFunction<dynamic>>(name);
    return cast(ptr);
  } catch (_) {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Simple / Global API
// ---------------------------------------------------------------------------

/// `char* CreateLoomNetwork(const char* jsonConfig)`
final Pointer<Utf8> Function(Pointer<Utf8>) createLoomNetwork = LoomLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Utf8>)>>('CreateLoomNetwork')
    .asFunction();

/// `void FreeLoomNetwork()`
final void Function() freeLoomNetwork = LoomLib
    .lookup<NativeFunction<Void Function()>>('FreeLoomNetwork')
    .asFunction();

/// `void FreeLoomString(char*)`
final void Function(Pointer<Utf8>) freeLoomString = LoomLib
    .lookup<NativeFunction<Void Function(Pointer<Utf8>)>>('FreeLoomString')
    .asFunction();

/// `char* LoomForward(float* inputs, int length)`
final Pointer<Utf8> Function(Pointer<Float>, int) loomForward = LoomLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Float>, Int32)>>('LoomForward')
    .asFunction();

/// `char* LoomBackward(float* gradients, int length)`
final Pointer<Utf8> Function(Pointer<Float>, int) loomBackward = LoomLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Float>, Int32)>>('LoomBackward')
    .asFunction();

/// `void LoomUpdateWeights(float learningRate)`
final void Function(double) loomUpdateWeights = LoomLib
    .lookup<NativeFunction<Void Function(Float)>>('LoomUpdateWeights')
    .asFunction();

/// `char* LoomTrain(const char* batchesJSON, const char* configJSON)`
final Pointer<Utf8> Function(Pointer<Utf8>, Pointer<Utf8>) loomTrain = LoomLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Utf8>, Pointer<Utf8>)>>('LoomTrain')
    .asFunction();

/// `char* LoomTrainStandard(const char* inputsJSON, const char* targetsJSON, const char* configJSON)`
final Pointer<Utf8> Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>) loomTrainStandard = LoomLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>)>>('LoomTrainStandard')
    .asFunction();

/// `char* LoomSaveModel(const char* modelID)`
final Pointer<Utf8> Function(Pointer<Utf8>) loomSaveModel = LoomLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Utf8>)>>('LoomSaveModel')
    .asFunction();

/// `char* LoomLoadModel(const char* jsonString, const char* modelID)`
final Pointer<Utf8> Function(Pointer<Utf8>, Pointer<Utf8>) loomLoadModel = LoomLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Utf8>, Pointer<Utf8>)>>('LoomLoadModel')
    .asFunction();

/// `char* LoomGetNetworkInfo()`
final Pointer<Utf8> Function() loomGetNetworkInfo = LoomLib
    .lookup<NativeFunction<Pointer<Utf8> Function()>>('LoomGetNetworkInfo')
    .asFunction();

/// `char* LoomEvaluateNetwork(const char* inputsJSON, const char* expectedJSON)`
final Pointer<Utf8> Function(Pointer<Utf8>, Pointer<Utf8>) loomEvaluateNetwork = LoomLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Utf8>, Pointer<Utf8>)>>('LoomEvaluateNetwork')
    .asFunction();

/// `void LoomSyncGPU()`
final void Function() loomSyncGPU = LoomLib
    .lookup<NativeFunction<Void Function()>>('LoomSyncGPU')
    .asFunction();

// ---------------------------------------------------------------------------
// StepState API
// ---------------------------------------------------------------------------

/// `long long LoomInitStepState(int inputSize)`
final int Function(int) loomInitStepState = LoomLib
    .lookup<NativeFunction<Int64 Function(Int32)>>('LoomInitStepState')
    .asFunction();

/// `void LoomSetInput(long long handle, float* input, int length)`
final void Function(int, Pointer<Float>, int) loomSetInput = LoomLib
    .lookup<NativeFunction<Void Function(Int64, Pointer<Float>, Int32)>>('LoomSetInput')
    .asFunction();

/// `long long LoomStepForward(long long handle)` → nanoseconds
final int Function(int) loomStepForward = LoomLib
    .lookup<NativeFunction<Int64 Function(Int64)>>('LoomStepForward')
    .asFunction();

/// `char* LoomGetOutput(long long handle)`
final Pointer<Utf8> Function(int) loomGetOutput = LoomLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Int64)>>('LoomGetOutput')
    .asFunction();

/// `char* LoomStepBackward(long long handle, float* gradients, int length)`
final Pointer<Utf8> Function(int, Pointer<Float>, int) loomStepBackward = LoomLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Int64, Pointer<Float>, Int32)>>('LoomStepBackward')
    .asFunction();

/// `void LoomApplyGradients(float learningRate)`
final void Function(double) loomApplyGradients = LoomLib
    .lookup<NativeFunction<Void Function(Float)>>('LoomApplyGradients')
    .asFunction();

/// `void LoomApplyGradientsAdamW(float lr, float beta1, float beta2, float weightDecay)`
final void Function(double, double, double, double) loomApplyGradientsAdamW = LoomLib
    .lookup<NativeFunction<Void Function(Float, Float, Float, Float)>>('LoomApplyGradientsAdamW')
    .asFunction();

/// `void LoomFreeStepState(long long handle)`
final void Function(int) loomFreeStepState = LoomLib
    .lookup<NativeFunction<Void Function(Int64)>>('LoomFreeStepState')
    .asFunction();

// ---------------------------------------------------------------------------
// TweenState API
// ---------------------------------------------------------------------------

/// `long long LoomCreateTweenState(int useChainRule)`
final int Function(int) loomCreateTweenState = LoomLib
    .lookup<NativeFunction<Int64 Function(Int32)>>('LoomCreateTweenState')
    .asFunction();

/// `float LoomTweenStep(long long handle, float* input, int inputLen, int targetClass, int outputSize, float lr)`
final double Function(int, Pointer<Float>, int, int, int, double) loomTweenStep = LoomLib
    .lookup<NativeFunction<Float Function(Int64, Pointer<Float>, Int32, Int32, Int32, Float)>>('LoomTweenStep')
    .asFunction();

/// `void LoomFreeTweenState(long long handle)`
final void Function(int) loomFreeTweenState = LoomLib
    .lookup<NativeFunction<Void Function(Int64)>>('LoomFreeTweenState')
    .asFunction();

// ---------------------------------------------------------------------------
// AdaptationTracker API
// ---------------------------------------------------------------------------

final int Function(int, int) loomCreateAdaptationTracker = LoomLib
    .lookup<NativeFunction<Int64 Function(Int32, Int32)>>('LoomCreateAdaptationTracker')
    .asFunction();

final void Function(int, Pointer<Utf8>, Pointer<Utf8>) loomTrackerSetModelInfo = LoomLib
    .lookup<NativeFunction<Void Function(Int64, Pointer<Utf8>, Pointer<Utf8>)>>('LoomTrackerSetModelInfo')
    .asFunction();

final void Function(int, Pointer<Utf8>, int) loomTrackerStart = LoomLib
    .lookup<NativeFunction<Void Function(Int64, Pointer<Utf8>, Int32)>>('LoomTrackerStart')
    .asFunction();

final int Function(int, int) loomTrackerRecordOutput = LoomLib
    .lookup<NativeFunction<Int32 Function(Int64, Int32)>>('LoomTrackerRecordOutput')
    .asFunction();

final int Function(int) loomTrackerGetCurrentTask = LoomLib
    .lookup<NativeFunction<Int32 Function(Int64)>>('LoomTrackerGetCurrentTask')
    .asFunction();

final Pointer<Utf8> Function(int) loomTrackerFinalize = LoomLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Int64)>>('LoomTrackerFinalize')
    .asFunction();

final void Function(int) loomFreeTracker = LoomLib
    .lookup<NativeFunction<Void Function(Int64)>>('LoomFreeTracker')
    .asFunction();

// ---------------------------------------------------------------------------
// Scheduler API
// ---------------------------------------------------------------------------

final int Function(double) loomCreateConstantScheduler = LoomLib
    .lookup<NativeFunction<Int64 Function(Float)>>('LoomCreateConstantScheduler')
    .asFunction();

final int Function(double, double, int) loomCreateLinearDecayScheduler = LoomLib
    .lookup<NativeFunction<Int64 Function(Float, Float, Int32)>>('LoomCreateLinearDecayScheduler')
    .asFunction();

final int Function(double, double, int) loomCreateCosineScheduler = LoomLib
    .lookup<NativeFunction<Int64 Function(Float, Float, Int32)>>('LoomCreateCosineScheduler')
    .asFunction();

final double Function(int, int) loomSchedulerGetLR = LoomLib
    .lookup<NativeFunction<Float Function(Int64, Int32)>>('LoomSchedulerGetLR')
    .asFunction();

final void Function(int) loomFreeScheduler = LoomLib
    .lookup<NativeFunction<Void Function(Int64)>>('LoomFreeScheduler')
    .asFunction();

// ---------------------------------------------------------------------------
// K-Means / Statistics
// ---------------------------------------------------------------------------

final Pointer<Utf8> Function(Pointer<Utf8>, int, int) loomKMeansCluster = LoomLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Utf8>, Int32, Int32)>>('LoomKMeansCluster')
    .asFunction();

final double Function(Pointer<Utf8>, Pointer<Utf8>) loomSilhouetteScore = LoomLib
    .lookup<NativeFunction<Float Function(Pointer<Utf8>, Pointer<Utf8>)>>('LoomSilhouetteScore')
    .asFunction();

final Pointer<Utf8> Function(Pointer<Utf8>) loomComputeCorrelation = LoomLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Utf8>)>>('LoomComputeCorrelation')
    .asFunction();

// ---------------------------------------------------------------------------
// Grafting API
// ---------------------------------------------------------------------------

final int Function(Pointer<Utf8>) loomCreateNetworkForGraft = LoomLib
    .lookup<NativeFunction<Int64 Function(Pointer<Utf8>)>>('LoomCreateNetworkForGraft')
    .asFunction();

final Pointer<Utf8> Function(Pointer<Utf8>, Pointer<Utf8>) loomGraftNetworks = LoomLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Utf8>, Pointer<Utf8>)>>('LoomGraftNetworks')
    .asFunction();

final void Function(int) loomFreeGraftNetwork = LoomLib
    .lookup<NativeFunction<Void Function(Int64)>>('LoomFreeGraftNetwork')
    .asFunction();
