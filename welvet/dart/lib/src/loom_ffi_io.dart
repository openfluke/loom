// loom_ffi_io.dart
// Native FFI bindings to loom_cabi (Windows DLL / Linux SO / macOS dylib / Android SO).
// Mirrors the C-ABI exported by loom/welvet/cabi/.

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'dart:developer' show log;

import 'package:ffi/ffi.dart';

// ── Library loader ─────────────────────────────────────────────────────────────

/// Optional override for tests / local dev (`WELVET_NATIVE_DIR` env var).
String? _welvetNativeDirFromEnv() {
  final dir = Platform.environment['WELVET_NATIVE_DIR'];
  if (dir != null && dir.isNotEmpty) return dir;
  return null;
}

DynamicLibrary _tryOpen(String path) {
  return DynamicLibrary.open(path);
}

DynamicLibrary _openFirst(List<String> candidates) {
  for (final path in candidates) {
    try {
      return _tryOpen(path);
    } catch (_) {}
  }
  throw StateError(
    'Could not load Welvet native library. Tried: ${candidates.join(", ")}',
  );
}

/// Installed `welvet` package root (pub-cache tarball) for VM / `flutter test`.
String? _welvetPackageRoot() {
  var dir = Directory.current;
  while (true) {
    final config = File('${dir.path}/.dart_tool/package_config.json');
    if (!config.existsSync()) {
      final parent = dir.parent;
      if (parent.path == dir.path) return null;
      dir = parent;
      continue;
    }
    try {
      final packages =
          (jsonDecode(config.readAsStringSync()) as Map)['packages'] as List?;
      if (packages == null) return null;
      final projectRoot = config.parent.parent.path;
      for (final pkg in packages) {
        final map = pkg as Map<String, dynamic>;
        if (map['name'] != 'welvet') continue;
        final rootUri = map['rootUri'] as String;
        if (rootUri.startsWith('file:')) {
          return Uri.parse(rootUri).toFilePath();
        }
        final resolved =
            Uri.directory(projectRoot).resolve(rootUri).toFilePath();
        if (Directory('$resolved/native').existsSync()) return resolved;
      }
    } catch (_) {}
    return null;
  }
}

String _macosAppFrameworksLib() {
  final frameworksDir = File(Platform.resolvedExecutable).parent.parent.path;
  return '$frameworksDir/Frameworks/libwelvet.dylib';
}

List<String> _packageNativeCandidates(String subdir, String fileName) {
  final env = _welvetNativeDirFromEnv();
  if (env != null) {
    return ['$env/$fileName', '$env/lib$fileName'];
  }
  final candidates = <String>[
    'native/$subdir/$fileName',
    '../native/$subdir/$fileName',
  ];
  final root = _welvetPackageRoot();
  if (root != null) {
    candidates.insert(0, '$root/native/$subdir/$fileName');
  }
  return candidates;
}

DynamicLibrary _loadLib() {
  if (Platform.isWindows) {
  return _openFirst([
      'welvet.dll',
      ..._packageNativeCandidates('windows_amd64', 'welvet.dll'),
      ..._packageNativeCandidates('windows_arm64', 'welvet.dll'),
    ]);
  } else if (Platform.isAndroid) {
    return _openFirst([
      'libwelvet.so',
      ..._packageNativeCandidates('android/arm64-v8a', 'libwelvet.so'),
      ..._packageNativeCandidates('android/x86_64', 'libwelvet.so'),
    ]);
  } else if (Platform.isLinux) {
    return _openFirst([
      'libwelvet.so',
      ..._packageNativeCandidates('linux_amd64', 'libwelvet.so'),
      ..._packageNativeCandidates('linux_arm64', 'libwelvet.so'),
    ]);
  } else if (Platform.isMacOS) {
    // SoulGlitch: libwelvet.dylib in app Frameworks; VM tests use package native/.
    return _openFirst([
      _macosAppFrameworksLib(),
      'libwelvet.dylib',
      ..._packageNativeCandidates('macos_universal', 'libwelvet.dylib'),
      ..._packageNativeCandidates('macos_arm64', 'libwelvet.dylib'),
      ..._packageNativeCandidates('macos_amd64', 'libwelvet.dylib'),
    ]);
  } else if (Platform.isIOS) {
    return DynamicLibrary.process();
  }
  throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
}

// ── Native typedefs (C side) ───────────────────────────────────────────────────

typedef _FreeLoomStringC = Void Function(Pointer<Utf8> ptr);
typedef _FreeLoomStringD = void Function(Pointer<Utf8> ptr);

typedef _VoidLongLongC = Void Function(Int64 handle);
typedef _VoidLongLongD = void Function(int handle);

typedef _LongLongFromStringC = Int64 Function(Pointer<Utf8> json);
typedef _LongLongFromStringD = int Function(Pointer<Utf8> json);

typedef _LongLongFromLongLongC = Int64 Function(Int64 handle);
typedef _LongLongFromLongLongD = int Function(int handle);

typedef _StringFromLongLongC = Pointer<Utf8> Function(Int64 handle);
typedef _StringFromLongLongD = Pointer<Utf8> Function(int handle);

typedef _CreateStepStateC = Int64 Function(Int64 networkHandle, Int32 dtype);
typedef _CreateStepStateD = int Function(int networkHandle, int dtype);

typedef _SetInputC = Void Function(
    Int64 stateHandle, Pointer<Float> data, Int32 length);
typedef _SetInputD = void Function(
    int stateHandle, Pointer<Float> data, int length);

typedef _LoomStepC = Int64 Function(
    Int64 networkHandle, Int64 stateHandle, Int32 captureHistory);
typedef _LoomStepD = int Function(
    int networkHandle, int stateHandle, int captureHistory);

typedef _GetOutputC = Pointer<Utf8> Function(Int64 stateHandle, Int32 layerIdx);
typedef _GetOutputD = Pointer<Utf8> Function(int stateHandle, int layerIdx);

typedef _SequentialForwardC = Pointer<Utf8> Function(
    Int64 networkHandle, Pointer<Float> inputData, Int32 inputLen);
typedef _SequentialForwardD = Pointer<Utf8> Function(
    int networkHandle, Pointer<Float> inputData, int inputLen);

typedef _MorphLayerC = Pointer<Utf8> Function(
    Int64 networkHandle, Int32 layerIdx, Int32 targetDType);
typedef _MorphLayerD = Pointer<Utf8> Function(
    int networkHandle, int layerIdx, int targetDType);

typedef _GetLayerSpecC = Void Function(Int64 networkHandle, Int32 layerIdx,
    Pointer<Int32> outSpec);
typedef _GetLayerSpecD = void Function(
    int networkHandle, int layerIdx, Pointer<Int32> outSpec);

typedef _GetLayerStatsC = Void Function(
    Int64 stateHandle, Int32 layerIdx, Pointer<Float> outStats);
typedef _GetLayerStatsD = void Function(
    int stateHandle, int layerIdx, Pointer<Float> outStats);

// LoomCompareDNA takes two DNA JSON strings, not handles
typedef _CompareDNAC = Pointer<Utf8> Function(
    Pointer<Utf8> dna1JSON, Pointer<Utf8> dna2JSON);
typedef _CompareDNAD = Pointer<Utf8> Function(
    Pointer<Utf8> dna1JSON, Pointer<Utf8> dna2JSON);

// Extra exports
typedef _StringVoidC = Pointer<Utf8> Function();
typedef _StringVoidD = Pointer<Utf8> Function();

typedef _ExtractBlueprintC = Pointer<Utf8> Function(Int64 handle, Pointer<Utf8> id);
typedef _ExtractBlueprintD = Pointer<Utf8> Function(int handle, Pointer<Utf8> id);

typedef _GetLayerTelemetryC = Pointer<Utf8> Function(Int64 handle, Int32 layerIdx);
typedef _GetLayerTelemetryD = Pointer<Utf8> Function(int handle, int layerIdx);

typedef _DefaultNEATConfigC = Pointer<Utf8> Function(Int32 dModel);
typedef _DefaultNEATConfigD = Pointer<Utf8> Function(int dModel);

typedef _SpliceDNAC = Int64 Function(Int64 handleA, Int64 handleB, Pointer<Utf8> cfgJSON);
typedef _SpliceDNAD = int Function(int handleA, int handleB, Pointer<Utf8> cfgJSON);

typedef _StepBackwardC = Pointer<Utf8> Function(
    Int64 networkHandle, Int64 stateHandle, Pointer<Float> gradData, Int32 gradLen);
typedef _StepBackwardD = Pointer<Utf8> Function(
    int networkHandle, int stateHandle, Pointer<Float> gradData, int gradLen);

typedef _ApplyGradientsC = Void Function(
    Int64 networkHandle, Int32 layerIdx, Pointer<Utf8> gradJSON, Float lr);
typedef _ApplyGradientsD = void Function(
    int networkHandle, int layerIdx, Pointer<Utf8> gradJSON, double lr);

typedef _IsGPUC = Int32 Function(Int64 networkHandle);
typedef _IsGPUD = int Function(int networkHandle);

typedef _InitWGPUC = Pointer<Utf8> Function(Int64 networkHandle);
typedef _InitWGPUD = Pointer<Utf8> Function(int networkHandle);

typedef _NewVolumetricNetworkC = Int64 Function(
    Int32 depth, Int32 rows, Int32 cols, Int32 layersPerCell);
typedef _NewVolumetricNetworkD = int Function(
    int depth, int rows, int cols, int layersPerCell);

// ── High-level LLM (llm_ext.go) ───────────────────────────────────────────────
typedef _LLMListModelsC = Pointer<Utf8> Function(Pointer<Utf8> hubDir);
typedef _LLMListModelsD = Pointer<Utf8> Function(Pointer<Utf8> hubDir);

typedef _LLMListInstalledModelsC = Pointer<Utf8> Function(Pointer<Utf8> hubDir);
typedef _LLMListInstalledModelsD = Pointer<Utf8> Function(Pointer<Utf8> hubDir);

typedef _CreateLLMC = Int64 Function(
    Pointer<Utf8> snapshotDir, Int32 execMode, Int32 precision, Int32 useGPU, Int32 deterministic);
typedef _CreateLLMD = int Function(
    Pointer<Utf8> snapshotDir, int execMode, int precision, int useGPU, int deterministic);

typedef _LLMGenerateC = Pointer<Utf8> Function(
    Int64 handle, Pointer<Utf8> systemPrompt, Pointer<Utf8> userMsg,
    Float temperature, Int32 topK, Int32 maxTokens);
typedef _LLMGenerateD = Pointer<Utf8> Function(
    int handle, Pointer<Utf8> systemPrompt, Pointer<Utf8> userMsg,
    double temperature, int topK, int maxTokens);

typedef _LLMResetHistoryC = Void Function(Int64 handle);
typedef _LLMResetHistoryD = void Function(int handle);

typedef _LLMFreeC = Void Function(Int64 handle);
typedef _LLMFreeD = void Function(int handle);

typedef _LLMUseGPUC = Int32 Function(Int64 handle);
typedef _LLMUseGPUD = int Function(int handle);

typedef _LLMStartGenerateC = Void Function(
    Int64 handle, Pointer<Utf8> systemPrompt, Pointer<Utf8> userMsg,
    Float temperature, Int32 topK, Int32 maxTokens);
typedef _LLMStartGenerateD = void Function(
    int handle, Pointer<Utf8> systemPrompt, Pointer<Utf8> userMsg,
    double temperature, int topK, int maxTokens);

typedef _LLMPollTokenC = Pointer<Utf8> Function(Int64 handle);
typedef _LLMPollTokenD = Pointer<Utf8> Function(int handle);

typedef _LoadTokenizerC = Int64 Function(Pointer<Utf8> path);
typedef _LoadTokenizerD = int Function(Pointer<Utf8> path);

typedef _TokenizeC = Pointer<Utf8> Function(
    Int64 handle, Pointer<Utf8> text);
typedef _TokenizeD = Pointer<Utf8> Function(int handle, Pointer<Utf8> text);

typedef _DetokenizeC = Pointer<Utf8> Function(
    Int64 handle, Pointer<Uint32> tokens, Int32 count);
typedef _DetokenizeD = Pointer<Utf8> Function(
    int handle, Pointer<Uint32> tokens, int count);

// Training / persistence (poly parity)
typedef _TrainC = Pointer<Utf8> Function(
    Int64 handle,
    Pointer<Float> inputData,
    Int32 inputLen,
    Pointer<Float> targetData,
    Int32 targetLen,
    Int32 batchSize,
    Int32 inDim,
    Int32 outDim,
    Pointer<Utf8> configJSON);
typedef _TrainD = Pointer<Utf8> Function(
    int handle,
    Pointer<Float> inputData,
    int inputLen,
    Pointer<Float> targetData,
    int targetLen,
    int batchSize,
    int inDim,
    int outDim,
    Pointer<Utf8> configJSON);

typedef _SerializeNetworkC = Pointer<Utf8> Function(Int64 handle);
typedef _SerializeNetworkD = Pointer<Utf8> Function(int handle);

typedef _DeserializeNetworkC = Int64 Function(Pointer<Utf8> wire, Int32 len);
typedef _DeserializeNetworkD = int Function(Pointer<Utf8> wire, int len);

typedef _SerializeEntityC = Pointer<Utf8> Function(Int64 handle);
typedef _SerializeEntityD = Pointer<Utf8> Function(int handle);

typedef _DeserializeEntityC = Int64 Function(Pointer<Uint8> wire, Int32 len);
typedef _DeserializeEntityD = int Function(Pointer<Uint8> wire, int len);

typedef _LayerPersistenceC = Pointer<Utf8> Function(
    Pointer<Uint8> wire, Int32 len, Int32 layerIdx);
typedef _LayerPersistenceD = Pointer<Utf8> Function(
    Pointer<Uint8> wire, int len, int layerIdx);

typedef _SyncInferenceWeightsC = Void Function(Int64 handle);
typedef _SyncInferenceWeightsD = void Function(int handle);

typedef _ConfigureTrainingModeC = Pointer<Utf8> Function(Int64 handle, Int32 mode);
typedef _ConfigureTrainingModeD = Pointer<Utf8> Function(int handle, int mode);

typedef _ForwardPolymorphicC = Pointer<Utf8> Function(
    Int64 handle, Pointer<Float> data, Int32 len, Pointer<Utf8> shapeJSON);
typedef _ForwardPolymorphicD = Pointer<Utf8> Function(
    int handle, Pointer<Float> data, int len, Pointer<Utf8> shapeJSON);

typedef _BackwardPolymorphicC = Pointer<Utf8> Function(
    Int64 handle,
    Pointer<Float> inp,
    Int32 inpLen,
    Pointer<Utf8> inShape,
    Pointer<Float> tgt,
    Int32 tgtLen,
    Pointer<Utf8> tgtShape);
typedef _BackwardPolymorphicD = Pointer<Utf8> Function(
    int handle,
    Pointer<Float> inp,
    int inpLen,
    Pointer<Utf8> inShape,
    Pointer<Float> tgt,
    int tgtLen,
    Pointer<Utf8> tgtShape);

// ── LoomLib ────────────────────────────────────────────────────────────────────

/// Singleton wrapper around the loom_cabi shared library.
class LoomLib {
  final DynamicLibrary _lib;

  // ── function pointers ──────────────────────────────────────────────────────
  late final _FreeLoomStringD _freeLoomString;
  late final _LongLongFromStringD _createNetwork;
  late final _VoidLongLongD _freeNetwork;
  late final _StringFromLongLongD _getNetworkInfo;
  late final _NewVolumetricNetworkD _newVolumetricNetwork;
  late final _CreateStepStateD _createStepState;
  late final _VoidLongLongD _freeStepState;
  late final _SetInputD _setInput;
  late final _LoomStepD _loomStep;
  late final _GetOutputD _getOutput;
  late final _SequentialForwardD _sequentialForward;
  late final _MorphLayerD _morphLayer;
  late final _StringFromLongLongD _extractDNA;
  late final _CompareDNAD _compareDNA;
  late final _StringVoidD _getMethodsJSON;
  late final _ExtractBlueprintD _extractNetworkBlueprint;
  late final _GetLayerTelemetryD _getLayerTelemetryIdx;
  late final _DefaultNEATConfigD _defaultNEATConfig;
  late final _SpliceDNAD _spliceDNA;
  late final _StepBackwardD _meshBackward;
  late final _StringVoidD _defaultTweenConfig;
  late final _ApplyGradientsD _applyGradients;
  late final _IsGPUD _isGPU;
  late final _InitWGPUD _initWGPU;
  late final _VoidLongLongD _destroyWGPU;
  // ── LLM ─────────────────────────────────────────────────────────────────────
  late final _LLMListModelsD _llmListModels;
  late final _LLMListInstalledModelsD _llmListInstalledModels;
  late final _CreateLLMD _createLLM;
  late final _LLMGenerateD _llmGenerate;
  late final _LLMResetHistoryD _llmResetHistory;
  late final _LLMFreeD _llmFree;
  _LLMUseGPUD? _llmUseGPU;
  late final _LLMStartGenerateD _llmStartGenerate;
  late final _LLMPollTokenD _llmPollToken;

  late final _LoadTokenizerD _loadTokenizer;
  late final _TokenizeD _tokenize;
  late final _DetokenizeD _detokenize;
  late final _VoidLongLongD _freeTokenizer;
  late final _StringFromLongLongD _getLayerTelemetry;

  late final _TrainD _train;
  late final _SerializeNetworkD _serializeNetwork;
  late final _DeserializeNetworkD _deserializeNetwork;
  _SerializeEntityD? _serializeEntity;
  _DeserializeEntityD? _deserializeEntity;
  _LayerPersistenceD? _layerPersistenceFromEntity;
  late final _SyncInferenceWeightsD _syncInferenceWeights;
  late final _ConfigureTrainingModeD _configureTrainingMode;
  late final _ForwardPolymorphicD _forwardPolymorphic;
  late final _BackwardPolymorphicD _backwardPolymorphic;

  LoomLib._(this._lib) {
    _freeLoomString = _lib
        .lookupFunction<_FreeLoomStringC, _FreeLoomStringD>('FreeLoomString');

    _createNetwork = _lib.lookupFunction<_LongLongFromStringC,
        _LongLongFromStringD>('LoomCreateNetwork');

    _freeNetwork = _lib
        .lookupFunction<_VoidLongLongC, _VoidLongLongD>('LoomFreeNetwork');

    _getNetworkInfo = _lib.lookupFunction<_StringFromLongLongC,
        _StringFromLongLongD>('LoomGetNetworkInfo');

    _newVolumetricNetwork = _lib.lookupFunction<_NewVolumetricNetworkC,
        _NewVolumetricNetworkD>('LoomNewVolumetricNetwork');

    _createStepState = _lib.lookupFunction<_CreateStepStateC,
        _CreateStepStateD>('LoomCreateStepState');

    _freeStepState = _lib.lookupFunction<_VoidLongLongC,
        _VoidLongLongD>('LoomFreeStepState');

    _setInput = _lib
        .lookupFunction<_SetInputC, _SetInputD>('LoomSetInput');

    _loomStep = _lib
        .lookupFunction<_LoomStepC, _LoomStepD>('LoomStep');

    _getOutput = _lib
        .lookupFunction<_GetOutputC, _GetOutputD>('LoomGetOutput');

    _sequentialForward = _lib.lookupFunction<_SequentialForwardC,
        _SequentialForwardD>('LoomSequentialForward');

    _morphLayer = _lib
        .lookupFunction<_MorphLayerC, _MorphLayerD>('LoomMorphLayer');

    _extractDNA = _lib.lookupFunction<_StringFromLongLongC,
        _StringFromLongLongD>('LoomExtractDNA');

    _compareDNA = _lib
        .lookupFunction<_CompareDNAC, _CompareDNAD>('LoomCompareDNA');

    _getMethodsJSON = _lib
        .lookupFunction<_StringVoidC, _StringVoidD>('LoomGetMethodsJSON');

    _extractNetworkBlueprint = _lib.lookupFunction<
        _ExtractBlueprintC, _ExtractBlueprintD>('LoomExtractNetworkBlueprint');

    _getLayerTelemetryIdx = _lib.lookupFunction<
        _GetLayerTelemetryC, _GetLayerTelemetryD>('LoomGetLayerTelemetry');

    _defaultNEATConfig = _lib.lookupFunction<
        _DefaultNEATConfigC, _DefaultNEATConfigD>('LoomDefaultNEATConfig');

    _spliceDNA = _lib
        .lookupFunction<_SpliceDNAC, _SpliceDNAD>('LoomSpliceDNA');

    _meshBackward = _lib.lookupFunction<
        _StepBackwardC, _StepBackwardD>('LoomStepBackward');

    _defaultTweenConfig = _lib
        .lookupFunction<_StringVoidC, _StringVoidD>('LoomDefaultTweenConfig');

    _applyGradients = _lib.lookupFunction<
        _ApplyGradientsC, _ApplyGradientsD>('LoomApplyGradients');

    _isGPU = _lib.lookupFunction<_IsGPUC, _IsGPUD>('LoomIsGPU');

    _initWGPU =
        _lib.lookupFunction<_InitWGPUC, _InitWGPUD>('LoomInitWGPU');

    _destroyWGPU = _lib
        .lookupFunction<_VoidLongLongC, _VoidLongLongD>('LoomDestroyWGPU');

    _llmListModels = _lib
        .lookupFunction<_LLMListModelsC, _LLMListModelsD>('LoomLLMListModels');

    _llmListInstalledModels = _lib.lookupFunction<_LLMListInstalledModelsC,
        _LLMListInstalledModelsD>('LoomLLMListInstalledModels');

    _createLLM = _lib
        .lookupFunction<_CreateLLMC, _CreateLLMD>('LoomCreateLLM');

    _llmGenerate = _lib
        .lookupFunction<_LLMGenerateC, _LLMGenerateD>('LoomLLMGenerate');

    _llmResetHistory = _lib
        .lookupFunction<_LLMResetHistoryC, _LLMResetHistoryD>('LoomLLMResetHistory');

    _llmFree = _lib
        .lookupFunction<_LLMFreeC, _LLMFreeD>('LoomFreeLLM');

    try {
      _llmUseGPU =
          _lib.lookupFunction<_LLMUseGPUC, _LLMUseGPUD>('LoomLLMUseGPU');
    } catch (_) {
      _llmUseGPU = null;
    }

    _llmStartGenerate = _lib.lookupFunction<_LLMStartGenerateC,
        _LLMStartGenerateD>('LoomLLMStartGenerate');

    _llmPollToken = _lib
        .lookupFunction<_LLMPollTokenC, _LLMPollTokenD>('LoomLLMPollToken');

    _loadTokenizer = _lib
        .lookupFunction<_LoadTokenizerC, _LoadTokenizerD>('LoomLoadTokenizer');

    _tokenize = _lib
        .lookupFunction<_TokenizeC, _TokenizeD>('LoomTokenize');

    _detokenize = _lib
        .lookupFunction<_DetokenizeC, _DetokenizeD>('LoomDetokenize');

    _freeTokenizer = _lib
        .lookupFunction<_VoidLongLongC, _VoidLongLongD>('LoomFreeTokenizer');

    _getLayerTelemetry = _lib.lookupFunction<_StringFromLongLongC,
        _StringFromLongLongD>('LoomGetLayerTelemetry');

    _train = _lib.lookupFunction<_TrainC, _TrainD>('LoomTrain');
    _serializeNetwork =
        _lib.lookupFunction<_SerializeNetworkC, _SerializeNetworkD>(
            'LoomSerializeNetwork');
    _deserializeNetwork = _lib.lookupFunction<_DeserializeNetworkC,
        _DeserializeNetworkD>('LoomDeserializeNetwork');
    try {
      _serializeEntity =
          _lib.lookupFunction<_SerializeEntityC, _SerializeEntityD>(
              'LoomSerializeEntity');
      _deserializeEntity = _lib.lookupFunction<_DeserializeEntityC,
          _DeserializeEntityD>('LoomDeserializeEntity');
      _layerPersistenceFromEntity = _lib.lookupFunction<_LayerPersistenceC,
          _LayerPersistenceD>('LoomLayerPersistenceFromEntity');
    } catch (_) {
      _serializeEntity = null;
      _deserializeEntity = null;
      _layerPersistenceFromEntity = null;
    }
    _syncInferenceWeights = _lib.lookupFunction<_SyncInferenceWeightsC,
        _SyncInferenceWeightsD>('LoomSyncInferenceWeights');
    _configureTrainingMode = _lib.lookupFunction<_ConfigureTrainingModeC,
        _ConfigureTrainingModeD>('LoomConfigureTrainingMode');
    _forwardPolymorphic = _lib.lookupFunction<_ForwardPolymorphicC,
        _ForwardPolymorphicD>('LoomForwardPolymorphic');
    _backwardPolymorphic = _lib.lookupFunction<_BackwardPolymorphicC,
        _BackwardPolymorphicD>('LoomBackwardPolymorphic');
  }

  // ── Memory ─────────────────────────────────────────────────────────────────

  /// Free a C string allocated by the loom bridge (call after consuming result).
  void freeString(Pointer<Utf8> ptr) => _freeLoomString(ptr);

  /// Helper: read a C string result, free the pointer, return a Dart String.
  String _consume(Pointer<Utf8> ptr) {
    final s = ptr.toDartString();
    _freeLoomString(ptr);
    return s;
  }

  // ── Network lifecycle ──────────────────────────────────────────────────────

  /// Create a network from a JSON config string. Returns handle or -1 on error.
  int createNetwork(String jsonConfig) {
    final p = jsonConfig.toNativeUtf8();
    final handle = _createNetwork(p);
    calloc.free(p);
    return handle;
  }

  /// Create an empty volumetric network with the given grid dimensions.
  int newVolumetricNetwork(
          {int depth = 1, int rows = 1, int cols = 1, int layersPerCell = 1}) =>
      _newVolumetricNetwork(depth, rows, cols, layersPerCell);

  /// Free a network handle.
  void freeNetwork(int handle) => _freeNetwork(handle);

  /// Returns JSON string with total_layers and grid info.
  String getNetworkInfo(int handle) =>
      _consume(_getNetworkInfo(handle));

  // ── Inference (step mesh) ─────────────────────────────────────────────────

  /// DType constants (mirrors poly.DType in Go)
  static const int dtypeFloat64 = 0;
  static const int dtypeFloat32 = 1;
  static const int dtypeFloat16 = 2;
  static const int dtypeBFloat16 = 3;
  static const int dtypeInt32 = 7;
  static const int dtypeInt8 = 9;

  /// Create step mesh state for a network. Returns handle or -1.
  int createStepState(int networkHandle, {int dtype = dtypeFloat32}) =>
      _createStepState(networkHandle, dtype);

  /// Free a step state handle.
  void freeStepState(int handle) => _freeStepState(handle);

  /// Set the float32 input tensor on step state.
  void setInput(int stateHandle, List<double> data) {
    final ptr = calloc<Float>(data.length);
    for (int i = 0; i < data.length; i++) {
      ptr[i] = data[i];
    }
    _setInput(stateHandle, ptr, data.length);
    calloc.free(ptr);
  }

  /// Run one mesh clock step (`LoomStep`). Returns elapsed nanoseconds.
  int meshStep(int networkHandle, int stateHandle,
          {bool captureHistory = false}) =>
      _loomStep(networkHandle, stateHandle, captureHistory ? 1 : 0);

  /// Get the output of a layer as a JSON string (list of floats).
  String getOutput(int stateHandle, int layerIdx) =>
      _consume(_getOutput(stateHandle, layerIdx));

  /// Run a full sequential forward pass. Returns JSON array of floats.
  String sequentialForward(int networkHandle, List<double> input) {
    final ptr = calloc<Float>(input.length);
    for (int i = 0; i < input.length; i++) {
      ptr[i] = input[i];
    }
    final result = _consume(_sequentialForward(networkHandle, ptr, input.length));
    calloc.free(ptr);
    return result;
  }

  /// Morph a layer to a new dtype. Returns JSON status/error.
  String morphLayer(int networkHandle, int layerIdx, int targetDType) =>
      _consume(_morphLayer(networkHandle, layerIdx, targetDType));

  /// Configure CPU SC (1) or MC (2) training mode. Returns JSON status.
  String configureTrainingMode(int networkHandle, int mode) =>
      _consume(_configureTrainingMode(networkHandle, mode));

  /// Polymorphic forward with explicit input shape. Returns JSON float array.
  String forwardPolymorphic(
    int networkHandle,
    List<double> data,
    List<int> shape,
  ) {
    final ptr = calloc<Float>(data.length);
    for (int i = 0; i < data.length; i++) {
      ptr[i] = data[i];
    }
    final shapeJson = jsonEncode(shape);
    final shapePtr = shapeJson.toNativeUtf8();
    final result = _consume(
      _forwardPolymorphic(networkHandle, ptr, data.length, shapePtr),
    );
    calloc.free(ptr);
    calloc.free(shapePtr);
    return result;
  }

  /// Polymorphic backward. Returns JSON with dx/dw gradients.
  String backwardPolymorphic(
    int networkHandle,
    List<double> input,
    List<int> inputShape,
    List<double> target,
    List<int> targetShape,
  ) {
    final inPtr = calloc<Float>(input.length);
    for (int i = 0; i < input.length; i++) {
      inPtr[i] = input[i];
    }
    final tgtPtr = calloc<Float>(target.length);
    for (int i = 0; i < target.length; i++) {
      tgtPtr[i] = target[i];
    }
    final inShapePtr = jsonEncode(inputShape).toNativeUtf8();
    final tgtShapePtr = jsonEncode(targetShape).toNativeUtf8();
    final result = _consume(
      _backwardPolymorphic(
        networkHandle,
        inPtr,
        input.length,
        inShapePtr,
        tgtPtr,
        target.length,
        tgtShapePtr,
      ),
    );
    calloc.free(inPtr);
    calloc.free(tgtPtr);
    calloc.free(inShapePtr);
    calloc.free(tgtShapePtr);
    return result;
  }

  /// Train via LoomTrain. Returns JSON with loss_history.
  String train(
    int networkHandle,
    List<double> flatInput,
    List<double> flatTarget, {
    required int batchSize,
    required int inDim,
    required int outDim,
    int epochs = 10,
    double learningRate = 0.01,
    String lossType = 'mse',
    bool useGPU = false,
    bool verbose = false,
    int mode = 0,
    List<int>? inputShape,
    List<int>? targetShape,
  }) {
    final inPtr = calloc<Float>(flatInput.length);
    for (int i = 0; i < flatInput.length; i++) {
      inPtr[i] = flatInput[i];
    }
    final tgtPtr = calloc<Float>(flatTarget.length);
    for (int i = 0; i < flatTarget.length; i++) {
      tgtPtr[i] = flatTarget[i];
    }
    final config = <String, dynamic>{
      'Epochs': epochs,
      'LearningRate': learningRate,
      'LossType': lossType,
      'UseGPU': useGPU,
      'Verbose': verbose,
      'GradientClip': 0.0,
    };
    if (mode != 0) config['Mode'] = mode;
    if (inputShape != null) config['InputShape'] = inputShape;
    if (targetShape != null) config['TargetShape'] = targetShape;
    final cfgPtr = jsonEncode(config).toNativeUtf8();
    final result = _consume(
      _train(
        networkHandle,
        inPtr,
        flatInput.length,
        tgtPtr,
        flatTarget.length,
        batchSize,
        inDim,
        outDim,
        cfgPtr,
      ),
    );
    calloc.free(inPtr);
    calloc.free(tgtPtr);
    calloc.free(cfgPtr);
    return result;
  }

  /// Serialize network weights to JSON wire string.
  String serializeNetwork(int networkHandle) =>
      _consume(_serializeNetwork(networkHandle));

  /// Deserialize network from JSON wire. Returns handle or -1.
  int deserializeNetwork(String wire) {
    final p = wire.toNativeUtf8();
    final handle = _deserializeNetwork(p, wire.length);
    calloc.free(p);
    return handle;
  }

  /// Serialize network to native .entity bytes (base64 in JSON from C-ABI).
  Uint8List serializeEntity(int networkHandle) {
    final fn = _serializeEntity;
    if (fn == null) {
      throw StateError('LoomSerializeEntity not available in this welvet build');
    }
    final json = _consume(fn(networkHandle));
    final m = jsonDecode(json) as Map<String, dynamic>;
    if (m.containsKey('error')) {
      throw StateError(m['error'].toString());
    }
    final b64 = m['b64'] as String;
    return base64Decode(b64);
  }

  /// Deserialize from native .entity bytes. Returns handle or -1.
  int deserializeEntity(Uint8List wire) {
    final fn = _deserializeEntity;
    if (fn == null) {
      throw StateError('LoomDeserializeEntity not available in this welvet build');
    }
    final ptr = calloc<Uint8>(wire.length);
    for (int i = 0; i < wire.length; i++) {
      ptr[i] = wire[i];
    }
    final handle = fn(ptr, wire.length);
    calloc.free(ptr);
    return handle;
  }

  /// Layer persistence metadata from entity blob.
  String layerPersistenceFromEntity(Uint8List wire, int layerIdx) {
    final fn = _layerPersistenceFromEntity;
    if (fn == null) {
      throw StateError(
        'LoomLayerPersistenceFromEntity not available in this welvet build',
      );
    }
    final ptr = calloc<Uint8>(wire.length);
    for (int i = 0; i < wire.length; i++) {
      ptr[i] = wire[i];
    }
    final result = _consume(fn(ptr, wire.length, layerIdx));
    calloc.free(ptr);
    return result;
  }

  /// Sync inference weights after training / entity reload.
  void syncInferenceWeights(int networkHandle) =>
      _syncInferenceWeights(networkHandle);

  // ── IO / DNA ───────────────────────────────────────────────────────────────

  /// Extract the DNA (architecture fingerprint) of a network as JSON.
  String extractDNA(int networkHandle) => _consume(_extractDNA(networkHandle));

  /// Compare two DNA JSON strings. Returns JSON diff.
  String compareDNA(String dna1JSON, String dna2JSON) {
    final p1 = dna1JSON.toNativeUtf8();
    final p2 = dna2JSON.toNativeUtf8();
    final result = _consume(_compareDNA(p1, p2));
    calloc.free(p1);
    calloc.free(p2);
    return result;
  }

  /// Get the methods manifest JSON (list of all exported function names).
  String getMethodsJSON() => _consume(_getMethodsJSON());

  /// Extract a network blueprint JSON (ID, grid dimensions).
  String extractNetworkBlueprint(int networkHandle, {String modelId = 'model'}) {
    final p = modelId.toNativeUtf8();
    final result = _consume(_extractNetworkBlueprint(networkHandle, p));
    calloc.free(p);
    return result;
  }

  /// Get telemetry JSON for a specific layer by index.
  String getLayerTelemetryAt(int networkHandle, int layerIdx) =>
      _consume(_getLayerTelemetryIdx(networkHandle, layerIdx));

  /// Get default NEAT config JSON for a given dModel size.
  String defaultNEATConfig({int dModel = 16}) =>
      _consume(_defaultNEATConfig(dModel));

  /// Splice two networks' DNA. Returns handle of child network or -1.
  int spliceDNA(int handleA, int handleB, {String cfgJSON = '{}'}) {
    final p = cfgJSON.toNativeUtf8();
    final result = _spliceDNA(handleA, handleB, p);
    calloc.free(p);
    return result;
  }

  /// Run step-mesh backward pass (`LoomStepBackward`). Returns JSON with grad results.
  String meshBackward(
      int networkHandle, int stateHandle, List<double> gradOutput) {
    final ptr = calloc<Float>(gradOutput.length);
    for (int i = 0; i < gradOutput.length; i++) ptr[i] = gradOutput[i];
    final result =
        _consume(_meshBackward(networkHandle, stateHandle, ptr, gradOutput.length));
    calloc.free(ptr);
    return result;
  }

  /// Get default tween config JSON (neural target propagation).
  String defaultTweenConfig() => _consume(_defaultTweenConfig());

  /// Apply gradients to a layer. gradWeightsJSON should be a JSON array of floats.
  void applyGradients(int networkHandle, int layerIdx, String gradWeightsJSON,
      double learningRate) {
    final p = gradWeightsJSON.toNativeUtf8();
    _applyGradients(networkHandle, layerIdx, p, learningRate);
    calloc.free(p);
  }

  /// Get per-layer telemetry JSON for a network.
  String getLayerTelemetry(int networkHandle) =>
      _consume(_getLayerTelemetry(networkHandle));

  // ── Acceleration (GPU/WGPU) ────────────────────────────────────────────────

  /// Returns 1 if the network is GPU-accelerated, 0 otherwise.
  bool isGPU(int networkHandle) => _isGPU(networkHandle) != 0;

  /// Initialise WGPU for a network. Returns JSON status.
  String initWGPU(int networkHandle) => _consume(_initWGPU(networkHandle));

  /// Destroy WGPU context for a network.
  void destroyWGPU(int networkHandle) => _destroyWGPU(networkHandle);

  // ── High-level LLM ────────────────────────────────────────────────────────

  /// Returns a JSON array of model names found in the HuggingFace hub dir.
  String llmListModels(String hubDir) {
    final p = hubDir.toNativeUtf8();
    final result = _consume(_llmListModels(p));
    calloc.free(p);
    return result;
  }

  /// JSON array of `{id,snapshot_dir}` for models under [hubDir] (Welvet ≥ bundled ABI).
  String llmListInstalledModels(String hubDir) {
    final p = hubDir.toNativeUtf8();
    final result = _consume(_llmListInstalledModels(p));
    calloc.free(p);
    return result;
  }

  /// Load a full LLM from a snapshot directory.
  /// execMode: 1=standard, 2=single-core tiled, 3=multi-core tiled
  /// precision: 4=Q4, 8=INT8, 32=FP32
  /// Returns handle >0 or -1 on error.
  int createLLM(String snapshotDir,
      {int execMode = 3,
      int precision = 4,
      bool useGPU = false,
      bool deterministic = true}) {
    final p = snapshotDir.toNativeUtf8();
    final handle = _createLLM(
        p, execMode, precision, useGPU ? 1 : 0, deterministic ? 1 : 0);
    calloc.free(p);
    return handle;
  }

  /// True when welvet actually initialized GPU for this LLM handle (not just requested).
  /// Null when this libwelvet build has no [LoomLLMUseGPU] export yet.
  bool? llmRuntimeUseGPU(int handle) {
    final fn = _llmUseGPU;
    if (fn == null) return null;
    return fn(handle) != 0;
  }

  /// Generate a reply. Returns JSON with response + stats.
  String llmGenerate(
    int handle,
    String systemPrompt,
    String userMsg, {
    double temperature = 0.7,
    int topK = 40,
    int maxTokens = 128,
  }) {
    final pSys = systemPrompt.toNativeUtf8();
    final pMsg = userMsg.toNativeUtf8();
    final result = _consume(
        _llmGenerate(handle, pSys, pMsg, temperature, topK, maxTokens));
    calloc.free(pSys);
    calloc.free(pMsg);
    return result;
  }

  /// Start generation asynchronously. Poll with [llmPollToken].
  void llmStartGenerate(
    int handle,
    String systemPrompt,
    String userMsg, {
    double temperature = 0.7,
    int topK = 40,
    int maxTokens = 128,
  }) {
    final pSys = systemPrompt.toNativeUtf8();
    final pMsg = userMsg.toNativeUtf8();
    _llmStartGenerate(handle, pSys, pMsg, temperature, topK, maxTokens);
    calloc.free(pSys);
    calloc.free(pMsg);
  }

  /// Poll for next streamed token. Returns JSON:
  ///   {"s":0,"t":"token"} — token chunk
  ///   {"s":1,...stats}    — done
  ///   {"s":2}             — empty, still generating
  String llmPollToken(int handle) => _consume(_llmPollToken(handle));

  /// Clear chat history and reset KV cache for the given LLM handle.
  void llmResetHistory(int handle) => _llmResetHistory(handle);

  /// Free an LLM handle and release all its resources.
  void freeLLM(int handle) => _llmFree(handle);

  // ── Tokenizer ──────────────────────────────────────────────────────────────

  /// Load a tokenizer from a JSON file path. Returns handle or -1.
  int loadTokenizer(String path) {
    final p = path.toNativeUtf8();
    final handle = _loadTokenizer(p);
    calloc.free(p);
    return handle;
  }

  /// Tokenize text. Returns JSON array of token IDs.
  String tokenize(int handle, String text) {
    final p = text.toNativeUtf8();
    final result = _consume(_tokenize(handle, p));
    calloc.free(p);
    return result;
  }

  /// Detokenize a list of token IDs to text. Returns JSON string.
  String detokenize(int handle, List<int> tokens) {
    final ptr = calloc<Uint32>(tokens.length);
    for (int i = 0; i < tokens.length; i++) {
      ptr[i] = tokens[i];
    }
    final result = _consume(_detokenize(handle, ptr, tokens.length));
    calloc.free(ptr);
    return result;
  }

  /// Free a tokenizer handle.
  void freeTokenizer(int handle) => _freeTokenizer(handle);
}

// ── Singleton accessor ─────────────────────────────────────────────────────────

LoomLib? _loomInstance;

/// Last failure from trying to construct [LoomLib] (e.g. missing `FreeLoomString`
/// dlsym on iOS when Welvet is not linked into Runner).
String? _loomLibLastError;

/// Text from the most recent failed FFI init; null after a successful [loomLib] load.
String? get loomLibLastError => _loomLibLastError;

/// The global LoomLib instance. Loaded lazily on first access.
/// Throws if the native library cannot be found.
LoomLib get loomLib {
  if (_loomInstance != null) return _loomInstance!;
  try {
    final lib = LoomLib._(_loadLib());
    _loomInstance = lib;
    _loomLibLastError = null;
    return lib;
  } catch (e) {
    _loomLibLastError = e.toString();
    rethrow;
  }
}

/// Returns true if the loom native library can be loaded.
bool get loomAvailable {
  try {
    loomLib; // triggers load
    _loomLibLastError = null;
    return true;
  } catch (e, st) {
    _loomLibLastError = e.toString();
    // Always print — this is the real reason (e.g. dlsym FreeLoomString not found).
    log('loom_ffi: LoomLib init failed: $e', name: 'welvet');
    log('$st', name: 'welvet');
    return false;
  }
}

// ── Convenience JSON helpers ───────────────────────────────────────────────────

/// Decode a JSON result from loom. Returns null if it contains an "error" key.
Map<String, dynamic>? loomParseResult(String json) {
  try {
    final m = jsonDecode(json) as Map<String, dynamic>;
    if (m.containsKey('error')) return null;
    return m;
  } catch (_) {
    return null;
  }
}

/// Decode a JSON array result from loom (e.g. sequential forward output).
List<double>? loomParseFloatArray(String json) {
  try {
    final list = jsonDecode(json) as List;
    return list.map((e) => (e as num).toDouble()).toList();
  } catch (_) {
    return null;
  }
}

/// Status codes from [LoomLib.llmPollToken] (`s` field).
abstract class LlmPollStatus {
  static const int token = 0;
  static const int done = 1;
  static const int waiting = 2;
}

/// Parsed [LoomLib.llmPollToken] JSON (`{"s":…}`).
class LlmPollEvent {
  LlmPollEvent({
    required this.status,
    this.tokenChunk,
    this.prefillTps = 0,
    this.decodeTps = 0,
    this.totalTps = 0,
    this.ramMB = 0,
    this.vramMB = 0,
  });

  final int status;
  final String? tokenChunk;
  final double prefillTps;
  final double decodeTps;
  final double totalTps;
  final double ramMB;
  final double vramMB;

  static LlmPollEvent parse(String raw) {
    final m = jsonDecode(raw) as Map<String, dynamic>;
    final s = (m['s'] as num?)?.toInt() ?? LlmPollStatus.done;
    return LlmPollEvent(
      status: s,
      tokenChunk: m['t'] as String?,
      prefillTps: (m['prefill_tps'] as num?)?.toDouble() ?? 0,
      decodeTps: (m['decode_tps'] as num?)?.toDouble() ?? 0,
      totalTps: (m['total_tps'] as num?)?.toDouble() ?? 0,
      ramMB: (m['ram_mb'] as num?)?.toDouble() ?? 0,
      vramMB: (m['vram_mb'] as num?)?.toDouble() ?? 0,
    );
  }
}
