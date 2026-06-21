// Seven-layer CPU suite via Welvet C-ABI — mirrors benchmark_seven_layer.py / lucy [7].

import 'dart:convert';
import 'dart:typed_data';

import 'package:welvet/loom_ffi.dart';
import 'package:welvet/src/seven_layer_spec.dart';

/// Result summary from a full or partial seven-layer run.
class SevenLayerRunResult {
  SevenLayerRunResult({required this.passed, required this.failed});

  final int passed;
  final int failed;

  bool get ok => failed == 0;
}

/// Run lucy-style seven-layer matrix over C-ABI (forward/backward parity, train, save/reload).
///
/// [layerFilter] — e.g. `Dense` to run one suite (like `--layer Dense` in Python).
/// [dtypeFilter] — e.g. `Float32` to run one dtype row.
/// [entityTests] — JSON + .entity roundtrip (skipped when LoomSerializeEntity missing).
SevenLayerRunResult runSevenLayerSuite({
  String? layerFilter,
  String? dtypeFilter,
  bool entityTests = true,
  void Function(String line)? onLog,
}) {
  void log(String line) {
    onLog?.call(line);
  }

  void logPart(String part) {
    onLog?.call(part);
  }

  if (!loomAvailable) {
    log('ERROR: Welvet not loaded: $loomLibLastError');
    return SevenLayerRunResult(passed: 0, failed: 1);
  }

  var passed = 0;
  var failed = 0;
  var entitySupported = entityTests;

  for (final suite in layerSuites) {
    if (layerFilter != null && suite.name != layerFilter) continue;

    for (final g in suite.grids) {
      final epochs = trainEpochsForGrid(g);
      final iters = benchItersForGrid(g);
      final label = '${suite.name} ${g.$1}×${g.$2}×${g.$3}';
      log('');
      log('${'═' * 70}');
      log('  $label — Dart → Loom CABI (CPU SC/MC · train · JSON + .entity save/reload)');
      log('${'═' * 70}');

      for (final row in allDtypes) {
        final (name, jsonName, dtype, tol) = row;
        if (dtypeFilter != null && name != dtypeFilter) continue;

        logPart('  · ${name.padRight(10)} ');
        try {
          final handle = _createAndMorph(suite.build(g, jsonName), dtype);
          final (inp, inShape) = suite.makeInput(g);
          final batch = suite.isEmbedding ? 1 : 4;

          final outSc = _benchForward(handle, inp, inShape, trainingModeCpuSc, iters);
          final outMc = _benchForward(handle, inp, inShape, trainingModeCpuMc, iters);
          final fwdScmc = maxAbsDiff(outSc, outMc);

          final tgt = sinTarget(outSc);
          final tgtShape = targetShape(
            inShape,
            tgt.length,
            isEmbedding: suite.isEmbedding,
          );

          final (dxSc, dwSc) = _benchBackward(
            handle,
            inp,
            inShape,
            tgt,
            tgtShape,
            trainingModeCpuSc,
          );
          final (dxMc, dwMc) = _benchBackward(
            handle,
            inp,
            inShape,
            tgt,
            tgtShape,
            trainingModeCpuMc,
          );
          final bwdScmc = maxAbsDiff(dxSc + dwSc, dxMc + dwMc);

          final detTol = tol > 1e-10 ? tol : 1e-10;
          final detOk = fwdScmc <= detTol && bwdScmc <= detTol * 10;

          final jsonBeforeOk = _checkSaveReloadJson(handle, inp, inShape, tol, false);
          var entityBeforeOk = true;
          var entityAfterOk = true;
          if (entitySupported) {
            try {
              entityBeforeOk = _checkSaveReloadEntity(handle, inp, inShape, tol, false);
            } on StateError catch (e) {
              if (e.message.contains('not available')) {
                entitySupported = false;
                entityBeforeOk = true;
                entityAfterOk = true;
              } else {
                throw e;
              }
            }
          }

          final lr = trainingLr(dtype);
          final histSc = _trainFresh(
            suite.build(g, jsonName),
            dtype,
            inp,
            tgt,
            inShape,
            tgtShape,
            batch,
            epochs,
            trainingModeCpuSc,
            lr,
          );

          final mcHandle = _createAndMorph(suite.build(g, jsonName), dtype);
          final histMc = _trainOnHandle(
            mcHandle,
            inp,
            tgt,
            inShape,
            tgtShape,
            batch,
            epochs,
            trainingModeCpuMc,
            lr,
          );

          final lossInit = histMc.isNotEmpty ? histMc.first : 0.0;
          final lossFinal = histMc.isNotEmpty ? histMc.last : 0.0;
          final requiresLearn =
              suite.primary != 'RESIDUAL' && !suite.noLearn;
          final learned =
              trainingOk(lossInit, lossFinal, dtype) || !requiresLearn;

          final jsonAfterOk = _checkSaveReloadJson(mcHandle, inp, inShape, tol, true);
          if (entitySupported) {
            try {
              entityAfterOk =
                  _checkSaveReloadEntity(mcHandle, inp, inShape, tol, true);
            } on StateError catch (e) {
              if (e.message.contains('not available')) {
                entitySupported = false;
                entityAfterOk = true;
              } else {
                throw e;
              }
            }
          }

          final overall = jsonBeforeOk &&
              entityBeforeOk &&
              jsonAfterOk &&
              entityAfterOk &&
              learned &&
              detOk;

          loomLib.freeNetwork(handle);
          loomLib.freeNetwork(mcHandle);
          // histSc network already freed inside _trainFresh

          if (overall) {
            passed++;
            log(
              'PASS  loss ${lossInit.toStringAsExponential(4)}→${lossFinal.toStringAsExponential(4)} '
              'det=$detOk json=$jsonAfterOk entity=$entityAfterOk',
            );
          } else {
            failed++;
            log(
              'FAIL  loss ${lossInit.toStringAsExponential(4)}→${lossFinal.toStringAsExponential(4)} '
              'learn=$learned json=${jsonBeforeOk && jsonAfterOk} '
              'entity=${entityBeforeOk && entityAfterOk} det=$detOk sc_train=${histSc.length}',
            );
          }
        } catch (e) {
          failed++;
          log('ERR   $e');
        }
      }
    }
  }

  return SevenLayerRunResult(passed: passed, failed: failed);
}

int _createAndMorph(String json, int dtype) {
  final h = loomLib.createNetwork(json);
  if (h < 0) throw StateError('createNetwork failed');
  _morphAll(h, dtype);
  loomLib.syncInferenceWeights(h);
  return h;
}

void _morphAll(int handle, int dtype) {
  final info = jsonDecode(loomLib.getNetworkInfo(handle)) as Map<String, dynamic>;
  final n = (info['total_layers'] as num?)?.toInt() ?? 0;
  for (var i = 0; i < n; i++) {
    loomLib.morphLayer(handle, i, dtype);
  }
}

List<double> _benchForward(
  int handle,
  List<double> inp,
  List<int> shape,
  int mode,
  int iters,
) {
  loomLib.configureTrainingMode(handle, mode);
  for (var i = 0; i < 3; i++) {
    _forward(handle, inp, shape);
  }
  List<double>? last;
  for (var i = 0; i < iters; i++) {
    last = _forward(handle, inp, shape);
  }
  return last ?? [];
}

List<double> _forward(int handle, List<double> inp, List<int> shape) {
  final out = loomParseFloatArray(loomLib.forwardPolymorphic(handle, inp, shape));
  if (out == null) throw StateError('forward_polymorphic failed');
  return out;
}

(List<double>, List<double>) _benchBackward(
  int handle,
  List<double> inp,
  List<int> inShape,
  List<double> tgt,
  List<int> tgtShape,
  int mode,
) {
  loomLib.configureTrainingMode(handle, mode);
  final raw = loomLib.backwardPolymorphic(handle, inp, inShape, tgt, tgtShape);
  final m = jsonDecode(raw) as Map<String, dynamic>;
  if (m.containsKey('error')) throw StateError(m['error'].toString());
  final dx = _gradList(m['dx']);
  final dw = _gradList(m['dw']);
  return (dx, dw);
}

List<double> _gradList(dynamic v) {
  if (v == null) return [];
  if (v is List) return v.map((e) => (e as num).toDouble()).toList();
  return [];
}

List<double> _trainFresh(
  String json,
  int dtype,
  List<double> inp,
  List<double> tgt,
  List<int> inShape,
  List<int> tgtShape,
  int batch,
  int epochs,
  int mode,
  double lr,
) {
  final h = _createAndMorph(json, dtype);
  final hist = _trainOnHandle(
    h,
    inp,
    tgt,
    inShape,
    tgtShape,
    batch,
    epochs,
    mode,
    lr,
  );
  loomLib.freeNetwork(h);
  return hist;
}

List<double> _trainOnHandle(
  int handle,
  List<double> inp,
  List<double> tgt,
  List<int> inShape,
  List<int> tgtShape,
  int batch,
  int epochs,
  int mode,
  double lr,
) {
  final raw = loomLib.train(
    handle,
    inp,
    tgt,
    batchSize: batch,
    inDim: inp.length ~/ batch,
    outDim: tgt.length ~/ batch,
    epochs: epochs,
    learningRate: lr,
    mode: mode,
    inputShape: inShape,
    targetShape: tgtShape,
  );
  final m = jsonDecode(raw) as Map<String, dynamic>;
  if (m.containsKey('error')) throw StateError(m['error'].toString());
  final hist = m['loss_history'] as List?;
  return hist?.map((e) => (e as num).toDouble()).toList() ?? [];
}

bool _checkSaveReloadJson(
  int handle,
  List<double> inp,
  List<int> shape,
  double tol,
  bool after,
) {
  final out0 = _forward(handle, inp, shape);
  final wire = loomLib.serializeNetwork(handle);
  final reloaded = loomLib.deserializeNetwork(wire);
  if (reloaded < 0) return false;
  loomLib.syncInferenceWeights(reloaded);
  final out1 = _forward(reloaded, inp, shape);
  loomLib.freeNetwork(reloaded);
  return maxAbsDiff(out0, out1) <= tol * (after ? 100 : 1);
}

bool _checkSaveReloadEntity(
  int handle,
  List<double> inp,
  List<int> shape,
  double tol,
  bool after,
) {
  final out0 = _forward(handle, inp, shape);
  final wire = loomLib.serializeEntity(handle);
  final reloaded = loomLib.deserializeEntity(wire);
  if (reloaded < 0) return false;
  loomLib.syncInferenceWeights(reloaded);
  final out1 = _forward(reloaded, inp, shape);
  final fwdOk = maxAbsDiff(out0, out1) <= tol * (after ? 100 : 1);
  final nativeOk = _entityNativeOk(handle, wire);
  loomLib.freeNetwork(reloaded);
  return fwdOk && nativeOk;
}

bool _entityNativeOk(int handle, Uint8List wire) {
  final info = jsonDecode(loomLib.getNetworkInfo(handle)) as Map<String, dynamic>;
  final n = (info['total_layers'] as num?)?.toInt() ?? 0;
  for (var i = 0; i < n; i++) {
    try {
      final raw = loomLib.layerPersistenceFromEntity(wire, i);
      final r = jsonDecode(raw) as Map<String, dynamic>;
      if (r.containsKey('error')) {
        final msg = r['error'].toString();
        if (msg.contains('no weight blob')) continue;
        return false;
      }
      if (r['native'] != true || r['weights'] == null) return false;
    } on StateError catch (e) {
      if (e.message.contains('no weight blob')) continue;
      throw e;
    }
  }
  return true;
}
