// Example 3: JSON wire + .entity checkpoint roundtrip.
//
//   cd examples && dart run 03_save_reload.dart

import 'dart:convert';
import 'dart:math' as math;

import 'package:welvet/loom_ffi.dart';
import 'package:welvet/src/seven_layer_spec.dart';

void main() {
  if (!loomAvailable) {
    print('Welvet not loaded: $loomLibLastError');
    return;
  }

  const grid = (1, 1, 1);
  final netJson = buildDenseJson(grid, 'FLOAT32');
  final (inp, inShape) = sinInput(4, [denseEndpoints(grid)[0]]);

  final handle = loomLib.createNetwork(netJson);
  loomLib.configureTrainingMode(handle, trainingModeCpuMc);
  final out0 = loomParseFloatArray(
    loomLib.forwardPolymorphic(handle, inp, inShape),
  );
  if (out0 == null) throw StateError('forward failed');

  // JSON wire
  final wire = loomLib.serializeNetwork(handle);
  final jsonHandle = loomLib.deserializeNetwork(wire);
  loomLib.syncInferenceWeights(jsonHandle);
  final outJson = loomParseFloatArray(
    loomLib.forwardPolymorphic(jsonHandle, inp, inShape),
  );
  final jsonDrift = maxAbsDiff(out0, outJson ?? []);
  loomLib.freeNetwork(jsonHandle);

  // .entity wire (skipped if binary lacks LoomSerializeEntity)
  var entityDrift = -1.0;
  try {
    final entityWire = loomLib.serializeEntity(handle);
    final entityHandle = loomLib.deserializeEntity(entityWire);
    loomLib.syncInferenceWeights(entityHandle);
    final outEntity = loomParseFloatArray(
      loomLib.forwardPolymorphic(entityHandle, inp, inShape),
    );
    entityDrift = maxAbsDiff(out0, outEntity ?? []);
    loomLib.freeNetwork(entityHandle);
  } on StateError catch (e) {
    if (!e.message.contains('not available')) throw e;
    entityDrift = -1.0;
  }

  loomLib.freeNetwork(handle);
  print(
    '03_save_reload OK — json_drift=$jsonDrift entity_drift=${entityDrift < 0 ? "skipped" : entityDrift}',
  );
}
