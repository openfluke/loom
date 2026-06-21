// Example 4: MHA forward with [batch, seq, d_model].
//
//   cd examples && dart run 04_mha_forward.dart

import 'dart:convert';
import 'dart:math' as math;

import 'package:welvet/loom_ffi.dart';

void main() {
  if (!loomAvailable) {
    print('Welvet not loaded: $loomLibLastError');
    return;
  }

  const batch = 4;
  const seq = 8;
  const dModel = 64;
  const heads = 4;

  final netJson = jsonEncode({
    'id': 'example-mha',
    'depth': 1,
    'rows': 1,
    'cols': 1,
    'layers_per_cell': 1,
    'layers': [
      {
        'z': 0,
        'y': 0,
        'x': 0,
        'l': 0,
        'type': 'MHA',
        'dtype': 'FLOAT32',
        'd_model': dModel,
        'num_heads': heads,
        'seq_length': seq,
        'activation': 'RELU',
      },
    ],
  });

  final handle = loomLib.createNetwork(netJson);
  final inp = List<double>.generate(
    batch * seq * dModel,
    (i) => 0.2 * math.sin(i * 0.11),
  );
  final inShape = [batch, seq, dModel];

  loomLib.configureTrainingMode(handle, 2);
  final out = loomParseFloatArray(
    loomLib.forwardPolymorphic(handle, inp, inShape),
  );
  if (out == null || out.isEmpty) throw StateError('MHA forward failed');

  loomLib.freeNetwork(handle);
  print('04_mha_forward OK — out_len=${out.length} out[0]=${out[0].toStringAsFixed(4)}');
}
