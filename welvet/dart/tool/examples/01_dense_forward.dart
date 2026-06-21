// Example 1: Build a dense stack and run forward (sequential + shape-aware).
//
// Run from welvet/dart:
//   cd examples && dart pub get && dart run 01_dense_forward.dart

import 'dart:convert';
import 'dart:math' as math;

import 'package:welvet/loom_ffi.dart';

void main() {
  if (!loomAvailable) {
    print('Welvet not loaded: $loomLibLastError');
    return;
  }

  final netJson = jsonEncode({
    'id': 'example-dense',
    'depth': 1,
    'rows': 1,
    'cols': 1,
    'layers_per_cell': 2,
    'layers': [
      {
        'z': 0,
        'y': 0,
        'x': 0,
        'l': 0,
        'type': 'dense',
        'dtype': 'float32',
        'input_height': 16,
        'output_height': 8,
        'activation': 'relu',
      },
      {
        'z': 0,
        'y': 0,
        'x': 0,
        'l': 1,
        'type': 'dense',
        'dtype': 'float32',
        'input_height': 8,
        'output_height': 4,
        'activation': 'linear',
      },
    ],
  });

  final handle = loomLib.createNetwork(netJson);
  if (handle < 0) throw StateError('createNetwork failed');

  final inp = List<double>.generate(16, (i) => 0.2 * math.sin(i * 0.2));
  const inShape = [1, 16];

  final out = loomParseFloatArray(
    loomLib.forwardPolymorphic(handle, inp, inShape),
  );
  if (out == null || out.length != 4) {
    throw StateError('expected 4 outputs, got ${out?.length}');
  }

  final outSeq = loomParseFloatArray(loomLib.sequentialForward(handle, inp));
  if (outSeq == null || outSeq.length != 4) {
    throw StateError('sequentialForward expected 4 outputs');
  }

  final info = jsonDecode(loomLib.getNetworkInfo(handle)) as Map<String, dynamic>;
  final layers = (info['total_layers'] as num?)?.toInt() ?? 0;

  loomLib.freeNetwork(handle);
  print('01_dense_forward OK — out[0]=${out[0].toStringAsFixed(4)} layers=$layers');
}
