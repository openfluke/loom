// Example 2: Morph to INT8 and train on CPU MC.
//
//   cd examples && dart run 02_morph_and_train.dart

import 'dart:convert';
import 'dart:math' as math;

import 'package:welvet/loom_ffi.dart';

void main() {
  if (!loomAvailable) {
    print('Welvet not loaded: $loomLibLastError');
    return;
  }

  final netJson = jsonEncode({
    'id': 'example-morph',
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
        'type': 'dense',
        'dtype': 'float32',
        'input_height': 16,
        'output_height': 8,
        'activation': 'relu',
      },
    ],
  });

  final handle = loomLib.createNetwork(netJson);
  loomLib.morphLayer(handle, 0, LoomLib.dtypeInt8);

  final inp = List<double>.generate(16, (i) => 0.2 * math.sin(i * 0.3));
  final tgt = List<double>.generate(8, (i) => 0.5 + 0.3 * math.sin(i * 0.17));
  const inShape = [1, 16];
  const outShape = [1, 8];

  loomLib.configureTrainingMode(handle, 2);
  final out = loomParseFloatArray(
    loomLib.forwardPolymorphic(handle, inp, inShape),
  );

  final trainRaw = loomLib.train(
    handle,
    inp,
    tgt,
    batchSize: 1,
    inDim: 16,
    outDim: 8,
    epochs: 5,
    learningRate: 0.05,
    mode: 2,
    inputShape: inShape,
    targetShape: outShape,
  );
  final hist = (jsonDecode(trainRaw)['loss_history'] as List?)?.cast<num>();
  if (hist == null || hist.isEmpty) throw StateError('empty loss history');

  loomLib.freeNetwork(handle);
  print(
    '02_morph_and_train OK — out_len=${out?.length} final_loss=${hist.last}',
  );
}
