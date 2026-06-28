import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:welvet/loom_ffi.dart';

/// Mirrors loom/welvet/python/consumer_smoke.py
void main() {
  test('consumer smoke: load, forward, morph, train, serialize', () {
    expect(loomAvailable, isTrue, reason: loomLibLastError);

    const inShape = [1, 16];
    const outShape = [1, 8];
    final inp = List<double>.generate(16, (i) => 0.2 * math.sin(i * 0.3));
    final tgt = List<double>.generate(8, (i) => 0.5 + 0.3 * math.sin(i * 0.17));

    final netJson = jsonEncode({
      'id': 'consumer-smoke',
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
          'input_height': 16,
          'output_height': 8,
          'activation': 'relu',
          'dtype': 'float32',
        },
      ],
    });

    final handle = loomLib.createNetwork(netJson);
    expect(handle, greaterThan(0));

    loomLib.configureTrainingMode(handle, 2); // CPU MC
    final out0 = loomParseFloatArray(
      loomLib.forwardPolymorphic(handle, inp, inShape),
    );
    expect(out0, isNotNull);
    expect(out0!.length, 8);

    loomLib.morphLayer(handle, 0, LoomLib.dtypeInt8);
    final out1 = loomParseFloatArray(
      loomLib.forwardPolymorphic(handle, inp, inShape),
    );
    expect(out1, isNotNull);

    final trainResult = jsonDecode(
      loomLib.train(
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
      ),
    ) as Map<String, dynamic>;
    final hist = (trainResult['loss_history'] as List?)?.cast<num>();
    expect(hist, isNotEmpty);
    expect(hist!.last.isFinite, isTrue);

    final wire = loomLib.serializeNetwork(handle);
    expect(wire.startsWith('{"error"'), isFalse);

    final reloaded = loomLib.deserializeNetwork(wire);
    expect(reloaded, greaterThan(0));

    final out2 = loomParseFloatArray(
      loomLib.forwardPolymorphic(reloaded, inp, inShape),
    );
    expect(out2, isNotNull);

    final drift = _maxAbsDiff(out1!, out2!);
    loomLib.freeNetwork(reloaded);
    loomLib.freeNetwork(handle);

    expect(drift, lessThan(0.25));
  });
}

double _maxAbsDiff(List<double> a, List<double> b) {
  var m = 0.0;
  for (var i = 0; i < math.min(a.length, b.length); i++) {
    m = math.max(m, (a[i] - b[i]).abs());
  }
  return m;
}
