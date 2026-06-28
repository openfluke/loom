import 'package:flutter_test/flutter_test.dart';
import 'package:welvet/src/seven_layer_runner.dart';

/// Quick CI slice: Dense · 1³ · Float32 (full suite via `dart run welvet:seven_layer`).
void main() {
  test('seven-layer Dense 1x1x1 Float32 (lucy [7] slice)', () {
    final lines = <String>[];
    final result = runSevenLayerSuite(
      layerFilter: 'Dense',
      dtypeFilter: 'Float32',
      onLog: lines.add,
    );
    expect(result.ok, isTrue, reason: lines.join('\n'));
    expect(result.passed, greaterThan(0));
  });
}
