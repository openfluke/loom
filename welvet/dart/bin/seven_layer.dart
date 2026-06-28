// CLI entry — lucy [7] seven-layer suite via Welvet C-ABI.
//
// Usage:
//   dart run welvet:seven_layer
//   dart run welvet:seven_layer -- --layer Dense
//   dart run welvet:seven_layer -- --layer Dense --dtype Float32

import 'package:welvet/src/seven_layer_runner.dart';

void main(List<String> args) {
  String? layer;
  String? dtype;
  for (var i = 0; i < args.length; i++) {
    if (args[i] == '--layer' && i + 1 < args.length) {
      layer = args[++i];
    } else if (args[i] == '--dtype' && i + 1 < args.length) {
      dtype = args[++i];
    }
  }

  print('=== welvet seven-layer suite — Dart FFI → Loom CABI ===\n');

  final result = runSevenLayerSuite(
    layerFilter: layer,
    dtypeFilter: dtype,
    onLog: print,
  );

  print('');
  if (result.ok) {
    print('✅ ALL PASSED (${result.passed} rows)');
  } else {
    print('❌ FAILURES: ${result.failed} failed, ${result.passed} passed');
  }

  if (!result.ok) {
    throw StateError('seven-layer suite failed');
  }
}
