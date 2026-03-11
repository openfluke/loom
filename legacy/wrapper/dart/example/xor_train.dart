/// XOR training example using the Loom Dart SDK.
///
/// Run with:
///   dart run example/xor_train.dart
///
/// Requires libloom.dll / libloom.so / libloom.dylib on the library path.
import 'package:loom/loom.dart';

void main() {
  print('🔷 Loom Dart – XOR training example\n');

  // Build a small network: 2-input, 2×2 grid, 3 layers/cell
  final net = LoomNetwork.create({
    'grid_rows': 2,
    'grid_cols': 2,
    'layers_per_cell': 3,
    'input_size': 2,
    'layers': [
      for (var i = 0; i < 11; i++)
        {'type': 'dense', 'activation': 'relu', 'input_size': 2, 'output_size': 2},
      {'type': 'dense', 'activation': 'sigmoid', 'input_size': 2, 'output_size': 1},
    ],
  });

  // XOR dataset
  final inputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
  ];
  final targets = [
    [0.0],
    [1.0],
    [1.0],
    [0.0],
  ];

  const epochs = 1000;
  const lr = 0.01;

  for (var epoch = 0; epoch < epochs; epoch++) {
    for (var i = 0; i < inputs.length; i++) {
      net.forward(inputs[i]);
      final grad = [targets[i][0] - 0.5]; // simple gradient signal
      net.backward(grad);
      net.updateWeights(lr);
    }
    if ((epoch + 1) % 200 == 0) {
      print('Epoch ${epoch + 1}/$epochs');
    }
  }

  // Test
  print('\nResults after $epochs epochs:');
  for (var i = 0; i < inputs.length; i++) {
    final out = net.forward(inputs[i]);
    final predicted = out.isNotEmpty ? out[0].toStringAsFixed(4) : '?';
    print('  ${inputs[i]} → $predicted (expected ${targets[i][0]})');
  }

  // Save / load round-trip
  final json = net.save(modelId: 'xor');
  print('\nModel saved (${json.length} bytes)');

  net.dispose();
  print('\n✅ Done.');
}
