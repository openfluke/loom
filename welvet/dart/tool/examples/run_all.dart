// Run all README examples (mirrors python/examples/run_all.py).
//
//   cd examples && dart run run_all.dart

import 'dart:io';

final scripts = [
  '01_dense_forward.dart',
  '02_morph_and_train.dart',
  '03_save_reload.dart',
  '04_mha_forward.dart',
  '05_dna_compare.dart',
];

Future<void> main() async {
  final dir = Directory.current.path;
  var failed = 0;
  for (final script in scripts) {
    print('── $script');
    final result = await Process.run('dart', ['run', script], workingDirectory: dir);
    stdout.write(result.stdout);
    stderr.write(result.stderr);
    if (result.exitCode != 0) {
      print('FAILED: $script (exit ${result.exitCode})');
      failed++;
    }
  }
  if (failed > 0) {
    print('\n❌ $failed example(s) failed');
    exit(1);
  }
  print('\n✅ all examples passed');
}
