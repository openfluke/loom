// Example 5: Extract and compare DNA fingerprints.
//
//   cd examples && dart run 05_dna_compare.dart

import 'dart:convert';
import 'dart:math' as math;

import 'package:welvet/loom_ffi.dart';

void main() {
  if (!loomAvailable) {
    print('Welvet not loaded: $loomLibLastError');
    return;
  }

  String smallNet(String id, int outH) => jsonEncode({
        'id': id,
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
            'input_height': 8,
            'output_height': outH,
            'activation': 'relu',
          },
        ],
      });

  final hA = loomLib.createNetwork(smallNet('dna-a', 4));
  final hB = loomLib.createNetwork(smallNet('dna-b', 8));

  final dnaA = loomLib.extractDNA(hA);
  final dnaB = loomLib.extractDNA(hB);
  final cmpSelf = jsonDecode(loomLib.compareDNA(dnaA, dnaA)) as Map<String, dynamic>;
  final cmpDiff = jsonDecode(loomLib.compareDNA(dnaA, dnaB)) as Map<String, dynamic>;

  loomLib.freeNetwork(hA);
  loomLib.freeNetwork(hB);

  print('05_dna_compare OK — self_sim=${cmpSelf['similarity']} diff_sim=${cmpDiff['similarity']}');
}
