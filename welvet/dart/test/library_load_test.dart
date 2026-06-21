import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:welvet/loom_ffi.dart';

void main() {
  test('library loads and exports methods manifest', () {
    expect(loomAvailable, isTrue, reason: loomLibLastError);

    final methodsJson = loomLib.getMethodsJSON();
    final decoded = jsonDecode(methodsJson);
    expect(decoded, isNotNull);

    if (decoded is List) {
      expect(decoded.length, greaterThan(10));
      final names = decoded.map((e) => e.toString()).toList();
      expect(
        names.any((n) => n.contains('CreateNetwork') || n.contains('BuildNetwork')),
        isTrue,
      );
    } else if (decoded is Map) {
      expect(decoded.isNotEmpty, isTrue);
    }
  });
}
