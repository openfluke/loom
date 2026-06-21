/// Welvet — Loom M-POLY-VTD engine bindings for Flutter/Dart.
///
/// FFI to prebuilt `libwelvet` natives (macOS, Windows, Linux x86/ARM,
/// iOS, Android). See [loomLib] and [LoomLib].
library;

export 'loom_ffi.dart';
export 'src/seven_layer_spec.dart';
export 'src/seven_layer_runner.dart';

/// Package version (matches loom/welvet/python PyPI release).
const String welvetVersion = '0.80.0';
