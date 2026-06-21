## 0.80.1

- Fix pub.dev tarball: include prebuilt `native/` and `macos/Frameworks/` binaries (0.80.0 omitted them due to `.gitignore`).

## 0.80.0

- Initial pub.dev release: Flutter FFI plugin for Loom Welvet C-ABI.
- Dart bindings (`loomLib` / `LoomLib`) with train, morph, polymorphic forward/backward, JSON + entity persistence.
- Prebuilt natives for Linux, macOS, Windows, Android, iOS (via `tool/copy_native.sh`).
- Lucy **[7]** seven-layer suite: `dart run welvet:seven_layer`.
- Tests mirroring Python `consumer_smoke.py` and `benchmark_seven_layer.py`.
