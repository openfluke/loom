## 0.80.6

- macOS natives moved to `welvet_apple` (with iOS); main `welvet` is Dart-only + federated deps.
- Depends on `welvet_apple` ^0.80.6 for iOS and macOS.

## 0.80.5

- **Federated FFI plugin** — split natives across `welvet_linux`, `welvet_windows`, `welvet_android`, `welvet_apple` so each pub.dev tarball stays under the 100 MB limit (x86_64 + ARM64 per platform).
- Main `welvet` ships macOS natives + all Dart APIs; Linux/Windows/Android/iOS resolve via `default_package` impl deps.
- `loom_ffi_io` loads natives from federated impl package roots via `package_config.json`.
- Publish: `tool/publish_all.sh` (impl packages first, then main).

## 0.80.4

- macOS: SoulGlitch-style loading — `libwelvet.dylib` in app Frameworks + package `native/` for tests; drop vendored-link / post_install hacks.

## 0.80.3

- macOS Flutter apps: vend `Frameworks/welvet.dylib` with `@rpath/welvet.dylib` install name; `force_load` into plugin; add `welvet_post_install.rb` Podfile hook for Runner dyld path.

## 0.80.2

- Fix native library discovery for pub.dev consumers: resolve `package:welvet` root via `package_config.json` instead of cwd-relative paths.

## 0.80.1

- Fix pub.dev tarball: include prebuilt `native/` and `macos/Frameworks/` binaries (0.80.0 omitted them due to `.gitignore`).

## 0.80.0

- Initial pub.dev release: Flutter FFI plugin for Loom Welvet C-ABI.
- Dart bindings (`loomLib` / `LoomLib`) with train, morph, polymorphic forward/backward, JSON + entity persistence.
- Prebuilt natives for Linux, macOS, Windows, Android, iOS (via `tool/copy_native.sh`).
- Lucy **[7]** seven-layer suite: `dart run welvet:seven_layer`.
- Tests mirroring Python `consumer_smoke.py` and `benchmark_seven_layer.py`.
