# Loom – Dart Bindings

Dart package providing `dart:ffi` bindings for the [Loom](https://github.com/openfluke/loom) neural-network inference engine.

Communicates directly with the native shared library produced by `cabi/` — no HTTP, no code generation, no intermediate JNI layer.

---

## Requirements

| Tool | Version |
|------|---------|
| Dart SDK | ≥ 3.0 |
| libloom | built from `cabi/build_windows.sh` (or equivalent) |

---

## Installation

```yaml
# pubspec.yaml
dependencies:
  loom:
    path: ../dart   # or pub.dev once published
```

```sh
dart pub get
```

---

## Quick Start

```dart
import 'package:loom/loom.dart';

void main() {
  final net = LoomNetwork.create({
    'grid_rows': 2,
    'grid_cols': 2,
    'layers_per_cell': 3,
    'input_size': 4,
  });

  final output = net.forward([1.0, 0.0, 0.5, 0.3]);
  print(output);

  final json = net.save(modelId: 'my_model');
  net.dispose();
}
```

---

## API

### `LoomNetwork`

| Method | Description |
|--------|-------------|
| `LoomNetwork.create(config)` | Build a network from a config map |
| `LoomNetwork.load(json)` | Restore a previously saved model |
| `forward(inputs)` | Run a forward pass → `List<double>` |
| `backward(grads)` | Backward pass with gradient vector |
| `updateWeights(lr)` | SGD weight update |
| `applyAdamW(...)` | AdamW weight update |
| `trainStandard(inputs, targets, cfg)` | Batch training pass |
| `save()` | Serialise to JSON string |
| `evaluate(inputs, expected)` | Compute accuracy metrics |
| `dispose()` | Release native resources |

### Other classes

- **`LoomStepState`** — fine-grained step-by-step execution
- **`LoomTweenState`** — online neural tweening
- **`LoomAdaptationTracker`** — benchmark task-switching
- **`LoomScheduler`** — constant / linear-decay / cosine LR schedulers

---

## Running the example

```sh
# Build libloom first (from repo root):
# bash cabi/build_windows.sh   (WSL / Git Bash)

dart pub get
dart run example/xor_train.dart
```

---

## Native library location

Place `libloom.dll` (Windows), `libloom.so` (Linux), or `libloom.dylib` (macOS) in a directory on your system library path, **or** set the `LD_LIBRARY_PATH` / `PATH` environment variable to point to the build output directory.

---

## Structure

```
dart/
├── pubspec.yaml
├── lib/
│   ├── loom.dart          # public barrel export
│   └── src/
│       ├── bindings.dart  # raw dart:ffi declarations (mirrors cabi/main.go)
│       └── loom_api.dart  # idiomatic Dart API
└── example/
    └── xor_train.dart
```
