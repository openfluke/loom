# welvet

**M-POLY-VTD AI Engine (Loom v0.80.0)** — Flutter/Dart FFI bindings for the [Loom](https://github.com/openfluke/loom) deterministic neural VM: 21 numerical types, volumetric 3D grids, CPU/GPU training paths, DNA evolution, JSON + native **`.entity`** checkpoints.

[![pub package](https://img.shields.io/pub/v/welvet.svg)](https://pub.dev/packages/welvet)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **Loom core:** [README](https://github.com/openfluke/loom/blob/main/README.md) · **Bedrock validation:** [`docs/bedrock_validation.md`](https://github.com/openfluke/loom/blob/main/docs/bedrock_validation.md)

This is the **pub.dev** package page content (same role as [`@openfluke/welvet` on npm](https://www.npmjs.com/package/@openfluke/welvet) or [`welvet` on PyPI](https://pypi.org/project/welvet/)).

## What this package is

`welvet` ships **prebuilt Welvet C-ABI natives** (`libwelvet.so`, `welvet.dll`, `Welvet.xcframework`, …) inside a **Flutter FFI plugin**. Your Dart/Flutter app calls Loom through `dart:ffi` — same engine as Go `poly/`, Python ctypes, and TypeScript WASM.

| Binding | Best for |
|--------|----------|
| **This package (Dart FFI)** | Flutter apps, Dart CLI on desktop/mobile with bundled natives |
| **[`@openfluke/welvet` npm](https://www.npmjs.com/package/@openfluke/welvet)** | Browser, Node.js — WASM |
| **[`welvet` PyPI](https://pypi.org/project/welvet/)** | Servers, notebooks — ctypes C-ABI |
| **Go `poly/`** | Reference, Lucy harness |

## Features

- **Flutter FFI plugin** — natives bundled per platform (Android, iOS, Linux, macOS, Windows).
- **21 DTypes** — runtime `morphLayer()` per layer index.
- **Volumetric grid** — `depth × rows × cols` cells, Lucy-style JSON networks.
- **Training** — `loomLib.train()` with CPU SC/MC modes (`configureTrainingMode`).
- **Polymorphic forward/backward** — shape-aware tensors (MHA `[batch, seq, d_model]`, CNN, etc.).
- **Persistence** — JSON wire (`serializeNetwork`) and **`.entity`** binary when the native build exports entity APIs.
- **LLM** — `createLLM`, `llmGenerate`, streaming `llmPollToken` (SoulGlitch-style apps).
- **DNA** — `extractDNA`, `compareDNA`, `spliceDNA`, NEAT configs.

## Installation

```yaml
dependencies:
  welvet: ^0.80.0
```

```dart
import 'package:welvet/loom_ffi.dart';
import 'package:welvet/welvet.dart'; // welvetVersion

print(welvetVersion); // 0.80.0
```

**Supported platforms (64-bit):** Linux (x86_64, ARM64), macOS (universal/arm64), Windows (x86_64, ARM64), Android (arm64-v8a, x86_64), iOS (device + simulator via XCFramework).

### Monorepo / path dependency

```yaml
dependencies:
  welvet:
    path: ../loom/welvet/dart
```

### Build natives from source

PyPI/npm ship prebuilt binaries. To refresh from your Loom checkout:

```bash
cd loom/welvet/cabi/internal/build
./build_unix.sh all          # or platform-specific target

cd ../../../dart
bash tool/copy_native.sh --all
flutter test
```

Without `native/` artifacts, `loomAvailable` is false — run `copy_native.sh` first.

## Examples (runnable)

Scripts in [`tool/examples/`](tool/examples/) mirror Python `welvet/python/examples/` and the npm README quick-start.

```bash
cd loom/welvet/dart
bash tool/copy_native.sh

cd tool/examples
dart pub get
dart run 01_dense_forward.dart
dart run run_all.dart          # runs 01–05
```

| Script | What it shows |
|--------|----------------|
| [`01_dense_forward.dart`](tool/examples/01_dense_forward.dart) | Volumetric JSON → `forwardPolymorphic` + `sequentialForward` |
| [`02_morph_and_train.dart`](tool/examples/02_morph_and_train.dart) | `morphLayer(INT8)`, CPU MC `train()` |
| [`03_save_reload.dart`](tool/examples/03_save_reload.dart) | JSON wire + `.entity` roundtrip |
| [`04_mha_forward.dart`](tool/examples/04_mha_forward.dart) | MHA with `[batch, seq, d_model]` |
| [`05_dna_compare.dart`](tool/examples/05_dna_compare.dart) | `extractDNA` + `compareDNA` |

Flutter demo app: [loom-flutter-quickstart](https://github.com/openfluke/loom-flutter-quickstart) on GitHub, or see [docs/flutter.md](../../docs/flutter.md).

## Quick start

Layers live on a 3D grid `(z, y, x, l)` — see [`docs/overview.md`](https://github.com/openfluke/loom/blob/main/docs/overview.md).

### 1. Load library and build a network

```dart
import 'dart:convert';
import 'package:welvet/loom_ffi.dart';

if (!loomAvailable) {
  throw StateError('Welvet missing: $loomLibLastError');
}

final json = jsonEncode({
  'id': 'demo-dense',
  'depth': 1,
  'rows': 1,
  'cols': 1,
  'layers_per_cell': 2,
  'layers': [
    {
      'z': 0, 'y': 0, 'x': 0, 'l': 0,
      'type': 'dense',
      'dtype': 'float32',
      'input_height': 16,
      'output_height': 8,
      'activation': 'relu',
    },
    {
      'z': 0, 'y': 0, 'x': 0, 'l': 1,
      'type': 'dense',
      'dtype': 'float32',
      'input_height': 8,
      'output_height': 4,
      'activation': 'linear',
    },
  ],
});

final handle = loomLib.createNetwork(json);
```

### 2. Forward (shape-aware)

```dart
final input = List<double>.generate(16, (i) => 0.1);
final inShape = [1, 16];

loomLib.configureTrainingMode(handle, 2); // CPU MC
final outJson = loomLib.forwardPolymorphic(handle, input, inShape);
final output = loomParseFloatArray(outJson); // List<double>?
```

### 3. Morph precision

```dart
loomLib.morphLayer(handle, 0, LoomLib.dtypeInt8);
loomLib.syncInferenceWeights(handle);
```

DType IDs: `LoomLib.dtypeFloat32`, `dtypeInt8`, etc. (0–20, see Loom numerical types doc).

### 4. Training

```dart
final target = List<double>.generate(4, (_) => 0.5);
final trainResult = loomLib.train(
  handle,
  input,
  target,
  batchSize: 1,
  inDim: 16,
  outDim: 4,
  epochs: 50,
  learningRate: 0.05,
  mode: 2, // CPU MC
  inputShape: [1, 16],
  targetShape: [1, 4],
);
final hist = (jsonDecode(trainResult)['loss_history'] as List).cast<num>();
```

### 5. Save / reload

**JSON wire** (always available):

```dart
final wire = loomLib.serializeNetwork(handle);
final reloaded = loomLib.deserializeNetwork(wire);
loomLib.syncInferenceWeights(reloaded);
// forward on reloaded, then loomLib.freeNetwork(reloaded)
```

**`.entity` wire** (when native exports `LoomSerializeEntity`):

```dart
final bytes = loomLib.serializeEntity(handle);
final h2 = loomLib.deserializeEntity(bytes);
loomLib.syncInferenceWeights(h2);
```

### 6. LLM (local inference)

```dart
final llm = loomLib.createLLM(
  '/path/to/snapshot',
  execMode: 3,
  precision: 4,
  useGPU: true,
);
final reply = loomLib.llmGenerate(llm, 'You are helpful.', 'Hello!');
loomLib.freeLLM(llm);
```

## Seven-layer validation (Dart → C-ABI → Loom)

Same Lucy menu **[7]** logic as `loom/lucy`, Python `benchmark_seven_layer.py`, and npm `test:seven-layer`.

```bash
cd loom/welvet/dart
bash tool/copy_native.sh

# Full suite — 10 layer types × 21 dtypes × grids (slow)
dart run welvet:seven_layer

# One layer
dart run welvet:seven_layer -- --layer Dense
dart run welvet:seven_layer -- --layer MHA

# Single dtype row (fast)
dart run welvet:seven_layer -- --layer Dense --dtype Float32
```

Layer names: `Dense`, `SwiGLU`, `MHA`, `CNN1`, `CNN2`, `CNN3`, `RNN`, `LSTM`, `Embedding`, `Residual`.

Each row checks: SC/MC forward & backward parity, train, JSON save/reload, `.entity` roundtrip (when exported).

**Automated tests** (CI-friendly subset):

```bash
flutter test
```

| Test | Mirrors |
|------|---------|
| `test/consumer_smoke_test.dart` | Python `consumer_smoke.py` |
| `test/seven_layer_suite_test.dart` | Dense · Float32 · all grids |
| `test/library_load_test.dart` | C-ABI load + methods manifest |

Cross-check with Python:

```bash
cd loom/welvet/python && python benchmark_seven_layer.py --layer Dense
```

## API surface (`loomLib` / `LoomLib`)

Import: `package:welvet/loom_ffi.dart`

| Category | Methods |
|----------|---------|
| **Lifecycle** | `createNetwork`, `freeNetwork`, `getNetworkInfo`, `newVolumetricNetwork` |
| **Inference** | `sequentialForward`, `forwardPolymorphic`, `backwardPolymorphic`, `configureTrainingMode` |
| **Step mesh** | `createStepState`, `setInput`, `meshStep`, `getOutput`, `meshBackward` |
| **Training** | `train`, `applyGradients` |
| **Morph** | `morphLayer`, `syncInferenceWeights` |
| **Persistence** | `serializeNetwork`, `deserializeNetwork`, `serializeEntity`, `deserializeEntity` |
| **DNA** | `extractDNA`, `compareDNA`, `spliceDNA`, `defaultNEATConfig` |
| **GPU** | `initWGPU`, `destroyWGPU`, `isGPU` |
| **LLM** | `createLLM`, `llmGenerate`, `llmStartGenerate`, `llmPollToken`, `freeLLM` |
| **Tokenizer** | `loadTokenizer`, `tokenize`, `detokenize` |

Helpers: `loomAvailable`, `loomLibLastError`, `loomParseFloatArray`, `loomParseResult`, `LlmPollEvent`.

## Publishing to pub.dev

Not a manual upload — use the CLI (like `npm publish` / `twine upload`):

```bash
flutter pub login
cd loom/welvet/dart
bash tool/publish.sh                         # dry-run (desktop: mac + linux + windows)
bash tool/publish.sh --publish               # upload desktop slice (~59 MB)
```

**Binaries stay out of git.** `publish.sh` runs `copy_native.sh` locally, copies the package to a temp dir outside the repo, strips iOS/Android natives, and uploads from there.

**Size:** pub.dev rejects uploads over **100 MB** compressed. Default **desktop** slice (macOS + Linux + Windows) fits; iOS/Android are omitted (use `copy_native.sh` in the monorepo for mobile).

## Version alignment

| Component | Version |
|-----------|---------|
| **Loom engine (poly)** | **0.80.0** |
| **pub `welvet`** | **0.80.1** |
| **npm `@openfluke/welvet`** | **0.80.0** |
| **PyPI `welvet`** | **0.80.0** |

## License

Apache-2.0 — see [LICENSE](LICENSE).
