# Flutter & Dart: `welvet` on pub.dev

Loom runs in **Flutter** and plain **Dart** through the [`welvet`](https://pub.dev/packages/welvet) FFI plugin. Same Welvet C-ABI as Python ctypes and TypeScript WASM — `createNetwork`, polymorphic forward/backward, CPU training, DNA, JSON + `.entity` checkpoints, mesh step, and LLM exports when your native build includes them.

> **v0.80.6** — Federated natives on pub.dev: main `welvet` (Dart API) plus `welvet_linux`, `welvet_windows`, `welvet_android`, `welvet_apple` (iOS + macOS) for per-OS binaries under the 100 MB limit.

---

## Installation

Add to `pubspec.yaml` — **no monorepo path required**:

```yaml
dependencies:
  welvet: ^0.80.4
```

```bash
flutter pub get
```

```dart
import 'package:welvet/loom_ffi.dart';
import 'package:welvet/welvet.dart'; // welvetVersion, seven_layer_runner

print(welvetVersion); // 0.80.4
print(loomAvailable); // true when libwelvet loaded
```

If `loomAvailable` is false, read `loomLibLastError` — usually a missing native library on that platform.

---

## Quick start: dense forward

Networks are JSON volumetric grids (`depth × rows × cols`, layers at `(z, y, x, l)`). See [overview.md](overview.md).

```dart
import 'dart:convert';
import 'dart:math' as math;
import 'package:welvet/loom_ffi.dart';

if (!loomAvailable) throw StateError(loomLibLastError ?? 'welvet not loaded');

final netJson = jsonEncode({
  'id': 'flutter-demo',
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

final handle = loomLib.createNetwork(netJson);
final inp = List<double>.generate(16, (i) => 0.2 * math.sin(i * 0.2));
const inShape = [1, 16];

loomLib.configureTrainingMode(handle, 2); // CPU MC
final out = loomParseFloatArray(
  loomLib.forwardPolymorphic(handle, inp, inShape),
);
// out.length == 4

loomLib.freeNetwork(handle);
```

---

## Runnable examples (Dart CLI)

`welvet/dart/tool/examples/` mirrors Python and npm quick-starts:

| Script | API surface |
|--------|-------------|
| `01_dense_forward.dart` | `forwardPolymorphic` + `sequentialForward` |
| `02_morph_and_train.dart` | `morphLayer(INT8)`, `train()` on CPU MC |
| `03_save_reload.dart` | `serializeNetwork` / `deserializeNetwork`, `.entity` roundtrip |
| `04_mha_forward.dart` | MHA with `[batch, seq, d_model]` |
| `05_dna_compare.dart` | `extractDNA`, `compareDNA` |

```bash
cd welvet/dart
bash tool/copy_native.sh
cd tool/examples && dart pub get && dart run run_all.dart
```

---

## Interactive Flutter showcase app

**[loom-flutter-quickstart](https://github.com/openfluke/loom-flutter-quickstart)** — Apache-2.0 demo app with **30+ CPU demos** (no `path:` dependency on the Loom repo):

```bash
git clone https://github.com/openfluke/loom-flutter-quickstart.git
cd loom-flutter-quickstart
flutter pub get
flutter run -d macos   # or linux / windows
```

Categories in the app:

- System — version, `loomAvailable`, C-ABI methods manifest
- Forward — dense stack, MHA, volumetric grid
- Training (CPU) — morph dtypes, `LoomTrain`, backward polymorphic, SC vs MC
- Mesh — `LoomStep`, mesh backward, `applyGradients`
- Persistence — JSON wire + `.entity` checkpoint
- DNA & meta — compare/splice DNA, blueprint, NEAT/tween defaults, telemetry
- Layer quick forward — Dense, MHA, CNN×3, RNN, LSTM, SwiGLU, Embedding, Residual
- Seven-layer suite — full CPU parity suite per layer type (Float32)

GPU training and bundled on-device LLM inference are intentionally **not** exercised in the showcase (GPU train is still flaky; LLM needs a local HuggingFace snapshot).

---

## Common patterns

### Morph precision

```dart
loomLib.morphLayer(handle, 0, LoomLib.dtypeInt8);
loomLib.syncInferenceWeights(handle);
```

### CPU training

```dart
final raw = loomLib.train(
  handle,
  flatInput,
  flatTarget,
  batchSize: 1,
  inDim: 16,
  outDim: 8,
  epochs: 10,
  learningRate: 0.05,
  mode: 2, // CPU MC — use 1 for CPU SC
  inputShape: [1, 16],
  targetShape: [1, 8],
);
final hist = (jsonDecode(raw)['loss_history'] as List).cast<num>();
```

### Backward (gradients)

```dart
final bwd = loomLib.backwardPolymorphic(
  handle, input, inShape, target, targetShape,
);
final m = loomParseResult(bwd); // dx, dw maps
```

### Mesh step

```dart
final state = loomLib.createStepState(handle);
loomLib.setInput(state, input);
loomLib.meshStep(handle, state);
final layerOut = loomParseFloatArray(loomLib.getOutput(state, 0));
loomLib.freeStepState(state);
```

### JSON checkpoint

```dart
final wire = loomLib.serializeNetwork(handle);
final reloaded = loomLib.deserializeNetwork(wire);
loomLib.syncInferenceWeights(reloaded);
```

### Seven-layer CPU suite (Lucy [7] parity)

```dart
import 'package:welvet/welvet.dart';

final result = runSevenLayerSuite(
  layerFilter: 'Dense',
  dtypeFilter: 'Float32',
  onLog: print,
);
print('passed=${result.passed} failed=${result.failed}');
```

Layer types: `Dense`, `SwiGLU`, `MHA`, `CNN1`, `CNN2`, `CNN3`, `RNN`, `LSTM`, `Embedding`, `Residual`.

### SoulGlitch-style LLM (advanced)

When natives include LLM symbols and you ship a model snapshot + tokenizer:

```dart
final llm = loomLib.createLLM(snapshotDir, execMode: 3, precision: 4);
final reply = loomLib.llmGenerate(llm, systemPrompt, userMessage);
loomLib.freeLLM(llm);
```

SoulGlitch uses path-vendored `welvet` + XCFramework on iOS; see SoulGlitch macOS/iOS Pod setup for production mobile.

---

## Platform notes

| Platform | Package | Notes |
|----------|---------|--------|
| **macOS** | [`welvet_apple`](https://pub.dev/packages/welvet_apple) | `libwelvet.dylib` in plugin Frameworks |
| **Linux** | [`welvet_linux`](https://pub.dev/packages/welvet_linux) | x86_64 + ARM64 `.so` (auto via `welvet` dep) |
| **Windows** | [`welvet_windows`](https://pub.dev/packages/welvet_windows) | x86_64 + ARM64 `.dll` |
| **Android** | [`welvet_android`](https://pub.dev/packages/welvet_android) | arm64-v8a + x86_64 |
| **iOS** | [`welvet_apple`](https://pub.dev/packages/welvet_apple) | `Welvet.xcframework` |
| **Web** | — | Use [@openfluke/welvet](deployment.md) WASM in browser |

Add only `welvet: ^0.80.6` in `pubspec.yaml` — Flutter pulls the correct impl package per platform.

Monorepo developers can refresh natives:

```bash
cd welvet/cabi/internal/build && ./build_unix.sh all
cd ../../dart && bash tool/copy_native.sh --all
```

Local monorepo dev uses `pubspec_overrides.yaml` for path impl packages (not published).

---

## Related docs

- [deployment.md](deployment.md) — TypeScript/npm and WASM
- [training.md](training.md) — CPU SC/MC modes, loss types
- [entity.md](entity.md) — `.entity` native checkpoints
- [dna.md](dna.md) — `extractDNA`, `compareDNA`, `spliceDNA`
- [step.md](step.md) — mesh clock engine
- [testing_and_validation.md](testing_and_validation.md) — seven-layer suite legend
- [welvet on pub.dev](https://pub.dev/packages/welvet)
- [welvet/dart README](../welvet/dart/README.md) — package source
