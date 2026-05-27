# @openfluke/welvet

**M-POLY-VTD AI Engine (Loom v0.79.0)** — Isomorphic TypeScript/WASM bindings for the [Loom](https://github.com/openfluke/loom) deterministic neural VM: 21 numerical types, volumetric 3D grids, CPU/GPU training paths, DNA evolution, and bit-packed save/reload.

[![npm version](https://img.shields.io/npm/v/@openfluke/welvet.svg)](https://www.npmjs.com/package/@openfluke/welvet)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **Loom core:** [README](https://github.com/openfluke/loom/blob/main/README.md) · **Docs index:** [`docs/index.md`](https://github.com/openfluke/loom/blob/main/docs/index.md) · **Bedrock validation (v0.79):** [`docs/bedrock_validation.md`](https://github.com/openfluke/loom/blob/main/docs/bedrock_validation.md)

## What this package is

`@openfluke/welvet` ships a Go→WASM build of the **`poly/`** engine. Your TypeScript or browser code calls **Loom through WASM** — same math as native Go and the Python/CABI `.so`, without reimplementing layers in JS.

| Binding | Best for |
|--------|----------|
| **This package (WASM)** | Browser, Node.js, Bun, edge — one `main.wasm` + `wasm_exec.js` |
| **[`welvet` on PyPI](https://pypi.org/project/welvet/)** | Servers, notebooks — ctypes C-ABI |
| **Go `poly/`** | Reference, Lucy harness, maximum CPU parallelism |

**v0.79.0 "Bedrock Validation"** (see [release notes](https://github.com/openfluke/loom/blob/main/docs/bedrock_validation.md)): seven-layer CPU regression (10 layer types × 21 dtypes × train × native save/reload), MHA `[B,S,D]` layout, KV train/decode split, C-ABI **461/461**. The TypeScript seven-layer suite exercises the same paths via WASM.

## Features

- **Isomorphic**: Node.js, Bun, and browser (React/Vite/etc.) from one API.
- **21 DTypes**: Float64 → Binary/Ternary — runtime `morphLayer()` per layer.
- **Volumetric grid**: `depth × rows × cols` cells, multiple layers per cell (Lucy-style JSON).
- **Training**: `train()` / `trainNetwork()` with CPU SC/MC modes (parity on WASM; true multicore on native CABI).
- **Polymorphic forward/backward**: shape-aware tensors (e.g. MHA `[batch, seq, d_model]`).
- **Persistence**: `serialize()` + `deserializeLoomNetwork()` wire checkpoints.
- **WebGPU** (browser): optional acceleration after `setupWebGPU()`.
- **DNA & NEAT**: extract/compare DNA, splice, populations.

## Installation

```bash
npm install @openfluke/welvet
# or
bun add @openfluke/welvet
```

### Build from source (monorepo)

WASM is not rebuilt on `npm install` — publish ships prebuilt `dist/main.wasm`. To refresh after pulling [loom](https://github.com/openfluke/loom):

```bash
# From repo root (or use npm run build:wasm inside welvet/typescript)
bash welvet/wasm/build.sh               # → welvet/typescript/assets/main.wasm
cd welvet/typescript && npm run build   # tsc + copy assets → dist/

# One shot from welvet/typescript:
npm run build:all
```

`npm test` / `test:consumer` call `loomEngineVersion()` on the loaded WASM and fail if it does not match `LOOM_ENGINE_VERSION` in the package (stale `dist/main.wasm` still prints the old init banner).

## Quick start

### 1. Initialize

```typescript
import { init, createNetwork, DType } from "@openfluke/welvet";

await init(); // Node: loads dist/main.wasm · Browser: fetch main.wasm
```

Browser (non-default asset path):

```typescript
import { initBrowser } from "@openfluke/welvet";

await initBrowser("/assets/main.wasm");
```

### 2. Volumetric network (Dense stack)

Layers are addressed by grid coordinates `(z, y, x, l)` — see [`docs/overview.md`](https://github.com/openfluke/loom/blob/main/docs/overview.md).

```typescript
import { init, createNetwork, DType } from "@openfluke/welvet";

await init();

const net = createNetwork({
  id: "demo-dense",
  depth: 1,
  rows: 1,
  cols: 1,
  layers_per_cell: 2,
  layers: [
    {
      z: 0, y: 0, x: 0, l: 0,
      type: "DENSE",
      dtype: "FLOAT32",
      input_height: 784,
      output_height: 256,
      activation: "RELU",
    },
    {
      z: 0, y: 0, x: 0, l: 1,
      type: "DENSE",
      dtype: "FLOAT32",
      input_height: 256,
      output_height: 10,
      activation: "LINEAR",
    },
  ],
});

const input = new Float32Array(784).fill(0.1);
const inShape = JSON.stringify([1, 784]);
const output = net.forwardPolymorphic(input, inShape); // Float32Array, length 10
console.log(output[0]);
```

### 3. Morph precision (21 types)

```typescript
import { DType } from "@openfluke/welvet";

const info = JSON.parse(net.getInfo());
for (let i = 0; i < info.total_layers; i++) {
  net.morphLayer(i, DType.INT8);
}

const outQ = net.forwardPolymorphic(input, inShape);
```

DType IDs match [`docs/numerical_types.md`](https://github.com/openfluke/loom/blob/main/docs/numerical_types.md). Use exported `DType.*` constants in TypeScript.

### 4. Shape-aware forward (MHA, CNN, etc.)

For attention and conv layers, pass explicit shapes (do not flatten to `[batch, features]` unless that is the real layout).

```typescript
const batch = 4;
const seq = 8;
const dModel = 64;

const mhaNet = createNetwork({
  id: "demo-mha",
  depth: 1,
  rows: 1,
  cols: 1,
  layers_per_cell: 1,
  layers: [
    {
      z: 0, y: 0, x: 0, l: 0,
      type: "MHA",
      dtype: "FLOAT32",
      d_model: dModel,
      num_heads: 4,
      seq_length: seq,
      activation: "RELU",
    },
  ],
});

const inp = new Float32Array(batch * seq * dModel);
for (let i = 0; i < inp.length; i++) inp[i] = 0.2 * Math.sin(i * 0.11);

const shapeJson = JSON.stringify([batch, seq, dModel]);
const out = mhaNet.forwardPolymorphic(inp, shapeJson); // Float32Array
```

### 5. Training (CPU)

`TrainingMode`: **1** = CPU single-core tiling (SC), **2** = CPU multi-core tiling (MC). On WASM both paths run on a single host thread; the suite checks **numerical parity** between them.

```typescript
import { trainNetwork, DType } from "@openfluke/welvet";

const batches = [
  {
    input: new Float32Array(784).fill(0.1),
    target: new Float32Array(10).fill(0.5),
    inputShape: [1, 784],   // batch × features — match layer input_height
    targetShape: [1, 10],
  },
];

// Simple helper (FP32, default mode)
const result = trainNetwork(net, batches, 10, 0.05);
console.log(result.loss_history);

// Full config (mode, clip, epochs) — call train() on the network handle
const cfg = JSON.stringify({
  Epochs: 50,
  LearningRate: 0.05,
  LossType: "mse",
  Mode: 2,              // CPUMC
  GradientClip: 1.0,
  Verbose: false,
  UseGPU: false,
});
const raw = JSON.parse(
  net.train(JSON.stringify([{
    input: { shape: [1, 784], data: [...batches[0].input] },
    target: { shape: [1, 10], data: [...batches[0].target] },
  }]), 50, 0.05, cfg)
);
console.log(raw.loss_history);
```

See [`docs/training.md`](https://github.com/openfluke/loom/blob/main/docs/training.md) for loss types, tween/target propagation, and GPU training on native builds.

### 6. Save / reload (native wire format)

```typescript
await init();

const wire = net.serialize(); // JSON wire string
const reloaded = (globalThis as any).deserializeLoomNetwork(wire);

const a = net.sequentialForward(input);
const b = reloaded.sequentialForward(input);
reloaded.free();
```

Same format as Go `SerializeNetwork` / Python `Network.deserialize()` — see [`docs/serialization.md`](https://github.com/openfluke/loom/blob/main/docs/serialization.md).

### 7. WebGPU (browser)

```typescript
import { init, setupWebGPU, createNetwork } from "@openfluke/welvet";

await init();
await setupWebGPU(); // sets window.webgpuDevice — required before initGPU()

const net = createNetwork({ /* ... */ });
await net.initGPU();
await net.syncToGPU();
// Forward/training may use GPU kernels when configured
```

Details: [`docs/gpu.md`](https://github.com/openfluke/loom/blob/main/docs/gpu.md) · [`docs/deployment.md`](https://github.com/openfluke/loom/blob/main/docs/deployment.md).

### 8. DNA comparison

```typescript
import { compareDNA } from "@openfluke/welvet";

const dnaA = net.extractDNA();
const dnaB = other.extractDNA();
const sim = compareDNA(dnaA, dnaB);
console.log(sim.similarity, sim.logic_shift);
```

### 9. NEAT population

```typescript
import { createNEATPopulation, getNEATConfig } from "@openfluke/welvet";

const cfg = getNEATConfig(64);
const pop = createNEATPopulation(net, 100, cfg);
pop.evolveWithFitnesses(new Float64Array(100).fill(Math.random()));
console.log(pop.bestFitness());
```

## Seven-layer validation (TypeScript → WASM → Loom)

The Lucy **[7]** suite logic lives in **`welvet/seven_layer/`** (copied to `dist/seven_layer/` on build). It is the bedrock gate from v0.79: forward/backward SC/MC, train, save/reload, 21 dtypes.

```bash
cd welvet/typescript
npm run build

# Full suite (slow — MHA/CNN3 take a while)
npm run test:seven-layer

# One layer type at a time
npm run test:seven-layer:dense
npm run test:seven-layer -- --layer SwiGLU
npm run test:seven-layer -- --layer MHA
npm run test:seven-layer -- --layer CNN1
npm run test:seven-layer -- --layer CNN2
npm run test:seven-layer -- --layer CNN3
npm run test:seven-layer -- --layer RNN
npm run test:seven-layer -- --layer LSTM
npm run test:seven-layer -- --layer Embedding
npm run test:seven-layer -- --layer Residual
```

Browser demo (after `npm run build`):

```bash
npm run serve:seven-layer
# → http://localhost:3000/benchmark_seven_layer.html
```

Cross-check with Python (CABI, multicore CPU):

```bash
cd welvet/python
python benchmark_seven_layer.py --layer Dense
```

## Other tests

```bash
npm test              # cabi + benchmarks + coverage + consumer_demo
npm run test:consumer # README / npm quick-start smoke (forwardPolymorphic, train, serialize)
npm run test:cabi     # WASM export / functional smoke
npm run test:bench    # Layer micro-benchmarks
npm run test:coverage # Export coverage vs Go parity list
```

## API surface (after `init()`)

Globals injected by WASM (also wrapped by this package):

| Category | Methods |
|----------|---------|
| **Lifecycle** | `createLoomNetwork(json)`, `deserializeLoomNetwork(wire)`, `loadLoomNetwork(path)` |
| **Network** | `sequentialForward`, `forwardPolymorphic(data, shapeJson)`, `backwardPolymorphic(...)`, `train(...)`, `serialize`, `morphLayer`, `setTrainingMode(mode)`, `getInfo`, `extractDNA`, `initGPU`, `syncToGPU` / `syncToCPU`, `free` |
| **Evolution** | `createLoomNEATPopulation`, `compareLoomDNA`, `defaultNEATConfig`, `defaultSpliceConfig` |
| **Browser** | `setupWebGPU()` |

TypeScript exports: `init`, `initBrowser`, `createNetwork`, `trainNetwork`, `compareDNA`, `createNEATPopulation`, `createTransformer`, `setupWebGPU`, `DType`, `LayerType`, `Activation`, and types in `dist/index.d.ts`.

## Documentation map (Loom repo)

| Topic | Doc |
|-------|-----|
| Architecture | [`docs/overview.md`](https://github.com/openfluke/loom/blob/main/docs/overview.md) |
| All layer types | [`docs/layers.md`](https://github.com/openfluke/loom/blob/main/docs/layers.md) |
| 21 dtypes / morph | [`docs/numerical_types.md`](https://github.com/openfluke/loom/blob/main/docs/numerical_types.md) |
| Training | [`docs/training.md`](https://github.com/openfluke/loom/blob/main/docs/training.md) |
| Transformers / MHA | [`docs/transformer.md`](https://github.com/openfluke/loom/blob/main/docs/transformer.md) |
| Save/load | [`docs/serialization.md`](https://github.com/openfluke/loom/blob/main/docs/serialization.md) |
| v0.79 bedrock suite | [`docs/bedrock_validation.md`](https://github.com/openfluke/loom/blob/main/docs/bedrock_validation.md) |
| Snippets | [`docs/quick_reference.md`](https://github.com/openfluke/loom/blob/main/docs/quick_reference.md) |

## Version alignment

| Component | Version |
|-----------|---------|
| **Loom engine (poly)** | **0.79.0** — Bedrock Validation |
| **npm `@openfluke/welvet`** | **0.79.0** (rebuild WASM from this repo to match latest `main`) |

## License

Apache-2.0 — see [LICENSE](LICENSE).
