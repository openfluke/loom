# @openfluke/welvet

**M-POLY-VTD AI Engine (Loom v0.74.0)** — Isomorphic TypeScript/WASM library for building, training, and evolving neural networks with 21 numerical types, WebGPU acceleration, and DNA-based evolution.

[![npm version](https://img.shields.io/npm/v/@openfluke/welvet.svg)](https://www.npmjs.com/package/@openfluke/welvet)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Features

- **Isomorphic Architecture**: Runs seamlessly in Node.js, Bun, and the Browser (React/Frontend).
- **Multi-Precision Support**: Native support for 21 numerical types (FP64, FP32, FP16, BF16, FP8, INT8... all the way down to Binary/Ternary).
- **Hybrid Training**: standard Backprop, Target Propagation, and NEAT-style structural evolution in a single unified engine.
- **Hardware Acceleration**: WebGPU support for high-performance inference and training in the browser.
- **DNA Evolution**: Network "DNA" extraction and comparison for architectural analysis and genetic recombination.

## Installation

```bash
npm install @openfluke/welvet
# or
bun add @openfluke/welvet
```

## Quick Start

### 1. Initialization

The engine initializes automatically after a single call to `init()`. It detects whether it's running in Node.js, Bun, or the Browser and loads the appropriate WASM environment.

```typescript
import { init } from "@openfluke/welvet";

await init(); // Just works™
```

> [!TIP]
> In browser environments, if your assets are served from a non-standard path, you can optionally pass a custom URL: `await init("/custom/path/to/main.wasm")`.

### 2. Create a Network

```typescript
import { createNetwork, DType } from "@openfluke/welvet";

const net = createNetwork({
  layers: [
    { type: "Dense", input_height: 784, output_height: 256, activation: "ReLU", dtype: DType.FLOAT32 },
    { type: "Dense", input_height: 256, output_height: 10, activation: "Linear", dtype: DType.FLOAT16 }
  ]
});

const input = new Float32Array(784).fill(0.5);
const output = net.sequentialForward(input);
console.log("Prediction:", output);
```

### 3. Training

```typescript
import { trainNetwork } from "@openfluke/welvet";

const batches = [
  { input: [0.1, 0.2, ...], target: [1, 0, ...] },
];

const result = trainNetwork(net, batches, 10, 0.001);
console.log("Final Loss:", result.final_loss);
```

### 4. NEAT Evolution

```typescript
import { createNEATPopulation } from "@openfluke/welvet";

const population = createNEATPopulation(net, 100);
const fitnesses = new Array(100).fill(0).map(() => Math.random());

population.evolveWithFitnesses(fitnesses);
const bestNet = population.best();
console.log("Best Fitness:", population.bestFitness());
```

## Testing & Benchmarks

The package includes a built-in TypeScript testing suite suitable for both CI and environment verification.

### Run tests in Node.js:

```bash
npm test
```

Or individual suites:
```bash
npm run test:cabi    # Check WASM exports and functional smoke tests
npm run test:bench   # Run layer-by-layer performance benchmarks
```

### Integrate tests into Frontend/React:

```typescript
import { runVerify } from "@openfluke/welvet/tests/cabi_verify";

// Run benchmarks or verification logic directly in your app's lifecycle
await runVerify();
```

## License

Apache-2.0
