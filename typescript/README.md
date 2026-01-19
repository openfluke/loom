# @openfluke/welvet - LOOM TypeScript/WASM Bindings
 
 **Wrapper for Embedding Loom Via External (WASM) Toolchain**
 
 High-performance neural network library with **full training in browser/Node.js** via WebAssembly. Zero external dependencies—just import and go.
 
 > **v0.3.0 Update**: Now includes a **Universal Test Suite** (2298 tests) with 100% parity across Browser and Node.js environments.


## Framework Comparison

| Feature Category | Feature | **Loom/welvet** | **TensorFlow.js** | **Brain.js** | **ONNX.js** | **Candle (WASM)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Core** | **Runtime** | WASM (Pure Go) | JS + WebGL | Pure JS | WASM | WASM (Rust) |
| | **Runtime Dependency** | **None** | Heavy | Light | Light | None |
| **Loading** | **Safetensors** | ✅ **Native** | ❌ | ❌ | ❌ | ✅ |
| | **Structure Inference** | ✅ **Auto-Detect** | ❌ | ❌ | ❌ | ❌ |
| **Training** | **Browser Training** | ✅ **Full** | ✅ (Slow) | ✅ | ❌ | ✅ |
| | **Neural Tweening** | ✅ **Hybrid Engine** | ❌ | ❌ | ❌ | ❌ |
| | **LR Schedulers** | ✅ **7 Types** | ✅ | ⚠️ | ❌ | ✅ |
| **Layer Support** | **Dense (MLP)** | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **Conv1D/2D** | ✅ **Native** | ✅ | ❌ | ✅ | ✅ |
| | **RNN / LSTM** | ✅ **Full Gate** | ✅ | ✅ | ✅ | ✅ |
| | **Transformer (MHA)** | ✅ (Explicit) | ✅ | ❌ | ✅ | ✅ |
| | **Parallel / MoE** | ✅ **Structure** | ❌ (Manual) | ❌ | ❌ | ❌ |
| | **Sequential Layers** | ✅ **Native** | ⚠️ | ⚠️ | ❌ | ⚠️ |
| **Advanced** | **Step-Based Forward** | ✅ **Unique** | ❌ | ❌ | ❌ | ❌ |
| | **Stitch Layers** | ✅ **Native** | ❌ | ❌ | ❌ | ❌ |
| | **K-Means / Stats** | ✅ **Parallel** | ❌ | ❌ | ❌ | ❌ |
| | **Cross-Lang ABI** | ✅ **Universal** | ❌ | ❌ | ❌ | ⚠️ |
| **Streaming** | **LLM Streaming** | ✅ | ✅ | ❌ | ❌ | ✅ |
| | **Pure Go Tokenizer** | ✅ | ❌ | ❌ | ❌ | ❌ |

For detailed comparison, see [`docs/loom_assessment_comparison.md`](../docs/loom_assessment_comparison.md).

## 🌍 Cross-Ecosystem Compatibility

Models trained in TypeScript can be loaded instantly in Python, C#, Go, or other WASM environments. **Bit-for-bit identical results** across all platforms.

| Platform | Package | Install |
|:---------|:--------|:--------|
| **TypeScript/Node** | [NPM](https://www.npmjs.com/package/@openfluke/welvet) | `npm install @openfluke/welvet` |
| **Python** | [PyPI](https://pypi.org/project/welvet/) | `pip install welvet` |
| **C#/.NET** | [NuGet](https://www.nuget.org/packages/Welvet) | `dotnet add package Welvet` |
| **Go** | [GitHub](https://github.com/openfluke/loom) | `go get github.com/openfluke/loom` |

### Key Strengths

- **True Embeddability**: Single WASM binary. Works in Node.js and browsers with the same API.
- **Hybrid Gradient/Geometric Engine**: "Neural Tweening" combines geometric gap-closing with backpropagation-guided momentum.
- **Structural Parallelism**: Native support for Inception, ResNeXt, Siamese, and MoE architectures via `LayerParallel`.
- **Native Mixed-Precision**: Generic tensor backend supports `int8`, `uint16`, and `float32`.
- **Complete Evaluation Suite**: Deviation metrics, training milestones, and adaptation tracking.
- **Network Telemetry**: Runtime introspection with `GetMethodsJSON()` and `ExtractNetworkBlueprint()`.

## Installation

```bash
npm install @openfluke/welvet
```

**Using Bun:**
```bash
bun add @openfluke/welvet
```

## Quick Start

### 🎉 Simple API (Recommended)

The simple API provides streamlined functions with **cross-platform consistency**:

```typescript
import { init, createNetworkFromJSON, loadLoomNetwork } from "@openfluke/welvet";

// Initialize LOOM WASM
await init();

// Create network from JSON config
const config = {
  batch_size: 1,
  grid_rows: 1,
  grid_cols: 3,
  layers_per_cell: 1,
  layers: [
    { type: "dense", input_size: 8, output_size: 16, activation: "relu" },
    {
      type: "parallel",
      combine_mode: "grid_scatter",
      grid_output_rows: 3, grid_output_cols: 1, grid_output_layers: 1,
      grid_positions: [
        { branch_index: 0, target_row: 0, target_col: 0, target_layer: 0 },
        { branch_index: 1, target_row: 1, target_col: 0, target_layer: 0 },
        { branch_index: 2, target_row: 2, target_col: 0, target_layer: 0 },
      ],
      branches: [
        { type: "parallel", combine_mode: "add", branches: [
          { type: "dense", input_size: 16, output_size: 8, activation: "relu" },
          { type: "dense", input_size: 16, output_size: 8, activation: "gelu" },
        ]},
        { type: "lstm", input_size: 16, hidden_size: 8, seq_length: 1 },
        { type: "rnn", input_size: 16, hidden_size: 8, seq_length: 1 },
      ],
    },
    { type: "dense", input_size: 24, output_size: 2, activation: "sigmoid" },
  ],
};

const network = createNetworkFromJSON(JSON.stringify(config));

// Training
const batches = [
  { Input: [0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8], Target: [1.0, 0.0] },
  { Input: [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1], Target: [0.0, 1.0] },
];

const trainingConfig = {
  Epochs: 800,
  LearningRate: 0.15,
  LossType: "mse",
  GradientClip: 1.0,
};

network.Train(JSON.stringify([batches, trainingConfig]));

// Forward pass
const [output] = network.ForwardCPU(JSON.stringify([[0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8]]));
console.log("Output:", JSON.parse(output)); // [0.95, 0.05]

// Save/Load
const [modelJSON] = network.SaveModelToString(JSON.stringify(["my_model"]));
const loadedNetwork = loadLoomNetwork(modelJSON, "my_model");
```

**Simple API Functions:**

| Function | Description |
|:---------|:------------|
| `createNetworkFromJSON(config)` | Create network from JSON |
| `loadLoomNetwork(json, id)` | Load saved model |
| `network.ForwardCPU(input)` | Forward pass |
| `network.BackwardCPU(gradients)` | Backward pass |
| `network.Train(params)` | Train network |
| `network.SaveModelToString(id)` | Save to JSON string |
| `network.EvaluateNetwork(params)` | Evaluate with metrics |

### ⚡ Stepping API - Fine-Grained Execution Control

Execute networks one step at a time for online learning:

```typescript
import { init, createNetwork, StepState } from "@openfluke/welvet";

await init();

const network = createNetwork({
  batch_size: 1,
  layers: [
    { type: "dense", input_height: 4, output_height: 8, activation: "relu" },
    { type: "lstm", input_size: 8, hidden_size: 12, seq_length: 1 },
    { type: "dense", input_height: 12, output_height: 3, activation: "softmax" }
  ]
});

// Initialize stepping state
const state: StepState = network.createStepState(4);

// Training loop - update weights after EACH step
for (let step = 0; step < 100000; step++) {
  state.setInput(new Float32Array([0.1, 0.2, 0.1, 0.3]));
  state.stepForward();
  const output = state.getOutput();
  
  // Backward pass
  const gradients = output.map((o, i) => o - target[i]);
  state.stepBackward(gradients);
  
  // Update weights immediately
  network.ApplyGradients(JSON.stringify([learningRate]));
}
```

### 🧠 Neural Tweening API - Gradient-Free Learning

Direct weight adjustment without backpropagation:

```typescript
import { init, createNetwork, TweenState } from "@openfluke/welvet";

await init();

const network = createNetwork(config);

// Create tween state (with optional chain rule)
const tweenState: TweenState = network.createTweenState(true); // useChainRule=true

// Training loop - direct weight updates
for (let step = 0; step < 10000; step++) {
  const loss = tweenState.TweenStep(
    new Float32Array([0.1, 0.2, 0.3, 0.4]),
    1,     // targetClass
    4,     // outputSize
    0.02   // learningRate
  );
}
```

### 📊 Adaptation Benchmark

Run the full multi-architecture adaptation benchmark:

```bash
cd example
bun run test18_adaptation.ts
```

Tests **5 architectures × 3 depths × 5 training modes** (75 tests total):
- **Architectures:** Dense, Conv2D, RNN, LSTM, Attention
- **Depths:** 3, 5, 9 layers
- **Modes:** NormalBP, StepBP, Tween, TweenChain, StepTweenChain

## Complete Test Suite
 
 The `universal_test.ts` example demonstrates all framework capabilities with **100% parity** to the Go/C core.
 
 ### Running in Browser (v0.3.0+)
 
 The universal test suite now runs directly in the browser with a full DOM report:
 
 ```bash
 cd typescript
 python3 serve.py
 # Open http://localhost:8081
 ```
 
 ### Running in Node/Bun
 
 ```bash
 cd example
 bun run universal_test.ts
 ```
 
 **Test Coverage (2298 Tests):**
 - ✅ **Serialization**: 12 Layer Types × 15 Data Types (2100 combinations)
 - ✅ **In-Memory WASM**: SafeTensors without filesystem (144 tests)
 - ✅ **Advanced Math**: K-Means, Correlation, Grafting, Ensembles
 - ✅ **GPU Parity**: Determinism checks for forward/backward passes
 - ✅ **Core**: Architecture generation, combinators, sequential layers
 
 See [`example/universal_test.ts`](./example/universal_test.ts) for the complete test implementation.

## Layer Types

| Layer | Type String | Description |
|:------|:------------|:------------|
| Dense | `dense` | Fully connected layer |
| LSTM | `lstm` | Long Short-Term Memory |
| RNN | `rnn` | Recurrent Neural Network |
| GRU | `gru` | Gated Recurrent Unit |
| Conv2D | `conv2d` | 2D Convolution |
| Conv1D | `conv1d` | 1D Convolution |
| Multi-Head Attention | `multi_head_attention` | Transformer attention |
| LayerNorm | `layer_norm` | Layer normalization |
| RMSNorm | `rms_norm` | RMS normalization |
| SwiGLU | `swiglu` | SwiGLU activation layer |
| Softmax | `softmax` | Softmax classification |
| Embedding | `embedding` | Token embedding |
| Parallel | `parallel` | Branching with combine modes |
| Sequential | `sequential` | Grouped sub-layers |

**Parallel Combine Modes:** `add`, `concat`, `multiply`, `average`, `grid_scatter`, `filter`

## Activation Functions

`relu`, `sigmoid`, `tanh`, `softmax`, `gelu`, `swish`, `mish`, `leaky_relu`, `elu`, `selu`, `linear`

## API Reference

### Initialization

```typescript
import { init, initBrowser, createNetwork, createNetworkFromJSON } from "@openfluke/welvet";

// Node.js
await init();

// Browser
await initBrowser();
```

### Network Methods

All Network methods follow the WASM calling convention:
- **Input:** JSON string of an array of parameters
- **Return:** JSON string of an array of results

```typescript
// Method with no parameters
const info = network.GetNetworkInfo(JSON.stringify([]));

// Method with parameters
const result = network.Train(JSON.stringify([batches, config]));

// Save model
const saved = network.SaveModelToString(JSON.stringify(["my-model"]));
```

**Available Methods:**

| Method | Parameters | Description |
|:-------|:-----------|:------------|
| `ForwardCPU` | `[inputs]` | CPU forward pass |
| `ForwardGPU` | `[inputs]` | GPU forward pass |
| `BackwardCPU` | `[gradients]` | CPU backward pass |
| `Train` | `[batches, config]` | Train network |
| `SaveModelToString` | `["modelID"]` | Save to JSON |
| `GetWeights` | `[row, col, layer]` | Get layer weights |
| `SetWeights` | `[row, col, layer, weights]` | Set layer weights |
| `GetBiases` | `[row, col, layer]` | Get layer biases |
| `SetBiases` | `[row, col, layer, biases]` | Set layer biases |
| `GetNetworkInfo` | `[]` | Get network info |
| `GetTotalParameters` | `[]` | Get parameter count |
| `Clone` | `[]` | Clone network |
| `TotalLayers` | `[]` | Get total layer count |

### Statistical Tools

```typescript
import welvet from "@openfluke/welvet";

// K-Means Clustering
const data = [[1, 1], [1.1, 1.1], [5, 5], [5.1, 5.1]];
const result = welvet.kmeans(data, 2, 100);
console.log(`Centroids: ${result.centroids}`);
console.log(`Silhouette Score: ${result.silhouette_score}`);

// Correlation Matrix
const matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
const corr = welvet.correlation(matrix);
console.log(`Pearson: ${corr.pearson}`);
```

### Network Grafting

Combine multiple trained networks:

```typescript
const h1 = welvet.createKHandle(config);
const h2 = welvet.createKHandle(config);

const result = welvet.graft([h1, h2], "concat");
console.log(`Grafted: ${result.num_branches} branches`);
```

## Examples

```bash
cd example

# Grid Scatter Multi-Agent
bun run grid-scatter.ts

# Stepping Training (LSTM)
bun run step_train_v3.ts

# Adaptation Benchmark (75 tests)
bun run test18_adaptation.ts

# Full Test Suite (77 tests)
bun run universal_test.ts
```

## TypeScript Types

```typescript
interface NetworkConfig {
  batch_size: number;
  grid_rows?: number;
  grid_cols?: number;
  layers_per_cell?: number;
  layers: LayerConfig[];
  dtype?: "float32" | "float64" | "int32" | "int16" | "int8" | "uint8";
}

interface TrainingConfig {
  Epochs: number;
  LearningRate: number;
  LossType?: string;
  Verbose?: boolean;
  UseGPU?: boolean;
  GradientClip?: number;
}

interface TrainingBatch {
  Input: number[];
  Target: number[];
}
```

## License

Apache-2.0

## Links

- **GitHub**: [github.com/openfluke/loom](https://github.com/openfluke/loom)
- **NPM**: [@openfluke/welvet](https://www.npmjs.com/package/@openfluke/welvet)
- **PyPI**: [welvet](https://pypi.org/project/welvet/)
- **NuGet**: [Welvet](https://www.nuget.org/packages/Welvet)
- **Documentation**: [`docs/loom_assessment_comparison.md`](../docs/loom_assessment_comparison.md)
