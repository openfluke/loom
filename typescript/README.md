# @openfluke/welvet

Isomorphic TypeScript/JavaScript wrapper for the LOOM WebAssembly neural network framework.

## Features

- 🎉 **NEW: Simple API** - Streamlined functions with cross-platform consistency
- 🚀 **Isomorphic WASM Wrapper** - Works in Node.js and browser with same API
- 🔄 **Mirrors main.go** - Direct 1:1 mapping to WASM exports
- 🎯 **Type-Safe** - Full TypeScript type definitions for all Network methods
- 🤖 **Multi-Agent Networks** - Grid scatter architecture for heterogeneous agents
- 📦 **JSON Configuration** - Build networks from simple JSON configs
- ⚡ **Fast Training** - Optimized training with configurable parameters
- 💾 **Model Persistence** - Save and load trained models as JSON
- ✅ **Cross-Platform Consistency** - Same API as Python, C#, C, WASM

## Installation

```bash
npm install @openfluke/welvet
```

## Quick Start

### 🎉 NEW: Simple API (Recommended)

The simple API provides streamlined functions with consistent behavior across all platforms:

```typescript
import {
  init,
  createNetworkFromJSON,
  loadLoomNetwork,
} from "@openfluke/welvet";

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
      grid_output_rows: 3,
      grid_output_cols: 1,
      grid_output_layers: 1,
      grid_positions: [
        { branch_index: 0, target_row: 0, target_col: 0, target_layer: 0 },
        { branch_index: 1, target_row: 1, target_col: 0, target_layer: 0 },
        { branch_index: 2, target_row: 2, target_col: 0, target_layer: 0 },
      ],
      branches: [
        {
          type: "parallel",
          combine_mode: "add",
          branches: [
            {
              type: "dense",
              input_size: 16,
              output_size: 8,
              activation: "relu",
            },
            {
              type: "dense",
              input_size: 16,
              output_size: 8,
              activation: "gelu",
            },
          ],
        },
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
  { Input: [0.7, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3], Target: [0.0, 1.0] },
  { Input: [0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7], Target: [1.0, 0.0] },
];

const trainingConfig = {
  Epochs: 800,
  LearningRate: 0.15,
  UseGPU: false,
  PrintEveryBatch: 0,
  GradientClip: 1.0,
  LossType: "mse",
  Verbose: false,
};

const [result] = network.Train(JSON.stringify([batches, trainingConfig]));
console.log("Training complete!");

// Forward pass
const [output] = network.ForwardCPU(
  JSON.stringify([[0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8]])
);
console.log("Output:", JSON.parse(output)); // [0.950, 0.050]

// Evaluate network
const inputs = batches.map((b) => b.Input);
const expected = [0, 1, 1, 0];
const [metrics] = network.EvaluateNetwork(JSON.stringify([inputs, expected]));
const metricsData = JSON.parse(metrics);
console.log(
  `Quality: ${metricsData.score}/100, Deviation: ${metricsData.avg_deviation}%`
);

// Save/Load
const [modelJSON] = network.SaveModelToString(JSON.stringify(["my_model"]));
console.log(`Model saved (${modelJSON.length} bytes)`);

// Load model
const loadedNetwork = loadLoomNetwork(modelJSON, "my_model");
const [output2] = loadedNetwork.ForwardCPU(
  JSON.stringify([[0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8]])
);
// output2 === output (bit-for-bit identical!)
```

**Simple API Functions:**

- `createNetworkFromJSON(jsonConfig)` - Create network from JSON
- `loadLoomNetwork(jsonString, modelID)` - Load saved model
- `network.ForwardCPU(inputJSON)` - Forward pass
- `network.BackwardCPU(gradientsJSON)` - Backward pass
- `network.Train(paramsJSON)` - Train network
- `network.SaveModelToString(idJSON)` - Save to JSON string
- `network.EvaluateNetwork(paramsJSON)` - Evaluate with metrics
- `network.UpdateWeights(lrJSON)` - Update weights

**Cross-Platform Results:**

- ✅ Same training: 99.5% improvement, 100/100 quality score
- ✅ Same save/load: 0.00 difference in predictions
- ✅ Same evaluation: Identical deviation metrics
- ✅ Same behavior as Python, C#, C, and WASM

See `example/grid-scatter.ts` for a complete working example.
{
type: "parallel",
combine_mode: "add",
branches: [
{
type: "dense",
input_size: 16,
output_size: 8,
activation: "relu",
},
{
type: "dense",
input_size: 16,
output_size: 8,
activation: "gelu",
},
],
},
{ type: "lstm", input_size: 16, hidden_size: 8, seq_length: 1 },
{ type: "rnn", input_size: 16, hidden_size: 8, seq_length: 1 },
],
},
{
type: "dense",
input_size: 24,
output_size: 2,
activation: "sigmoid",
},
],
});

// Train multi-agent network
const batches: TrainingBatch[] = [
{ Input: [0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8], Target: [1.0, 0.0] },
{ Input: [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1], Target: [0.0, 1.0] },
];

const config: TrainingConfig = {
Epochs: 800,
LearningRate: 0.15,
LossType: "mse",
Verbose: false,
};

const result = agentNetwork.Train(JSON.stringify([batches, config]));

````

## API Reference

### Functions

#### `async init(): Promise<void>`

Initialize LOOM WASM module for Node.js environment.

#### `async initBrowser(): Promise<void>`

Initialize LOOM WASM module for browser environment.

#### `createNetwork(config: object | string): Network`

Create a new neural network from JSON configuration object or string.

**Note:** This is the only global function exposed by the WASM (mirrors `createLoomNetwork` from main.go). To load a saved model, just pass the saved JSON string to `createNetwork()`.

### Network Interface

The `Network` object returned by `createNetwork()` has all methods from the Go `nn.Network` type automatically exposed via reflection.

**Important:** All Network methods follow the WASM calling convention:

- Take a single parameter: JSON string of an array of parameters
- Return a JSON string of an array of results

Example:

```typescript
// Method with no parameters
const info = network.GetNetworkInfo(JSON.stringify([]));
const parsed = JSON.parse(info)[0];

// Method with parameters
const result = network.Train(JSON.stringify([batches, config]));
const data = JSON.parse(result)[0];

// Save model (requires modelID parameter)
const saved = network.SaveModelToString(JSON.stringify(["my-model"]));
const json = JSON.parse(saved)[0];
````

#### Available Network Methods

- `ForwardCPU(paramsJSON)` - CPU forward pass: `[inputs]`
- `ForwardGPU(paramsJSON)` - GPU forward pass: `[inputs]`
- `BackwardCPU(paramsJSON)` - CPU backward pass: `[gradients]`
- `BackwardGPU(paramsJSON)` - GPU backward pass: `[gradients]`
- `UpdateWeights(paramsJSON)` - Update weights: `[learningRate]`
- `Train(paramsJSON)` - Train network: `[batches, config]`
- `SaveModelToString(paramsJSON)` - Save model: `["modelID"]`
- `GetWeights(paramsJSON)` - Get layer weights: `[row, col, layer]`
- `SetWeights(paramsJSON)` - Set layer weights: `[row, col, layer, weights]`
- `GetBiases(paramsJSON)` - Get layer biases: `[row, col, layer]`
- `SetBiases(paramsJSON)` - Set layer biases: `[row, col, layer, biases]`
- `GetActivation(paramsJSON)` - Get activation: `[row, col, layer]`
- `GetLayerType(paramsJSON)` - Get layer type: `[row, col, layer]`
- `GetLayerSizes(paramsJSON)` - Get layer sizes: `[row, col, layer]`
- `GetBatchSize(paramsJSON)` - Get batch size: `[]`
- `GetGridDimensions(paramsJSON)` - Get grid dimensions: `[]`
- `GetNetworkInfo(paramsJSON)` - Get network info: `[]`
- `GetTotalParameters(paramsJSON)` - Get parameter count: `[]`
- `InitializeWeights(paramsJSON)` - Initialize weights: `[]` or `[method]`
- `Clone(paramsJSON)` - Clone network: `[]`
- And 10+ more methods...
- `GetLastOutput(): string` - Get last forward pass output

### Types

#### `NetworkConfig`

```typescript
interface NetworkConfig {
  batch_size: number;
  grid_rows?: number; // Required for grid networks (use 1 for sequential)
  grid_cols?: number; // Required for grid networks (use 1 for sequential)
  layers_per_cell?: number; // Required for grid networks
  layers: LayerConfig[];
}
```

#### `LayerConfig`

```typescript
interface LayerConfig {
  type: string;
  input_size?: number;
  output_size?: number;
  hidden_size?: number;
  seq_length?: number;
  activation?: string;
  combine_mode?: string;
  grid_output_rows?: number;
  grid_output_cols?: number;
  grid_output_layers?: number;
  grid_positions?: GridPosition[];
  branches?: LayerConfig[];
}
```

#### `TrainingBatch`

```typescript
interface TrainingBatch {
  Input: number[];
  Target: number[];
}
```

#### `TrainingConfig`

```typescript
interface TrainingConfig {
  Epochs: number;
  LearningRate: number;
  LossType?: string;
  Verbose?: boolean;
  UseGPU?: boolean;
  PrintEveryBatch?: number;
  GradientClip?: number;
}
```

## Example

See `example/grid-scatter.ts` for a complete multi-agent training demo:

```bash
cd example
bun install
bun run grid-scatter.ts
```

Expected output:

```
🤖 Running Grid Scatter Multi-Agent Training...
✅ Agent network created!
Training for 800 epochs with learning rate 0.150
✅ Training complete!
Training time: 0.47 seconds
Initial Loss: 0.252249
Final Loss: 0.001374
Improvement: 99.46%
Total Epochs: 800
```

## Layer Types

- `dense` - Fully connected layer
- `lstm` - Long Short-Term Memory layer
- `rnn` - Recurrent Neural Network layer
- `gru` - Gated Recurrent Unit layer
- `cnn` - Convolutional layer
- `parallel` - Parallel branches with combine modes:
  - `add` - Element-wise addition
  - `concat` - Concatenation
  - `multiply` - Element-wise multiplication
  - `grid_scatter` - Multi-agent grid routing

## Activation Functions

`relu`, `sigmoid`, `tanh`, `softmax`, `gelu`, `swish`, `mish`, `leaky_relu`, `elu`, `selu`

## License

MIT

## Links

- [GitHub](https://github.com/openfluke/loom)
- [WASM Documentation](../wasm/README.md)
- [Go Examples](../examples/)
