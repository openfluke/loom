# @openfluke/welvet

> TypeScript/JavaScript bindings for the LOOM neural network framework via WebAssembly

**Welvet** (Wrapper for Embedding Loom Via External Toolchain) provides a complete neural network API in the browser and Node.js/Bun environments. Built on WebAssembly, it delivers high-performance deep learning with zero dependencies.

[![npm version](https://img.shields.io/npm/v/@openfluke/welvet.svg)](https://www.npmjs.com/package/@openfluke/welvet)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## ✨ Features

- 🤖 **Transformer Inference (NEW!)** - Run LLMs like SmolLM2-135M with streaming generation
- 🚀 **6.0MB WASM Binary** - Complete neural network framework + transformer inference
- 🧠 **7 Layer Types (All CPU)** - Dense, Conv2D, Multi-Head Attention, LayerNorm, RNN, LSTM, Softmax (10 variants)
- ✅ **Full CPU Implementation** - Every layer works with complete forward/backward passes
- 🎯 **Registry-based Initialization** - Dynamic layer creation via `CallLayerInit()` with zero manual exports
- 🔍 **Runtime Introspection** - Discover methods, signatures, and parameters dynamically
- 💾 **Model Serialization** - Save/load models as JSON (no filesystem required)
- ⚡ **Full Training Support** - Train networks with `network.Train()` API and automatic gradients
- 📘 **Full TypeScript Support** - Complete type definitions for IntelliSense
- 🎯 **Zero Dependencies** - Pure WASM + Go runtime, no external libs
- 🌐 **Isomorphic** - Works in browsers, Node.js, Bun, and Deno
- 🎨 **Multiple Activation Functions** - ReLU, Sigmoid, Tanh, Softplus, LeakyReLU, Linear
- ⚠️ **CPU-Only** (GPU support via WebGPU coming soon)

## 📦 Installation

```bash
npm install @openfluke/welvet
```

Or with your preferred package manager:

```bash
# Yarn
yarn add @openfluke/welvet

# pnpm
pnpm add @openfluke/welvet

# Bun
bun add @openfluke/welvet
```

## 🚀 Quick Start

### 🤖 Transformer Inference (NEW!)

Run Large Language Models with streaming generation:

```typescript
import { initLoom, createTransformerAPI } from "@openfluke/welvet";

// Initialize WASM
await initLoom();

// Create transformer API
const transformer = await createTransformerAPI();

// Load tokenizer
const tokenizerData = await fetch("models/SmolLM2-135M-Instruct/tokenizer.json")
  .then((r) => r.arrayBuffer())
  .then((buf) => new Uint8Array(buf));
await transformer.loadTokenizer(tokenizerData);

// Load model
const configData = await fetch("models/SmolLM2-135M-Instruct/config.json")
  .then((r) => r.arrayBuffer())
  .then((buf) => new Uint8Array(buf));
const weightsData = await fetch(
  "models/SmolLM2-135M-Instruct/model.safetensors"
)
  .then((r) => r.arrayBuffer())
  .then((buf) => new Uint8Array(buf));
await transformer.loadModel(configData, weightsData);

// Stream generation token-by-token
for await (const token of transformer.generateStream(
  "The capital of France is",
  50,
  0.7
)) {
  process.stdout.write(token); // Paris...
}
```

**Live Demo:** See `wasm/inference.html` for a beautiful web UI with real-time token streaming!

### The Easy Way: Load Complete Models

Instead of manually configuring layers, **load a complete model with ONE line**:

```typescript
import { initLoom } from "@openfluke/welvet";

const loom = await initLoom();

// Load model from JSON (architecture + weights all at once!)
const modelJSON = await fetch("test.json").then((r) => r.json());
const network = loom.LoadModelFromString(
  JSON.stringify(modelJSON),
  "all_layers_test"
);

// That's it! All 16 layers, weights, biases loaded automatically
const input = new Array(10).fill(0).map(() => Math.random());
const [output, duration] = JSON.parse(
  network.ForwardCPU(JSON.stringify([input]))
);
console.log("Output:", output);
```

**Live Demo:** See `wasm/all_layers_test.html` for a complete working example that loads a 26.4KB model with 16 layers (Dense, Conv2D, Attention, RNN, LSTM) and runs inference in the browser!

### Manual Configuration (for building models from scratch)

```typescript
import { initLoom, ActivationType } from "@openfluke/welvet";

// Initialize the WASM module
const loom = await initLoom();

// Create a neural network: 784 inputs → 392 hidden → 10 outputs
const network = loom.NewNetwork(784, 1, 1, 2);

// Configure layers using registry-based initialization
const layer0 = loom.CallLayerInit(
  "InitDenseLayer",
  JSON.stringify([784, 392, ActivationType.ReLU])
);
const layer1 = loom.CallLayerInit(
  "InitDenseLayer",
  JSON.stringify([392, 10, ActivationType.Sigmoid])
);

network.SetLayer(JSON.stringify([0, 0, 0, JSON.parse(layer0)]));
network.SetLayer(JSON.stringify([0, 0, 1, JSON.parse(layer1)]));

// Forward pass
const input = new Array(784).fill(0).map(() => Math.random());
const resultJSON = network.ForwardCPU(JSON.stringify([input]));
const [output, duration] = JSON.parse(resultJSON);

console.log("Output:", output);
console.log("Inference time:", duration / 1e6, "ms");

// Training with the high-level API
const batches = [
  { Input: input, Target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] }, // Example: classify as "9"
];
const config = {
  Epochs: 10,
  LearningRate: 0.01,
  GradientClip: 1.0,
  LossType: "mse",
};

const trainingResult = network.Train(JSON.stringify([batches, config]));
const result = JSON.parse(trainingResult);
console.log("Final loss:", result.FinalLoss);
```

## 📚 API Reference

### Initialization

```typescript
interface InitOptions {
  wasmUrl?: string | URL;      // Custom WASM file location
  injectGoRuntime?: boolean;   // Include Go runtime (default: true)
}

const loom = await initLoom(options?);
```

### Creating Networks

```typescript
const network = loom.NewNetwork(
  inputSize: number,     // Input layer size
  gridRows: number,      // Grid rows (use 1 for simple networks)
  gridCols: number,      // Grid columns (use 1 for simple networks)
  layersPerCell: number  // Number of layers
);
```

### Layer Types

All layer types are created via the registry system using `CallLayerInit()`:

#### Dense (Fully-Connected) Layer

```typescript
const config = loom.CallLayerInit(
  "InitDenseLayer",
  JSON.stringify([
    inputSize: number,
    outputSize: number,
    activation: ActivationType,
  ])
);
```

#### Conv2D Layer

```typescript
const config = loom.CallLayerInit(
  "InitConv2DLayer",
  JSON.stringify([
    height: number,        // Input height
    width: number,         // Input width
    channels: number,      // Input channels
    filters: number,       // Number of output filters
    kernelSize: number,    // Kernel size (e.g., 3 for 3x3)
    stride: number,        // Stride (typically 1 or 2)
    padding: number,       // Padding (typically 0 or 1)
    activation: ActivationType,
  ])
);
```

#### Multi-Head Attention Layer

```typescript
const config = loom.CallLayerInit(
  "InitMultiHeadAttentionLayer",
  JSON.stringify([
    seqLength: number,     // Sequence length
    dModel: number,        // Model dimension
    numHeads: number,      // Number of attention heads
    activation: ActivationType,
  ])
);
```

#### RNN Layer

```typescript
const config = loom.CallLayerInit(
  "InitRNNLayer",
  JSON.stringify([
    inputSize: number,     // Input feature size
    hiddenSize: number,    // Hidden state size
    seqLength: number,     // Sequence length
    outputSize: number,    // Output size (hiddenSize * seqLength)
  ])
);
```

#### LSTM Layer

```typescript
const config = loom.CallLayerInit(
  "InitLSTMLayer",
  JSON.stringify([
    inputSize: number,     // Input feature size
    hiddenSize: number,    // Hidden/cell state size
    seqLength: number,     // Sequence length
    outputSize: number,    // Output size (hiddenSize * seqLength)
  ])
);
```

**Activation Types:**

```typescript
enum ActivationType {
  ReLU = 0, // Scaled ReLU (1.1x)
  Sigmoid = 1, // Logistic sigmoid
  Tanh = 2, // Hyperbolic tangent
  Softplus = 3, // Smooth ReLU
  LeakyReLU = 4, // ReLU with 0.1x negative slope
  Linear = 5, // Identity (no activation)
}
```

activation: ActivationType // ReLU, Sigmoid, Tanh, Linear
);

network.SetLayer(JSON.stringify([
gridRow, // Grid row index
gridCol, // Grid column index
layerIndex, // Layer index within cell
JSON.parse(config)
]));

````

#### Multi-Head Attention Layer

```typescript
const config = loom.InitMultiHeadAttentionLayer(
  dModel: number,       // Model dimension
  numHeads: number,     // Number of attention heads
  seqLength: number,    // Sequence length
  activation: ActivationType
);

network.SetLayer(JSON.stringify([0, 0, layerIndex, JSON.parse(config)]));
````

### Training Operations

#### Forward Pass

```typescript
const input = [0.1, 0.2, 0.3, 0.4];
const resultJSON = network.ForwardCPU(JSON.stringify([input]));
const [output, duration] = JSON.parse(resultJSON);
```

#### Backward Pass

```typescript
const gradOutput = new Array(outputSize).fill(0.01);
const backwardJSON = network.BackwardCPU(JSON.stringify([gradOutput]));
const [gradInput, duration] = JSON.parse(backwardJSON);
```

#### Update Weights

```typescript
const learningRate = 0.01;
network.UpdateWeights(JSON.stringify([learningRate]));
```

### Model Persistence

#### Load Model (The Easy Way - ONE LINE!)

```typescript
// Fetch model from server
const savedModel = await fetch("model.json").then((r) => r.json());

// Load complete network with ONE function call!
const network = loom.LoadModelFromString(
  JSON.stringify(savedModel),
  "model_name"
);

// Or from localStorage
const savedModel = JSON.parse(localStorage.getItem("my_model")!);
const network = loom.LoadModelFromString(
  JSON.stringify(savedModel),
  "model_name"
);
```

**That's it!** All layers, weights, biases, and configurations are automatically restored. No manual layer setup needed!

#### Save Model

```typescript
const modelJSON = network.SaveModelToString(JSON.stringify(["model_name"]));
const model = JSON.parse(JSON.parse(modelJSON)[0]);

// Store anywhere (localStorage, IndexedDB, etc.)
localStorage.setItem("my_model", JSON.stringify(model));
```

#### Save Model

```typescript
const modelJSON = network.SaveModelToString(JSON.stringify(["model_name"]));
const model = JSON.parse(JSON.parse(modelJSON)[0]);

// Store anywhere (localStorage, IndexedDB, backend API, etc.)
localStorage.setItem("my_model", JSON.stringify(model));
```

#### Cross-Platform Model Loading

The same JSON model file works across **all three platforms**:

```typescript
// JavaScript/WASM
const network = loom.LoadModelFromString(modelJSON, "model_id");
```

```python
# Python
network = welvet.load_model_from_string(model_json, "model_id")
```

```go
// Go
network, _ := nn.LoadModelFromString(modelJSON, "model_id")
```

See `examples/all_layers_validation.go` for a complete demo that generates test.json (26.4KB with 16 layers) and verifies all three platforms load it identically!

## 🤖 Transformer API

### Loading Models

```typescript
import { initLoom, createTransformerAPI } from "@openfluke/welvet";

// Initialize WASM
await initLoom();

// Create transformer API
const transformer = await createTransformerAPI();

// Load tokenizer from bytes
const tokenizerData = await fetch("models/SmolLM2-135M/tokenizer.json")
  .then((r) => r.arrayBuffer())
  .then((buf) => new Uint8Array(buf));

const tokResult = await transformer.loadTokenizer(tokenizerData);
console.log(`Tokenizer loaded: ${tokResult.vocab_size} tokens`);

// Load model from config and weights
const configData = await fetch("models/SmolLM2-135M/config.json")
  .then((r) => r.arrayBuffer())
  .then((buf) => new Uint8Array(buf));

const weightsData = await fetch("models/SmolLM2-135M/model.safetensors")
  .then((r) => r.arrayBuffer())
  .then((buf) => new Uint8Array(buf));

const modelResult = await transformer.loadModel(configData, weightsData);
console.log(
  `Model loaded: ${modelResult.num_layers} layers, ${modelResult.hidden_size} hidden size`
);
```

### Text Encoding/Decoding

```typescript
// Encode text to token IDs
const encodeResult = await transformer.encode("Hello world", true);
console.log(encodeResult.ids); // [1, 9906, 2088]

// Decode token IDs to text
const decodeResult = await transformer.decode([1, 9906, 2088], true);
console.log(decodeResult.text); // "Hello world"
```

### Text Generation

#### Blocking Generation

```typescript
const result = await transformer.generate(
  "The capital of France is",
  50, // maxTokens
  0.7 // temperature
);
console.log(result.generated_text);
```

#### Streaming Generation

```typescript
// Stream tokens one at a time
process.stdout.write("Generated: ");
for await (const token of transformer.generateStream(
  "Once upon a time",
  50, // maxTokens
  0.7 // temperature
)) {
  process.stdout.write(token); // Print each token as it's generated
}
console.log();
```

### Transformer API Reference

```typescript
interface TransformerAPI {
  // Load tokenizer from JSON bytes
  loadTokenizer(tokenizerData: Uint8Array): Promise<TokenizerLoadResult>;

  // Load model from config + weights bytes
  loadModel(
    configData: Uint8Array,
    weightsData: Uint8Array
  ): Promise<TransformerLoadResult>;

  // Encode text to token IDs
  encode(text: string, addSpecialTokens?: boolean): Promise<EncodeResult>;

  // Decode token IDs to text
  decode(
    tokenIds: number[],
    skipSpecialTokens?: boolean
  ): Promise<DecodeResult>;

  // Generate text (blocking)
  generate(
    prompt: string,
    maxTokens?: number,
    temperature?: number
  ): Promise<GenerateResult>;

  // Generate text (streaming)
  generateStream(
    prompt: string,
    maxTokens?: number,
    temperature?: number
  ): AsyncGenerator<string, void, unknown>;
}
```

#### Load Model (Legacy API)

````

### Runtime Introspection

#### Get All Methods

```typescript
const methodsJSON = network.GetMethods();
const methods = JSON.parse(methodsJSON);

methods.forEach((method) => {
  console.log(
    `${method.method_name}(${method.parameters.map((p) => p.type).join(", ")})`
  );
});
````

#### Check Method Availability

```typescript
if (network.HasMethod("ForwardGPU")) {
  const signature = network.GetMethodSignature(JSON.stringify(["ForwardGPU"]));
  console.log(signature);
}
```

#### List Method Names

```typescript
const names = JSON.parse(network.ListMethods());
console.log("Available methods:", names);
```

## 🎨 Activation Functions

```typescript
enum ActivationType {
  ReLU = 0, // Rectified Linear Unit
  Sigmoid = 1, // Sigmoid (logistic)
  Tanh = 2, // Hyperbolic tangent
  Linear = 3, // No activation (identity)
}
```

## 💡 Complete Examples

### MNIST-Style Classifier

```typescript
import { initLoom, ActivationType } from "@openfluke/welvet";

async function trainMNIST() {
  const loom = await initLoom();

  // Network: 784 → 128 → 64 → 10
  const network = loom.NewNetwork(784, 1, 1, 3);

  const layer0 = loom.InitDenseLayer(784, 128, ActivationType.ReLU);
  const layer1 = loom.InitDenseLayer(128, 64, ActivationType.ReLU);
  const layer2 = loom.InitDenseLayer(64, 10, ActivationType.Sigmoid);

  network.SetLayer(JSON.stringify([0, 0, 0, JSON.parse(layer0)]));
  network.SetLayer(JSON.stringify([0, 0, 1, JSON.parse(layer1)]));
  network.SetLayer(JSON.stringify([0, 0, 2, JSON.parse(layer2)]));

  // Training loop
  const epochs = 50;
  const learningRate = 0.01;

  for (let epoch = 0; epoch < epochs; epoch++) {
    // Your training data here
    const input = new Array(784).fill(0).map(() => Math.random());
    const target = new Array(10).fill(0);
    target[Math.floor(Math.random() * 10)] = 1;

    // Forward
    const [output] = JSON.parse(network.ForwardCPU(JSON.stringify([input])));

    // Compute loss (MSE)
    const loss =
      output.reduce((sum, val, i) => sum + Math.pow(val - target[i], 2), 0) /
      output.length;

    // Backward
    const gradOutput = output.map(
      (val, i) => (2 * (val - target[i])) / output.length
    );
    network.BackwardCPU(JSON.stringify([gradOutput]));

    // Update
    network.UpdateWeights(JSON.stringify([learningRate]));

    if (epoch % 10 === 0) {
      console.log(`Epoch ${epoch}: Loss = ${loss.toFixed(6)}`);
    }
  }

  // Save model
  const modelJSON = network.SaveModelToString(JSON.stringify(["mnist"]));
  localStorage.setItem("mnist_model", JSON.parse(modelJSON)[0]);
}
```

### XOR Problem

```typescript
import { initLoom, ActivationType } from "@openfluke/welvet";

const loom = await initLoom();
const network = loom.NewNetwork(2, 1, 1, 2);

// 2 → 4 → 1 (XOR needs hidden layer)
const layer0 = loom.InitDenseLayer(2, 4, ActivationType.ReLU);
const layer1 = loom.InitDenseLayer(4, 1, ActivationType.Sigmoid);

network.SetLayer(JSON.stringify([0, 0, 0, JSON.parse(layer0)]));
network.SetLayer(JSON.stringify([0, 0, 1, JSON.parse(layer1)]));

const trainingData = [
  { input: [0, 0], target: [0] },
  { input: [0, 1], target: [1] },
  { input: [1, 0], target: [1] },
  { input: [1, 1], target: [0] },
];

for (let epoch = 0; epoch < 1000; epoch++) {
  let totalLoss = 0;

  for (const sample of trainingData) {
    const [output] = JSON.parse(
      network.ForwardCPU(JSON.stringify([sample.input]))
    );
    const loss = Math.pow(output[0] - sample.target[0], 2);
    totalLoss += loss;

    const gradOutput = [2 * (output[0] - sample.target[0])];
    network.BackwardCPU(JSON.stringify([gradOutput]));
    network.UpdateWeights(JSON.stringify([0.1]));
  }

  if (epoch % 100 === 0) {
    console.log(`Epoch ${epoch}: Loss = ${(totalLoss / 4).toFixed(6)}`);
  }
}

// Test
trainingData.forEach((sample) => {
  const [output] = JSON.parse(
    network.ForwardCPU(JSON.stringify([sample.input]))
  );
  console.log(
    `${sample.input} → ${output[0].toFixed(4)} (expected ${sample.target[0]})`
  );
});
```

## 🌐 Browser Usage

### Via CDN (UMD)

```html
<!DOCTYPE html>
<html>
  <head>
    <script src="https://unpkg.com/@openfluke/welvet"></script>
  </head>
  <body>
    <script>
      (async () => {
        const { initLoom, ActivationType } = window.Welvet;
        const loom = await initLoom();

        const network = loom.NewNetwork(4, 1, 1, 1);
        console.log("LOOM ready!");
      })();
    </script>
  </body>
</html>
```

### Via ES Modules

```html
<!DOCTYPE html>
<html>
  <head>
    <script type="module">
      import {
        initLoom,
        ActivationType,
      } from "https://unpkg.com/@openfluke/welvet/dist/esm/index.js";

      const loom = await initLoom();
      const network = loom.NewNetwork(4, 1, 1, 1);
      console.log("LOOM ready!");
    </script>
  </head>
</html>
```

## ⚛️ Framework Integration

### React

```tsx
import { useEffect, useState } from "react";
import { initLoom, type LoomAPI } from "@openfluke/welvet";

function NeuralNetworkComponent() {
  const [loom, setLoom] = useState<LoomAPI | null>(null);
  const [prediction, setPrediction] = useState<number[] | null>(null);

  useEffect(() => {
    initLoom().then((api) => {
      setLoom(api);

      // Initialize network
      const network = api.NewNetwork(4, 1, 1, 2);
      const layer0 = api.InitDenseLayer(4, 8, 0); // ReLU
      const layer1 = api.InitDenseLayer(8, 2, 1); // Sigmoid

      network.SetLayer(JSON.stringify([0, 0, 0, JSON.parse(layer0)]));
      network.SetLayer(JSON.stringify([0, 0, 1, JSON.parse(layer1)]));

      // Make prediction
      const input = [0.5, 0.3, 0.2, 0.1];
      const [output] = JSON.parse(network.ForwardCPU(JSON.stringify([input])));
      setPrediction(output);
    });
  }, []);

  if (!loom) return <div>Loading neural network...</div>;

  return (
    <div>
      <h2>Prediction: {prediction?.map((v) => v.toFixed(4)).join(", ")}</h2>
    </div>
  );
}
```

### Vue 3

```vue
<script setup lang="ts">
import { ref, onMounted } from "vue";
import { initLoom, type LoomAPI } from "@openfluke/welvet";

const loom = ref<LoomAPI | null>(null);
const output = ref<number[] | null>(null);

onMounted(async () => {
  const api = await initLoom();
  loom.value = api;

  const network = api.NewNetwork(2, 1, 1, 1);
  const layer = api.InitDenseLayer(2, 1, 1); // Sigmoid
  network.SetLayer(JSON.stringify([0, 0, 0, JSON.parse(layer)]));

  const [result] = JSON.parse(network.ForwardCPU(JSON.stringify([[0.5, 0.5]])));
  output.value = result;
});
</script>

<template>
  <div v-if="!loom">Loading...</div>
  <div v-else>
    <h2>Neural Network Output</h2>
    <pre>{{ output }}</pre>
  </div>
</template>
```

### Svelte

```svelte
<script lang="ts">
  import { onMount } from 'svelte';
  import { initLoom, type LoomAPI } from '@openfluke/welvet';

  let loom: LoomAPI | null = null;
  let result: number[] = [];

  onMount(async () => {
    loom = await initLoom();

    const network = loom.NewNetwork(3, 1, 1, 1);
    const layer = loom.InitDenseLayer(3, 2, 0); // ReLU
    network.SetLayer(JSON.stringify([0, 0, 0, JSON.parse(layer)]));

    const [output] = JSON.parse(network.ForwardCPU(JSON.stringify([[1, 2, 3]])));
    result = output;
  });
</script>

{#if !loom}
  <p>Loading neural network...</p>
{:else}
  <h2>Result: {result.join(', ')}</h2>
{/if}
```

## 🔧 Advanced Configuration

### Custom WASM Location

```typescript
const loom = await initLoom({
  wasmUrl: "/custom/path/loom.wasm",
});
```

### Skip Go Runtime Injection

```typescript
// Useful if you're loading Go runtime separately
const loom = await initLoom({
  injectGoRuntime: false,
});
```

## 📊 Performance Tips

1. **Batch Processing** - Process multiple inputs together when possible
2. **Model Caching** - Save trained models to avoid retraining
3. **Layer Sizing** - Start with smaller layers and scale up as needed
4. **Learning Rate** - Tune learning rate for faster convergence (typically 0.001 - 0.1)
5. **Activation Functions** - ReLU often trains faster than Sigmoid/Tanh

## 🐛 Troubleshooting

### WASM fails to load

Ensure your server serves `.wasm` files with the correct MIME type:

```
Content-Type: application/wasm
```

### Module not found errors

Make sure to await the initialization:

```typescript
const loom = await initLoom(); // Don't forget await!
```

### JSON parsing errors

All network methods use JSON string parameters:

```typescript
// ✅ Correct
network.ForwardCPU(JSON.stringify([input]));

// ❌ Wrong
network.ForwardCPU(input);
```

## 🔗 Related Projects

- **Python Package**: [`welvet`](https://pypi.org/project/welvet/) - Python bindings for LOOM
- **Go Framework**: [LOOM](https://github.com/openfluke/loom) - Original Go implementation
- **Legacy Package**: [`@openfluke/portal`](https://github.com/openfluke/portal) - Previous generation framework

## 📄 License

Apache-2.0 © 2025 OpenFluke

## 🤝 Contributing

Contributions are welcome! Please see the [main repository](https://github.com/openfluke/loom) for guidelines.

## 📞 Support

- 🐛 [Report Issues](https://github.com/openfluke/loom/issues)
- 💬 [Discussions](https://github.com/openfluke/loom/discussions)
- 📖 [Documentation](https://github.com/openfluke/loom/tree/main/typescript)

---

**Built with ❤️ by the OpenFluke team**
