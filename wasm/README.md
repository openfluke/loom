# LOOM WASM Module

WebAssembly bindings for the LOOM neural network framework, enabling neural network creation, training, and transformer inference directly in the browser with zero dependencies.

## 🚀 Quick Start

```bash
cd wasm
./build_wasm.sh          # Build main.wasm
python3 -m http.server 8080
# Open http://localhost:8080/grid_scatter_demo.html
```

## 🎉 NEW: Simple API

Streamlined functions for common operations with **cross-platform consistency**:

```javascript
// Create network from JSON
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

const [result, error] = network.Train(
  JSON.stringify([batches, trainingConfig])
);
console.log("Training complete! Final loss:", JSON.parse(result).FinalLoss);

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
  `Quality Score: ${metricsData.score}/100, Avg Deviation: ${metricsData.avg_deviation}%`
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

- `createNetworkFromJSON(jsonConfig)` - Create from JSON configuration
- `loadLoomNetwork(jsonString, modelID)` - Load saved model
- `network.ForwardCPU(inputJSON)` - Forward pass
- `network.BackwardCPU(gradientsJSON)` - Backward pass
- `network.Train(batchesJSON)` - Train network
- `network.SaveModelToString(idJSON)` - Save to JSON string
- `network.EvaluateNetwork(inputsJSON)` - Evaluate with metrics
- `network.UpdateWeights(lrJSON)` - Update weights

**Cross-Platform Consistency:**
The simple API matches Python, TypeScript, C#, and C - identical behavior and results!

See `grid_scatter_demo.html` and `grid_scatter_demo.js` for complete working examples.

## 🚀 What's New: Dynamic Method Exposure

**Every Network method automatically available in JavaScript!**

The WASM wrapper uses Go reflection to dynamically expose ALL `nn.Network` methods - no manual bindings required!

```javascript
// Create network from JSON
const network = createLoomNetwork(jsonConfig);

// All 27+ methods automatically available:
network.ForwardCPU(inputJSON);
network.BackwardCPU(gradientsJSON);
network.Train(batchesJSON);
network.SaveModelToString(idJSON);
network.GetMethodsJSON();
// ... and 20+ more!
```

## ✨ Features

- ✅ **Zero Manual Bindings**: All Network methods auto-exposed via reflection
- ✅ **27+ Methods**: Complete API including Train, Forward, Backward, SaveModel, LoadModel
- ✅ **JSON-Based Network Creation**: Build networks from JSON config (no Go code needed)
- ✅ **Full Training Support**: `Train(batches, config)` with automatic gradient computation
- ✅ **All Layer Types**: Dense, Conv2D, LSTM, RNN, MHA, Parallel, Grid Scatter, Softmax (10+ variants)
- ✅ **Grid Scatter Demo**: Multi-agent heterogeneous neural networks training in browser
- ✅ **Runtime Introspection**: Query available methods and signatures
- ✅ **Type Conversion**: Automatic JavaScript ↔ Go type conversion
- ✅ **6.4MB Binary**: Complete framework in single WASM module
- ✅ **Pure CPU**: All operations (GPU via WebGPU coming soon)

## 📝 API Overview

### Creating Networks from JSON

```javascript
// Define network architecture
const config = {
  input_size: 10,
  batch_size: 1,
  grid_rows: 1,
  grid_cols: 1,
  layers_per_cell: 3,
  layers: [
    {
      type: "dense",
      output_size: 8,
      activation: "relu",
    },
    {
      type: "dense",
      output_size: 4,
      activation: "relu",
    },
    {
      type: "dense",
      output_size: 2,
      activation: "sigmoid",
    },
  ],
};

// Create network (returns object with ALL methods)
const network = createLoomNetwork(JSON.stringify(config));
```

### Training Networks

```javascript
// Prepare training data
const batches = [
  {
    Input: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    Target: [1.0, 0.0],
  },
  // ... more samples
];

// Training configuration
const config = {
  Epochs: 100,
  LearningRate: 0.01,
  UseGPU: false,
  PrintEveryBatch: 0,
  GradientClip: 1.0,
  LossType: "mse",
  Verbose: false,
};

// Train the network
const result = JSON.parse(network.Train(JSON.stringify([batches, config])));

console.log("Initial Loss:", result[0].LossHistory[0]);
console.log("Final Loss:", result[0].FinalLoss);
console.log(
  "Improvement:",
  (
    ((result[0].LossHistory[0] - result[0].FinalLoss) /
      result[0].LossHistory[0]) *
    100
  ).toFixed(2) + "%"
);
```

### Forward Pass

```javascript
// Run inference
const input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
const result = JSON.parse(network.ForwardCPU(JSON.stringify([input])));

const output = result[0]; // Output array
const duration = result[1]; // Execution time (nanoseconds)

console.log("Predictions:", output);
```

### Method Introspection

```javascript
// List all available methods
const methods = JSON.parse(network.ListMethods())[0];
console.log("Available methods:", methods);
// Output: ["Activations", "BackwardCPU", "ForwardCPU", "GetMethodsJSON",
//          "InitializeWeights", "ListMethods", "SaveModelToString", "Train", ...]

// Get detailed method information
const methodsJSON = JSON.parse(network.GetMethodsJSON())[0];
const parsedMethods = JSON.parse(methodsJSON);

parsedMethods.forEach((method) => {
  console.log(
    `${method.method_name}(${method.parameters
      .map((p) => p.type)
      .join(", ")}) -> ${method.returns.join(", ")}`
  );
});
```

### Save/Load Models

```javascript
// Save model to JSON string
const modelJSON = JSON.parse(
  network.SaveModelToString(JSON.stringify(["my_model"]))
)[0];

// Store in localStorage
localStorage.setItem("loom_model", modelJSON);

// Load model later
const savedModel = localStorage.getItem("loom_model");
const loadedNetwork = createLoomNetwork(savedModel);

// Use loaded network immediately
const output = JSON.parse(loadedNetwork.ForwardCPU(JSON.stringify([input])))[0];
```

## 🎮 Interactive Demos

### test.html - Complete Neural Network Demo

**Features:**

- 🎨 Beautiful gradient UI with multiple test sections
- 🔧 JSON config editor for network architecture
- 🏋️ Training demo with pattern recognition
- 🤖 Grid Scatter multi-agent training
- 📊 Real-time loss tracking and predictions
- 💾 Method introspection and network info

**Test 1: Grid Scatter - Multi-Agent Coordination**

- 3 heterogeneous agents (Feature Extractor, LSTM, RNN)
- Binary classification task
- 800 epochs training in ~0.4 seconds
- 99.5% improvement (0.25 → 0.001 loss)
- 100% classification accuracy

**Test 2: Pattern Recognition Training**

- Learns to classify patterns: high values in first half vs second half
- Configurable epochs, learning rate, and sample count
- Real-time loss display
- Post-training validation

**Try it:**

```bash
cd wasm
python3 -m http.server 8080
# Open http://localhost:8080/test.html
```

### Example Output: Grid Scatter Training

```
🤖 Running Grid Scatter Multi-Agent Training...
Task: 3 agents learn to collaborate for binary classification

Architecture:
  Shared Layer → Grid Scatter (3 agents) → Decision
  Agent 0: Feature Extractor (ensemble of 2 dense)
  Agent 1: Transformer (LSTM)
  Agent 2: Integrator (RNN)

✅ Training complete!
Training time: 0.36 seconds
Initial Loss: 0.252357
Final Loss: 0.001175
Improvement: 99.53%

Final predictions:
Sample 0: [0.989, 0.011] → Class 0 (expected 0) ✓
Sample 1: [0.023, 0.977] → Class 1 (expected 1) ✓
Sample 2: [0.049, 0.951] → Class 1 (expected 1) ✓
Sample 3: [0.960, 0.040] → Class 0 (expected 0) ✓
```

## 🏗️ Architecture

### Dynamic Method Wrapping

The WASM module (`wasm/main.go`) uses reflection to expose methods:

1. **Network Creation**: `createLoomNetwork(jsonConfig)` builds network from JSON
2. **Method Discovery**: Uses `reflect.ValueOf(network)` to find all methods
3. **Dynamic Wrapping**: Each method wrapped with `js.FuncOf(methodWrapper)`
4. **Type Conversion**: Automatic JSON ↔ Go type conversion
5. **Result Serialization**: All results returned as JSON arrays

```go
// Pseudo-code of the wrapper
func createNetworkFromJSON(jsonConfig string) js.Value {
    network := nn.BuildNetworkFromJSON(jsonConfig)
    networkObj := js.Global().Get("Object").New()

    // Wrap EVERY method dynamically
    networkValue := reflect.ValueOf(network)
    for i := 0; i < networkValue.NumMethod(); i++ {
        method := networkValue.Type().Method(i)
        networkObj.Set(method.Name, js.FuncOf(methodWrapper(network, method.Name)))
    }

    return networkObj
}
```

### Key Components

- **`main.go`**: WASM entry point, reflection-based method exposure
- **`grid_scatter_demo.js`**: Multi-agent training demo
- **`test.html`**: Interactive UI with all demos
- **`build_wasm.sh`**: Build script
- **`nn/introspection.go`**: Method discovery and signature extraction

## 🔧 Available Network Methods

All 27+ methods automatically exposed:

**Core Operations:**

- `ForwardCPU([]float32)` - Forward pass
- `BackwardCPU([]float32)` - Backward pass
- `UpdateWeights(float32)` - Update weights with learning rate
- `Train([]TrainingBatch, *TrainingConfig)` - Full training loop

**Model Management:**

- `SaveModelToString(string)` - Export model as JSON
- `LoadModelFromString(string, string)` - Import model from JSON
- `InitializeWeights()` - Initialize random weights
- `ResetState()` - Reset RNN/LSTM hidden states

**Introspection:**

- `GetMethodsJSON()` - Get all method signatures
- `ListMethods()` - Get method names only
- `GetMethodSignature(string)` - Get specific method signature
- `HasMethod(string)` - Check if method exists

**Layer Operations:**

- `GetLayer(int, int, int)` - Get layer config
- `SetLayer(int, int, int, LayerConfig)` - Set layer config
- `TotalLayers()` - Get layer count
- `Activations()` - Get activation outputs

**And many more!** Use `network.ListMethods()` to see all available methods.

## 📊 Supported Layer Types

All layer types from the Go framework:

- **Dense**: Fully connected layers with 15+ activation functions
- **Conv2D**: 2D convolution (CPU implementation)
- **LSTM**: Long Short-Term Memory
- **RNN**: Recurrent Neural Network
- **Multi-Head Attention**: Transformer attention mechanism
- **Layer Norm**: Layer normalization
- **RMS Norm**: Root Mean Square normalization
- **SwiGLU**: Gated linear units
- **Softmax**: 10+ variants (standard, temperature, grid, spatial, etc.)
- **Parallel**: 4 combine modes (concat, add, avg, grid_scatter)

## 🎯 Grid Scatter: Multi-Agent Networks

**What makes this special:**

Grid Scatter enables **heterogeneous multi-agent neural networks** where each agent has a completely different architecture:

```javascript
{
  type: "parallel",
  combine_mode: "grid_scatter",
  grid_output_rows: 3,
  grid_output_cols: 1,
  branches: [
    {
      type: "parallel",
      combine_mode: "add",
      branches: [
        {type: "dense", output_size: 8, activation: "relu"},
        {type: "dense", output_size: 8, activation: "gelu"}
      ]
    },
    {type: "lstm", hidden_size: 8},
    {type: "rnn", hidden_size: 8}
  ]
}
```

**Key Features:**

- ✅ **Heterogeneous Architectures**: LSTM + RNN + Dense ensemble in same layer
- ✅ **Spatial Topology**: Explicit 2D/3D grid positioning
- ✅ **Emergent Specialization**: Agents learn complementary roles
- ✅ **Trainable**: Full gradient flow through all agents

**Real-world Applications:**

- Multi-robot coordination (heterogeneous robots)
- Hierarchical reinforcement learning
- Multi-agent game playing
- Distributed sensor networks
- Ensemble methods with architectural diversity

## 🚧 Current Limitations

- **CPU Only**: WebGPU support coming soon
- **No Transformer Inference**: Removed in this version (see separate transformer branch)
- **4GB Memory Limit**: Standard 32-bit WASM
- **Performance**: 2-3x slower than native Go

## 🔮 Future Enhancements

- [ ] WebGPU acceleration for GPU support
- [ ] Memory64 support (unlimited memory)
- [ ] Optimize binary size (tree shaking)
- [ ] Add transformer inference back
- [ ] Web Workers for parallel training
- [ ] Quantization support (int8/int4)
- [ ] Streaming model loading
- [ ] Performance profiling tools

## 🛠️ Building

```bash
cd wasm
./build_wasm.sh
```

**Output:**

- `main.wasm` (6.4MB) - Complete LOOM framework
- `wasm_exec.js` (17KB) - Go WASM runtime

**Requirements:**

- Go 1.21+ with WASM support
- Any modern browser with WebAssembly support

## 🌐 Browser Compatibility

**Requires WebAssembly support:**

- ✅ Chrome/Edge 57+
- ✅ Firefox 52+
- ✅ Safari 11+

**Tested on:**

- ✅ Firefox 120+ (Linux)
- ✅ Chrome 119+ (Linux, macOS, Windows)

## 📄 License

Apache License 2.0 - see [LICENSE](../LICENSE) file for details.

## Transformer Inference API

### JavaScript API

```javascript
// 1. Load tokenizer
const tokenizerData = new Uint8Array(
  await (
    await fetch("models/SmolLM2-135M-Instruct/tokenizer.json")
  ).arrayBuffer()
);
const tokResult = JSON.parse(LoadTokenizerFromBytes(tokenizerData));

// 2. Load transformer model
const configData = new Uint8Array(
  await (await fetch("models/SmolLM2-135M-Instruct/config.json")).arrayBuffer()
);
const weightsData = new Uint8Array(
  await (
    await fetch("models/SmolLM2-135M-Instruct/model.safetensors")
  ).arrayBuffer()
);
const modelResult = JSON.parse(
  LoadTransformerFromBytes(configData, weightsData)
);

// 3. Generate text
const result = JSON.parse(GenerateText("Once upon a time", 50, 0.7));
console.log(result.generated_text);
```

### Available Functions

- `LoadTokenizerFromBytes(tokenizerData)` - Load BPE tokenizer from tokenizer.json
- `LoadTransformerFromBytes(configData, weightsData)` - Load transformer from config + safetensors
- `EncodeText(text, addSpecialTokens)` - Tokenize text to token IDs
- `DecodeTokens(tokenIDs, skipSpecialTokens)` - Convert token IDs back to text
- `GenerateText(prompt, maxTokens, temperature)` - Generate text (auto-handles tokenization)
- `GenerateNextToken(tokenIDs, temperature)` - Generate single next token

## Neural Network API

### Building

```bash
cd wasm
./build.sh
```

This produces:

- `loom.wasm` (6.0MB) - The compiled WebAssembly binary with transformer support
- `wasm_exec.js` (17KB) - Go's WASM runtime

### Running Traditional NN Demos

```bash
./serve.sh  # Starts server on port 8080
# Open http://localhost:8080/example.html
# Open http://localhost:8080/all_layers_test.html
```

**Demos included:**

- `example.html` - Network creation, training, introspection
- `all_layers_test.html` - ✨ **Load complete models from JSON!**
- `inference.html` - 🚀 **Transformer text generation!**

### ✨ Model Loading (The Easy Way)

```javascript
// Load complete model from JSON string
const network = LoadModelFromString(modelJSONString, "model_id");

// That's it! Network has all layers + weights loaded
// Use it immediately:
const output = JSON.parse(network.ForwardCPU(JSON.stringify([inputData])))[0];

// Train it:
const batches = [{ Input: inputData, Target: targetData }];
const config = {
  Epochs: 10,
  LearningRate: 0.01,
  LossType: "mse",
};
network.Train(JSON.stringify([batches, config]));

// Save it:
const savedJSON = JSON.parse(
  network.SaveModelToString(JSON.stringify(["model_id"]))
)[0];
```

### Creating Networks from Scratch

```javascript
// Create a network: 784 input → 392 hidden → 10 output
const network = NewNetwork(784, 1, 1, 2);

// Use registry-based initialization for all layer types
const layer0Config = CallLayerInit(
  "InitDenseLayer",
  JSON.stringify([784, 392, 0]) // ReLU activation
);
const layer1Config = CallLayerInit(
  "InitDenseLayer",
  JSON.stringify([392, 10, 1]) // Sigmoid activation
);

// Apply configurations to network
network.SetLayer(JSON.stringify([0, 0, 0, JSON.parse(layer0Config)]));
network.SetLayer(JSON.stringify([0, 0, 1, JSON.parse(layer1Config)]));
```

### Runtime Introspection

Discover all available methods at runtime:

```javascript
// Get all methods with metadata
const methodsJSON = network.GetMethods();
const methods = JSON.parse(methodsJSON);

console.log(`Network has ${methods.length} methods`);

methods.forEach((method) => {
  const params = method.parameters
    .map((p) => `${p.name}: ${p.type}`)
    .join(", ");
  const returns = method.returns.join(", ");
  console.log(`${method.method_name}(${params}) -> ${returns}`);
});

// Check if specific method exists
if (network.HasMethod("ForwardCPU")) {
  const sig = network.GetMethodSignature(JSON.stringify(["ForwardCPU"]));
  console.log("Signature:", sig);
  // Output: "ForwardCPU([]float32) ([]float32, time.Duration)"
}

// List all method names
const names = JSON.parse(network.ListMethods());
console.log("Available:", names);
// Output: ["BackwardCPU", "BackwardGPU", "ForwardCPU", "ForwardGPU", ...]
```

### Forward Pass

```javascript
// Create input (784 values for MNIST-sized input)
const input = new Array(784).fill(0).map(() => Math.random());

// Run forward pass
const resultJSON = network.ForwardCPU(JSON.stringify([input]));
const result = JSON.parse(resultJSON);

const output = result[0]; // Output array (10 values)
const duration = result[1]; // Execution time (nanoseconds)

console.log("Output:", output);
console.log("Time:", duration / 1e6, "ms");
console.log("Mean:", output.reduce((a, b) => a + b) / output.length);
```

### Training Loop

```javascript
// Manual training (forward passes only)
for (let epoch = 0; epoch < 5; epoch++) {
  const input = new Array(784).fill(0).map(() => Math.random());
  const resultJSON = network.ForwardCPU(JSON.stringify([input]));
  const output = JSON.parse(resultJSON)[0];

  // Compute loss (MSE against zero)
  const loss = output.reduce((sum, val) => sum + val * val, 0) / output.length;

  console.log(`Epoch ${epoch + 1}: Loss = ${loss.toFixed(6)}`);

  // For full training, you would:
  // 1. Compute gradients from loss
  // 2. Call BackwardCPU with gradients
  // 3. Update weights (manual or via Train method)
}
```

### Save/Load Models

```javascript
// Save model to JSON string
const modelJSON = network.SaveModelToString(JSON.stringify(["my_model"]));
const model = JSON.parse(JSON.parse(modelJSON)[0]);

console.log("Model size:", JSON.stringify(model).length, "bytes");
console.log("Layers:", model.models[0].cfg.layers.length);

// Store in localStorage (persistent across page reloads)
localStorage.setItem("loom_model", JSON.stringify(model));

// Load model from localStorage
const savedModel = JSON.parse(localStorage.getItem("loom_model"));
const loadedNetwork = LoadModelFromString(
  JSON.stringify(savedModel),
  "my_model"
);

// Verify loaded model works
const testInput = new Array(784).fill(0).map(() => Math.random());
const output = JSON.parse(
  loadedNetwork.ForwardCPU(JSON.stringify([testInput]))
)[0];
console.log("Loaded model output:", output);
```

### Helper Functions

```javascript
// Create layer configurations
const denseConfigJSON = InitDenseLayer(784, 128, 0); // 0 = ScaledReLU
const denseConfig = JSON.parse(denseConfigJSON);

// Create a multi-head attention layer
const mhaConfigJSON = InitMultiHeadAttentionLayer(
  512, // dModel
  8, // numHeads
  32, // batchSize
  256 // seqLength
);
const mhaConfig = JSON.parse(mhaConfigJSON);
```

## Training Example

```javascript
// Create network
const network = NewNetwork(784, 1, 1, 2);

// Training parameters
const epochs = 10;
const learningRate = 0.01;
const batchSize = 32;

// Training data (simplified)
const trainData = generateTrainingData(); // Your data

for (let epoch = 0; epoch < epochs; epoch++) {
  let totalLoss = 0;

  for (let i = 0; i < trainData.length; i += batchSize) {
    const batch = trainData.slice(i, i + batchSize);

    // Forward pass
    batch.forEach((sample) => {
      const input = sample.input;
      const target = sample.target;

      // Forward
      const outputJSON = network.ForwardCPU(JSON.stringify([input]));
      const output = JSON.parse(outputJSON)[0];

      // Compute loss (MSE example)
      const loss =
        output.reduce(
          (sum, val, idx) => sum + Math.pow(val - target[idx], 2),
          0
        ) / output.length;
      totalLoss += loss;

      // Compute gradient
      const gradOutput = output.map(
        (val, idx) => (2 * (val - target[idx])) / output.length
      );

      // Backward
      network.BackwardCPU(JSON.stringify([gradOutput]));
    });
  }

  console.log(`Epoch ${epoch + 1}: Loss = ${totalLoss / trainData.length}`);
}
```

## Method Calling Convention

All methods follow this pattern:

```javascript
// Parameters are passed as JSON array
const params = [param1, param2, param3];
const paramsJSON = JSON.stringify(params);

// Call the method
const resultJSON = network.MethodName(paramsJSON);

// Parse results (returns are also JSON array)
const results = JSON.parse(resultJSON);
const result1 = results[0];
const result2 = results[1];
```

## Type Conversion

JavaScript types are automatically converted to Go types:

| JavaScript Type | Go Type                                              |
| --------------- | ---------------------------------------------------- |
| `number`        | `int`, `float32`, `float64`                          |
| `boolean`       | `bool`                                               |
| `string`        | `string`                                             |
| `Array`         | `[]T` (slice)                                        |
| `Object`        | `map[string]T` or `struct`                           |
| Custom integers | `LayerType`, `ActivationType` (automatic conversion) |

For complex types (structs), pass as objects:

```javascript
const config = {
  Type: 0, // LayerType (automatically converted to Go type)
  Activation: 1, // ActivationType
  Kernel: [1.0, 2.0], // []float32
  Bias: [0.1, 0.2], // []float32
};

network.SetLayer(JSON.stringify([0, 0, 0, config]));
```

The WASM wrapper handles:

- ✅ **Nil values**: JavaScript `null` → Go `nil` for optional fields
- ✅ **Custom types**: Type conversion for enums like `LayerType`, `ActivationType`
- ✅ **Nested structs**: Recursive conversion of complex objects
- ✅ **Slices**: Multi-dimensional arrays properly converted

## Demo Results

The included `example.html` demo successfully demonstrates:

### Network Creation

```
Network: 784 → 392 → 10
Layer 0: 307,328 weights initialized
Layer 1: 3,920 weights initialized
Total: 2 layers with real weights
```

### Forward Pass

```
Input: 784 random values [0, 1]
Output: [0.3456, 0.6710, 0.4669, 0.5165, 0.5758, 0.6556, 0.5595, 0.6136, 0.6537, 0.3036]
Range: [0.3036, 0.6710]
Mean: 0.5362
```

### Training (5 epochs)

```
Epoch 1/5: Loss = 0.2946
Epoch 2/5: Loss = 0.2401 ✓ (improving)
Epoch 3/5: Loss = 0.2857
Epoch 4/5: Loss = 0.3392
Epoch 5/5: Loss = 0.3121
```

### Model Serialization

```
Saved: 1,486 bytes
Loaded: Successfully restored network
Verified: Forward pass produces identical outputs
```

## Current Limitations

- **No KV Cache**: Transformer generation reprocesses full sequence each token (slow but correct)
- **No GPU Support**: WebGPU integration coming soon
- **CPU Only**: All operations run on the CPU
- **Performance**: WASM is 2-3x slower than native Go (but instant deployment!)
- **Memory Limit**: 4GB for standard WASM (32-bit addressing)
- **Binary Size**: 6.0MB (includes full framework + transformer support)

## Future Enhancements

- [ ] **KV Caching** for transformers (10-100x speedup)
- [ ] WebGPU integration for GPU acceleration
- [ ] Memory64 support (unlimited memory when Go supports it)
- [ ] Streaming model loading for large models
- [ ] Web Workers for parallel training
- [ ] Quantization (int8/int4) for smaller models
- [ ] Optimize binary size (tree shaking, compression)
- [ ] Model visualization tools
- [ ] Performance benchmarking tools

## Architecture

### Transformer Inference

The transformer inference system (`wasm/inference.go`) provides:

1. **Byte-based Loading**: Load models from `Uint8Array` (no file system needed)
2. **Pure Go BPE Tokenizer**: Complete tokenizer implementation in `/tokenizer` package
3. **Safetensors Support**: Direct loading of HuggingFace model weights
4. **Full Sequence Context**: Processes entire token sequence for proper attention
5. **Auto-architecture Detection**: Supports LLaMA, GPT-2, and other architectures

### Neural Network Architecture

The WASM module uses reflection to automatically wrap all public methods of the `nn.Network` struct:

1. **Introspection** (`nn/introspection.go`): Discovers methods and signatures via reflection
2. **Method Wrapper** (`main.go`): Dynamically wraps methods for JavaScript calls
3. **Type Conversion** (`convertParameter`): Converts between JavaScript and Go types
4. **Result Serialization** (`serializeResults`): Returns results as JSON arrays

This approach means:

- ✅ **Zero manual bindings** - new methods automatically available
- ✅ **24+ methods exposed** - all Network methods callable from JavaScript
- ✅ **Type-safe** - runtime validation with helpful error messages
- ✅ **Self-documenting** - introspection reveals complete API

## Examples

### Transformer Inference Demo

The `inference.html` demo showcases transformer text generation in the browser:

**Features:**

- 🚀 Load models from local files (downloaded via `huggingface-cli`)
- 🎨 Beautiful gradient UI with model selection cards
- ⚡ Real-time text generation with progress tracking
- 📊 Live statistics (tokens/sec, time elapsed)
- 🔧 Adjustable temperature and max tokens

**Example Output (SmolLM2-135M-Instruct):**

```
Prompt: "Once upon a time"
Generated: "hi

I'm excited to see what you come up with! Let me know if you have any"
```

### Traditional Neural Network Demos

See `example.html` for a complete interactive demo including:

- ✅ Network creation with layer initialization
- ✅ Method introspection (24 methods discovered)
- ✅ Forward/backward passes with real outputs
- ✅ Model save/load with localStorage
- ✅ Training workflow with loss tracking
- ✅ Console logging for debugging

**Try it now:**

```bash
# Transformer inference:
cd wasm
bash serve_wasm.sh
# Open http://localhost:8888/inference.html

# Traditional neural networks:
cd wasm
./serve.sh
# Open http://localhost:8080/example.html
# Open http://localhost:8080/all_layers_test.html
```

## Browser Compatibility

Requires WebAssembly support (all modern browsers):

- Chrome/Edge 57+
- Firefox 52+
- Safari 11+

Tested and working on:

- ✅ Firefox 120+ (Linux)
- ✅ Chrome 119+ (Linux, macOS, Windows)

## License

Apache License 2.0 - see [LICENSE](../LICENSE) file for details.
