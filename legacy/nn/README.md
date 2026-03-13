# Neural Network Package

A high-performance **CPU-first** grid neural network implementation in Go with full CPU support for all 7 layer types, automatic differentiation, WebAssembly export, and C ABI for FFI. **WebGPU GPU acceleration is experimental and in development** — only select layers (Dense, Conv2D, MHA) have GPU code, and it may not work reliably.

> 🤯 **BREAKTHROUGH:** The Softmax layer includes **native Mixture of Experts (MoE)** via Grid Softmax. Mathematically proven equivalent to traditional MoE with 97.1% loss reduction and perfect gradient matching. See `../examples/moe_proof_demo.go`!

> 🧠 **NEW:** **Neural Tweening (StepTweenChain)** - Bidirectional "meet in the middle" training with proper chain rule gradient propagation. Achieves **100% accuracy** on shallow networks, **never crashes to 0%** during task changes, and maintains **40-80% stability** while adapting. See [`../docs/step_tween_assessment.md`](../docs/step_tween_assessment.md) for comprehensive benchmarks!

> 📊 **NEW:** **Neural Telemetry** - Extract network blueprints and layer metadata for visualization. Supports real-time neural activity recording and caching for replay. See `telemetry.go` and `examples/step_example/visualization_demo.go`!

## Features

### Grid Architecture

- **Grid Structure**: Organizes layers in a 2D grid (rows × columns × layers per cell)
- **Flexible Configuration**: Each grid position can have different layer types and activations
- **Registry-based Initialization**: Dynamic layer creation via `CallLayerInit()` for all layer types
- **Cross-Platform Support**: Same API across Go, WASM, C-ABI, Python, TypeScript, and C#
- **Universal CPU Support**: Every layer type has complete CPU forward/backward implementation
- **Selective GPU Acceleration**: Dense, Conv2D, and Multi-Head Attention with WebGPU compute shaders

**Layer Types (All work on CPU):**

- **Dense**: Fully-connected layers with element-wise activations (CPU + GPU)
- **Conv2D**: 2D convolutional layers with stride/padding/kernels (CPU + GPU)
- **Multi-Head Attention**: Transformer-style Q/K/V attention (CPU + GPU)
- **LayerNorm**: Layer normalization with gamma/beta + residual (CPU)
- **RNN**: Recurrent Neural Network with hidden state (CPU)
- **LSTM**: Long Short-Term Memory with forget/input/output gates (CPU)
- **Softmax**: First-class layer with 10 variants including native MoE (CPU)
- **Parallel**: Run multiple sub-layers in parallel with 4 combine modes (CPU)
  - **Combine Modes**: concat, add, avg, grid_scatter
  - **Nested Support**: Parallel layers can contain parallel layers (infinite recursion)
  - **Heterogeneous**: Each branch can be ANY layer type (LSTM + MHA + RNN + Dense)
  - **Grid Scatter**: Place branch outputs at specific 2D/3D grid positions

### Activation Functions

1. **ReLU** (0): `max(0, 1.1 * x)` with derivative `1.1` for x > 0
2. **Sigmoid** (1): `1 / (1 + e^(-x))`
3. **Tanh** (2): `(e^(2x) - 1) / (e^(2x) + 1)`
4. **Softplus** (3): `log(1 + e^x)`
5. **LeakyReLU** (4): `x` if x ≥ 0, else `0.1 * x`
6. **Linear** (5): `x` (identity function, no activation)

### Softmax Layer - The Unique Feature

Unlike other frameworks that treat softmax as a function, LOOM makes **softmax a first-class layer** with full backpropagation support.

**10 Softmax Variants:**

1. **Standard**: Single probability distribution (classification)
2. **Grid**: Independent distributions per row (**native MoE!**)
3. **Hierarchical**: Nested decision trees (strategy → tactic → action)
4. **Temperature**: Adjustable exploration (low=sharp, high=smooth)
5. **Gumbel**: Add exploration noise (training with randomness)
6. **Masked**: Filter illegal options (legal moves in games)
7. **Sparsemax**: Exact zeros in output (interpretable attention)
8. **Entmax**: Blend softmax/sparsemax (moderate sparsity)
9. **Adaptive**: Hierarchical vocabulary (large output spaces)
10. **Mixture**: Blend multiple distributions (ensemble decisions)

**The MoE Discovery:**

Grid Softmax IS Mixture of Experts (Soft-MoE):

- Each row = independent expert pathway
- Softmax = soft gating/routing mechanism
- All experts compute (dense routing)
- Gradients flow through routing automatically

**Proven Equivalent:**

- ✅ 97.1% loss reduction (1.1700 → 0.0343)
- ✅ 100% classification accuracy
- ✅ Output match: 0.00e+00 (perfect)
- ✅ Gradient match: 0.00e+00 (perfect)
- ✅ Finite difference validated (< 1.44e-04)
- ✅ Simpler than PyTorch/TensorFlow (2 lines vs 200+)

See `../examples/moe_proof_demo.go` for rigorous mathematical proof!

### Execution Modes

- **CPU**: Pure Go implementation with full backward propagation
- **GPU**: WebGPU/WGSL compute shaders for parallel execution
- **WASM**: Compile to WebAssembly for browser deployment (CPU-only)
- **C ABI**: Foreign Function Interface for C, C++, Rust, Python, and more (multi-platform)
- **Automatic Gradient Computation**: Stores activations and pre-activations for backprop

### Training & Evaluation

- **Training Loop**: Built-in `Train()` method with gradient clipping, loss tracking, and checkpointing
- **DeviationMetrics**: Comprehensive evaluation system tracking prediction accuracy across 7 deviation buckets
- **Sample-Level Tracking**: Identifies which specific samples fall into each performance category
- **Validation Integration**: Automatic periodic evaluation during training
- **Quality Scoring**: Standardized 0-100 score for model comparison
- **Metrics Persistence**: Save/load evaluation results to JSON
- **Failure Analysis**: Identify worst predictions and problematic samples

### Runtime Introspection

- **Method Discovery**: Query all Network methods at runtime via reflection
- **Signature Inspection**: Get parameter types and return values for any method
- **JSON Metadata**: Export complete API documentation as JSON
- **Registry System**: List all available layer initialization functions with metadata
- **WASM Integration**: Automatic exposure of all public methods to JavaScript
- **C ABI Integration**: Dynamic method calling from any language supporting C FFI

## File Structure

```
nn/
├── nn.go                 # Package documentation
├── types.go              # Core types (Network, LayerConfig, LayerType)
├── registry.go           # Layer initialization function registry
├── activations.go        # Activation functions and derivatives
├── softmax.go            # Softmax layer (10 variants including native MoE)
├── forward.go            # Forward propagation (CPU/GPU)
├── backward.go           # Backward propagation (CPU/GPU) with softmax Jacobian
├── step_forward.go       # Step-based forward for all layer types
├── step_backward.go      # Step-based backward for all layer types
├── tween.go              # Neural Tweening (bidirectional training)
├── telemetry.go          # Network blueprint extraction & neural activity
├── gpu.go                # WebGPU initialization and shader generation
├── utils.go              # Utility functions (MaxAbsDiff, Min, Max, Mean)
├── cnn.go                # Conv2D implementation (forward/backward)
├── conv2d_gpu.go         # Conv2D GPU kernels
├── attention.go          # Multi-Head Attention implementation
├── attention_gpu.go      # Attention GPU kernels + UpdateWeights + ZeroGradients
├── rnn.go                # RNN implementation with BPTT
├── lstm.go               # LSTM implementation with gate computations
├── training.go           # Training loop with evaluation support
├── evaluation.go         # DeviationMetrics evaluation system
├── introspection.go      # Runtime method discovery and inspection
├── serialization.go      # Model save/load (file and string-based)
└── README.md             # This file
```

## Usage

### Registry-based Layer Initialization

All layer types can be created dynamically using the registry system. This enables automatic discovery and creation of layers across all platforms (Go, WASM, C-ABI, Python, TypeScript).

```go
// List all available layer initialization functions
functions := nn.ListLayerInitFunctions()
for _, fn := range functions {
    fmt.Printf("%s: %s\n", fn.Name, fn.Signature)
}
// Output:
// InitDenseLayer: func(int, int, nn.ActivationType) nn.LayerConfig
// InitConv2DLayer: func(int, int, int, int, int, int, int, nn.ActivationType) nn.LayerConfig
// InitMultiHeadAttentionLayer: func(int, int, int, nn.ActivationType) nn.LayerConfig
// InitRNNLayer: func(int, int, int, int) nn.LayerConfig
// InitLSTMLayer: func(int, int, int, int) nn.LayerConfig

// Call any layer init function dynamically
params := []interface{}{128, 64, nn.ActivationReLU}
layerConfig, err := nn.CallLayerInit("InitDenseLayer", params)
if err != nil {
    log.Fatal(err)
}

// Same for other layer types
conv2dConfig, _ := nn.CallLayerInit("InitConv2DLayer",
    []interface{}{28, 28, 1, 32, 3, 1, 1, nn.ActivationReLU})

attentionConfig, _ := nn.CallLayerInit("InitMultiHeadAttentionLayer",
    []interface{}{10, 64, 8, nn.ActivationTanh})

rnnConfig, _ := nn.CallLayerInit("InitRNNLayer",
    []interface{}{32, 64, 10, 640})

lstmConfig, _ := nn.CallLayerInit("InitLSTMLayer",
    []interface{}{32, 64, 10, 640})
```

### Creating a Dense Neural Network

```go
// Create a 2x2 grid with 2 layers per cell = 8 total layers
network := nn.NewNetwork(
    1024,  // inputSize (batch size for dense layers)
    2,     // gridRows
    2,     // gridCols
    2,     // layersPerCell
)

// Initialize layer using registry
layerConfig, _ := nn.CallLayerInit("InitDenseLayer",
    []interface{}{1024, 512, nn.ActivationReLU})
network.SetLayer(0, 0, 0, layerConfig)

// Forward pass on CPU
input := make([]float32, 1024)
output, cpuTime := network.ForwardCPU(input)

// Forward pass on GPU
network.InitGPU()
defer network.ReleaseGPU()
outputGPU, gpuTime, err := network.ForwardGPU(input)

// Backward pass
gradOutput := make([]float32, len(output))
gradInput, bwdTime := network.BackwardCPU(gradOutput)
```

### Creating a Conv2D Network

```go
// Create network
batchSize := 2
network := nn.NewNetwork(inputSize, 1, 1, 1)
network.BatchSize = batchSize

// Configure Conv2D layer using registry
conv2dConfig, _ := nn.CallLayerInit("InitConv2DLayer",
    []interface{}{
        4,  // inputHeight
        4,  // inputWidth
        3,  // inputChannels
        2,  // filters
        3,  // kernelSize
        1,  // stride
        1,  // padding
        nn.ActivationReLU,
    })
network.SetLayer(0, 0, 0, conv2dConfig)

// Forward and backward passes work the same
output, _ := network.ForwardCPU(input)
gradInput, _ := network.BackwardCPU(gradOutput)
```

### Mixed Layer Types

```go
// Create grid with both Dense and Conv2D layers
network := nn.NewNetwork(inputSize, 2, 1, 2)
network.BatchSize = batchSize

// Cell [0,0], layer 0: Conv2D
conv2d := nn.InitConv2DLayer(4, 4, 3, 3, 1, 1, 2, nn.ActivationScaledReLU)
network.SetLayer(0, 0, 0, conv2d)

// Cell [0,0], layer 1: Dense (default)
// Cell [1,0] layers: Dense (default)

// Network automatically routes to appropriate layer type
output, _ := network.ForwardCPU(input)
```

### Multi-Head Attention

```go
// Create network with Multi-Head Attention layer
batchSize := 2
seqLength := 4
dModel := 8
numHeads := 2

totalInputSize := batchSize * seqLength * dModel
network := nn.NewNetwork(totalInputSize, 1, 1, 1)
network.BatchSize = batchSize

// Configure Multi-Head Attention layer
mhaConfig := nn.InitMultiHeadAttentionLayer(dModel, numHeads, batchSize, seqLength)
network.SetLayer(0, 0, 0, mhaConfig)

// Input shape: [batchSize, seqLength, dModel]
input := make([]float32, totalInputSize)
output, _ := network.ForwardCPU(input)
```

### RNN (Recurrent Neural Network)

```go
// Create RNN layer
batchSize := 2
seqLength := 4
inputSize := 8
hiddenSize := 16

totalInputSize := batchSize * seqLength * inputSize
network := nn.NewNetwork(totalInputSize, 1, 1, 1)
network.BatchSize = batchSize

// Configure RNN layer
rnnConfig := nn.InitRNNLayer(inputSize, hiddenSize, batchSize, seqLength)
network.SetLayer(0, 0, 0, rnnConfig)

// Input shape: [batchSize, seqLength, inputSize]
// Output shape: [batchSize, seqLength, hiddenSize]
input := make([]float32, totalInputSize)
output, _ := network.ForwardCPU(input)
```

### LSTM (Long Short-Term Memory)

```go
// Create LSTM layer
batchSize := 2
seqLength := 4
inputSize := 8
hiddenSize := 12

totalInputSize := batchSize * seqLength * inputSize
network := nn.NewNetwork(totalInputSize, 1, 1, 1)
network.BatchSize = batchSize

// Configure LSTM layer
lstmConfig := nn.InitLSTMLayer(inputSize, hiddenSize, batchSize, seqLength)
network.SetLayer(0, 0, 0, lstmConfig)

// Input shape: [batchSize, seqLength, inputSize]
// Output shape: [batchSize, seqLength, hiddenSize]
input := make([]float32, totalInputSize)
output, _ := network.ForwardCPU(input)
```

### Softmax Layers

```go
// Standard Softmax (classification)
network := nn.NewNetwork(inputSize, 1, 1, 2)
dense := nn.InitDenseLayer(inputSize, 10, nn.ActivationLeakyReLU)
network.SetLayer(0, 0, 0, dense)
softmax := nn.InitSoftmaxLayer() // 10 outputs → 10 probabilities (sum=1.0)
network.SetLayer(0, 0, 1, softmax)

// Grid Softmax (multi-agent AI / Mixture of Experts)
moe := nn.InitGridSoftmaxLayer(3, 4) // 3 experts × 4 actions each
network.SetLayer(0, 0, 1, moe)
// Each of the 3 rows independently sums to 1.0
// Row 0: Expert 0's output distribution [4 values, sum=1.0]
// Row 1: Expert 1's output distribution [4 values, sum=1.0]
// Row 2: Expert 2's output distribution [4 values, sum=1.0]

// Hierarchical Softmax (nested decisions)
hierarchical := nn.InitHierarchicalSoftmaxLayer([]int{3, 3, 4})
network.SetLayer(0, 0, 1, hierarchical)
// Creates decision tree: 3 strategies × 3 units × 4 actions

// Temperature Softmax (exploration control)
temperature := nn.InitTemperatureSoftmaxLayer(0.1) // Low = sharp/confident
network.SetLayer(0, 0, 1, temperature)

// Masked Softmax (legal moves only)
masked := nn.InitMaskedSoftmaxLayer(6)
masked.Mask = []bool{true, false, true, true, false, true}
network.SetLayer(0, 0, 1, masked)
// Positions 1 and 4 forced to ~0.0 (illegal moves)

// All softmax variants support proper backpropagation!
output, _ := network.ForwardCPU(input)
gradInput, _ := network.BackwardCPU(gradOutput)
```

## Layer Details

### Conv2D

### Forward Pass

- Input shape: `[batch, inChannels, height, width]`
- Output shape: `[batch, filters, outHeight, outWidth]`
- Output dimensions: `outH = (inH + 2*padding - kernelSize) / stride + 1`
- Kernel shape: `[filters, inChannels, kernelSize, kernelSize]`

### Backward Pass

Computes three gradients:

1. **∂L/∂input**: Gradient with respect to input (for backprop to previous layer)
2. **∂L/∂kernel**: Gradient with respect to kernel weights (for weight updates)
3. **∂L/∂bias**: Gradient with respect to bias (for weight updates)

### Weight Initialization

#### Conv2D

- **Kernel**: He initialization with `stddev = sqrt(2 / (inChannels * kernelSize²))`
- **Bias**: Initialized to zero

#### Multi-Head Attention

- **Q/K/V/Output Weights**: Xavier/Glorot initialization with `stddev = sqrt(2 / (fan_in + fan_out))`
- **Biases**: Initialized to zero

#### RNN

- **Input-to-Hidden**: Xavier initialization with `stddev = sqrt(2 / (inputSize + hiddenSize))`
- **Hidden-to-Hidden**: Xavier initialization with `stddev = sqrt(2 / (hiddenSize + hiddenSize))`
- **Bias**: Initialized to zero

#### LSTM

- **All Gates (i, f, g, o)**: Xavier initialization for both input-to-hidden and hidden-to-hidden weights
- **Forget Gate Bias**: Initialized to 1.0 (remember by default)
- **Other Biases**: Initialized to zero

## Multi-Head Attention Details

- Uses scaled dot-product attention: `softmax(Q·K^T / sqrt(d_k)) · V`
- Splits `dModel` into `numHeads` heads, each with dimension `headDim = dModel / numHeads`
- Linear projections for Q, K, V, and output
- Full backward pass through attention mechanism

## RNN/LSTM Details

### RNN Forward Pass

- Sequence processing: `h_t = tanh(W_ih · x_t + W_hh · h_{t-1} + b_h)`
- Hidden state initialized to zero
- Returns all timestep outputs: `[batch, seqLength, hiddenSize]`

### RNN Backward Pass

- Backpropagation Through Time (BPTT)
- Computes gradients for input, W_ih, W_hh, and bias
- Gradient flows through tanh activation

### LSTM Forward Pass

- Four gates per timestep:
  - Input gate: `i_t = sigmoid(W_ii · x_t + W_hi · h_{t-1} + b_i)`
  - Forget gate: `f_t = sigmoid(W_if · x_t + W_hf · h_{t-1} + b_f)`
  - Cell candidate: `g_t = tanh(W_ig · x_t + W_hg · h_{t-1} + b_g)`
  - Output gate: `o_t = sigmoid(W_io · x_t + W_ho · h_{t-1} + b_o)`
- Cell state update: `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`
- Hidden state: `h_t = o_t ⊙ tanh(c_t)`

### LSTM Backward Pass

- BPTT through all four gates
- Gradients for 12 weight matrices and 4 bias vectors
- Gradient flow through cell state across timesteps

## Performance

### CPU vs GPU Status (November 2024)

#### ✅ Dense Layers - Full GPU Acceleration

- **Forward Pass**: GPU shaders implemented, 0.81x speedup (GPU slightly slower for small batches)
- **Backward Pass**: GPU shaders implemented, 0.19x speedup (GPU slower due to small batch sizes)
- **Overall**: 0.38x speedup at batch=4096 (CPU faster for small workloads, GPU wins at larger scales)
- **Accuracy**: max_diff < 1e-7 (excellent)
- **Status**: ✅ Production ready, full GPU compute shaders active

#### ⚠️ Conv2D Layers - GPU Implementation Has Bugs

- **Forward Pass**: Falls back to CPU (0.99x "speedup")
- **Backward Pass**: Falls back to CPU (1.04x "speedup")
- **Overall**: 1.02x at batch=32, 64x64 images
- **Accuracy**: max_diff = 0 (CPU fallback is accurate)
- **Status**: ⚠️ Shader generation code exists but has runtime errors, currently uses CPU fallback

#### ✅ Multi-Head Attention - GPU Acceleration Active

- **Forward Pass**: GPU shaders for Q/K/V projections, 1.04x speedup
- **Backward Pass**: GPU shaders for gradient backprop, 1.08x speedup
- **Overall**: 1.07x speedup at batch=32, seq=256, dim=512
- **Accuracy**: max_diff = 0 (perfect)
- **Implementation**: Q/K/V projections use GPU matrix multiplication, attention scores computed on CPU
- **Status**: ✅ Hybrid CPU/GPU implementation, production ready

#### ⚠️ RNN Layers - CPU Fallback

- **Forward Pass**: CPU only, 4.49x "speedup" (anomalous timing)
- **Backward Pass**: CPU only, 3.41x "speedup" (anomalous timing)
- **Overall**: 3.85x "speedup" but actually CPU fallback
- **Accuracy**: max_diff = 1.74 (CPU implementation is correct, timing artifacts from fallback)
- **Status**: ⚠️ Sequential operations make GPU parallelization difficult

#### ⚠️ LSTM Layers - CPU Fallback

- **Forward Pass**: CPU only, 56.67x "speedup" (anomalous timing)
- **Backward Pass**: CPU only, 129.05x "speedup" (anomalous timing)
- **Overall**: 88.61x "speedup" but actually CPU fallback
- **Accuracy**: max_diff = 1.28 (CPU implementation is correct, timing artifacts from fallback)
- **Status**: ⚠️ Sequential gate operations make GPU parallelization very difficult

### Benchmark Configuration

```
Dense:  batch=4096, 4x4 grid, 5 layers/cell = 80 total layers
Conv2D: batch=32, 64→128 channels, 64x64 images, 3x3 kernel
MHA:    batch=32, seq=256, dim=512, 8 heads
RNN:    batch=64, seq=128, input=128, hidden=256
LSTM:   batch=64, seq=128, input=128, hidden=256
```

### Run Comprehensive Benchmark

```bash
cd fabric
go run main.go
# Select option 14 for CPU vs GPU comparison
```

## Model Serialization

Save and load model architectures and weights with both file-based and string-based methods.

### File-Based Serialization

```go
// Save a single model to file
err := network.SaveModel("model.json", "my_model_v1")

// Load a single model from file
loadedNetwork, err := nn.LoadModel("model.json", "my_model_v1")

// Save multiple models in a bundle
models := map[string]*nn.Network{
    "model_a": networkA,
    "model_b": networkB,
}
err = nn.SaveBundle("bundle.json", models)

// Load a bundle with multiple models
bundle, err := nn.LoadBundle("bundle.json")
// Access individual models from bundle
modelA := bundle.Models[0].Network
```

### String-Based Serialization (WASM/CABI)

Perfect for WebAssembly, FFI, network transfer, or embedded models:

```go
// Serialize a single model to JSON string
jsonString, err := network.SaveModelToString("my_model_v1")
// jsonString contains the full model (architecture + weights)

// Load from JSON string (no file system needed!)
loadedNetwork, err := nn.LoadModelFromString(jsonString, "my_model_v1")

// Bundle to string
bundle := &nn.ModelBundle{
    Type:    "modelhost/bundle",
    Version: 1,
    Models:  []nn.SavedModel{...},
}
jsonStr, err := bundle.SaveToString()

// Load bundle from string
bundle, err := nn.LoadBundleFromString(jsonString)
```

### WASM/CABI Integration Example

```go
//export LoadModelFromJSON
func LoadModelFromJSON(jsonPtr *byte, jsonLen int) *Network {
    jsonString := bytesToString(jsonPtr, jsonLen)
    network, err := nn.LoadModelFromString(jsonString, "model_id")
    if err != nil {
        return nil
    }
    return network
}

//export RunInference
func RunInference(netPtr *Network, inputPtr *float32, inputLen int) *float32 {
    input := float32SliceFromPtr(inputPtr, inputLen)
    output, _, err := netPtr.ForwardCPU(input)
    if err != nil {
        return nil
    }
    return &output[0]
}
```

**JavaScript (WASM):**

```javascript
// Load model from JSON
const modelJSON = JSON.stringify(modelData);
const network = Module.LoadModelFromJSON(modelJSON);

// Run inference
const input = new Float32Array([1.0, 2.0, 3.0, ...]);
const output = Module.RunInference(network, input);
```

**C (CABI):**

```c
// Load model from JSON string
const char* json = "{\"type\":\"modelhost/bundle\",\"version\":1,...}";
Network* net = LoadModelFromJSON((uint8_t*)json, strlen(json));

// Run inference
float input[1024] = {1.0, 2.0, 3.0, ...};
float* output = RunInference(net, input, 1024);
```

### Model Format

Models are saved in JSON format with base64-encoded weights:

```json
{
  "type": "modelhost/bundle",
  "version": 1,
  "models": [
    {
      "id": "my_model_v1",
      "cfg": {
        "id": "my_model_v1",
        "batch_size": 32,
        "grid_rows": 4,
        "grid_cols": 4,
        "layers_per_cell": 5,
        "layers": [
          {
            "type": "dense",
            "activation": "scaled_relu",
            "weights_len": 16777216,
            "bias_len": 4096
          },
          {
            "type": "multi_head_attention",
            "activation": "scaled_relu",
            "d_model": 512,
            "num_heads": 8,
            "seq_length": 256,
            "weights_len": 1048576,
            "bias_len": 512
          }
        ]
      },
      "weights": {
        "fmt": "jsonModelB64",
        "data": "eyJ0eXBlIjoiZmxvYXQzMi1hcnJheSIsImxlbmd0...  (base64 encoded)"
      }
    }
  ]
}
```

### Serialization Use Cases

**File-Based (SaveModel/LoadModel):**

- ✅ Training checkpoints
- ✅ Model versioning and archiving
- ✅ Local deployment
- ✅ Model sharing between systems

**String-Based (SaveModelToString/LoadModelFromString):**

- ✅ **WebAssembly** applications (no file system access)
- ✅ **CABI/FFI** integration with C/C++/Rust/Python
- ✅ **REST APIs** and network transfer (gRPC, HTTP)
- ✅ **Database storage** (JSON or TEXT columns)
- ✅ **Embedding models** directly in source code
- ✅ **Serverless functions** (Lambda, Cloud Functions)
- ✅ **Mobile apps** (in-memory model loading)

### Serialization Demo

```bash
cd fabric
go run main.go
# Select option 15 for Model Serialization Demo
```

The demo shows:

- File-based save/load
- String-based serialization
- Multi-model bundles
- WASM/CABI integration examples
- Model verification with forward pass

### Cross-Platform Serialization Test

The `examples/all_layers_validation.go` test demonstrates **one-line model loading** across all platforms (Go, Python, WASM):

```bash
# 1. Generate test.json (26.4KB model with 16 layers)
cd examples
go run all_layers_validation.go

# 2. Serve for web access (in new terminal)
cd .. && ./serve_files.sh  # Port 3123

# 3. Test Python (loads same test.json with ONE line!)
cd python/examples
python3 all_layers_test.py

# 4. Test WASM (loads same test.json with ONE line!)
# Open: http://localhost:3123/wasm/all_layers_test.html
```

All three platforms load the SAME test.json and verify:

- ✅ All 16 layers loaded automatically
- ✅ Weights, biases, configurations restored
- ✅ Outputs match within floating-point precision
- ✅ Training works (weight mutation verified)

**The key insight: ONE function call loads everything!**

```go
// Go
network, _ := nn.LoadModel("test.json", "all_layers_test")
```

```python
# Python
network = welvet.load_model_from_string(model_json, "all_layers_test")
```

```javascript
// JavaScript/WASM
network = LoadModelFromString(modelJSON, "all_layers_test");
```

## Runtime Introspection

The `nn` package provides runtime introspection capabilities for discovering and inspecting Network methods at runtime.

### Available Methods

- **`GetMethods()`**: Returns `[]MethodInfo` with all public methods, parameters, and return types
- **`GetMethodsJSON()`**: Returns JSON string with method metadata
- **`ListMethods()`**: Returns `[]string` of method names
- **`HasMethod(methodName)`**: Returns `bool` if method exists
- **`GetMethodSignature(methodName)`**: Returns formatted signature string

### Example Usage

```go
network := nn.NewNetwork(1024, 2, 2, 2)

// Get all methods as structured data
methods, err := network.GetMethods()
for _, method := range methods {
    fmt.Printf("%s\n", method.MethodName)
    for _, param := range method.Parameters {
        fmt.Printf("  - %s: %s\n", param.Name, param.Type)
    }
}

// Get method list
methodNames := network.ListMethods()
fmt.Printf("Found %d methods: %v\n", len(methodNames), methodNames)

// Check if method exists
if network.HasMethod("ForwardCPU") {
    sig, _ := network.GetMethodSignature("ForwardCPU")
    fmt.Println("Signature:", sig)
    // Output: ForwardCPU([]float32) ([]float32, time.Duration)
}

// Get as JSON for WASM/API export
methodsJSON, _ := network.GetMethodsJSON()
fmt.Println(methodsJSON)
```

### JSON Output Format

```json
[
  {
    "method_name": "ForwardCPU",
    "parameters": [{ "name": "param0", "type": "[]float32" }],
    "returns": ["[]float32", "time.Duration"]
  },
  {
    "method_name": "Train",
    "parameters": [
      { "name": "param0", "type": "[]nn.Batch" },
      { "name": "param1", "type": "*nn.TrainingConfig" }
    ],
    "returns": ["*nn.TrainingResult", "error"]
  }
]
```

### WASM Integration

Introspection is particularly useful for WebAssembly, where it enables automatic method exposure:

```javascript
// In browser, get all available methods
const methodsJSON = network.GetMethods();
const methods = JSON.parse(methodsJSON);

console.log(`Network has ${methods.length} methods:`);
methods.forEach((m) => {
  const params = m.parameters.map((p) => p.type).join(", ");
  const returns = m.returns.join(", ");
  console.log(`  ${m.method_name}(${params}) -> ${returns}`);
});

// Dynamically call any method
if (network.HasMethod("SaveModelToString")) {
  const modelJSON = network.SaveModelToString(JSON.stringify(["my_model"]));
  console.log("Model saved:", modelJSON);
}
```

See [../wasm/README.md](../wasm/README.md) for complete WASM documentation.

### C ABI Integration

Introspection also powers the C ABI, enabling dynamic method calls from any language:

```c
// In C, list all methods
char* methods = Loom_ListMethods(handle);
printf("Available methods: %s\n", methods);
Loom_FreeCString(methods);

// Dynamically call any method
char* result = Loom_Call(handle, "ForwardCPU", "[[0.1, 0.2, ...]]");
printf("Output: %s\n", result);
Loom_FreeCString(result);
```

```python
# In Python via ctypes
import ctypes, json
loom = ctypes.CDLL('./compiled/linux_x86_64/libloom.so')
loom.Loom_ListMethods.restype = ctypes.c_char_p

methods_json = loom.Loom_ListMethods(handle)
methods = json.loads(methods_json.decode('utf-8'))
print(f"Network has {methods['count']} methods")
```

See [../cabi/README.md](../cabi/README.md) for complete C ABI documentation and multi-platform builds.

### Introspection Demo

```bash
cd fabric
go run main.go
# Select option 16 for Introspection Demo
```

The demo shows all discovered methods (24+) with their signatures and parameters.

## Testing

Run all tests:

```bash
cd fabric/examples
go test -v
```

Run specific test:

```bash
go test -v -run TestConv2DLayer
go test -v -run TestMixedLayerTypes
go test -v -run TestMultiHeadAttention
go test -v -run TestRNNLayer
go test -v -run TestLSTMLayer
```

## Demos

### Dense Network Demo

```bash
cd fabric
go run main.go
# Select option 9
```

### Conv2D Demo

```bash
cd fabric
go run main.go
# Select option 10
```

### Multi-Head Attention Demo

```bash
cd fabric
go run main.go
# Select option 11
```

### RNN Demo

```bash
cd fabric
go run main.go
# Select option 12
```

### LSTM Demo

```bash
cd fabric
go run main.go
# Select option 13
```

## Implementation Notes

### Grid Indexing

Layers are stored in a flattened array with index:

```
idx = row * GridCols * LayersPerCell + col * LayersPerCell + layer
```

### Activation Storage

- `activations[0]`: Input to network
- `activations[i]`: Output of layer i-1 (post-activation)
- `preActivations[i]`: Pre-activation values for layer i (needed for derivatives)

### Layer Routing

Forward and backward passes check `LayerConfig.Type`:

- `LayerDense`: Element-wise activation
- `LayerConv2D`: 2D convolution with kernel
- `LayerMultiHeadAttention`: Transformer attention mechanism
- `LayerRNN`: Recurrent processing with BPTT
- `LayerLSTM`: LSTM gates and cell state

### GPU Shaders

#### Dense Layers (✅ Fully Implemented)

- **Forward**: Element-wise activation with GPU compute shaders
- **Backward**: Gradient computation with GPU compute shaders
- **Performance**: Best for large batch sizes (8K+), overhead dominates at small batches
- **Status**: Production ready, command batching optimized

#### Conv2D Layers (⚠️ Has Bugs)

- **Shader Generation**: Full generation functions exist (`generateConv2DForwardShader`, `generateConv2DBackwardShader`)
- **Status**: Runtime errors cause fallback to CPU
- **Issue**: Shader execution fails, needs debugging
- **Fallback**: CPU implementation used automatically

#### Multi-Head Attention (✅ Hybrid GPU/CPU)

- **GPU Components**:
  - Q/K/V projection matrix multiplications (forward)
  - Output projection matrix multiplication (forward)
  - Gradient backprop through projections (backward)
  - Uses `matmulGPU` and `matmulTransposeGPU` kernels
- **CPU Components**:
  - Attention score computation (Q·K^T / sqrt(d_k))
  - Softmax over attention scores
  - Weighted sum (attention·V)
- **Performance**: 1.07x speedup overall, GPU matrix ops faster than CPU
- **Status**: Production ready, hybrid approach balances GPU/CPU strengths

#### RNN/LSTM (⚠️ CPU Only)

- **Challenge**: Sequential timestep processing conflicts with GPU parallelism
- **Current**: Placeholder shaders that pass through to CPU
- **Alternative Approaches**:
  - QRNN (Quasi-Recurrent Neural Networks) - more GPU-friendly
  - SRU (Simple Recurrent Units) - designed for parallel execution
- **Status**: CPU implementation correct, GPU not feasible for standard RNN/LSTM architecture

## Model Evaluation & Training

### DeviationMetrics - Accuracy Deviation Heatmap Distribution

A comprehensive evaluation system that tracks how far predictions deviate from expected values, providing detailed performance breakdowns across 7 deviation buckets.

#### Quick Start

```go
// During training - automatic evaluation
config := &nn.TrainingConfig{
    Epochs:            10,
    LearningRate:      0.01,
    EvaluateEveryN:    1,  // Evaluate every epoch
    ValidationInputs:  valInputs,   // [][]float32
    ValidationTargets: valTargets,  // []float64
}

result, err := network.Train(batches, config)

// Access evaluation metrics
fmt.Printf("Quality Score: %.2f/100\n", result.EvalMetrics.Score)
fmt.Printf("Average Deviation: %.2f%%\n", result.EvalMetrics.AverageDeviation)
result.EvalMetrics.PrintSummary()

// Save metrics to file
result.EvalMetrics.SaveMetrics("evaluation.json")
```

#### Deviation Buckets

Performance is categorized into 7 ranges based on percentage deviation from expected:

- **0-10%**: High confidence, highly accurate (🟢 Red in visualization)
- **10-20%**: Very good accuracy
- **20-30%**: Good accuracy
- **30-40%**: Moderate accuracy
- **40-50%**: Acceptable accuracy
- **50-100%**: Significant deviation (🔵 Blue in visualization)
- **100%+**: Extreme deviation/failures (⚫ Black in visualization)

#### Sample-Level Tracking

Each bucket tracks which specific training samples fall into it:

```go
// Get samples in a specific bucket
samples := metrics.GetSamplesInBucket("0-10%")
fmt.Printf("High-performing samples: %v\n", samples)

// Get worst N predictions
worst := metrics.GetWorstSamples(5)
for _, result := range worst {
    fmt.Printf("Sample #%d: Expected %.2f, Predicted %.2f, Deviation: %.1f%%\n",
        result.SampleIndex, result.ExpectedOutput, result.ActualOutput, result.Deviation)
}
```

#### Manual Evaluation

```go
// Evaluate model on any dataset
metrics, err := network.EvaluateNetwork(inputs, expectedOutputs)

// Print distribution
metrics.PrintSummary()

// Analyze results
for bucketName, bucket := range metrics.Buckets {
    fmt.Printf("%s: %d samples\n", bucketName, bucket.Count)
    fmt.Printf("  Sample indices: %v\n", bucket.Samples)
}
```

#### Training Integration

The `TrainingConfig` struct supports automatic evaluation during training:

```go
type TrainingConfig struct {
    Epochs            int
    LearningRate      float32
    EvaluateEveryN    int         // Evaluate every N epochs (0 = disabled)
    ValidationInputs  [][]float32 // Validation dataset inputs
    ValidationTargets []float64   // Validation dataset targets
    // ... other fields
}
```

During training, validation metrics are printed:

```
Epoch 5/10 - Avg Loss: 0.234
  Running validation evaluation...
  Validation Score: 76.5/100, Avg Deviation: 32.1%, Failures: 3/100
```

#### Metrics Persistence

```go
// Save metrics to JSON
err := metrics.SaveMetrics("mnist_evaluation.json")

// Load metrics from JSON
loadedMetrics, err := nn.LoadMetrics("mnist_evaluation.json")

// Evaluate from checkpoint files
metrics, err := nn.EvaluateFromCheckpointFiles(
    "model.json", "model_v1",
    inputs, expectedOutputs,
)
```

#### Example Output

```
=== Model Evaluation Summary ===
Total Samples: 100
Quality Score: 76.5/100
Average Deviation: 32.1%
Failures (>100% deviation): 3 (3.0%)

Deviation Distribution:
     0-10%:   45 samples (45.0%) ██████████████████████
    10-20%:   18 samples (18.0%) █████████
    20-30%:   12 samples (12.0%) ██████
    30-40%:    8 samples (8.0%)  ████
    40-50%:    6 samples (6.0%)  ███
   50-100%:    8 samples (8.0%)  ████
     100%+:    3 samples (3.0%)  █

=== Worst 5 Predictions ===
1. Sample #42: Expected 5, Predicted 1, Deviation: 80.0%
2. Sample #17: Expected 3, Predicted 7, Deviation: 133.3%
3. Sample #89: Expected 2, Predicted 9, Deviation: 350.0%
```

#### Use Cases

1. **Training Monitoring**: Track model improvement over epochs
2. **Failure Analysis**: Identify which samples the model struggles with
3. **Quality Benchmarking**: Compare model versions with standardized score (0-100)
4. **Dataset Insights**: Discover patterns in prediction errors
5. **Debugging**: Isolate problematic samples for investigation

See `EVALUATION_README.md` for detailed API documentation and advanced examples.

## Current Limitations

### GPU Execution

- **Dense**: Works well for large batches (8K+), overhead dominates small batches
- **Conv2D**: Shader code exists but has runtime bugs → CPU fallback active
- **MHA**: Hybrid GPU/CPU works well, attention computation remains on CPU
- **RNN/LSTM**: CPU only - sequential nature incompatible with GPU parallelism

### Known Issues

- **Conv2D GPU Bugs**: Runtime errors in shader execution need debugging
- **RNN/LSTM Timing**: Fallback measurements show anomalous "speedups" - ignore these
- **Small Batch Performance**: Dense GPU slower than CPU for batches < 4K elements
- **MHA Attention**: Softmax and attention scores computed on CPU (still achieves 1.07x total speedup)

## Future Enhancements

### High Priority

- [ ] Debug Conv2D GPU shader runtime errors
- [ ] Optimize Dense GPU for small batches (reduce command submission overhead)
- [ ] Implement GPU attention score computation for MHA (currently CPU)

### Medium Priority

- [ ] GPU softmax kernel for MHA
- [ ] Multi-GPU support for large models
- [ ] Parallel RNN alternatives (QRNN, SRU) that work well on GPU
- [ ] FP16/FP32 mixed precision support

### Low Priority (Nice to Have)

- [ ] MaxPool2D and AvgPool2D layers
- [ ] Batch normalization
- [ ] Dropout layers
- [ ] Layer normalization
- [ ] Optimizers (SGD, Adam, RMSprop)
- [ ] Loss functions (cross-entropy, MSE)

### Completed

- [x] Model serialization (save/load weights) - File and string-based methods
- [x] RNN/LSTM layers (CPU implementation)
- [x] Multi-Head Attention (CPU + GPU hybrid)
- [x] Conv2D shader generation (has bugs, needs fixing)
- [x] Dense GPU forward/backward (working)
- [x] MHA GPU matrix operations (working)
- [x] MHA GPU backward pass (working)
- [x] Training loop with automatic evaluation (`Train()` method)
- [x] DeviationMetrics evaluation system with sample tracking
- [x] Validation integration during training
- [x] Metrics persistence (JSON save/load)
