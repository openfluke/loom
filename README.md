# LOOM - Layered Omni-architecture Openfluke Machine

A high-performance GPU-accelerated neural network framework written in Go, featuring WebGPU compute shaders for parallel execution and WebAssembly export for browser deployment.

[![Go Version](https://img.shields.io/badge/Go-1.24+-blue.svg)](https://golang.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## Overview

Loom is a modern neural network framework that combines the simplicity of Go with the power of GPU acceleration via WebGPU. It supports multiple layer types, flexible grid-based architectures, and provides both CPU and GPU execution paths with automatic gradient computation. The framework can be compiled to WebAssembly for running neural networks directly in the browser.

## Key Features

### 🚀 GPU Acceleration

- **WebGPU Compute Shaders**: Native GPU acceleration using WGSL (WebGPU Shading Language)
- **Hybrid CPU/GPU**: Intelligent routing between CPU and GPU execution
- **Multi-layer Support**: Dense, Conv2D, Multi-Head Attention with GPU acceleration

### 🌐 WebAssembly Support

- **Browser Deployment**: Compile to WASM for client-side inference
- **Reflection-based API**: Automatic method exposure with 24+ discoverable functions
- **Runtime Introspection**: Query available methods, signatures, and parameters from JavaScript
- **Zero Dependencies**: Pure WASM + Go stdlib, no external libraries needed
- **Model Serialization**: Save/load models as JSON strings in the browser

### 🔗 C ABI (Foreign Function Interface)

- **Language Interop**: Call LOOM from C, C++, Rust, Python (ctypes/cffi), and more
- **Handle-based Management**: Safe object lifecycle with automatic cleanup
- **JSON Parameters**: Simple, language-agnostic API
- **Dynamic Method Calling**: Access all Network methods via reflection
- **Shared Library**: Build as .so/.dylib/.dll for system-wide integration

### 🧠 Neural Network Layers

- **Dense Layers**: Fully-connected layers with element-wise activations
- **Conv2D**: 2D convolutional layers with configurable kernels
- **Multi-Head Attention**: Transformer-style attention mechanism with GPU matrix operations
- **RNN**: Recurrent Neural Networks with BPTT (Backpropagation Through Time)
- **LSTM**: Long Short-Term Memory with gated cells

### 🏗️ Grid Architecture

- **Flexible Structure**: Organize layers in a 2D grid (rows × columns × layers per cell)
- **Mixed Layer Types**: Different layer types at different grid positions
- **Deep Networks**: Support for 100+ layers in a single network

### 📊 Activation Functions

- ScaledReLU, Sigmoid, Tanh, Softplus, LeakyReLU

### 🎯 Training & Evaluation

- **Built-in Training Loop**: `Train()` method with gradient clipping, loss tracking, and checkpointing
- **DeviationMetrics System**: Comprehensive evaluation tracking prediction accuracy across 7 deviation buckets
- **Sample-Level Tracking**: Identifies which specific samples fall into each performance category
- **Validation Integration**: Automatic periodic evaluation during training
- **Quality Scoring**: Standardized 0-100 score for model comparison
- **Metrics Persistence**: Save/load evaluation results to JSON

### 💾 Model Serialization

- Save and load model architectures and weights
- JSON-based model bundles with base64-encoded weights
- Compatible with model hosting systems

### 🔍 Runtime Introspection

- **Method Discovery**: Query all available network methods at runtime
- **Signature Inspection**: Get parameter types and return values for any method
- **JSON Metadata**: Export complete API documentation as JSON
- **WASM Integration**: Automatic exposure of Go methods to JavaScript

## Project Structure

```
loom/
├── nn/                  # Neural network package
│   ├── types.go         # Core types and structures
│   ├── forward.go       # Forward propagation (CPU/GPU)
│   ├── backward.go      # Backward propagation (CPU/GPU)
│   ├── gpu.go           # WebGPU initialization and shaders
│   ├── attention.go     # Multi-Head Attention implementation
│   ├── attention_gpu.go # MHA GPU kernels
│   ├── cnn.go           # Conv2D implementation
│   ├── rnn.go           # RNN implementation
│   ├── lstm.go          # LSTM implementation
│   ├── training.go      # Training loop with evaluation support
│   ├── evaluation.go    # DeviationMetrics evaluation system
│   ├── introspection.go # Runtime method discovery
│   ├── serialization.go # Model save/load
│   └── README.md        # Detailed package documentation
│
├── wasm/                # WebAssembly module
│   ├── main.go          # WASM wrapper with type conversion
│   ├── build.sh         # Build script for WASM compilation
│   ├── example.html     # Interactive browser demo
│   └── README.md        # WASM documentation and examples
│
├── cabi/                # C ABI for FFI
│   ├── main.go          # C foreign function interface
│   ├── simple_bench.c   # C benchmark program
│   ├── build.sh         # Build script for shared library
│   └── README.md        # C API reference and examples
│
├── fabric/              # Demo application
│   ├── main.go          # Interactive demo menu
│   ├── demos/           # Individual layer demos
│   └── examples/        # Benchmarks and tests
│
├── pods/                # GPU compute pods (primitives)
│   ├── ml_gemm.go       # Matrix multiplication
│   ├── ml_softmax_norm.go # Softmax and normalization
│   ├── primitives_scan.go # Parallel prefix scan
│   └── ...
│
└── detector/            # GPU device detection
    ├── detector.go      # Hardware capability detection
    └── detector_wasm.go # WASM stub (GPU N/A in browser)
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/openfluke/loom.git
cd loom

# Install dependencies
go mod download

# Build the demo application
cd fabric
go build
```

### Run Interactive Demo

```bash
cd fabric
./fabric
```

**Menu Options:**

- **Option 9**: Dense Neural Network demo
- **Option 10**: Conv2D demo
- **Option 11**: Multi-Head Attention demo
- **Option 12**: RNN demo
- **Option 13**: LSTM demo
- **Option 14**: **CPU vs GPU Comprehensive Benchmark** (recommended!)
- **Option 15**: **Model Serialization Demo** (file & string-based)

### Simple Dense Network Example

```go
package main

import (
    "fmt"
    "github.com/openfluke/loom/nn"
)

func main() {
    // Create a 4x4 grid with 5 layers per cell = 80 total layers
    network := nn.NewNetwork(
        4096,  // batch size / input size
        4,     // grid rows
        4,     // grid cols
        5,     // layers per cell
    )

    // Initialize GPU
    if err := network.InitGPU(); err != nil {
        panic(err)
    }
    defer network.ReleaseGPU()

    // Create input data
    input := make([]float32, 4096)
    for i := range input {
        input[i] = float32(i) * 0.001
    }

    // Forward pass on GPU
    output, gpuTime, err := network.ForwardGPU(input)
    if err != nil {
        panic(err)
    }

    fmt.Printf("GPU Forward time: %v\n", gpuTime)
    fmt.Printf("Output size: %d\n", len(output))
}
```

### Multi-Head Attention Example

```go
// Create network with MHA layer
batchSize := 32
seqLen := 256
dModel := 512
numHeads := 8

network := nn.NewNetwork(batchSize*seqLen*dModel, 1, 1, 1)
network.BatchSize = batchSize

// Configure MHA layer
config := nn.InitMultiHeadAttentionLayer(dModel, numHeads, seqLen, nn.ActivationScaledReLU)
network.SetLayer(0, 0, 0, config)

// Initialize GPU
network.InitGPU()
defer network.ReleaseGPU()

// Forward pass (GPU-accelerated Q/K/V projections)
input := make([]float32, batchSize*seqLen*dModel)
output, gpuTime, _ := network.ForwardGPU(input)

// Backward pass (GPU-accelerated gradient computation)
gradOutput := make([]float32, len(output))
gradInput, bwdTime, _ := network.BackwardGPU(gradOutput)
```

### Training with Automatic Evaluation

```go
// Prepare training data
trainBatches := []nn.Batch{
    {Inputs: batch1Inputs, Targets: batch1Targets},
    {Inputs: batch2Inputs, Targets: batch2Targets},
    // ... more batches
}

// Prepare validation data
valInputs := [][]float32{ /* validation inputs */ }
valTargets := []float64{ /* expected outputs */ }

// Configure training with automatic evaluation
config := &nn.TrainingConfig{
    Epochs:            10,
    LearningRate:      0.01,
    UseGPU:            true,
    GradientClip:      5.0,
    LossType:          "mse",
    EvaluateEveryN:    1,  // Evaluate every epoch
    ValidationInputs:  valInputs,
    ValidationTargets: valTargets,
}

// Train the model
result, err := network.Train(trainBatches, config)
if err != nil {
    panic(err)
}

// Training output:
// Epoch 1/10 - Avg Loss: 0.234
//   Running validation evaluation...
//   Validation Score: 76.5/100, Avg Deviation: 32.1%, Failures: 3/100
// ...

// Access evaluation metrics
fmt.Printf("Final Quality Score: %.2f/100\n", result.EvalMetrics.Score)
fmt.Printf("Average Deviation: %.2f%%\n", result.EvalMetrics.AverageDeviation)

// Print detailed distribution
result.EvalMetrics.PrintSummary()

// Save evaluation metrics
result.EvalMetrics.SaveMetrics("evaluation.json")

// Get worst predictions
worst := result.EvalMetrics.GetWorstSamples(5)
for _, pred := range worst {
    fmt.Printf("Sample #%d: Expected %.2f, Got %.2f, Deviation: %.1f%%\n",
        pred.SampleIndex, pred.ExpectedOutput, pred.ActualOutput, pred.Deviation)
}

// Analyze specific buckets
highPerformers := result.EvalMetrics.GetSamplesInBucket("0-10%")
fmt.Printf("High-performing samples: %v\n", highPerformers)
```

### Evaluation Output Example

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

=== Samples by Performance ===
   0-10%: 45 samples - [3 4 13 19 24] ... (40 more)
  10-20%: 18 samples - [1 8 15 21 22] ... (13 more)
   100%+: 3 samples - [17 42 89]
```

## WebAssembly (Browser Deployment)

Loom can be compiled to WebAssembly for running neural networks directly in the browser with zero dependencies.

### Building the WASM Module

```bash
cd wasm
./build.sh

# Serve the demo
python3 -m http.server 8080
# Open http://localhost:8080/example.html
```

### JavaScript API

The WASM module automatically exposes all Network methods via reflection:

```javascript
// Create a network
const network = NewNetwork(784, 1, 1, 2); // 784→392→10 architecture

// Initialize layers
const layer0Config = InitDenseLayer(784, 392, 0); // ReLU activation
const layer1Config = InitDenseLayer(392, 10, 1); // Sigmoid activation

network.SetLayer(JSON.stringify([0, 0, 0, JSON.parse(layer0Config)]));
network.SetLayer(JSON.stringify([0, 0, 1, JSON.parse(layer1Config)]));

// Run forward pass
const input = new Array(784).fill(0).map(() => Math.random());
const resultJSON = network.ForwardCPU(JSON.stringify([input]));
const output = JSON.parse(resultJSON)[0];

console.log("Output:", output); // [0.34, 0.67, 0.46, ...]

// Save model
const modelJSON = network.SaveModelToString(JSON.stringify(["my_model"]));
const model = JSON.parse(JSON.parse(modelJSON)[0]);

// Load model
const loadedNetwork = LoadModelFromString(JSON.stringify(model), "my_model");

// Introspection - discover all available methods
const methodsJSON = network.GetMethods();
const methods = JSON.parse(methodsJSON);
console.log("Available methods:", methods.length); // 24 methods

methods.forEach((method) => {
  console.log(
    `${method.method_name}(${method.parameters.map((p) => p.type).join(", ")})`
  );
});
```

### WASM Features

- ✅ **5.4MB binary** (includes full framework)
- ✅ **24+ methods** automatically exposed via reflection
- ✅ **Runtime introspection** - query methods, signatures, parameters
- ✅ **Type conversion** - automatic JavaScript ↔ Go type mapping
- ✅ **Model persistence** - save/load as JSON strings (no file system)
- ✅ **CPU-only** - GPU support via WebGPU coming soon

See [wasm/README.md](wasm/README.md) for complete documentation and examples.

## C ABI (Foreign Function Interface)

Call LOOM from C, C++, Rust, Python (ctypes/cffi), and any language with C FFI support.

### Building the Shared Library

```bash
cd cabi

# Quick build (current platform)
./build.sh

# Multi-platform builds
./build_all.sh linux arm64          # Linux ARM64
./build_all.sh macos universal      # macOS Universal Binary
./build_all.sh windows x86_64       # Windows 64-bit
./build_all.sh android arm64        # Android ARM64
./build_all.sh ios xcframework      # iOS XCFramework

# Build all architectures for current platform
./build_all.sh all
```

**Supported Platforms**: Linux (x86_64, arm64, armv7, x86), macOS (x86_64, arm64, universal), Windows (x86_64, x86, arm64), Android (arm64, armv7, x86_64, x86), iOS (arm64, simulators, xcframework)

**Output**: All builds organized in `compiled/<platform>_<arch>/` with `.so`/`.dylib`/`.dll`, headers, and benchmark.

### C API Example

```c
#include <stdio.h>
#include <stdint.h>

extern char* Loom_NewNetwork(int, int, int, int, bool);
extern char* Loom_InitDenseLayer(int, int, int);
extern char* Loom_SetLayer(int64_t, int, int, int, char*);
extern char* Loom_Call(int64_t, char*, char*);
extern void Loom_Free(int64_t);
extern void Loom_FreeCString(char*);

int main() {
    // Create network (784→392→10)
    char* result = Loom_NewNetwork(784, 2, 1, 1, false);
    int64_t handle = extractHandle(result); // Parse JSON for handle
    Loom_FreeCString(result);

    // Initialize layers
    char* layer0 = Loom_InitDenseLayer(784, 392, 1); // ReLU
    Loom_SetLayer(handle, 0, 0, 0, layer0);
    Loom_FreeCString(layer0);

    char* layer1 = Loom_InitDenseLayer(392, 10, 0); // Linear
    Loom_SetLayer(handle, 1, 0, 0, layer1);
    Loom_FreeCString(layer1);

    // Forward pass
    char* input = "[[0.1, 0.2, ...]]"; // 784 values
    char* output = Loom_Call(handle, "ForwardCPU", input);
    printf("Output: %s\n", output);
    Loom_FreeCString(output);

    // Cleanup
    Loom_Free(handle);
    return 0;
}
```

Compile:

```bash
gcc -o my_program my_program.c -L./compiled/linux_x86_64 -lloom -Wl,-rpath,'$ORIGIN'
```

### Python Example (ctypes)

```python
import ctypes
import json

loom = ctypes.CDLL('./cabi/libloom.so')
loom.Loom_NewNetwork.restype = ctypes.c_char_p
loom.Loom_Call.restype = ctypes.c_char_p

# Create network
result = loom.Loom_NewNetwork(784, 2, 1, 1, False)
data = json.loads(result.decode('utf-8'))
handle = data['handle']

# Forward pass
input_json = json.dumps([[0.1] * 784])
output = loom.Loom_Call(handle, b"ForwardCPU", input_json.encode())
print(json.loads(output.decode('utf-8')))

# Cleanup
loom.Loom_Free(handle)
```

### Benchmark Results

From `simple_bench.c` (784→392→10 network, 100 iterations):

```
CPU Forward: 100 iterations in 36.93 ms (avg: 0.3693 ms/iter)
GPU Forward: 100 iterations in 296.38 ms (avg: 2.9638 ms/iter)
Speedup: 8.03x (CPU faster for small batches)
```

### C ABI Features

- ✅ **Multi-platform support** - Linux, macOS, Windows, Android, iOS
- ✅ **Cross-compilation** - Build for multiple architectures from a single machine
- ✅ **17MB shared library** - Includes full framework + CGO runtime
- ✅ **Handle-based management** - Safe object lifecycle with sync.Mutex
- ✅ **JSON parameters** - Language-agnostic API
- ✅ **Dynamic method calling** - Access all 24+ Network methods via reflection
- ✅ **Introspection** - List methods, get signatures, query object info
- ✅ **GPU support** - Enable/disable GPU acceleration at runtime
- ✅ **Model persistence** - Save/load as JSON strings

See [cabi/README.md](cabi/README.md) for complete API reference, multi-platform build instructions, and language bindings (Python, Rust, C++, etc.).

## Performance Benchmarks

Results from **Option 14** (CPU vs GPU Comprehensive Benchmark):

### Dense Layers ✅

- **Forward**: 0.81x speedup (GPU: 4.8ms vs CPU: 3.9ms)
- **Backward**: 0.19x speedup (GPU: 10.6ms vs CPU: 2.0ms)
- **Total**: 0.38x at batch=4096, 80 layers
- **Status**: Full GPU acceleration (overhead dominates at small batches)

### Multi-Head Attention ✅

- **Forward**: 1.04x speedup (GPU: 693ms vs CPU: 721ms)
- **Backward**: 1.08x speedup (GPU: 2.39s vs CPU: 2.58s)
- **Total**: **1.07x speedup** at batch=32, seq=256, dim=512
- **Status**: Hybrid GPU/CPU - Q/K/V projections on GPU, attention on CPU

### Conv2D ⚠️

- **Status**: GPU implementation has bugs, falls back to CPU
- **Total**: 1.02x at batch=32, 64x64 images

### RNN/LSTM ⚠️

- **Status**: CPU only (sequential operations incompatible with GPU parallelism)

_GPU: Intel Arc Graphics (MTL), Vulkan backend_

## Model Serialization

Save and load trained models with both file-based and string-based methods:

### File-Based Serialization

```go
// Save a single model
network.SaveModel("model.json", "my_model_v1")

// Load a single model
loadedNetwork, err := nn.LoadModel("model.json", "my_model_v1")

// Save multiple models in a bundle
models := map[string]*nn.Network{
    "model_a": networkA,
    "model_b": networkB,
}
nn.SaveBundle("models.json", models)

// Load bundle
bundle, err := nn.LoadBundle("models.json")
```

### String-Based Serialization (WASM/CABI)

Perfect for WebAssembly, FFI, network transfer, or embedded models:

```go
// Serialize to JSON string
jsonString, err := network.SaveModelToString("my_model_v1")

// Load from JSON string (no file system needed!)
loadedNetwork, err := nn.LoadModelFromString(jsonString, "my_model_v1")

// Bundle to string
bundle := &nn.ModelBundle{...}
jsonStr, err := bundle.SaveToString()

// Load bundle from string
bundle, err := nn.LoadBundleFromString(jsonString)
```

**WASM Integration Example:**

```go
//export LoadModelFromJSON
func LoadModelFromJSON(jsonPtr *byte, jsonLen int) *Network {
    jsonString := bytesToString(jsonPtr, jsonLen)
    network, _ := nn.LoadModelFromString(jsonString, "model_id")
    return network
}

// From JavaScript:
// const modelJSON = JSON.stringify(modelData);
// const network = loadModelFromJSON(modelJSON);
```

**Use Cases for String-Based Serialization:**

- ✅ WebAssembly (no file system access)
- ✅ CABI/FFI integration with C/C++/Rust
- ✅ REST APIs and network transfer
- ✅ Database storage (JSON columns)
- ✅ Embedding models in source code

**Model Format:**

```json
{
  "type": "modelhost/bundle",
  "version": 1,
  "models": [
    {
      "id": "my_model_v1",
      "cfg": {
        "batch_size": 32,
        "grid_rows": 4,
        "grid_cols": 4,
        "layers_per_cell": 5,
        "layers": [ ... ]
      },
      "weights": {
        "fmt": "jsonModelB64",
        "data": "eyJ0eXBlIjoiZmxvYXQzMi... (base64)"
      }
    }
  ]
}
```

## GPU Architecture

### WebGPU Compute Shaders

Loom uses WGSL (WebGPU Shading Language) for GPU compute:

- **Dense Forward/Backward**: Element-wise activation and gradient computation
- **MHA Matrix Ops**: `matmulGPU` and `matmulTransposeGPU` kernels
- **Optimizations**: Command batching, efficient buffer management

### GPU Status by Layer Type

| Layer Type | Forward GPU | Backward GPU | Status                           |
| ---------- | ----------- | ------------ | -------------------------------- |
| Dense      | ✅ Active   | ✅ Active    | Production ready                 |
| MHA        | ✅ Hybrid   | ✅ Hybrid    | Production ready (1.07x speedup) |
| Conv2D     | ⚠️ Buggy    | ⚠️ Buggy     | Falls back to CPU                |
| RNN        | ❌ CPU      | ❌ CPU       | Sequential nature                |
| LSTM       | ❌ CPU      | ❌ CPU       | Sequential nature                |

## Documentation

- [Neural Network Package](nn/README.md) - Detailed API documentation
- [Evaluation System](nn/EVALUATION_README.md) - DeviationMetrics comprehensive guide
- [Examples](fabric/examples/) - Code examples and benchmarks
- [Demos](fabric/demos/) - Interactive demonstrations

## Building from Source

```bash
# Build the library
go build ./nn

# Run tests
cd fabric/examples
go test -v

# Run benchmarks
cd fabric
go build
./fabric
# Select option 14 for comprehensive CPU vs GPU benchmark
```

## Requirements

- **Go**: 1.24 or higher
- **GPU**: WebGPU-compatible GPU (Vulkan, Metal, or D3D12)
- **OS**: Linux, macOS, or Windows

## Roadmap

### High Priority

- [ ] Fix Conv2D GPU shader bugs
- [ ] Optimize Dense GPU for small batches
- [ ] GPU softmax kernel for MHA

### Medium Priority

- [ ] Multi-GPU support
- [ ] FP16/FP32 mixed precision
- [ ] Parallel RNN alternatives (QRNN, SRU)

### Future Enhancements

- [ ] Batch normalization
- [ ] Dropout layers
- [ ] Model visualization tools

### Completed ✅

- [x] **Training Loop**: Built-in `Train()` method with gradient clipping and loss tracking
- [x] **DeviationMetrics Evaluation**: 7-bucket accuracy tracking with sample-level analysis
- [x] **Validation Integration**: Automatic periodic evaluation during training
- [x] **Metrics Persistence**: JSON save/load for evaluation results
- [x] **Multi-Head Attention**: GPU-accelerated with hybrid CPU/GPU execution (1.07x speedup)
- [x] **Model Serialization**: File and string-based save/load (WASM/FFI compatible)
- [x] **RNN/LSTM**: Full CPU implementation with BPTT
- [x] **Dense GPU**: Forward/backward with WebGPU compute shaders
- [x] **Optimizers**: SGD with momentum, gradient clipping, learning rate scheduling
- [x] **Loss Functions**: MSE, Cross-Entropy with softmax

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- WebGPU compute shader architecture
- Inspired by modern deep learning frameworks (PyTorch, TensorFlow)
- Built with Go's simplicity and performance

## Contact

For questions and support, please open an issue on GitHub.

---

**Made with ❤️ by Openfluke**
