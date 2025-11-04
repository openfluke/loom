# LOOM - Layered Omni-architecture Openfluke Machine

A high-performance GPU-accelerated neural network framework written in Go, featuring WebGPU compute shaders for parallel execution.

[![Go Version](https://img.shields.io/badge/Go-1.24+-blue.svg)](https://golang.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## Overview

Loom is a modern neural network framework that combines the simplicity of Go with the power of GPU acceleration via WebGPU. It supports multiple layer types, flexible grid-based architectures, and provides both CPU and GPU execution paths with automatic gradient computation.

## Key Features

### 🚀 GPU Acceleration

- **WebGPU Compute Shaders**: Native GPU acceleration using WGSL (WebGPU Shading Language)
- **Hybrid CPU/GPU**: Intelligent routing between CPU and GPU execution
- **Multi-layer Support**: Dense, Conv2D, Multi-Head Attention with GPU acceleration

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

### 💾 Model Serialization

- Save and load model architectures and weights
- JSON-based model bundles with base64-encoded weights
- Compatible with model hosting systems

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
│   ├── serialization.go # Model save/load
│   └── README.md        # Detailed package documentation
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
    └── detector.go      # Hardware capability detection
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

- [ ] Optimizers (SGD, Adam, RMSprop)
- [ ] Loss functions (cross-entropy, MSE)
- [ ] Batch normalization
- [ ] Dropout layers
- [ ] Model visualization tools

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
