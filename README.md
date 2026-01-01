# LOOM - Layered Omni-architecture Openfluke Machine

A high-performance **CPU-first** neural network framework written in Go, with **experimental** WebGPU compute shaders for GPU acceleration (in development, only select layers supported). Features WebAssembly export for browser deployment. **Now with transformer inference support!**

> 🎉 **NEW:** Full transformer inference in browser WASM! SmolLM2-135M-Instruct successfully generates coherent text entirely in the browser with pure Go implementation.

> 🤯 **BREAKTHROUGH:** LOOM's Softmax layer includes **native Mixture of Experts (MoE)** via Grid Softmax - the same architecture used in GPT-4, Switch Transformer, and Mixtral. **Mathematically proven** equivalent with 97.1% loss reduction and perfect gradient matching. See `examples/moe_proof_demo.go` for rigorous proof!

> ⚡ **NEW:** **Grid Scatter Mode** - Place parallel branch outputs at **specific 2D/3D grid positions** instead of concatenating! Build multi-agent systems with heterogeneous architectures (LSTM + MHA + RNN + Dense in same layer), hierarchical RL with spatial decomposition, and ensemble methods with explicit topology. **Impossible in traditional neural networks!** See `examples/json_grid_scatter_demo.go` and `examples/json_grid_scatter_agents.go` for mind-bending examples.

> 🧠 **NEW:** **Neural Tweening (StepTweenChain)** - A paradigm shift for **real-time embodied AI**. Train and run simultaneously with all layers processing in parallel. Achieves **100% accuracy** on shallow networks, **never crashes to 0%** during task changes (maintains 40-80% while adapting), and provides **minimal decision latency**. Statistically validated with 100 runs per config showing **0.8-1.9% StdDev** (vs 4-10% for traditional methods). See [`docs/step_tween_assessment.md`](docs/step_tween_assessment.md) for comprehensive benchmarks across 19 tests!

[![Go Version](https://img.shields.io/badge/Go-1.24+-blue.svg)](https://golang.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/welvet.svg)](https://pypi.org/project/welvet/)
[![npm](https://img.shields.io/npm/v/@openfluke/welvet.svg)](https://www.npmjs.com/package/@openfluke/welvet)
[![NuGet](https://img.shields.io/nuget/v/Welvet.svg)](https://www.nuget.org/packages/Welvet/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![.NET](https://img.shields.io/badge/.NET-9.0+-512BD4.svg)](https://dotnet.microsoft.com/)

## Overview

Loom is a modern neural network framework that combines the simplicity of Go with the power of GPU acceleration via WebGPU. It supports multiple layer types, flexible grid-based architectures, and provides both CPU and GPU execution paths with automatic gradient computation. The framework can be compiled to WebAssembly for running neural networks and **transformer inference** directly in the browser.

**Example transformer output (SmolLM2-135M in browser):**

```
Prompt: "Once upon a time"
Output: "hi

I'm excited to see what you come up with! Let me know if you have any"
```

## Framework Comparison

**Loom** is a specialized, lightweight, embeddable AI framework. Unlike general-purpose research frameworks, it is designed for **native embedding** into Go applications, targeting edge devices, CLIs, and backend microservices.

### Capabilities Comparison Matrix

| Feature | **Loom** (Go) | **PyTorch** | **TensorFlow** | **GoMLX** | **Spago** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Runtime Dependency** | **None** (Binary) | Heavy (Pip) | Binary (Edge) | CGo / XLA | None |
| **Safetensors** | ✅ **Native** | ✅ | ✅ | ✅ | ❌ |
| **Neural Tweening** | ✅ **Hybrid Engine** | ❌ | ❌ | ❌ | ❌ |
| **LR Schedulers** | ✅ **7 Types** | ✅ | ✅ | ✅ | ⚠️ Basic |
| **Optimizers** | ✅ **3 (SGD/AdamW/RMSprop)** | ✅ Many | ✅ Many | ✅ | ✅ |
| **RNN / LSTM** | ✅ **Full Gate** | ✅ | ✅ | ✅ | ✅ |
| **Transformer (MHA)** | ✅ (Explicit) | ✅ | ✅ | ✅ | ✅ (BERT) |
| **SwiGLU** | ✅ **Native** | ✅ | ✅ | ✅ | ❌ |
| **Parallel / MoE** | ✅ **Structure** | ❌ (Manual) | ❌ (Manual) | ❌ | ❌ |
| **LayerNorm / RMSNorm** | ✅ **Native** | ✅ | ✅ | ✅ | ✅ |
| **Dynamic Arch Gen** | ✅ **Built-in** | ❌ | ❌ | ❌ | ❌ |
| **Step-Based Forward** | ✅ **Unique** | ❌ | ❌ | ❌ | ❌ |
| **K-Means Clustering** | ✅ **Parallel** | ❌ | ❌ | ❌ | ❌ |
| **WASM Training** | ✅ **Full** | ❌ | ❌ | ❌ | ❌ |
| **Cross-Lang ABI** | ✅ **Universal** | ❌ | ❌ | ❌ | ❌ |
| **Tokenizer** | ✅ **Pure Go** | ❌ (Rust/C++) | ❌ (C++) | ❌ | ❌ |

### Go Ecosystem Comparison

| Feature | **Loom** | **GoMLX** | **Gorgonia** | **Spago** | **Go-Deep** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Implementation** | Pure Go | CGo (XLA) | Pure Go + CGo | Pure Go | Pure Go |
| **Load Safetensors** | ✅ **Native** | ✅ | ❌ | ❌ | ❌ |
| **Hybrid Tweening** | ✅ **Unique** | ❌ | ❌ | ❌ | ❌ |
| **Softmax Variants** | ✅ **10 Types** | ⚠️ Standard | ⚠️ Standard | ⚠️ Standard | ⚠️ Standard |
| **Network Grafting** | ✅ **Unique** | ❌ | ❌ | ❌ | ❌ |
| **C-ABI (Polyglot)** | ✅ **Universal** | ❌ | ❌ | ❌ | ❌ |
| **WASM Training** | ✅ **Full** | ❌ (XLA) | ❌ | ❌ | ❌ |

**Verdict**: Loom is the only Pure Go framework capable of loading and running modern **Llama-style LLMs** (Safetensors + SwiGLU + MHA) without CGo. Choose Loom for pure Go-native embedding, single-binary deployment, or to experiment with the Neural Tweening training paradigm.

For detailed comparison, see [`docs/loom_assessment_comparison.md`](docs/loom_assessment_comparison.md).

## Key Features

### 🚀 GPU Acceleration (Experimental - Untested)

- **WebGPU Compute Shaders**: Native GPU acceleration using WGSL (WebGPU Shading Language) - _code exists but untested_
- **Hybrid CPU/GPU**: Intelligent routing between CPU and GPU execution - _primarily Dense layer only_
- **CPU-First Focus**: All layers work reliably on CPU with full backward pass; GPU is experimental side feature

### 🌐 WebAssembly Support

- **Browser Deployment**: Compile to WASM for client-side inference
- **🚀 Transformer Inference**: Run LLaMA, GPT-2, and other transformers entirely in browser
- **Pure Go Tokenizer**: Complete BPE tokenizer implementation (no Python dependencies)
- **Safetensors Loading**: Direct loading of HuggingFace model weights from bytes
- **Local Model Files**: Load models from local filesystem (downloaded via `huggingface-cli`)
- **Interactive UI**: Beautiful web interface with model selection and generation controls
- **Working Models**: SmolLM2-135M (✅), Pythia-70M/160M (✅)
- **Registry-based Layer Initialization**: Dynamic layer creation via `CallLayerInit()` for all layer types
- **Reflection-based API**: Automatic method exposure with 24+ discoverable functions
- **Runtime Introspection**: Query available methods, signatures, and parameters from JavaScript
- **Zero Dependencies**: Pure WASM + Go stdlib, no external libraries needed
- **Model Serialization**: Save/load models as JSON strings in the browser
- **Full Training Support**: Train networks with all layer types (Dense, Conv2D, Attention, LayerNorm, RNN, LSTM, Softmax) in browser
- **Simple API**: New `createNetworkFromJSON`, `loadLoomNetwork`, `forward`, `train`, `evaluate` functions
- **CPU-Only in Browser**: GPU/WebGPU code exists but is untested; all demos run on CPU

### 🔗 C ABI (Foreign Function Interface)

- **Language Interop**: Call LOOM from C, C++, Rust, Python (ctypes/cffi), and more
- **Simple API**: New streamlined functions - `CreateLoomNetwork`, `LoomForward`, `LoomTrain`, `LoomSaveModel`, `LoomLoadModel`, `LoomEvaluateNetwork`
- **Global Network Pattern**: Single active network, no handle management needed
- **JSON Parameters**: Simple, language-agnostic API
- **Registry-based Layer Creation**: Dynamic layer initialization for all layer types via `CallLayerInit()`
- **Dynamic Method Calling**: Access all Network methods via reflection (legacy API)
- **Shared Library**: Build as .so/.dylib/.dll for system-wide integration
- **Multi-Platform**: Linux, macOS, Windows, Android, iOS with cross-compilation support
- **Cross-Language Consistency**: Same API across Python, C#, TypeScript, and C/C++/Rust
- **CPU-First Design**: Reliable CPU execution; GPU code exists but untested

### 🧠 Neural Network Layers

**All layer types support full CPU implementation:**

- ✅ **Complete CPU Forward/Backward**: Every layer works on CPU with full gradient computation
- ✅ **GPU Acceleration (Selective)**: Dense, Conv2D, and Multi-Head Attention with WebGPU compute shaders
- ✅ **Registry System**: Dynamic layer initialization via `CallLayerInit()` across all platforms (Go, WASM, C-ABI, Python, TypeScript)
- ✅ **Automatic Differentiation**: Complete backpropagation through all layer types
- ✅ **Cross-Platform**: Works everywhere (Go, Python, TypeScript/Node.js, C#, browser WASM, C/C++/Rust via FFI)

**Supported Layer Types (All with full CPU support):**

- **Dense Layers**: Fully-connected layers with element-wise activations (CPU fully tested, GPU exists but untested)
- **Conv2D**: 2D convolutional layers with configurable kernels, stride, padding (CPU fully tested, GPU code exists)
- **Multi-Head Attention**: Transformer-style attention with Q/K/V projections (CPU fully tested, GPU code exists)
- **LayerNorm**: Layer normalization with learned gamma/beta parameters and residual connections (CPU)
- **RNN**: Recurrent Neural Networks with BPTT (Backpropagation Through Time) (CPU)
- **LSTM**: Long Short-Term Memory with forget/input/output gates (CPU)
- **Softmax**: First-class layer with 10 variants (CPU) - Standard, Grid, Hierarchical, Temperature, Gumbel, Masked, Sparsemax, Entmax, Adaptive, Mixture
- **Parallel**: Run multiple sub-layers in parallel with 4 combine modes (CPU) - concat, add, avg, **grid_scatter**
  - **Nested Support**: Parallel layers can contain parallel layers (infinite recursion)
  - **Heterogeneous Branches**: Each branch can be ANY layer type (LSTM + MHA + RNN + Dense in same layer!)
  - **Grid Scatter**: Place outputs at specific 2D/3D grid positions for spatial topology

**Performance:** CPU implementations are production-ready, tested, and reliable. GPU acceleration code exists (WebGPU shaders) but is untested/experimental - use at your own risk!

### 🎨 Softmax Layer - The Unique Feature

LOOM makes **softmax a first-class layer** (not just a function), enabling:

- **10 Built-in Variants**: Standard, Grid, Hierarchical, Temperature, Gumbel, Masked, Sparsemax, Entmax, Adaptive, Mixture
- **Use Anywhere**: Hidden layers OR output layers
- **Grid Softmax**: Independent probability distributions per row (perfect for multi-agent AI)
- **Native MoE**: Grid Softmax IS Mixture of Experts (mathematically proven!)
- **Serialization**: All variants save/load correctly

**MoE Proof:** `examples/moe_proof_demo.go` demonstrates:

- ✅ 97.1% loss reduction (1.1700 → 0.0343)
- ✅ Perfect output/gradient matching (0.00e+00 difference)
- ✅ 100% classification accuracy
- ✅ Validated with finite difference check
- ✅ Simpler than PyTorch/TensorFlow (2 lines vs 200+)

### 🏗️ Grid Architecture & Parallel Layers

- **Flexible Structure**: Organize layers in a 2D grid (rows × columns × layers per cell)
- **Mixed Layer Types**: Different layer types at different grid positions
- **Deep Networks**: Support for 100+ layers in a single network
- **Parallel Layers**: Run multiple heterogeneous branches simultaneously with 4 combine modes:
  - `concat` - Concatenate outputs sequentially (default)
  - `add` - Element-wise addition (all branches must have same output size)
  - `avg` - Element-wise average (all branches must have same output size)
  - `grid_scatter` - **Place outputs at specific 2D/3D grid positions** (NEW!)

**Grid Scatter Mode** enables impossible architectures:

- **Multi-Agent Systems**: Each agent (grid position) has different architecture (LSTM, MHA, RNN, Dense)
- **Hierarchical RL**: Strategy → Tactics → Actions decomposed spatially using grid depth
- **Ensemble Learning**: Diverse architectures at different spatial locations
- **Multi-Scale Processing**: Different resolutions in different grid layers
- **Nested Grid Scatter**: Grid scatter within grid scatter for hierarchical spatial decomposition

Example:

```json
{
  "type": "parallel",
  "combine_mode": "grid_scatter",
  "grid_output_rows": 2,
  "grid_output_cols": 2,
  "grid_output_layers": 1,
  "grid_positions": [
    { "branch_index": 0, "target_row": 0, "target_col": 0, "target_layer": 0 },
    { "branch_index": 1, "target_row": 0, "target_col": 1, "target_layer": 0 },
    { "branch_index": 2, "target_row": 1, "target_col": 0, "target_layer": 0 },
    { "branch_index": 3, "target_row": 1, "target_col": 1, "target_layer": 0 }
  ],
  "branches": [
    { "type": "lstm", "hidden_size": 10 },
    { "type": "mha", "num_heads": 4 },
    { "type": "rnn", "hidden_size": 10 },
    { "type": "dense", "output_size": 10 }
  ]
}
```

See `examples/json_grid_scatter_demo.go` and `examples/json_grid_scatter_agents.go` for complete examples!

### 📊 Activation Functions

Supported across all layer types and platforms:

- **ReLU** (0): Rectified Linear Unit with 1.1x scaling
- **Sigmoid** (1): Logistic sigmoid function
- **Tanh** (2): Hyperbolic tangent
- **Softplus** (3): Smooth approximation of ReLU
- **LeakyReLU** (4): ReLU with negative slope (0.1x for x < 0)
- **Linear** (5): Identity function (no activation)

### 🎯 Training & Evaluation

- **Built-in Training Loop**: `Train()` method with gradient clipping, loss tracking, and checkpointing
- **DeviationMetrics System**: Comprehensive evaluation tracking prediction accuracy across 7 deviation buckets
- **Sample-Level Tracking**: Identifies which specific samples fall into each performance category
- **Validation Integration**: Automatic periodic evaluation during training
- **Quality Scoring**: Standardized 0-100 score for model comparison
- **Metrics Persistence**: Save/load evaluation results to JSON
- **Cross-Platform Evaluation**: `EvaluateNetwork()` available in Go, Python, TypeScript, C#, and C

### ⚡ Stepping API - Fine-Grained Execution Control

**NEW:** Execute networks one step at a time with full control over input/output at each layer:

- **Step-by-Step Execution**: Process inputs incrementally instead of all at once
- **Stateful Processing**: Maintain layer states across multiple steps (perfect for LSTMs/RNNs)
- **Manual Gradient Control**: Apply gradients when YOU want, not automatically
- **Real-Time Training**: Update weights after each step for online learning
- **Cross-Platform**: Available in Go, Python, C#, TypeScript, and WASM

**Example (Python):**
```python
from welvet import create_network_from_json, StepState, apply_gradients

# Create network
config = {"batch_size": 1, "layers": [...]}
network = create_network_from_json(config)

# Initialize stepping state
state = StepState(input_size=4)

# Training loop
for step in range(100000):
    state.set_input([0.1, 0.2, 0.1, 0.3])
    state.step_forward()
    output = state.get_output()
    
    # Calculate gradients
    gradients = [output[i] - target[i] for i in range(len(output))]
    
    # Backward pass
    state.step_backward(gradients)
    
    # Update weights
    apply_gradients(learning_rate=0.01)
```

**Available in all platforms:**
- ✅ **Go**: `network.InitStepState()`, `network.StepForward()`, `network.StepBackward()`, `network.ApplyGradients()`
- ✅ **Python**: `StepState(size)`, `state.step_forward()`, `state.step_backward()`, `apply_gradients()`
- ✅ **C#**: `new StepState(size)`, `state.StepForward()`, `state.StepBackward()`, `Network.ApplyGradients()`
- ✅ **TypeScript**: `network.createStepState()`, `state.stepForward()`, `state.stepBackward()`, `network.ApplyGradients()`
- ✅ **WASM/Browser**: Same as TypeScript, works in browser!

See examples:
- **Go**: `examples/step_example/step_train_v3.go`
- **Python**: `python/examples/step_train_v3.py`
- **C#**: `csharp/examples/StepTrainV3.cs`
- **TypeScript**: `typescript/example/step_train_v3.ts`
- **WASM**: `wasm/step_example.html`


### 🌍 Cross-Platform API Consistency

**All platforms now share the same simple API:**

| Function       | Go                       | Python                       | TypeScript/JS             | C#                      | C/C++/Rust              |
| -------------- | ------------------------ | ---------------------------- | ------------------------- | ----------------------- | ----------------------- |
| Create Network | `BuildNetworkFromJSON()` | `create_network_from_json()` | `createNetworkFromJSON()` | `CreateLoomNetwork()`   | `CreateLoomNetwork()`   |
| Forward Pass   | `ForwardCPU()`           | `forward_simple()`           | `forward()`               | `LoomForward()`         | `LoomForward()`         |
| Train          | `Train()`                | `train_simple()`             | `train()`                 | `LoomTrain()`           | `LoomTrain()`           |
| Save Model     | `SaveModelToString()`    | `save_model_simple()`        | `saveModel()`             | `LoomSaveModel()`       | `LoomSaveModel()`       |
| Load Model     | `LoadModelFromString()`  | `load_model_simple()`        | `loadLoomNetwork()`       | `LoomLoadModel()`       | `LoomLoadModel()`       |
| Evaluate       | `EvaluateNetwork()`      | `evaluate_network_simple()`  | `evaluate()`              | `LoomEvaluateNetwork()` | `LoomEvaluateNetwork()` |

**Verified identical behavior:**

- ✅ Same training results (99.3-99.5% improvement, 100/100 quality score)
- ✅ Bit-for-bit identical predictions after save/load (0.00 difference)
- ✅ Same evaluation metrics (7-bucket deviation distribution)
- ✅ Same model serialization format (~25-26KB JSON)

See platform-specific demos:

- **Python**: `python/examples/grid_scatter_demo.py`
- **TypeScript**: `typescript/example/grid-scatter.ts`
- **JavaScript/WASM**: `wasm/grid_scatter_demo.js`
- **C#**: `csharp/examples/GridScatterDemo.cs`
- **C**: `cabi/simple_bench.c`

### 💾 Model Serialization

- Save and load model architectures and weights
- JSON-based model bundles with base64-encoded weights
- Compatible with model hosting systems

### � Pre-trained Model Import

- **Import HuggingFace Models**: Convert BERT, GPT-2, and other transformers to LOOM format
- **Full Transformer Support**: Multi-head attention, LayerNorm, residual connections, FFN
- **Verified Accuracy**: 54% cosine similarity with real BERT (weights working correctly!)
- **Easy Conversion**: `python3 model_conversion/convert_tiny.py` - select from BERT-Tiny, Mini, Small
- **Automatic Verification**: Built-in tools compare LOOM vs original model outputs
- See [`model_conversion/README.md`](model_conversion/README.md) for detailed guide

### �🔍 Runtime Introspection

- **Method Discovery**: Query all available network methods at runtime
- **Signature Inspection**: Get parameter types and return values for any method
- **JSON Metadata**: Export complete API documentation as JSON
- **WASM Integration**: Automatic exposure of Go methods to JavaScript

## Project Structure

```
loom/
├── nn/                  # Neural network package
│   ├── types.go         # Core types and structures
│   ├── registry.go      # Layer initialization function registry
│   ├── forward.go       # Forward propagation (CPU/GPU)
│   ├── backward.go      # Backward propagation (CPU/GPU)
│   ├── step_forward.go  # Step-based forward for all layer types
│   ├── step_backward.go # Step-based backward for all layer types
│   ├── tween.go         # Neural Tweening (bidirectional training)
│   ├── telemetry.go     # Network blueprint & neural activity
│   ├── gpu.go           # WebGPU initialization and shaders
│   ├── attention.go     # Multi-Head Attention implementation
│   ├── attention_gpu.go # MHA GPU kernels
│   ├── cnn.go           # Conv2D implementation
│   ├── conv2d_gpu.go    # Conv2D GPU kernels
│   ├── rnn.go           # RNN implementation
│   ├── lstm.go          # LSTM implementation
│   ├── training.go      # Training loop with evaluation support
│   ├── evaluation.go    # DeviationMetrics evaluation system
│   ├── introspection.go # Runtime method discovery
│   ├── serialization.go # Model save/load
│   ├── transformer.go   # Transformer model loading and inference
│   └── README.md        # Detailed package documentation
│
├── docs/                # Documentation
│   ├── README.md        # Documentation hub
│   └── step_tween_assessment.md  # Neural Tweening benchmarks (19 tests)
│
├── tokenizer/           # Pure Go BPE tokenizer
│   ├── bpe.go           # Byte Pair Encoding implementation
│   ├── tokenizer.go     # HuggingFace tokenizer.json loader
│   └── README.md        # Tokenizer documentation and examples
│
├── wasm/                # WebAssembly module
│   ├── main.go          # WASM wrapper with type conversion
│   ├── inference.go     # Transformer inference exports for WASM
│   ├── build.sh         # Build script for WASM compilation
│   ├── example.html     # Interactive browser demo
│   ├── inference.html   # Transformer inference demo
│   └── README.md        # WASM documentation and examples
│
├── cabi/                # C ABI for FFI
│   ├── main.go          # C foreign function interface
│   ├── transformer.go   # Transformer inference C exports
│   ├── simple_bench.c   # C benchmark program
│   ├── build.sh         # Build script for shared library
│   └── README.md        # C API reference and examples
│
├── python/              # Python package (welvet)
│   ├── pyproject.toml   # Python package configuration
│   ├── README.md        # Python package documentation
│   ├── src/welvet/      # Python bindings via ctypes
│   │   ├── __init__.py  # Package initialization
│   │   ├── utils.py     # High-level Python API
│   │   └── */           # Multi-platform C libraries
│   └── examples/        # Python examples
│       ├── test_transformer.py         # CLI inference example
│       └── transformer_web_interface.py # Web UI with streaming
│
├── model_conversion/    # Model import & pure Go inference
│   ├── README.md        # Conversion documentation
│   ├── requirements.txt # Python dependencies
│   ├── convert_tiny.py  # BERT/tiny model converter
│   ├── convert_model.py # General model converter
│   ├── serve_model_bytes.go    # Pure Go model serving
│   ├── web_interface.go        # Pure Go web interface
│   └── verify_bert_weights.py  # Weight verification tool
│
├── typescript/          # TypeScript/WASM package
│   ├── package.json     # npm package configuration
│   ├── README.md        # TypeScript package documentation
│   ├── src/             # TypeScript bindings
│   │   ├── index.ts     # Main WASM loader
│   │   ├── transformer.ts # Transformer API wrapper
│   │   └── types.ts     # TypeScript type definitions
│   └── examples/        # TypeScript examples
│       ├── transformer.ts   # Node.js inference example
│       └── transformer.html # Browser demo with streaming
│
├── csharp/              # C#/.NET package (Welvet)
│   ├── Welvet.csproj    # NuGet package configuration
│   ├── NativeMethods.cs # P/Invoke declarations (C-ABI)
│   ├── Network.cs       # High-level managed API
│   ├── Transformer.cs   # Transformer inference API (NEW!)
│   ├── Activation.cs    # Activation enum
│   ├── README.md        # C# package documentation
│   ├── runtimes/        # Native libraries per platform
│   └── examples/        # C# example programs
│       ├── TransformerTest.cs          # CLI inference example
│       └── TransformerWebInterface.cs  # Web UI with streaming
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

### Option A: Import Pre-trained Models

Convert and use pre-trained transformer models from HuggingFace:

```bash
# Install Python dependencies
cd model_conversion
pip install -r requirements.txt

# Convert BERT-Tiny (4MB, 2 layers)
python3 convert_tiny.py
# Select option 1 for BERT-Tiny

# Verify the conversion
python3 verify_bert_weights.py
# ✅ Expected: 54% similarity (weights working!)

# Test in Go
go run run_bert_tiny.go
```

See [`model_conversion/README.md`](model_conversion/README.md) for complete guide.

### Option B: Run Interactive Demo

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

### ✨ Model Serialization - Save & Load Complete Networks

**The Easy Way - One Function Call:**

```go
// Save a trained model (includes all weights and configuration)
err := network.SaveModel("model.json", "my_model")

// Load it back - ONE LINE! Everything restored automatically
loadedNet, err := nn.LoadModel("model.json", "my_model")
// Done! All layers, weights, and configuration loaded

// Or use strings (great for APIs/databases)
jsonString, err := network.SaveModelToString("my_model")
loadedNet, err := nn.LoadModelFromString(jsonString, "my_model")
```

**Works everywhere:**

- ✅ **Go**: `nn.LoadModel()` / `nn.LoadModelFromString()`
- ✅ **Python**: `welvet.load_model_from_string(json_str, "model_id")`
- ✅ **JavaScript/WASM**: `LoadModelFromString(jsonString, "model_id")`
- ✅ **C#/.NET**: `Network.LoadFromString(jsonString, "model_id")`
- ✅ **C/C++/Rust**: `Loom_LoadModel(jsonCStr, modelID)`

**Example Test:** See `examples/all_layers_validation.go` for a complete demo with all 6 layer types + 10 softmax variants (16 layers total)

```bash
cd examples
go run all_layers_validation.go
# Creates: test.json, inputs.txt, outputs.txt
# Tests: save → load → verify → train
```

### 🤖 Transformer Inference - Run LLMs in Browser or Python

Run pretrained transformer models like SmolLM2-135M entirely client-side:

**Python (Server or CLI):**

```python
import welvet

# Load tokenizer and model
tokenizer = welvet.load_tokenizer_from_bytes(open("tokenizer.json", "rb").read())
model = welvet.load_transformer_from_bytes(
    open("config.json", "rb").read(),
    open("model.safetensors", "rb").read()
)

# Generate text with streaming
for token in welvet.generate_text_stream("The capital of France is", max_tokens=50):
    print(token, end="", flush=True)
```

**TypeScript/Browser (100% Client-Side):**

```typescript
import { initLoom, createTransformerAPI } from "@openfluke/welvet";

await initLoom();
const transformer = await createTransformerAPI();

// Load from URLs (or File API)
await transformer.loadTokenizer(tokenizerData);
await transformer.loadModel(configData, weightsData);

// Stream tokens in real-time
for await (const token of transformer.generateStream(prompt, 50, 0.7)) {
  console.log(token); // Updates UI immediately
}
```

**C# (.NET 9+):**

```csharp
using Welvet;

var transformer = new Transformer();
await transformer.LoadTokenizerAsync("tokenizer.json");
await transformer.LoadModelAsync("config.json", "model.safetensors");

await foreach (var token in transformer.GenerateStreamAsync(prompt, 50, 0.7f))
{
    Console.Write(token);
}
```

**Supported Models:**

- ✅ SmolLM2-135M-Instruct (tested, working)
- ✅ Pythia-70M/160M (tested, working)
- ✅ Any HuggingFace model with similar architecture (LLaMA, GPT-2, etc.)

**Download models:**

```bash
pip install huggingface-hub
huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct \
  --local-dir models/SmolLM2-135M-Instruct
```

See language-specific READMEs for detailed examples:

- [Python README](python/README.md) - Server & CLI examples
- [TypeScript README](typescript/README.md) - Browser WASM demo
- [C# README](csharp/README.md) - .NET console & web interface
- [WASM README](wasm/README.md) - Pure WASM implementation

**Cross-Platform Tests:**

- **Python/C-ABI**: `python/examples/all_layers_test.py`
- **WebAssembly**: `wasm/all_layers_test.html` (open in browser)
- **TypeScript/Bun**: `typescript/examples/all_layers_test.js`
- **C#/.NET**: `csharp/examples/Program.cs`
- **Go Native**: `examples/all_layers_validation.go`

All tests load the same `test.json` model file and verify outputs match!

## Validation

All 5 layer types (Dense, Conv2D, Multi-Head Attention, RNN, LSTM) have been empirically validated through end-to-end training:

- **Dense-only baseline**: 98.6% loss reduction, perfect classification in 50 epochs
- **Full 6-layer stack** (Dense→Conv2D→Attention→RNN→LSTM→Dense): 93.6% loss reduction, perfect classification in 200 epochs
- **Cross-platform verified**: Native Go, WebAssembly, TypeScript, and Python bindings tested

Run the validation test:

```bash
cd examples
go run all_layers_validation.go
```

Expected output: Clean convergence and perfect binary classification demonstrating all layer types learn correctly.

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

### Pre-trained BERT Model Example

Load and use converted BERT models from HuggingFace:

```go
package main

import (
    "fmt"
    "github.com/openfluke/loom/nn"
)

func main() {
    // Load converted BERT-Tiny model
    network, err := nn.LoadImportedModel("model_conversion/bert-tiny.json", "bert-tiny")
    if err != nil {
        panic(err)
    }

    fmt.Printf("Loaded BERT with %d layers\n", network.TotalLayers())
    // Output: Loaded BERT with 10 layers
    // 2 transformer blocks: [MHA, LayerNorm, Dense, Dense, LayerNorm] × 2

    // Create embeddings (from tokenizer + embedding layer)
    seqLength := 128
    hiddenSize := 128
    embeddings := make([]float32, seqLength*hiddenSize)
    // ... fill with word + position embeddings from BERT tokenizer

    // Run forward pass through transformer
    output, _ := network.ForwardCPU(embeddings)

    // Output: contextual embeddings for each token
    fmt.Printf("Output shape: %d values (%d tokens × %d hidden)\n",
        len(output), seqLength, hiddenSize)
}
```

**Convert your own models:**

```bash
cd model_conversion
python3 convert_tiny.py  # Select BERT-Tiny, Mini, or custom
python3 verify_bert_weights.py  # Verify 54% similarity
go run run_bert_tiny.go  # Test in Go
```

See [`model_conversion/README.md`](model_conversion/README.md) for complete guide including:

- Architecture details (attention, LayerNorm, residuals, FFN)
- Verification tools and similarity metrics
- Adding support for GPT-2, T5, Vision Transformers
- Troubleshooting and debugging

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

## Python Package (welvet)

**Wrapper for Embedding Loom Via External (C-ABI) Toolchain**

High-level Python bindings for LOOM with GPU acceleration support.

### Installation

```bash
pip install welvet
```

### Quick Example

```python
import welvet

# Create network with GPU acceleration
network = welvet.create_network(
    input_size=4,
    grid_rows=1,
    grid_cols=1,
    layers_per_cell=2,
    use_gpu=True
)

# Configure: 4 -> 8 -> 2
welvet.configure_sequential_network(
    network,
    layer_sizes=[4, 8, 2],
    activations=[welvet.Activation.RELU, welvet.Activation.SIGMOID]
)

# Training data
inputs = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
targets = [[1.0, 0.0], [0.0, 1.0]]

# Train
for epoch in range(10):
    loss = welvet.train_epoch(network, inputs, targets, learning_rate=0.1)
    print(f"Epoch {epoch+1}: loss = {loss:.4f}")

# Predict
output = welvet.forward(network, [0.1, 0.2, 0.3, 0.4])
print(f"Output: {output}")

# Cleanup
welvet.cleanup_gpu(network)
welvet.free_network(network)
```

### Features

- ✅ **Simple API** - High-level helpers for common tasks
- ✅ **GPU Support** - WebGPU acceleration via C-ABI
- ✅ **Multi-platform** - Linux, macOS, Windows, Android binaries included
- ✅ **Lightweight** - ctypes-based, no compilation required
- ✅ **Type Safe** - Proper error handling and validation

See [python/README.md](python/README.md) for complete documentation.

**PyPI**: https://pypi.org/project/welvet/

## .NET/C# Package (Welvet)

High-level C# bindings for LOOM with full P/Invoke support for .NET 9.0+.

### Installation

```bash
dotnet add package Welvet
```

### Quick Example

```csharp
using Welvet;

// Create network with GPU acceleration
using var network = Network.Create(
    inputSize: 4,
    gridRows: 1,
    gridCols: 1,
    layersPerCell: 2,
    useGpu: true
);

// Configure: 4 -> 8 -> 2
network.ConfigureSequential(
    layerSizes: new[] { 4, 8, 2 },
    activations: new[] { Activation.ScaledReLU, Activation.Sigmoid }
);

// Training data
var inputs = new float[][] {
    new[] { 0.1f, 0.2f, 0.3f, 0.4f },
    new[] { 0.5f, 0.6f, 0.7f, 0.8f }
};
var targets = new float[][] {
    new[] { 1.0f, 0.0f },
    new[] { 0.0f, 1.0f }
};

// Train
for (int epoch = 0; epoch < 10; epoch++)
{
    float loss = network.TrainEpoch(inputs, targets, learningRate: 0.1f);
    Console.WriteLine($"Epoch {epoch + 1}: loss = {loss:F4}");
}

// Predict
var output = network.Forward(new[] { 0.1f, 0.2f, 0.3f, 0.4f });
Console.WriteLine($"Output: [{string.Join(", ", output)}]");
```

### One-Line Model Loading

```csharp
// Load complete model from JSON string
using var network = Network.LoadFromString(modelJson, "my_model");

// Save model to JSON string
string json = network.SaveToString("my_model");
```

### Features

- ✅ **Modern C# API** - IDisposable, nullable reference types, async-ready
- ✅ **GPU Support** - WebGPU acceleration via P/Invoke to C-ABI
- ✅ **Multi-platform** - Linux, macOS, Windows with native library packaging
- ✅ **Type Safe** - Strong typing with proper exception handling
- ✅ **.NET 9.0+** - Built for latest .NET runtime
- ✅ **Zero Dependencies** - Pure P/Invoke, no external packages

See [csharp/README.md](csharp/README.md) for complete documentation.

**NuGet**: https://www.nuget.org/packages/Welvet/

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

> ⚠️ **Experimental Feature**: GPU support is currently in active development. Results may vary across hardware configurations.

### WebGPU Compute Shaders

Loom uses WGSL (WebGPU Shading Language) for GPU compute:

- **Dense Forward/Backward**: Element-wise activation and gradient computation
- **MHA Matrix Ops**: `matmulGPU` and `matmulTransposeGPU` kernels
- **Optimizations**: Command batching, efficient buffer management

### GPU Status by Layer Type

| Layer Type | Forward GPU | Backward GPU | Status                              |
| ---------- | ----------- | ------------ | ----------------------------------- |
| Dense      | ✅ Active   | ✅ Active    | Development (functional)            |
| MHA        | ⚠️ Hybrid   | ⚠️ Hybrid    | Experimental (may have issues)      |
| Conv2D     | ❌ Buggy    | ❌ Buggy     | Falls back to CPU                   |
| RNN        | ❌ CPU      | ❌ CPU       | CPU only (sequential nature)        |
| LSTM       | ❌ CPU      | ❌ CPU       | CPU only (sequential nature)        |

## Documentation

- [Neural Network Package](nn/README.md) - Detailed API documentation
- [Neural Tween Assessment](docs/step_tween_assessment.md) - Comprehensive benchmarks for Neural Tweening (19 tests)
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

- [x] **Neural Tweening (StepTweenChain)**: Bidirectional training for real-time embodied AI (validated across 19 tests)
- [x] **Neural Telemetry**: Network blueprint extraction and activity visualization
- [x] **Step Forward/Backward**: All layer types now support stepping (Dense, Conv2D, RNN, LSTM, Attention, Norm, SwiGLU)
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
