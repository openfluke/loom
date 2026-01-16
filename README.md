# LOOM - Layered Omni-architecture Openfluke Machine

**"The SQLite of AI" — A Universal Neural Runtime & Engine**

Loom is a compiled, cross-platform **Neural Runtime Environment (NRE)** designed for structural interoperability and embedded intelligence. It bridges the gap between training frameworks and inference engines, acting as the **JVM for Neural Networks**:

*   **Write Once:** Define architecture in JSON.
*   **Run Anywhere:** Go, Browser (WASM), Desktop (Python/C#/.NET), Mobile.
*   **Universal Bytecode:** The JSON model definition is the bytecode.

Unlike heavy frameworks, Loom compiles to a **single binary** with zero dependencies. It features a **Virtual Execution Layer** that transparently routes operations to AVX2 (CPU) or WebGPU/Vulkan (GPU) without changing a line of user code.

[![Go Version](https://img.shields.io/badge/Go-1.24+-blue.svg)](https://golang.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/welvet.svg)](https://pypi.org/project/welvet/)
[![npm](https://img.shields.io/npm/v/@openfluke/welvet.svg)](https://www.npmjs.com/package/@openfluke/welvet)
[![NuGet](https://img.shields.io/nuget/v/Welvet.svg)](https://www.nuget.org/packages/Welvet/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![.NET](https://img.shields.io/badge/.NET-9.0+-512BD4.svg)](https://dotnet.microsoft.com/)

## 🌍 Cross-Ecosystem Compatibility

Models trained in **any platform** work instantly in all others. **Bit-for-bit identical results** across Go, Python, C#, TypeScript, and browser WASM.

| Platform | Package | Install |
|:---------|:--------|:--------|
| **Go** | [GitHub](https://github.com/openfluke/loom) | `go get github.com/openfluke/loom` |
| **Python** | [PyPI](https://pypi.org/project/welvet/) | `pip install welvet` |
| **C#/.NET** | [NuGet](https://www.nuget.org/packages/Welvet) | `dotnet add package Welvet` |
| **TypeScript/Node** | [NPM](https://www.npmjs.com/package/@openfluke/welvet) | `npm install @openfluke/welvet` |
| **Browser** | WASM | `import { init } from "@openfluke/welvet"` |

### Supported Platforms

Pre-compiled binaries for:
- **Linux**: x86_64, ARM64, ARMv7
- **Windows**: x86_64, x86, ARM64
- **macOS**: Apple Silicon (M1/M2/M3), Intel, Universal
- **Android**: ARM64, ARMv7
- **iOS**: ARM64 (XCFramework)

---

## Key Strengths

- **True Embeddability**: Single binary. Zero external dependencies. No Python runtime needed.
- **Hybrid Gradient/Geometric Engine**: [Neural Tweening](docs/step_tween_assessment.md) combines geometric gap-closing with backpropagation-guided momentum for real-time adaptation.
- **Structural Parallelism**: Native support for Inception, ResNeXt, Siamese, and MoE architectures via `LayerParallel` with 6 combine modes.
- **Native Mixed-Precision**: Generic tensor backend supports `int8`, `uint16`, `float32`, `float64` natively.
- **Complete Training Infrastructure**: 7 LR schedulers, 3 optimizers (SGD/AdamW/RMSprop), 10 softmax variants.
- **Pure Go Tokenizer**: HuggingFace-compatible BPE tokenizer for LLM inference.
- **Step-Based Execution**: Real-time inference with layer-by-layer control via `StepForward` API.
- **Network Telemetry**: Runtime introspection via `GetMethodsJSON()` and `ExtractNetworkBlueprint()`.

### Key Limitations

- **Ecosystem Maturity**: No central "Model Zoo" or pip-installable convenience; relies on loading external checkpoints.
- **GPU Support**: **WebGPU** acceleration is implemented (Dense, Conv2D, MHA) but is **beta/experimental** and less stable than CuDNN/CUDA.
- **Operator Coverage**: While "Deep" support is good (MHA, LSTM), "Broad" support (e.g., 3D Conv, Deformable Attn, FFTs) is missing compared to SciPy/JAX.
- **Math Backend**: Relies on custom explicit forward/backward passes rather than a general-purpose symbolic autograd graph.

---

## Recommended Configurations

Based on exhaustive benchmarks (300+ combinations tested), here are the optimal configurations:

### Training Mode Selection

| Scenario | Recommended Mode | Why |
|:---------|:-----------------|:----|
| **Real-time / Robotics** | `StepBP` or `StepTweenChain` | 100% availability, 0ms blocking |
| **Noisy / Adversarial Data** | `StepTweenChain` | 94% robustness vs 86% for NormalBP |
| **Offline Batch Training** | `NormalBP` | Highest accuracy when blocking is acceptable |
| **Multi-Agent Systems** | `StepBP` | 12x better coordination vs blocked training |
| **Continuous Adaptation** | `StepTweenChain` | Maintains competence during distribution shift |

### Layer × Training Mode (float32)

| Layer | Best Mode | Score | Accuracy | Availability |
|:------|:----------|------:|:---------|:-------------|
| **Conv2D** | StepTweenChain | **1187** | 98.7% | 100% |
| **Conv2D** | StepTween | 1012 | 98.7% | 100% |
| **Attention** | StepTween | 830 | 90.1% | 100% |
| **RNN** | StepTween | 663 | 76.5% | 100% |
| **Dense** | StepTween | 379 | 42.5% | 100% |
| **LSTM** | NormalBP | 49 | 53.6% | 28.7% |

> [!TIP]
> **Conv2D + StepTweenChain + float32** is the optimal configuration for most real-time scenarios, achieving 98.7% accuracy with 100% availability.

### Numeric Type Selection

| Type | Best For | Notes |
|:-----|:---------|:------|
| **float32** | Most use cases | 18/30 benchmark wins, best accuracy |
| **float64** | Scientific computing | Higher precision, slower, wins with NormalBP |
| **int16** | LSTM layers | Only type that works for step-based LSTM |
| **uint16** | Edge/embedded | Good balance of range and speed |

> [!NOTE]
> Integer types (`int8`, `uint8`, etc.) work but achieve only ~13-23% accuracy on adaptive tasks. Use floats for training, integers for quantized inference.

---

## Benchmarks and Repro

Benchmark methodology and results live in [docs/step_tween_assessment.md](docs/step_tween_assessment.md). Results are hardware- and build-dependent; use CPU runs as the reference baseline when comparing.

---

## What's New

> 🎉 **Transformer Inference**: SmolLM2-135M-Instruct runs entirely in browser WASM with pure Go implementation.

> 🤯 **Grid Softmax = Native MoE**: Mathematically proven equivalent to PyTorch MoE with 97.1% loss reduction. See `examples/moe_proof_demo.go`.

> ⚡ **Grid Scatter Mode**: Place parallel branch outputs at specific 2D/3D grid positions for multi-agent systems, hierarchical RL, and ensemble methods with explicit topology.

> 🧠 **Neural Tweening**: Train and run simultaneously with 100% accuracy on shallow networks, never crashes to 0% during task changes. [Benchmarks →](docs/step_tween_assessment.md)

> 📦 **Recursive Safetensors**: Full support for deeply nested architectures (MoE, Sequential, Parallel) with 100% bitwise save/load consistency. Verified with `tva/testing/safetensors_recursive.go`.

> 🔢 **MNIST Verification**: New end-to-end demo `tva/demo/mnist/main.go` proving exact CPU/GPU consistency and training convergence.

---

## Framework Comparison

### Global AI Landscape

| Feature Category | Feature | **Loom** (Go) | **PyTorch** (Py) | **TF / TFLite** | **GoMLX** (Go) | **Spago** (Go) | **Core ML** | **TF.js** | **Candle** (Rust) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Core** | **Primary Language** | Go | Python | Python / C++ | Go | Go | Swift / ObjC | JS / TS | Rust |
| | **Runtime Dependency** | **None** (Binary) | Heavy (Pip) | Binary (Edge) | CGo / XLA | None | OS-Native | Browser | None |
| | **Auto-Differentiation** | ⚠️ Hybrid/Manual | ✅ Full | ✅ Full | ✅ Full (XLA) | ✅ Manual | ❌ (Inference) | ✅ Full | ✅ Full |
| | **Safetensors** | ✅ **Native** | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| | **ONNX Support** | ❌ | ✅ (Export) | ✅ | ⚠️ | ❌ | ✅ (Import) | ✅ | ⚠️ |
| | **Structure Inference** | ✅ **Auto-Detect** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Training** | **Gradient Descent** | ✅ Manual Chain | ✅ Standard | ✅ Standard | ✅ Standard | ✅ Standard | ✅ (On-device) | ✅ Standard | ✅ Standard |
| | **Neural Tweening** | ✅ **Hybrid Engine** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **LR Schedulers** | ✅ **7 Types** | ✅ | ✅ | ✅ | ⚠️ Basic | ✅ | ✅ | ✅ |
| | **Optimizers** | ✅ **3 (SGD/AdamW/RMSprop)** | ✅ Many | ✅ Many | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| **Layer Support** | **Dense (MLP)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **Conv2D** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| | **Conv1D** | ✅ **Native** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| | **RNN / LSTM** | ✅ **Full Gate** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **Transformer (MHA)** | ✅ (Explicit) | ✅ | ✅ | ✅ | ✅ (BERT) | ✅ | ✅ | ✅ |
| | **SwiGLU** | ✅ **Native** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| | **Parallel / MoE** | ✅ **Structure** | ❌ (Manual) | ❌ (Manual) | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **Sequential Layers** | ✅ **Native** | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ | ✅ | ⚠️ |
| | **Embeddings** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **Tokenizer** | ✅ **Pure Go** | ❌ (Rust/C++) | ❌ (C++) | ❌ | ❌ | ✅ | ❌ | ✅ |
| **Normalization** | **LayerNorm** | ✅ **Native** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **RMSNorm** | ✅ **Native** | ⚠️ (Manual) | ⚠️ (Manual) | ✅ | ❌ | ❌ | ❌ | ✅ |
| | **Residual/Skip** | ✅ **Native** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Advanced** | **Stitch Layers** | ✅ **Native** | ❌ (Manual) | ❌ (Manual) | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **Dynamic Arch Gen** | ✅ **Built-in** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **Step-Based Forward** | ✅ **Unique** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **K-Means Clustering** | ✅ **Parallel** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **Correlation Analysis** | ✅ **Pearson/Spearman** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **Model Evaluation** | ✅ **Deviation/Metrics** | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| | **Network Telemetry** | ✅ **Blueprint API** | ❌ | ⚠️ | ❌ | ❌ | ❌ | ⚠️ | ❌ |
| | **Runtime Introspection** | ✅ **Reflection** | ⚠️ (Python) | ⚠️ | ❌ | ❌ | ❌ | ⚠️ | ❌ |
| **Platform** | **WASM Training** | ✅ **Full** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (Slow) | ✅ |
| | **Cross-Lang ABI** | ✅ **Universal** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ |
| **Ecosystem** | **HuggingFace Hub** | ⚠️ (Read/Inspect) | ✅ Native | ✅ Native | ❌ | ✅ | ❌ | ✅ | ✅ |
| | **Pre-trained Zoo** | ❌ | ✅ Massive | ✅ Massive | ❌ | ✅ (Small) | ✅ (Apple) | ✅ Large | ⚠️ Growing |
| | **Mobile/Web** | ✅ **WASM / C-ABI** | ✅ (Mobile) | ✅ **King** | ❌ | ❌ | ✅ **King (iOS)** | ✅ **King (Web)** | ✅ (WASM) |

### Go Ecosystem Comparison

| **Category** | **Feature** | **Loom** | **GoMLX** | **Gorgonia** | **Spago** | **Go-Deep** | **Gonum** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Foundation** | **Primary implementation** | Pure Go | CGo (XLA) | Pure Go + CGo | Pure Go | Pure Go | Pure Go |
| | **Tensor Backend** | Custom (Generic) | XLA (C++) | Custom | Custom (Dense) | Custom | Dense Matrix |
| | **Autograd** | ⚠️ Hybrid | ✅ Full | ✅ Symbolic | ✅ Dynamic | ✅ Backprop | ❌ |
| **Model** | **Load Safetensors** | ✅ **Native** | ✅ | ❌ | ❌ | ❌ | ❌ |
| | **Model Export** | binary/json | XLA format | Onnx (Import) | Gob | Json | ❌ |
| **Architecture** | **Dense (MLP)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (Matrix Mul) |
| | **Conv2D** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| | **Conv1D** | ✅ **Native** | ✅ | ⚠️ (via 2D) | ⚠️ (via 2D) | ❌ | ❌ |
| | **RNN / LSTM** | ✅ **Full Gate** | ✅ | ⚠️ Basic | ✅ BiLSTM | ❌ | ❌ |
| | **Transformer (MHA)** | ✅ **Explicit** | ✅ | ⚠️ Hard | ✅ (BERT) | ❌ | ❌ |
| | **SwiGLU** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| | **Embeddings** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| | **Parallel / MoE** | ✅ **MoE + Gating** | ❌ (Manual) | ❌ | ❌ | ❌ | ❌ |
| | **Sequential Layers** | ✅ **Native + Nested** | ⚠️ (Manual) | ⚠️ (Manual) | ⚠️ (Manual) | ❌ | ❌ |
| | **Tokenizer** | ✅ **Pure Go** | ❌ (Deps) | ❌ | ✅ (WordPiece) | ❌ | ❌ |
| **Training** | **Gradient Descent** | ✅ Manual | ✅ Standard | ✅ Standard | ✅ Standard | ✅ Standard | ❌ |
| | **Hybrid Tweening** | ✅ **Unique** | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **LR Schedulers** | ✅ **7 Types** | ✅ | ✅ | ⚠️ Basic | ❌ | ❌ |
| | **Optimizers** | ✅ **SGD/AdamW/RMSprop** | ✅ | ✅ | ✅ | ⚠️ SGD | ❌ |
| | **Softmax Variants** | ✅ **10 Types** | ⚠️ Standard | ⚠️ Standard | ⚠️ Standard | ⚠️ Standard | ❌ |
| **Normalization** | **LayerNorm** | ✅ **Native** | ✅ | ⚠️ Manual | ✅ | ❌ | ❌ |
| | **RMSNorm** | ✅ **Native** | ✅ | ❌ | ❌ | ❌ | ❌ |
| | **Residual/Skip** | ✅ **Native** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Advanced** | **RoPE Embeddings** | ✅ **GQA Support** | ✅ | ❌ | ❌ | ❌ | ❌ |
| | **Network Grafting** | ✅ **Unique** | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **Step-Based Forward** | ✅ **Unique** | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **Dynamic Arch Gen** | ✅ **Unique** | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **K-Means Clustering** | ✅ **Parallel** | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **Correlation Analysis** | ✅ **Pearson/Spearman** | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **Model Evaluation** | ✅ **Full Suite** | ⚠️ | ⚠️ | ⚠️ | ❌ | ❌ |
| | **Network Telemetry** | ✅ **Blueprint** | ❌ | ⚠️ | ❌ | ❌ | ❌ |
| | **Runtime Introspection** | ✅ **Reflection** | ❌ | ⚠️ | ❌ | ❌ | ❌ |
| **Platform** | **C-ABI (Polyglot)** | ✅ **Universal** | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **WASM Training** | ✅ **Full** | ❌ (XLA) | ❌ | ❌ | ❌ | ❌ |
| **Ecosystem** | **HuggingFace** | ⚠️ (Load) | ❌ | ❌ | ✅ (Load) | ❌ | ❌ |
| | **Documentation** | ⚠️ Growing | ✅ Good | ✅ Good | ✅ Good | ⚠️ Minimal | ✅ Excellent |
| | **Maintenance** | 🔥 Active | 🔥 Active | ⚠️ Slow | ⏸️ Paused | ⚠️ Slow | 🔥 Active |

### Native Numerical Type & Precision Support

| **Layer Type** | **Numerical Type** | **Loom** | **GoMLX** | **Gorgonia** | **Spago** | **PyTorch** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **All Layers** | **Float32** | ✅ | ✅ | ✅ | ✅ (Float64) | ✅ |
| (Dense, Conv, | **Float64 (High Prec)** | ✅ **Native** | ✅ | ✅ | ✅ | ✅ |
| RNN, Attn) | **Float16 / BF16** | ⚠️ (Storage) | ✅ (XLA) | ❌ | ❌ | ✅ |
| | **Int8 (Training)** | ✅ **Native** | ❌ | ❌ | ❌ | ⚠️ (QAT Wrapper) |
| | **Int8 (Inference)** | ✅ | ❌ | ❌ | ❌ | ✅ (Quant) |
| | **Int16, Int32, Int64** | ✅ **Native** | ✅ (XLA) | ⚠️ (Tensor) | ❌ | ❌ (Tensor Only) |
| | **Uint8, Uint16, Uint32** | ✅ **Native** | ✅ (XLA) | ⚠️ (Tensor) | ❌ | ✅ (Uint8 Only) |

> [!NOTE]
> **Complete Type System**: Unlike frameworks that treat integers primarily as storage formats for quantization, Loom's Generics allow **native training and inference** on exotic types like `uint16` (common in medical imaging), `int32`, or `float64` (scientific sim) across **every layer type** without changes to the model code.

### Summary Verdict

- **Choose PyTorch** if you are doing **Research**, need the latest SOTA models, or rely on complex dynamic architectures.
- **Choose TensorFlow / TFLite** if you need robust **Mobile/Edge Deployment**.
- **Choose GoMLX** if you need **High-Performance Training in Go** and can tolerate CGo/C++ dependencies.
- **Choose Core ML** if you are targeting **iOS/macOS** exclusively.
- **Choose Loom** if you need **Pure Go-Native Embedding** (Cloud/CLI/Server), want a single binary with zero dependencies, need to experiment with the **Neural Tweening** training paradigm, or need unique features like **Step-Based Forward Pass** for real-time inference and **Dynamic Architecture Generation** for automated model exploration.

---

## Layer Types & Features

### Supported Layer Types

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
| Softmax | `softmax` | 10 variants (Standard, Grid, Hierarchical, Temperature, Gumbel, Masked, Sparsemax, Entmax, Adaptive, Mixture) |
| Embedding | `embedding` | Token embedding |
| Parallel | `parallel` | Branching with 6 combine modes (add, concat, multiply, average, grid_scatter, filter) |
| Sequential | `sequential` | Grouped sub-layers |

### Activation Functions

`relu`, `sigmoid`, `tanh`, `softmax`, `gelu`, `swish`, `mish`, `leaky_relu`, `elu`, `selu`, `linear`

---

## SafeTensors & Model Interoperability
 
Loom features a **universal SafeTensors engine** capable of standardizing models from any framework (PyTorch, TensorFlow, HuggingFace) into a highly optimized, single-file format. It proactively handles complex **nested architectures** (like Mixture-of-Experts within Parallel layers) via recursive serialization.
 
### 1. Universal "Any-to-Any" Quantization
Load a model in high precision (`float32`/`float64`) and instantly quantize it to any supported type for deployment. The file format handles the type conversion automatically.
 
- **Input**: Model weights in `F32` (e.g., from HuggingFace)
- **Output**: Quantized weights in `F4`, `I8`, `BF16`, `U16` etc.
- **Verification**: 100% round-trip integrity verified for all 143 layer/type combinations.
 
```go
// Load standard model
tensors, _ := nn.LoadSafetensors("llama.safetensors")
 
// Save as 4-bit optimized web model (automatically quantizes)
for name, t := range tensors { t.DType = "F4" }
nn.SaveSafetensors("llama-web-4bit.safetensors", tensors)
```
 
### 2. WASM / In-Memory Operation
Loom's SafeTensors implementation can operate **purely in memory** (using `[]byte` buffers) without any filesystem access, making it perfect for **WebAssembly (WASM)** and constrained environments.
 
```go
// Serialize directly to memory (for sending to browser/client)
bytes, _ := nn.SerializeSafetensors(myModelWeights)
 
// Load directly from memory (no disk I/O required)
tensors, _ := nn.LoadSafetensorsWithShapes(bytes)
```
 
### 3. Full Layer Support
The interoperability layer supports every component in the Loom ecosystem:
 
| Category | Supported Layers | 
|:---|:---|
| **Core** | `Dense`, `Embedding`, `Parallel`, `Sequential` |
| ** Convolution** | `Conv1D`, `Conv2D` |
| **Sequence** | `RNN`, `LSTM`, `GRU` |
| **Attention** | `MultiHeadAttention`, `SwiGLU` |
| **Norm/Act** | `LayerNorm`, `RMSNorm`, `Softmax` (10 variants) |
 
---
 
## GPU Acceleration (WebGPU)

**Experimental** GPU acceleration via WebGPU compute shaders. Treat all GPU paths (forward and backward) as experimental for now. Use with:

```go
network.GPU = true
network.WeightsToGPU()           // Mount weights to GPU
output, _ := network.Forward(input)  // Auto-routes to GPU!
network.Backward(dOutput)     // GPU backward pass
network.ReleaseGPUWeights()      // Cleanup
```

### GPU Support Matrix

| Layer Type | Forward | Backward (Training) | Notes |
|:-----------|:-------:|:-------------------:|:------|
| **Dense** | ✅ **Stable** | ⚠️ **Experimental** | Production speedup (20x) on large layers. |
| **Conv2D** | ✅ **Stable** | ⚠️ **Experimental** | Works well, optimized for 32+ filters. |
| **Conv1D** | ✅ **Stable** | ⚠️ **Experimental** | Gradients implemented, accuracy tuning needed. |
| **RNN** | ✅ **Stable** | ⚠️ **Experimental** | Weights update, but BPTT limited to batch=1. |
| **LSTM** | ✅ **Stable** | ⚠️ **Experimental** | Same limitations as RNN. |
| **LayerNorm** | ✅ **Stable** | ⚠️ **Experimental** | Forward is stable, backward can be numeric unstable. |
| **RMSNorm** | ✅ **Stable** | ⚠️ **Experimental** | Same as LayerNorm. |
| **SwiGLU** | ✅ **Stable** | ⚠️ **Experimental** | High performance. |
| **MHA** | ✅ **Stable** | ⚠️ **Experimental** | Functional parity verified. |
| **Softmax** | ✅ **Stable** | ⚠️ **Experimental** | Functional. |


## Quick Start

Quick docs:
- [NN Overview](docs/nn/overview.md)
- [NN Quick Reference](docs/nn/quick_reference.md)

### Installation

```bash
# Clone the repository
git clone https://github.com/openfluke/loom.git
cd loom

# Install dependencies
go mod download
```

### Simple Example

```go
package main

import (
    "fmt"
    "github.com/openfluke/loom/nn"
)

func main() {
    network := nn.NewNetwork(4096, 4, 4, 5)  // 80 total layers

    if err := network.InitGPU(); err != nil {
        panic(err)
    }
    defer network.ReleaseGPU()

    input := make([]float32, 4096)
    output, gpuTime, _ := network.ForwardGPU(input)

    fmt.Printf("GPU Forward time: %v, Output size: %d\n", gpuTime, len(output))
}
```

### Model Serialization

```go
// Save a trained model
err := network.SaveModel("model.json", "my_model")

// Load it back - ONE LINE!
loadedNet, err := nn.LoadModel("model.json", "my_model")

// Or use strings (great for APIs/databases/WASM)
jsonString, err := network.SaveModelToString("my_model")
loadedNet, err := nn.LoadModelFromString(jsonString, "my_model")
```

### Cross-Platform API

| Function | Go | Python | TypeScript | C# | C |
|:---------|:---|:-------|:-----------|:---|:--|
| Create | `BuildNetworkFromJSON()` | `create_network_from_json()` | `createNetworkFromJSON()` | `CreateLoomNetwork()` | `CreateLoomNetwork()` |
| Forward | `Forward()` | `forward_simple()` | `forward()` | `LoomForward()` | `LoomForward()` |
| Train | `Train()` | `train_simple()` | `train()` | `LoomTrain()` | `LoomTrain()` |
| Save | `SaveModelToString()` | `save_model_simple()` | `saveModel()` | `LoomSaveModel()` | `LoomSaveModel()` |
| Load | `LoadModelFromString()` | `load_model_simple()` | `loadLoomNetwork()` | `LoomLoadModel()` | `LoomLoadModel()` |
| Evaluate | `EvaluateNetwork()` | `evaluate_network_simple()` | `evaluate()` | `LoomEvaluateNetwork()` | `LoomEvaluateNetwork()` |

---

## Language Bindings

### Python

```bash
pip install welvet
```

```python
import welvet

config = {"batch_size": 1, "layers": [...]}
welvet.create_network_from_json(config)
output = welvet.forward_simple([0.1, 0.2, 0.3, 0.4])
```

See [python/README.md](python/README.md) for complete documentation.

### TypeScript / Node.js

```bash
npm install @openfluke/welvet
```

```typescript
import { init, createNetworkFromJSON } from "@openfluke/welvet";

await init();
const network = createNetworkFromJSON(JSON.stringify(config));
const output = network.Forward(JSON.stringify([[0.1, 0.2, 0.3, 0.4]]));
```

See [typescript/README.md](typescript/README.md) for complete documentation.

### C# / .NET

```bash
dotnet add package Welvet
```

```csharp
using Welvet;

Network.CreateFromJson(config);
var output = NativeMethods.LoomForward(input, input.Length);
```

See [csharp/README.md](csharp/README.md) for complete documentation.

---

## Project Structure

```
loom/
├── nn/                  # Neural network package (core)
├── tokenizer/           # Pure Go BPE tokenizer
├── wasm/                # WebAssembly module
├── cabi/                # C ABI for FFI
├── python/              # Python package (welvet)
├── typescript/          # TypeScript/WASM package
├── csharp/              # C#/.NET package (Welvet)
├── fabric/              # Demo application
├── pods/                # GPU compute pods
├── model_conversion/    # HuggingFace model import
├── docs/                # Documentation
└── detector/            # GPU device detection
```

---

## Documentation

- [Neural Network Package](nn/README.md) - Detailed API documentation
- [Neural Tweening Benchmarks](docs/step_tween_assessment.md) - 19-test comprehensive benchmark
- [Python Bindings](python/README.md) - PyPI package docs
- [TypeScript Bindings](typescript/README.md) - NPM package docs
- [C# Bindings](csharp/README.md) - NuGet package docs
- [WASM Module](wasm/README.md) - Browser deployment
- [C ABI](cabi/README.md) - FFI reference
- [Model Conversion](model_conversion/README.md) - HuggingFace import guide

**More Examples:** See [github.com/openfluke/tva](https://github.com/openfluke/tva) for additional examples and experiments.

## Comprehensive Test Suite

Loom includes a rigorous verification suite in `tva/muniversal_testing.go` that validates functional correctness across all layers, numeric types, and backend engines (CPU/GPU).

### Coverage Summary

| Test Section | Description | Status |
|:-------------|:------------|:-------|
| **Part 1: Core** | Forward/backward pass correctness for basic layers | ✅ Covered |
| **Part 2: Serialization** | Save/Load permutations for all layers and types (`float32`, `int8`, `uint16`, etc.) | ✅ Covered |
| **Part 3: Advanced** | Complex layers (MHA, Grid Softmax, K-Means) and math ops | ✅ Covered |
| **Part 5: GPU Determinism** | Validates GPU forward pass matches CPU results | ✅ Covered |
| **Part 6: GPU Training** | Verifies GPU learning convergence vs CPU baseline | ✅ Covered |

> [!NOTE]
> **GPU Acceleration Limits:** As of v0.0.8, WebGPU acceleration is enabled for standard `Forward/Backward` passes. 
> The structural API `nn/step_forward.go` (Step-based execution) and `nn/tween.go` (Neural Tweening) currently run on **CPU only**.

### Verified Advanced Architectures

The test suite also verifies complex, production-ready architectural patterns:

- **Heterogenous MoE**: Using `LayerParallel` with `CombineMode: "filter"` to route inputs to experts of different depths/types (e.g., CNN expert vs Dense expert).
- **Stitched Experts**: Using `LayerStitch` to harmonize outputs from parallel branches with different dimensions (e.g., 5-dim output and 7-dim output stitched to common 10-dim).
- **Neural Grafting**: Training *only* the gating mechanism of an MoE while keeping experts frozen, using `TweenStep` for precise surgical updates.
- **Bit-Exact Determinism**: Verifying that GPU forward passes match CPU results to within machine epsilon (often exactly bit-matching for integer ops).

### Runnable Demos

- **MNIST Consistency (`tva/demo/mnist/main.go`)**:
  - Trains a digit classifier on MNIST.
  - Saves model to JSON and Safetensors.
  - Reloads model and verifies **0.000000000 max difference** in predictions.
  - Proves robustness of `SaveWeightsToSafetensors` / `LoadWeightsFromSafetensors`.

- **Recursive Safetensors (`tva/testing/safetensors_recursive.go`)**:
  - Constructs a complex nested Network: `MoE (Gate) -> [Parallel -> [Dense, Sequential -> [Conv1D, RNN]]]`.
  - Saves and reloads to prove structural integrity of serialization for arbitrary depths.


---

## Requirements

- **Go**: 1.24 or higher
- **GPU**: WebGPU-compatible GPU (Vulkan, Metal, or D3D12) - *optional*
- **OS**: Linux, macOS, or Windows

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

---

**Made with ❤️ by Openfluke**
