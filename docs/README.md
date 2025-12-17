# LOOM Documentation Hub

**LOOM (Layered Omni-architecture Openfluke Machine)** - A high-performance **CPU-first** neural network framework written in Go. WebGPU GPU acceleration is **experimental and in development** — only select layers are supported. Features WebAssembly export for browser deployment.

---

## 🧠 Neural Tweening: The Breakthrough for Real-Time AI

> **For comprehensive benchmarks across 19 tests, see: [Step Tween Assessment](step_tween_assessment.md)**

Neural Tweening (StepTweenChain) represents a **paradigm shift** for embodied AI and real-time decision making:

### The Problem with Traditional Neural Networks

Traditional neural networks only have **one layer active at a time**. Input enters layer 1, propagates through layers 2, 3, ... N, and only then produces an output:

- **Action is derived from X layers of sequential propagation**
- **Decision latency = sum of all layer forward times**
- The network must "think" before it can "act"

For real-time embodied AI (robotics, games, virtual agents), this propagation delay is unacceptable.

### The Stepping + Tween Solution

The **stepping mechanism** runs all layers simultaneously in parallel:

```
Traditional:  Layer1 → Layer2 → Layer3 → Output  (sequential, slow)
Stepping:     [Layer1 | Layer2 | Layer3 | Output] (parallel, fast)
```

Combined with **Neural Tweening** (bidirectional "meet in the middle" weight updates), you can:

1. **Train and run simultaneously** — no separate "training mode" vs "inference mode"
2. **Achieve minimal decision latency** — only 1 layer delay between input and output
3. **Never crash to 0%** — maintains 40-80% accuracy even during task changes
4. **Statistically validated stability** — 0.8-1.9% StdDev vs 4-10% for traditional methods

### Key Results (Tests 16-19)

| Metric | NormalBP | StepTweenChain |
|--------|----------|----------------|
| Shallow (3-5L) Dense/Conv2D | 98-99% | 90-99% |
| Speed to First Milestone | ~300ms | **~100ms** |
| **Stability During Task Changes** | Crashes to 0% | **Never below 40%** |
| Standard Deviation (100 runs) | 4-10% | **0.8-1.9%** |

**Key Insight:** For embodied AI, a consistent 45% accuracy beats oscillating between 100% and 0%. An agent that maintains baseline competence while adapting beats one that freezes during transitions.

---

## 📚 Documentation Index

### Core Documentation

| Document | Description |
|----------|-------------|
| [Step Tween Assessment](step_tween_assessment.md) | Comprehensive benchmarks for Neural Tweening across 19 tests |
| [Main README](../README.md) | Project overview, quick start, and API reference |
| [Neural Network Package](../nn/README.md) | Detailed nn/ package documentation |
| [Evaluation System](../nn/EVALUATION_README.md) | DeviationMetrics evaluation guide |

### Platform-Specific Guides

| Platform | Documentation |
|----------|---------------|
| [Python (welvet)](../python/README.md) | Python bindings via ctypes |
| [TypeScript/WASM](../typescript/README.md) | Browser WASM and Node.js |
| [C#/.NET (Welvet)](../csharp/README.md) | .NET 9+ bindings |
| [C ABI/FFI](../cabi/README.md) | C foreign function interface |
| [WASM](../wasm/README.md) | WebAssembly module |
| [Tokenizer](../tokenizer/README.md) | Pure Go BPE tokenizer |
| [Model Conversion](../model_conversion/README.md) | HuggingFace model import |

---

## 🎯 Capability Overview

LOOM is a **CPU-first framework** with full forward/backward passes for 10 layer types. All layers are **fully tested and reliable on CPU**. GPU/WebGPU code exists but is experimental.

### Layer Types (All with Full CPU Support)

| Layer | Description | GPU Status |
|-------|-------------|------------|
| **Dense** | Fully-connected with activations | ✅ Production |
| **Conv2D** | 2D convolution with stride/padding | ⚠️ Buggy |
| **Multi-Head Attention** | Transformer-style Q/K/V | ✅ Hybrid |
| **LayerNorm** | Layer normalization with residual | CPU only |
| **RNN** | Recurrent with BPTT | CPU only |
| **LSTM** | Long Short-Term Memory with gates | CPU only |
| **Softmax** | 10 variants including native MoE | CPU only |
| **Parallel** | Multiple branches with 4 combine modes | CPU only |
| **RMSNorm** | Root Mean Square normalization | CPU only |
| **SwiGLU** | Swish-Gated Linear Unit | CPU only |

### Special Features

- **Native MoE via Grid Softmax** — Mathematically proven equivalent to traditional MoE
- **Grid Scatter Mode** — Place outputs at specific 2D/3D grid positions
- **Stepping API** — Fine-grained execution control for real-time training
- **Neural Tweening** — Bidirectional training for stable embodied AI
- **Neural Telemetry** — Network blueprint extraction for visualization
- **Cross-Platform** — Go, Python, TypeScript, C#, C, WASM

---

## 📊 Training Modes Reference

| Mode | Description | Best For |
|------|-------------|----------|
| **NormalBP** | Traditional epoch-based backpropagation | Deep networks, static tasks |
| **NormTween** | Epoch-based Neural Tweening | Shallow Dense/Conv2D (100% accuracy) |
| **Step+BP** | Stepping + backpropagation | LSTM, Attention architectures |
| **StepTween** | Stepping + legacy tween gradients | Legacy compatibility |
| **TChain** | Stepping + Tween + Chain Rule | Recommended for stepping |
| **BatchTween** | Batch-mode Neural Tweening | Not recommended (WIP) |
| **StepBatch** | Stepping + batch accumulated | Not recommended (WIP) |

### Recommended Mode Selection

```
FOR REAL-TIME EMBODIED AI:
  → Use StepTweenChain for adaptive systems
  → Keep networks shallow (3-5 layers)
  → Consider multi-network architecture

FOR STATIC TASKS:
  → Use NormTween for shallow Dense/Conv2D
  → Use NormalBP for deep networks (15+)
  → Use Step+BP for LSTM and Attention
```

---

## 🚀 Quick Start

```go
// Create network with stepping support
net := nn.NewNetwork(inputSize, gridRows, gridCols, layersPerCell)
state := net.InitStepState(inputSize)

// Configure Neural Tweening
ts := nn.NewTweenState(net)
ts.Config.UseChainRule = true           // Enable chain rule gradients
ts.Config.ExplosionDetection = false    // Disable rate dampening

// Stepping loop - train and run simultaneously
for {
    state.SetInput(input)
    net.StepForward(state)
    output := state.GetOutput()
    
    // Train while running
    ts.TweenStep(net, input, targetClass, learningRate)
}
```

---

## 📈 Cross-Platform API Consistency

All platforms share the same simple API with identical behavior:

| Function | Go | Python | TypeScript | C# | C |
|----------|----|----|----|----|---|
| Create Network | `BuildNetworkFromJSON()` | `create_network_from_json()` | `createNetworkFromJSON()` | `CreateLoomNetwork()` | `CreateLoomNetwork()` |
| Forward Pass | `ForwardCPU()` | `forward_simple()` | `forward()` | `LoomForward()` | `LoomForward()` |
| Train | `Train()` | `train_simple()` | `train()` | `LoomTrain()` | `LoomTrain()` |

**Verified Identical Behavior:**
- ✅ Same training results (99.3-99.5% improvement)
- ✅ Bit-for-bit identical predictions after save/load
- ✅ Same evaluation metrics across platforms

---

## 📁 Examples

| Example | Description |
|---------|-------------|
| `examples/step_example/visualization_demo.go` | Neural telemetry and activity visualization |
| `examples/ex1/test19_architecture_adaptation_sparta.go` | SPARTA benchmark (100 runs statistical validation) |
| `examples/moe_proof_demo.go` | MoE proof via Grid Softmax |
| `examples/json_grid_scatter_demo.go` | Grid Scatter mode demo |
| `examples/all_layers_validation.go` | Cross-platform serialization test |

---

**For the latest information, see the [main README](../README.md).**

**Made with ❤️ by OpenFluke**
