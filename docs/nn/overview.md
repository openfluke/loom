# Understanding the Neural Network Package

This document explains how Loom's neural network system actually works—not just what functions to call, but what's really happening under the hood when you build and train a network.

---

## Loom as a Deterministic Neural Virtual Machine

Loom is a **Deterministic Neural Virtual Machine (DNVM)** — a portable execution environment for neural networks that guarantees **bitwise-identical results** across all platforms, backends, and language bindings.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LOOM ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  │
│  │   Python    │   │  TypeScript │   │     C#      │   │    WASM     │  │
│  │   Binding   │   │   Binding   │   │   Binding   │   │   Browser   │  │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘  │
│         │                 │                 │                 │         │
│         └────────────────┬┴─────────────────┴─────────────────┘         │
│                          ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        C-ABI (FFI Layer)                          │  │
│  │         Handle-based state management, JSON marshalling           │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                          │                                              │
│                          ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    EXECUTION ENGINE (nn/)                         │  │
│  │   Forward/Backward passes, Optimizers, Schedulers, Tweening       │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│         │                                         │                     │
│         ▼                                         ▼                     │
│  ┌─────────────────┐                    ┌─────────────────────────┐     │
│  │   CPU Backend   │                    │    GPU JIT Compiler     │     │
│  │   (Pure Go)     │                    │   (WGSL Generation)     │     │
│  │                 │                    │         ▼               │     │
│  │  Deterministic  │                    │  ┌─────────────────┐    │     │
│  │  IEEE-754 Math  │◄────────────────►  │  │  WebGPU Runtime │    │     │
│  └─────────────────┘   Bit-identical    │  └─────────────────┘    │     │
│                           results       └─────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Architectural Components

| Layer | Component | Role |
|:------|:----------|:-----|
| **IR (Bytecode)** | JSON network configs, `serialization.go` | Portable, declarative network specification |
| **Type System** | `types.go` with `Tensor[T Numeric]` | Multi-precision tensors (F64→I8), generic operations |
| **Execution** | `forward.go`, `backward.go` | Deterministic layer-by-layer forward/backward |
| **JIT Backend** | `gpu/*.go` | Runtime WGSL generation → WebGPU pipelines |
| **FFI Runtime** | `cabi/main.go` | Handle-based API, state management, memory safety |
| **Bindings** | `python/`, `csharp/`, `typescript/`, `wasm/` | Thin wrappers exposing the C-ABI |

### Determinism Guarantee

Unlike typical ML runtimes that disclaim cross-platform reproducibility, Loom enforces **bit-exact determinism**:

```
┌──────────────────────────────────────────────────────────────────────┐
│ Testing: Dense                                                       │
├──────────────────────────────────────────────────────────────────────┤
│  • Max Diff:  0.0000000000 (Idx: -1)                                 │
│  • Mean Diff: 0.0000000000                                           │
│  ✅ [GOLD STANDARD] Exact Bit-Determinism                            │
│     Perfect match. CPU and GPU logic are identical down to the bit.  │
│     CPU: 0.5010004044 | GPU: 0.5010004044 | Diff: 0.0000000000       │
└──────────────────────────────────────────────────────────────────────┘
```

**Verified across:** CPU (Go) ↔ GPU (WebGPU/WGSL), x86_64 ↔ ARM64 ↔ ARMv7, Linux ↔ Windows ↔ macOS ↔ Android ↔ iOS, Native ↔ WASM (Browser)

---

## The Parallel Revolution: 600Hz Real-Time Intelligence

As of **v5.0 (2026)**, the Loom architecture achieved a significant breakthrough in real-time execution. By refactoring the main engine to support **Parallel Worker Pools**, we unlocked the "Parallel Turbo Mode."

### Key Findings:
*   **Bicameral Maximum Throughput**: Using the **Bicameral Configuration** (offloading training to a parallel "Right Hemisphere"), we observed **100.0% Availability** with zero-latency blocking.
*   **Throughput Scaling**:
    *   **Baseline (Sequential)**: ~7 Hz
    *   **Turbo (Parallel)**: **600 Hz** (a **85x improvement**)
*   **Constant State Neural Dynamics**: This frequency allows the network to stay in a "Constant State," where feature extraction occurs faster than the speed of light in a biological nervous system.

This validates Loom's mission to be a the **Neural Operating System** for robotics and cybersecurity—capable of learning and reacting at MHz frequencies on standard hardware.

---

## FP4 Weight Quantization for LLMs

To support massive language models on consumer hardware, Loom features built-in **NVFP4 E2M1** (4-bit floating point) quantization for Dense and SwiGLU layers. 

By packing two weights into every byte and sharing scale factors across micro-blocks, Loom reduces memory consumption by ~81%. The GPU backend includes specialized WebGPU shaders (`FP4DenseLayer` and `FP4SwiGLULayer`) that multiply directly against the packed 4-bit representation in VRAM, eliminating the need to decompress the model.

---

## The Big Picture: What Makes Loom Different

Most neural network frameworks organize layers in a simple chain: input flows through layer 1, then layer 2, then layer 3, and so on. Loom does something different. It organizes layers in a **2D grid**, like cells in a spreadsheet.

Why does this matter? Because real neural architectures aren't always linear chains. Transformers have parallel attention heads. Mixture-of-Experts models have multiple expert pathways. Residual networks have skip connections. The grid structure lets you express all of these patterns naturally.

Think of it like this:

```
Traditional Framework (Linear Chain):

    Input → [Layer 1] → [Layer 2] → [Layer 3] → Output
    
    Simple, but you can only do one thing at a time.


Loom's Grid Architecture:

    ┌─────────────┬─────────────┬─────────────┐
    │             │             │             │
    │  Cell(0,0)  │  Cell(0,1)  │  Cell(0,2)  │
    │  [Dense]    │  [Conv2D]   │  [Attention]│
    │  [Dense]    │  [Pool]     │  [Dense]    │
    │             │             │             │
    ├─────────────┼─────────────┼─────────────┤
    │             │             │             │
    │  Cell(1,0)  │  Cell(1,1)  │  Cell(1,2)  │
    │  [LSTM]     │  [Dense]    │  [Softmax]  │
    │  [Norm]     │  [ReLU]     │             │
    │             │             │             │
    └─────────────┴─────────────┴─────────────┘
    
    Each cell can contain multiple layers stacked on top of each other.
    Data flows through cells in a predictable pattern.
```

---

## How Data Flows Through the Grid

When you call `Forward(input)`, here's what actually happens:

1. **Your input enters cell (0,0)**—the top-left corner
2. **Data flows through all layers in that cell** from bottom to top
3. **The output moves to the next cell** in reading order (left→right, then down)
4. **This continues until reaching the bottom-right cell**
5. **The final output emerges**

Here's a visual:

```
Input Data: [1.0, 2.0, 3.0, ...]
     │
     ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Cell(0,0)   │────▶│ Cell(0,1)   │────▶│ Cell(0,2)   │
│ Layer 0: ─┐ │     │ Layer 0: ─┐ │     │ Layer 0: ─┐ │
│ Layer 1: ─┘ │     │ Layer 1: ─┘ │     │ Layer 1: ─┘ │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
     ┌─────────────────────────────────────────┘
     │
     ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Cell(1,0)   │────▶│ Cell(1,1)   │────▶│ Cell(1,2)   │
│ Layer 0: ─┐ │     │ Layer 0: ─┐ │     │ Layer 0     │
│ Layer 1: ─┘ │     │ Layer 1: ─┘ │     │   FINAL     │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                    Output: [0.1, 0.7, 0.2, ...]
```

Within each cell, layers execute from index 0 upward:

```
Inside Cell(0,0):
    
    Output from this cell
           ▲
           │
    ┌──────┴──────┐
    │   Layer 2   │  ← Third to execute (if exists)
    ├─────────────┤
    │   Layer 1   │  ← Second to execute
    ├─────────────┤
    │   Layer 0   │  ← First to execute
    └──────┬──────┘
           │
    Input to this cell
```

---

## The Network Object: What It Actually Contains

When you create a network with `NewNetwork(inputSize, rows, cols, layersPerCell)`, you're allocating a data structure that holds:

```
Network {
    InputSize: 1024          ← How big is the input vector?
    GridRows: 2              ← How many rows of cells?
    GridCols: 3              ← How many columns of cells?
    LayersPerCell: 2         ← Max layers per cell
    BatchSize: 1             ← For batched operations
    
    Layers: [][][]LayerConfig      ← 3D array: [row][col][layer]
           ↑
           This is where all the layer definitions live
    
    Optimizer: nil           ← Optional optimizer (AdamW, SGD, etc.)
    Observer: nil            ← Optional observer for monitoring
}
```

The `Layers` array is the heart of the network. It's a 3D array where:
- First index = row in the grid
- Second index = column in the grid  
- Third index = layer within that cell

So `Layers[1][2][0]` means "row 1, column 2, layer 0" (the first layer in that cell).

---

## LayerConfig: The Blueprint for Each Layer

Every layer is defined by a `LayerConfig` struct. This is where things get interesting because different layer types need different information:

```
LayerConfig {
    Type: LayerDense        ← What kind of layer? (Dense, Conv2D, LSTM, etc.)
    Activation: ReLU        ← What activation function?
    
    // Size information
    InputSize: 1024
    OutputSize: 512
    
    // The actual learnable parameters!
    Weights: [524288]float32    ← 1024 × 512 = 524,288 weight values
    Bias: [512]float32          ← One bias per output neuron
    
    // For backpropagation - stores intermediate values
    Activations: [...]float32   ← Cached outputs (before activation)
    PreActivations: [...]float32 ← Cached outputs (after activation)
    
    // Gradients - computed during backward pass
    WeightGradients: [...]float32
    BiasGradients: [...]float32
}
```

For different layer types, additional fields come into play:

```
Conv2D Layer:
    InputHeight, InputWidth: 28, 28    ← Image dimensions
    InputChannels: 3                    ← RGB = 3 channels
    Filters: 32                         ← Number of output filters
    KernelSize: 3                       ← 3×3 convolution kernel
    Stride: 1                           ← Move 1 pixel at a time
    Padding: 1                          ← Pad edges to preserve size

Attention Layer:
    DModel: 512                         ← Model dimension
    NumHeads: 8                         ← Number of attention heads
    SeqLength: 128                      ← Sequence length
    
    QWeights, KWeights, VWeights        ← Query, Key, Value projections
    OutputWeights                       ← Final projection

LSTM Layer:
    HiddenSize: 256                     ← Hidden state dimension
    Wi, Wf, Wg, Wo                      ← Gate weight matrices
    Ui, Uf, Ug, Uo                      ← Recurrent weight matrices
    Bi, Bf, Bg, Bo                      ← Gate biases
    HiddenState, CellState              ← Persistent state across steps
```

---

## The Forward Pass: What Happens Inside

Let's trace through exactly what happens when you call `network.Forward(input)`:

### Step 1: Start the Clock
```go
startTime := time.Now()
```
Loom tracks execution time for performance monitoring.

### Step 2: Initialize Current Data
```go
currentData := input  // Start with your input
```

### Step 3: Loop Through the Grid
```go
for row := 0; row < gridRows; row++ {
    for col := 0; col < gridCols; col++ {
        for layer := 0; layer < layersPerCell; layer++ {
            // Process this layer...
```

### Step 4: For Each Layer, Switch on Type

This is where the real work happens. For a Dense layer:

```
Dense Layer Forward Pass:

    Input Vector: [x₁, x₂, x₃, ..., xₙ]     (size: 1024)
           │
           ▼
    ┌─────────────────────────────────────────────┐
    │  For each output neuron j:                  │
    │                                             │
    │    preActivation[j] = bias[j] +             │
    │                       Σ(weights[j,i] × xᵢ)  │
    │                                             │
    │  This is a dot product + bias               │
    └─────────────────────────────────────────────┘
           │
           ▼
    Pre-activation: [z₁, z₂, z₃, ..., zₘ]   (size: 512)
           │
           ▼
    ┌─────────────────────────────────────────────┐
    │  Apply activation function to each element: │
    │                                             │
    │  ReLU:    max(0, z)                         │
    │  Sigmoid: 1 / (1 + e⁻ᶻ)                     │
    │  Tanh:    (e²ᶻ - 1) / (e²ᶻ + 1)             │
    └─────────────────────────────────────────────┘
           │
           ▼
    Output Vector: [y₁, y₂, y₃, ..., yₘ]    (size: 512)
```

### Step 5: Cache for Backprop

Crucially, Loom saves intermediate values:

```go
layer.PreActivations = preActivation  // Before activation
layer.Activations = output            // After activation
```

Why? Because during backpropagation, we need to know what the values were at this layer to compute gradients correctly.

### Step 6: Output Becomes Next Input

```go
currentData = output  // This layer's output is the next layer's input
```

### Step 7: Return Final Output

After traversing all cells and layers:

```go
return currentData, time.Since(startTime)
```

---

## The Backward Pass: Gradients Flow in Reverse

The backward pass is like running the forward pass in reverse, but instead of computing outputs, we're computing *how much each weight contributed to the error*.

```
Forward Pass Direction:
    Input ──────────────────────────────────────▶ Output
    
Backward Pass Direction:
    Input ◀────────────────────────────────────── Output
                                                    │
                                                    │ We start here with
                                                    │ "how wrong were we?"
                                                    ▼
                                              gradOutput
```

Here's what happens:

### Step 1: Start with Output Gradient

```go
gradOutput := lossGradient  // e.g., (predicted - target) for MSE
```

This gradient tells us: "For each output value, how much should it change to reduce the loss?"

### Step 2: Reverse Through Layers

```go
for row := gridRows-1; row >= 0; row-- {      // Bottom to top
    for col := gridCols-1; col >= 0; col-- {  // Right to left
        for layer := layersPerCell-1; layer >= 0; layer-- {  // Top to bottom in cell
            // Backprop through this layer...
```

### Step 3: For Each Layer, Compute Three Things

For a Dense layer, we need:

```
Backward Pass Through Dense Layer:

    gradOutput: "How should each output change?"
         │
         ├──────────────────────────────────────────────┐
         │                                              │
         ▼                                              ▼
    ┌─────────────────┐                    ┌────────────────────────┐
    │ Gradient w.r.t. │                    │ Gradient w.r.t.        │
    │ INPUT           │                    │ WEIGHTS & BIAS         │
    │                 │                    │                        │
    │ gradInput[i] =  │                    │ gradWeight[j,i] =      │
    │  Σ(gradOut[j] × │                    │  gradOut[j] × input[i] │
    │   weights[j,i] ×│                    │                        │
    │   act'(pre[j])) │                    │ gradBias[j] =          │
    │                 │                    │  gradOut[j]            │
    └────────┬────────┘                    └────────────────────────┘
             │                                        │
             ▼                                        ▼
    Becomes gradOutput                      Accumulated for
    for previous layer                      weight updates
```

The key insight: **gradients tell us which direction to move weights to reduce error**.

---

## Tensors: Multi-Dimensional Data Containers

Loom uses a generic `Tensor[T]` type for handling data. This is more than just a slice—it understands shape and memory layout.

```
Tensor[float32] {
    Data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]   ← Flat storage in memory
    Shape: [2, 3]                           ← Conceptual shape: 2 rows, 3 cols
    Strides: [3, 1]                         ← How to navigate dimensions
}

Visual representation:
    
    ┌─────┬─────┬─────┐
    │ 1.0 │ 2.0 │ 3.0 │  ← Row 0
    ├─────┼─────┼─────┤
    │ 4.0 │ 5.0 │ 6.0 │  ← Row 1
    └─────┴─────┴─────┘
      ↑     ↑     ↑
     Col0  Col1  Col2

To access element [1, 2]:
    index = 1 * stride[0] + 2 * stride[1]
          = 1 * 3 + 2 * 1
          = 5
    Data[5] = 6.0  ✓
```

Why does this matter? Because neural network operations work on tensors of various shapes:

- **Dense**: Input `[batch, features]`, Weights `[in, out]`
- **Conv2D**: Input `[batch, channels, height, width]`
- **Attention**: Input `[batch, sequence, features]`

The tensor abstraction handles all these uniformly.

---

## Weight Initialization: The Starting Point Matters

When you create a layer, the weights need initial values. Random noise won't work—if weights are too large, activations explode. Too small, and gradients vanish.

Loom uses different initialization strategies depending on layer type:

```
Dense Layer - Xavier/Glorot Initialization:
    
    stddev = sqrt(2 / (fan_in + fan_out))
    
    Example: Dense 1024 → 512
        stddev = sqrt(2 / (1024 + 512)) = sqrt(2/1536) ≈ 0.036
        
    Weights drawn from Normal(mean=0, stddev=0.036)
    
    Why this works:
    ├── Keeps variance roughly constant through the network
    ├── Gradients don't explode or vanish
    └── Works well with tanh and sigmoid activations


Conv2D Layer - He Initialization:
    
    stddev = sqrt(2 / fan_in)
    
    Where fan_in = input_channels × kernel_height × kernel_width
    
    Example: Conv2D with 3 input channels, 3×3 kernel
        fan_in = 3 × 3 × 3 = 27
        stddev = sqrt(2/27) ≈ 0.27
    
    Why this works:
    └── Specifically designed for ReLU activations
        (ReLU kills half the values, so we compensate with larger init)


LSTM Layer - Orthogonal + Forget Bias:
    
    Hidden-to-hidden weights: Orthogonal initialization
    (Preserves norm during recurrent steps)
    
    Forget gate bias: Initialized to 1.0
    (Encourages "remembering" by default)
```

---

## File Organization: Where to Find Things

The `nn/` directory has 58 Go files. Here's how they're organized logically:

```
Core Architecture (start here to understand the system):
├── nn.go              ← Package documentation
├── types.go           ← Network, LayerConfig, LayerType definitions
└── backend.go         ← Backend interface for compute abstraction

Layer Implementations (one file per layer type):
├── dense.go           ← Fully-connected layers
├── cnn.go             ← Convolutional layers
├── conv1d.go          ← 1D convolution for sequences
├── attention.go       ← Multi-head attention
├── rnn.go             ← Simple recurrent network
├── lstm.go            ← LSTM with gates
├── softmax.go         ← 10 softmax variants (!)
├── layernorm.go       ← Layer normalization
├── rmsnorm.go         ← RMS normalization (Llama-style)
├── embedding.go       ← Token embeddings
├── swiglu.go          ← Gated linear unit (modern LLMs)
├── rope.go            ← Rotary position embeddings
├── sequential.go      ← Wrapper for sequential layers
├── parallel.go        ← Run layers in parallel
└── residual.go        ← Skip connections

Execution (forward and backward passes):
├── forward.go         ← High-level forward propagation
├── backward.go        ← High-level backward propagation
├── step_forward.go    ← Step-by-step forward with state
├── step_backward.go   ← Step-by-step backward with state
├── activations.go     ← ReLU, sigmoid, tanh implementations
└── tween.go           ← Neural Tweening algorithm (3600+ lines!)

Training:
├── training.go        ← Training loop, loss functions
├── training_utils.go  ← High-level TrainWithStepping
├── optimizer.go       ← SGD, AdamW, RMSprop
└── scheduler.go       ← Learning rate schedules

Serialization:
├── serialization.go               ← Save/load models
├── serialization_multiprecision.go ← Multi-precision weights
├── safetensors.go                 ← Load HuggingFace format
├── load_generic.go                ← Auto-detect model format
└── load_transformer.go            ← Load Llama-style transformers

Observability:
├── introspection.go   ← Runtime method discovery
├── telemetry.go       ← Network blueprints
├── observer.go        ← Layer monitoring
├── evaluation.go      ← Accuracy metrics
└── registry.go        ← Dynamic layer creation

Utilities and Analysis:
├── import_model.go    ← Build networks from external weights
├── grouping.go        ← Tensor grouping for complex layers
├── grafting.go        ← Graft parallel branches from multiple models
├── ensemble.go        ← Complementary model matching
├── correlation.go     ← Feature correlation analysis
└── clustering.go      ← K-means clustering helpers

GPU Acceleration:
├── gpu.go             ← WebGPU initialization
├── gpu_integration.go ← Upload/download weights + GPU wiring
├── apply_gradients_gpu.go ← GPU gradient updates
├── conv2d_gpu.go      ← Conv2D GPU kernels
└── attention_gpu.go   ← Attention GPU kernels
```

---

## Next Steps

Now that you understand the architecture, explore these topics in depth:

- **[Layers Guide](./layers.md)** - How each layer type works internally
- **[GPU Layers Guide](./gpu_layers.md)** - MHA, KV cache, GQA, RoPE, all GPU layer types
- **[FP4 Quantization](./fp4_quantization.md)** - NVFP4 E2M1 bitwise quantization and VRAM optimization
- **[Transformer Inference](./transformer.md)** - Prefill/decode loop, KV caching, GPU LLM inference
- **[Tokenizer Package](./tokenizer.md)** - BPE tokenizer, LLMEngine, sampling, chat templates
- **[Architecture Generation](./architecture.md)** - BrainType, ArchConfig, NAS random generation, diverse network builder
- **[Clustering Utilities](./clustering.md)** - KMeansCluster, silhouette score, NAS analysis pipeline
- **[Network Grafting](./grafting.md)** - Combining trained networks into a Super-Hive
- **[Type Conversion](./type_conversion.md)** - Multi-precision dtypes, F16/BF16/F4 encoding, ConvertSlice
- **[KMeans Layer](./kmeans.md)** - Differentiable clustering and recursive concept learning
- **[Training Guide](./training.md)** - The complete training process
- **[Tween Guide](./tween.md)** - The bidirectional training algorithm
- **[Serialization](./serialization.md)** - Save/load models, safetensors, geometry loader
- **[Quick Reference](./quick_reference.md)** - Concise code examples
