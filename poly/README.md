# M-POLY-VTD Architecture
**Multi-numerical POLYmorphic Volumetric Tiled-tensor Dispatcher**

M-POLY-VTD is a next-generation neural inference engine designed for high-performance, mixed-precision workloads. It treats the neural network not as a sequential stack, but as a **spatial 3D grid** where layers can morph their numerical precision on-the-fly.

## Core Pillars

### I. Multi-Numerical Architecture (M-POLY)
The engine provides a "Universal Dispatcher" supporting native forward and backward passes across **21 distinct numerical types**.

*   **Supported Types**: 
    *   **High-Precision**: Float64, Int64, Uint64
    *   **Standard**: Float32, Int32, Uint32, Int16, Uint16
    *   **Optimized**: Float16, BFloat16, Int8, Uint8
    *   **Low-Bit**: FP8 (E4M3/E5M2), Int4, Uint4, FP4 (E2M1)
    *   **Extreme**: Int2, Uint2, Ternary (-1, 0, 1), Binary (1-bit)
*   **CNN/ConvTransposed Support**: Native support for **LayerCNN1-3** and **LayerConvTransposed1-3** (1D, 2D, and 3D) across all numerical types.
*   **Transformer Support**: Native **LayerMultiHeadAttention** (with RoPE, GQA/MQA, and Causal Masking), **LayerSwiGLU**, **LayerRMSNorm**, and **LayerLayerNorm**.
*   **RNN Support**: Native support for **LayerRNN** and **LayerLSTM** with full polymorphism.
*   **Universal Softmax Engine**: Exhaustive **LayerSoftmax** support including Standard, Grid, Hierarchical, Gumbel, Masked, Sparsemax, and Entmax (1.5) across all 21 types.
*   **Universal Nesting & Training**: Support for **LayerParallel** (add, avg, concat, filter/MoE) and **LayerSequential** with recursive **Activation/Gradient Trees** for deep, trainable hierarchies.
*   **Embedding/KMeans Support**: Efficient **LayerEmbedding** lookups and differentiable **LayerKMeans** clustering.
*   **Bandwidth Optimization**: Targets a 75-80% reduction in weight size, specifically designed to break the memory bandwidth bottleneck on consumer hardware (e.g., Turing/GTX 1650 Super).

### II. Polymorphic Layer-Morphing (POLY)
Every layer is a polymorphic unit capable of **metamorphosis**.
*   **Dynamic DType Management**: Uses a `WeightStore` system with an FP32 "Master" source of truth.
*   **Metamorphosis**: Layers can swap between active numerical representations (e.g., FP32 -> INT8 -> FP4) instantly during Quantization-Aware Training (QAT) or inference benchmarks.
*   **Native Fast-Paths**: The dispatcher automatically selects specialized arithmetic paths for standard Go types to ensure "actual" performance gains rather than mere simulation.

### III. Volumetric Tensor Dispatch (VTD)
Replaces the traditional 2D sequential execution with a **3D Volumetric Coordinate System** (Depth, Row, Col, Layer).
*   **Spatial Hopping**: Enables recursive passing and 3D spatial routing via `IsRemoteLink`. Any layer can "hop" across coordinates, simulating biological feedback loops.
*   **Recursive Backpropagation**: A hierarchical training system that caches intermediates in a "Neural Tree," allowing signals to flow bidirectionally through arbitrary nesting.
*   **Tiling Strategy**: Built for future GPU integration where each 3D coordinate maps to a Shared Memory workgroup tile, aiming for a **70+ token/s** performance ceiling for models like SmolLM2.

## Performance Benchmarking
A comprehensive benchmarking suite is provided to measure the speed and memory efficiency of the polymorphic dispatcher.

### Running the Benchmarks
To see the system in action and view the performance/memory metrics for all 21 types:
```bash
go run tva/poly/example.go
```

### Key Performance Insights
*   **Memory Savings**: Low-bit types achieve up to **96.9% memory reduction** across all major layers (MHA, Conv, LSTM, SwiGLU).
*   **Deterministic CPU Anchor**: On CPU, the engine handles all 21 types with explicit loops, providing a low-power, **Deterministic Reference** for hardware-agnostic research.
*   **GPU "High Gear" (Tiling)**: Moving from CPU to GPU flips the bottleneck. By using **Tiling (L1 Caching)** and massive parallelism, the GPU bypasses the 192 GB/s bandwidth wall. The same deterministic logic from the CPU is executed at hardware-saturation speeds, aiming for **90+ tokens/s** for models like SmolLM2 135M.

---

## The Bedrock Philosophy
M-POLY-VTD is a **"Bedrock Edition"** neural engine. Unlike standard frameworks that build on top of high-level abstractions, this architecture is designed at the bit-level to bypass the physical memory limitations of consumer hardware.

*   **Shader-First Design**: The Go implementation is a direct blueprint for GPU kernels.
*   **Hardware-Agnostic**: By supporting 21 numerical types, we can run on anything from a GTX 1650 to an H100 by simply "Morphing" the precision to what the specific silicon prefers.

## Architectural Design Choices

### 1. Unified Package Structure (`poly/`)
**Decision**: Keeping all layers (`dense.go`, `mha.go`, etc.) in the same `poly` package.
*   **Rationale**: To avoid circular dependency hell common in Go polymorphic systems. All layers share a unified view of the `VolumetricLayer` and `WeightStore` types, allowing for seamless, fast internal dispatch without the overhead of public interfaces.

### 2. The Morphic WeightStore (`WeightStore`)
**Decision**: Using a master `float32` weight-set with a `map[DType]any` versioning system.
*   **Rationale**: This is the heart of "Metamorphosis." It allows a layer to hold multiple numerical personalities at once. We can keep "Master" weights for training and instantly swap to packed `FP4` for an inference burst without re-allocating buffers.

### 3. Volumetric 3D Dispatch (`VTD`)
**Decision**: Replacing 1D sequential stacks with a 3D coordinate-based grid (`Depth`, `Row`, `Col`).
*   **Rationale**: Standard 1D stacks are a bottleneck. The 3D grid maps directly to **GPU workgroup tiles**. It also enables "Spatial Hopping"—recursive feedback loops that mimic biological neural firing. By treating the network as a mesh, we unlock non-linear data flows (Parallel Expert Gating, Skip-Connections) that are impossible in sequential pipelines.

### 4. Recursive Neural Trees (`Tensor.Nested`)
**Decision**: Implementing a recursive `Nested` field in the `Tensor` struct.
*   **Rationale**: To support nesting (`Parallel`/`Sequential`) without losing the ability to train. This creates an **Activation Tree** during the forward pass and a **Gradient Tree** during the backward pass, establishing a "Plug-and-Learn" bedrock where any complex sub-architecture is automatically differentiable.

### 4. Explicit Numerical Fast-Paths
**Decision**: Using manual `switch` statements and type-casting instead of reflection.
*   **Rationale**: In high-speed inference, reflection is too slow. We write the `INT8` and `FLOAT32` loops explicitly to ensure the compiler generates the fastest possible arithmetic for the "Reference Logic."

### 5. The "Simulation vs. Throughput" Strategy
**Decision**: Supporting types the CPU doesn't natively have (like FP4, 2-bit, 1-bit).
*   **Rationale**: We are building the **Logic Bedrock** first. On CPU, these incur a "Simulation Tax," but on GPU they become **Native Bit-Packed Payloads**, which is where the 10x performance leap occurs.

---

## The 3 Planes of Polymorphism (Hardcore Edition)
M-POLY-VTD pushes Go’s type system into a realm of fluid identity that exceeds standard AI frameworks. It operates across three distinct planes:

### 1. Parametric Polymorphism (Generics)
Utilizes the `[T Numeric]` constraint system. The engine is "Tensor-Blind"; it doesn't care if the underlying signal is `float32`, `int16`, or `uint8`. It processes the mesh math as a universal operation, enabling a single codebase to support any tensor format.

### 2. Ad-hoc Polymorphism (The Dispatcher)
The `DispatchLayer` registry acts as a high-speed Runtime Jump Table. A 3D coordinate in the mesh only assumes its "Functional Identity" (Dense, MHA, SwiGLU) at the moment of execution, allowing for infinite spatial variety within the same volumetric structure.

### 3. Numerical Metamorphosis (Dynamic Identity)
This is the "Bedrock" secret. Unlike static frameworks where a layer has a fixed type, our layers exhibit **Metamorphosis**. A single layer can exist as **FP32** (for precision), morph to **INT8** (for training stability), and project into **FP4/Binary** (for inference throughput) instantly without re-allocating memory.

---

## The GPU "Fusion" Secret: Why the Dispatcher Refactor Matters
You might wonder why we moved the switch statement into a `DispatchLayer` registry. On CPU, it looks like a simple "cleanup," but on GPU, it is a **Mission-Critical Optimization**:

### 1. Avoiding "Thread Divergence"
On a GPU, thousands of threads run in blocks. If those threads hit a messy, nested switch statement inside a loop, they will "diverge" (some threads wait while others branch). By isolating the dispatch, we enable **Kernel Fusion**—the GPU can launch one massive shader that handles an entire "Tile" of the 3D grid if the layers are the same type.

### 2. Batched Metamorphosis
When a block of layers needs to "Morph" (e.g., FP32 -> FP4), the GPU is most efficient when it does this in **Parallel Batches**. The `DispatchLayer` structure allows the engine to group these memory switches together, performing a single "Massive Bit-Pack" rather than 100 small ones.

### 3. Asynchronous Predispatch
Because the Dispatcher is decoupled from the 3D Coordinate loop, the GPU driver can "look ahead." While it calculates the math for Layer (Z=1), it can already be "Predispatching" the weights for Layer (Z=2) into the fast Shared Memory (SRAM).

---

## The Path to 70+ Tokens/Sec
This architecture is specifically optimized for Turing-class GPUs (like the GTX 1650 Super).
1.  **Stage 1 (Current)**: Build the Universal Dispatcher and bit-logic in Go.
2.  **Stage 2**: Implement the Volumetric Tiling in Shared Memory.
3.  **Stage 3**: Move the "Unpacking Logic" (e.g., eight FP4 values per U32) into WebGPU Shaders.
4.  **Stage 4**: Achieve token generation speeds that bypass the 192 GB/s memory wall by using 80% smaller weight payloads.

*M-POLY-VTD: Universal precision. Volumetric freedom. Bedrock performance.*
