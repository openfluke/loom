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
*   **Bandwidth Optimization**: Targets a 75-80% reduction in weight size, specifically designed to break the memory bandwidth bottleneck on consumer hardware (e.g., Turing/GTX 1650 Super).

### II. Polymorphic Layer-Morphing (POLY)
Every layer is a polymorphic unit capable of **metamorphosis**.
*   **Dynamic DType Management**: Uses a `WeightStore` system with an FP32 "Master" source of truth.
*   **Metamorphosis**: Layers can swap between active numerical representations (e.g., FP32 -> INT8 -> FP4) instantly during Quantization-Aware Training (QAT) or inference benchmarks.
*   **Native Fast-Paths**: The dispatcher automatically selects specialized arithmetic paths for standard Go types to ensure "actual" performance gains rather than mere simulation.

### III. Volumetric Tensor Dispatch (VTD)
Replaces the traditional 2D sequential execution with a **3D Volumetric Coordinate System** (Depth, Row, Col, Layer).
*   **Spatial Hopping**: Enables recursive passing and spatial layer-hopping, simulating the feedback loops of biological neural systems.
*   **Tiling Strategy**: Built for future GPU integration where each 3D coordinate maps to a Shared Memory workgroup tile, aiming for a **70+ token/s** performance ceiling for models like SmolLM2.

## Performance Benchmarking
A comprehensive benchmarking suite is provided to measure the speed and memory efficiency of the polymorphic dispatcher.

### Running the Benchmarks
To see the system in action and view the performance/memory metrics for all 21 types:
```bash
go run tva/poly/example.go
```

### Key Performance Insights
*   **Memory Savings**: Low-bit types like **Binary (1-bit)** achieve up to **96.9% memory reduction** compared to FP32.
*   **The "Simulation Tax"**: On CPU, low-bit types may appear slower due to the logic required to simulate bit-level constraints.
*   **The WebGPU Promise**: Moving from CPU to GPU flips the bottleneck; the massive memory savings of FP4/INT8 allow the GPU to bypass the 192 GB/s bandwidth wall, leading to 10x speedups in token generation.
