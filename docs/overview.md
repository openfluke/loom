# M-POLY-VTD: Architecture Overview

**Multi-numerical POLYmorphic Volumetric Tiled-tensor Dispatcher**

M-POLY-VTD is a neural inference and training engine built from first principles in Go. It treats a neural network not as a sequential stack of layers, but as a **spatial 3D grid** where each cell can hold any layer type, and every layer can morph its numerical precision on demand.

> [!NOTE]
> Current version: **0.80.0 (Native Ship)**. Previous: **0.79.0 (Bedrock Validation)**. The Loom stack is **Go + `poly/asm` + WebGPU** (`github.com/openfluke/webgpu@v1.0.4`, wgpu-native v29). **Numerical Tiling (SC/MC)** is live across all 21 DTypes; **Dense forward** can use Plan 9 assembly via `UseAsmForward`. **v0.80** ships **ENTITY** (`.entity`) native checkpoints, Lucy **[8] ENTITY Talk**, and multi-platform GPU validation. **Planet Bridging** POC (planets→Loom) is complete in [`planetbridging/`](../planetbridging/) and releases after Loom. See [`v080_release.md`](v080_release.md) and [`entity.md`](entity.md).

---

## The Full Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        M-POLY-VTD ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │               POLYGLOT BINDINGS (C-ABI FFI Layer)                    │   │
│  │  Python │ TS (@openfluke/welvet) │ C# │ Java │ Dart │ WASM Browser    │   │
│  └─────────────────────────────┬────────────────────────────────────────┘   │
│                                │                                            │
│                                ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                  VolumetricNetwork (3D Grid)                          │   │
│  │                                                                      │   │
│  │   Depth × Rows × Cols × LayersPerCell                                │   │
│  │                                                                      │   │
│  │   ┌───────────┐  ┌───────────┐  ┌───────────┐                       │   │
│  │   │ (0,0,0,0) │  │ (0,0,1,0) │  │ (0,0,2,0) │   ← Depth=0, Row=0  │   │
│  │   │VolumetricL│  │VolumetricL│  │VolumetricL│                       │   │
│  │   │ayer       │  │ayer       │  │ayer       │                       │   │
│  │   └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                       │   │
│  │         │               │               │                            │   │
│  │   ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐                       │   │
│  │   │ (0,1,0,0) │  │ (0,1,1,0) │  │ (0,1,2,0) │   ← Depth=0, Row=1  │   │
│  │   └───────────┘  └───────────┘  └───────────┘                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                │                                            │
│              ┌─────────────────┼──────────────────────┐                     │
│              ▼                 ▼                      ▼                     │
│  ┌───────────────┐  ┌──────────────────┐  ┌───────────────────────────┐    │
│  │  CPU Backend  │  │  Step mesh engine │  │  WebGPU Backend (WGPU)    │    │
│  │               │  │                  │  │                           │    │
│  │ ForwardPoly-  │  │ StepForward  │  │ BeginFrame / FlushFrame   │    │
│  │ morphic[T]    │  │ StepBackward │  │ DispatchForwardLayer      │    │
│  │               │  │ Tween (NTP) │  │ DispatchBackwardLayer     │    │
│  │ All 21 DTypes │  │                  │  │ WGSL compute shaders      │    │
│  └───────────────┘  └──────────────────┘  └───────────────────────────┘    │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │              WeightStore (Morphic Precision Engine)                   │   │
│  │                                                                      │   │
│  │  Master []float32  ──┬──▶  Versions[DTypeFP4]  []int8               │   │
│  │  (Source of Truth)   ├──▶  Versions[DTypeInt8] []int8               │   │
│  │                      ├──▶  Versions[DTypeBinary] []int8             │   │
│  │                      └──▶  GPUWeights[DTypeFloat32] *wgpu.Buffer    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     DNA Engine                                        │   │
│  │  ExtractDNA ──▶ LayerSignature[] ──▶ CompareNetworks ──▶ SI Score    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Six Core Pillars

### I. Multi-Numerical Architecture (the "M")

The engine natively dispatches forward and backward passes across **21 distinct numerical types** (DTypes), from `float64` all the way down to 1-bit `binary`. Each layer stores its weights in a `WeightStore` that holds a `float32` master copy plus optional converted versions for inference.

```
DType Hierarchy:
┌────────────────────────────────────────────────────────┐
│ High-Precision  │ Float64, Int64, Uint64               │
│ Standard        │ Float32, Int32, Uint32, Int16, Uint16│
│ Optimized       │ Float16, BFloat16, Int8, Uint8       │
│ Low-Bit         │ FP8E4M3, FP8E5M2, Int4, Uint4, FP4  │
│ Extreme         │ Int2, Uint2, Ternary, Binary         │
└────────────────────────────────────────────────────────┘
```

Layers are not restricted to a single precision. The dispatcher reads `layer.DType`, fetches the right version from the `WeightStore`, and falls back to the master FP32 weights if no converted version exists. See [numerical_types.md](./numerical_types.md) for the full breakdown.

### II. Polymorphic Layer-Morphing (the "POLY")

Every layer is a **polymorphic processing unit**. Its numerical representation can be changed at any time via `WeightStore.Morph(dtype)` without reallocating the layer structure. The master FP32 weights are never destroyed—they remain the source of truth.

```
Metamorphosis sequence:
  FP32 (training) ──▶ Morph(INT8) ──▶ Morph(FP4) ──▶ Morph(Binary)
       ▲                                                     │
       └──── Unpack(dtype) ──── always recoverable ─────────┘
```

After gradients are applied via `WeightStore.ApplyGradients`, all cached low-bit versions are **automatically cleared**, forcing re-quantization on the next forward pass.

### III. Volumetric Tensor Dispatch (the "VTD")

The network is a **4D array** of `VolumetricLayer` values indexed by `(Depth, Row, Col, LayerIndex)`. The flattened index is:

```
idx = z * Rows * Cols * LayersPerCell
    + y * Cols * LayersPerCell
    + x * LayersPerCell
    + l
```

Data flows through the grid in reading order: Z outer loop, then Y, then X, then L. This gives the programmer a spatial metaphor to compose complex non-linear topologies.

#### Remote Links (Spatial Hopping)

Any layer can set `IsRemoteLink = true` and point to any other coordinate via `TargetZ / TargetY / TargetX / TargetL`. When the step mesh engine fires that layer, it reads input from the *target* coordinate's output buffer instead of the preceding layer. This enables biological-style feedback loops anywhere in the grid.

```
Normal flow:          Remote link (skip connection):
 (0,0,0)               (0,0,0)
    │                     │    ◄────────────────────────┐
    ▼                     ▼                             │
 (0,0,1)               (0,0,1)  ─ IsRemoteLink ──▶ (0,2,3)
    │                     │
    ▼                     ▼
 (0,0,2)               (0,0,2)
```

### IV. The Dispatcher Pattern

`DispatchLayer[T]` and `DispatchLayerBackward[T]` are **generic runtime jump tables**. They inspect `layer.Type` and call the correct polymorphic function, returning `(preAct, postAct)` tensors of the same type `T`. The separation from the grid traversal loop makes GPU kernel fusion possible—the driver can look ahead and pre-load the next tile's weights while the current tile computes.

```go
func DispatchLayer[T Numeric](layer *VolumetricLayer, input, skip *Tensor[T]) (preAct, postAct *Tensor[T])
```

There are 19 `LayerType` values routed here. An unknown type falls through to `DenseForwardPolymorphic`.

**Numerical tiling** is orthogonal to volumetric traversal: `ForwardPolymorphic` can walk the grid in spatial tiles or sequentially (`network.UseTiling`). **CPU** layers use a **single** tile map (`CPUTileSizes`). **GPU** layers carry **two** maps (`GPUSCTileSizes`, `GPUMCTileSizes`); **`EnableMultiCoreTiling`** on `VolumetricNetwork` selects MC vs SC dispatch (see [dispatch.md](./dispatch.md) and [gpu.md](./gpu.md)).

### V. The Step Mesh Engine

Unlike `ForwardPolymorphic`, which executes the entire network per input in one pass, `StepForward` fires **all layers simultaneously** every clock cycle. Each layer reads from the previous cycle's output buffer (`LayerData`) and writes to `NextBuffer`. After all layers have fired, the buffers are swapped. This double-buffering pattern is race-condition-free and supports parallel tile dispatch via goroutines.

### VI. The DNA Engine

`ExtractDNA` converts a network into a slice of `LayerSignature` values. Each signature contains the layer's 3D coordinates, type, DType, and a **normalized** (unit-vector) representation of its weights after precision simulation. `CompareNetworks(dna1, dna2)` then uses cosine similarity to produce an `OverallOverlap` score and identifies `LogicShift` events where a functional pattern has migrated to a different spatial coordinate.

---

## Key Types at a Glance

| Type | File | Role |
|:-----|:-----|:-----|
| `VolumetricNetwork` | `poly.go` | The 3D grid container |
| `VolumetricLayer` | `poly.go` | A single processing unit with coordinates |
| `WeightStore` | `weights.go` | Master FP32 + versioned low-bit storage |
| `Tensor[T Numeric]` | `poly.go` | Generic data container with `Shape` and `Nested` |
| `DType` | `poly.go` | 21-value enum for numerical types |
| `LayerType` | `poly.go` | 19-value enum for layer kinds |
| `WGPUContext` | `wgpu_context.go` | GPU device, queue, pipeline cache |
| `StepState[T]` | `step.go` | Double-buffered temporal mesh state |
| `NetworkDNA` | `dna.go` | `[]LayerSignature` topological blueprint |
| `TrainingConfig` | `training.go` | Epochs, LR, loss type, GPU flag |

---

## The `Tensor[T]` Type

```go
type Tensor[T Numeric] struct {
    Data   []T
    DType  DType
    Shape  []int
    Nested []*Tensor[T]  // activation tree for Parallel/Sequential layers
}
```

`Nested` is the key structural innovation. During a `ParallelForward` pass, each branch produces its own `preAct` tensor, and these are stored in `Nested` on the returned preAct. The backward pass reads them back, routing gradients to the correct branch without any external bookkeeping. This recursive tree property makes arbitrary nesting of `Parallel` and `Sequential` layers fully differentiable.

---

## Performance Snapshot

From the README benchmark table, measured on GTX 1650 Super (Vulkan/WebGPU). **v0.80** production GPU uses **openfluke/webgpu v1.0.4** (wgpu-native v29); Lucy Poly Talk on RTX 3050 Mobile reaches ~**69 tok/s decode** / ~**492 tok/s prefill** for SmolLM2-135M Q4 (custom WGSL — not Ollama-class, but validated on Metal, Win ARM64, Intel, and NVIDIA Vulkan).

| Layer type | CPU Tiled | GPU | Speedup |
|:-----------|:----------|:----|:--------|
| Dense | 5.42ms | 400µs | 13.6x |
| CNN 1D | 4.34ms | 195µs | 22.3x |
| CNN 2D | 182ms | 100µs | 1826x |
| CNN 3D | 1522ms | 200µs | 7602x |
| RMSNorm | 1.16ms | 103µs | 11.3x |

End-to-end GPU training (20 epochs):

| Architecture | CPU | GPU | Speedup |
|:-------------|:----|:----|:--------|
| Dense MLP (128→512→512→8) | 12.1s | 693ms | 17.5x |
| CNN 2D (3ch×32×32 → 16f→32f→8) | 1m57s | 1.81s | 64.8x |
| Deep Dense (128→512×4→8) | 31.7s | 1.23s | 25.7x |

---

## Next Steps

- [v080_release.md](./v080_release.md) — **v0.80.0** ENTITY, WebGPU v1.0.4, GPU validation, Planet Bridging POC
- [entity.md](./entity.md) — `.entity` native checkpoint format
- [numerical_types.md](./numerical_types.md) — DType system, WeightStore, Metamorphosis
- [layers.md](./layers.md) — Every layer type in detail
- [dispatch.md](./dispatch.md) — The dispatcher pattern and 3D coordinates
- [training.md](./training.md) — Forward/backward, optimizers, Tween
- [gpu.md](./gpu.md) — WebGPU backend and BeginFrame/FlushFrame pattern
- [step.md](./step.md) — The step mesh engine
- [quick_reference.md](./quick_reference.md) — Common code snippets
