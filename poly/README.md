# M-POLY-VTD Architecture
**Multi-numerical POLYmorphic Volumetric Tiled-tensor Dispatcher**

M-POLY-VTD is a next-generation neural inference engine designed for high-performance, mixed-precision workloads. It treats the neural network not as a sequential stack, but as a **spatial 3D grid** where layers can morph their numerical precision on-the-fly.

### The Loom stack

**Loom’s portable core runs on Go and WebGPU**, with **vendor silicon** (Intel NPU + Qualcomm NPU + Apple Metal GPU today; Google TPU next) plugged in through **`poly/accel`** + external C ABI plugins in [chaosglue `npu/`](https://github.com/openfluke/chaosglue/tree/main/npu). **Networked execution** (`donate_compute_*.go`) is the distributed complement. One volumetric dispatcher; pick the backend per device and workload.

| Backend | Role | Status |
| :--- | :--- | :--- |
| **Go** | Portable CPU: SC/MC tiled loops + Plan 9 SIMD (DotTile fwd, Saxpy bwd on 7 layer types), 21 dtypes — reference + `TrainingModeCPUSimd` | ✅ baseline |
| **WebGPU** | GPU forward / backward / training (WGSL from Go) | ✅ production — **[openfluke/webgpu](https://github.com/openfluke/webgpu) v1.0.4** (wgpu-native v29) |
| **`poly/accel`** | Vendor NPU/TPU/GPU via `libloom_accel_*` (`dlopen`/`LoadLibrary`, CGO) | 🧪 **experimental** — **Intel CPU+NPU** (Linux, OpenVINO) · **Qualcomm/Hexagon NPU** (Windows ARM64, QNN) · **Apple GPU** (macOS, Metal/MPSGraph) · **Google TPU** planned |
| **Network** | Remote inference & compute offload (`donate_compute_*.go`) | 🚧 protocol shipped · live inference wiring next |

**Performance work is GPU-first** for decoder stacks, with **NPU offload** for medium/large MAC-heavy layers where compile tax amortizes. CPU Go paths remain the bedrock parity surface (Lucy **[7]**).

**Vendor accel** = init-once compile + steady infer through `DispatchLayer`. See **[`docs/accelerators.md`](../docs/accelerators.md)** and Lucy menu **[9]** (`nine_layer`).

### Where we are now — **v0.83.0 “Apple Bridge”** (was **v0.82.0 “Snapdragon Bridge”**)

**Device-aware** = **Go** on CPU (Plan 9 **SIMD**: AVX2 `DotTile` + `SaxpyF32AccF64` on x86-64, NEON on ARM64 — forward + backward on Dense, SwiGLU, MHA, CNN1–3, RNN, LSTM via `TrainingModeCPUSimd` / `SetSimdForwardRecursive`), **WebGPU** when `UseGPU` is on, and three experimental accelerator vendors online through `poly/accel`: **Intel CPU+NPU** on Linux (OpenVINO), **Qualcomm/Hexagon** on Windows ARM64 (QNN), and **Apple GPU** on macOS Apple silicon (Metal / MPSGraph). Production GPU uses **openfluke/webgpu v1.0.4** (wgpu-native **v29**).

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
*   **Bandwidth Optimization**: Targets a 75-80% reduction in weight size to ease memory bandwidth limits on typical consumer GPUs via WebGPU.

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

### IV. Hierarchical Spatial Correlation Engine (DNA)
The DNA engine converts neural structures into topological "signatures," enabling high-fidelity comparison across disparate numerical families (e.g., FP64 vs. Binary).
*   **Topological Reconstruction**: Generates a 3D genetic blueprint of the network via `ExtractDNA`.
*   **Similarity Index (SI)**: Quantifies model overlap using directional geometry (Cosine Similarity) rather than raw weight parity.
*   **Logic Drift Detection**: Automatically tracks "Logic Shifts" where functional behavior has migrated across 3D coordinates.
*   **Comparative Evolution**: Designed to map neural development down to the neuron level, identifying overlapping structures in heterogeneous models.

### V. Native Bit-Packed Persistence
The framework provides an **Idempotent Serialization Tunnel** designed for extreme storage efficiency.
*   **Transparent Bit-Packing**: Low-bit models (`FP4`, `Binary`, etc.) are natively packed into bit-streams during I/O, achieving up to **98.4% compression** on disk.
*   **Automated Unpacking**: Models are stored in their native DType but automatically `Unpack` into RAM-compatible formats during deserialization, ensuring high-speed inference.
*   **Bit-Perfect Identity**: Verified across **378/378 permutations** (18 Layers x 21 DTypes) with **0.000000% mathematical divergence**. 
*   **Per-dtype JSON checkpoints**: `SerializeNetwork` writes each layer’s **active `dtype`** plus **native-packed** Base64 (`Native: true`, `Scale` preserved) — not an FP32-only blob. Lucy Dense training reports **Save/Reload PASS** for all 21 types (see `File` KB column: Binary ~17 KB vs Float64 ~5.4 MB on the standard bench).
*   **Idempotency Verified**: Serializing a reloaded model produces a byte-for-byte identical JSON to the original (when round-tripping the same dtype path).

### VI. GPU Backward & Accelerator Roadmap
**WebGPU** is the primary compute path for inference and training. The immediate engineering focus is **finishing GPU backward** for every layer type used in decoder stacks, then extending the same compile model to **NPUs** and **networked execution**.

*   **GPU backward (now):** Dense, RMSNorm, and CNN 1D/2D/3D train end-to-end on GPU. **SwiGLU**, **MHA**, and **Embedding** backward kernels exist but are not fully wired through `DispatchBackwardLayer` — see [GPU layer matrix](#gpu-layer-matrix-status).
*   **Command graphs:** Record full decoder forward/backward into fewer submits (partial precedent: `wgpu_forward.go` single encoder per token).
*   **Intel NPU:** Lower ENTITY / morphed weights to Intel NPU runtime (OpenVINO / driver path) for laptop-class inference.
*   **Qualcomm NPU (shipped — experimental):** QNN / Hexagon delegate for Snapdragon on **Windows ARM64** via a C ABI plugin (`loom_accel_qualcomm.dll`); per-layer `DispatchLayer` offload + drift spectrum through Lucy **[12]** (`snapdragon`). Same weight layouts as ENTITY native packing where possible. See [`docs/snapdragon_npu.md`](../docs/snapdragon_npu.md).
*   **Apple GPU (shipped — experimental):** Metal / MPSGraph plugin (`libloom_accel_apple.dylib`) for **macOS Apple silicon**; a CPU reference (parity anchor) + Metal GPU backend, per-layer `DispatchLayer` offload + drift spectrum through Lucy **[13]** (`apple`), plus a **BF16** wire dtype. See [`docs/apple_metal.md`](../docs/apple_metal.md).
*   **Networking:** Expand **Donate Compute** (`docs/donate_compute.md`) from framed TCP stub to live Transformer/ENTITY offload; remote volumetric segments later.
*   **HA step mesh (later):** Checkpoint/restore per step pulse, failover coordinator, and replicated step state for long-running “living network” deployments — builds on `StepForward` / `StepBackward`.

See **[Performance roadmap](#-performance-roadmap)** and **[Accelerators & distributed](#3-accelerators-networking--distributed)** below.

### VII. Tween (neural target propagation)
A bidirectional learning alternative to traditional backpropagation that bridges the gap between actual activations and idealized targets. **Tween** is our code name; papers often say *target propagation* or *difference target propagation*.
*   **True Target Estimation**: Heuristically estimates what a layer *should* have produced by aggregating importance signals through weights (high-fidelity support for **RNN/LSTM** weight mappings).
*   **Gap-Based Learning**: Updates weights using a Hebbian-style `delta = learningRate * input * gap` logic, bypassing the chain rule for localized, non-differentiable optimization.
*   **Mesh Fidelity (Link Budgets)**: Accurately calculates info-preservation (Cosine Similarity) across the mesh.
*   **Gated Learning**: Automatically prevents weight corruption in "dead layers" (Alignment < 0.2) via dynamic Link Budget gating.

## Performance & Verification
A comprehensive suite is provided to measure the speed, memory, and bit-level fidelity of the polymorphic dispatcher.

### Running checks in this repo

Layer matrices, GPU parity tables, and training transcripts are exercised from **`lucy/`** (`lucy_testing_output/log.txt`). The **seven-layer CPU bedrock suite** writes `lucy_testing_output/seven_layer.txt` (menu **[7]**). The **Intel NPU bridge suite** writes `lucy_testing_output/nine_layer.txt` (menu **[9]**). See [`docs/testing_and_validation.md`](../docs/testing_and_validation.md), [`docs/bedrock_validation.md`](../docs/bedrock_validation.md), and [`docs/accelerators.md`](../docs/accelerators.md).

```bash
go test ./poly/...
```

C-ABI vs public `poly/` surface (export names):

```bash
cd welvet/cabi/internal/check && go run check.go
```

### Lucy GPU & bedrock benchmarks

From repo root, run **`lucy/`** → **[7] seven-layer suite** (CPU parity) or **[8] ENTITY Talk** (GPU inference). Logs: `lucy_testing_output/seven_layer.txt`, `lucy_testing_output/log.txt`.

```bash
cd loom/lucy && go run .
```

**GPU forward** on real layer shapes shows **13×–7600×** vs CPU tiled (see tables below). **Next validation target:** GPU backward parity for **SwiGLU** and **MHA** once wired into `DispatchBackwardLayer`.

---

## GPU layer matrix (status)

What exists **today** for **Go CPU** vs **WebGPU**. Legend: **✅** done · **~** partial · **—** not started · **n/a** no grad

| Layer | Fwd Go CPU | Fwd GPU | Bwd Go CPU | Bwd GPU | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dense** | ✅ SC+MC | ✅ SC+MC | ✅ MC tiled | ✅ | Training E2E ✅ |
| RMSNorm / LayerNorm | ✅ | ✅ | ✅ | ✅ | Training E2E ✅ |
| CNN1 / CNN2 / CNN3 | ✅ SC+MC | ✅ SC+MC | ~ | ✅ | Training E2E ✅ |
| **MHA** | ✅ SC+MC | ✅ SC+MC | ✅ MC | **~** | dQ partial; full bwd **pending** |
| **SwiGLU** | ✅ SC+MC | ✅ SC+MC | ✅ MC | **—** | Not in `DispatchBackwardLayer` yet |
| Embedding | ✅ SC+MC | ✅ SC+MC | ✅ MC | **—** | DX intentionally zero |
| RNN / LSTM | ✅ SC+MC | ✅ SC+MC | ✅ MC | — | CPU-first |
| Residual | ✅ SC+MC | ✅ SC+MC | ✅ MC | n/a | Elementwise |
| ConvTransposed 1/2/3 | ✅ | ✅ | ~ | ~ | |
| Softmax / KMeans / Parallel / Sequential | ✅ | varies | ~ | — | |

### GPU backward completion queue (priority)

1. **SwiGLU** — wire GPU backward into `DispatchBackwardLayer`; parity vs CPU MC
2. **MHA** — complete dK/dV/dW; unify with decoder block training
3. **Embedding** — GPU weight grad path for untied `lm_head`
4. **Command graph** — single compiled encoder per training step / decode token
5. **INT8 / FP4 WGSL** — on-device dequant + quant matmul (see checklist §1.4)

---

### TypeScript / WASM Implementation Verification
To verify the **@openfluke/welvet** isomorphic (Browser/Node.js) bridge, a comprehensive 36-count diagnostic and performance suite is provided.

**Run verification:**
```bash
cd welvet/typescript
npm test
```

**Verified Results (Loom v0.80.0):**
- **[PASS]** Internal WASM Exports (8/8)
- **[PASS]** Network Wrapper Methods (16/16)
- **[PASS]** NEAT Population Methods (8/8)
- **[PASS]** Functional Smoke Tests (Sequential, DNA, SwiGLU, Transformers) (5/5)

### Numerical Tiling Profiles (SC vs MC)
**SC (single-core)** and **MC (multi-core)** tiled dispatch are the CPU baseline on all layers.

*   **SC**: One worker, smaller tiles (e.g. 8×8) — WASM-friendly, deterministic, low overhead.
*   **MC**: Parallel tiles across `runtime.NumCPU()` — Apple Silicon / Ryzen / desktop class.
*   **GPU**: Same layer math recorded into WebGPU command encoders; preferred for inference and training at scale.

### VII. Step Mesh Stability
The **step mesh engine** has been fundamentally stabilized in v0.75.0:
*   **Volumetric Coordinate Guarding**: Fixed nil-pointer panics in sparse grids by implementing explicit `IsDisabled` flags for uninitialized cells.
*   **Coordinate-Based Hopping**: Data flow is now strictly governed by 3D volumetric coordinates (`z, y, x, l`), ensuring stable "Neural Mesh" propagation even in complex recursive skip-connections.
*   **Deterministic Wavefront**: The double-buffering logic was refined to guarantee that the signal wavefront remains bit-perfect across all 21 numerical types.

### TS/WASM Training Showdown Benchmark
Measured in `test.ts` (Node.js/tsx) using the isomorphic `@openfluke/welvet` bridge:

| Layer           | Fwd ms/it   | Train ms   | Init Loss   | Final Loss  | Sanity |
| :-------------- | :---------- | :--------- | :---------- | :---------- | :----- |
| Dense (Linear)  | 0.985       | 166.4      | 0.1488      | 0.1488      | REAL   |
| RMSNorm         | 0.269       | 19.7       | 0.0063      | 0.0063      | REAL   |
| SwiGLU (MLP)    | 5.375       | 1713.9     | 0.0000      | 0.0000      | REAL   |
| Embedding       | 0.498       | 44.6       | 0.0067      | 0.0067      | REAL   |
| Residual Add    | 0.204       | 17.1       | 0.0859      | 0.0859      | REAL   |
| MHA (Fused)     | 0.457       | 32.8       | 0.0216      | 0.0101      | REAL   |

### Key Performance Insights
*   **98.4% Storage Compression**: Binary models are compressed from multi-byte pointers down to 1-bit payloads, breaking the memory bandwidth wall.
*   **0.000000% Divergence**: Verified bit-perfect parity across 378 model permutations.
*   **GPU Inference**: WebGPU kernels deliver massive speedups over CPU tiling, especially for volumetric operations like CNNs (up to 7000x+).
*   **GPU Training**: Full end-to-end GPU backward training is live. **17x–65x** training speedups over CPU on real workloads, with a single command buffer submission per batch (`BeginFrame`/`FlushFrame` pattern).

#### GPU Forward / Inference (CPU Tiling vs GPU)
```text
=== M-POLY-VTD Performance Showdown: CPU Tiling vs GPU Acceleration ===
| Layer type      | CPU (Simple) | CPU (Tiled)  | GPU (WebGPU) | Speedup (vs Tiled) | Deterministic | Sanity        |
|-----------------|--------------|--------------|--------------|-------------------|---------------|---------------|
| Dense (Linear)  | 4.79952ms    | 5.42286ms    | 400.08µs     | 13.55x            | SLIGHTLY OFF ⚠️ | REAL 💎       |
| RNN Cell        | 2.09993ms    | 2.61017ms    | 231µs        | 11.30x            | EXACT ⭐       | REAL 💎       |
| LSTM Cell       | 8.14321ms    | 7.03973ms    | 153.46µs     | 45.87x            | EXACT ⭐       | REAL 💎       |
| CNN 1D          | 8.12412ms    | 4.33881ms    | 194.54µs     | 22.30x            | EXACT ⭐       | REAL 💎       |
| CNN 2D          | 362.33425ms  | 182.6935ms   | 100.07µs     | 1825.66x          | EXACT ⭐       | REAL 💎       |
| CNN 3D          | 10.07534167s | 1.5223089s   | 200.24µs     | 7602.42x          | EXACT ⭐       | REAL 💎       |
| Embedding       | 320.86µs     | 217.05µs     | 109.77µs     | 1.98x             | EXACT ⭐       | REAL 💎       |
| RMSNorm         | 1.16247ms    | 1.15767ms    | 102.77µs     | 11.26x            | INDUSTRY ✅    | REAL 💎       |
| MHA (Attn)      | 210.01µs     | 417.27µs     | 258.55µs     | 1.61x             | BROKEN ❌      | REAL 💎       |
| SwiGLU (MLP)    | 11.48634ms   | 7.83584ms    | 3.08049ms    | 2.54x             | BROKEN ❌      | REAL 💎       |
| Residual Add    | 0s           | 0s           | 953.41µs     | N/A               | BROKEN ❌      | REAL 💎       |
```

#### GPU End-to-End Training (20 epochs, CPU vs GPU)
All runs share a single pre-initialised `WGPUContext`. Weights are copied CPU→GPU before each GPU run for a fair starting-point comparison.

```text
=== M-POLY-VTD Multi-Architecture Training Showdown ===
| Architecture                         | CPU Time | GPU Time | Speedup | CPU Loss Δ | GPU Loss Δ |
|--------------------------------------|----------|----------|---------|------------|------------|
| Dense MLP  (128→512→512→8)           | 12.1s    | 693ms    | 17.5x   | –72.3%     | –71.8%     |
| CNN 1D     (3ch×128 → 32f→64f → 8)  | 29.7s    | 811ms    | 36.6x   | –68.4%     | –67.9%     |
| CNN 2D     (3ch×32×32 → 16f→32f→8)  | 1m57s    | 1.81s    | 64.8x   | –61.2%     | –60.5%     |
| CNN 3D     (2ch×8×8×8 → 8f → 8)     | 3.2s     | 461ms    | 6.9x    | –55.1%     | –54.7%     |
| RMSNorm MLP (128→Dense512→Norm→512→8)| 12.6s    | 711ms    | 17.7x   | –73.1%     | –72.6%     |
| Deep Dense (128→512×4→8)             | 31.7s    | 1.23s    | 25.7x   | –69.8%     | –69.2%     |
```
> Measured on WebGPU (Vulkan), Windows 10. Batch sizes: 64 (Dense/RMSNorm), 32 (CNN1D), 16 (CNN2D), 8 (CNN3D).

#### Per-Layer Gradient Correctness (DX / DW parity, CPU vs GPU)
```text
| Layer          | DX (input grad)     | DW (weight grad)    | Notes                                    |
|----------------|---------------------|---------------------|------------------------------------------|
| Dense          | EXACT ⭐             | EXACT ⭐             | Tiling bug fixed (dyTile indexing)       |
| RMSNorm        | EXACT ⭐             | EXACT ⭐             |                                          |
| CNN 1D         | EXACT ⭐             | EXACT ⭐             |                                          |
| CNN 2D         | EXACT ⭐             | EXACT ⭐             |                                          |
| CNN 3D         | EXACT ⭐             | EXACT ⭐             |                                          |
| Embedding      | — (discrete)        | EXACT ⭐             | DX intentionally zero (index lookup)    |
| MHA            | OK ✅ (dQ)          | — (pending)         | Writes separate dQ/dK/dV buffers        |
| SwiGLU         | BROKEN ❌           | —                   | Not yet in DispatchBackwardLayer         |
```

**GPU backward training support status:**
- **Full end-to-end GPU training**: Dense · RMSNorm · CNN 1D/2D/3D
- **Pending wiring into `DispatchBackwardLayer`**: SwiGLU · MHA · Embedding

---

## The Bedrock Philosophy
M-POLY-VTD is a **"Bedrock Edition"** neural engine: bit-level dtypes, volumetric dispatch, and a clear split between **reference CPU (Go)** and **throughput backends (WebGPU → NPU → network)**.

*   **Go** owns correctness, portability, and Lucy **[7]** regression — SC/MC tiled loops across all 21 dtypes.
*   **WebGPU** owns GPU inference and training; **backward completion** for decoder layers is the current perf milestone.
*   **NPU** (Intel, Qualcomm) extends the same ENTITY / morph pipeline to fixed-function AI silicon on laptops and phones.
*   **Network + step mesh (later)** distribute inference and enable HA for long-running volumetric meshes.
*   **21-type morphing**: Same volumetric mesh can run FP32, INT8, FP4, Binary, etc., by swapping active weights in `WeightStore`.

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

### 4. Step Mesh Propagation (Neural Mesh)
Unlike the standard sequential flow, the **step mesh engine** treats the 3D grid as a cycle-accurate discrete-time mesh.

- **Neural Clock**: Every coordinate fires simultaneously in a single "pulse" or clock cycle.
- **Double Buffering**: Prevents race conditions, ensuring a stable wave of data through space-time.
- **Spatial Feedback**: Remote links can hop signals backwards in coordinates, creating dynamic recurrence (RNN-like behavior) across the 3D mesh.
- **BPTT (Backpropagation Through Time)**: Gradients are unrolled through clock cycles and spatial junctions, allowing the grid to learn complex temporal patterns.
- **Dynamic Learning Bridge**: Supports `poly.StepApplyTween` for localized, gap-based learning that updates the mesh in real-time based on temporal performance. o_O

> [!TIP]
> Use `poly.StepForward` and `poly.StepApplyTween` when you need a "living network" that evolves and learns over time rather than a static pipeline. o_O

### 4b. HF decoder CPU forward schedules (Lucy / `Transformer`)

The **3D step mesh** (`step.go`, `StepForward`) is a **volumetric-grid clock** for arbitrary `VolumetricNetwork` topologies. It is **not** what Lucy chat uses for HuggingFace decoder inference.

For **Llama-style decoder stacks** (`InitHFDecoderBlocks`: RMSNorm → MHA → RMSNorm → SwiGLU per block), `poly.Transformer` exposes a separate **CPU forward schedule** selected by `ForwardMode` (configured in **`lucy/`** after model load):

| Mode | Constant | Behavior |
| :--- | :--- | :--- |
| **1 Normal** | `TransformerForwardNormal` | Fused block loop — default, fastest on CPU. |
| **2 Stepped** | `TransformerForwardSteppedCPU` | Same math; one **sub-layer** per internal step; auto-drains each `ForwardFull` / `forwardOne`. |
| **3 Queued** | `TransformerForwardQueuedCPU` | Same as stepped; optional **Enter** pause per sub-layer (`QueueTickPause`). |
| **4 Pipeline** | `TransformerForwardPipelineCPU` | **Wavefront scheduler**: multiple prompt tokens can sit at different decoder blocks; each `PipelineTick` is one global clock that advances every **ready** job by one sub-layer. |

**Implementation files:** `transformer_forward.go` (modes 1–3), `transformer_pipeline.go` (mode 4), `transformer_layer_trace.go` (optional per-sub-layer recording during `Generate`).

#### What “pipeline / wavefront” means here

This is **classic wavefront / pipeline scheduling** (dependencies along token position and block depth; independence across the diagonal), applied to the HF decoder’s **six sub-steps per block** (pre-attn RMSNorm, MHA, attn residual, pre-MLP RMSNorm, SwiGLU, MLP residual) plus optional final RMSNorm.

- **One `PipelineTick` ≠ one sampled vocabulary token.** A tick is one sub-layer pulse for each ready `(token position, block, phase)` job. A full new decode token still needs on the order of **`numBlocks × 6 + 1`** ticks (~181 for 30 blocks) before `ApplyLMHead` and sampling.
- **Prefill** can overlap work: e.g. token 4 at block 8 while token 7 is at block 5 (turn on Lucy’s sub-layer log or interactive pipeline to see mixed `tok N block M/…` lines on the same tick).
- **Autoregressive decode** (one new embedding per step) usually has **one token in flight**, so overlap does not beat fused forward on a single CPU thread — expect **similar tok/s** to mode 1, not “one word per tick.”

**KV / position rules:** injection starts at `batchStartPos` (block-0 MHA `KVOffset` after prefill). MHA at block `b` uses **that block’s** MHA layer cursor (`Layers[b*4+1].KVOffset`), not block 0’s, so continuation after prefill does not deadlock when block 0 has already advanced.

#### Honest scope (NLP industry)

This is **not a new language-model algorithm** or paper claim. Serving stacks already use related ideas under other names (**pipeline parallelism**, **continuous batching**, **prefill/decode scheduling**, **speculative decoding**). Loom’s contribution is an **explicit, debuggable CPU scheduler** in-tree, aligned with the mesh/step mental model, and a stepping stone toward **multi-sequence / multi-token** overlap when parallel backends exist.

**When to use which mode**

| Goal | Mode |
| :--- | :--- |
| Fastest Lucy chat on CPU | **1 Normal** (GPU if enabled) |
| Step through one token, one sub-layer | **2** or **3** |
| See wavefront / debug KV order / prefill overlap | **4** + sub-layer logging or interactive pipeline ticks |
| Record every sub-layer to a trace | **Layer trace** in Lucy (`GenOptions`; uses traced CPU path, not the pipeline scheduler) |

**Tests:** `poly/tests/pipeline_forward_test.go` — prefill fused, pipeline decode, no stall (`TestPipelineDecodeAfterPrefillNoStall`).

### 5. Recursive Neural Trees (`Tensor.Nested`)
**Decision**: Implementing a recursive `Nested` field in the `Tensor` struct.
*   **Rationale**: To support nesting (`Parallel`/`Sequential`) without losing the ability to train. This creates an **Activation Tree** during the forward pass and a **Gradient Tree** during the backward pass, establishing a "Plug-and-Learn" bedrock where any complex sub-architecture is automatically differentiable.

### 6. Explicit Numerical Fast-Paths
**Decision**: Using manual `switch` statements and type-casting instead of reflection.
*   **Rationale**: In high-speed inference, reflection is too slow. We write the `INT8` and `FLOAT32` loops explicitly to ensure the compiler generates the fastest possible arithmetic for the "Reference Logic."

### 7. The "Simulation vs. Throughput" Strategy
**Decision**: Supporting types the CPU doesn't natively have (like FP4, 2-bit, 1-bit).
*   **Rationale**: We are building the **Logic Bedrock** first. On CPU, these incur a "Simulation Tax," but on GPU they become **Native Bit-Packed Payloads**, which is where the 10x performance leap occurs.

### 8. Donate compute — `donate_compute_*.go` (TCP)
**Canonical documentation** lives in [`../docs/donate_compute.md`](../docs/donate_compute.md) (wire format, modes, client/server API, security). This package only implements the protocol; inference/prompt handling is **stub** until wired to the real engine.

### 9. TANHI — `tanhi.go` (UDP telemetry)
**Canonical documentation** lives in [`../docs/tanhi.md`](../docs/tanhi.md) (JSON-line UDP protocol, SoulGlitch HUD, defaults port **17481**, Welvet `tanhi_ext.go`). Sparse, non-blocking layer events during forward/backward and GPU transformer hooks.

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

## ⚡ Performance roadmap

### v0.81 — GPU backward & decoder training

1. **SwiGLU GPU backward** — wire into `DispatchBackwardLayer`; Lucy gradient parity.
2. **MHA GPU backward** — dQ/dK/dV + projection weight grads; full decoder block GPU train.
3. **Embedding GPU backward** — untied head weight updates on device.
4. **Command graph / fusion** — fewer submits per token; prefill/decode scheduling on GPU.

### v0.82 — NPU backends

1. **Intel NPU** — ENTITY → OpenVINO / Intel NPU driver path; SmolLM-class smoke on Core Ultra.
2. **Qualcomm NPU** — ENTITY → QNN / Hexagon; Windows ARM64 + Android-class targets.
3. **Shared compile plan** — one load-time graph from `WeightStore` / `.entity`, target-select WebGPU vs NPU.

### v0.83+ — Networking & distributed mesh

1. **Donate Compute live path** — TCP offload runs real `Transformer` / ENTITY inference (not stub).
2. **Remote volumetric segments** — execute grid partitions on peers; merge activations at hop boundaries.
3. **TANHI bus** — optional fan-out of layer telemetry across nodes (`docs/tanhi.md`).

### Later — high-availability step mesh

1. **Step pulse checkpointing** — save/restore `StepState` per clock for failover.
2. **HA coordinator** — leader election, partition recovery for `StepForward` meshes.
3. **Replicated step-wise execution** — active/passive living networks for 24/7 mesh workloads.

Shaders and kernels stay in **Go + WGSL** on GPU; NPU uses vendor SDKs behind a thin `poly` backend interface. CPU **Go** remains the parity reference — see historical benchmarks below.

---

## GPU benchmarks (WebGPU track)

CNN 3D and training showdown tables below are the primary performance evidence for **GPU vs Go CPU**.


*M-POLY-VTD: Go + WebGPU + NPU (roadmap). Universal precision. Volumetric freedom.*

---

# Omni-Neural Framework: The Road to v1.0.0

To build a true "Universal AI Framework" from first principles, we must map out every theoretical and practical requirement across the entire AI industry. 

**Version 1.0.0 will only be achieved when EVERY SINGLE ITEM on this exhaustive checklist is natively supported.** 

Our semantic version number directly reflects our progress against this absolute, industry-scale roadmap. By calculating the ratio of completed features to the total required features, we derive our exact technical version.

---

## 1. Core Engine & Numerical Precision

### 1.1 Standard Floating-Point Types
- [x] FP64 (Double Precision - Scientific / Accumulation)
- [x] FP32 (Single Precision - Baseline)
- [x] FP16 (Half Precision)
- [x] BF16 (Brain Float - ML Standard)

### 1.2 Low-Precision & Bit-Level Types
- [x] FP8 E4M3 (Activations / Weights)
- [ ] INT8 WebGPU Inference Kernels (quantized matmul natively in WGSL shader)
- [x] FP4 E2M1 (Standard Bitwise Extreme Compression)

### 1.3 Integer & Fixed-Point Infrastructure
- [x] INT64, INT32, INT16, INT8
- [x] UINT64, UINT32, UINT16, UINT8
- [x] INT4 / UINT4 (Packed Weight Storage)
- [x] Bit-Packed Nibble Tensors (4-bit representation)
- [x] Quantization-Aware Scaling (Fixed-point factor logic)

### 1.4 GPU Numerical Acceleration
- [ ] FP16/BF16 GPU Training (native half-precision WGSL forward + backward kernels)
- [ ] Mixed Precision Training Loop (FP16 forward, FP32 gradient accumulation)
- [ ] On-Device Weight Dequant Shader (FP4/INT8 weight unpacking inside WGSL, no CPU roundtrip)

### 1.5 Quantization & Numerical Deep-Dive
- [x] Bitwise MAC (Multiply-Accumulate) for E2M1 CPU
- [x] Bitwise MatMul for E2M1 GPU (WebGPU)
- [x] On-the-fly Max/Min Statistics Collection (Layer Observers)
- [x] Dynamic Scale Calibration (Row-wise quantization)
- [ ] Gradient Checkpointing (recompute activations to reduce peak VRAM)
- [x] Post-Training Quantization (PTQ) weight conversion passes
- [x] True on-the-fly / load-path quantization without a mandatory FP32 master copy in host RAM (native dtype staging)
- [ ] Truncated BPTT (windowed gradient unroll for step-mesh long-sequence training)

### 1.6 GPU Backward Pass Completion
- [x] Real-valued Automatic Differentiation
- [ ] SwiGLU GPU Backward Wiring (resolve BROKEN status in benchmark table)
- [ ] MHA GPU Backward Wiring (resolve PENDING status in benchmark table)

### 1.7 Parallel Tiled Dispatch
- [x] Multi-core CPU tiling (18 layers × 21 dtypes, Go paths)
- [x] GPU register tiling (WebGPU, layer-dependent)
- [x] SIMD CPU forward — `DotTile` AVX2 (x86-64) + NEON (ARM64) (`poly/simd`, `SetSimdForwardRecursive`; see [`docs/simd.md`](../docs/simd.md))
- [x] SIMD CPU backward — `SaxpyF32AccF64` on all seven compute layers (Dense, SwiGLU, MHA, CNN1–3, RNN, LSTM; `*_simd_backward.go`)
- [ ] GPU execution plan — compile volumetric visit order to batched device dispatches

**Numerical Progress: 25 / 34**

---

## 2. Architectural Components & Layers

### 2.1 Foundational Layers
- [x] Linear / Dense / Fully Connected
- [x] Convolutional 1D
- [x] Convolutional 2D
- [x] Convolutional 3D / Volumetric
- [x] Embeddings & Lookup Tables

### 2.2 Sequence & Temporal Layers
- [x] Basic RNN (Recurrent Neural Network)
- [x] LSTM (Long Short-Term Memory)
- [x] GRU (Gated Recurrent Unit)

### 2.3 Attention & Transformer Mechanisms
- [x] Multi-Head Attention (MHA)
- [x] Grouped-Query Attention (GQA) & Multi-Query Attention (MQA)
- [x] RoPE (Rotary Position Embedding)
- [ ] Sliding Window / Sparse Attention (O(n) local attention for long contexts)
- [ ] GPU Command Graph Buffering (compile full forward pass into a single dispatch call)

### 2.4 Feed-Forward & Activations
- [x] Standard Activations (ReLU, GELU, Tanh, Sigmoid, Swish, Mish)
- [x] Softmax (10 variants: Standard, Grid, Hierarchical, Temperature, Gumbel, Masked, Sparsemax, Entmax, Adaptive, Mixture)
- [x] SwiGLU / Gated Linear Units

### 2.5 Normalization & Modern Layer Architectures
- [x] LayerNorm
- [x] RMSNorm
- [ ] Depthwise Separable Conv (1D/2D/3D) (edge-optimized mobile convolutions)
- [ ] Mamba / SSM Layer (state space model, O(n) sequence modeling alternative to transformers)

### 2.6 Advanced Topological Structures
- [x] Residual & Skip Connections
- [x] Sequential & Parallel Branching
- [x] Mixture of Experts (MoE) Routing Mechanisms
- [x] Parallel Grid Scattering (Spatial Distribution)
- [x] Layer Ensembles & Complementary Match Discovery
- [ ] LoRA Adapter Layer (low-rank fine-tuning primitive wrapping existing Dense layer)
- [x] DNA Splice / Genetic Crossover (merge two trained network DNAs into a child architecture)
- [x] NEAT-style Topology Evolution (structural NAS: add/remove nodes and edges genetically)
- [x] K-Means / Differentiable Clustering Layers

### 2.7 Introspection & Telemetry
- [x] Network Blueprint Extraction (Structure & Parameter Counts)
- [x] Recursive Layer Inspection
- [x] Memory Usage Analysis
- [x] Dynamic Grid Topology Visualization
- [x] Reflection-based Method Discovery (JSON API Export)
- [x] Observer-pattern Layer Monitoring
- [x] TANHI — UDP JSON-line layer-wise telemetry for external HUDs (e.g. SoulGlitch); see `docs/tanhi.md` and Welvet `tanhi_ext`
- [x] Allocator-level **memory footprint** reporting (`memory_footprint.go` and related rollups)

**Architectural Progress: 32 / 37**

---

## 3. Accelerators, Networking & Distributed

Part of the **Go + WebGPU + NPU + network** stack. **Device-aware** = **Go** on CPU, **WebGPU** on GPU, **NPU** on Intel/Qualcomm silicon, **TCP mesh** for remote segments.

### 3.1 GPU backward (priority)
- [x] Dense GPU backward — training E2E
- [x] RMSNorm GPU backward — training E2E
- [x] CNN 1D/2D/3D GPU backward — training E2E
- [ ] SwiGLU GPU backward wiring (`DispatchBackwardLayer`)
- [ ] MHA GPU backward wiring (dK/dV/dW + decoder block parity)
- [ ] Embedding GPU backward wiring (untied `lm_head`)

### 3.2 NPU / accelerator backends
- [x] Intel NPU path (OpenVINO / Intel NPU driver) — per-layer `poly/accel` dispatch, Lucy **[9]** (experimental, Linux)
- [x] Qualcomm NPU path (QNN / Hexagon) — per-layer `poly/accel` dispatch, Lucy **[12]** (experimental, Windows ARM64)
- [x] Apple GPU path (Metal / MPSGraph) — per-layer `poly/accel` dispatch, Lucy **[13]** (experimental, macOS Apple silicon)
- [ ] Shared compile plan: `.entity` → backend-specific graph (whole-model, not just per-layer)
- [ ] NPU parity suite vs WebGPU reference (SmolLM-class smoke)

### 3.3 Networking & offload
- [x] Donate Compute TCP framing (`donate_compute_*.go`, `docs/donate_compute.md`)
- [ ] Donate Compute → live `Transformer` / ENTITY inference
- [ ] Remote volumetric segment execution across peers
- [ ] Multi-client request routing / load spread

### 3.4 High availability — step mesh (later)
- [ ] Step pulse checkpoint / restore (`StepState` per clock)
- [ ] Failover coordinator for step mesh deployments
- [ ] Replicated step-wise execution (active/passive)
- [ ] Partition-tolerant mesh merge after node recovery

**Accelerators & Distributed Progress: 7 / 19**

---

## 4. Advanced Training Logic & Automation

### 4.1 Execution Flow
- [x] Static Computation Graphs
- [ ] Dynamic Computation Graphs (Define-by-run)
- [x] Atomic Time-Step execution (StepForward/StepBackward)
- [x] Neural Tweening / Hybrid Geometric Training
- [x] Neural Tweening Chain Rule Support
- [x] Gradient Explosion Detection & Damping
- [x] **Tiled dispatch as the primary path** — forward/backward unified on CPU/GPU tiling; legacy non-tiled paths removed or gated for a single source of truth
- [x] **`TrainingModeCPUSimd`** — MC tiling + recursive Plan 9 SIMD forward/backward (`training.go`, seven-layer SIMD train column)

### 4.2 Optimizers & Schedulers
- [x] Standard Optimizers (SGD, AdamW, RMSProp)
- [ ] Higher-order Optimizers (L-BFGS, K-FAC)
- [x] 8 Variants of Learning Rate Schedulers
- [x] Adaptive Rate Calculation (VGStepBP)
- [x] Tweening Momentum & Link-Budgeting
- [x] Adaptation Performance Tracking (Recovery Metrics)
- [x] GPU Accelerated Training Loop (FP32 end-to-end WebGPU: forward + backward + weight update in a single command buffer submission)

### 4.3 Automated Evolutionary Logic
- [ ] DARTS (gradient-based architecture search via differentiable mixed-op supernet)
- [x] Neural Architecture Search (NAS)
- [x] Random Architecture Generation & Mutation
- [ ] Speculative Decoding (draft model + verify for faster autoregressive token generation)

**Automation Progress: 15 / 19**

---

## 5. Deployment, Compilation & Ecosystem

### 5.1 Backends (Loom)
- [x] **Go** — tiled CPU loops (all layers) + Plan 9 SIMD fwd/bwd (7 compute types, AVX2/NEON) + `TrainingModeCPUSimd` — reference & Lucy [7]
- [x] **WebGPU** — GPU via WGPU (Metal / Vulkan / DX12)
- [x] **Intel NPU** — laptop AI accelerator path (experimental, OpenVINO, Lucy [9])
- [x] **Qualcomm NPU** — Snapdragon / Hexagon path (experimental, QNN, Windows ARM64, Lucy [12])
- [x] **Apple GPU** — Metal / MPSGraph path (experimental, macOS Apple silicon, Lucy [13])
- [x] **Network** — Donate Compute TCP protocol (inference wiring pending)

### 5.2 Compiler Integration
- [ ] Kernel Fusion (Translating sequential operations into single SRAM-bound kernels to eliminate memory bottleneck)
- [ ] Triton eDSL / WGSL AST transpilation
- [ ] MLIR (Multi-Level Intermediate Representation) Lowering passes

### 5.3 Polyglot Ecosystem & I/O
- [x] Universal C-ABI Core API
- [x] Python Bindings (`welvet`) — Published to PyPI
- [x] Node.js / TypeScript Bindings (@openfluke/welvet)
- [x] C# / .NET Bindings
- [x] Java Bindings
- [x] Dart Bindings
- [x] WebAssembly (WASM) browser execution
- [x] Universal SafeTensors Support (Load / Save / V2 Multi-type)
- [x] HuggingFace Checkpoint Interoperability (Weight Extraction)
- [x] **Donate Compute** — TCP LAN volunteer / offload framing (`donate_compute_*.go`, `docs/donate_compute.md`)
- [x] **Lucy** — HuggingFace model download, compile-on-the-go workflow, conversational smoke (`lucy/`)
- [x] **Qwen3-family** checkpoints in the HF ingestion / LM pipeline

### 5.4 Benchmarks & Validation
- [x] ARC-AGI Task Benchmark (K-Means Implementation)
- [x] Numerical Deviation Metrics (Accuracy Heatmaps)
- [x] Task-Switching Adaptation Benchmarks
- [x] Model Ensemble Diversity Metrics
- [x] Training Method Comparison Analysis
- **Lucy / log interpretation** — [`docs/testing_and_validation.md`](../docs/testing_and_validation.md) (parity legend, `lucy/lucy_testing_output/log.txt`, poly file map for suites)
- [x] **Lucy seven-layer CPU suite** — 10 layer types × 21 dtypes × 1³/2³/3³ grids, **SC/MC/SIMD** fwd/bwd, train (incl. SIMD), native save/reload (`lucy/examples/seven_layer/`, `seven_layer.txt`)
- [x] **ENTITY native checkpoints** — `.entity` binary format; Lucy [7] JSON+entity parity; Lucy [8] HF→entity→GPU chat ([`docs/entity.md`](../docs/entity.md))
- [x] **WebGPU v29 module** — `github.com/openfluke/webgpu@v1.0.4`; cross-platform GPU validated (Metal, Win ARM64 Vulkan, Linux Intel + NVIDIA)
- [x] **Planet Bridging POC** — planets→Loom complete for 12 layer types ([`planetbridging/`](../planetbridging/)); **separate release after Loom 0.80**
- [x] **C-ABI functional parity** — `welvet/cabi/internal/check` at **461/461** (100%); includes `LoomSyncInferenceWeights`
- [x] **MHA volumetric layout + KV** — `[B,S,D]` parsing, training vs decode cache policy, backward Q/K norm parity (`mha_layout.go`, `mha.go`)

**Ecosystem Progress: 28 / 28**

---

## 6. LLM Engine & Tokenization

### 6.1 Tokenization Core
- [x] BPE (Byte-Pair Encoding) Implementation
- [x] HuggingFace `tokenizer.json` Compatibility
- [x] ChatML & Prompt Template Engine
- [x] Recursive Multi-turn Turn Tracking

### 6.2 Generation Logic
- [x] KV Cache Optimization (Stateful incremental inference)
- [x] CPU forward schedules for HF decoder (normal / stepped / queued / **wavefront pipeline**) — see **§4b**
- [x] Batched Prefill & Autoregressive Decoding
- [x] Sampling Suite (Top-K, Temperature, Nucleus Placeholder)
- [x] Repetition Penalty & Windowed Logit Bias
- [x] Deterministic vs Stochastic Inference Modes
- [x] Real-time Token Streaming (Streamer primitives)

### 6.3 LLM Tooling & Profiling
- [x] HuggingFace Hub Cache Auto-Discovery
- [x] FP4 Quantized Specialist Chat Implementation
- [x] WebGPU LM-Head Offloading
- [x] VRAM Usage Profiling & Distribution Metrics

**LLM Progress: 15 / 15**

---

## v0.83.0 — *Apple Bridge* (current)

- **Apple GPU / Metal (experimental)** — third accelerator vendor: `libloom_accel_apple.dylib` built with CMake against Apple's **Metal Performance Shaders Graph** (no SDK to vendor — Metal ships with macOS). Two devices behind the vendor-neutral `loom_accel.h` C ABI: a portable **CPU reference** (deterministic parity anchor) and a **Metal GPU** (MPSGraph for MatMul/MHA/ReLU/Sigmoid/Softmax/Add/Multiply, per-op CPU fallback for the rest). Darwin CGO loader (`plugin_darwin.go`), `ExecAppleCPU`/`ExecAppleGPU` targets, per-layer `DispatchLayer` offload, and Lucy **[13]** `apple` bench, mirroring the Intel **[9]** and Qualcomm **[12]** suites.
- **BF16 wire dtype** — the shared accel bridge (`poly/accel_intel.go`) now packs/unpacks **bfloat16** (top 16 bits of FP32, round-to-nearest-even), the native low-precision type on Apple silicon. Apple advertises **FP32/FP16/BF16/INT16/INT8/INT4** via its `bench_manifest.json`.
- **Determinism** — 180/180 💎 EXACT repeat-forward on both Apple CPU and Metal GPU; GPU parity 132/180 ≤ INDUS; up to **5.4×** faster than Loom CPU on large MatMul/MHA (GPU) and up to **94×** on elementwise (CPU reference).
- **Honest scope** — forward-only per-layer offload; Conv/GELU and the norms are CPU-reference-only (no MPSGraph path yet), norm parity broken (no weight bake), INT8 MatMul drift breaks on large tiers, ANE not wired (Metal only). Good enough for a release, not for prod — see [`docs/apple_metal.md`](../docs/apple_metal.md).
- **SIMD backward (current tree)** — extends v0.82 forward-only `DotTile` with **`SaxpyF32AccF64` backward** on all seven compute layers; `TrainingModeCPUSimd`; seven-layer **SC/MC/SIMD** parity (0 diff Float32 1×1×1 on amd64 + arm64). Best fwd: Dense **3.6×** (amd64) / **3.0×** (arm64) vs SC; best bwd: CNN3 **2.8×** (amd64) / CNN2 **1.7×** (arm64). See [`docs/simd.md`](../docs/simd.md).
- **Docs** — [`docs/apple_metal.md`](../docs/apple_metal.md), [`docs/accelerators.md`](../docs/accelerators.md), [`docs/v083_release.md`](../docs/v083_release.md), [`docs/simd.md`](../docs/simd.md).

## v0.82.0 — *Snapdragon Bridge* (previous)

- **SIMD CPU fast-path** — `poly/simd`: AVX2 `DotTile` + NEON on **x86-64** / **ARM64**; v0.82 shipped **forward only** — current tree adds **backward `SaxpyF32AccF64`** on seven layer types + `TrainingModeCPUSimd`. See [`docs/simd.md`](../docs/simd.md).
- **Qualcomm / Hexagon NPU (experimental)** — second vendor plugin: `loom_accel_qualcomm.dll` built with `llvm-mingw clang++` against the **QNN AI Engine Direct** SDK (QAIRT). Windows-ARM64 CGO loader (`plugin_qualcomm_windows.go`), `ExecQualcommCPU`/`ExecQualcommNPU` targets, per-layer `DispatchLayer` offload, and Lucy **[12]** `snapdragon` bench (timing + drift spectrum across FP32/FP16/INT16/INT8/INT4), mirroring the Intel NPU **[9]** suite.
- **Robustness** — unique per-build QNN graph names (`loom_<op>_<dtype>_<seq>`) to avoid `QnnGraph_create` collisions; QNN context reset on `CompiledLayer` release (fixes graph leaks / `0xc0000005` on long runs); QNN log level clamped to ERROR by default (`LOOM_QNN_VERBOSE=1` to restore) plus a terminal noise filter.
- **Install / build** — `accel/qualcomm/install_qairt.ps1 -Persist` (machine-wide `QNN_SDK_ROOT` / `LOOM_QUALCOMM_RUNTIME`), `accel/qualcomm/build_clang.ps1`; `webgpu` `go.mod` `replace` workaround for the Windows ARM64 MSVC/GNU ABI mismatch documented in `accel/qualcomm/README.md`.
- **Honest scope** — forward-only per-layer offload; not whole-model `.entity` lowering, no NPU training/backward. Good enough for a release, not for prod — see [`docs/snapdragon_npu.md`](../docs/snapdragon_npu.md).
- **Docs** — [`docs/snapdragon_npu.md`](../docs/snapdragon_npu.md), [`docs/simd.md`](../docs/simd.md), [`docs/accelerators.md`](../docs/accelerators.md), [`docs/v082_release.md`](../docs/v082_release.md).

## v0.81.0 — *Accelerator Bridge* (previous)

- **`poly/accel/`** — runtime `dlopen` of vendor plugins; stable C ABI (`loom_accel.h` in chaosglue).
- **Intel OpenVINO** — `libloom_accel_intel.so`: CPU + NPU forward via `DispatchLayer`; `SyncToAccel` (compile once, infer many); weight upload for MatMul / Conv / MHA-MatMul.
- **Lucy [9]** — `nine_layer` suite: timing + seven-style drift spectrum; menu **[4]/[5]** DispatchLayer matrix (90 cells on full run).
- **Experimental** — Linux + `CGO_ENABLED=1` + OpenVINO/NPU driver on `LD_LIBRARY_PATH`. Forward only; training/backward still Loom CPU.
- **Planned vendors** — Qualcomm NPU (`libloom_accel_qcom.so`), Google TPU (`libloom_accel_google.so`) — same ABI, separate chaosglue trees.
- **Docs** — [`docs/v081_release.md`](../docs/v081_release.md), [`docs/accelerators.md`](../docs/accelerators.md). Evidence: [chaosglue integration assessment](https://github.com/openfluke/chaosglue/blob/main/npu/docs/2025-06-26-loom-dispatch-integration-assessment.md).

## v0.80.0 — *Native Ship* (previous)

- **ENTITY** — native `.entity` checkpoints (`poly/entity.go`); topology + native-packed weights; ~25% smaller than JSON; idempotent round-trip tests.
- **Lucy [8] ENTITY Talk** — HF → `ImportHFToEntity` → optional Q4 bake → GPU chat without runtime safetensors.
- **WebGPU v29** — `github.com/openfluke/webgpu@v1.0.4`; standalone openfluke module; futures + `WGPUStringView` Go bindings (no C shims).
- **GPU validated** — Metal (Apple M5), Windows ARM64 Vulkan (Snapdragon), Linux Intel Iris Xe + NVIDIA RTX 3050 Mobile (Lucy Poly Talk / ENTITY Talk, SmolLM2-135M Q4).
- **Planet Bridging POC** — v0.5.0 in [`planetbridging/`](../planetbridging/): 13 bedrocks, PyTorch/TF/JAX live stream → Loom; **releases after Loom 0.80**.
- **Docs** — [`docs/v080_release.md`](../docs/v080_release.md), [`docs/entity.md`](../docs/entity.md).

## v0.79.0 — *Bedrock Validation* (previous)

- **Lucy [7] seven-layer suite** — volumetric JSON grids, **10 layer families** × **21 dtypes**, CPU **SC/MC/SIMD**, **train**, **save/reload** before and after training (`lucy/examples/seven_layer/`).
- **MHA** — `mhaParseLayout` for `[B,S,D]`; `mhaPrepareKVForForward` (train reset vs decode cache); backward matches forward Q/K RMS norm; Poly Talk KV offset fixed.
- **Native checkpoints** — BitNet/ternary save uses native `Versions` packing (`bitnet_cpu.go`); signed low-bit persistence on `[]uint8`.
- **C-ABI** — **461/461** export parity; **`LoomSyncInferenceWeights`** for `ReleaseFP32MasterWhenIdle` inference RAM.
- **Docs** — [`docs/bedrock_validation.md`](../docs/bedrock_validation.md).

## v0.78.0 — *ASM CPU experiment* (previous, archived)

- Early Plan 9 dense-forward POC — **not** the long-term CPU strategy. Lessons archived in [`docs/asm-and-volumetric-exploration.md`](../docs/asm-and-volumetric-exploration.md). Performance investment moved to **WebGPU backward**, **NPU**, and **networked mesh**.

## v0.76.0 — *Operation Mesh* (previous)

Step mesh stability, TANHI, Donate Compute, Lucy HF pipeline, on-the-fly quantization, Qwen3 ingest, tiled-first dispatch, memory footprint telemetry — see checklist rows marked in earlier releases.

---

## 📊 True Version Calculation

Instead of arbitrarily bumping version numbers, we derive our exact semantic version by measuring the framework's strictly verified capabilities against the absolute "Universal Version 1.0.0" checklist.

| Category | Completed | Total |
| :--- | :---: | :---: |
| 1. Numerical Core | 25 | 34 |
| 2. Architectural Layers | 32 | 37 |
| 3. Accelerators & Distributed | 7 | 19 |
| 4. Training Automation | 15 | 19 |
| 5. Deployment Ecosystem | 28 | 28 |
| 6. LLM & Tokenization | 15 | 15 |
| **GRAND TOTAL** | **122** | **152** |

### **Completion Ratio: 80.3%**

## **Version 0.83.0 — CURRENT**
*(**v0.83.0 "Apple Bridge"** — from **0.82.0**. Third experimental accelerator vendor: **Apple GPU** (Metal / MPSGraph, macOS Apple silicon, Lucy **[13]**) alongside **Intel CPU+NPU** (Lucy **[9]**) and **Qualcomm/Hexagon** (Lucy **[12]**). Adds a **BF16** wire dtype to the shared accel bridge. **Current tree** also completes **Plan 9 SIMD backward** on seven compute layers + `TrainingModeCPUSimd` (seven-layer SC/MC/SIMD parity). Checklist **122 / 152** (80.3%). **Next:** whole-model `.entity` → NPU lowering, NPU parity suite, GPU backward (SwiGLU/MHA), AccelPlanner + JSON `exec`.)*

## **Version 0.82.0** (previous)
*(**v0.82.0 "Snapdragon Bridge"** — **SIMD** CPU fast-path (AVX2/NEON); second experimental NPU vendor: **Qualcomm/Hexagon** (QNN, Windows ARM64, Lucy **[12]**) alongside **Intel CPU+NPU** (Lucy **[9]**). Checklist **117 / 147** (79.6%).)*

## **Version 0.81.0** (previous)
*(**v0.81.0 "Accelerator Bridge"** — Experimental **Intel CPU+NPU** via `poly/accel`; Lucy **[9]**. Checklist **112 / 146** (76.7%).)*

## **Version 0.80.0** (previous)
*(**v0.80.0 "Native Ship"** — ENTITY, openfluke/webgpu v1.0.4, multi-platform GPU.)*

## **v0.79.0 — Bedrock Validation** (previous)
*(Seven-layer CPU regression, MHA/KV, C-ABI 100%.)*

## **v0.81–v0.82 Roadmap**
*(**AccelPlanner** + JSON `exec`; Intel parity for MatMul/norms; **Qualcomm** + **Google TPU** CABI stubs; complete **GPU backward** for SwiGLU/MHA/Embedding; **Donate Compute** live inference; HA step mesh — later.)*


