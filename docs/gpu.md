# GPU Backend: WebGPU (WGPU)

This document covers the WebGPU backend: initialization, the `BeginFrame`/`FlushFrame` command batching pattern, the buffer pool and pipeline cache, which layers have GPU support, and the tiling strategy.

---

## Why WebGPU

M-POLY-VTD uses the `github.com/openfluke/webgpu/wgpu` Go bindings for hardware acceleration. WebGPU compiles to:

- **Vulkan** on Windows/Linux
- **Metal** on macOS/iOS
- **DX12** on Windows
- **WebGPU** in browser via WASM

No CUDA, no CGO beyond the wgpu bindings. All shaders are WGSL (WebGPU Shading Language) strings generated at runtime by Go functions in `wgpu_shaders.go`, `wgpu_kernels.go`, and `wgpu_backward_shaders.go`.

---

## WGPUContext

```go
type WGPUContext struct {
    Instance       *wgpu.Instance
    Adapter        *wgpu.Adapter
    Device         *wgpu.Device
    Queue          *wgpu.Queue

    PipelineCache  map[string]*wgpu.ComputePipeline   // keyed by shader source hash
    ActivationPool map[string]*wgpu.Buffer             // named activation buffers
    LayoutCache    map[string]*wgpu.BindGroupLayout
    BindGroupCache map[uint64]*wgpu.BindGroup          // keyed by buffer-set hash

    UniformPool    []*wgpu.Buffer   // pre-allocated uniform buffer pool
    UniformIdx     int

    ActiveEncoder  *wgpu.CommandEncoder   // non-nil during BeginFrame/FlushFrame
    PendingDestroys []*wgpu.Buffer        // temp bufs destroyed after FlushFrame

    GPUTileSize    int    // auto-detected optimal tile size
    Limits         wgpu.Limits
}
```

### Initialization

```go
err := network.InitWGPU()
```

`InitWGPU` performs three WebGPU steps:

1. Create an `Instance` and request a `HighPerformance` `Adapter`
2. Query the default device for its limits, then boost `MaxStorageBufferBindingSize` to 1 GB and `MaxBufferSize` to 2 GB for large embedding tables
3. Request the final `Device` with boosted limits, then auto-detect the optimal `GPUTileSize` from the workgroup storage and invocation limits

```
CalculateOptimalGPUTileSizeFromLimits(
    MaxComputeWorkgroupStorageSize,
    MaxComputeInvocationsPerWorkgroup,
    headDim=64,
) → GPUTileSize (e.g., 8 or 16)
```

After init, call `network.SyncAllToGPU()` to upload all layer weights to VRAM. This also creates GPU KV cache buffers for MHA layers and pre-allocates named activation buffers (`hidden_A`, `hidden_B`, `norm_out`, etc.).

---

## BeginFrame / FlushFrame Pattern

The most important design decision in the GPU backend. Instead of submitting a command buffer per layer (which would mean 100+ GPU driver calls per token), all operations are recorded into a single shared encoder:

```
ctx.BeginFrame()
    ← creates ctx.ActiveEncoder
    ← resets ctx.PendingDestroys

    // All Dispatch* calls record into ActiveEncoder:
    ctx.DispatchForwardLayer(...)
    ctx.DispatchActivation(...)
    ctx.DispatchMSEGradPartialLoss(...)
    ctx.DispatchBackwardLayer(...)
    ctx.DispatchApplyGradients(...)

ctx.FlushFrame()
    ← enc.Finish() + Queue.Submit(cmd)
    ← destroys PendingDestroys buffers
    ← resets UniformIdx
```

Temporary uniform buffers (holding layer parameters like `batchSize`, `inputSize`, etc.) must stay alive until `FlushFrame` because the GPU reads them asynchronously. They are collected in `PendingDestroys` and destroyed only after the submit.

`Queue.WriteBuffer` calls (to upload inputs, targets, and zero DW buffers) are **queue-level operations** — they are safe to call between `BeginFrame` and `FlushFrame` because the WebGPU spec guarantees they complete before the encoder submit executes.

---

## Buffer Management

### ActivationPool

Named persistent buffers that survive across frames:

```go
buf := ctx.GetActivationBuffer("hidden_A", size, wgpu.BufferUsageStorage)
```

If a buffer with this name already exists and is large enough, it is reused. Otherwise a new one is created and cached. This avoids per-step allocations during inference.

### CreatePersistentBuffer

```go
buf, err := ctx.CreatePersistentBuffer(data []float32, label string)
```

Uploads a `[]float32` to a VRAM storage buffer with `Storage | CopySrc | CopyDst` usage. Used for weight buffers that stay resident across many forward passes.

### ReadBuffer

```go
values, err := ctx.ReadBuffer(buf *wgpu.Buffer)
```

Copies a GPU buffer to a CPU staging buffer, maps it, and returns `[]float32`. This is the only synchronous GPU→CPU roundtrip in the training path; it is called once per batch to read back the partial loss sums.

### BindGroup Cache

`GetBindGroup(pipeline, buffers...)` hashes the pipeline pointer and buffer pointers into a `uint64` key. If a matching `BindGroup` already exists, it is returned without re-creating it. This avoids rebuilding the descriptor set on every frame for stable weight+activation buffer pairs.

---

## Weight Sync Strategies

`SyncToGPU()` on a `VolumetricLayer` uses different strategies depending on layer type and DType:

```
RMSNorm:
    Always uploads FP32 master. Quantization destroys normalization precision.

SwiGLU (FP32):
    Splits Master into Gate, Up, Down slices.
    Uploads three separate persistent buffers.

SwiGLU (INT4 / Q4_0):
    Calls syncQuantizedSwiGLU which quantizes each slice independently.
    Each component gets a scales buffer + packed uint32 buffer.

Dense (INT4 / Q4_0):
    syncQuantizedDense: 32-weight blocks, scale per block, packed nibbles.

MHA (FP32):
    Splits into Q/K/V/O weight buffers at internal DType codes 200/201/202/203.
    Also uploads optional q_norm/k_norm buffers at 204/205 when present.

MHA (INT4):
    syncQuantizedMHA: quantizes each of Q/K/V/O separately.
```

The internal DType codes (100–102 for SwiGLU components, 200–203 for MHA projections) are a namespacing trick to store multiple named GPU buffers in the single `GPUWeights map[DType]any` without adding new struct fields.

---

## Forward Dispatch (wgpu_forward.go)

`ctx.DispatchForwardLayer(l, batchSize, inBuf, outBuf)` routes to the correct WGSL shader. Key functions:

| Function | WGSL kernel | Notes |
|:---------|:------------|:------|
| `DispatchDenseForward` | matmul shader | register-tiled |
| `DispatchRMSNorm` | RMSNorm shader | always FP32 weights |
| `DispatchCNN1Forward` | 1D conv shader | |
| `DispatchCNN2Forward` | 2D conv shader | 1826x vs CPU |
| `DispatchCNN3Forward` | 3D conv shader | 7602x vs CPU |
| `DispatchRNNForward` | RNN cell shader | |
| `DispatchLSTMForward` | LSTM cell shader | |
| `DispatchEmbedding` | gather shader | |
| `DispatchMHAForward` | Q/K/V + attention | separate kernels |
| `DispatchSwiGLUForward` | gate+up+down | BROKEN determinism |

`DispatchActivation(n, act, inBuf, outBuf)` dispatches a shader that applies ReLU, SiLU, GELU, Tanh, or Sigmoid elementwise over `n` elements.

---

## Backward Dispatch (wgpu_backward_shaders.go)

WGSL shaders for gradient computation:

**Dense DX shader** (`ShaderDenseBackwardDX`):
```wgsl
dx[b, i] = Σ_o  dy[b, o] × W[o, i]

// Implemented as tiled matmul using shared memory tiles:
var<workgroup> dyTile: array<f32, tileSize*tileSize>;
var<workgroup> wTile:  array<f32, tileSize*tileSize>;
```

**Dense DW shader** (`ShaderDenseBackwardDW`):
```wgsl
dW[o, i] = Σ_b  dy[b, o] × x[b, i]
// Uses atomic add for race-free accumulation across batch
```

**CNN DX/DW shaders**: Implement the "strided convolution" backward pass — the input gradient is the transposed convolution of the output gradient with the kernel, and the weight gradient is the correlation of the input with the output gradient.

**Activation backward**: `DispatchActivationBackward` applies the activation derivative elementwise: `gradPre[i] = gradOut[i] × act'(preAct[i])`.

**MSE gradient + partial loss** (`DispatchMSEGradPartialLoss`):
```wgsl
grad[i] = (2.0 / N) × (pred[i] - target[i])
partial[wg] = Σ_{i in group}  (pred[i] - target[i])²
```

**Apply gradients** (`DispatchApplyGradients`):
```wgsl
weights[i] -= lr × dw[i]
```

---

## GPU support: layer × `DType` (one table)

Scope: **`VolumetricLayer.SyncToGPU`** + **`(*WGPUContext).DispatchForwardLayer`** in `poly.go` / `wgpu_kernels.go`. Symbol **`T`** means **`Transformer.ForwardTokenIDsWGPU`** / **`wgpu_forward.go`** (LLM inference) for that layer+dtype, not generic batch dispatch. Activations are **`f32`** WGSL; **`DTypeFloat64`** is coerced to the **`Float32`** weight-buffer path in the `hasSpecialPath` / morph block (see `SyncToGPU`).

| Symbol | Meaning |
|:------:|---------|
| **Y** | **Generic GPU forward OK**: `SyncToGPU` does not skip the `MorphToFloat32ForGPU` upload **or** uses a matching native path (`DispatchDenseQ4` for **Dense+Int4** only; **CNN1** packed when `isCNN1NativeGPUQuantDType`). |
| **T** | **Transformer path only** (`wgpu_forward.go`): QKV/O use **`DispatchDenseQ4`** / **`DispatchDenseI8`**; SwiGLU gate/up may use **`DispatchSwiGLUQ4`**. **Not** correct for generic **`DispatchForwardLayer`** on that dtype (quantized buffers + **`DispatchDense`** / **`DispatchSwiGLUWithActCache`** mismatch). |
| **–** | **Not supported** after vanilla `SyncToGPU` + generic `DispatchForwardLayer` (skipped morph with no valid weight buffer, or packed weights fed to an **`f32`** matmul / SwiGLU shader). |
| **·** | **DType N/A** (no weight tensor for that layer). |

**Dense:** only **`DTypeInt4`** selects **`DispatchDenseQ4`**. Wider dtypes (**2–13, 15–20** except **14**) hit **`hasSpecialPath`** with no quant branch → morph skipped → **–**. Eight-bit dtypes on Dense get **`syncQuantizedDenseI8`** but **`DispatchDenseTiled`** expects **`f32`** layout → **–**. **`ensureGPUFloat32Weights`** (training) can still attach **`GPUWeights[Float32]`** so matmul runs on the **FP32 master** regardless of `l.DType` (not reflected as **Y** here).

| ID | `DType` | Dense | RMSNorm | CNN1 | CNN2 | CNN3 | RNN | LSTM | Embedding | Softmax | MHA | SwiGLU | Residual |
|---:|---------|:-----:|:-------:|:----:|:----:|:----:|:---:|:----:|:---------:|:-------:|:---:|:------:|:--------:|
| 0 | Float64 | Y | Y | Y | Y | Y | Y | Y | Y | · | Y | Y | · |
| 1 | Float32 | Y | Y | Y | Y | Y | Y | Y | Y | · | Y | Y | · |
| 2 | Float16 | – | Y | Y | Y | Y | Y | Y | Y | · | Y | Y | · |
| 3 | BFloat16 | – | Y | Y | Y | Y | Y | Y | Y | · | Y | Y | · |
| 4 | FP8 E4M3 | – | Y | Y | Y | Y | Y | Y | Y | · | T | T | · |
| 5 | FP8 E5M2 | – | Y | Y | Y | Y | Y | Y | Y | · | T | T | · |
| 6 | Int64 | – | Y | Y | Y | Y | Y | Y | Y | · | Y | Y | · |
| 7 | Int32 | – | Y | Y | Y | Y | Y | Y | Y | · | Y | Y | · |
| 8 | Int16 | – | Y | Y | Y | Y | Y | Y | Y | · | Y | Y | · |
| 9 | Int8 | – | Y | Y | Y | Y | Y | Y | Y | · | T | T | · |
| 10 | Uint64 | – | Y | Y | Y | Y | Y | Y | Y | · | Y | Y | · |
| 11 | Uint32 | – | Y | Y | Y | Y | Y | Y | Y | · | Y | Y | · |
| 12 | Uint16 | – | Y | Y | Y | Y | Y | Y | Y | · | Y | Y | · |
| 13 | Uint8 | – | Y | Y | Y | Y | Y | Y | Y | · | T | T | · |
| 14 | Int4 | Y | Y | Y | Y | Y | Y | Y | Y | · | T | T | · |
| 15 | Uint4 | – | Y | Y | Y | Y | Y | Y | Y | · | T | T | · |
| 16 | FP4 | – | Y | Y | Y | Y | Y | Y | Y | · | T | T | · |
| 17 | Int2 | – | Y | Y | Y | Y | Y | Y | Y | · | T | T | · |
| 18 | Uint2 | – | Y | Y | Y | Y | Y | Y | Y | · | T | T | · |
| 19 | Ternary | – | Y | Y | Y | Y | Y | Y | Y | · | T | T | · |
| 20 | Binary | – | Y | Y | Y | Y | Y | Y | Y | · | T | T | · |

**CNN1 column:** **Y** = either **`DispatchCNN1Packed`** (dtype in `isCNN1NativeGPUQuantDType`: Int8, Int4, Int2, FP4, Ternary, Binary, FP8×2, Uint8, Uint4, Uint2, Float16, BFloat16, Int16) or **`DispatchCNN1`** on **`MorphToFloat32ForGPU`** otherwise.

**Not in this table:** `LayerLayerNorm`, `LayerConvTransposed*`, `LayerKMeans`, `LayerParallel`, `LayerSequential`, `LayerMetacognition` (no `DispatchForwardLayer` arm). See [numerical_types.md](numerical_types.md) for the **`DType`** enum and **`WeightStore`**.

**GPU training:** `gpuTrainingNeedsCPUFallback` in `training.go` forces a **CPU** optimizer step when the net includes **MHA**, **SwiGLU**, **Dense+Int4**, or **RNN/LSTM** with **Int8/Int4**.

---

The project uses **Numerical Tiling** to map 3D volumetric layers to GPU workgroups.

### SC (single-workgroup) vs MC (multi-workgroup) profiles

Loom differentiates two dispatch profiles for GPU kernels (attention, dense, SwiGLU, CNN, etc.):

- **SC**: Smaller workgroups / tiles — lower register pressure, friendlier to tight limits (edge GPUs, WASM).
- **MC**: Larger tiles where limits allow — higher throughput on desktop-class GPUs.

At **inference**, transformer-style forwards (`wgpu_forward.go`) choose per-layer tile sizes with `layer.GetGPUSCTileSize(dtype)` vs `layer.GetGPUMCTileSize(dtype)` according to **`VolumetricNetwork.EnableMultiCoreTiling`** (with the same field mirrored on layers when set). That is the primary switch — not `GPUTileSize` alone.

`WGPUContext.GPUTileSize` is still the device-tuned baseline derived from `CalculateOptimalGPUTileSizeFromLimits` and feeds into how SC/MC maps are built in `refreshRuntimeGPUTileSizes`. **GPU training** may ignore the network flag and pick SC vs MC directly via `TrainingModeGPUSC` / `TrainingModeGPUMC` (`training.go`).

**CPU:** poly does **not** expose SC vs MC as two tile maps on the CPU side — layers use **`CPUTileSizes` / `GetCPUTileSize` only**. See the **“GPU: two tile maps…”** and **“CPU: one tile map…”** subsections in [dispatch.md](dispatch.md).

---

## Transformer GPU Forward (wgpu_forward.go)

`Transformer.ForwardTokenIDsWGPU` is the optimized path for LLM inference:

1. If `tokens != nil` and GPU embeddings are loaded, dispatch a gather shader to convert token IDs → hidden states entirely on-GPU
2. `BeginFrame()` — all subsequent ops recorded into one encoder
3. For each transformer block (4 layers: RMSNorm → MHA → RMSNorm → SwiGLU):
   - Dispatch `DispatchRMSNorm`
   - Dispatch Q/K/V projections separately (supports expanded QueryDim)
   - Optional Q/K RMSNorm using q_norm/k_norm buffers
   - Dispatch RoPE rotation
   - Dispatch attention score + softmax
   - Dispatch output projection
   - Add residual
4. Final norm + LM head if on GPU
5. `FlushFrame()` — single submit
6. Read back only the logits (one small buffer)

This path achieves the "260+ tokens/s prefill on M4" figure mentioned in the README.

### Qwen / Expanded-Query Notes

Loom's GPU path now supports architectures where `query_dim != d_model` (for example Qwen3-0.6B with `head_dim=128`, `num_heads=16`, `query_dim=2048`, `d_model=1024`).

Key implementation details:
- MHA shader workgroup width scales with `head_dim` (not hardcoded to 64).
- Q projection and attention output buffers use `query_dim`.
- O projection uses `input=query_dim`, `output=d_model`.
- RMSNorm epsilon is propagated from checkpoint config (`rms_norm_eps`) for parity with CPU.
