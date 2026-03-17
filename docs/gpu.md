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

## GPU Support Matrix

From the README benchmark table:

```
┌─────────────────┬────────────────┬────────────────┬──────────────┐
│ Layer           │ Forward (GPU)  │ Backward (GPU) │ Determinism  │
├─────────────────┼────────────────┼────────────────┼──────────────┤
│ Dense           │ REAL           │ EXACT          │ SLIGHTLY OFF │
│ RNN Cell        │ REAL           │ —              │ EXACT        │
│ LSTM Cell       │ REAL           │ —              │ EXACT        │
│ CNN 1D          │ REAL           │ EXACT          │ EXACT        │
│ CNN 2D          │ REAL           │ EXACT          │ EXACT        │
│ CNN 3D          │ REAL           │ EXACT          │ EXACT        │
│ Embedding       │ REAL           │ EXACT (DW)     │ EXACT        │
│ RMSNorm         │ REAL           │ EXACT          │ INDUSTRY ✅  │
│ MHA (Attn)      │ REAL           │ pending        │ BROKEN ❌    │
│ SwiGLU (MLP)    │ REAL           │ not wired      │ BROKEN ❌    │
│ Residual Add    │ REAL           │ —              │ BROKEN ❌    │
└─────────────────┴────────────────┴────────────────┴──────────────┘
```

"BROKEN" means the GPU forward result diverges from the CPU reference — these are known bugs. Full end-to-end GPU training is verified for Dense, CNN 1D/2D/3D, and RMSNorm.

---

## The Tiling Strategy

Each layer's GPU shader uses **register-level tiling**: a portion of the weight matrix is loaded into workgroup shared memory, threads compute a partial dot product, then the next tile is loaded. This keeps data in ultra-fast SRAM and avoids redundant global memory reads.

```
Dense 8×8 Tile:

  Workgroup: 8 threads × 8 threads = 64 invocations

  For tile t:
  ┌─────────────────────────────────────┐
  │  Load dy[batch, o_tile] into SRAM  │
  │  Load W[o_tile, input] into SRAM   │
  │  workgroupBarrier()                 │
  │  Compute partial sums               │
  │  workgroupBarrier()                 │
  └─────────────────────────────────────┘
  Accumulate across all tiles → dx[b, i]
```

The tile size is auto-detected from `MaxComputeWorkgroupStorageSize` and `MaxComputeInvocationsPerWorkgroup` at device init time, then stored in `WGPUContext.GPUTileSize`.

---

## Transformer GPU Forward (wgpu_forward.go)

`Transformer.ForwardTokenIDsWGPU` is the optimized path for LLM inference:

1. If `tokens != nil` and GPU embeddings are loaded, dispatch a gather shader to convert token IDs → hidden states entirely on-GPU
2. `BeginFrame()` — all subsequent ops recorded into one encoder
3. For each transformer block (4 layers: RMSNorm → MHA → RMSNorm → SwiGLU):
   - Dispatch `DispatchRMSNorm`
   - Dispatch Q/K/V projections separately
   - Dispatch RoPE rotation
   - Dispatch attention score + softmax
   - Dispatch output projection
   - Add residual
4. Final norm + LM head if on GPU
5. `FlushFrame()` — single submit
6. Read back only the logits (one small buffer)

This path achieves the "260+ tokens/s prefill on M4" figure mentioned in the README.
