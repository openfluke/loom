# Training: Forward Pass, Backward Pass, Optimizers, and Learning

This document covers the full training pipeline: the forward and backward pass mechanics, loss computation, weight update strategies, gradient clipping, Tween, and the `VGStepBP` adaptive rate.

---

## The Training Loop

```go
result, err := poly.Train[float32](network, batches, config)
```

`Train[T Numeric]` is the high-level entry point. It wraps `trainBatchCPU` or `trainBatchWGPU` depending on `config.UseGPU`.

```go
type TrainingConfig struct {
    Epochs       int
    LearningRate float32
    LossType     string   // "mse" or "cross_entropy"
    GradientClip float32  // 0 = no clipping
    Verbose      bool
    UseGPU       bool
    DeviceID     int
    TrackPerf    bool
}
```

A `TrainingBatch[T]` pairs `Input *Tensor[T]` with `Target *Tensor[T]`. Multiple batches are provided as a slice — the loop iterates over batches for each epoch, averages the loss, and prints progress if `Verbose = true`.

---

## Runtime tiling (`ConfigureNetworkForMode`)

Before the training loop runs, `Train` wires the network through `ConfigureNetworkForMode` (`training.go`), which aligns tiling flags with the selected `TrainingMode`:

| Mode | Tiling | Multi-core | SIMD |
|------|--------|------------|------|
| `TrainingModeCPUNormal` | off | off | off |
| `TrainingModeCPUSC` | on | off | off |
| `TrainingModeCPUMC` | on | on | off |
| `TrainingModeCPUSimd` | on | on | **on** (`SetSimdForwardRecursive`) |

- **CPU modes** (`TrainingModeCPUNormal` … `TrainingModeCPUSimd`): `ConfigureNetworkForMode` sets `UseTiling` and `EnableMultiCoreTiling` per row above, calls `RefreshRuntimeTileSizes()`, and syncs every layer. **`TrainingModeCPUSimd`** additionally enables Plan 9 SIMD on compute layers (Dense, SwiGLU, MHA, CNN1–3, RNN, LSTM, Embedding, Residual): forward `DotTile` / int8 dots, backward `SaxpyF32AccF64` / int8 saxpy. Other CPU modes call `SetSimdForwardRecursive(false)`.
- **GPU modes** (`TrainingModeGPUNormal`, `TrainingModeGPUSC`, `TrainingModeGPUMC`): initializes WebGPU if needed, `RefreshRuntimeTileSizes()`, resets the bind-group cache, `SyncToGPU()`, and ensures FP32 master buffers exist for backward. **`trainBatchWGPU`** uses **`TrainingModeGPUSC`** vs **`TrainingModeGPUMC`** to select **`GetGPUSCTileSize`** vs **`GetGPUMCTileSize`** per layer; **`GPUNormal`** uses untiled or generic dispatch per layer type.

For **interactive inference** (no explicit training mode), toggling **`VolumetricNetwork.EnableMultiCoreTiling`** chooses GPU SC vs MC tile maps (`wgpu_forward.go`), the same underlying maps training uses.

### CPU-SIMD training

```go
config := poly.DefaultTrainingConfig()
config.Mode = poly.TrainingModeCPUSimd
result, err := poly.Train[float32](network, batches, config)
```

The seven-layer harness (Lucy `examples/seven_layer`) benchmarks SC, MC, and SIMD training in its summary tables. On a Dense Float32 1×1×1 micro-benchmark (50 epochs), typical wall times are:

| Platform | SC | MC | SIMD |
|----------|-----|-----|------|
| amd64 | ~14 ms | ~16 ms | **~10 ms** |
| arm64 | ~9 ms | ~12 ms | **~10 ms** |

Per-layer forward/backward SIMD vs SC tables (amd64 and arm64) are in [simd.md — seven-layer benchmark results](simd.md#seven-layer-benchmark-results). SIMD is a CPU path only; it does not replace GPU training.

---

## Training paradigms: default (QAT-like) vs native exact

Loom has **two CPU training semantics**. They share the same `Train()` loop, loss, and backward walk — but differ in **which forward/backward kernels run** and **where weights are updated**.

| | **Default path** (`dense.go`, `rnn.go`, …) | **Native exact** (`*_native.go`, menu **[14]**) |
|---|---------------------------------------------|--------------------------------------------------|
| **Flag** | `UseExactDType = false` (default) | `UseExactDType = true` |
| **Forward weights** | `GetActive(dtype)` → dequant to **FP32**, matmul in float | `GetNative` / per-dot MAC rules in **storage dtype** |
| **Backward** | Gradients in **FP32** | Dtype-native gradient rules (int MAC for true integers) |
| **Optimizer** | `ApplyGradients` → updates **FP32 Master**, clears `Versions`, re-`Morph` on next forward | `ApplyGradientsNative` (MAC dtypes) or **in-place int8 update in backward** (true integers) |
| **What it models** | Train with a float surrogate; storage dtype is a cached view | Train as if the network literally runs in that dtype |

The native menu prints the split explicitly:

```
layer.go = GetActive FP32 dequant · *_native.go = GetNative MAC rules
```

### Default path ≈ QAT-like surrogate training

When `UseExactDType` is off, a layer tagged `DTypeInt8` (or Int32, FP8, …) still does most of its math in **float**:

1. **Forward:** `WeightStore.GetActive(layer.DType)` returns `Versions[dtype]` if present; tiled/SIMD paths cast or dequant to FP32 for `DotTile` / matmul.
2. **Backward:** `gradWeights` and `gradInput` accumulate in FP32.
3. **Update:** `ApplyGradients` subtracts `lr × grad` from **Master** (`[]float32`), then clears all cached `Versions`.
4. **Next forward:** `Morph(dtype)` re-quantizes Master into storage dtype.

This is **not** textbook QAT (no fake-quant nodes, no straight-through estimator in the graph), but it is **QAT-adjacent**: *quantize → dequant → train in float → re-quant*. The low-bit dtype is a **storage and inference view**; learning happens in FP32 master space.

`MorphToFloat32ForGPU` at upload time is pure **PTQ** (precision loss at inference only). Default CPU training is the same family: float math with periodic re-quantization.

### Native exact — two sub-flavors

Enable with:

```go
net.UseExactDType = true
```

Layers route to `DenseForwardNativeExact`, `RNNForwardNativeExact`, `LSTMForwardNativeExact`, etc. (`layer_native.go` lists supported types: Dense, SwiGLU, MHA, CNN1–3, RNN, LSTM, Embedding, Residual).

#### 1. Native MAC dtypes (Int32, Int64, FP8, Float16, …)

- Forward/backward use **per-dtype MAC rules** via `GetNative` — no bulk FP32 dequant buffer on the hot path.
- Gradients follow those MAC rules (e.g. `nativeGradW` / `nativeGradX` in `native_weight_ctx.go`).
- Optimizer: `ApplyGradientsNative` updates Master **and** the native storage slice (e.g. round to `int64` after each step).
- **SIMD:** `*_native_simd.go` materializes f32 tiles once per pass (`materializeF32Weights`) then runs `DotTile` / `SaxpyF32AccF64` — same numerics, faster kernels.

#### 2. True native integers (Int8, Int4, Ternary, Uint8, …)

This is the largest departure from QAT:

| Stage | Behaviour |
|-------|-----------|
| Forward | Real **int8 × int8 → int32** MAC, `>> 8`, clamp; activations cached as int8 (`ExactDense` cache) |
| Backward | Integer gate grads; `SaxpyI8ScaleI32Acc` / `SaxpyI8ShiftedInputGradAcc` for weight/input accumulation |
| Update | **`applyStochasticInt8Update` inside backward** — weights change in-place in int8 storage; external optimizer step is **skipped** (`ExactDense.WeightsUpdated`) |

The network is **literally int8 during training**, not a float copy with int8 snapshots.

### SIMD on top (not a third paradigm)

| SIMD mode | Math semantics |
|-----------|----------------|
| Default `*_simd.go` | Still `GetActive` FP32 — faster QAT-like path |
| Native `*_native_simd.go` | Same native-exact rules; `DotI8Tile` / int8 saxpy for true integers |

`SetSimdForward(true)` / `TrainingModeCPUSimd` choose the fast kernels; `UseExactDType` chooses **which training paradigm**.

### Lucy menus

| Menu | Path | Purpose |
|------|------|---------|
| **[7]** seven-layer suite | Default (`UseExactDType` off for most runs) | SC/MC/SIMD parity, save/reload, training across 21 dtypes |
| **[14]** native layer suite | `UseExactDType = true` per dtype | Per-layer native fwd/bwd/train × 21 dtypes; SIMD speedup columns when linked |

Benchmark results (amd64/arm64): [native_layers.md](native_layers.md).

See [quantization.md — Three training/inference modes](quantization.md#three-traininginference-modes) for how this relates to PTQ and [simd.md](simd.md) for native-exact SIMD file layout.

---

## CPU Training: Step by Step

```go
func trainBatchCPU[T Numeric](n *VolumetricNetwork, batch TrainingBatch[T], config *TrainingConfig) float64
```

### 1. Forward Pass with History Capture

```
histIn  [numLayers]*Tensor[T]  ← input to each layer
histPre [numLayers]*Tensor[T]  ← preAct from each layer

curr = batch.Input
for each layer idx:
    histIn[idx] = curr
    pre, post = DispatchLayer(layer, curr, nil)
    histPre[idx] = pre
    curr = post
```

The history arrays are what make backpropagation possible without a tape. Every layer caches what it received and what it produced before activation.

### 2. Loss and Gradient Computation

```
gradOut = ComputeLossGradient(curr, batch.Target, "mse")
lossVal = CalculateLoss(curr, batch.Target, "mse")
```

**MSE loss:**
```
L = (1/N) Σᵢ (output[i] - target[i])²

gradOut[i] = (2/N) × (output[i] - target[i])
```

### 3. Backward Pass

```go
_, layerGradients, _ := BackwardPolymorphic(n, gradOut, histIn, histPre)
```

`BackwardPolymorphic` walks the grid in **reverse** order (Z high to low, Y high to low, X high to low, L high to low). At each step:

```
gIn, gW = DispatchLayerBackward(layer, currentGrad, histIn[idx], nil, histPre[idx])
currentGrad = gIn                   ← flows back to previous layer
layerGradients[idx] = {gIn, gW}    ← stored for weight update
```

The backward pass for Dense computes:

```
gradPre[b,o] = gradOutput[b,o] × activation'(preAct[b,o])

gradWeights[o,i] += input[b,i] × gradPre[b,o]   (accumulated over batch)
gradInput[b,i]   += W[o,i] × gradPre[b,o]
```

### 4. Weight Update

```go
for idx := range n.Layers {
    if l.ExactDense != nil && l.ExactDense.WeightsUpdated {
        l.ExactDense.WeightsUpdated = false
        continue  // true-int8 layers already updated in backward
    }
    if layerGradients[idx][1] != nil {
        gW := ConvertTensor[T, float32](layerGradients[idx][1])
        ApplyRecursiveGradients(l, gW, config.LearningRate)
    }
}
```

**Default path** (`UseExactDType = false`): `ApplyRecursiveGradients` calls `WeightStore.ApplyGradients(gW, lr)`:

```
Master[i] -= lr × gradWeights[i]
```

After this, all cached `Versions` and `GPUWeights` are cleared, forcing re-quantization on the next forward pass.

**Native exact path** (`UseExactDType = true`):

- **MAC dtypes:** `ApplyGradientsNative` updates Master and the native storage slice together.
- **True integers (Int8, …):** backward already ran `applyStochasticInt8Update`; the loop above skips the layer when `ExactDense.WeightsUpdated` is set.

See [Training paradigms](#training-paradigms-default-qat-like-vs-native-exact) for the full comparison.

`ApplyRecursiveGradients` also recurses into `ParallelBranches` and `SequentialLayers`, using the `Nested` structure of the returned `gradWeights` tensor to route updates to the correct sub-layer.

---

## GPU Training: BeginFrame / FlushFrame

The GPU training path batches the entire forward + backward + weight-update into **one command buffer**:

```
ctx.BeginFrame()         ← create shared CommandEncoder
  │
  ├── forward pass: DispatchForwardLayer per layer
  ├── loss grad: DispatchMSEGradPartialLoss
  ├── backward: DispatchActivationBackward + DispatchBackwardLayer per layer
  └── update: DispatchApplyGradients per layer

ctx.FlushFrame()         ← ONE submit + destroy temp uniform bufs
  │
ReadBuffer(partialsBuf) ← only reads back numWG × float32 scalars
```

The loss value is computed from partial sums: `numWG = (totalOutput + 255) / 256` workgroups each sum 256 elements. The Go side only reads back `numWG` floats rather than the full output tensor.

GPU weight updates are applied directly in VRAM via `DispatchApplyGradients`, which runs a WGSL shader:

```wgsl
weights[i] -= lr * gradients[i]
```

This means the CPU master weights become stale after GPU training. A `ReadBuffer` + `Unpack` cycle is required if you want to access updated weights on the CPU.

---

## Loss Functions

| `LossType` | Formula | Gradient |
|:-----------|:--------|:---------|
| `"mse"` | `(1/N) Σ (out-target)²` | `(2/N)(out-target)` |
| `"cross_entropy"` | (not yet in `training.go`) | — |

The GPU MSE gradient shader (`DispatchMSEGradPartialLoss`) computes both the gradient tensor and partial sums in a single pass.

---

## Tween (neural target propagation)

**Tween** is the name used in this codebase for layer-local target propagation. In papers it often appears as *target propagation*, *difference target propagation*, or similar. Implementation: `tween.go`.

Tween is a gradient-free alternative that estimates what each layer *should* have produced rather than computing exact chain-rule gradients.

### Two Modes

**Chain Rule mode** (`UseChainRule = true`):

```
target = actual + gradient × GradientScale
```

This uses backpropagation to compute gradients, then shifts the target in the gradient direction. It is standard backprop dressed in Tween clothing.

**Pure Tween mode** (`UseChainRule = false`):

```
target[i] = Σⱼ w[i,j] × currentTarget[j] / totalWeight[j]
```

Estimates input targets using weighted importance from the layer's own weights, without computing derivatives. This is the biologically-motivated "local learning" variant. Supported for Dense, RNN, LSTM, MHA, and SwiGLU.

### The TweenState

```go
type TweenState[T Numeric] struct {
    ForwardActs     []*Tensor[T]    // what layers produced
    BackwardTargets []*Tensor[T]    // what they should have produced
    Gradients       []*Tensor[float32]
    LinkBudgets     []float32       // cosine similarity: actual vs target
    Gaps            []float32       // RMS distance: actual vs target
    Config          *TweenConfig
}
```

### Usage Pattern

```go
state := poly.NewTweenState[float32](network, poly.DefaultTweenConfig())
output := poly.TweenForward(network, state, input)
poly.TweenBackward(network, state, target)
state.CalculateLinkBudgets()
poly.ApplyTweenGaps(network, state, lr)
```

### Link Budget Gating

Before applying any weight update, the engine checks the layer's `LinkBudget` (cosine similarity between actual output and backward target, normalized to [0,1]):

```
if budget < 0.2 {
    skip update  // prevent corrupting "dead" layers
}
layerRate = lr × (0.5 + budget × 0.5)  // good signal = higher rate
```

This prevents gradient corruption in layers where the signal has been destroyed.

---

## VGStepBP Adaptive Rate

The README mentions `VGStepBP` (Variable Gradient Step Backpropagation) as an adaptive rate calculation. This integrates with the Tween `DepthScaleFactor` field:

```go
DepthScaleFactor: 1.1   // each deeper layer gets 1.1× the base rate
```

Deeper layers receive slightly higher learning rates to compensate for gradient attenuation through the network depth. This is a simple heuristic that avoids the full computation of per-layer adaptive optimizers.

---

## Gradient Explosion Detection

The `GradientClip` field in `TrainingConfig` (when non-zero) clips gradient norms. Additionally, the Tween gap system implicitly detects explosion: if `Gaps[i]` grows very large, the gap-based update `delta = lr × input × gap` will also be large, but the Link Budget gating prevents this from firing if the cosine similarity is low.

The README references "Gradient Explosion Detection & Damping" as a completed feature in the training automation section.

---

## Activation Functions (Forward and Backward)

All activation derivatives are computed analytically in `ActivateDerivative[T]`:

```
ReLU:    dA/dx = 1 if x > 0, else 0
SiLU:    dA/dx = σ(x)(1 + x(1-σ(x)))
GELU:    dA/dx ≈ CDF(x) + x × PDF(x)
Tanh:    dA/dx = 1 - tanh(x)²
Sigmoid: dA/dx = σ(x)(1 - σ(x))
Linear:  dA/dx = 1
```

In the backward pass, `gradOutput` is multiplied elementwise by the derivative of `preAct` before accumulating `gradWeights` and `gradInput`.

---

## The Full Training Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  EPOCH LOOP                                                     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  BATCH                                                   │   │
│  │                                                         │   │
│  │  batch.Input                                            │   │
│  │       │                                                 │   │
│  │       ▼                                                 │   │
│  │  [Forward Pass]  ──▶  histIn, histPre captured          │   │
│  │       │                                                 │   │
│  │       ▼                                                 │   │
│  │  prediction                                             │   │
│  │       │                                                 │   │
│  │       ▼                                                 │   │
│  │  [Loss + gradOut]  ◀── batch.Target                     │   │
│  │       │                                                 │   │
│  │       ▼                                                 │   │
│  │  [Backward Pass]  ──▶  layerGradients                   │   │
│  │       │                                                 │   │
│  │       ▼                                                 │   │
│  │  [ApplyRecursiveGradients]  ──▶  Master updated         │   │
│  │                                  Versions cleared       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  LossHistory appended, EpochTimes recorded                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## TrainingResult

```go
type TrainingResult struct {
    FinalLoss   float64
    TotalTime   time.Duration
    LossHistory []float64          // one entry per epoch
    EpochTimes  []time.Duration
}
```

`Train` returns this struct regardless of CPU or GPU path, making it easy to log or compare runs.
