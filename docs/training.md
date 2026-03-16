# Training: Forward Pass, Backward Pass, Optimizers, and Learning

This document covers the full training pipeline: the forward and backward pass mechanics, loss computation, weight update strategies, gradient clipping, TargetProp, and the `VGStepBP` adaptive rate.

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
    if layerGradients[idx][1] != nil {
        gW := ConvertTensor[T, float32](layerGradients[idx][1])
        ApplyRecursiveGradients(l, gW, config.LearningRate)
    }
}
```

`ApplyRecursiveGradients` calls `WeightStore.ApplyGradients(gW, lr)`:

```
Master[i] -= lr × gradWeights[i]
```

After this, all cached `Versions` and `GPUWeights` are cleared, forcing re-quantization on the next forward pass.

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

## TargetProp: Alternative to Backpropagation

Neural Target Propagation (`target_prop.go`) is a gradient-free alternative that estimates what each layer *should* have produced rather than computing exact chain-rule gradients.

### Two Modes

**Chain Rule mode** (`UseChainRule = true`):

```
target = actual + gradient × GradientScale
```

This uses backpropagation to compute gradients, then shifts the target in the gradient direction. It is standard backprop dressed in TargetProp clothing.

**Pure TargetProp mode** (`UseChainRule = false`):

```
target[i] = Σⱼ w[i,j] × currentTarget[j] / totalWeight[j]
```

Estimates input targets using weighted importance from the layer's own weights, without computing derivatives. This is the biologically-motivated "local learning" variant. Supported for Dense, RNN, LSTM, MHA, and SwiGLU.

### The TargetPropState

```go
type TargetPropState[T Numeric] struct {
    ForwardActs     []*Tensor[T]    // what layers produced
    BackwardTargets []*Tensor[T]    // what they should have produced
    Gradients       []*Tensor[float32]
    LinkBudgets     []float32       // cosine similarity: actual vs target
    Gaps            []float32       // RMS distance: actual vs target
    Config          *TargetPropConfig
}
```

### Usage Pattern

```go
state := poly.NewTargetPropState[float32](network, poly.DefaultTargetPropConfig())
output := poly.TargetPropForward(network, state, input)
poly.TargetPropBackward(network, state, target)
state.CalculateLinkBudgets()
poly.ApplyTargetPropGaps(network, state, lr)
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

The README mentions `VGStepBP` (Variable Gradient Step Backpropagation) as an adaptive rate calculation. This integrates with the TargetProp `DepthScaleFactor` field:

```go
DepthScaleFactor: 1.1   // each deeper layer gets 1.1× the base rate
```

Deeper layers receive slightly higher learning rates to compensate for gradient attenuation through the network depth. This is a simple heuristic that avoids the full computation of per-layer adaptive optimizers.

---

## Gradient Explosion Detection

The `GradientClip` field in `TrainingConfig` (when non-zero) clips gradient norms. Additionally, the TargetProp gap system implicitly detects explosion: if `Gaps[i]` grows very large, the gap-based update `delta = lr × input × gap` will also be large, but the Link Budget gating prevents this from firing if the cosine similarity is low.

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
