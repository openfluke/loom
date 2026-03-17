# The Dispatcher Pattern and 3D Coordinate System

This document explains how `DispatchLayer` and `DispatchLayerBackward` work as runtime jump tables, how the 3D coordinate system maps to `VolumetricLayer` positions, and how `IsRemoteLink` enables spatial hopping across the grid.

---

## Why a Dispatcher?

A naive implementation of a polymorphic neural network would embed a large `switch` inside the forward loop:

```go
// Naive — thread-divergence on GPU, hard to fuse
for _, layer := range layers {
    switch layer.Type {
    case LayerDense:   output = denseForward(layer, input)
    case LayerCNN2:    output = cnn2Forward(layer, input)
    // ...
    }
}
```

M-POLY-VTD separates concerns: the **traversal loop** iterates coordinates, and the **dispatcher** makes the type-specific call. This decoupling is what makes GPU kernel fusion possible in the future — the driver can inspect a group of same-type layers and launch a single batched shader rather than 19 separate ones.

---

## DispatchLayer

```go
func DispatchLayer[T Numeric](
    layer *VolumetricLayer,
    input, skip *Tensor[T],
) (preAct, postAct *Tensor[T])
```

This is a generic function. The type parameter `T` is inferred from `input`. Every call returns two tensors:

- `preAct` — the layer's internal state before the final activation. For Parallel/Sequential layers this carries the nested activation tree in `preAct.Nested`.
- `postAct` — the result of applying the activation function to `preAct`. This is what flows to the next layer.

The full routing table:

```
layer.Type ──switch──▶ function called
───────────────────────────────────────────────────────────────
LayerResidual          ResidualForwardPolymorphic(layer, input, skip)
LayerDense             DenseForwardPolymorphic(layer, input)
LayerCNN1              CNN1ForwardPolymorphic(layer, input)
LayerCNN2              CNN2ForwardPolymorphic(layer, input)
LayerCNN3              CNN3ForwardPolymorphic(layer, input)
LayerRNN               RNNForwardPolymorphic(layer, input)
LayerLSTM              LSTMForwardPolymorphic(layer, input)
LayerMultiHeadAttention MHAForwardPolymorphic(layer, input)
LayerSwiGLU            SwiGLUForwardPolymorphic(layer, input)
LayerRMSNorm           RMSNormForwardPolymorphic(layer, input)
LayerLayerNorm         LayerNormForwardPolymorphic(layer, input)
LayerConvTransposed1D  ConvTransposed1DForwardPolymorphic(layer, input)
LayerConvTransposed2D  ConvTransposed2DForwardPolymorphic(layer, input)
LayerConvTransposed3D  ConvTransposed3DForwardPolymorphic(layer, input)
LayerEmbedding         EmbeddingForwardPolymorphic(layer, input)
LayerKMeans            KMeansForwardPolymorphic(layer, input)
LayerSoftmax           SoftmaxForwardPolymorphic(layer, input)
LayerParallel          ParallelForwardPolymorphic(layer, input)
LayerSequential        SequentialForwardPolymorphic(layer, input)
default                DenseForwardPolymorphic(layer, input)
───────────────────────────────────────────────────────────────
```

---

## DispatchLayerBackward

```go
func DispatchLayerBackward[T Numeric](
    layer *VolumetricLayer,
    gradOutput, input, skip, preAct *Tensor[T],
) (gradInput, gradWeights *Tensor[T])
```

The mirror of `DispatchLayer`. Returns:

- `gradInput` — the gradient to pass to the layer that produced `input` (propagates error upstream)
- `gradWeights` — the gradient for this layer's own weights (used to update `WeightStore.Master`)

The routing table is symmetric to the forward pass. The `skip` argument is used only by `ResidualBackwardPolymorphic`.

---

## The 3D Grid Traversal

`ForwardPolymorphic[T]` iterates the grid in reading order:

```go
for z := 0; z < n.Depth; z++ {
    for y := 0; y < n.Rows; y++ {
        for x := 0; x < n.Cols; x++ {
            for l := 0; l < n.LayersPerCell; l++ {
                idx := n.GetIndex(z, y, x, l)
                layer := &n.Layers[idx]
                // ...
                _, post := DispatchLayer(layer, currentTensor, nil)
                currentTensor = post
            }
        }
    }
}
```

The flattened index formula:

```
idx = z * (Rows * Cols * LayersPerCell)
    + y * (Cols * LayersPerCell)
    + x * (LayersPerCell)
    + l
```

Visually, for a (Depth=1, Rows=2, Cols=3, LayersPerCell=1) network:

```
z=0:
  ┌─────────────┬─────────────┬─────────────┐
  │ (0, 0, 0,0) │ (0, 0, 1,0) │ (0, 0, 2,0) │  ← idx 0,1,2
  │   idx=0     │   idx=1     │   idx=2     │
  ├─────────────┼─────────────┼─────────────┤
  │ (0, 1, 0,0) │ (0, 1, 1,0) │ (0, 1, 2,0) │  ← idx 3,4,5
  │   idx=3     │   idx=4     │   idx=5     │
  └─────────────┴─────────────┴─────────────┘

Data flows: idx=0 ──▶ idx=1 ──▶ idx=2 ──▶ idx=3 ──▶ idx=4 ──▶ idx=5
```

`BackwardPolymorphic` walks in reverse (z, y, x, l all reversed), using cached `inputs[idx]` and `preActs[idx]` from the forward pass.

---

## Tiled Traversal

When `n.UseTiling = true`, `ForwardPolymorphic` uses a blocked spatial traversal with tile size 4:

```
for zTile := 0; zTile < Depth; zTile += 4 {
  for yTile := 0; yTile < Rows; yTile += 4 {
    for xTile := 0; xTile < Cols; xTile += 4 {
      // Process 4×4×4 tile of cells
    }
  }
}
```

This is the CPU-side analogue of the GPU workgroup tile strategy. The intent is to improve data locality: all layers in a 4×4×4 spatial neighborhood execute together, keeping their weight data warm in L2/L3 cache.

---

## VolumetricLayer: The Coordinate Record

Every `VolumetricLayer` contains its own position:

```go
type VolumetricLayer struct {
    Network     *VolumetricNetwork  // back-pointer
    Type        LayerType
    Activation  ActivationType
    DType       DType
    WeightStore *WeightStore

    Z int  // Depth coordinate
    Y int  // Row coordinate
    X int  // Col coordinate
    L int  // Layer index within cell

    // Spatial Routing
    IsRemoteLink bool
    TargetZ, TargetY, TargetX, TargetL int

    // ... configuration fields
}
```

The `(Z, Y, X, L)` fields are set during `NewVolumetricNetwork` and are the canonical address. `GetLayer(z, y, x, l)` returns a pointer into the flat `Layers` slice using `GetIndex`.

---

## IsRemoteLink: Spatial Hopping

A layer with `IsRemoteLink = true` does not receive its input from the previous layer in reading order. Instead, it reads from the output of whatever layer lives at `(TargetZ, TargetY, TargetX, TargetL)`.

This enables:

1. **Skip connections** — hop over several layers in the grid
2. **Feedback loops** — target a layer at an *earlier* coordinate (biological recurrence)
3. **Parallel expert routing** — multiple layers at different positions all reading the same source
4. **Cross-depth signals** — connect depth=0 outputs to depth=2 inputs

```
Standard flow:             Remote link (skip):

 (0,0,0) → (0,0,1)          (0,0,0) ────────────────────┐
              │               (0,0,1) → (0,0,2) → ...    │
           (0,0,2)                                        │
              │               (0,2,0) ←── IsRemoteLink ──┘
           (0,0,3)             └── reads output of (0,0,0)

Feedback loop:

 (0,0,0)
    │
 (0,0,1)
    │
 (0,0,2) ─── IsRemoteLink ──▶ TargetZ=0, TargetY=0, TargetX=0
                                (reads from cycle N-1's output
                                 of layer (0,0,0) — Systolic only)
```

In `ForwardPolymorphic`, a remote-linked layer simply receives `currentTensor` like any other layer; the remote link semantic is only fully honored by `SystolicForward`, which maintains per-layer output buffers across time steps.

In `ParallelForwardPolymorphic` and `SequentialForwardPolymorphic`, remote links are resolved by calling `layer.Network.GetLayer(branch.TargetZ, ...)` and dispatching the resolved layer pointer.

---

## The GPU Dispatch Path

When `n.UseGPU = true`, the training loop calls `ctx.DispatchForwardLayer(l, batchSize, curBuf, preBuf)` instead of `DispatchLayer`. This function is in `wgpu_forward.go` and routes to the appropriate WGSL compute shader based on `l.Type`.

The same dispatcher philosophy applies: one function, one switch, explicit routing. The difference is that inputs and outputs are `*wgpu.Buffer` handles in VRAM rather than `*Tensor[T]` in RAM.

```
trainBatchWGPU:

  BeginFrame()  ← create shared CommandEncoder
     │
     ├── for each layer forward:
     │   └── ctx.DispatchForwardLayer(l, ...) ← records into encoder
     │
     ├── DispatchMSEGradPartialLoss(...)       ← records into encoder
     │
     ├── for each layer backward (reverse):
     │   ├── ctx.DispatchActivationBackward(...)
     │   ├── ctx.DispatchBackwardLayer(l, ...)
     │   └── ctx.DispatchApplyGradients(...)
     │
  FlushFrame()  ← ONE submit for entire forward + backward + weight update
     │
  ReadBuffer(partialsBuf) ← only reads back tiny loss scalars
```

This single-submission design reduces Go-to-GPU driver overhead from ~150+ round trips per batch to exactly 1.

---

## Disabled Layers

Setting `layer.IsDisabled = true` causes both `ForwardPolymorphic` and `SystolicForward` to skip the layer entirely. In `SystolicForward`, a disabled layer passes its input buffer through to `NextBuffer` unchanged. This is the mechanism for implementing sparse MoE expert activation — gate layers can conditionally disable branches.
