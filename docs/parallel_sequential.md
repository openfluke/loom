# Parallel and Sequential Layers

This document explains `LayerParallel` and `LayerSequential` in depth: how they fan out and chain sub-layers, the five combination modes, the recursive activation tree, and how backpropagation flows through nested structures.

---

## LayerParallel

`ParallelForwardPolymorphic` fans the input to every branch simultaneously and then combines the results.

### Configuration

```go
layer.Type        = poly.LayerParallel
layer.CombineMode = "concat"   // or "add", "avg", "filter", "grid_scatter"
layer.ParallelBranches = []poly.VolumetricLayer{
    {Type: poly.LayerDense, InputHeight: 64, OutputHeight: 32, ...},
    {Type: poly.LayerRNN,   InputHeight: 64, OutputHeight: 32, ...},
    {Type: poly.LayerCNN1,  InputHeight: 64, ...},
}
```

Each entry in `ParallelBranches` is a full `VolumetricLayer` — it can itself be a `LayerParallel` or `LayerSequential`, enabling unlimited nesting.

### Combination Modes

#### "add"

Element-wise sum of all branch outputs. All branches must produce the same output shape.

```
Input ──▶ Branch 0 ──▶ [32]
Input ──▶ Branch 1 ──▶ [32]   →   [32] (sum of all)
Input ──▶ Branch 2 ──▶ [32]
```

Use for: residual-style ensembles, multi-path feature accumulation.

#### "avg"

Element-wise average of all branch outputs. Same shape requirement as "add".

```
Output[i] = (Branch0[i] + Branch1[i] + ... + BranchN[i]) / N
```

Use for: soft ensemble averaging where no single branch should dominate.

#### "concat" / "grid_scatter"

Concatenates all branch outputs into one flat tensor. Branch output sizes can differ.

```
Input ──▶ Branch 0 ──▶ [32]
Input ──▶ Branch 1 ──▶ [16]   →   [32, 16, 64] = [112]
Input ──▶ Branch 2 ──▶ [64]
```

`"grid_scatter"` behaves identically to `"concat"` in the current implementation — they share the same code path. The name signals intent: scatter the input across a grid of experts, then collect all outputs.

Use for: multi-scale feature extraction, heterogeneous expert outputs before a routing layer.

#### "filter" (Soft Mixture of Experts)

Uses a separate gate sub-layer to produce per-branch weights, then computes a weighted sum:

```go
layer.FilterGateConfig = &poly.VolumetricLayer{
    Type:         poly.LayerDense,
    InputHeight:  64,
    OutputHeight: 3,   // one scalar per branch
    Activation:   poly.ActivationLinear,
}
```

At forward time:

```
Input ──▶ FilterGateConfig ──▶ [numBranches]
                │
         Softmax(gate_logits)
                │
          [w0, w1, w2]  ← learned routing weights

Input ──▶ Branch 0 ──▶ [32] × w0
Input ──▶ Branch 1 ──▶ [32] × w1  →  [32] (weighted sum)
Input ──▶ Branch 2 ──▶ [32] × w2
```

Use for: differentiable Mixture of Experts (MoE), learned feature gating, adaptive multi-scale fusion.

---

## The Activation Tree (Tensor.Nested)

The key to making arbitrary nesting differentiable is the `Nested []*Tensor[T]` field on `Tensor`.

During `ParallelForwardPolymorphic`, each branch produces its own `(bPre, bOut)` pair. The branch `preAct` tensors are collected into a slice and stored as `Nested` on the returned `preAct`:

```go
preAct = &Tensor[T]{
    Data:   input.Data,     // proxy — carries input shape
    Shape:  input.Shape,
    DType:  input.DType,
    Nested: branchPreActs,  // [branch0.preAct, branch1.preAct, ...]
}
```

During `ParallelBackwardPolymorphic`, the backward function reads `preAct.Nested[i]` to get the correct cached state for each branch:

```go
var bPre *Tensor[T]
if preAct != nil && i < len(preAct.Nested) {
    bPre = preAct.Nested[i]
}
gIn, gW := DispatchLayerBackward(target, scaledGrad, input, nil, bPre)
```

This creates a recursive tree of activation caches that mirrors the nesting depth of the network:

```
preAct.Nested:
├── Branch 0 preAct
│     └── (if branch 0 is also Parallel)
│           └── .Nested
│                 ├── Sub-branch 0 preAct
│                 └── Sub-branch 1 preAct
├── Branch 1 preAct
└── Branch 2 preAct
```

The backward pass recursively walks this tree, ensuring each sub-layer gets the exact cached pre-activation it needs to compute its gradient.

---

## Gradient Flow Through Parallel

For "add" and "avg" modes, the same `gradOutput` (or a scaled version) is sent to every branch:

```
gradOutput
    │
    ├──── scaledGrad ──▶ Branch 0 backward ──▶ gradInput_0 + gradWeights_0
    ├──── scaledGrad ──▶ Branch 1 backward ──▶ gradInput_1 + gradWeights_1
    └──── scaledGrad ──▶ Branch 2 backward ──▶ gradInput_2 + gradWeights_2

gradInput = gradInput_0 + gradInput_1 + gradInput_2  (accumulated)
```

For "avg" mode, `scaledGrad = gradOutput / N` before dispatching.

For "concat" mode, the gradient is **sliced** by branch output size:

```
gradOutput [112]:
  branch 0 slice: gradOutput[0:32]   → Branch 0 backward
  branch 1 slice: gradOutput[32:48]  → Branch 1 backward
  branch 2 slice: gradOutput[48:112] → Branch 2 backward
```

For "concat" backward, the branch output size is determined by running a forward pass to measure `len(out.Data)`. This is a known overhead — for large models, consider caching branch output sizes.

The `gradWeights` returned by `ParallelBackwardPolymorphic` is a synthetic tensor with no `Data` — only `Nested`:

```go
gradWeights = &Tensor[T]{
    Nested: branchGradWeights,  // per-branch weight gradients
}
```

`ApplyRecursiveGradients` recognizes this pattern and dispatches weight updates to each branch recursively.

---

## LayerSequential

`SequentialForwardPolymorphic` chains sub-layers in order, each receiving the output of the previous one.

```go
layer.Type = poly.LayerSequential
layer.SequentialLayers = []poly.VolumetricLayer{
    {Type: poly.LayerDense,   InputHeight: 128, OutputHeight: 256, ...},
    {Type: poly.LayerRMSNorm, InputHeight: 256, ...},
    {Type: poly.LayerDense,   InputHeight: 256, OutputHeight: 64, ...},
}
```

This is how transformer blocks are typically assembled: `RMSNorm → MHA → RMSNorm → SwiGLU`.

### Step Containers

For each sub-layer, the forward pass stores a "step container" — a tensor whose `Nested` holds `[bPre, bInput, bSkip]`:

```go
stepContainer := &Tensor[T]{
    Nested: []*Tensor[T]{
        bPre,    // Nested[0]: preAct from this sub-layer
        current, // Nested[1]: the input this sub-layer received
        lastInput, // Nested[2]: the previous input (for skip connections)
    },
}
stepIntermediates[i] = stepContainer
```

The outer `preAct` returned by `SequentialForwardPolymorphic` carries all step containers in its `Nested`:

```go
preAct = &Tensor[T]{
    Data:   input.Data,
    Nested: stepIntermediates,  // [step0container, step1container, step2container]
}
```

### Sequential Backward

The backward pass iterates sub-layers in **reverse** order:

```go
for i := len(layer.SequentialLayers) - 1; i >= 0; i-- {
    container := preAct.Nested[i]
    bPre   = container.Nested[0]
    bInput = container.Nested[1]
    bSkip  = container.Nested[2]

    stepGradOutput = currentGrad
    if skipGradients[i+1] != nil {
        stepGradOutput.Add(skipGradients[i+1])  // add skip gradient
    }

    gIn, gW = DispatchLayerBackward(target, stepGradOutput, bInput, bSkip, bPre)
    currentGrad = gIn
}
```

`skipGradients` is a slice that accumulates gradients flowing back through skip connections inside the sequence. If a sub-layer (like `LayerResidual`) produces a gradient flowing back to an earlier step, it is accumulated here.

---

## Remote Links Inside Branches

Both `ParallelForwardPolymorphic` and `SequentialForwardPolymorphic` support `IsRemoteLink` on individual branches:

```go
if branch.IsRemoteLink && layer.Network != nil {
    if remote := layer.Network.GetLayer(branch.TargetZ, branch.TargetY, branch.TargetX, branch.TargetL); remote != nil {
        target = remote
    }
}
```

This allows a branch to redirect to any layer in the parent `VolumetricNetwork`, enabling cross-cell feature reuse without duplicating layer definitions.

---

## Tiling Propagation

When `layer.UseTiling = true` on the parent Sequential layer, the flag is propagated to each sub-layer before dispatch:

```go
if layer.UseTiling {
    target.UseTiling = true
    target.TileSize  = layer.TileSize
}
```

This means you can set tiling on the top-level Sequential layer and all its sub-layers inherit it automatically.

---

## Practical Example: Transformer Block as Sequential

```go
block := poly.VolumetricLayer{
    Type: poly.LayerSequential,
    SequentialLayers: []poly.VolumetricLayer{
        {
            Type:        poly.LayerRMSNorm,
            InputHeight: 512,
            OutputHeight: 512,
        },
        {
            Type:       poly.LayerMultiHeadAttention,
            DModel:     512,
            NumHeads:   8,
            NumKVHeads: 8,
            HeadDim:    64,
            MaxSeqLen:  2048,
        },
        {
            Type:        poly.LayerRMSNorm,
            InputHeight: 512,
            OutputHeight: 512,
        },
        {
            Type:         poly.LayerSwiGLU,
            InputHeight:  512,
            OutputHeight: 1364,  // ~2.67× hidden size
        },
    },
}
```

The entire block is a single `VolumetricLayer` entry in the grid. It runs as a mini-pipeline with the `preAct.Nested` tree tracking all four sub-layer states for backpropagation.
