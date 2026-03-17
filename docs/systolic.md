# The Systolic Grid Engine

This document covers the `SystolicState`, `SystolicForward`, `SystolicBackward`, and `SystolicApplyTargetProp` functions that implement a clock-cycle-accurate discrete-time neural mesh.

---

## What Is the Systolic Grid?

Standard `ForwardPolymorphic` runs the entire network in one sequential sweep — input enters at coordinate (0,0,0,0) and the final output exits at the last coordinate. This is a **one-shot** pass.

The **Systolic Engine** treats the 3D grid as a living mesh. Each "tick" of the neural clock fires every layer simultaneously. Each layer reads from the previous tick's output buffers and writes to a new set of output buffers. After all layers have fired, the buffers swap. This is classical **double buffering** applied to neural computation.

```
Standard ForwardPolymorphic:

  Input ──▶ L0 ──▶ L1 ──▶ L2 ──▶ L3 ──▶ Output
  (one complete pass per call)


Systolic Grid (one clock cycle):

  Tick N:                        Tick N+1:
  ┌──────┬──────┬──────┐          ┌──────┬──────┬──────┐
  │ L0   │ L1   │ L2   │          │ L0   │ L1   │ L2   │
  │fires │fires │fires │ ──swap──▶│fires │fires │fires │
  │      │      │      │ buffers  │      │      │      │
  └──────┴──────┴──────┘          └──────┴──────┴──────┘
  All layers process simultaneously    Same pattern
```

The key insight: **every layer in the grid has the opportunity to update its output every clock cycle**, not just when an input happens to flow through it sequentially.

---

## SystolicState

```go
type SystolicState[T Numeric] struct {
    LayerData  []*Tensor[T]     // current output of every layer
    NextBuffer []*Tensor[T]     // write target for the current tick

    HistoryIn  [][]*Tensor[T]   // [step][layerIdx] → input to that layer at that step
    HistoryPre [][]*Tensor[T]   // [step][layerIdx] → preAct at that step

    StepCount uint64
    mu        sync.RWMutex

    tpState   *TargetPropState[T]  // optional TargetProp bridge
    lastInput *Tensor[T]
}
```

`LayerData[idx]` is what layer `idx` produced in the **previous** clock cycle. `NextBuffer[idx]` is what layer `idx` will produce in the **current** cycle. After the cycle, they swap.

Create with:

```go
state := poly.NewSystolicState[float32](network)
state.SetInput(inputTensor)  // loads input into LayerData[0]
```

---

## SystolicForward: One Clock Cycle

```go
elapsed := poly.SystolicForward(network, state, captureHistory bool)
```

Each call advances the mesh by exactly one discrete time step. All layers execute during this one call.

### Sequential Mode (UseTiling = false)

```go
for idx := range n.Layers {
    l := &n.Layers[idx]
    if l.IsDisabled { pass through; continue }

    // Resolve input source
    var input *Tensor[T]
    if l.IsRemoteLink {
        tIdx := n.GetIndex(l.TargetZ, l.TargetY, l.TargetX, l.TargetL)
        input = s.LayerData[tIdx]          // reads from REMOTE layer's output
    } else if idx > 0 {
        input = s.LayerData[idx-1]         // reads from preceding layer
    } else {
        input = s.LayerData[0]             // reads injection point
    }

    pre, post := DispatchLayer(l, input, nil)
    s.NextBuffer[idx] = post
}

// Swap double buffers
copy(s.LayerData, s.NextBuffer)
s.StepCount++
```

### Parallel Tiled Mode (UseTiling = true)

When `n.UseTiling = true`, goroutines process 4×4×4 spatial tiles concurrently:

```go
var wg sync.WaitGroup
for zTile ...:
  for yTile ...:
    for xTile ...:
      wg.Add(1)
      go func(zT, zE, yT, yE, xT, xE int) {
          defer wg.Done()
          for z := zT; z < zE; z++ {
              for y := yT; y < yE; y++ {
                  for x := xT; x < xE; x++ {
                      // dispatch layers in this tile
                  }
              }
          }
      }(...)
wg.Wait()
```

The mutex (`s.mu`) is held for the duration of the sequential path, and for individual history writes in the parallel path. The `NextBuffer` slice is pre-allocated so concurrent writes to different indices are safe.

### History Capture

If `captureHistory = true`, each tick appends to `HistoryIn` and `HistoryPre`:

```
After tick N:
  HistoryIn[N][idx]  = what layer idx received
  HistoryPre[N][idx] = preAct that layer idx produced
```

This history is the foundation for `SystolicBackward` (BPTT) and is required before calling `SystolicBackward`. It consumes memory proportional to `Steps × Layers × FeatureSize` — use only when training.

---

## Spatial Feedback (Remote Links in Systolic Mode)

The systolic engine is where `IsRemoteLink` reaches its full potential. Because `s.LayerData[tIdx]` is always the **previous tick's** output (not the current tick's), a remote link to an earlier coordinate creates genuine recurrence:

```
Tick N-1:
  Layer A (0,0,0) produces output → stored in LayerData[0]

Tick N:
  Layer B (0,2,0) has IsRemoteLink pointing to (0,0,0)
  → Layer B reads LayerData[0]  (from tick N-1, not current tick)
  → Layer B effectively "remembers" what A produced one cycle ago

This is the discrete-time equivalent of an RNN hidden state.
```

```
┌────────────────────────────────────────────────────────────────┐
│  SPATIAL FEEDBACK DIAGRAM                                       │
│                                                                │
│  Tick N-1:    A ──output──▶ LayerData[A]                       │
│                                                                │
│  Tick N:      B ──IsRemoteLink──▶ reads LayerData[A] from N-1  │
│               B produces new output → LayerData[B]             │
│                                                                │
│  Tick N+1:    A reads updated B output if A is also remote     │
│               → Full spatial RNN at mesh scale                 │
└────────────────────────────────────────────────────────────────┘
```

---

## SystolicBackward: BPTT Through the Mesh

```go
gradIn, layerGradients, err := poly.SystolicBackward(network, state, gradOutput)
```

This implements **Backpropagation Through Time (BPTT)** across the systolic history. It walks backwards through both time steps and spatial coordinates.

### Algorithm

```
gradBuffers[numLayers-1] = gradOutput   // seed with final error

for step from (numSteps-1) downto 0:
    nextGradBuffers = new zero buffers

    for idx from (numLayers-1) downto 0:
        input = HistoryIn[step][idx]
        pre   = HistoryPre[step][idx]
        grad  = gradBuffers[idx]

        gIn, gW = DispatchLayerBackward(l, grad, input, nil, pre)

        // Accumulate weight gradients across all time steps
        layerGradients[idx][1] += gW   (if exists)

        // Route gIn back to the source of input for this layer
        accumulateMeshGrad(network, nextGradBuffers, idx, gIn)

    gradBuffers = nextGradBuffers

return gradBuffers[0]   // gradient with respect to the initial input
```

`accumulateMeshGrad` determines where to send `gIn`:

- If `IsRemoteLink`: send to `TargetZ/Y/X/L` coordinates
- Otherwise: send to `idx - 1`
- If `idx == 0`: send to the input site

This correctly routes gradients through the spatial topology — remote links receive their share of the gradient from every layer that consumed their output.

---

## SystolicApplyTargetProp

```go
poly.SystolicApplyTargetProp(network, state, globalTarget, lr)
```

Bridges the systolic mesh with the `TargetProp` machinery. At each call:

1. If `state.tpState == nil`, create a new `TargetPropState` with `UseChainRule = false` (gap-based learning — appropriate for the continuous-time mesh)
2. Copy current `LayerData` into `tpState.ForwardActs` (the mesh's current "what is" state)
3. Call `TargetPropBackward(n, tpState, globalTarget)` to compute what each layer *should* produce
4. `CalculateLinkBudgets()` — measure cosine similarity between actual and target at each node
5. `ApplyTargetPropGaps(n, tpState, lr)` — update weights using the gap signal, gated by link budgets

This enables **online, asynchronous learning** on a live mesh — you can inject a global target at any time and the weights update locally at each node based on their current output gap.

---

## Double Buffer Guarantees

The double buffer swap (`copy(s.LayerData, s.NextBuffer)`) happens after all layers have written to `NextBuffer`. This guarantees:

1. A layer at coordinate (0,0,2) cannot see the output of (0,0,1) from the *current* tick, only from the previous tick
2. Concurrent goroutines in tiled mode write to different indices of `NextBuffer` without conflict
3. Remote links always see stable, previous-tick values regardless of which goroutine happens to fire first

This is the "clock cycle accuracy" mentioned in the README.

---

## When to Use the Systolic Engine

Use `SystolicForward` / `SystolicApplyTargetProp` when you need:

- **Continuous operation**: the network runs indefinitely, processing new inputs each tick
- **Spatial feedback**: remote links that create mesh-level recurrence
- **Online learning**: weight updates interleaved with forward passes
- **Parallel processing**: the tiled mode can saturate multi-core CPUs

Use `ForwardPolymorphic` / `BackwardPolymorphic` when you need:

- **Batch training**: multiple training examples per weight update
- **GPU acceleration**: the GPU path uses `trainBatchWGPU`, not the systolic engine
- **Deterministic single-pass inference**: no history overhead

> [!TIP]
> The README's phrase "use `SystolicForward` and `SystolicApplyTargetProp` when you need a living network that evolves and learns over time rather than a static pipeline" captures this distinction perfectly.
