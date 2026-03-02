# Network Grafting

Grafting is a technique for combining the learned expertise of multiple independently-trained networks into a single, more capable composite model — without retraining from scratch.

> [!NOTE]
> `grafting.go` is marked in-progress. The core `GraftNetworks` function is stable but deep weight copying and full training of grafted models has known caveats (see Shallow Copy section below).

---

## Concept

Each loom network following the standard Evolutionary Framework (`ef.go`) structure has this layout:

```
Network:
  Layer 0: Input / Adapter
  Layer 1: Hive (Parallel branches — the "expertise" layer)
  Layer 2: Merger
  Layer 3: Output
```

Grafting targets **Layer 1 (The Hive)**. It extracts all parallel branches from multiple networks and builds a **Super-Hive** — a single parallel layer containing all branches from all source networks.

```
Network A (Hive)         Network B (Hive)         Network C (Hive)
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│Branch A0│Branch A1│    │Branch B0│Branch B1│    │Branch C0│Branch C1│
└──────────────────┘    └──────────────────┘    └──────────────────┘
         │                       │                       │
         └──────────────┬────────────────────────────────┘
                        ▼
               Super-Hive (6 branches)
         ┌────────────────────────────────────┐
         │A0 │ A1 │ B0 │ B1 │ C0 │ C1        │
         └────────────────────────────────────┘
                        │
                [Combine Mode: concat / add / filter]
                        │
                     Output
```

---

## Usage

```go
// Train networks A, B, C independently
networkA := trainOnDomainA(...)
networkB := trainOnDomainB(...)
networkC := trainOnDomainC(...)

// Graft their hives together
superHive, err := nn.GraftNetworks(
    []*nn.Network{networkA, networkB, networkC},
    "filter",  // or "concat", "add", "avg", "grid_scatter"
)

// superHive is a *LayerConfig for a parallel layer
// Insert it into a new network at layer 1
```

---

## Combine Modes

| Mode | Behaviour | Best For |
|---|---|---|
| `"concat"` | Concatenate all branch outputs | When branch output sizes differ |
| `"add"` | Element-wise sum | When all branches have same output size |
| `"avg"` | Element-wise average | Ensemble averaging |
| `"filter"` | Learned softmax routing (MoE) | When you want adaptive expert selection |
| `"grid_scatter"` | Place each branch at a grid position | Spatial/structured output |

---

## Filter Mode (Mixture of Experts)

When `combineMode == "filter"`, a **gate network** is automatically created:

```go
// Automatically built gate: Dense(inputSize → numExperts)
// Uses ActivationScaledReLU
// Followed by standard softmax to produce routing weights
```

The gate learns to route each input to the most appropriate branch(es) dynamically:

```
Input
  │
  ├──▶ Gate(Dense → Softmax) ──▶ [0.7, 0.2, 0.1]  (routing weights)
  │
  ├──▶ Branch A ──▶ out_A
  ├──▶ Branch B ──▶ out_B
  └──▶ Branch C ──▶ out_C
          │
          ▼
  0.7×out_A + 0.2×out_B + 0.1×out_C  (weighted sum)
```

---

## Grid Scatter Mode

Arranges branches spatially in a square grid. Useful when branches represent independent sub-problems in a 2D structure:

```go
superHive, _ := nn.GraftNetworks(networks, "grid_scatter")
// Auto-computes grid dimensions: ceil(sqrt(numBranches)) × ceil(sqrt(numBranches))
// GridPositions: branch i → row (i/side), col (i%side)
```

---

## Shallow Copy Warning

> [!WARNING]
> Branch weight slices are currently **shallow-copied** — the grafted model shares the underlying `[]float32` arrays with the source networks. This means:
> - If you train the grafted model, the source network weights are also mutated.
> - If you want independent weights after grafting, manually deep-copy weight slices before training.

Deep copy helper:

```go
func deepCopyWeights(src []float32) []float32 {
    dst := make([]float32, len(src))
    copy(dst, src)
    return dst
}
```

A `LayerConfig.Clone()` method is planned for v0.1.0 to make this automatic.

---

## ScaleWeights Helper

```go
// Scale all weights in a slice by a factor (in-place)
nn.ScaleWeights(layer.Kernel, 0.5)  // Halve the weights
```

Useful for blending grafted branches — you can downscale all branches equally before merging to prevent the combined output from being N× larger.
