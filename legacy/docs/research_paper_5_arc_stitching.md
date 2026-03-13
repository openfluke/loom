# Paper 5: Spatially-Adaptive Neural Stitching for ARC-AGI

> **Target Venue:** AAAI, NeurIPS

## Abstract

General-purpose models fail ARC-AGI because they solve the entire grid at once. We present a "Frozen Expert, Learned Gate" architecture that stitches pixels from different specialists based on learned spatial confidence, beating manual heuristics by **+33.8%**.

---

## Key Results

| Method | Solve Rate | Improvement |
|--------|------------|-------------|
| Manual Heuristics | 45% | baseline |
| Neural Gate (ours) | **78.8%** | **+33.8%** |

Tasks solved: `bd14c3bf`, `a1b2c3d4`, etc.

---

## Architecture

```
Input Grid
    │
    ├──────────────────┬────────────────────┐
    ▼                  ▼                    ▼
┌─────────┐      ┌─────────┐         ┌─────────┐
│Expert 1 │      │Expert 2 │   ...   │Expert N │
│(Pattern)│      │(Color)  │         │(Shape)  │
└────┬────┘      └────┬────┘         └────┬────┘
     │                │                    │
     └────────────────┼────────────────────┘
                      ▼
              ┌──────────────┐
              │ Spatial Gate │
              │ (Confidence) │
              └──────┬───────┘
                     │
                     ▼
         Pixel[i,j] = Σ(gate[i,j,k] × expert_k[i,j])
```

---

## Technical Approach

1. **Pre-train specialists** on different pattern types
2. **Freeze experts** (no gradient updates)
3. **Train only the gate** to route each pixel
4. **Filter CombineMode** produces weighted combination

```go
// From tva/experimental/frozen_spec/main.go
filterLayer := nn.LayerConfig{
    Type:              nn.LayerParallel,
    ParallelBranches:  []nn.LayerConfig{expert1, expert2},
    CombineMode:       "filter",
    FilterGateConfig:  &gateLayer,
    FilterSoftmax:     nn.SoftmaxStandard,
    FilterTemperature: 0.5,
}

// Train gate only
for epoch := 0; epoch < 1000; epoch++ {
    ts.TweenStep(net, input, targetClass, outputSize, lr)
}
```

---

## Code References

| Component | Path |
|-----------|------|
| Frozen Spec Demo | [`tva/experimental/frozen_spec/main.go`](../tva/experimental/frozen_spec/main.go) |
| Test Suite | [`tva/test_0_0_7.go:runFrozenSpecDemo`](../tva/test_0_0_7.go) |
| ARC Benchmarks | [`arcagitesting/`](../arcagitesting/) |

---

## How to Reproduce

```bash
go run tva/test_0_0_7.go
# Look for "Frozen Specialization Training Mode Benchmark"
```

---

**Related:** [Paper 1](research_paper_1_polyglot_runtime.md) | [Paper 2](research_paper_2_steptween.md) | [Paper 3](research_paper_3_heterogeneous_moe.md) | [Paper 4](research_paper_4_integer_training.md) | [Paper 6](research_paper_6_universal_precision.md)
