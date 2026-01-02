# Paper 4: Native Integer Training

> **Target Venue:** TinyML, MLSys

## Abstract

Edge devices often lack FPUs. Loom provides **native int8/int16 forward and backward passes** via generics, enabling on-device training without float32.

---

## Key Results

From `tva/test_0_0_7.go` Part 2:

| Layer | float32 | int8 | Reduction |
|-------|---------|------|-----------|
| MHA | 116.5KB | 29.9KB | **4x** |
| SwiGLU | 29.9KB | 8.2KB | **3.6x** |

All 55 layer/dtype combinations pass round-trip tests.

---

## Technical Approach

```go
// Generic tensor backend
type Numeric interface {
    ~int8 | ~int16 | ~int32 | ~float32 | ~float64
}

type Tensor[T Numeric] struct {
    Data  []T
    Shape []int
}
```

Layers implement generic forward/backward:
```go
func DenseForwardGeneric[T Numeric](input, weights, bias *Tensor[T]) *Tensor[T]
func DenseBackwardGeneric[T Numeric](input, gradOutput, weights *Tensor[T]) (gradIn, gradW, gradB)
```

---

## Code References

| Component | Path |
|-----------|------|
| Generic Tensor | [`nn/tensor.go`](../nn/tensor.go) |
| Serialization | [`nn/serialization.go`](../nn/serialization.go) |
| Test Suite | [`tva/test_0_0_7.go`](../tva/test_0_0_7.go) Part 2 |

---

## How to Reproduce

```bash
go run tva/test_0_0_7.go
# Look for "PART 2: MULTI-PRECISION SAVE/LOAD"
```

---

**Related:** [Paper 1](research_paper_1_polyglot_runtime.md) | [Paper 2](research_paper_2_steptween.md) | [Paper 3](research_paper_3_heterogeneous_moe.md) | [Paper 5](research_paper_5_arc_stitching.md)
