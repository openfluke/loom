# Softmax Variants

`LayerSoftmax` (type 15) implements ten distinct softmax variants, controlled by the `SoftmaxType` field on `VolumetricLayer`. All variants are fully differentiable and work across all 21 DTypes.

---

## The Standard Formula

All variants start from the numerically stable form:

```
logits_shifted = logits - max(logits)   ← prevents overflow
exp_vals[i]    = exp(logits_shifted[i])
probs[i]       = exp_vals[i] / sum(exp_vals)
```

This is implemented in `Softmax(logits []float32) []float32`.

---

## SoftmaxType Constants

```go
const (
    SoftmaxStandard     SoftmaxType = 0
    SoftmaxGrid         SoftmaxType = 1
    SoftmaxHierarchical SoftmaxType = 2
    SoftmaxTemperature  SoftmaxType = 3
    SoftmaxGumbel       SoftmaxType = 4
    SoftmaxMasked       SoftmaxType = 5
    SoftmaxSparse       SoftmaxType = 6
    SoftmaxAdaptive     SoftmaxType = 7
    SoftmaxMixture      SoftmaxType = 8
    SoftmaxEntmax       SoftmaxType = 9
)
```

---

## Variant 0: Standard

```
probs = softmax(logits)
```

The classic form. All outputs are positive and sum to 1. Smooth gradient everywhere.

**When to use:** Classification heads, final output layers, any time you need a valid probability distribution.

```
Input:  [2.0, 1.0, 0.1]
           ▼
Shifted: [1.9, 0.9, 0.0]
           ▼
Exps:    [6.69, 2.46, 1.00]
Sum = 10.15
           ▼
Output:  [0.66, 0.24, 0.10]  ← sums to 1.0
```

---

## Variant 3: Temperature

```
probs = softmax(logits / temperature)
```

Temperature `T` (stored in `VolumetricLayer.Temperature`) controls sharpness.

```
┌──────────────────────────────────────────────────────────────┐
│  temperature = 0.1 (sharp):                                  │
│    Input: [2.0, 1.8, 0.1] → Output: ≈[0.99, 0.01, 0.00]    │
│    Effect: "confident" — almost winner-takes-all              │
│                                                              │
│  temperature = 1.0 (standard):                               │
│    Input: [2.0, 1.8, 0.1] → Output: ≈[0.55, 0.45, 0.00]    │
│                                                              │
│  temperature = 5.0 (smooth):                                 │
│    Input: [2.0, 1.8, 0.1] → Output: ≈[0.40, 0.38, 0.22]    │
│    Effect: "uncertain" — options spread more evenly          │
└──────────────────────────────────────────────────────────────┘
```

**When to use:** Token sampling in language models (low T = greedy, high T = diverse), exploration vs. exploitation in RL.

---

## Variant 4: Gumbel

```
noise[i] = -log(-log(Uniform(0,1)))   ← Gumbel noise
probs = softmax(logits + noise)
```

Adds independent Gumbel noise to each logit before computing softmax. This produces stochastic samples that are biased toward higher logits but not deterministic. The Gumbel distribution is the natural noise for the `argmax` operation.

**When to use:** Discrete sampling without the `argmax` non-differentiability. Training generative models with categorical outputs. Controlled exploration in MoE routing.

```
Same logits, three calls:
  Call 1: [0.71, 0.24, 0.05]  ← high logit usually wins
  Call 2: [0.48, 0.40, 0.12]  ← noise sometimes shifts result
  Call 3: [0.82, 0.14, 0.04]
```

---

## Variant 5: Masked

```
masked_logits[i] = logits[i]  if mask[i] == true
                 = -1e9        if mask[i] == false
probs = softmax(masked_logits)
```

The `mask` field is `[]bool` on `VolumetricLayer`. Positions where `mask[i] = false` get `-1e9` in the logit, making their `exp` output effectively zero. After softmax, those positions have probability 0.

The backward pass respects the mask: gradients are zeroed for masked positions.

**When to use:**
- Causal attention (prevent attending to future tokens)
- Legal-move filtering (board games, planning)
- Expert routing where some experts are unavailable

```
Logits: [2.0, 1.0, 0.5, 1.5]
Mask:   [T,   F,   T,   T  ]

After masking: [2.0, -1e9, 0.5, 1.5]
After softmax: [0.63, 0.00, 0.11, 0.26]
               masked position → 0 ✓
```

---

## Variant 6: Sparse (Sparsemax)

Sparsemax is an alternative to softmax that can produce **exact zeros** — true sparsity rather than just very small values.

```
Algorithm:
1. Sort logits descending: z₁ ≥ z₂ ≥ ... ≥ zₙ
2. Find k = max { k : z_k - (Σᵢ≤ₖ zᵢ - 1)/k > 0 }
3. τ = (Σᵢ≤ₖ zᵢ - 1) / k
4. output[i] = max(0, z[i] - τ)
```

Implemented in `SoftmaxSparseHelper(logits)`.

```
Logits: [3.0, 1.0, -1.0, -3.0]
Standard softmax: [0.87, 0.12, 0.01, 0.00]   ← all non-zero
Sparsemax:        [0.75, 0.25, 0.00, 0.00]   ← exact zeros!
```

**When to use:**
- Attention when you want the model to focus on exactly a few tokens
- Interpretability (fewer non-zero attention weights to explain)
- MoE routing (hard assignment to a subset of experts)

---

## Variant 9: Entmax

Entmax is a family of distributions parameterized by `alpha`. It interpolates between softmax and sparsemax:

- `alpha = 1.0` → standard softmax
- `alpha = 2.0` → sparsemax
- `alpha = 1.5` → the recommended default (used in original paper)

```go
layer.EntmaxAlpha = 1.5   // set on VolumetricLayer
```

Implemented in `SoftmaxEntmaxHelper(logits, alpha)`:

```go
weight := alpha - 1.0
s1 := Softmax(logits)
s2 := SoftmaxSparseHelper(logits)
result[i] = (1-weight)*s1[i] + weight*s2[i]
// renormalize to sum to 1
```

**When to use:** When you want controllable sparsity. Start with `alpha=1.5` and tune toward 2.0 for sparser attention.

---

## Variant 1: Grid

Grid softmax applies standard softmax independently to each **row** of a 2D interpretation of the input:

```
Input flat tensor reinterpreted as [SoftmaxRows, SoftmaxCols]:
  Row 0: softmax([logits[0:cols]])    → row probs sum to 1
  Row 1: softmax([logits[cols:2cols]]) → row probs sum to 1
  ...
```

Each row is an independent probability distribution.

**When to use:**
- Native Mixture of Experts: each row represents one expert's output distribution
- Multi-label classification where each "group" of labels is mutually exclusive
- Per-head attention normalization without the full MHA overhead

```
Input (flat): [2.0, 1.0, | 0.5, 3.0, | 1.5, 1.5]
Rows=3, Cols=2:

  Row 0: softmax([2.0, 1.0]) = [0.73, 0.27]
  Row 1: softmax([0.5, 3.0]) = [0.08, 0.92]
  Row 2: softmax([1.5, 1.5]) = [0.50, 0.50]
```

---

## Variant 2: Hierarchical

Hierarchical softmax uses `HierarchyLevels []int` to define a tree structure. The last level of `HierarchyLevels` is used as the column count, with rows computed from `n / cols`. In practice it reduces to Grid softmax with the last level defining the partition.

**When to use:** Large vocabulary prediction where the vocabulary has a natural hierarchical structure (e.g., word categories → words).

---

## Variant 7: Adaptive

Adaptive softmax selects the softmax type based on input statistics (currently implemented as a fallback to standard softmax, intended for future dynamic routing logic).

---

## Variant 8: Mixture

Mixture softmax is a placeholder for weighted combinations of multiple softmax outputs. Currently falls back to standard softmax.

---

## Backward Pass

All variants share the standard softmax Jacobian:

```
gradLogits[j] = probs[j] × (gradOutput[j] - Σᵢ gradOutput[i] × probs[i])
             = probs[j] × (gradOutput[j] - dotProduct)
```

Implemented in `SoftmaxBackward(gradOutput, softmaxOutput []float32)`.

For Grid and Hierarchical variants, the Jacobian is applied independently to each row. For Masked, gradients are zeroed at masked positions before computing the Jacobian.

---

## GetLogits

`GetLogits[T Numeric](data []T, temp float64, dtype DType)` converts any `Tensor[T]` to `[]float32` with temperature scaling. It has specialized fast-paths for the most common types (float32, float64, int8, etc.) to avoid generic conversion overhead.

---

## Summary Table

| Variant | Produces zeros | Stochastic | Key parameter | Best for |
|:--------|:--------------|:-----------|:--------------|:---------|
| Standard | No | No | — | General classification |
| Temperature | No | No | `Temperature` | Sampling sharpness |
| Gumbel | No | Yes | — | Differentiable sampling |
| Masked | Yes (at mask) | No | `Mask []bool` | Causal attention |
| Sparse | Yes | No | — | Hard sparse attention |
| Entmax | Maybe | No | `EntmaxAlpha` | Tunable sparsity |
| Grid | No | No | `SoftmaxRows/Cols` | MoE, multi-group |
| Hierarchical | No | No | `HierarchyLevels` | Tree vocabularies |
| Adaptive | No | No | — | (future) |
| Mixture | No | No | — | (future) |
