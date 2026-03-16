# The DNA Engine: Topological Network Comparison

This document covers `ExtractDNA`, `CompareNetworks`, `CosineSimilarity`, and the `LogicShift` detection system that allows high-fidelity comparison between networks of different architectures or precisions.

---

## Motivation

When you train two networks starting from different random seeds, or morph a network from FP32 to INT8, or graft one network's layers onto another's, you want answers to questions like:

- How similar are these two networks, fundamentally?
- Have any functional patterns migrated to different spatial positions?
- Can I tell that a Binary network is doing the "same job" as its FP64 counterpart?

Standard weight comparison fails here — FP4 and FP64 weights cannot be directly compared. The DNA engine solves this by converting weights to **unit vectors** after precision simulation, then comparing directions rather than magnitudes.

---

## LayerSignature

```go
type LayerSignature struct {
    Z, Y, X, L int      // 3D coordinates in the grid
    Type        LayerType
    DType       DType
    Weights     []float32  // normalized (unit-vector) simulated weights
}
```

The `Weights` field is **not** the raw master weights. It is:

1. The master weights, passed through `SimulatePrecision(w, DType, Scale)` — this simulates what the weights actually behave like at their active precision
2. Then normalized to a unit vector via `Normalize(simulated)`

This means two layers with the same functional behavior but different scales will have the same (or very similar) `Weights` direction vector.

---

## ExtractDNA

```go
func ExtractDNA(n *VolumetricNetwork) NetworkDNA
```

`NetworkDNA` is `[]LayerSignature`. `ExtractDNA` iterates all `n.Layers` and builds a signature for each:

```go
for _, l := range n.Layers {
    simulated := make([]float32, len(l.WeightStore.Master))
    for i, w := range l.WeightStore.Master {
        simulated[i] = SimulatePrecision(w, l.DType, scale)
    }
    norm = Normalize(simulated)

    dna = append(dna, LayerSignature{
        Z: l.Z, Y: l.Y, X: l.X, L: l.L,
        Type:    l.Type,
        DType:   l.DType,
        Weights: norm,
    })
}
```

For layers with no weights (e.g., Softmax), the signature gets `Weights = []float32{1.0}` — a neutral presence marker that says "something exists here."

---

## Normalize

```go
func Normalize(v []float32) []float32
```

Computes the L2 norm of `v`, then divides each element:

```
mag = sqrt(Σᵢ vᵢ²)
output[i] = v[i] / mag
```

If `mag == 0` (zero weights), returns a zero vector. Two zero-weight layers will be considered identical (`CosineSimilarity` returns 1.0 for two zero vectors).

---

## CosineSimilarity

```go
func CosineSimilarity(s1, s2 LayerSignature) float32
```

Returns a similarity score in `[-1.0, 1.0]`:

- `+1.0` — identical direction (same functional behavior, regardless of scale)
- `0.0` — orthogonal (no correlation)
- `-1.0` — opposite direction

The function first checks for architectural compatibility:

```go
if s1.Type != s2.Type || s1.DType != s2.DType {
    return 0  // architectural mismatch
}
if len(s1.Weights) != len(s2.Weights) {
    return 0  // dimension mismatch
}
```

Then computes the dot product of the already-normalized weight vectors (the result is directly the cosine similarity, since `|v| = 1`).

> [!NOTE]
> The DType check in `CosineSimilarity` means networks of different precisions will always score 0 in direct layer comparison. Cross-precision comparison requires normalizing to a common type first. This is intentional — it prevents spurious high scores between, say, an INT8 layer and a Binary layer with the same weights.

---

## CompareNetworks

```go
func CompareNetworks(dna1, dna2 NetworkDNA) NetworkComparisonResult
```

Returns:

```go
type NetworkComparisonResult struct {
    OverallOverlap  float32
    LayerOverlaps   map[string]float32  // "z,y,x,l" → cosine score
    LogicShifts     []LogicShift
}
```

### Phase 1: Hierarchical Direct Overlap

For each layer in `dna1`, find the layer at the same `(Z, Y, X, L)` position in `dna2` and compute their similarity:

```go
for _, sig1 := range dna1 {
    posKey := fmt.Sprintf("%d,%d,%d,%d", sig1.Z, sig1.Y, sig1.X, sig1.L)
    for _, sig2 := range dna2 {
        if sig1.Z == sig2.Z && sig1.Y == sig2.Y && sig1.X == sig2.X && sig1.L == sig2.L {
            overlap := CosineSimilarity(sig1, sig2)
            res.LayerOverlaps[posKey] = overlap
            totalOverlap += overlap
            matchedCount++
        }
    }
}
OverallOverlap = totalOverlap / matchedCount
```

This gives the **Similarity Index (SI)** — how much of the network's functional structure was preserved between the two snapshots.

### Phase 2: Logic Drift Detection

For each layer in `dna1`, search **all** layers in `dna2` (not just the same position) for the best cosine match:

```go
for _, sig1 := range dna1 {
    bestOverlap := float32(-1.0)
    bestSig2 := LayerSignature{}

    for _, sig2 := range dna2 {
        overlap := CosineSimilarity(sig1, sig2)
        if overlap > bestOverlap {
            bestOverlap = overlap
            bestSig2 = sig2
        }
    }

    if bestOverlap > 0.8 && pos1 != pos2 {
        res.LogicShifts = append(res.LogicShifts, LogicShift{
            SourcePos: pos1,
            TargetPos: pos2,
            Overlap:   bestOverlap,
        })
    }
}
```

A `LogicShift` is recorded when the best match for a layer in `dna1` is found at a **different spatial position** in `dna2`. This detects functional migration: the layer learned the same thing, but its position in the grid moved.

```go
type LogicShift struct {
    SourcePos string   // "z,y,x,l" in dna1
    TargetPos string   // "z,y,x,l" in dna2 (different position)
    Overlap   float32  // similarity score > 0.8
}
```

The 0.8 threshold is hardcoded. Below 0.8, the match is considered coincidental.

---

## Full Comparison Flow

```
Network A (FP32, trained)           Network B (FP32, further trained)
         │                                    │
         ▼                                    ▼
   ExtractDNA(A)                       ExtractDNA(B)
         │                                    │
         │  LayerSignature[]                  │  LayerSignature[]
         │  - simulate precision              │  - simulate precision
         │  - normalize to unit vec           │  - normalize to unit vec
         │                                    │
         └──────────────┬─────────────────────┘
                        │
                        ▼
               CompareNetworks(dnaA, dnaB)
                        │
              ┌─────────┴──────────┐
              │                    │
              ▼                    ▼
   Phase 1: Direct          Phase 2: Cross-position
   position matching        best-match search
              │                    │
              ▼                    ▼
   LayerOverlaps           LogicShifts
   OverallOverlap           (migrations)
              │
              └──────────────────▶ NetworkComparisonResult
```

---

## Practical Applications

### Measuring Quantization Fidelity

Compare a FP32 network against itself after `Morph(DTypeInt8)`:

```go
dnaFP32 := poly.ExtractDNA(network)
// ... morph all layers to INT8 ...
dnaINT8 := poly.ExtractDNA(network)
result := poly.CompareNetworks(dnaFP32, dnaINT8)
// result.OverallOverlap close to 1.0 → quantization preserved function
```

### Detecting Training Convergence

Track `OverallOverlap` between consecutive epoch checkpoints. When it stops changing significantly, the network has converged.

### Finding Structural Overlap Between Different Architectures

Two networks with different layer counts or grid dimensions can be partially compared — `CompareNetworks` will only match positions that exist in both DNAs.

### Logic Drift Monitoring

If a specific layer's function migrates to a new grid coordinate during training (common in plastic/growing networks), `LogicShifts` will catch it with position keys like `"0,0,2,0" → "0,1,0,0"`.

---

## Roadmap Note

The README's TODO for v0.74.0 includes "DNA Splice / Genetic Crossover — extend dna.go". This would use `CompareNetworks` output to selectively merge two trained networks' weight vectors at the layer level, creating a child network that inherits functional structures from both parents.
