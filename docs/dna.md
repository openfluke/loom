# The DNA Engine: Topological Network Fingerprinting

This document covers `ExtractDNA`, `CosineSimilarity`, `CompareNetworks`, `LogicShift` detection, and the recursive signature extraction for all 19 layer types in `dna.go`.

For the **Evolution Engine** (DNA Splice + NEAT mutations), see [evolution.md](evolution.md).

---

## Why DNA?

Standard weight comparison breaks across precisions — you can't directly compare an INT8 weight against an FP32 weight. The DNA engine solves this by converting every layer's weights to a **unit direction vector** after simulating precision loss. Comparing direction vectors (cosine similarity) instead of raw values means:

- FP32 and INT8 representations of the same model look nearly identical
- Two networks trained on the same task converge toward the same DNA
- Structural changes (different layer order, different grid positions) are detectable as **logic shifts**

```
Raw FP32 weights  ──►  scale (× ws.Scale)  ──►  Normalize  ──►  unit vector
       │                                         (L2 norm)         "DNA strand"
       │
       └── FP4 weights ──►  scale (× ws.Scale)  ──►  Normalize  ──►  same direction ≈ 1.0 similarity
```

---

## Core Types

```go
// The "DNA strand" of a single layer
type LayerSignature struct {
    Z, Y, X, L int       // 3D grid coordinates
    Type        LayerType
    DType       DType
    Weights     []float32 // L2-normalized, scale-applied master weights
}

// The complete genetic blueprint of a network
type NetworkDNA []LayerSignature
```

---

## ExtractDNA — all 19 layer types

```go
func ExtractDNA(n *VolumetricNetwork) NetworkDNA
```

Iterates every layer in the network, calls `extractLayerSignature(l)`, and wraps the result with position and type metadata. The signature extraction logic handles all 19 layer types:

```
                    VolumetricNetwork
                           │
              ┌────────────┼────────────┐
              │            │            │
         LayerDense   LayerParallel  LayerSoftmax
         LayerRNN     LayerSequential LayerResidual
         LayerLSTM    (recursive)    (weightless)
         LayerMHA
         LayerSwiGLU
         LayerRMSNorm
         LayerLayerNorm
         LayerCNN1/2/3
         LayerConvT1/2/3D
         LayerEmbedding
         LayerKMeans
              │
              ▼
    extractLayerSignature(l)
              │
    ┌─────────┼──────────────┐
    │         │              │
    ▼         ▼              ▼
 weighted  recursive     weightless
 layers    containers    layers
    │         │              │
    ▼         ▼              ▼
 Master   flatten all    []float32{1.0}
 weights  branches
    │         │
    ▼         ▼
scale(×ws.Scale)   Normalize(concat)
    │
    ▼
 Normalize
    │
    ▼
 []float32 unit vector
```

### Weighted layers (Dense, RNN, LSTM, MHA, CNN*, ConvTransposed*, SwiGLU, RMSNorm, LayerNorm, Embedding, KMeans)

```go
// All weighted layers follow this path:
scale := l.WeightStore.Scale
if scale == 0 { scale = 1.0 }
simulated := make([]float32, len(l.WeightStore.Master))
for i, w := range l.WeightStore.Master {
    if scale != 1.0 {
        simulated[i] = w * scale
    } else {
        simulated[i] = w
    }
}
return Normalize(simulated)
```

Applying the layer's scale factor before normalizing means the DNA of an INT8 Dense layer and an FP32 Dense layer with the same trained weights will be nearly identical — both normalize to the same unit direction.

### Structural containers (Parallel, Sequential) — recursive extraction

Parallel and Sequential layers contain nested layers (`ParallelBranches`, `SequentialLayers`). A naive approach that returned `{1.0}` for both would make any two parallel layers look identical regardless of what's inside them. Instead, the engine recurses:

```
LayerParallel
├── Branch 0 (Dense 32×32)   ──► extractLayerSignature ──► unit vec A  ─┐
├── Branch 1 (RMSNorm 32)    ──► extractLayerSignature ──► unit vec B  ─┤ concat
└── FilterGateConfig (Dense) ──► extractLayerSignature ──► unit vec C  ─┘
                                                                         │
                                                                    Normalize(flat)
                                                                         │
                                                                    single unit vec
                                                                    representing ALL
                                                                    nested weights
```

```go
case LayerParallel:
    var flat []float32
    for _, branch := range l.ParallelBranches {
        if branch.IsRemoteLink { continue }   // remote links have no local weights
        flat = append(flat, extractLayerSignature(branch)...)
    }
    if l.FilterGateConfig != nil {
        flat = append(flat, extractLayerSignature(*l.FilterGateConfig)...)
    }
    if len(flat) == 0 { return []float32{1.0} }
    return Normalize(flat)

case LayerSequential:
    var flat []float32
    for _, sub := range l.SequentialLayers {
        flat = append(flat, extractLayerSignature(sub)...)
    }
    if len(flat) == 0 { return []float32{1.0} }
    return Normalize(flat)
```

Remote links (`IsRemoteLink = true`) are spatial hops with no local weights — they are skipped during extraction.

### Weightless layers (Softmax, Residual)

```go
case LayerSoftmax, LayerResidual:
    return []float32{1.0}
```

A `{1.0}` vector is a neutral presence marker. Two Softmax layers at the same position will score `1.0` similarity (identical), which is correct — they are architecturally identical by definition.

---

## Normalize

```go
func Normalize(v []float32) []float32
```

Converts a weight vector to a unit vector:

```
mag = sqrt(v[0]² + v[1]² + ... + v[n]²)
output[i] = v[i] / mag
```

- If `mag == 0` (all-zero weights), returns a zero vector
- Two zero vectors score `1.0` similarity (both represent an untrained/zeroed layer)
- One zero + one nonzero scores `0.0` (orthogonal by convention)

---

## CosineSimilarity

```go
func CosineSimilarity(s1, s2 LayerSignature) float32
```

Returns a score in `[-1.0, 1.0]` comparing two layer signatures:

```
         s1.Weights · s2.Weights
sim  =  ─────────────────────────   =  dot product  (since |s1| = |s2| = 1)
              |s1| × |s2|
```

Guard rails:

| Condition | Returns |
|:----------|:--------|
| `s1.Type != s2.Type` | `0.0` — architectural mismatch |
| `s1.DType != s2.DType` | `0.0` — precision mismatch |
| `len(s1.Weights) != len(s2.Weights)` | `0.0` — dimension mismatch |
| Both zero vectors | `1.0` — identical untrained layers |
| One zero, one nonzero | `0.0` — no similarity |

Similarity values to interpret:

```
-1.0  ────────────  0.0  ────────────  +1.0
  │                  │                   │
opposite          no match          identical
direction                           direction
(learned to      (different         (same functional
do opposite)      purpose)           role)
```

---

## CompareNetworks

```go
func CompareNetworks(dna1, dna2 NetworkDNA) NetworkComparisonResult

type NetworkComparisonResult struct {
    OverallOverlap float32
    LayerOverlaps  map[string]float32   // "z,y,x,l" → score
    LogicShifts    []LogicShift
}
```

Two-phase comparison:

### Phase 1 — Direct Position Matching

Match each layer in `dna1` with the layer at the same `(Z, Y, X, L)` position in `dna2`:

```
dna1:   [L0: Dense]  [L1: RNN]  [L2: Dense]
              │              │          │
              │ same pos     │          │ same pos
              ▼              ▼          ▼
dna2:   [L0: Dense]  [L1: Dense]  [L2: Dense]
              │              │          │
          sim=0.94       sim=0.0    sim=0.87   (0.0 because type mismatch)
              │              │          │
              └──────────────┴──────────┘
                              │
                         avg = 0.60
                    OverallOverlap = 0.60
```

### Phase 2 — Logic Drift Detection

For each layer in `dna1`, search **all** positions in `dna2` for the best cosine match — not just the same position:

```
dna1 L0 (Dense, sim vector A)
    │
    ├──► compare vs dna2 L0 → sim=0.72
    ├──► compare vs dna2 L1 → sim=0.31
    └──► compare vs dna2 L2 → sim=0.91  ← best match!

Best match (0.91) is at position L2, not L0.
Since 0.91 > 0.8 threshold AND positions differ:
→ LogicShift { SourcePos:"0,0,0,0", TargetPos:"0,0,0,2", Overlap:0.91 }
```

```go
type LogicShift struct {
    SourcePos string   // "z,y,x,l" in dna1
    TargetPos string   // "z,y,x,l" in dna2
    Overlap   float32  // cosine score > 0.8
}
```

Logic shifts appear when:
- A network was restructured and layers were reordered
- A NEAT mutation moved a functional pattern to a different grid position
- Two networks converged to the same function at different coordinates

---

## Full DNA Pipeline

```
  Network A (trained)                   Network B (trained)
       │                                      │
       ▼                                      ▼
 ExtractDNA(A)                          ExtractDNA(B)
       │                                      │
  for each layer:                        for each layer:
  ┌──────────────────────────────┐       ┌──────────────────────────────┐
  │ Parallel/Sequential:         │       │ Parallel/Sequential:         │
  │   recurse into branches      │       │   recurse into branches      │
  │   concat + Normalize         │       │   concat + Normalize         │
  │ Weighted:                    │       │ Weighted:                    │
  │   scale(w × ws.Scale)        │       │   scale(w × ws.Scale)        │
  │   Normalize(simulated)       │       │   Normalize(simulated)       │
  │ Weightless:                  │       │ Weightless:                  │
  │   {1.0}                      │       │   {1.0}                      │
  └──────────────────────────────┘       └──────────────────────────────┘
       │                                      │
       │  NetworkDNA ([]LayerSignature)        │  NetworkDNA
       └─────────────────┬────────────────────┘
                         │
                         ▼
               CompareNetworks(dnaA, dnaB)
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
    Phase 1: Direct            Phase 2: Cross-pos
    position matching          best-match search
            │                         │
    LayerOverlaps              LogicShifts
    OverallOverlap             (migrations)
            │
            └────────────────────────▶ NetworkComparisonResult
```

---

## Use Cases

### Measuring Quantization Fidelity

```go
dnaFP32 := poly.ExtractDNA(net)
// morph all layers to INT8...
poly.MorphAllLayers(net, poly.DTypeInt8)
dnaINT8 := poly.ExtractDNA(net)
result := poly.CompareNetworks(dnaFP32, dnaINT8)
// result.OverallOverlap near 1.0 → quantization preserved behavior
// result.OverallOverlap near 0.0 → quantization destroyed the model
```

### Detecting Training Convergence

Sample DNA every N epochs. When `OverallOverlap` between consecutive snapshots stabilizes above 0.99, the network has converged.

```
Epoch 0   → Epoch 10  : overlap = 0.12  (learning fast)
Epoch 10  → Epoch 50  : overlap = 0.61  (settling)
Epoch 50  → Epoch 100 : overlap = 0.94  (nearly converged)
Epoch 100 → Epoch 150 : overlap = 0.99  (converged)
```

### Cross-Architecture Similarity

Two networks with different layer counts share coordinates for only the positions they have in common. `CompareNetworks` will match only those overlapping positions, and the `OverallOverlap` is averaged over matched layers only.

### Logic Drift After NEAT Mutations

After a NEAT topology mutation moves a Dense layer from position `0,0,0,0` to `0,0,0,2`, the logic shift detector will report:

```
LogicShift {
    SourcePos: "0,0,0,0",
    TargetPos: "0,0,0,2",
    Overlap:   0.93,
}
```

This is how you track functional identity across structural mutations.

---

## DNA Signature Sizes by Layer Type

| Layer Type | Signature Length | Notes |
|:-----------|:-----------------|:------|
| Dense (32) | 1024 | inputH × outputH |
| MHA (32, 4 heads) | 4224 | Q+K+V+O projections + biases |
| SwiGLU (32) | 6144 | gate + up + down × 3 projections |
| RMSNorm (32) | 32 | scale vector only |
| LayerNorm (32) | 64 | gamma + beta |
| CNN1/2 (8f, 1c, k3) | 72 | filters × channels × k² |
| CNN3 (8f, 1c, k3) | 216 | filters × channels × k³ |
| RNN (32) | 2080 | Wx + Wh + bias |
| LSTM (32) | 8320 | 4 gates × (Wx + Wh + bias) |
| Embedding (256 vocab, 32 dim) | 8192 | vocab × dim |
| KMeans (8 clusters, 32 dim) | 256 | clusters × dim |
| Softmax | 1 | neutral marker |
| Residual | 1 | neutral marker |
| Parallel (2× Dense 32) | 1056 | concat of branches, renormalized |
| Sequential (2× Dense 32) | 2048 | concat of sub-layers, renormalized |
