# The Evolution Engine: DNA Splice & NEAT Topology Evolution

This document covers `SpliceDNA`, `SpliceDNAWithReport`, `NEATMutate`, and `NEATPopulation` from `evolution.go`. The evolution engine builds on the DNA fingerprinting system described in [dna.md](dna.md).

---

## Two Evolutionary Mechanisms

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    Evolution Engine                         │
  │                                                             │
  │   ┌────────────────────┐    ┌──────────────────────────┐   │
  │   │   DNA Splice       │    │   NEAT-style Mutation    │   │
  │   │  (Crossover)       │    │  (Topology Evolution)    │   │
  │   │                    │    │                          │   │
  │   │  ParentA + ParentB │    │  Network ──► mutated     │   │
  │   │      ──►  Child    │    │             clone        │   │
  │   │                    │    │                          │   │
  │   │  merges weights    │    │  changes layer types,    │   │
  │   │  guided by DNA     │    │  activations, topology   │   │
  │   │  similarity        │    │  weights                 │   │
  │   └────────────────────┘    └──────────────────────────┘   │
  │              │                          │                   │
  │              └──────────┬───────────────┘                   │
  │                         ▼                                   │
  │              NEATPopulation.Evolve()                        │
  │         (combines both in a generation loop)                │
  └─────────────────────────────────────────────────────────────┘
```

---

## Part 1 — DNA Splice / Genetic Crossover

### Concept

Given two trained parent networks `A` and `B`, produce a child network whose weights are a blend of both. The blend is **guided by DNA similarity** — layers that are more similar between parents get blended more aggressively; layers that diverged get a heavier bias toward the fitter parent.

```
  ParentA (trained)        ParentB (trained)
       │                        │
  ExtractDNA(A)            ExtractDNA(B)
       │                        │
  sigA per layer          sigB per layer
       │                        │
       └────────┬───────────────┘
                │
       for each layer position (z,y,x,l):
                │
         CosineSimilarity(sigA, sigB)
                │
         ┌──────┴──────┐
         │             │
      blend         skip
    weights       (keep A's
    from A+B       weights)
         │
         ▼
     Child network
```

### SpliceConfig

```go
type SpliceConfig struct {
    CrossoverMode string   // "blend", "point", or "uniform"
    BlendAlpha    float32  // interpolation factor (blend mode): 0=all A, 1=all B
    SplitRatio    float64  // fraction from A in point mode (e.g. 0.5)
    FitnessA      float64  // optional: used to bias toward fitter parent
    FitnessB      float64
}

func DefaultSpliceConfig() SpliceConfig {
    return SpliceConfig{CrossoverMode: "blend", BlendAlpha: 0.5, SplitRatio: 0.5}
}
```

### Three Crossover Modes

#### Mode: "blend" (default)

Interpolates weights per element. Alpha is modulated by the layer's cosine similarity and relative fitness:

```
alpha = FitnessB / (FitnessA + FitnessB)   ← bias toward fitter parent
alpha = alpha × (0.5 + 0.5 × similarity)   ← scale by how similar layers are

child[i] = wA[i] × (1 - alpha) + wB[i] × alpha
```

When similarity is high (layers learned the same thing), alpha blends freely. When similarity is low (layers diverged), alpha is pulled toward the fitter parent.

```
similarity = 1.0  ──►  free blend (both parents contribute equally)
similarity = 0.0  ──►  take mostly from fitter parent (layers are unrelated)
similarity = -1.0 ──►  heavily bias toward fitter parent (opposite patterns)
```

#### Mode: "point"

Splits weights at a single cut point. First `SplitRatio` fraction from A, rest from B:

```
wA: [a0 a1 a2 a3 a4 a5 a6 a7]
wB: [b0 b1 b2 b3 b4 b5 b6 b7]
                │
           SplitRatio=0.5
                │
child: [a0 a1 a2 a3 b4 b5 b6 b7]
        ─── from A ──── from B ──
```

#### Mode: "uniform"

Each weight is randomly drawn from A or B, with probability biased toward the fitter parent:

```
threshold = FitnessA / (FitnessA + FitnessB)

for each weight i:
    if rand < threshold → child[i] = wA[i]
    else               → child[i] = wB[i]
```

### SpliceDNA

```go
func SpliceDNA(parentA, parentB *VolumetricNetwork, cfg SpliceConfig) *VolumetricNetwork
```

- The child is always a **deep clone of parentA** (architecture inherited from A)
- Only layers where both parents have matching positions **and matching weight dimensions** are blended
- If `parentB` has no layer at that position, or the weight counts differ, A's weights are kept unchanged

```go
// Guard: skip if dimensions don't match
if wB == nil || len(wB) != len(wA) {
    continue // keep A's weights
}
```

### SpliceDNAWithReport

```go
func SpliceDNAWithReport(parentA, parentB *VolumetricNetwork, cfg SpliceConfig) SpliceResult

type SpliceResult struct {
    Child        *VolumetricNetwork
    ParentADNA   NetworkDNA
    ParentBDNA   NetworkDNA
    ChildDNA     NetworkDNA
    Similarities map[string]float32  // "z,y,x,l" → cosine score used for blending
    BlendedCount int                  // how many layers were actually blended
}
```

Returns the same child as `SpliceDNA` plus a full diagnostic report. Use this when debugging crossover behavior or logging ancestry.

---

## Part 2 — NEAT-style Topology Evolution

### Concept

NEAT (NeuroEvolution of Augmenting Topologies) mutates both weights and structure. The implementation here applies six mutation types to a cloned network, leaving the original untouched.

```
  Original Network (immutable)
       │
  cloneNetwork()
       │
  mutated clone
       │
  ┌────┴────────────────────────────────────────────┐
  │  Per-layer mutations (applied sequentially):    │
  │                                                 │
  │  1. Weight perturbation  ── add Gaussian noise  │
  │  2. Activation mutation  ── swap act function   │
  │  3. Node mutation        ── change layer type   │
  │  4. Layer toggle         ── enable/disable      │
  │                                                 │
  │  Network-level mutations (applied once):        │
  │                                                 │
  │  5. Connection add  ── insert remote link       │
  │  6. Connection drop ── remove remote link       │
  └─────────────────────────────────────────────────┘
       │
  returns mutated clone
```

### NEATConfig

```go
type NEATConfig struct {
    WeightPerturbRate  float64  // prob of perturbing a layer's weights (default 0.8)
    WeightPerturbScale float32  // noise magnitude (default 0.05)
    NodeMutateRate     float64  // prob of changing a layer's type (default 0.1)
    ConnectionAddRate  float64  // prob of adding a remote link (default 0.05)
    ConnectionDropRate float64  // prob of removing a remote link (default 0.02)
    ActivationMutRate  float64  // prob of changing activation function (default 0.1)
    LayerToggleRate    float64  // prob of toggling IsDisabled (default 0.02)
    DModel             int      // reference dimension for weight reinitialization
    AllowedLayerTypes  []LayerType // types a node can mutate to
    // Type-specific defaults used by neatReinitLayer:
    DefaultNumHeads    int
    DefaultInChannels  int
    DefaultFilters     int
    DefaultKernelSize  int
    DefaultVocabSize   int
    DefaultNumClusters int
    Seed               int64
}
```

`DefaultNEATConfig(dModel)` returns conservative rates with all 17 mutable layer types in `AllowedLayerTypes`.

### NEATMutate

```go
func NEATMutate(n *VolumetricNetwork, cfg NEATConfig) *VolumetricNetwork
```

The original network `n` is **never modified**. The function clones it and applies mutations:

```
For each layer i:

  Step 1 — Weight Perturbation (WeightPerturbRate = 0.8)
  ┌─────────────────────────────────────────────────────┐
  │ master[i] += rand(-1, 1) × WeightPerturbScale       │
  │ (clears cached DType versions as weights changed)   │
  └─────────────────────────────────────────────────────┘

  Step 2 — Activation Mutation (ActivationMutRate = 0.1)
  ┌──────────────────────────────────────────────────────┐
  │ layer.Activation = random from {ReLU, SiLU, GELU,   │
  │                                 Tanh, Sigmoid, Linear}│
  └──────────────────────────────────────────────────────┘

  Step 3 — Node Mutation (NodeMutateRate = 0.1)
  ┌──────────────────────────────────────────────────────┐
  │ newType = random from AllowedLayerTypes (≠ current)  │
  │ neatReinitLayer(child, i, newType, cfg)              │
  │   → sets new Type, InputHeight, OutputHeight         │
  │   → creates fresh WeightStore with correct wCount    │
  └──────────────────────────────────────────────────────┘

  Step 4 — Layer Toggle (LayerToggleRate = 0.02)
  ┌──────────────────────────────────────────────────────┐
  │ layer.IsDisabled = !layer.IsDisabled                 │
  │ (disabled layers are skipped during forward pass)    │
  └──────────────────────────────────────────────────────┘

After all layers:

  Step 5 — Connection Add (ConnectionAddRate = 0.05)
  ┌──────────────────────────────────────────────────────┐
  │ Pick two random layers src and dst (src ≠ dst)       │
  │ Append IsRemoteLink branch to src.ParallelBranches   │
  │   TargetZ/Y/X/L point to dst                        │
  │ Creates a spatial "skip connection" in the 3D grid   │
  └──────────────────────────────────────────────────────┘

  Step 6 — Connection Drop (ConnectionDropRate = 0.02)
  ┌──────────────────────────────────────────────────────┐
  │ Find a layer with ParallelBranches containing        │
  │ IsRemoteLink entries                                 │
  │ Remove one at random                                 │
  └──────────────────────────────────────────────────────┘
```

### Node Mutation: Weight Counts for All 19 Layer Types

When `neatReinitLayer` changes a layer's type, it creates a fresh `WeightStore` with the correct number of weights for the new type:

| New Layer Type | Formula | Example (dModel=32) |
|:---------------|:--------|:--------------------|
| Dense | `dModel × dModel` | 1024 |
| RNN | `dModel² + dModel² + dModel` | 2080 |
| LSTM | `4 × (dModel² + dModel² + dModel)` | 8320 |
| SwiGLU | `dModel × (dModel×2) × 3` | 6144 |
| RMSNorm | `dModel` | 32 |
| LayerNorm | `dModel × 2` | 64 |
| MHA | `2×dModel² + 2×dModel×kv + 2×dModel + 2×kv` | 4224 (4 heads) |
| CNN1 / CNN2 | `filters × inChannels × kSize²` | 72 (8f, 1c, k3) |
| CNN3 | `filters × inChannels × kSize³` | 216 (8f, 1c, k3) |
| ConvTransposed1D/2D | `inChannels × filters × kSize²` | 72 |
| ConvTransposed3D | `inChannels × filters × kSize³` | 216 |
| Embedding | `vocabSize × dModel` | 8192 (256 vocab) |
| KMeans | `numClusters × dModel` | 256 (8 clusters) |
| Softmax | `0` — no WeightStore | — |
| Residual | `0` — no WeightStore | — |
| Parallel / Sequential | unchanged — keep existing branches | — |

Parallel and Sequential are structural containers. Mutating a non-container to Parallel/Sequential would destroy branch structure, so `neatReinitLayer` leaves them untouched (just returns) when the target type is Parallel or Sequential.

### Connection Add — Remote Links

`neatAddConnection` adds a **spatial skip connection** between two layers anywhere in the 3D grid:

```
Layer at (0,0,0,0) ──────────────────────────► Layer at (0,0,0,2)
                                                      │
                    ┌─ ParallelBranches ──────────────┘
                    │   [IsRemoteLink=true,
                    │    TargetZ=0, TargetY=0,
                    │    TargetX=0, TargetL=2]
```

During `ForwardPolymorphic`, `ParallelForwardPolymorphic` follows remote links and routes activations to the target layer. Remote links are skipped during DNA extraction (`extractLayerSignature` skips `IsRemoteLink=true` branches since they have no local weights).

---

## Part 3 — NEATPopulation: Full Evolutionary Loop

`NEATPopulation` manages a pool of networks across generations using fitness-based selection.

```go
type NEATPopulation struct {
    Networks  []*VolumetricNetwork
    Fitnesses []float64
    Config    NEATConfig
    rng       *rand.Rand
}
```

### Initialization

```go
pop := poly.NewNEATPopulation(seedNetwork, populationSize, cfg)
```

Creates `populationSize` networks, each a `NEATMutate` of the seed. This gives diverse starting points from day 0.

```
seedNetwork
    │
    ├── NEATMutate (seed1) ──► Network[0]
    ├── NEATMutate (seed2) ──► Network[1]
    ├── NEATMutate (seed3) ──► Network[2]
    └── ...                    Network[N-1]
```

### One Generation of Evolution

```go
pop.Evolve(fitnessFn)
```

```
  Generation N:  [net0, net1, net2, ..., netN]
                      │
              fitnessFn(net) for each
                      │
              sort descending by fitness
                      │
          ┌───────────┴───────────┐
          │                       │
     Top 25%                 Bottom 75%
     (elites)                (replaced)
          │                       │
     carry over              pick 2 elites A, B
     unchanged               SpliceDNA(A, B, blend)
                                  │
                             NEATMutate(child)
                                  │
                             new offspring
          │                       │
          └───────────┬───────────┘
                      │
              Generation N+1
```

**Elites**: The top `populationSize / 4` networks survive unchanged. The rest are replaced by:
1. Pick two random elites `A` and `B`
2. Produce a child via `SpliceDNA(A, B, cfg)` — inherits weights from both
3. Apply `NEATMutate(child)` — adds structural noise

### Helper Methods

```go
pop.Best()           // returns the highest-fitness network (index 0 after sort)
pop.BestFitness()    // returns the best fitness score
pop.Summary(gen)     // returns a one-line status string:
                     // "Gen 5 | best=-0.0012  avg=-0.0045  worst=-0.2300  pop=16"
```

### Fitness Function Contract

The fitness function receives a network and returns `float64` — higher is better. Penalize with a large negative (e.g., `-1e9`) for architecturally incompatible networks (dimension mismatches from mutations):

```go
fitnessFn := func(net *poly.VolumetricNetwork) (result float64) {
    defer func() {
        if r := recover(); r != nil {
            result = -1e9 // incompatible architecture
        }
    }()
    out, _, _ := poly.ForwardPolymorphic[float32](net, input)
    if out == nil || len(out.Data) == 0 {
        return -1e9
    }
    // compute your task loss here
    mse := computeMSE(out.Data, target)
    return -mse   // negate: lower loss = higher fitness
}
```

---

## Combined Flow: SpliceDNA + NEAT in a Population

```
                 ┌──────────────────────────────────────────┐
                 │           NEATPopulation.Evolve           │
                 │                                          │
  Generation N:  │  [A] [B] [C] [D]  ... [P]               │
                 │   │                                      │
                 │   fitnessFn() for all                    │
                 │   sort: A=best, P=worst                  │
                 │                                          │
                 │  Elites (keep): [A] [B] [C] [D]         │
                 │                                          │
                 │  Offspring:                              │
                 │                                          │
                 │   SpliceDNA(A, B)  ──► child_AB          │
                 │   NEATMutate(child_AB)                   │
                 │        ├── perturb weights               │
                 │        ├── maybe swap activation         │
                 │        ├── maybe change layer type       │
                 │        └── maybe add/drop connection     │
                 │            ──► mutated_AB                │
                 │                                          │
                 │   ... repeat for all offspring slots ... │
                 │                                          │
  Generation N+1:│  [A] [B] [C] [D] [mut_AB] ... [mut_XY] │
                 └──────────────────────────────────────────┘
```

---

## DNA Tracking Across Generations

Because every `NEATMutate` and `SpliceDNA` call touches only a clone, you can always extract DNA from any network in the population and compare it against a reference:

```go
// Track how far the best network has drifted from the initial seed
seedDNA := poly.ExtractDNA(seedNetwork)
for gen := 1; gen <= 50; gen++ {
    pop.Evolve(fitnessFn)
    bestDNA := poly.ExtractDNA(pop.Best())
    result := poly.CompareNetworks(seedDNA, bestDNA)
    fmt.Printf("Gen %d | seed→best overlap=%.4f  logic_shifts=%d\n",
        gen, result.OverallOverlap, len(result.LogicShifts))
}
```

Expected pattern:
```
Gen  1 | overlap=0.98  logic_shifts=0   (small weight nudges)
Gen  5 | overlap=0.73  logic_shifts=1   (one node mutated type)
Gen 20 | overlap=0.41  logic_shifts=3   (topology diverging)
Gen 50 | overlap=0.12  logic_shifts=7   (heavily evolved)
```

---

## Multi-Parent Splice Chain

You can chain splices to merge three or more trained networks:

```go
cfgA := poly.DefaultSpliceConfig()
cfgA.FitnessA, cfgA.FitnessB = fitnessA, fitnessB

cfgB := poly.DefaultSpliceConfig()
cfgB.FitnessA, cfgB.FitnessB = fitnessMid, fitnessC

mid   := poly.SpliceDNA(netA, netB, cfgA)    // A + B → mid
final := poly.SpliceDNA(mid, netC, cfgB)     // mid + C → final
```

```
netA ──┐
        ├── SpliceDNA ──► mid ──┐
netB ──┘                        ├── SpliceDNA ──► final
                          netC ──┘
```

---

## Immutability Guarantee

Both `SpliceDNA` and `NEATMutate` always operate on **clones** of the input networks. The originals are never modified:

```go
// Verify: run 5 aggressive mutations, original unchanged
original := buildDenseMLP(32, 3)
dnaOrig  := poly.ExtractDNA(original)

aggressiveCfg := poly.NEATConfig{
    NodeMutateRate: 1.0, WeightPerturbRate: 1.0,
    WeightPerturbScale: 10.0, DModel: 32, Seed: 42,
    AllowedLayerTypes: poly.DefaultNEATConfig(32).AllowedLayerTypes,
}
for i := 0; i < 5; i++ {
    _ = poly.NEATMutate(original, aggressiveCfg)
}

dnaAfter := poly.ExtractDNA(original)
result   := poly.CompareNetworks(dnaOrig, dnaAfter)
// result.OverallOverlap == 1.0 — original untouched
```

---

## Quick Reference

| Function | What it does |
|:---------|:-------------|
| `SpliceDNA(A, B, cfg)` | Blend weights from A and B into a child (A's architecture) |
| `SpliceDNAWithReport(A, B, cfg)` | Same + diagnostic report with per-layer similarities |
| `DefaultSpliceConfig()` | Returns blend mode, alpha=0.5, split=0.5 |
| `NEATMutate(n, cfg)` | Returns a structurally mutated clone of n |
| `DefaultNEATConfig(dModel)` | Conservative rates, all 17 mutable types allowed |
| `NewNEATPopulation(seed, size, cfg)` | Create diverse initial population from seed |
| `pop.Evolve(fitnessFn)` | Run one generation: evaluate → sort → elites → offspring |
| `pop.Best()` | Highest-fitness network from last Evolve |
| `pop.BestFitness()` | Fitness score of the top network |
| `pop.Summary(gen)` | One-line status: best/avg/worst fitness |
