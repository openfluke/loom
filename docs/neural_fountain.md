# Neural Fountain — shard specialists · LT peel · Master ensemble

**Neural Fountain** is a training / assembly paradigm in `poly/` that mirrors LT fountain codes on **network weights** instead of data bytes.

Pixel fountain recovers **K source blocks** byte-exact via spray / peel.  
Neural Fountain recovers **K specialist weight blobs** byte-exact the same way, then exposes a **Master** ensemble.

| Pixel fountain | Neural Fountain |
|:---------------|:----------------|
| K image / data blocks | K specialist **weight** blocks |
| XOR spray + peel | same LT codec (`poly/fountain_lt.go`) |
| 100% byte-exact cargo | 100% byte-exact specialists |
| reconstructed dataset | **Master** = recovered experts + averaged outputs |

Learning happens in `poly.Train` on each shard. Fountain is **transport / reassembly**, not the optimizer. No `layer_seed` search — contrast [seed_manifests.md](seed_manifests.md).

Companion MNIST demo: `chaosglue/loom_neural_fountain`. Related (data-only LT): `chaosglue/loom_fountain_codes`.

---

## Pipeline

1. **Partition** training batches into **K** shards (every sample covered).
2. **Specialize** — `NetworkFactory(i)` builds any architecture; `Train` on shard `i` (optional `UseExactDType` / `UniformDType`).
3. **Pack** — recursive FP32 Master (+ aux + Scale) over the **full layer tree** → `[]byte`.
4. **Fountain** — LT XOR spray (lossy OK) → peel until **K/K** recovered.
5. **Unpack** → Masters restored → **`ForceMorph(layer.DType)`** for each layer’s numerical type → Master ensemble.

Specialists must share an **identical parameter layout** (same walk order / lengths). Architecture is free via `NetworkFactory` (dense, CNN, residual, MHA, parallel/sequential nests, …).

---

## Any layer type / any numerical type

### Layers
`PackNetworkWeights` / `UnpackNetworkWeights` recursively visit:

- top-level `Layers`
- `ParallelBranches`, `SequentialLayers`
- `FilterGateConfig`, `MetaObservedLayer`

and pack, per layer:

- `WeightStore.Master` (FP32 persistence space — Loom’s SoT)
- `WeightStore.Scale`
- `QNormWeight`, `KNormWeight`, `InnerNormWeight` (MHA / BitNet-style aux)

Any layer that stores trainable state there participates. Nested stacks included.

### Numerical types
Loom persists through **FP32 Masters**, then morphs to the layer’s `DType` (any of the supported storage types):

- After unpack: `MorphNetworkToLayerDTypes` → `ForceMorph(layer.DType)`
- `ApplyUniformDType(net, dtype)` forces one dtype on the whole tree before train
- `UseExactDType: true` enables native-dtype train/forward paths during specialize

So fountain cargo is bit-exact FP32 Masters; **runtime dtype** is whatever each layer is set to (float16, int8, …), including exact-dtype training.

Activations / `TrainingBatch` remain float32 (standard Loom train API); weights can be exact-dtype.

---

## Core API

| File | Role |
|:-----|:-----|
| `poly/neural_fountain.go` | `NeuralFountain`, `FountainMaster`, `DenseSpecialistFactory`, config |
| `poly/weight_pack.go` | recursive pack/unpack, morph/wire helpers |
| `poly/fountain_lt.go` | LT encoder / decoder |

```go
// Dense example with mixed layer dtypes:
factory := poly.DenseSpecialistFactory("net",
    []int{784, 128, 64, 10},
    []string{"float16", "float16", "float32"})

// Or any architecture:
factory := func(i int) (*poly.VolumetricNetwork, error) {
    return buildMyCNNOrMHA(i) // identical layout across i
}

cfg := poly.DefaultNeuralFountainConfig()
cfg.K = 16
cfg.UseExactDType = true
// cfg.UniformDType = poly.DTypeFloat16  // optional blanket morph

master, err := poly.NeuralFountain(factory, batches, cfg)
out, err := master.Forward(input)
```

### Config knobs

| Field | Meaning |
|:------|:--------|
| `K` | specialist / shard count |
| `Epochs` / `LR` / `LossType` / `Mode` | per-specialist `Train` |
| `UseExactDType` | native-dtype train/forward |
| `UniformDType` | if set, morph all layers to this dtype before train |
| `LossRate` / `MaxOverhead` / `Seed` | LT channel + spray budget |
| `Verbose` | specialize / recover logs |

---

## Master semantics

- **`Forward` / `ForwardArgmax`** — average specialist outputs (deployable).
- **`OracleForward` / `OracleArgmax`** — shard expert owns sample `i` (coverage check).
- Prefer ensemble Master; do not average differently trained weight blobs into one net.

---

## Honest scope

- Fountain does not invent weights without specialist `Train`.
- Pack path is Loom’s FP32 Master SoT (+ aux); natives are rematerialized via `ForceMorph`.
- Layout equality across specialists is required (same recursive float count).
- LT recover is probabilistic; large blobs need enough spray overhead.
