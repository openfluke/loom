# Seed manifests — topology + layer seeds (no weight blobs)

Loom can represent a neural network as **recipes**, not checkpoints:

| Piece | Role |
|:------|:-----|
| `topology_seed` | Hash of network **shape** (layer widths, name tag) |
| `layer_seed` (per layer) | Expands to **all weights** for that layer via He-init |
| `weight_fp` / `forward_fp` | Optional fingerprints for verify (not weights) |

Weights are **never** stored in a seed manifest. Reload calls `InitWeightStoreHeSeeded` (or `InitFloat32HeSeeded`) to regenerate the full weight matrix from each `layer_seed`.

This is complementary to [serialization.md](serialization.md) (JSON / `.entity` checkpoints with packed weight bytes) and [entity.md](entity.md) (HF → native ship lane).

---

## Core API (`poly/seed_core.go`, `poly/seed_init.go`)

```go
topo := poly.SeedFrom("my-net", []int{4, 8, 4, 2}...)
layerSeed := poly.DeriveLayerSeed(topo, layerIndex, "dense.0")

poly.InitFloat32HeSeeded(weights, inputSize, layerSeed)
poly.InitWeightStoreHeSeeded(ws, inputSize, layerSeed)
```

- `SeedFrom(parts...)` — deterministic uint64 mixer (golden-ratio / SplitMix64).
- `DeriveLayerSeed(initSeed, index, path)` — per-layer slot (path disambiguates parallel branches in entity manifests).
- `NewSeedRNG(seed)` — xorshift64* PRNG; He-init draws `NormFloat64() * sqrt(2/fan_in)`.

**Invariant:** same `layer_seed` + same `inputSize` → same weights (all 21 dtypes via `InitLayerWeightsSeeded`).

---

## Dense manifests (`poly/seed_dense.go`)

```go
topo := poly.DenseTopologySeed("tag", []int{4, 8, 4, 2})
m, _ := poly.BuildDenseManifest(topo, sizes, []string{"float32", "float32", "float32"})
net, _ := poly.BuildDenseVolumetricFromManifest(m)

// weights → seeds (only when weights still match He-init from layer_seed)
extracted, err := poly.ManifestFromDenseNetwork(net, topo, sizes, dtypes)
```

| Function | Direction |
|:---------|:----------|
| `BuildDenseManifest` | topology + dtypes → per-layer `layer_seed`, `weight_fp`, `forward_fp` |
| `BuildDenseVolumetricFromManifest` | manifest → `VolumetricNetwork` (He-init per layer) |
| `ManifestFromDenseNetwork` | built net → manifest (verifies each layer matches its seed) |
| `RebuildDenseManifest` | seeds-only rebuild + fingerprint check |
| `MarshalDenseManifest` / `ParseDenseManifest` | JSON round trip (~hundreds of bytes) |

`DenseLayerWeightSeed(topologySeed, i)` is `DeriveLayerSeed(topologySeed, i, "dense."+i)` — the default init recipe from topology alone.

---

## Other layer families

Same pattern under `poly/seed_*.go`:

| File | Layers |
|:-----|:-------|
| `seed_swiglu.go` | SwiGLU |
| `seed_mha.go` | Multi-head attention |
| `seed_rnn.go`, `seed_lstm.go` | RNN, LSTM |
| `seed_cnn.go` | CNN1/2/3 |
| `seed_embedding.go` | Embedding tables |
| `seed_residual.go` | Dense + skip (dense branch seed only) |
| `seed_entity.go` | Entity transformer topology + globals |
| `seed_manifest.go` | Tiny `.wseed` entity manifests (`loom-seed-manifest-v3`) |
| `seed_dtypes.go`, `seed_dtypes_layers.go` | 21-dtype matrix (210 layer×dtype round trips) |

Lucy **[19]** runs the full round-trip matrix: `loom/lucy_bloom_rivers/examples/seed_roundtrip/`.

---

## Weights ↔ seeds: what works and what does not

### On the seed manifold (works)

1. Build net from `layer_seed` → He-init weights.
2. Forward / train **only by changing `layer_seed`** (weights always re-derived from seed).
3. `ManifestFromDenseNetwork` recovers the same `layer_seed` values.
4. Save manifest JSON → reload → bit-exact outputs.

### Off the seed manifold (does not work with manifest extract)

`poly.Train` updates **weight tensors** in place. After SGD/MSE training, weights generally **no longer** equal `He-init(layer_seed)` for any topology-derived seed. `ManifestFromDenseNetwork` then returns:

```
dense: layer N weights do not match seed 0x…
```

That is expected: a seed manifest is **not** a trained checkpoint format unless training stayed on the seed manifold.

| Training style | Save trained state as seeds? |
|:-------------|:----------------------------|
| Optimize `layer_seed` (mutate seed, reinit weights each eval) | **Yes** — weights always from seed |
| `poly.Train` on weight tensors | **No** — use `.entity` / JSON persistence |

---

## Lucy [20] — seed proof (`chaosglue-seed-proof-v4`)

**Repo path:** `loom/lucy_bloom_rivers/examples/seed_proof/`  
**Menu:** Lucy **[20]**  
**Output:** `lucy_testing_output/proof.seeds`  
**Headless:** `LOOM_SEED_PROOF=1 go run .`

End-to-end demo: build → show init outputs → **train layer seeds** → save trained seeds → reload with **no training** → same trained outputs.

### First run (no `proof.seeds`)

1. `DenseTopologySeed` + `BuildDenseManifest` → init `layer_seed` per layer.
2. `BuildDenseVolumetricFromManifest` → He-init weights.
3. Print **before train** chained forwards + 10 final outputs.
4. `ManifestFromDenseNetwork` — proves init weights↔seeds.
5. **Train:** hill-climb each `layer_seed` (mutate → `InitWeightStoreHeSeeded` → MSE). Weights never updated outside He-init.
6. Print **after train** outputs (different from init).
7. Verify each layer’s weights match its **trained** `layer_seed`.
8. Save `proof.seeds` (trained seeds only + `trained_outputs` baseline for verify).
9. Reload check from file before exit.

### Rerun (`proof.seeds` exists)

1. Load JSON (topology + 3× trained `layer_seed`).
2. `BuildDenseVolumetricFromManifest` — weights generated in RAM from seeds.
3. Forward pass (chained + 10 outputs recomputed, **not** read from file).
4. Compare to `trained_outputs` in file — fail if seeds did not rebuild the net.

No `trainLayerSeeds`, no `poly.Train`, no weight file.

### `proof.seeds` format (v4)

```json
{
  "format": "chaosglue-seed-proof-v4",
  "topology_seed": 10459346120451217710,
  "sizes": [4, 8, 4, 2],
  "layers": [
    { "index": 0, "in": 4, "out": 8, "layer_seed": 16912650198654748781, "dtype": "float32" },
    { "index": 1, "in": 8, "out": 4, "layer_seed": 15008752656474397499, "dtype": "float32" },
    { "index": 2, "in": 4, "out": 2, "layer_seed": 15710426925220086453, "dtype": "float32" }
  ],
  "init_outputs": [ … ],
  "trained_outputs": [ … ]
}
```

- **`layers[].layer_seed`** — the trained model state (3× uint64). Reload expands to full weight matrices.
- **`trained_outputs`** — verification baseline only; rerun recomputes forwards and checks bit-exact match.
- **Not in file:** weight arrays, Base64 blobs, per-weight “seeds”.

Delete `lucy_testing_output/proof.seeds` to repeat the first-run train + save flow.

### Thin wrapper

`chaosglue/seed_proof/` delegates to the same package (`LOOM_SEED_PROOF=1` from `loom/lucy_bloom_rivers`).

---

## Related Lucy menus

| Menu | Suite | Doc |
|:-----|:------|:----|
| **[18]** | Seed topology POC (shape → recipe seeds) | — |
| **[19]** | Seed round trip (dense + all layer families, 21 dtypes) | this doc |
| **[20]** | Seed proof (train layer seeds, save, reload) | this doc |

---

## When to use seeds vs checkpoints

| Goal | Use |
|:-----|:----|
| Tiny init recipe from topology | Seed manifest (`.wseed`, `proof.seeds`, dense manifest JSON) |
| Ship trained model to disk | [entity.md](entity.md) `.entity` or [serialization.md](serialization.md) JSON |
| HF import | SafeTensors → `.entity` ([entity.md](entity.md)) |
| Prove seeds↔weights without weight blobs | Lucy **[20]** |

---

## Package map

```
poly/
├── seed_core.go          SeedFrom, DeriveLayerSeed, NewSeedRNG, InitFloat32HeSeeded
├── seed_init.go          InitLayerWeightsSeeded, InitSeededNetwork, fingerprints
├── seed_dense.go         Dense manifests, BuildDenseVolumetricFromManifest
├── seed_*.go             Per-layer-family manifests
├── seed_manifest.go      Entity weight-seed files (loom-seed-manifest-v3)
└── seed_dtypes*.go       21-dtype verification matrix

lucy_bloom_rivers/examples/
├── seed_poc/             Menu [18]
├── seed_roundtrip/       Menu [19]
└── seed_proof/           Menu [20]
```
