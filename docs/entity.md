# ENTITY format (`.entity`)

**E**very **N**umerical **T**ype **I**n **N**ative **T**opolog**Y**

Native Loom checkpoint files. One `.entity` file = one saved brain: **full volumetric topology + all native-packed weights** in a single binary artifact.

Implementation: [`poly/entity.go`](../poly/entity.go)

Validated in Lucy menu **[7] Seven-layer CPU suite** — JSON and `.entity` save/reload run side by side for all 21 dtypes (`lucy/examples/seven_layer/runner.go`).

---

## Why we built this

HuggingFace **`.safetensors`** is the right **import** lane for PyTorch/HF checkpoints. Loom uses it for model download and HF decode (`poly/safetensors.go`, SoulGlitch, `LoomCreateLLM`).

It is **not** a native Loom checkpoint format. That is not a bug in SafeTensors — Loom simply does more than flat named tensors:

| Loom needs | SafeTensors |
|:-----------|:------------|
| **21 DTypes** with native on-disk packing (Int4 nibbles, Binary 8:1, Ternary, FP4, …) | Fixed HF dtype strings; sub-byte types are awkward; Loom **export** is F32-only |
| **Per-layer `Scale`** (quant mapping used at save time) | No standard field |
| **Volumetric grid** `(Z, Y, X, L)` per layer | Flat string keys only (`model.layers.0…`) |
| **Topology** — parallel branches, sequential stacks, metacognition | Requires separate `config.json`; no recursion |
| **Bit-perfect reload** of trained native dtypes | Import path usually decodes to FP32 master |

We already had full fidelity in **`persistence.go`** (`SerializeNetwork` / `DeserializeNetwork`) — JSON + Base64 native blobs. That works (Lucy save/reload PASS on all 21 dtypes) and remains the **transparent debug lane**, but it is **large and slow** for shipping brains to phones or edge nodes.

**ENTITY** is the native binary path:

- Same semantics as JSON persistence (same topology spec, same packing rules)
- SafeTensors-*like* wire safety: length-prefixed header + indexed blob section
- Raw weight bytes (no Base64)
- **Different file extension** so HF tooling does not assume HuggingFace semantics

```
Import:   model.safetensors     ← HF yarn (read-only in product flow)
Native:   fluffy.entity         ← ENTITY (train, save, reload, ship)
Debug:    model.json            ← JSON persistence (same brain, verbose)
```

---

## One file = topology + weights

Unlike HuggingFace’s split of `config.json` + `model.safetensors`, ENTITY keeps everything together:

```
┌─────────────────────────────────────────┐
│  Fixed header (magic, version, flags)   │
├─────────────────────────────────────────┤
│  JSON header                            │
│    • network topology (grid + layers)   │  ← PersistenceNetworkSpec
│    • blob index (path, offset, dtype…)  │
├─────────────────────────────────────────┤
│  Binary payload                         │  ← native-packed weight blobs
└─────────────────────────────────────────┘
```

After `LoadEntity`, you get a full `VolumetricNetwork` — grid dimensions, every layer’s type/activation/dtype/config, recursive branches, and quantized weights ready for forward or further training.

---

## Name and identity

| | |
|---|---|
| **Format** | ENTITY |
| **Expansion** | **E**very **N**umerical **T**ype **I**n **N**ative **T**opolog**Y** |
| **Extension** | `.entity` |
| **Magic** | `ENTITY\0\0` (8 bytes) |
| **Format version** | `1` (v1, implemented) |

---

## Wire layout (v1)

```
Offset  Size  Content
──────  ────  ───────
0       8     magic "ENTITY\0\0"
8       2     u16 format_version  (= 1)
10      2     u16 flags           (reserved; 0 today)
12      8     u64 header_byte_length (LE)
20      N     header JSON (see below)
20+N    …     native-packed weight blobs (contiguous)
```

### Header JSON

The header is one JSON object:

```json
{
  "format_version": 1,
  "network": { /* PersistenceNetworkSpec — topology only, no weight strings */ },
  "blobs": [
    {
      "path": "layers.0",
      "offset": 0,
      "length": 1234,
      "dtype": "INT4",
      "scale": 0.01,
      "native": true
    }
  ]
}
```

| Field | Role |
|:------|:-----|
| `network` | Same shape as [`PersistenceNetworkSpec`](../poly/persistence.go): `depth`, `rows`, `cols`, `layers_per_cell`, and every `PersistenceLayerSpec` (type, activation, dtype, z/y/x/l, MHA/CNN dims, parallel/sequential recursion). **No** `weights` Base64 strings — those live in the blob section. |
| `blobs[]` | Index into the payload. Each entry points at one weight store (main layer or nested branch). |

**Blob paths** mirror the in-memory tree:

| Path example | Weight store |
|:-------------|:---------------|
| `layers.0` | Top-level layer index 0 |
| `layers.3.sequential_layers.1` | Nested sequential sub-layer |
| `layers.2.parallel_branches.0` | Parallel branch |
| `layers.5.meta_observed_layer` | Metacognition observed layer |

Each blob carries its own **`dtype`**, **`scale`**, and **`native`** flag — so a single checkpoint can hold **different numerical types per layer** (e.g. layer 0 Int4, layer 12 BFloat16, layer 40 Binary).

### Weight blobs

Payload bytes use the **same bit-packing** as JSON persistence:

- Implemented via `EncodeNativeWeightsRaw` / `DecodeNativeWeightsRaw` in [`persistence.go`](../poly/persistence.go)
- Documented in [serialization.md](serialization.md#the-bit-packing-system)

No Base64. No FP32-only export constraint (unlike `SaveSafetensors`).

---

## API (`poly/entity.go`)

| Function | Purpose |
|:---------|:--------|
| `SerializeEntity(net)` | Network → `.entity` bytes |
| `DeserializeEntity(data)` | Bytes → full network (topology + all weights) |
| `DeserializeEntityWithOptions(data, opts)` | Selective weight load (`EntityLoadOptions.LayerIndices`) |
| `DeserializeEntityLayer(data, layerIndex)` | Topology + one top-level layer’s weights |
| `SaveEntity(path, net)` / `LoadEntity(path)` | File I/O |
| `ParseEntityHeader(data)` | Header only (no weight decode; mmap-friendly planning) |
| `LayerPersistenceFromEntity(data, layerIndex)` | Raw blob + scale + native for one layer (parity checks) |
| `EntityBlobBytes(data, blobIndex)` | Raw bytes for blob `i` without dtype decode |

**Hub model:** load any format → `VolumetricNetwork` → save as JSON, `.entity`, or (lossy) safetensors F32 export.

```go
net, err := poly.LoadEntity("brain.entity")
// … inference or re-quantize per layer (Morph) …
poly.SaveEntity("brain-v2.entity", net)
jsonWire, _ := poly.SerializeNetwork(net) // still valid debug export
```

---

## Size vs JSON — observed compression (Lucy [7])

Source: [`lucy/lucy_testing_output/seven_layer.txt`](../lucy/lucy_testing_output/seven_layer.txt) — full **[7] Seven-layer CPU suite** run (10 layer types × up to three grids × 21 dtypes). Checkpoints are written after MC training as `tag_DType.json` and `tag_DType.entity`.

### Headline numbers

| Metric | Result |
|:-------|:-------|
| **Runs compared** | 546 dtype×suite rows (26 memory tables) |
| **Save/reload** | `json=PASS entity=PASS` on all 546 trained reload checks; `entity=FAIL`: 0 |
| **Average disk saving** | `.entity` is **~27.6% smaller** than JSON (typical band **25–28%**) |
| **Runtime heap** | Unchanged — savings are **on-disk only** (same trained-native weight RAM in the log’s Weights column) |

ENTITY v1 removes **Base64 weight strings** and **pretty-printed JSON weights**. It does **not** re-quantize. The header is still **full topology JSON**, so this is not safetensors-class compression.

### Sample checkpoints (trained, after MC train)

**SwiGLU 2×2×2** (8 cells × 7 layers = **56-layer stack**) — the grid from the first live ENTITY comparison:

| DType | JSON ckpt | `.entity` ckpt | Saving |
|:------|----------:|---------------:|-------:|
| Float64 | 496.79 KiB | 372.49 KiB | 25% |
| Float32 | 258.17 KiB | 193.35 KiB | 25% |
| Int4 | 49.92 KiB | 36.90 KiB | 26% |
| Binary | 27.66 KiB | 20.29 KiB | 27% |

**Dense 1×1×1** (7-layer pyramid stack):

| DType | JSON ckpt | `.entity` ckpt | Saving |
|:------|----------:|---------------:|-------:|
| Float32 | 57.95 KiB | 43.47 KiB | 25% |
| Int4 | 9.53 KiB | 7.12 KiB | 25% |
| Binary | 4.35 KiB | 3.24 KiB | 26% |

**Dense 3×3×3** (27 cells × 7 = **189-layer stack**):

| DType | JSON ckpt | `.entity` ckpt | Saving |
|:------|----------:|---------------:|-------:|
| Float32 | 83.90 KiB | 60.95 KiB | 27% |
| Int4 | 69.93 KiB | 49.68 KiB | 29% |
| Binary | 68.61 KiB | 49.09 KiB | 28% |

Across **all 21 dtypes** on SwiGLU 2×2×2, the ENTITY/JSON ratio stays in a **25.0–26.6%** band — the saving is almost entirely **Base64 removal**, not dtype-specific magic.

### Three things the log teaches

**1. ENTITY vs JSON ≈ fixed ~25% discount, not 10×**

The ratio is stable because both formats carry the same topology JSON header and the same native weight bits; only the weight *encoding in the file* changes (Base64 strings → raw blob section).

**2. Quant dtype still dominates absolute file size**

Same topology, different dtype — SwiGLU 2×2×2:

| DType | JSON | `.entity` |
|:------|-----:|----------:|
| Float64 | 497 KiB | 372 KiB |
| Int4 | 50 KiB | 37 KiB |

Int4 JSON is ~**10%** of Float64 JSON on the same brain. Picking Int4/Binary matters far more than picking `.entity` over `.json`.

**3. Topology overhead grows with grid size; ENTITY shrinks the gap**

SwiGLU 2×2×2 **Float32** breakdown (from the log’s Weights vs checkpoint columns):

| Component | Size |
|:----------|-----:|
| Trained-native weights in RAM | 178.50 KiB |
| JSON checkpoint | 258.17 KiB (+**80** KiB overhead ≈ **31%** of file) |
| `.entity` checkpoint | 193.35 KiB (+**15** KiB overhead ≈ **8%** of file) |

On **Dense 3×3×3 Float32**, trained-native weights are only **~12 KiB** in RAM but the JSON checkpoint is **~84 KiB** — topology metadata dominates. ENTITY drops that to **~61 KiB** (~27% saving), but the file is still mostly header, not weights.

**Residual 3×3×3** is an extreme case in the log: **~42%** smaller `.entity` vs JSON when per-layer weight blobs are tiny relative to the 189-layer spec.

### Where to read it in the log

Each layer-type × grid block ends with a **memory & weight footprint** table:

```text
| DType | Heap | Sys | Heap+train | Weights | JSON ckpt | .entity ckpt |
```

Pass lines also print both sizes inline, e.g. `json=496.79 KiB entity=372.49 KiB` on SwiGLU 2×2×2 Float64.

### Why not safetensors-small?

1. **Weights** are already at the bit-width floor for each dtype (Int4 nibbles, Binary 8:1, …).
2. **Topology JSON** is a large fixed cost on big grids (56–189 layer specs with type, dtype, dims, z/y/x/l).
3. **SafeTensors** omits that graph entirely — flat tensor names only.

See [Future: smaller files with full topology](#future-smaller-files-with-full-topology) for the planned binary topology + optional zstd path.

---

## Comparison to SafeTensors

| | SafeTensors | ENTITY v1 | JSON persistence |
|--|-------------|-----------|------------------|
| Weights on disk | Raw HF dtypes | Raw Loom native packing | Base64 in JSON |
| Topology in same file | ❌ | ✅ (JSON header) | ✅ |
| Per-layer Loom dtype + Scale | ❌ | ✅ | ✅ |
| Volumetric (Z,Y,X,L) | ❌ | ✅ | ✅ |
| Parallel / sequential tree | ❌ | ✅ | ✅ |
| Typical header size | Tiny | Large on big grids | Largest (includes Base64 weights) |

SafeTensors wins on **flat LLM weight dumps**. ENTITY wins on **native Loom brains** you trained and need to reload exactly.

---

## Idempotency

`SerializeEntity` → `DeserializeEntity` → `SerializeEntity` yields **identical bytes** for a given network state (tested in `poly/tests/entity_test.go`).

Topology fields are canonicalized on save (e.g. default `seq_length` omitted for non-sequence layers) so reload does not inflate the header.

---

## Validation

| Suite | What it checks |
|:------|:---------------|
| Lucy **[7]** (`seven_layer/runner.go`) | Before/after train: JSON **and** `.entity` save/reload PASS; memory table shows both checkpoint sizes |
| `poly/tests/entity_test.go` | Round-trip, idempotent bytes, selective layer load, entity smaller than JSON |

Checkpoints land in `lucy/lucy_testing_output/` as `tag_DType.json` and `tag_DType.entity`. Full-run numbers and compression observations: [entity.md — observed compression](entity.md#size-vs-json--observed-compression-lucy-7) (from `seven_layer.txt`).

---

## Relationship to other I/O

| File | Role |
|:-----|:-----|
| [`safetensors.go`](../poly/safetensors.go) | Read HF `.safetensors`; `SaveSafetensors` is F32-only export |
| [`persistence.go`](../poly/persistence.go) | JSON save/load — semantic reference for ENTITY topology and packing |
| [`entity.go`](../poly/entity.go) | Native `.entity` binary save/load |
| [`serialization.go`](../poly/serialization.go) | Architecture-only JSON (`BuildNetworkFromJSON`) — random init, no trained weights |
| [`universal_loader.go`](../poly/universal_loader.go) | Auto-detect from safetensors shapes — import only |

---

## Future: smaller files with full topology

ENTITY v1 prioritizes **correctness and debuggability**. Planned v2+ improvements (same full topology, smaller wire):

1. **Binary topology section** — string tables, dtype/layer-type enums, grid-implied `(z,y,x,l)` where regular
2. **Compact blob index** — fixed records (`node_id`, `u8 dtype`, `f32 scale`, offsets) instead of JSON path strings
3. **Optional zstd** (lossless) on header/index/payload via `flags` bits
4. **`ConvertSafetensorsToEntity`** — import HF weights into a Loom topology wrapper
5. **Welvet C-ABI** — `LoomSaveEntity`, `LoomLoadEntity`; loaders accept `.entity` or `.safetensors`
6. **SoulGlitch** — prefer `.entity` for on-device trained saves; keep HF download as `.safetensors`

Weights stay on `EncodeNativeWeightsRaw`; the big disk wins are in **topology + index**, not re-quantizing weights.

---

## See also

- [serialization.md](serialization.md) — JSON persistence, bit-packing, SafeTensors import, three save paths
- [numerical_types.md](numerical_types.md) — 21 DTypes
- [quantization.md](quantization.md) — Scale, Morph, native packing
- [testing_and_validation.md](testing_and_validation.md) — Lucy [7] logs and tables
- [bedrock_validation.md](bedrock_validation.md) — seven-layer CPU suite overview
