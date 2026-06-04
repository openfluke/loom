# Serialization, Persistence, and Loading

This document covers how `VolumetricNetwork` instances are saved and loaded, the bit-packed persistence format for low-bit types, the idempotency guarantee, and SafeTensors support.

---

## Two Serialization Paths

`poly/` provides two complementary serialization systems:

| File | Functions | Use case |
|:-----|:---------|:---------|
| `serialization.go` | `BuildNetworkFromJSON` | Architecture-only: creates a network from a spec with randomly initialized weights |
| `persistence.go` | `SerializeNetwork` / `DeserializeNetwork` | Full save/load: architecture + trained weights |

---

## Full Save/Load (persistence.go)

### Saving

```go
jsonData, err := poly.SerializeNetwork(network)
os.WriteFile("model.json", jsonData, 0644)
```

`SerializeNetwork` walks every layer and builds a `PersistenceNetworkSpec`:

```go
type PersistenceNetworkSpec struct {
    ID            string                   `json:"id"`
    Depth         int                      `json:"depth"`
    Rows          int                      `json:"rows"`
    Cols          int                      `json:"cols"`
    LayersPerCell int                      `json:"layers_per_cell"`
    Layers        []PersistenceLayerSpec   `json:"layers"`
}
```

Each `PersistenceLayerSpec` contains all configuration fields plus:

```go
DType   string   `json:"dtype"`              // active numerical type for this layer (e.g. "Uint8", "FP4")
Weights string   `json:"weights,omitempty"`  // Base64-encoded **native-packed** payload for that dtype
Native  bool     `json:"native,omitempty"`   // true = weights are native-packed (current default on save)
Scale   float32  `json:"scale,omitempty"`    // morph/quant scale used when the checkpoint was written
```

### Native JSON per dtype (not FP32-only)

`SerializeNetwork` no longer dumps a single FP32 master blob for every layer. On save it:

1. Reads each layer’s live `DType` and writes it to `PersistenceLayerSpec.DType`.
2. Calls `WeightStore.Morph(dt)` for that dtype and `encodeNativeWeights(active, dt)` — Int8 as 1 byte/weight, FP4/Int4 as nibbles, Binary as bit-packs, Float64 as LE uint64, etc.
3. Sets `Native: true` and persists `Scale` so reload uses the same quant mapping training saw.

**Implication:** a **Uint8** Dense checkpoint is ~**0.8 KB** on disk for the Lucy 8×1024→512 bench; **Float64** is ~**5.4 MB** for the same topology — see the **File** column in Lucy’s training matrix (`lucy/lucy_testing_output/log.txt`). You can train, save, and reload **each of the 21 dtypes** independently; Lucy’s Dense suite reports **Save/Reload PASS** on all of them in the latest full run.

Older checkpoints with `Native: false` (FP32 master only) still load via `decodeWeights`; new saves prefer native packing.

### Loading

```go
jsonData, _ := os.ReadFile("model.json")
network, err := poly.DeserializeNetwork(jsonData)
```

`DeserializeNetwork` reconstructs the `VolumetricNetwork`, initializes fresh `WeightStore`s, then calls `applyPersistenceLayerSpec` for each layer which:

1. Parses all config fields
2. Calls `initializeWeights(l)` to allocate the correct `WeightStore` size
3. Decodes the `Weights` string — using `decodeNativeWeights` if `Native=true`, or `decodeWeights` (FP32 master) if `Native=false`
4. If native format (`Native=true`): stores in `Versions[dtype]`, then calls `Unpack(dtype)` to reconstruct the FP32 master for training paths that still use master weights
5. Recursively applies the same process to `ParallelBranches` and `SequentialLayers`

---

## The Bit-Packing System

The core serialization innovation is `encodeNativeWeights(data any, dt DType) string`.

This function takes the `active` version from the `WeightStore.Versions` map and packs it into the most compact binary representation before Base64 encoding:

```
DType          Packing                    Ratio vs FP32
──────────────────────────────────────────────────────
Float64        8 bytes/weight (LE uint64)   0.5x size reduction
Float32        4 bytes/weight (LE uint32)   1x (baseline)
Float16        4 bytes (stored as float32)  not yet compact
BFloat16       4 bytes (stored as float32)  not yet compact
Int8/Uint8     1 byte/weight                4x reduction
Int4/FP4/Uint4 0.5 bytes (2 per byte)      8x reduction
Int2/Uint2     0.25 bytes (4 per byte)     16x reduction
Ternary        0.25 bytes (4 per byte)     16x reduction
Binary         0.125 bytes (8 per byte)    32x reduction
```

### 4-bit Packing Detail

```go
// Pack 2 int8 weights into 1 byte using upper and lower nibbles:
buf[i/2] |= (byte(v & 0x0F) << 4)  // high nibble for even index
buf[i/2] |= (byte(v & 0x0F))       // low nibble for odd index
```

Unpacking sign-extends the nibble: if the 4-bit value is > 7, subtract 16 to recover the signed value.

### 2-bit/Ternary Packing Detail

```go
// Pack 4 values into 1 byte using 2-bit fields:
shift := uint(6 - (i%4)*2)   // 6, 4, 2, 0
buf[i/4] |= (val & 0x03) << shift
```

Unpacking reverses the shift and sign-extends from 2-bit.

### Binary Packing Detail

```go
// Pack 8 weights into 1 byte, MSB first:
if v > 0 { buf[i/8] |= (1 << uint(7-(i%8))) }
```

Unpacking reads each bit and maps `1 → +1`, `0 → -1`.

---

## Idempotency Guarantee

The README states: "Serializing a reloaded model produces a byte-for-byte identical JSON to the original."

This holds because:

1. `DeserializeNetwork` calls `Unpack(dtype)` which reconstructs `Master` from the packed data
2. The next `SerializeNetwork` call reads `Master`, calls `Morph(dtype)` again (if needed), and re-packs
3. Since `Morph` is deterministic (same formula, same scale), and the `Master` was faithfully reconstructed by `Unpack`, the output bytes are identical

Verified across 378 permutations (18 layer types × 21 DTypes) with **0.000000% mathematical divergence**.

---

## Architecture-Only JSON (serialization.go)

`BuildNetworkFromJSON` creates a network from a spec but uses **random weight initialization** (via `initializeWeights` which calls `Randomize`). This is for defining network topologies without weights.

```go
type LayerSpec struct {
    Z, Y, X, L    int
    Type           string   // "Dense", "CNN2", etc.
    Activation     string   // "ReLU", "Tanh", etc.
    DType          string   // "float32", "int8", etc.
    InputHeight    int
    OutputHeight   int
    // ... all configuration fields
    ParallelBranches []LayerSpec   // recursive
    SequentialLayers []LayerSpec   // recursive
}
```

`ParseLayerType`, `ParseActivationType`, and `ParseDType` accept case-insensitive strings plus common aliases.

---

## SafeTensors support (import only)

`safetensors.go` and `prefix_safetensor.go` implement **loading** from the HuggingFace SafeTensors format. Export via `SaveSafetensors` is F32-only. For native Loom saves (all 21 dtypes, volumetric topology, native packing), use **ENTITY** (`.entity`) — see [entity.md](entity.md).

`universal_loader.go` provides auto-detection of the model format.

The `Transformer[T]` type has dedicated loading support in `transformer.go` for assembling a full LLM from SafeTensors files: it maps weight tensor names (e.g., `"model.layers.0.self_attn.q_proj.weight"`) to the correct `VolumetricLayer` positions and weight sub-slices.

---

## Compression Ratios in Practice

From the README, for a network with 1M weights:

```
┌──────────────────────────────────────────────────────────────┐
│  DType     RAM (uncompressed)   JSON size   Ratio            │
├──────────────────────────────────────────────────────────────┤
│  Float32   4.0 MB               ~5.5 MB     1.38x (base64)  │
│  Int8      1.0 MB               ~1.4 MB     0.34x vs FP32   │
│  Int4      0.5 MB               ~0.7 MB     0.17x           │
│  Binary    0.125 MB             ~0.18 MB    0.045x ← 98.4%  │
└──────────────────────────────────────────────────────────────┘
```

Base64 encoding adds ~33% overhead over the raw binary size. The 98.4% figure is relative to FP32 on disk (including the base64 overhead).

---

## Weight Encoding Flow

```
Training produces Master []float32
                │
                ▼ (if layer.DType != DTypeFloat32)
         Morph(layer.DType)
                │
                ▼
         Versions[dtype] = []int8 / []int4 / etc.
                │
                ▼
    encodeNativeWeights(active, dtype)
                │
         ┌──────┴──────┐
         │             │
         ▼             ▼
    bit-packing    Base64 encode
         │             │
         └──────┬──────┘
                │
                ▼
    PersistenceLayerSpec.Weights = "base64string..."
    PersistenceLayerSpec.Native  = true
    PersistenceLayerSpec.Scale   = ws.Scale
```

---

## Deserialization and Unpack Flow

```
JSON string
    │
    ▼ json.Unmarshal
PersistenceNetworkSpec
    │
    ▼ applyPersistenceLayerSpec
For each layer:
    1. ParseLayerType / ParseActivationType / ParseDType
    2. initializeWeights → fresh WeightStore allocated
    3. if ls.Native:
          decodeNativeWeights → Versions[dtype] = packed slices
          ws.Unpack(dtype) → Master reconstructed
       else:
          decodeWeights → Master loaded directly
    4. Recurse for ParallelBranches, SequentialLayers
```

After `DeserializeNetwork`, every layer's `WeightStore.Master` is a valid FP32 weight array ready for forward inference or further training.
