# Quantization: DType Conversion and PTQ Pipeline

This document covers the Post-Training Quantization (PTQ) pipeline in `poly/`: how weights move from FP32 masters into lower-precision formats, the `WeightStore` versioning system, the `Q4_0Block` block-quantization format, and how `SimulatePrecision` approximates low-bit arithmetic on CPU.

---

## Why Quantization?

Running a 7B-parameter model at FP32 requires ~28 GB of RAM. Quantization trades a small amount of numerical fidelity for dramatic memory and compute savings:

```
┌──────────────────────────────────────────────────────────────────┐
│  DType       Bits/weight   1B params   Theoretical speedup       │
├──────────────────────────────────────────────────────────────────┤
│  Float64     64            8 GB        0.5× (slower than FP32)   │
│  Float32     32            4 GB        1× baseline               │
│  BFloat16    16            2 GB        2×                        │
│  Int8        8             1 GB        4×                        │
│  Int4/FP4    4             0.5 GB      8×                        │
│  Int2        2             0.25 GB     16×                       │
│  Binary      1             0.125 GB    32×                       │
└──────────────────────────────────────────────────────────────────┘
```

`poly/` supports all 21 DTypes in the same training and inference loop. Switching precision is a single function call — no retraining required.

---

## The WeightStore: Three-Layer Storage

Every `VolumetricLayer` holds a `*WeightStore`:

```go
type WeightStore struct {
    Master     []float32          // Source of truth — always FP32
    Versions   map[DType]any      // CPU-resident quantized versions
    GPUWeights map[DType]any      // VRAM-resident wgpu.Buffer versions
    GPUScales  map[DType]*wgpu.Buffer  // Per-dtype scale buffers on VRAM
    Scale      float32            // Quantization scale factor
}
```

### Layer 1: Master

`Master` is the FP32 weight array that training operates on. Gradient updates always modify `Master`. No other layer is ever trained directly.

### Layer 2: Versions

`Versions` is a cache of quantized representations derived from `Master`. Each key is a `DType`. The value type varies:

```
DType             Value type in Versions
───────────────────────────────────────
Float64           []float64
Float16/BFloat16  []float32  (simulated — stored as float32 but treated as 16-bit)
Int32/Int16/Int8  []int32 / []int16 / []int8
Int4/FP4/Binary   []int8  (unpacked — one value per element; bit-packing is for disk only)
```

### Layer 3: GPUWeights / GPUScales

`GPUWeights` holds `wgpu.Buffer` references to VRAM. They are populated via `layer.SyncToGPU()` and consumed by the GPU forward/backward shaders. `GPUScales` holds the quantization scale as a separate GPU buffer used by quantized shader kernels.

---

## Morph: Producing a Quantized Version

```go
func (ws *WeightStore) Morph(dtype DType)
```

`Morph` converts `ws.Master` to the target `dtype` and stores the result in `ws.Versions[dtype]`. It is idempotent — if the target version already exists, it returns immediately.

```
ws.Master ([]float32)
      │
      ├── dtype == Float32 → return immediately (Master is already FP32)
      │
      ├── dtype == Float64 → []float64: direct cast
      │
      ├── dtype == Float16/BFloat16 → []float32: SimulatePrecision per element
      │
      ├── dtype == Int8/Uint8/FP8* → []int8: v / ws.Scale, clamped to [-128, 127]
      │
      ├── dtype == Int16/Uint16 → []int16: v / ws.Scale
      │
      ├── dtype == Int32/Uint32 → []int32: v / ws.Scale
      │
      └── dtype == Int4/FP4/Int2/Ternary/Binary → []int8 (one per weight):
              Int4/FP4/Int2: v / ws.Scale, truncated to range
              Ternary: round to {-1, 0, +1}
              Binary: +1 if v > 0, else -1
```

> [!NOTE]
> Sub-byte types (Int4, Int2, Binary) are stored in `Versions` as unpacked `[]int8` with one element per weight. The bit-packing into nibbles and pairs happens only during serialization (`encodeNativeWeights`). This keeps the forward pass simple — no runtime unpacking overhead during inference.

### Clearing Versions After Training

When `ApplyGradients` runs, it updates `Master` and then clears `Versions`:

```go
ws.Versions = make(map[DType]any)
```

This ensures stale quantized copies are not used after a weight update. The next forward pass calls `Morph` again to regenerate the needed version. This lazy invalidation pattern means training overhead is minimal — quantized versions are only regenerated on the first forward pass of each new batch.

---

## Unpack: Reconstructing Master from a Quantized Version

```go
func (ws *WeightStore) Unpack(dtype DType)
```

`Unpack` is the inverse of `Morph`. It reads `ws.Versions[dtype]` and reconstructs `ws.Master`. This is used after deserialization — the JSON stores the quantized version, and `Unpack` brings `Master` back to FP32 so the network is ready for inference or further training.

```
ws.Versions[dtype]
      │
      ├── []float64 → cast to float32
      ├── []float32 → copy directly (Float16/BFloat16 simulation)
      ├── []int8    → v * ws.Scale (for Int8, FP8, Int4, Int2, etc.)
      ├── []int16   → v * ws.Scale
      └── []int32   → v * ws.Scale
```

---

## SimulatePrecision: Quantization-Aware Arithmetic on CPU

```go
func SimulatePrecision(v float32, dtype DType, scale float32) float32
```

For types that have no native Go representation (Float16, BFloat16, FP8, FP4, Int4, Int2, Ternary, Binary), `SimulatePrecision` applies the quantization rounding and dequantization in one step. The CPU forward pass uses this during inference when a layer's `DType` is set to one of these types.

```
┌──────────────────────────────────────────────────────────────────────┐
│  How SimulatePrecision works for Int8 (scale = 0.01):               │
│                                                                      │
│  Input: v = 0.437                                                    │
│  Step 1: quantize   →  q = round(0.437 / 0.01) = 44                │
│  Step 2: clamp      →  q = clamp(44, -128, 127) = 44               │
│  Step 3: dequantize →  result = 44 * 0.01 = 0.44                   │
│                                                                      │
│  The rounding error is: |0.437 - 0.44| = 0.003                     │
│  This error is what Int8 quantization "costs"                        │
└──────────────────────────────────────────────────────────────────────┘
```

For Binary:
```
if v > 0: return +1.0
else:     return -1.0
```

For Ternary:
```
threshold = scale * 0.5
if |v| < threshold: return 0.0
if v > 0:           return scale
else:               return -scale
```

For FP8E4M3 / FP8E5M2:
```
The value is clamped to the FP8 representable range, then rounded to the
nearest FP8 representable value by simulating the mantissa truncation.
```

---

## Scale Calibration

`ws.Scale` is the per-layer quantization scale. It is computed during `Morph` using the **absolute-maximum** calibration strategy:

```
scale = max(|weight|) / maxQuantValue

For Int8:  maxQuantValue = 127
For Int4:  maxQuantValue = 7
For Int2:  maxQuantValue = 1
For Int1:  maxQuantValue = 1  (binary: +1/-1)
```

This is the simplest calibration method — no calibration data required. It is a Post-Training Quantization (PTQ) approach: train at FP32, then call `MorphLayer` to convert to the target dtype. The scale is derived analytically from the weight distribution alone.

> [!TIP]
> For activation-aware quantization (computing scale from representative inputs rather than from weights alone), you would need to run a calibration forward pass and inject the computed scale into `ws.Scale` before calling `Morph`. The current pipeline does not implement observer-based calibration for activations — only weight calibration.

---

## MorphLayer: Network-Wide Conversion

```go
func MorphLayer(n *VolumetricNetwork, dtype DType)
```

`MorphLayer` iterates all layers in the network and calls `ws.Morph(dtype)` on each. This is the primary entry point for converting a trained FP32 network to a lower-precision format:

```go
// Train at FP32
poly.Train(network, trainingData, config)

// Convert to Int8 for deployment
poly.MorphLayer(network, poly.DTypeInt8)

// The network is now ready for Int8 inference
// All new forward passes will use Versions[DTypeInt8]
```

For layers that already have a version for the target `dtype`, `Morph` skips them. To force a re-quantization (e.g., after manual scale adjustment), clear the version first:

```go
delete(layer.WeightStore.Versions, poly.DTypeInt8)
layer.WeightStore.Morph(poly.DTypeInt8)
```

---

## Q4_0Block: Block Quantization

In addition to the global-scale quantization in `WeightStore.Morph`, `poly/` implements the **Q4_0 block format** used by llama.cpp and GGUF:

```go
type Q4_0Block struct {
    Scale   float32   // one float32 scale per block
    Weights [16]byte  // 32 nibbles (4-bit signed values)
}
// Total: 4 + 16 = 20 bytes per block
// Bandwidth: 20 bytes / 32 weights = 0.625 bytes/weight
```

### QuantizeQ4_0

```go
func QuantizeQ4_0(weights []float32) []Q4_0Block
```

Converts a flat FP32 slice into Q4_0 blocks:

```
For each block of 32 weights:
  1. Find maxAbs = max(|weights[i]|) in the block
  2. scale = maxAbs / 7.0         ← 4-bit signed range is [-8, 7]
  3. For each weight pair (w1, w2):
       q1 = round(w1 / scale), clamped to [-8, 7]
       q2 = round(w2 / scale), clamped to [-8, 7]
       byte[j] = (q1 & 0xF) | ((q2 & 0xF) << 4)   ← pack 2 values per byte
```

The per-block scale means every 32 weights have their own scale factor, which is significantly more accurate than a single global scale for the entire layer. This is why Q4_0 retains much higher fidelity than naive Int4.

### DequantizeQ4_0

```go
func DequantizeQ4_0(blocks []Q4_0Block, n int) []float32
```

Unpacks nibbles and applies the per-block scale:

```
For each block:
  For each byte b:
    q1 = (b & 0xF)       → sign-extend: if q1 > 7, q1 -= 16
    q2 = (b >> 4)        → sign-extend: if q2 > 7, q2 -= 16
    res[idx1] = float32(q1) * block.Scale
    res[idx2] = float32(q2) * block.Scale
```

### Q4_0 vs Global Int4

```
┌───────────────────────────────────────────────────────────────────┐
│  Comparison for a Dense layer with 4096×4096 weights             │
│                                                                   │
│  Format         Scale count  Bytes        Notes                   │
│─────────────────────────────────────────────────────────────────  │
│  FP32           1 (implicit) 67.1 MB      No quantization        │
│  Global Int4    1            8.4 MB       One scale for all      │
│  Q4_0 blocks    524288       8.6 MB       One scale per 32 wts   │
│                                           (2% overhead, 10× fidelity) │
└───────────────────────────────────────────────────────────────────┘
```

Q4_0 is the preferred format for loading HuggingFace/GGUF checkpoints. The `universal_loader.go` and `safetensors.go` paths use `QuantizeQ4_0` internally when importing Q4_0 tensors.

---

## The Full PTQ Workflow

```
┌──────────────────────────────────────────────────────────────────────┐
│  1. Train at FP32                                                    │
│                                                                      │
│     poly.Train[float32](network, data, config)                       │
│     → Master updated each batch                                      │
│     → Versions map is cleared after each update                      │
│                                                                      │
│  2. (Optional) Calibrate scale                                       │
│                                                                      │
│     For each layer:                                                  │
│       maxAbs := findMaxAbs(layer.WeightStore.Master)                 │
│       layer.WeightStore.Scale = maxAbs / targetRange                 │
│                                                                      │
│  3. Morph to target dtype                                            │
│                                                                      │
│     poly.MorphLayer(network, poly.DTypeInt4)                         │
│     → Versions[DTypeInt4] = []int8{...} created for each layer      │
│     → Scale stored in WeightStore.Scale                              │
│                                                                      │
│  4. Save the quantized model                                         │
│                                                                      │
│     jsonData, _ := poly.SerializeNetwork(network)                    │
│     os.WriteFile("model_int4.json", jsonData, 0644)                 │
│     → encodeNativeWeights packs []int8 into nibbles (0.5 bytes/wt)  │
│                                                                      │
│  5. Load and run inference                                           │
│                                                                      │
│     network, _ := poly.DeserializeNetwork(jsonData)                  │
│     → Unpack(DTypeInt4) reconstructs Master from nibbles             │
│     → Versions[DTypeInt4] restored for fast inference                │
│     → forward passes use Versions[DTypeInt4], not Master             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Forward Pass with Quantized Weights

During a forward pass, `DispatchLayer` calls the layer-specific function (e.g., `DenseForwardPolymorphic`). Inside that function, the active weights are retrieved via:

```go
weights := layer.WeightStore.GetActive(layer.DType)
if weights == nil {
    weights = layer.WeightStore.Master
}
```

`GetActive` returns `Versions[dtype]` if it exists, otherwise `nil`. If the version is missing (e.g., after a gradient update), the forward pass falls back to `Master` and uses `SimulatePrecision` on each weight during the matrix multiply. This is slower but always correct.

For the GPU path, `GetActive` for GPU dtypes reads from `GPUWeights[dtype]` via the shader's bind group. The CPU never sees these weights once they are on VRAM.

---

## Accuracy vs. Compression Trade-offs

From empirical benchmarks in the README:

```
┌─────────────────────────────────────────────────────────────────┐
│  DType      Similarity to FP32 (cosine)   Size factor          │
├─────────────────────────────────────────────────────────────────┤
│  Float64    1.000                          2.0× larger         │
│  BFloat16   0.999+                         0.5×                │
│  Int8       0.998+                         0.25×               │
│  Int4/FP4   0.99+                          0.125×              │
│  Int2       0.97+                          0.0625×             │
│  Ternary    0.96+                          0.0625×             │
│  Binary     0.90+                          0.03125×            │
└─────────────────────────────────────────────────────────────────┘
```

The similarity scores are measured with `poly.CompareNetworks` (see `dna.md`) — comparing the cosine angle between normalized weight vectors after precision simulation. A score of 0.999 means the quantized layer points in essentially the same direction as the FP32 layer, meaning functional behavior is preserved.

> [!NOTE]
> Binary (1-bit) networks at 0.90 cosine similarity will show measurable accuracy degradation on complex tasks. Binary quantization is best suited for embedding layers, lookup tables, or architectures specifically designed for 1-bit operation (e.g., BitNet). For most tasks, Int8 or Int4 provides the best accuracy/compression balance.
