# Numerical Types, DType System, and WeightStore

This document covers all 21 `DType` values, the `Numeric` generic constraint, the `WeightStore` master/versioned architecture, and the Metamorphosis mechanism that lets a layer switch precision on the fly.

---

## The 21 DTypes

```go
type DType int
```

Every `VolumetricLayer` carries a `DType` field that controls which numerical format its weights are active in. The full set:

```
┌─────┬───────────────┬──────────────────────────────────────────────┐
│ ID  │ Name          │ Description                                  │
├─────┼───────────────┼──────────────────────────────────────────────┤
│  0  │ DTypeFloat64  │ IEEE 754 double (8 bytes per weight)         │
│  1  │ DTypeFloat32  │ Standard single (4 bytes) — Master baseline  │
│  2  │ DTypeFloat16  │ 16-bit float (simulated, stored as f32)      │
│  3  │ DTypeBFloat16 │ Brain Float: 8 exp bits, 7 mantissa          │
│  4  │ DTypeFP8E4M3  │ 8-bit FP, 4-exponent 3-mantissa             │
│  5  │ DTypeFP8E5M2  │ 8-bit FP, 5-exponent 2-mantissa             │
│  6  │ DTypeInt64    │ 64-bit signed integer                        │
│  7  │ DTypeInt32    │ 32-bit signed integer                        │
│  8  │ DTypeInt16    │ 16-bit signed integer                        │
│  9  │ DTypeInt8     │ 8-bit signed integer (0.625–1.0 B/weight)   │
│ 10  │ DTypeUint64   │ 64-bit unsigned integer                      │
│ 11  │ DTypeUint32   │ 32-bit unsigned integer                      │
│ 12  │ DTypeUint16   │ 16-bit unsigned integer                      │
│ 13  │ DTypeUint8    │ 8-bit unsigned integer                       │
│ 14  │ DTypeInt4     │ 4-bit signed (2 weights per byte)           │
│ 15  │ DTypeUint4    │ 4-bit unsigned (2 weights per byte)         │
│ 16  │ DTypeFP4      │ 4-bit floating point E2M1 (2 per byte)     │
│ 17  │ DTypeInt2     │ 2-bit signed (4 weights per byte)           │
│ 18  │ DTypeUint2    │ 2-bit unsigned (4 weights per byte)         │
│ 19  │ DTypeTernary  │ 2-bit ternary: -1, 0, +1                    │
│ 20  │ DTypeBinary   │ 1-bit XNOR-Net (8 weights per byte)        │
└─────┴───────────────┴──────────────────────────────────────────────┘
```

### Storage Size per Weight

```
┌────────────────────────────────────────────────────────┐
│  DType        Bits/weight   Bytes/1024 weights          │
├────────────────────────────────────────────────────────┤
│  Float64      64            8192                        │
│  Float32      32            4096                        │
│  Float16      16            2048                        │
│  BFloat16     16            2048                        │
│  FP8E4M3      8             1024                        │
│  FP8E5M2      8             1024                        │
│  Int8/Uint8   8             1024                        │
│  Int4/Uint4   4              512   (2 per byte)         │
│  FP4          4              512   (2 per byte)         │
│  Int2/Uint2   2              256   (4 per byte)         │
│  Ternary      2              256   (4 per byte)         │
│  Binary       1              128   (8 per byte) ← 98.4% │
│                                    compression vs FP32  │
└────────────────────────────────────────────────────────┘
```

### Parsing DTypes from Strings

`ParseDType(s string) DType` accepts aliases:

| Input strings | Result |
|:-------------|:-------|
| `"float32"`, `"fp32"`, `"f32"` | `DTypeFloat32` |
| `"bfloat16"`, `"bf16"` | `DTypeBFloat16` |
| `"fp8e4m3"`, `"fp8"` | `DTypeFP8E4M3` |
| `"int4"` | `DTypeInt4` |
| `"fp4"`, `"f4"` | `DTypeFP4` |
| `"ternary"` | `DTypeTernary` |
| `"binary"` | `DTypeBinary` |

---

## The `Numeric` Constraint

```go
type Numeric interface {
    ~int | ~int8 | ~int16 | ~int32 | ~int64 |
        ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
        ~float32 | ~float64
}
```

This constraint makes `Tensor[T]`, `DispatchLayer[T]`, `ForwardPolymorphic[T]`, and all other generic functions work across any of Go's numeric primitives. The constraint is deliberately limited to types the compiler can generate native arithmetic for—no reflection, no `interface{}` boxing at the hot path.

> [!NOTE]
> FP4, FP8, BFloat16, and other non-native types are **simulated** on CPU. The layer holds them as `float32` or `int8` with a scale factor, and `SimulatePrecision` quantizes each multiply. On GPU they become native packed payloads in WGSL shaders.

---

## The WeightStore

```go
type WeightStore struct {
    Master     []float32          // Source of truth — always FP32
    Versions   map[DType]any      // Cached conversions (e.g., []int8 for INT8)
    GPUWeights map[DType]any      // VRAM-resident wgpu.Buffer references
    GPUScales  map[DType]*wgpu.Buffer  // Per-block scale buffers for quantized types
    Scale      float32            // Global quantization scale factor
}
```

The `Master` slice is allocated with `AlignedFloat32(n)` which aligns to 64-byte boundaries (one CPU cache line), enabling AVX-width SIMD operations.

### Creating and Initializing

```go
ws := NewWeightStore(inputSize * outputSize)
ws.Scale = 1.0
ws.Randomize(seed, 0.1)  // fills Master with uniform [-0.1, 0.1]
```

After `Randomize`, all `Versions` and `GPUWeights` maps are cleared, ensuring no stale low-bit versions survive.

### The Morphic Version System

```
WeightStore.Morph(dtype DType):

  Master (FP32)
       │
       ▼
  DTypeFloat64  ──▶  []float64  (direct cast)
  DTypeBFloat16 ──▶  []float32  (bits masked to 16-bit BF16)
  DTypeInt8     ──▶  []int8     (quantized: int8(v / Scale))
  DTypeInt4     ──▶  []int8     (quantized, stored 1-per-int8)
  DTypeBinary   ──▶  []int8     (sign bit only: +1 or -1)
```

The BFloat16 path uses a bit-masking trick:

```go
u32 := math.Float32bits(wVal)
u32 &= 0xFFFF0000   // zero the lower 16 mantissa bits
return math.Float32frombits(u32)
```

This preserves the exponent and upper mantissa exactly as BFloat16 would.

### Metamorphosis: Switching Precision On the Fly

A layer starts life as FP32. Before inference you can call:

```go
layer.WeightStore.Morph(DTypeInt8)
layer.DType = DTypeInt8
```

Now `DenseForwardPolymorphic` will find the `[]int8` version in `Versions[DTypeInt8]` and use the native INT8 fast-path loop. The FP32 master is untouched.

After training (`ApplyGradients`), the master is updated and **all cached versions are automatically purged**:

```go
func (ws *WeightStore) ApplyGradients(gradWeights *Tensor[float32], lr float32) {
    for i := 0; i < limit; i++ {
        ws.Master[i] -= lr * gradWeights.Data[i]
    }
    // Stale — force re-quantize on next forward:
    ws.Versions = make(map[DType]any)
    ws.GPUWeights = make(map[DType]any)
}
```

This guarantees the layer never silently uses outdated quantized weights.

```
┌──────────────────────────────────────────────────────────────┐
│                   Metamorphosis Lifecycle                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  NewWeightStore(n)                                           │
│       │                                                      │
│       ▼                                                      │
│  Randomize(seed, scale) ──▶ Master filled, Versions={}      │
│       │                                                      │
│       ▼                                                      │
│  layer.DType = DTypeInt8                                     │
│       │                                                      │
│       ▼                                                      │
│  Forward() ──▶ Morph(DTypeInt8) if Versions[INT8]==nil      │
│       │                 │                                    │
│       │          Versions[DTypeInt8] = []int8{...}          │
│       │                                                      │
│       ▼                                                      │
│  INT8 fast-path arithmetic executes                         │
│       │                                                      │
│       ▼                                                      │
│  ApplyGradients(gW, lr) ──▶ Master updated                  │
│                         ──▶ Versions = {} (cleared)         │
│                                                              │
│  Next Forward() ──▶ Morph(DTypeInt8) again from new Master  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Unpacking for Deserialization

When loading a model saved in a low-bit format:

```go
ws.Versions[dtype] = decoded  // e.g., []int8 from bit-packed JSON
ws.Unpack(dtype)              // reconstructs Master: Master[i] = packed[i] * Scale
```

This ensures the FP32 master is always available for gradient-based fine-tuning, even on a model that was serialized in INT4.

---

## SimulatePrecision

This is the universal quantization simulation function used by all CPU layers at the weight-multiply step:

```go
func SimulatePrecision(wVal float32, dtype DType, scale float32) float32
```

| DType | Behavior |
|:------|:---------|
| Float64, Int64, Uint64, Int32, Uint32 | Identity (no quantization) |
| BFloat16 | Bit-mask to upper 16 bits |
| FP8, Int8, Uint8, Int16, Uint16 | `int8(wVal/scale) * scale` |
| Int4, Uint4, FP4 | `int(wVal/scale) * scale` |
| Int2, Uint2 | 4-level: `int(wVal*2/scale) * scale/2` |
| Ternary | Threshold: `+scale`, `0`, or `-scale` |
| Binary | Sign: `+scale` if positive, else `-scale` |

The `scale` parameter comes from `WeightStore.Scale`, computed from the max absolute value of the master weights during morphing.

---

## The Q4_0 Block Format (GPU Quantization)

For GPU inference, the engine uses the Q4_0 block format, matching llama.cpp compatibility:

```
Q4_0Block:
┌────────────────────────────────────────────────────────┐
│  Scale: float32  (4 bytes)                             │
│  Weights: [16]byte  (32 nibbles = 32 × 4-bit weights)  │
│                                                        │
│  Total: 20 bytes for 32 weights = 0.625 bytes/weight   │
└────────────────────────────────────────────────────────┘
```

`QuantizeQ4_0(weights []float32) []Q4_0Block` finds the max absolute value in each block of 32, sets `scale = maxAbs / 7.0`, then quantizes each weight to a signed 4-bit integer (`-8` to `7`) packed two-per-byte.

On the GPU, the WGSL shader receives the packed uint32 array plus the float32 scales array, and dequantizes on the fly inside the shader without a CPU roundtrip.

---

## CastWeights

`CastWeights[T Numeric](weights any) []T` is the universal extraction helper. It type-switches on all 10 concrete slice types and uses `ConvertSlice[In, Out]` to re-cast the values into the requested type `T`. When `DispatchLayer` cannot find a dedicated fast-path for the layer's DType, it falls through to `CastWeights` followed by `SimulatePrecision`.

---

## Bit-Packed Serialization Ratios

From the README, verified across 378 model permutations:

| DType | Bytes/weight (serialized) | vs FP32 |
|:------|:--------------------------|:--------|
| Float32 | 4 | 1.0x |
| Float16 | 2 | 0.5x |
| Int8 | 1 | 0.25x |
| Int4/FP4 | 0.5 | 0.125x |
| Int2/Ternary | 0.25 | 0.0625x |
| Binary | 0.125 | 0.0313x ← **98.4% reduction** |

The packing/unpacking logic lives in `encodeNativeWeights` and `decodeNativeWeights` in `persistence.go`. Binary packs 8 weights per byte using bit shifts; Ternary packs 4 per byte using 2-bit fields; FP4 packs 2 per byte using nibbles.
