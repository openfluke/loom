# Multi-Precision Type Conversion

The `type_conversion.go` module provides universal, lossless-where-possible conversion between all numeric data types supported by Loom. This is the foundation for the multi-dtype inference and training system.

---

## Supported Types

Loom supports 13 numeric types, spanning float, integer, and quantised formats:

| Constant | String | Size | Range | Use Case |
|---|---|---|---|---|
| `TypeF64` | `"F64"` | 8 bytes | ±10³⁰⁸ | Research, high precision |
| `TypeF32` | `"F32"` | 4 bytes | ±10³⁸ | Standard training |
| `TypeF16` | `"F16"` | 2 bytes | ±65504 | GPU inference (packed as `uint16`) |
| `TypeBF16` | `"BF16"` | 2 bytes | ±10³⁸ | LLM inference (packed as `uint16`) |
| `TypeF4` | `"F4"` | 0.5 bytes | [-6, 6] | Extreme compression (packed as `uint8`, E2M1) |
| `TypeI8` | `"I8"` | 1 byte | [-128, 127] | Quantised edge/mobile |
| `TypeI16` | `"I16"` | 2 bytes | ±32767 | Quantised mid-range |
| `TypeI32` | `"I32"` | 4 bytes | ±2.1×10⁹ | Integer networks |
| `TypeI64` | `"I64"` | 8 bytes | ±9.2×10¹⁸ | Large integer |
| `TypeU8` | `"U8"` | 1 byte | [0, 255] | Image data |
| `TypeU16` | `"U16"` | 2 bytes | [0, 65535] | Texture data |
| `TypeU32` | `"U32"` | 4 bytes | [0, 4.3×10⁹] | Indices |
| `TypeU64` | `"U64"` | 8 bytes | [0, 1.8×10¹⁹] | Large offsets |

> [!NOTE]
> F16 and BF16 are stored as `uint16` in memory (raw bits). F4 is stored as `uint8`. All conversions go through `float64` as the universal intermediate.

---

## Conversion Strategy

All conversions route through a `float64` intermediate to guarantee correctness across any pair of types:

```
Source → float64 (lossless for integers, best-effort for floats) → Target
```

This means you can convert freely between any pair without a custom conversion path per pair, at the cost of one extra step through float64.

```
F32 → F64 → I8  ✓  (with saturation clamping)
I8  → F64 → F16 ✓  (exact, I8 range fits in F16)
BF16→ F64 → F32 ✓  (via bfloat16ToFloat32 decode)
```

---

## API

### Single Value

```go
result, err := nn.ConvertValue(float32(3.14), nn.TypeF32, nn.TypeF16)
// result is uint16 (raw F16 bits)
```

### Whole Slice

```go
f32weights := []float32{0.5, -0.3, 1.2}

// F32 → F16 (for GPU upload)
f16raw, err := nn.ConvertSlice(f32weights, nn.TypeF32, nn.TypeF16)
// f16raw is []uint16

// F32 → I8 (quantisation)
i8weights, err := nn.ConvertSlice(f32weights, nn.TypeF32, nn.TypeI8)
// i8weights is []int8, values clamped to [-128, 127]
```

`ConvertSlice` handles type assertion internally — just pass the correctly-typed slice.

---

## Type Utilities

```go
// Size in bytes per element
nn.GetTypeSize(nn.TypeF32)   // 4
nn.GetTypeSize(nn.TypeF16)   // 2
nn.GetTypeSize(nn.TypeI8)    // 1
nn.GetTypeSize(nn.TypeF4)    // 0  ← special: 2 values per byte

// Category checks
nn.IsNumericTypeFloat(nn.TypeBF16)      // true
nn.IsNumericTypeSignedInt(nn.TypeI8)    // true
nn.IsNumericTypeUnsigned(nn.TypeU32)    // true

// Range of representable values
min, max := nn.GetTypeRange(nn.TypeF16)  // -65504, 65504
min, max := nn.GetTypeRange(nn.TypeI8)   // -128, 127
```

---

## Clamping Behaviour

Integer conversions saturate at the type boundary rather than wrapping or panicking:

```go
// Float value 999.0 → I8: clamped to 127
// Float value -999.0 → I8: clamped to -128
// Float value -1.0 → U8: clamped to 0
// Float value 300.0 → U8: clamped to 255
```

This is safe by default — no silent overflow.

---

## F16 / BF16 Encoding Details

F16 and BF16 are not native Go types. They are stored as `uint16`:

| Format | Bits | Notes |
|---|---|---|
| F16 (IEEE 754 half) | S=1, E=5, M=10 | Range ±65504, good precision |
| BF16 (Brain Float) | S=1, E=8, M=7 | Same exponent range as F32, lower mantissa precision |

Both use the same Go type (`uint16`) as their storage container. When you receive an F16 result from `ConvertSlice`, cast accordingly:

```go
result, _ := nn.ConvertSlice(f32s, nn.TypeF32, nn.TypeF16)
f16Bits := result.([]uint16)  // raw F16 bits, ready for GPU upload
```

---

## F4 (4-bit Float) Details

F4 uses E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit. It represents only 16 distinct values in [-6, 6]:

```
±0.25, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
```

Values outside this range are clipped to ±6. Two F4 values pack into one byte (`uint8`). Best used for extreme compression where quality degradation is acceptable (e.g., embedding layers at the edge).

---

## Roadmap

The type conversion system is the foundation for the v0.2.0 GPU dtype work. Once integrated:
- F16 weights will be uploaded to GPU as `[]uint16` (no CPU dequant)
- The GPU shader will dequantise in WGSL at read time
- I8 inference path will quantise weights on load and dequant in shader
