# `poly/asm` — Plan 9 CPU kernels

Hand-written **Plan 9 assembly** (`.s` files, **not** CGO) for hot inner loops. Go owns tiling, batching, and layer dispatch; asm owns dot products and tight matmul tiles.

**Toggle from poly:** `VolumetricNetwork.UseAsmForward` or `VolumetricLayer.UseAsmForward` (see `poly/dense_asm.go`, `poly/dense_asm_native.go`).

**Platforms:** `amd64` and `arm64` ship `.s` kernels (`asm.Enabled()`). Other `GOARCH` uses pure-Go fallbacks in `*_stub.go` — no crash.

---

## Layout

```
asm/
├── doc.go              # package root
├── enabled_*.go        # asm.Enabled() per arch
├── dot/                # dot products (shared by all layers)
├── matmul/             # tiled GEMV / forward orchestration
└── dense/              # LayerDense forward entry (more layers later)
```

| Package | Role |
| :--- | :--- |
| **`dot/`** | Scalar and tile dots: float (`f32`/`f64`), native integer (`IMUL` + int64 acc), legacy packed row dots |
| **`matmul/`** | Output-parallel tiled forward, MC via `OverOutputTiles`, wires `dot` into tile callbacks |
| **`dense/`** | Thin forward API used by `poly` (`asmdense.Forward` for floats) |

Future: `asm/mha/`, `asm/swiglu/`, `asm/cnn/` — same `dot` + `matmul` building blocks.

---

## How Dense forward reaches asm

```
poly.DenseForwardPolymorphic
  └─ layerUseAsmForward && asm.Enabled()
       └─ denseForwardAsm (dense_asm.go)
            ├─ isNativeQuantDType → denseForwardAsmNative (dense_asm_native.go)
            │    ├─ Int16/32/64, Uint16/32/64 → width-native tiled forward
            │    └─ Int8, Int4, Ternary, Binary, … → denseForwardAsmMorphU8
            │         · morphed []uint8 weights (WeightStore.Morph)
            │         · quantize input (nativeQuantValue + scale)
            │         · matmul.ForwardNativeU8 (signed or unsigned dot)
            │         · dequant to float pre-activation (scale²)
            └─ else → denseForwardAsmFloatPath → asm/dense.Forward (float tiled)
```

**Important:** CPU dense hot path for Int4/Ternary/Binary uses **one quant byte per weight** in RAM (`[]uint8` from morph). That is **not** the GPU bit-packed `[]uint32` stream.

---

## Compute paths (dot / matmul)

| Path | Go entry | Accumulator | FP inside dot? | Used for |
| :--- | :--- | :--- | :---: | :--- |
| **Float tiled** | `matmul.ForwardTiledF32/F64`, `asm/dense.Forward` | `float32` / `float64` | yes | Float32/64/16/BF16, FP8 (for now) |
| **Native integer** | `matmul.ForwardNativeU8/I8/…/I64` | `int64` | **no** | Int8–Int64, morphed-u8 low-bit |
| **Int acc (legacy)** | `matmul` + `dot.*TileAccF64` | `float64` via `CVTSQ2SD` | yes | superseded for dense; kept in tree |
| **Packed row** | `dot.NibblePackedRowNativeI64`, etc. | `int64` | no | GPU bitstreams, BitNet; **not** dense CPU hot path |

### Native integer rules

- Multiply and accumulate in the **storage width** (or morphed byte interpreted as `int8`).
- **No** widening activations/weights to `float32` inside the inner loop (that is dequantization, not low-bit math).
- Quantization / dequantization happens **once** at the tensor boundary (`scale`, `scale²`).
- Clamp to output dtype range when storing tile partials (e.g. `int8` → `clampInt8`).

### Morphed-u8 signed vs unsigned

| Signed dot (`U8BytesTileNativeI64`) | Unsigned dot (`U8TileNativeI64`) |
| :--- | :--- |
| Int8, Int4, Int2, Ternary, Binary | Uint8, Uint4, Uint2, FP4 |

---

## Assembly sources (`.s`)

Plan 9 files under `dot/` (`*_amd64.s`, `*_arm64.s`) are **checked in** and edited directly — no Python codegen in-tree. After changing `.s` or stubs, run `go test ./asm/...`.

| File family | Purpose |
| :--- | :--- |
| `f32_*.s`, `f64_*.s`, `f32_acc_*.s` | float dot / f32×f64 acc tiles |
| `native_int_*.s` | `I8`–`U64` native dots (`dot*NativeI64`) |
| `int_acc_*.s` | integer → float64 acc (legacy / reference) |
| `native_packed_*.s` | packed row dots (GPU / BitNet layouts) |

---

## Tests

Tests live in per-package **`test/`** subfolders (`package foo_test`, external tests). Add new asm tests there, not beside production `.go` files.

```bash
cd loom/poly
go test ./asm/... ./tests/... -count=1
```

| Package | Test directory |
| :--- | :--- |
| `asm/dot` | [`asm/dot/test/`](dot/test/) |
| `asm/matmul` | [`asm/matmul/test/`](matmul/test/) |
| `asm/dense` | [`asm/dense/test/`](dense/test/) |
| `poly` (asm integration) | [`poly/tests/dense_asm_parity_test.go`](../tests/dense_asm_parity_test.go) |

---

## Lucy benchmarks

Run from `loom/lucy` → **Dense** → **L1 Caching** or **GPU Forward Parity**. Log: `loom/lucy/lucy_testing_output/log.txt`.

| Column | Meaning |
| :--- | :--- |
| **Go SC / Go MC** | CPU reference (`DenseForwardTiled`, float64 acc for floats) |
| **ASM SC / ASM MC** | `UseAsmForward` dense path |
| **Go/Asm↑** | `Go_time / ASM_time` — **> 1.0** = asm wins |
| **D(G,SC)** | max abs diff Go CPU vs GPU forward |
| **Forward Parity** | 💎 exact · ✅ industry · 🟤 heavy drift |

**Interpreting parity:** Lucy compares ASM to Go **float64 tiled** forward. Native-int asm (exact integer math + boundary dequant) often shows 🟤 **heavy drift** vs that reference while still being correct for the native path. Float dtypes may show ~2.5 `D(G,SC)` from Go f64 vs GPU f32 — separate from asm.

### Recent Dense L1 snapshot (arm64 / Metal, post morph-u8 fix)

| DType | Go/Asm↑ SC | Go/Asm↑ MC | Notes |
| :--- | ---: | ---: | :--- |
| Uint8 | ~2.5× | ~2.4× | best overall |
| Int4, Ternary, Binary | ~1.7–2.0× | ~2.5–3.0× | was ~0.5× / ~0.17× on packed path |
| FP4, Uint4 | ~2.3–2.4× | ~2.5–3.0× | |
| Int8 | ~1.6× | ~2.6× | |
| Float32 | ~1.1× | ~0.8–1.0× | |
| Float64 | ~0.9× | ~1.2× | SC still slightly behind Go |
| Int64 / Uint64 | ~0.8× | ~1.0× | SC tuning opportunity |

Training matrix (84 runs) remains 💎 with asm forward-only.

---

## Shipped vs TODO

### Shipped ✅

- [x] `asm/dot` — `f32`, `f64`, `f32_acc_f64` tile dots
- [x] `asm/dot` — native `I8`–`U64` tile dots (`native_int_*.s`)
- [x] `asm/dot` — packed row dots (`native_packed_*`) for non-dense consumers
- [x] `asm/matmul` — float tiled forward SC+MC
- [x] `asm/matmul` — native integer tiled forward SC+MC
- [x] `asm/dense` — float forward entry
- [x] `poly` — `denseForwardAsmNative` morph-u8 route for all low-bit dtypes
- [x] `poly` — `denseForwardAsm` routing for 21 dtypes on Dense forward

### In progress / next 🚧

- [ ] **Dense backward** asm (dW, dX tiled)
- [ ] **Buffer pooling** for `qIn`/`qOut` in `denseForwardAsmMorphU8` (~half wall time is alloc + quant/dequant)
- [ ] **FP8 native** dot (e4m3/e5m2 bit math, no float widen in kernel)
- [ ] **Float64 SC** parity with Go (currently ~0.9× on Lucy)
- [ ] **Int64 SC** speed (native dot OK; load width / SIMD)
- [ ] **NEON / AVX** unrolled tile dots (4–8 lanes)
- [ ] **SwiGLU** asm package (3× matmul per block)
- [ ] **MHA** asm package (Q/K/V/O)
- [ ] **CNN** asm inner loops

### Explicit non-goals (for now)

- CGO / external BLAS — stack stays Go + Plan 9 `.s` + WebGPU
- Repacking morphed CPU weights to `[]uint32` on every forward — use morph storage instead

---

## Adding a new layer package

1. Add `asm/<layer>/forward.go` calling shared `matmul` + `dot`.
2. Wire from `poly/<layer>_asm.go` behind `UseAsmForward` (or layer-specific flag).
3. Add or extend `*_amd64.s` / `*_arm64.s` (and Go stubs) for new element widths or packed layouts.
4. Add `*_test.go` + one Lucy row when the layer suite supports it.
5. Update the [ASM layer matrix](../README.md#asm-layer-matrix-status) in `poly/README.md`.

---

## Related files (poly)

| File | Role |
| :--- | :--- |
| `poly/dense_asm.go` | asm toggle, float vs native dispatch |
| `poly/dense_asm_native.go` | native / morph-u8 forward |
| `poly/weights.go` | `Morph`, `nativeQuantValue` |
| `poly/dense.go` | Go tiled reference forward |
