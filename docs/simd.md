# Plan 9 SIMD Forward

Loom ships hand-written **Plan 9 assembly** SIMD kernels (`poly/simd/*.s`, **not** CGO)
for CPU forward passes. They are opt-in per network, cover the eight compute-heavy
layer types across all 21 dtypes, and include a dedicated BitNet **ternary MAD**
kernel so 1.58-bit models accelerate without dequantizing to float.

**Related docs:** [`bitnet_cpu.md`](bitnet_cpu.md), [`dispatch.md`](dispatch.md),
[`numerical_types.md`](numerical_types.md), [`transformer.md`](transformer.md).
For the archived (removed) ASM/executor experiment, see
[`asm-and-volumetric-exploration.md`](asm-and-volumetric-exploration.md) — that is a
**different, superseded** effort; this document describes the **current, in-tree** path.

---

## The two kernels

Everything routes through the `poly/simd` package, which exposes exactly two
compute primitives plus enable/detect helpers.

| Kernel | Signature | Used for |
| :--- | :--- | :--- |
| **Float32 tile dot** | `DotTile(x, w []float32, i0, i1 int, prev float64) float64` | Every float / dequantized dtype |
| **BitNet ternary MAD** | `BitNetTernaryCodeRowDot(codes []uint8, acts []int8, nBytes int) int32` | Packed 1.58-bit ternary (`DTypeTernary` + `UseExactDType`) |

Backend per architecture:

| GOARCH | Float32 `DotTile` | Ternary MAD | Notes |
| :--- | :--- | :--- | :--- |
| **amd64** | AVX2 + FMA (`avx2_amd64.s`) | AVX2 `VPMADDUBSW` (`bitnet_ternary_amd64.s`) | `float64` accumulation |
| **arm64** | 4-wide scalar-unrolled Go (`neon_arm64.go`) | scalar Go fallback | correct, not yet true NEON asm |
| other | scalar Go (`stub.go`) | scalar Go | `SimdEnabled()` reports `false` on amd64/arm64 only |

Both `DotTile` implementations accumulate in `float64` for numerical stability
even though inputs are `float32`, matching the reference kernels in
`loom-dense-bench`.

---

## What is covered

### Layer types

`SetSimdForward` enables the Plan 9 path for these eight layer types:

| Layer | SIMD file | Kernel |
| :--- | :--- | :--- |
| Dense | `poly/dense_simd.go` | float `DotTile` / ternary MAD |
| SwiGLU | `poly/swiglu_simd.go` | float `DotTile` / ternary MAD |
| MHA (attention) | `poly/mha_simd.go` | float `DotTile` / ternary MAD |
| CNN1 | `poly/cnn1_simd.go` | float `DotTile` (im2col patch dot) |
| CNN2 | `poly/cnn2_simd.go` | float `DotTile` |
| CNN3 | `poly/cnn3_simd.go` | float `DotTile` |
| RNN | `poly/rnn_simd.go` | float `DotTile` |
| LSTM | `poly/lstm_simd.go` | float `DotTile` (4 gate projections) |

`RMSNorm`, `Residual`, `Embedding`, and `Softmax` have **no** SIMD kernel — they
are elementwise / gather / normalization ops with no multiply-accumulate to
vectorize, so the flag is a no-op for them (they stay on their scalar path).

Check programmatically:

```go
poly.LayerSupportsSimdForward(poly.LayerDense)      // true
poly.Plan9SimdForwardForLayer(poly.LayerRMSNorm)    // false
poly.Plan9SimdEnabled()                             // hardware SIMD linked?
```

### Numerical types

There is only **one** float kernel. Non-float dtypes are dequantized to `float32`
once by `WeightStore.GetActive(dtype)` and then run through `DotTile`:

| Dtype group | Path |
| :--- | :--- |
| Float32 | direct `DotTile` |
| Float64, Float16, BFloat16, FP8-E4M3/E5M2, Int/Uint 8/16/32/64, Int/Uint 4/2, FP4, Binary | dequantize → `float32` → `DotTile` |
| **Ternary** (`DTypeTernary` + `UseExactDType`) | packed **ternary MAD** (no dequant) |

So an int4 checkpoint like Qwen3-0.6B accelerates through the same float kernel as
a float32 model; only BitNet's packed ternary takes the dedicated integer path.

---

## Enabling it

```go
net.SetSimdForward(true)            // flat layer list
net.SetSimdForwardRecursive(true)   // also parallel/sequential/meta sub-layers
```

Both set `Network.UseSimdForward` and each layer's `UseSimdForward`, **and** toggle
the BitNet ternary AVX2 flag (`simd.SetBitNetTernarySimdForward`). Per-layer
dispatch then checks:

```go
// e.g. dense.go
if layerUseSimdForward(layer) && simd.SimdEnabled() {
    if pre, post, ok := tryDenseForwardSimd(layer, input); ok {
        return pre, post
    }
}
return DenseForwardTiled(layer, input) // scalar / tiled fallback
```

**Explicit means explicit.** There is no width-based "viability" auto-gating: if
you enable SIMD, the SIMD kernel runs regardless of layer size. The only reasons a
layer falls back are *correctness* constraints (non-`float32` tensor handed to the
try-path, or a packed representation the kernel does not implement). Very narrow
layers may see negligible or slightly negative speedup — that is expected physics
(vector setup not amortized), not a gate.

### Multi-core is orthogonal

SIMD (per-dot vectorization) and multi-core tiling (`EnableMultiCoreTiling`, rows
across `GOMAXPROCS`) compose. The transformer bench crosses them as
`cpu_sc / cpu_mc / cpu_simd_sc / cpu_simd_mc`.

### Tile sizes

Each layer/dtype gets an L1-aware SIMD tile size via
`layer.GetCPUSimdTileSize(dtype)` (see `poly/tile_detection.go`), populated by
`EnsureRuntimeTileSizes` / `RefreshRuntimeTileSizes`.

---

## BitNet ternary MAD kernel

BitNet weights are `{-1, 0, +1}` packed 16-per-`uint32`. The scalar path decodes
2 bits at a time; the SIMD path mirrors Microsoft's `ggml-bitnet-mad.cpp` MAD
approach:

1. **Unpack once, cached.** `ensureBitNetCodes` expands the packed weights into one
   unsigned byte per weight (code ∈ `{0,1,2}`), each row zero-padded to a multiple
   of 32. Built a single time per matrix and reused every token — **no on-the-fly
   unpacking during inference**.
2. **MAD hot loop** (`VPMADDUBSW` unsigned-code × signed-int8-activation →
   `VPMADDWD` widen → `VPADDD` accumulate), 32 weights per instruction.
3. **Ternary identity.** Since `weight = code − 1`:

   ```text
   dot = Σ(code_i · act_i) − Σ(act_i)
   ```

   The `Σ(act)` correction is computed once per activation vector. Activations are
   quantized per row to int8 (`activation_scale = 127 / max(|input|)`), and the
   final value is `dot · weight_absmean / activation_scale`.

The MAD path is bit-identical to the scalar path (both do exact integer
accumulation), toggled by `SetSimdForward*` and gated by `BitNetTernarySimdActive()`.

**Memory trade-off:** the cached byte-per-weight `Codes` buffer is ~1 byte/weight
(e.g. ~2 GB for the 2B BitNet model) on top of the packed 2-bit weights — the cost
of avoiding per-call unpack. A future variant could keep weights at ~0.5 GB with
vectorized *in-register* unpack (PSHUFB-style) at some per-call cost.

---

## Benchmarks

### Seven-layer suite — Lucy `[7]`

Per layer × per dtype, on the `1³ / 2³ / 3³` grids: parity (SIMD vs scalar), train,
save/reload. Speedups scale with layer width; narrow micro-layers (e.g. `4×4` on
`3³`) can be flat or slightly negative — vector setup is not amortized.

### Transformer decode — Lucy `[11]`

CPU decode throughput on quantized `.entity` models across
`cpu_sc / cpu_mc / cpu_simd_sc / cpu_simd_mc`, greedy, output parity checked
(`lucy/examples/transformer_simd_bench`). Representative single-core SIMD vs scalar:

| Model | Dtype path | decode SIMD speedup |
| :--- | :--- | :--- |
| Qwen3-0.6B | int4 → float32 `DotTile` | ~2.26× |
| BitNet b1.58-2B | packed ternary MAD | ~2.61× |
| SmolLM2-135M | float32 `DotTile` | ~1.29× |

---

## Testing

```bash
# Parity: SIMD kernels vs scalar (poly external tests)
cd loom && go test ./poly/tests/ -run 'Simd|BitNet' -count=1

# Seven-layer per-layer suites (Lucy menu [7])
cd loom/lucy && go run .   # → [7]

# Transformer SIMD-vs-MC decode bench (Lucy menu [11])
cd loom/lucy && go run .   # → [11]
```

---

## Package layout

```
poly/simd/
├── dot.go                 DotTile dispatcher + scalar fallback + SimdEnabled
├── avx2_amd64.go/.s       AVX2+FMA float32 tile dot (amd64)
├── neon_arm64.go          4-wide scalar-unrolled float dot (arm64)
├── stub.go                scalar fallback (other GOARCH)
├── bitnet_ternary.go      BitNetTernaryCodeRowDot dispatcher + Go fallback
├── bitnet_ternary_amd64.go/.s   AVX2 VPMADDUBSW ternary MAD kernel
├── bitnet_ternary_flag.go SetBitNetTernarySimdForward / BitNetTernarySimdActive
└── bitnet_ternary_stub.go non-amd64 ternary fallback

poly/
├── simd_forward.go        SetSimdForward(Recursive), LayerSupportsSimdForward, dispatch gate
├── bitnet_simd.go         poly re-exports of the ternary SIMD toggles
├── {dense,swiglu,mha,cnn1,cnn2,cnn3,rnn,lstm}_simd.go   per-layer try-paths
└── bitnet_cpu.go          packed ternary matvec (scalar + SIMD MAD), Codes cache
```
