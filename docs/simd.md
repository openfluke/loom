# Plan 9 SIMD — Forward and Backward

Loom's CPU SIMD path uses **Plan 9 assembly** (`.s` files) for hot GEMV-style kernels on **amd64** (AVX2) and **arm64** (NEON). Forward passes use **`DotTile`**; backward passes use **`SaxpyF32AccF64`** (and strided variants). Both are wired through the same **`UseSimdForward`** flag and **`TrainingModeCPUSimd`** training mode.

## Overview

| Kernel | Role | amd64 | arm64 |
|--------|------|-------|-------|
| `DotTile` | Forward GEMV (dot products) | `avx2_amd64.s` | `dot_neon_arm64.s` |
| `SaxpyF32AccF64` | Backward weight/input accumulation | `saxpy_avx2_amd64.s` | `saxpy_neon_arm64.s` |
| `BitNetTernaryMAD` | Ternary weight forward (optional) | `bitnet_ternary_amd64.s` | `bitnet_ternary_arm64.s` |

**Layers with SIMD forward + backward:** Dense, SwiGLU, MHA, CNN1, CNN2, CNN3, RNN, LSTM, Embedding, Residual.

**Native-exact SIMD** (`*_native_simd.go`): same numerics as `*_native.go`, faster kernels — MAC dtypes via `materializeF32Weights` + `DotTile`; true integers via `DotI8Tile` / `SaxpyI8*`. Enabled when both `UseExactDType` and `UseSimdForward` are on (Lucy menu **[14]**).

**Layers without heavy GEMV SIMD:** RMSNorm, Softmax (attention softmax/RoPE in MHA remain scalar).

## Enabling SIMD

```go
// Per-network (training helper)
poly.ConfigureNetworkForMode(net, poly.TrainingModeCPUSimd)

// Or manually
net.SetSimdForwardRecursive(true) // enables forward DotTile + backward saxpy on all layers
```

`TrainingModeCPUSimd` is MC-tiled CPU training with SIMD: same tiling/parallelism as `TrainingModeCPUMC`, plus recursive `UseSimdForward` on every compute layer.

## Package layout

```
poly/simd/
├── dot.go, saxpy.go           # Go wrappers, feature detection
├── avx2_amd64.s               # DotTile (AVX2)
├── dot_neon_arm64.s           # DotTile (NEON)
├── saxpy_avx2_amd64.s         # SaxpyF32AccF64 (AVX2)
├── saxpy_neon_arm64.s         # SaxpyF32AccF64 (NEON)
├── bitnet_ternary_amd64.s     # optional ternary forward
└── stub.go                    # scalar fallback when !amd64 && !arm64

poly/
├── simd_forward.go            # layerUseSimdForward, SetSimdForwardRecursive
├── {dense,swiglu,mha,cnn1,cnn2,cnn3,rnn,lstm}_simd.go          # default forward (GetActive FP32)
├── {dense,swiglu,mha,cnn1,cnn2,cnn3,rnn,lstm}_simd_backward.go # default backward
├── {dense,swiglu,mha,cnn1,cnn2,cnn3,rnn,lstm,embedding,residual}_native_simd.go  # native-exact SIMD
├── embedding_simd.go          # parallel lookup / scatter (MAC native bridge)
└── residual_simd.go           # polymorphic residual add (non-native-exact)
```

## Forward path

When `layerUseSimdForward(layer) && simd.SimdEnabled()`:

- **Dense / SwiGLU / RNN / LSTM:** `simd.DotTile` over weight rows (same tile structure as MC tiled path).
- **MHA:** DotTile on Q/K/V/O projections; attention softmax and RoPE stay scalar.
- **CNN1/2/3:** DotTile over filter patches where the receptive field is contiguous.

Accumulation uses **float64** internally where the tiled path does, for dtype parity.

## Backward path

When the same SIMD gate is true, each layer's `*BackwardPolymorphic` tries `try*BackwardSimd` first:

| Layer | ∂L/∂W | ∂L/∂X / hidden |
|-------|-------|----------------|
| Dense, SwiGLU | `SaxpyF32AccF64` per output unit | saxpy scatter into input grad |
| MHA | saxpy on projection paths | DotTile for Q recompute; rest scalar |
| CNN1/2/3 | saxpy over contiguous input patches | output-centric saxpy scatter (∂L/∂X) |
| RNN, LSTM | saxpy per hidden unit (LSTM: 4 gates) | BPTT; hidden-hidden / cell carry scalar |

CNN backward uses an **output-centric saxpy scatter** for ∂L/∂X (matches tiled exactly; stride-1 full-kernel fast paths, scalar at edges).

RNN/LSTM: parallel over batch; ∂L/∂W_IH and ∂L/∂X via saxpy; hidden-hidden, bias, and cell state paths match tiled scalar code.

## Native-exact SIMD (`*_native_simd.go`)

When `UseExactDType` and `UseSimdForward` are both enabled, native-exact layers try SIMD before scalar native fallback:

```
LayerForwardNativeExact
  └─ layerUseSimdForward → try*ForwardNativeSimd
       ├─ use*TrueNative  → DotI8Tile / SaxpyI8* (int8 MAC)
       └─ else             → materializeF32Weights → *SimdF32WithWeights
  └─ fallback → scalar *_native.go
```

Lucy **[14]** reports SIMD fwd/bwd speedup columns for layers that implement this bridge (Dense through Residual). Default menu **[7]** SIMD uses `GetActive` FP32 tiles — same fast kernels, **QAT-like** numerics. Full **[14]** pass matrix and native-exact speedup tables: [native_layers.md](native_layers.md).

## Validation

**Unit tests** (`poly/tests/*_backward_test.go`): SIMD backward vs tiled forward/backward; SC vs MC where applicable.

**Seven-layer suite** (`lucy/examples/seven_layer/`):

- `TestSimdParityAllLayers_Float32_1x1` — fwd/bwd SC ↔ MC ↔ SIMD, all seven compute layers
- Per-layer grid tests (e.g. `TestCNN3SimdParityAllGrids_Float32`, `TestLSTMSimdParityAllGrids_Float32`)
- Training table includes **CPU-SIMD** column alongside SC and MC

Run:

```bash
go test ./lucy/examples/seven_layer/ -run Simd -count=1
go test ./poly/tests/ -run Backward -count=1
```

## Seven-layer benchmark results

Logs: `seven_layer_amd.txt`, `seven_layer_arm.txt` (Float32, 1×1×1 grid, **avg 25 passes** per cell). Speedup = SC time ÷ SIMD time (>1 means SIMD is faster).

### Forward (SIMD vs SC)

| Layer | AMD | ARM |
|-------|-----|-----|
| Dense | **3.6×** (72% faster) | **3.0×** (67% faster) |
| SwiGLU | **2.0×** (49% faster) | **1.5×** (34% faster) |
| MHA | **1.7×** (42% faster) | **1.6×** (36% faster) |
| CNN1 | **1.2×** (16% faster) | **1.8×** (44% faster) |
| CNN2 | **1.4×** (26% faster) | ~1.0× |
| CNN3 | 0.65× (52% slower) | 0.85× (15% slower) |
| RNN | **1.6×** (39% faster) | **2.0×** (51% faster) |
| LSTM | **1.2×** (15% faster) | **1.1×** (9% faster) |

Dense forward SIMD is the largest win on both platforms (wide GEMV). CNN3 forward can be slower than SC on small grids (patch layout / overhead); backward still wins (below).

### Backward (SIMD vs SC)

| Layer | AMD | ARM |
|-------|-----|-----|
| Dense | **1.7×** (40% faster) | **1.1×** (11% faster) |
| SwiGLU | 0.85× (15% slower) | **1.3×** (23% faster) |
| MHA | **1.8×** (46% faster) | **1.2×** (20% faster) |
| CNN1 | **1.3×** (21% faster) | **1.5×** (32% faster) |
| CNN2 | **2.3×** (56% faster) | **1.7×** (40% faster) |
| CNN3 | **2.8×** (64% faster) | **1.6×** (38% faster) |
| RNN | ~1.0× | **1.6×** (38% faster) |
| LSTM | **1.1×** (7% faster) | **1.6×** (37% faster) |

CNN2/CNN3 backward saxpy paths show the strongest gains. SwiGLU backward on AMD can trail SC slightly (strided weights + gate structure); ARM sees a modest win.

### Training (Dense, Float32, 50 epochs, 1×1×1)

| Platform | SC | MC | SIMD |
|----------|-----|-----|------|
| AMD | 14.3 ms (~3496 steps/s) | 16.4 ms (~3046 steps/s) | **9.72 ms (~5144 steps/s)** |
| ARM | 8.96 ms (~5576 steps/s) | 12.4 ms (~4029 steps/s) | **9.88 ms (~5063 steps/s)** |

On AMD, SIMD training is ~**47% faster** than SC for this Dense micro-benchmark. On ARM, SIMD sits between SC (fastest for this tiny grid) and MC.

### Parity

On both platforms, seven-layer **Float32 1×1×1** parity tables report **0** fwd/bwd diff across SC, MC, and SIMD for all seven layer types (21 dtype rows in full suite). Larger grids (2³, 3³) are covered by per-layer `*SimdParityAllGrids*` tests.

## BitNet (optional)

Ternary / packed BitNet forward kernels live in the same `poly/simd` package (`bitnet_ternary_*.s`, TL1 variants on arm64). These are separate from the main float32 training SIMD path; see `poly/simd/doc.go` and BitNet tests.

## See also

- [Training](training.md) — `TrainingModeCPUSimd`, `UseExactDType`, and [training paradigms](training.md#training-paradigms-default-qat-like-vs-native-exact)
- [Quantization](quantization.md) — [three training/inference modes](quantization.md#three-traininginference-modes)
- [Testing and validation](testing_and_validation.md) — seven-layer SC/MC/SIMD suite
- [Bedrock validation](bedrock_validation.md) — menu option [7] and artifact layout
