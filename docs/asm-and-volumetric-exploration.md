# ASM & Volumetric Executor Exploration (Archive)

**Status:** Exploratory work — **not merged as a long-term direction.**  
**Date:** June 2026  
**Context:** Lucy menu `[7]` (seven-layer CPU suite), Dense forward only.

This document captures everything built and learned during the **Plan 9 ASM dense forward** push and the **volumetric fused executor (v1)** experiment. Copy this file before deleting the exploration code; the complexity-to-gain ratio did not justify keeping it in-tree.

**Related existing docs:** [`poly/asm/README.md`](../poly/asm/README.md), [`bitnet_cpu.md`](bitnet_cpu.md), [`dispatch.md`](dispatch.md).

---

## Executive summary

### What we tried

1. **Native-quant ASM dense forward** — BitNet-style W8A8: morphed weights stay in native storage, activations quantized per row to int8, integer dot in Plan 9 asm, one output scale per row. **No** expanding morphed weights to float32 for matmul on the ASM path.
2. **BitNet I2_S scaffolding** — Microsoft `ggml-bitnet-mad.cpp` I2_S pack + scalar/asm row dots for ternary.
3. **Volumetric executor v1** — `BuildDenseExecPlan` + `ForwardDenseExecutor`: skip `DispatchLayer` per cell×layer, pre-plan dtype→kernel, morph weights once, direct dense kernel calls.

### What we learned

| Finding | Detail |
| :--- | :--- |
| ASM wins on **fat layers** | 1×1×1 with pyramid dims (16→64): quant native ASM ~**1.5–2.4×** vs Go float-dequant tiled path |
| ASM **loses on thin grids** | 3×3×3 with `flatEndpoints(4)` (4×4 micro-layers): ASM ~**0.5–0.93×** vs Go — setup + 189 tiny GEMVs dominate |
| Executor v1 saves **~15–38%** | Skipping interpret dispatch on 3×3×3 dense: consistent **1.15–1.38×** (Interp/Exec↑), perfect parity |
| Dispatch is **not** the main 3³ problem | Executor proves interpret tax ≈ **25%** of forward; remaining **~75%** is per-layer 4×4 work |
| Quant does **not** win bigger via executor | Int8/Int4 gains ≈ Float32 — no special breakout |
| **Not worth the complexity** | Many files, dual parity paths, Lucy table sprawl, MC races, dtype routing — for ~20–35% on a narrow CPU interpret path |

### Decision

Archive the exploration. Keep the **interpreter** (`ForwardPolymorphic` + `DispatchLayer`) as the research path. Peak inference belongs on **GPU** (`poly/wgpu_forward.go`) or a future **batched executor** that fuses matmul across cells — not per-hop ASM + per-hop dispatch trimming.

---

## Part I — Architecture before and after

### Two tiers (intended end state)

```
┌─────────────────────────────────────────────────────────────┐
│  Tier 1 — Interpreter (keep)                                │
│  ForwardPolymorphic → spatial tile loops → DispatchLayer    │
│  Build, train, parity, volumetric experiments               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Tier 2 — Executor (explored, v1 only)                      │
│  Plan at load: dtype → kernel map, pooled buffers           │
│  Fused visit order, few CPU entry points                    │
│  v1: no cross-cell matmul batching yet                      │
└─────────────────────────────────────────────────────────────┘
```

### Interpret path (baseline)

```
ForwardPolymorphic (forward.go)
  → spatial 4×4×4 tile loops (or classic z→y→x→l)
  → DispatchLayer (jump table)
  → DenseForwardPolymorphic (dense.go)
       → asm: denseForwardAsm
       → bitnet: DenseForwardBitNetNativeQuantCPU
       → default: DenseForwardTiled (float dequant via denseActiveWeights)
```

Lucy `[7]` **Go column** for quant dtypes timed `DenseForwardTiled` (float dequant), **not** native BitNet — so ASM-vs-Go tables compare different semantics unless you read the captions.

### ASM dense path

```
DenseForwardPolymorphic
  └─ layerUseAsmForward && asm.Enabled()
       └─ denseForwardAsm (dense_asm.go)
            ├─ ternary: denseForwardAsmBitNetTernary → I2S asm if available
            ├─ quant / low-bit float morph:
            │    denseForwardAsmNativeQuant (dense_asm_native_quant.go)
            │    · morphed []uint8 weights (WeightStore.Morph)
            │    · per-row int8 activation quant (bitNetQuantizeActivationNumeric)
            │    · dot.U8BytesTileNativeI64 / packed nibble / 2-bit / binary paths
            │    · one output scale per row → float preAct
            └─ Float32/64: denseForwardAsmByDType → asm/matmul GEMV
```

**Key principle:** weights use proper native/entity layouts; forward does not re-morph from FP32 on every hop if inference sync already populated `Versions[dtype]`.

### Executor v1 path

```
BuildDenseExecPlan(net)
  · forwardVisitLayerIndices — same order as ForwardPolymorphic
  · verify all visited layers are LayerDense
  · prepDenseExecWeights — Morph once per layer
  · pickDenseExecKernel — dtype → DenseExecKernel enum

ForwardDenseExecutor(net, plan, input)
  · for step in plan.Steps:
       denseExecForwardStep(kernel, layer, current)  // no DispatchLayer
  · kernels mirror DenseForwardPolymorphic (no ASM): Tiled / BitNet native / I2S try
```

**What v1 did *not* do:** fuse tiles under the grid, batch 27 cells into one GEMV, or call ASM from the executor loop.

---

## Part II — ASM exploration (detail)

### Goals

- Hand-written **Plan 9 assembly** (`*.s`, not CGO) on **amd64** and **arm64**.
- **21 dtypes** on Dense forward via `UseAsmForward` / `VolumetricNetwork.SetAsmForward`.
- **Native integer dots** — multiply/accumulate in storage width; dequant once at boundary.
- **BitNet W8A8** reference in Go (`DenseForwardBitNetNativeQuantCPU`) for quant parity vs ASM.
- Lucy `[7]` timing: Go tiled vs ASM, all dtypes when `ForwardCapable`.

### Package layout (`poly/asm/`)

| Package | Role |
| :--- | :--- |
| `asm/dot/` | f32/f64 dots, native int dots (`native_int_*.s`), packed row dots (`native_packed_*.s`) |
| `asm/matmul/` | Tiled GEMV, `OverOutputTiles` MC, wires dot into tile callbacks |
| `asm/dense/` | Thin float forward entry for poly |
| `asm/bitnet/` | `rowDotI2SI8`, `ternaryWordDot16` (scalar asm stubs on arm64/amd64) |

### BitNet I2_S (`poly/bitnet/`)

Reference: Microsoft `ggml-bitnet-mad.cpp` (cloned under `loom/BitNet/` during exploration).

| Constant / type | Meaning |
| :--- | :--- |
| `QK_I2S = 128` | Block size; **cols must be `% 128 == 0`** for I2S matrix |
| `I2SMatrix` | Row-major ternary in ACT_PARALLEL layout |
| Codes `0,1,2` | → weights `-1, 0, +1` |
| `RowDotI2SI8Go` | Go scalar reference |
| ASM | `poly/asm/bitnet/i2s_amd64.s`, `i2s_arm64.s` — **not** full AVX2 `maddubs` loop from Microsoft |

**Not ported:** full AVX2 I2_S inner loop from `BitNet/src/ggml-bitnet-mad.cpp`.

### BitNet W8A8 native quant flow

Per output row:

1. Read morphed weight row from `WeightStore.Versions[dtype]` (uint8 storage).
2. Quantize input row to int8 (`bitNetQuantizeActivationNumeric`).
3. Integer dot (Go ref or `dot.U8BytesTileNativeI64` in asm).
4. `outScale = weightScale * actMax / 127` → float pre-activation.
5. Apply layer activation.

Dtype-specific branches in `denseForwardAsmNativeQuant`: wide int16/32/64, FP8 morph, nibble Int4, 2-bit Int2/Ternary, binary.

### Poly files (ASM exploration)

| File | Role |
| :--- | :--- |
| `poly/dense_asm.go` | `SetAsmForward`, `denseForwardAsm` routing |
| `poly/dense_asm_dtypes.go` | Dtype → asm matmul dispatch |
| `poly/dense_asm_native.go` | Legacy native width paths |
| `poly/dense_asm_native_matmul.go` | Native tiled matmul wiring |
| `poly/dense_asm_native_quant.go` | BitNet W8A8 asm path |
| `poly/dense_bitnet_w8a8_cpu.go` | Go scalar BitNet reference |
| `poly/bitnet_cpu.go` | Ternary word dot delegates to asm |
| `poly/bitnet_i2s_cpu.go` | `GetBitNetI2SMatrix`, `DenseForwardBitNetI2SCPU` |
| `poly/bitnet/i2s.go` | I2_S pack + Go row dot |
| `poly/bitnet/ternary_u32.go` | `TernaryWordDot16Go` |
| `poly/dense.go` | `layerUseBitNetNativeQuant`, `DenseForwardPolymorphic` routing |

### Network flags (`poly/poly.go`)

```go
UseAsmForward         bool  // Plan 9 assembly CPU kernels
UseBitNetNativeQuant  bool  // Dense: BitNet W8A8 Go ref (Lucy quant parity column)
```

### Bugs fixed during exploration

| Bug | Symptom | Fix |
| :--- | :--- | :--- |
| Lucy gated ASM timing on float-only determinism | Blank ASM columns for quant dtypes | Time ASM for all dtypes when `ForwardCapable` |
| Shared `xq`/`xu` buffers in `runRows` closure | MC parity ~1e19 on Float16 | Allocate quant buffers **inside** each `runRows` goroutine |
| `GetBitNetI2SMatrix` with cols=16 | Panic — I2_S needs cols % 128 | Guard / fallback to packed ternary |
| Duplicate `bitNetTernaryWordDotTail` | Build error | Cleaned `bitnet_cpu.go` |

### Lucy `[7]` ASM integration

**Files:** `lucy/examples/seven_layer/runner.go`, `summary.go`, `common.go`

- `requiresAsmGoTiledParity` — Float32/64: ASM vs Go tiled
- `requiresAsmBitNetParity` — quant: ASM vs `captureForwardBitNetNative` (Go BitNet ref)
- Table: `printDenseForwardAsmTimingTable` — Go SC/MC, ASM SC/MC, Go/Asm↑, |Δ|

**Captions:**

- Go = `DenseForwardTiled` (float dequant)
- ASM = native quant (BitNet W8A8) or float GEMV
- Go/Asm↑ = Go÷ASM (>1 means assembly wins)

### ASM benchmark findings (Lucy Dense)

#### 1×1×1 (pyramid 16→24→32→48→64→48→32→8)

| Category | Go/Asm↑ (approx) |
| :--- | :--- |
| Int8, FP8, Float16 | ~1.5–2.4× ASM wins |
| Int4, Binary, Ternary | ~0.5–1.0× (scalar packed dots; ternary may hit Go matvec) |
| Float32/64 | ~1.0–1.1× |

#### 3×3×3 (`flatEndpoints(4)` → 4×4 layers, 189 hops)

| Category | Go/Asm↑ (approx) |
| :--- | :--- |
| **Everything** | ~**0.5–0.93×** — ASM **slower** than Go tiled |

**Why 3³ is bad for ASM:**

- 27 cells × 7 layers = **189 serial dense forwards**
- Each layer is **4×4** GEMV — too small to amortize asm entry, quant buffers, tile setup
- Interpreter adds overhead too, but Go tiled float-dequant path is cheap per micro-op
- Native quant correctness is fine; **kernel economics** are wrong at this shape

### Tests added

| Test | Package |
| :--- | :--- |
| `dense_asm_parity_test.go` | Float → Go tiled; quant → BitNet native ref |
| `bitnet_i2s_test.go` | I2S row dot, ternary word dot, I2S forward smoke |
| `bitnet_cpu_test.go` | BitNet CPU helpers |

```bash
cd loom/poly
go test ./asm/... ./tests/ -run 'DenseAsm|BitNet|I2S' -count=1
```

---

## Part III — Volumetric executor (detail)

### Motivation

User goal: *"dtype-aware volumetric executor that fuses tiles under the grid, instead of re-interpreting every cell×layer on the way through."*

**v1 scope:** prove interpret tax measurable; skip `DispatchLayer`; pre-plan kernels. **Not** true tile fusion under the grid.

### API (`poly/volumetric_executor.go`)

```go
type DenseExecKernel uint8  // Tiled | BitNetNative | BitNetI2SThenTiled

type DenseExecStep struct {
    LayerIdx int
    Kernel   DenseExecKernel
}

type DenseExecPlan struct {
    Steps []DenseExecStep
}

func BuildDenseExecPlan(n *VolumetricNetwork) (*DenseExecPlan, error)
func ForwardDenseExecutor(n, plan, input) (*Tensor[float32], time.Duration)
```

`ErrNotDenseOnly` if any visited layer is not `LayerDense`.

### Lucy `[7]` executor table

Printed after ASM table for Dense suites: **Interpret vs Fused Executor**.

| Column | Meaning |
| :--- | :--- |
| Interp SC/MC | `ForwardPolymorphic` + `DispatchLayer` (CPU Go, no ASM) |
| Exec SC/MC | `ForwardDenseExecutor` with pre-built plan |
| Interp/Exec↑ | Interpret÷Executor (>1 = executor wins) |
| \|Δ\| | max \|Interpret − Executor\| — parity |

**Files:** `runner.go` (capture), `common.go` (`benchmarkExecutorForward`), `summary.go` (`printDenseExecutorTimingTable`), `dense_executor_3x3_test.go` (smoke).

### Measured results — Dense 3×3×3 (5 passes, user run)

```
╔══════════════════════════════════════════════════════════════════════╗
║  Dense 3×3×3 — forward: Interpret vs Fused Executor (avg of 5 passes) ║
╚══════════════════════════════════════════════════════════════════════╝

| DType      | Interp SC  | Exec SC    | Interp/Exec↑ | |Δ| SC    | Interp MC  | Exec MC    | Interp/Exec↑ | |Δ| MC    |
| Float64    | 31.4µs     | 25.9µs     | 1.21×        | 0.00e+00  | 152.8µs    | 131.0µs    | 1.17×        | 0.00e+00  |
| Float32    | 31.2µs     | 22.6µs     | 1.38×        | 0.00e+00  | 136.1µs    | 140.1µs    | 0.97×        | 0.00e+00  |
| Float16    | 38.7µs     | 32.9µs     | 1.18×        | 0.00e+00  | 152.9µs    | 121.9µs    | 1.25×        | 0.00e+00  |
| BFloat16   | 36.5µs     | 27.4µs     | 1.33×        | 0.00e+00  | 136.0µs    | 122.7µs    | 1.11×        | 0.00e+00  |
| FP8-E4M3   | 38.0µs     | 32.9µs     | 1.15×        | 0.00e+00  | 145.8µs    | 120.9µs    | 1.21×        | 0.00e+00  |
| FP8-E5M2   | 38.5µs     | 33.2µs     | 1.16×        | 0.00e+00  | 151.4µs    | 131.0µs    | 1.16×        | 0.00e+00  |
| Int64      | 36.0µs     | 30.4µs     | 1.19×        | 0.00e+00  | 139.2µs    | 114.8µs    | 1.21×        | 0.00e+00  |
| Uint64     | 36.2µs     | 29.7µs     | 1.22×        | 0.00e+00  | 144.7µs    | 127.7µs    | 1.13×        | 0.00e+00  |
| Int32      | 37.1µs     | 30.0µs     | 1.24×        | 0.00e+00  | 159.2µs    | 126.7µs    | 1.26×        | 0.00e+00  |
| Uint32     | 36.4µs     | 29.9µs     | 1.22×        | 0.00e+00  | 132.2µs    | 129.3µs    | 1.02×        | 0.00e+00  |
| Int16      | 36.7µs     | 28.0µs     | 1.31×        | 0.00e+00  | 140.5µs    | 114.2µs    | 1.23×        | 0.00e+00  |
| Uint16     | 35.9µs     | 29.2µs     | 1.23×        | 0.00e+00  | 131.4µs    | 117.9µs    | 1.11×        | 0.00e+00  |
| Int8       | 38.7µs     | 30.8µs     | 1.26×        | 0.00e+00  | 141.3µs    | 111.1µs    | 1.27×        | 0.00e+00  |
| Uint8      | 38.1µs     | 27.8µs     | 1.37×        | 0.00e+00  | 155.2µs    | 133.0µs    | 1.17×        | 0.00e+00  |
| Int4       | 36.8µs     | 30.3µs     | 1.21×        | 0.00e+00  | 141.9µs    | 114.9µs    | 1.23×        | 0.00e+00  |
| Uint4      | 40.4µs     | 32.1µs     | 1.26×        | 0.00e+00  | 143.7µs    | 125.6µs    | 1.14×        | 0.00e+00  |
| FP4        | 36.9µs     | 28.9µs     | 1.28×        | 0.00e+00  | 130.8µs    | 115.4µs    | 1.13×        | 0.00e+00  |
| Int2       | 38.7µs     | 30.6µs     | 1.27×        | 0.00e+00  | 145.5µs    | 110.2µs    | 1.32×        | 0.00e+00  |
| Uint2      | 37.4µs     | 29.4µs     | 1.27×        | 0.00e+00  | 151.8µs    | 122.7µs    | 1.24×        | 0.00e+00  |
| Ternary    | 54.9µs     | 73.5µs     | 0.75×        | 0.00e+00  | 56.6µs     | 44.0µs     | 1.29×        | 0.00e+00  |
| Binary     | 37.3µs     | 31.6µs     | 1.18×        | 0.00e+00  | 138.7µs    | 126.8µs    | 1.09×        | 0.00e+00  |

Best Interp/Exec↑ SC: Float32 at 1.38×  |  Best MC: Int2 at 1.32×
```

**Anomaly:** Ternary SC executor **slower** (0.75×) while MC faster (1.29×) — same |Δ|=0; likely kernel path + 5-pass noise. Worth noting if revisiting.

### 1×1×1 executor (reference)

Most dtypes ~1.0–1.1× — fat matmul dominates; dispatch tax invisible.

### When executor wins (rule of thumb)

```
noticeability ∝ (dispatch hops) / (compute per hop)
```

| Workload | Executor feel |
| :--- | :--- |
| Dense 3×3×3, width 4 | **Noticeable** (~1.2–1.4×) |
| Dense 1×1×1, pyramid | **Barely** (~1.0–1.1×) |
| MHA / SwiGLU / CNN | **Less** — heavy per-hop compute |
| Training | **None** — executor forward-only |
| GPU path | **Different tier** — `wgpu_forward.go` |

---

## Part IV — Lessons learned

### 1. Interpret dispatch is real but bounded

Executor v1 isolates ~**20–35%** overhead from `DispatchLayer` + routing on 3×3×3 dense. That is **not** enough to explain ASM being **0.5×** on the same grid — the bottleneck is **189 × tiny GEMV**, not the jump table alone.

### 2. ASM economics need layer mass

Native quant ASM needs enough FLOPs per call to amortize:

- Plan 9 entry
- Per-row activation quant buffers
- Tile orchestration
- Output scale + activation

4×4 on 3³ fails; 16×48 on 1³ succeeds.

### 3. Compare apples to apples in benchmarks

Lucy Go column used float-dequant tiled forward for quant dtypes; ASM used native BitNet. Parity was split (`requiresAsmGoTiledParity` vs `requiresAsmBitNetParity`) but the **timing** comparison was semantically asymmetric.

### 4. Executor v1 ≠ "fused volumetric"

True fusion means e.g. **one batched GEMV per layer-index across all 27 cells**, pooled activations, ASM called once per slab — not 189 direct kernel calls with dispatch removed.

### 5. Complexity inventory

- Dual forward paths (interpret / executor / asm / bitnet ref)
- Per-dtype asm branches in Go + `.s` stubs per arch
- Lucy table proliferation (ASM + executor + determinism + save/reload)
- MC threading bugs in quant buffer pooling
- I2S layout constraints (cols % 128)

**Gain:** ~1.3× on a narrow CPU interpret forward for one layer type on one grid shape.

**Verdict:** Good experiment, wrong ROI to maintain.

---

## Part V — What v2 would have required

If revisiting (e.g. after GPU path matures):

1. **Cross-cell batching** — for each `l` in `0..LayersPerCell-1`, stack 27 cell activations → one `batch×4` GEMV (or wider).
2. **ASM inside executor** — single plan entry point; no `DispatchLayer` and no per-layer asm re-entry.
3. **Buffer pool** — `xq`/`xu`/output tiles allocated per frame, not per row batch.
4. **Full I2_S SIMD** — port AVX2/NEON from Microsoft BitNet for ternary at scale.
5. **GPU executor tier** — mirror plan on device; `BeginFrame` + block dispatches (partial precedent in `wgpu_forward.go`).

---

## Part VI — File inventory (exploration code to delete)

Use `git status` / `git log` for the exact commit set. Approximate list:

### `poly/` — executor

- `volumetric_executor.go`
- `tests/dense_executor_test.go`

### `poly/` — BitNet W8A8 + I2S (if removing entire exploration)

- `dense_asm.go`, `dense_asm_dtypes.go`, `dense_asm_native.go`, `dense_asm_native_matmul.go`, `dense_asm_native_quant.go`
- `dense_bitnet_w8a8_cpu.go`
- `bitnet_cpu.go`, `bitnet_i2s_cpu.go`
- `bitnet/` (package)
- `tests/dense_asm_parity_test.go`, `tests/bitnet_i2s_test.go` (and related)

### `poly/asm/bitnet/`

- `dot.go`, `dot_decl.go`, `dot_stub.go`
- `i2s_amd64.s`, `i2s_arm64.s`
- `ternary_amd64.s`, `ternary_arm64.s`

### `lucy/examples/seven_layer/`

- Executor fields in `summary.go` (`DTypeRow` executor columns, `printDenseExecutorTimingTable`)
- Executor capture in `runner.go`, `benchmarkExecutorForward` in `common.go`
- `dense_executor_3x3_test.go`

### Revert / keep decisions

| Keep | Remove (exploration-only) |
| :--- | :--- |
| `poly/asm/dot`, `matmul`, `dense` if predating this work and used elsewhere | `volumetric_executor.go` + Lucy executor table |
| `ForwardPolymorphic`, `DispatchLayer` | BitNet I2S asm if unused |
| `wgpu_forward.go` GPU path | `dense_asm_native_quant.go` if reverting to pre-BitNet asm |
| `docs/bitnet_cpu.md` (older ternary doc) | This archive doc **keep** |

**Note:** `poly/asm/README.md` may describe paths you're removing — update or revert that README when deleting code.

---

## Part VII — How to reproduce (before delete)

```bash
# Poly parity
cd loom/poly
go test ./tests/ -run 'DenseExecutor|DenseAsm|BitNet' -count=1

# Lucy Dense 1³ regression (21 dtypes + ASM + executor tables)
cd loom/lucy/examples/seven_layer
go test -run TestRunLayerSuiteDense1x1AllDTypes -v -count=1

# Lucy 3³ executor smoke
go test -run TestDenseExecutor3x3Float32Smoke -v -count=1

# Full menu
cd loom/lucy && go run .   # → [7] Dense
```

---

## Closing

The exploration validated:

- Native quant ASM **can** beat float-dequant Go on **fat** dense layers.
- It **cannot** rescue **thin volumetric grids** without batching.
- Skipping dispatch saves **~25%**, not **2×**.
- A real volumetric performance story needs **fusion below the grid**, not more interpreter variants.

Copy this document, then delete the code. The interpreter + GPU paths remain the production-shaped architecture.
