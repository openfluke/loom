# Lucy Bloom Rivers

Architecture shorthand for a Loom stack that combines **multi-region volumetric** layout, **bicameral** train vs run, **discrete-time stepping** (`step.go`), and **streaming** inference on the outside — summarized as **MRBiVS** (**M**ulti-**R**egion · **Bi**cameral · **V**olumetric · **S**tep).

**Loom version:** **0.80.0 "Native Ship"** — see [`../docs/v080_release.md`](../docs/v080_release.md).

---

## Quick start

```bash
cd lucy
go mod tidy          # pulls github.com/openfluke/webgpu@v1.0.4
go run .
```

**GPU (Linux NVIDIA example):**

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export WGPU_ADAPTER_NAME=NVIDIA
go run .
```

Requires **Go 1.26.2+** (`GOTOOLCHAIN=auto` if your system Go is older).

---

## Menu

| # | Mode | What it does |
|---|------|----------------|
| **1** | **Poly Talk** | Chat from HuggingFace cache (safetensors each run); optional GPU, Q4, block-wise upload |
| **2** | Tests | Dense mid-stream adaptation benchmark |
| **3** | Layer testing | CPU/GPU suites → `lucy_testing_output/` |
| **4** | Download | Approved HF models (SoulGlitch-style) |
| **5** | Forward benchmark | BitNet b1.58 CPU: normal vs stepped vs pipeline |
| **6** | Five-layer examples | Per-layer `.go` tutorials |
| **7** | Seven-layer CPU suite | 10 layer types × 21 dtypes × SC/MC/ASM · JSON **and** `.entity` save/reload |
| **8** | **ENTITY Talk** | HF cache → `.entity` convert (optional Q4 bake) → GPU chat without runtime safetensors |

Native checkpoints: [`docs/entity.md`](../docs/entity.md). Converted models land in [`lucy_entities/`](lucy_entities/).

---

## Letter expansion

| Word | Letters | Meaning |
|------|---------|---------|
| **Lucy** | — | Spoken handle only (no MRBiVS letters required here). |
| **Bloom** | **B**, **M** | **Bi**cameral · **M**ulti-region mesh |
| **Rivers** | **R**, **i**, **V**, **S** | **R**outing / regional links · **i** completes **Bi** (with **B** from Bloom) · **V**olumetric grid · **S**tep mesh + streaming |

### Initialisms

**L.U.C.Y.** — *Lattice Unified Clock Yoked-net.*

**B.L.O.O.M.** — *Bicameral Loom Open-grid Orchestration Multi-region.*

**R.I.V.E.R.S.** — *Routed In Volumetric Engines Rhythmically Stepping.*

---

## Architecture

- **Volumetric network** — Grid of layers (`VolumetricNetwork`), not just depth stacked one way. Multi-region layouts: branches, combine modes, optional remote regional links (e.g. `glitch/measure/regional_mix`).

- **Bicameral** — Train vs run hemispheres with periodic mirror/sync (e.g. `glitch/systolic_demo_bicameo`).

- **Step mesh** — Inner state advances in ticks: `StepState`, `StepForward` in [`poly/step.go`](../poly/step.go); see [`docs/step.md`](../docs/step.md).

- **Streaming decode** — Outer loop can stay standard autoregressive / KV-style; mesh stepping is the inner temporal loop.

- **KV cache** — Ordinary attention cache where used; align with mesh ticks per design.

---

## Test output

### Logs

| Log | Contents |
|-----|----------|
| [`lucy_testing_output/seven_layer.txt`](lucy_testing_output/seven_layer.txt) | Menu **[7]** seven-layer CPU suite — reset each run |
| [`lucy_testing_output/log.txt`](lucy_testing_output/log.txt) | Older layer-matrix runs |

Run from repo root: build `lucy`, open the menu, choose **[7]** (or **[0]** for all layer types). Example: `cd lucy && go run .` → `[7]` → `[0]`.

Harness: [`examples/seven_layer/`](../lucy/examples/seven_layer/) — JSON build, **21 numerical types**, **CPU single-core (SC)** vs **multi-core (MC)**, **ASM** (Dense forward only), 50-epoch train, save/reload before and after train.

Symbol legend: **✓** PASS · **✗** FAIL · **·** N/A (not implemented for this layer).

Broader testing notes (H-DRIFT buckets, legacy matrices): [`docs/testing_and_validation.md`](../docs/testing_and_validation.md). **v0.79.0 Bedrock Validation** (MHA/KV, native save, C-ABI): [`docs/bedrock_validation.md`](../docs/bedrock_validation.md). **v0.80.0 Native Ship** (ENTITY, WebGPU v1.0.4, GPU validation): [`docs/v080_release.md`](../docs/v080_release.md).

---

### What each check measures

The suite runs **separate** checks for forward, backward, training, and weights. SC / MC / ASM are **execution variants**, not separate dtypes.

| Area | Check | SC | MC | ASM | Pass criterion |
|------|--------|----|----|-----|----------------|
| **Forward** | Output parity | `EnableMultiCoreTiling=false` | `true` | Dense + float dtypes only: Go tiled vs `UseAsmForward` | Max abs diff vs dtype tolerance (`Fwd SC↔MC`, `Go↔ASM` in log) |
| **Forward** | Timing | 25-pass avg | 25-pass avg | — | Informational only (not gated on **Overall**) |
| **Backward** | Gradient parity | SC | MC | Not implemented | Max abs diff on stacked `dx`+`dw` (`Bwd SC↔MC`; 10× fwd tol) |
| **Backward** | Timing | 25-pass avg | 25-pass avg | — | Informational only |
| **Training** | Loss decrease | `TrainingModeCPUSC` (runs) | `TrainingModeCPUMC` (reported loss) | — | `trainingOK` on MC loss init→final |
| **Training** | Timing | SC wall time | MC wall time | — | Informational only |
| **Weights** | Save/reload **before** train | — | — | — | JSON **and** `.entity` serialize → deserialize → forward + native blob match (`B-OK`) |
| **Weights** | Save/reload **after** MC train | — | — | — | Same on trained net (`A-OK`, `Native`); `.entity` ~25% smaller than JSON on disk |

**FP32 Master lifecycle (`ReleaseFP32MasterWhenIdle`):**

| Phase | RAM weights | What happens |
|-------|-------------|--------------|
| **Load / first mount** | Native `Versions` at layer `DType` (Int8, Binary, …) | JSON init → `Morph`; optional `SyncInferenceWeights()` drops Master |
| **Forward / infer** | Native only | `GetActive(dtype)`; no FP32 duplicate |
| **Training** | Native + **FP32 Master** mounted | `Train()` → `EnsureTrainingWeights()`; grads update Master; native `Versions` refreshed (native train path) |
| **After training** | Native only again (when flag true) | `Train()` ends with `SyncInferenceWeights()` — morph from Master, then **release** Master |

Menu `[7]`: forward path uses `(infer)` weight bytes; after MC train, `(trained-native)` — same width as checkpoint on disk. Set `ReleaseFP32MasterWhenIdle = false` to keep Master resident (legacy / further training in-process).
| **Overall** | Gate | — | — | — | `B-OK` ∧ `A-OK` ∧ **Learn** ∧ **Det** (ASM reported; not required except inside **Det** for float Dense) |

**Det** = forward SC↔MC ∧ backward SC↔MC ∧ (for Float64/32/16/BF16 on Dense: Go↔ASM). Non-Dense layers: ASM column is **·**; toggling `UseAsmForward` must not change outputs.

**Numerical types in this suite (not full native tensor math):**

| Tensor | Storage | Forward / backward compute |
|--------|---------|----------------------------|
| **Weights** | Morphed `Versions` per layer `DType` + float32 `Master` | Loaded via `GetActive(dtype)`; MAC mostly **float32** on cast weights (Dense **ASM** = native integer matmul for many quant dtypes) |
| **Activations** | Always **float32** | `ForwardPolymorphic` / `Train` batches |
| **Gradients** | float32 | `ApplyGradientsNative` when `UseExactDType` |

---

### Session manifest (from latest `seven_layer.txt`)

**210 / 210** dtype checks passed · **10 / 10** layer types fully green (latest `seven_layer.txt`).

| Layer | Passed | Failed | Total | All dtypes OK |
|-------|--------|--------|-------|---------------|
| Dense | 21 | 0 | 21 | ✓ |
| SwiGLU | 21 | 0 | 21 | ✓ |
| MHA | 21 | 0 | 21 | ✓ |
| CNN1 | 21 | 0 | 21 | ✓ |
| CNN2 | 21 | 0 | 21 | ✓ |
| CNN3 | 21 | 0 | 21 | ✓ |
| RNN | 21 | 0 | 21 | ✓ |
| LSTM | 21 | 0 | 21 | ✓ |
| Embedding | 21 | 0 | 21 | ✓ |
| Residual | 21 | 0 | 21 | ✓ |

---

### Per-layer results (21 numerical types)

Latest full run: **210/210 PASS** — see [`lucy_testing_output/seven_layer.txt`](lucy_testing_output/seven_layer.txt) for live tables. Snapshot tables below may lag; re-run menu `[7]` → `[0]` after changes.

Columns: **Fwd SC↔MC** / **Bwd SC↔MC** (determinism), **Go↔ASM**, **Train**, **Save before** / **Save after**, **Native** persistence, **Overall**.

#### Dense — 21/21 ✓

| DType | Fwd SC↔MC | Bwd SC↔MC | Go↔ASM | Train | Save before | Save after | Native | Overall |
|-------|-----------|-----------|--------|-------|-------------|------------|--------|---------|
| Float64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Float32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Float16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| BFloat16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP8-E4M3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP8-E5M2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ternary | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Binary | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

#### SwiGLU — 18/21 · MHA — 18/21

Identical matrix: **Int4**, **Int2**, **Ternary** fail **Save before/after**; all forward/backward determinism ✓; ASM **·**.

| DType | Fwd SC↔MC | Bwd SC↔MC | Go↔ASM | Train | Save before | Save after | Native | Overall |
|-------|-----------|-----------|--------|-------|-------------|------------|--------|---------|
| Float64 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Float32 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Float16 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| BFloat16 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP8-E4M3 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP8-E5M2 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int64 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint64 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int32 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint32 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int16 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint16 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int8 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint8 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int4 | ✓ | ✓ | · | ✓ | ✗ | ✗ | ✓ | ✗ |
| Uint4 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP4 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int2 | ✓ | ✓ | · | ✓ | ✗ | ✗ | ✓ | ✗ |
| Uint2 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ternary | ✓ | ✓ | · | ✓ | ✗ | ✗ | ✓ | ✗ |
| Binary | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |

#### CNN1 — 13/21

| DType | Fwd SC↔MC | Bwd SC↔MC | Go↔ASM | Train | Save before | Save after | Native | Overall |
|-------|-----------|-----------|--------|-------|-------------|------------|--------|---------|
| Float64 | ✓ | ✓ | · | ✗ | ✓ | ✓ | ✓ | ✗ |
| Float32 | ✓ | ✓ | · | ✗ | ✓ | ✓ | ✓ | ✗ |
| Float16 | ✓ | ✓ | · | ✗ | ✗ | ✗ | ✓ | ✗ |
| BFloat16 | ✓ | ✓ | · | ✗ | ✗ | ✗ | ✓ | ✗ |
| FP8-E4M3 | ✓ | ✓ | · | ✗ | ✗ | ✗ | ✓ | ✗ |
| FP8-E5M2 | ✓ | ✓ | · | ✗ | ✗ | ✗ | ✓ | ✗ |
| Int64 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint64 | ✓ | ✓ | · | ✗ | ✓ | ✓ | ✓ | ✗ |
| Int32 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint32 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int16 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint16 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int8 | ✓ | ✓ | · | ✗ | ✗ | ✗ | ✓ | ✗ |
| Uint8 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int4 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint4 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP4 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int2 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint2 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ternary | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Binary | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |

#### CNN2 — 21/21 ✓ · CNN3 — 21/21 ✓

All 21 dtypes: every column ✓ except **Go↔ASM** (**·**). See log for per-dtype timing and memory tables.

#### RNN — 15/21

| DType | Fwd SC↔MC | Bwd SC↔MC | Go↔ASM | Train | Save before | Save after | Native | Overall |
|-------|-----------|-----------|--------|-------|-------------|------------|--------|---------|
| Float64 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Float32 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Float16 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| BFloat16 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP8-E4M3 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP8-E5M2 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int64 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint64 | ✓ | ✓ | · | ✗ | ✓ | ✓ | ✓ | ✗ |
| Int32 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint32 | ✓ | ✓ | · | ✗ | ✓ | ✓ | ✓ | ✗ |
| Int16 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint16 | ✓ | ✓ | · | ✗ | ✓ | ✓ | ✓ | ✗ |
| Int8 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint8 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int4 | ✓ | ✓ | · | ✓ | ✗ | ✗ | ✓ | ✗ |
| Uint4 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP4 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int2 | ✓ | ✓ | · | ✓ | ✗ | ✗ | ✓ | ✗ |
| Uint2 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ternary | ✓ | ✓ | · | ✓ | ✓ | ✗ | ✓ | ✗ |
| Binary | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |

#### LSTM — 19/21

| DType | Fwd SC↔MC | Bwd SC↔MC | Go↔ASM | Train | Save before | Save after | Native | Overall |
|-------|-----------|-----------|--------|-------|-------------|------------|--------|---------|
| Float64 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Float32 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Float16 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| BFloat16 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP8-E4M3 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP8-E5M2 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int64 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint64 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int32 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint32 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int16 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint16 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int8 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint8 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int4 | ✓ | ✓ | · | ✓ | ✗ | ✗ | ✓ | ✗ |
| Uint4 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP4 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int2 | ✓ | ✓ | · | ✓ | ✗ | ✗ | ✓ | ✗ |
| Uint2 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ternary | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Binary | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |

#### Embedding — 17/21 · Residual — 17/21

Float dtypes: forward/backward determinism and save/reload ✓, but **Train ✗** (flat loss — Residual has no weights; Embedding float tables do not meet `trainingOK`). Quant dtypes: all ✓.

| DType | Fwd SC↔MC | Bwd SC↔MC | Go↔ASM | Train | Save before | Save after | Native | Overall |
|-------|-----------|-----------|--------|-------|-------------|------------|--------|---------|
| Float64 | ✓ | ✓ | · | ✗ | ✓ | ✓ | ✓ | ✗ |
| Float32 | ✓ | ✓ | · | ✗ | ✓ | ✓ | ✓ | ✗ |
| Float16 | ✓ | ✓ | · | ✗ | ✓ | ✓ | ✓ | ✗ |
| BFloat16 | ✓ | ✓ | · | ✗ | ✓ | ✓ | ✓ | ✗ |
| FP8-E4M3 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP8-E5M2 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int64 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint64 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int32 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint32 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int16 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint16 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int8 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint8 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int4 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint4 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP4 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Int2 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Uint2 | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ternary | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| Binary | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ |

---

### SC / MC / ASM coverage summary

| Variant | Forward | Backward | Training | Notes |
|---------|---------|----------|----------|-------|
| **CPU SC** | Benchmark + parity vs MC | Benchmark + parity vs MC | 50 epochs | `EnableMultiCoreTiling=false` |
| **CPU MC** | Benchmark + parity vs SC | Benchmark + parity vs MC | 50 epochs (loss reported) | `EnableMultiCoreTiling=true` |
| **ASM** | Dense: Go vs ASM on F64/F32/F16/BF16 | — | — | `net.UseAsmForward` after JSON build; backward ASM not implemented |

In the latest full run, **Fwd SC↔MC** and **Bwd SC↔MC** passed for every dtype×layer combination; failures are dominated by **train** (CNN1 floats, Embedding/Residual floats, RNN unsigned) and **save/reload** on low-bit types (SwiGLU, MHA, RNN, LSTM).
