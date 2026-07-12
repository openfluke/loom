# Testing, validation, and Lucy logs

This page ties together **how we stress `poly/`**, where **artifacts land**, and how to read **parity tables** in captured logs (for example `lucy/lucy_testing_output/log.txt`).

---

## Where logs come from

The **Lucy** tree (`lucy/`) drives broad layer suites: forward/backward parity, training matrices, save/reload checks, and GPU timing tables. Typical transcripts:

| Log | Menu | Contents |
|-----|------|----------|
| `lucy/lucy_testing_output/log.txt` | Dense L1 / GPU parity / layer matrices | Forward/backward parity, ASM timers, GPU tables |
| `lucy/lucy_testing_output/seven_layer.txt` | **[7] Seven-layer CPU suite** | 10 layer types × 21 dtypes × 1³/2³/3³ grids, **SC/MC/SIMD** fwd+bwd+train, **JSON + `.entity` save/reload** |
| `lucy/lucy_testing_output/nine_layer.txt` | **[9] Intel NPU bridge** | 15 layers × FP32/FP16/INT8 × small/medium/large — Loom vs Intel CPU/NPU timing + drift manifest |

Per-dtype checkpoints are written under the same folder: `tag_DType.json` (debug lane) and `tag_DType.entity` (native lane). The memory table compares both file sizes side by side.

**Observed compression (full [7] run):** `.entity` averages **~28% smaller** than JSON across 546 dtype×suite rows; all `json=PASS entity=PASS`. Quant dtype (Int4 vs Float64) still dominates absolute size — ENTITY removes Base64 overhead, not topology JSON. Details and sample tables: [entity.md — observed compression](entity.md#size-vs-json--observed-compression-lucy-7).

Both files are meant for human review and regression diffing (adapter name, per-dtype rows, summary tallies).

**Seven-layer suite (v0.79+):** See [`bedrock_validation.md`](bedrock_validation.md) for what the harness gates (MHA layout, KV decode, native ternary save, C-ABI `SyncInferenceWeights`, **SC/MC/SIMD parity** on all seven compute layers). Run `cd lucy && go run .` → **[7]** or **[0]**.

External benchmark logs (amd64 / arm64, Float32 timing tables): `seven_layer_amd.txt`, `seven_layer_arm.txt` — summarized in [simd.md](simd.md#seven-layer-benchmark-results).

**GPU load memory timeline (Lucy [1] / [8]):** Enable *Measure memory during GPU load* at the prompt, or set `LOOM_MEMORY_HISTORY=1`. After load, Lucy prints a braille chart, sample table, and diagnosis (block release, sequential globals, peak host+gpu overlap). See [memory_history.md](memory_history.md).

**HF → `.entity` convert:** Block-wise import is visible in the terminal (one `Finished loading weights with prefixes` per decoder block). Convert is not charted yet; see [entity.md — convert memory](entity.md#hf--entity-convert-memory).

---

## How to read parity summary lines

Sections often end with a line shaped like:

```text
>> [Forward Parity] 84 Tests | 💎 42 | ✅ 24 | 🟨 0 | 🟠 0 | 🟤 18 | ❌ 0 | 💀 0
```

Rough meaning (exact thresholds live in the test harness, not duplicated here):

| Symbol | Typical meaning |
|--------|-----------------|
| **💎** | Exact / diamond-grade agreement within the tightest tolerance |
| **✅** | Pass within configured industry-grade tolerance |
| **🟨 / 🟠** | Elevated drift bands (still classified by the harness) |
| **🟤** | Heavy drift (e.g. **H-DRIFT** in backward tables) — worth investigating dtype + path |
| **❌** | Hard failure (assert or threshold breach) |
| **💀** | Fatal / panic / infrastructure failure |

Backward tables may label columns **INDUS** (industry tolerance) vs **H-DRIFT** (heavy drift). Treat **🟤** rows as “numerically alive but not interchangeable with FP32 reference at the same tolerance,” not necessarily as engine bugs: some combinations are expected to diverge when the reference path is float32-simulated and the subject path is true low-bit or integer-native.

---

## May 2026 full-suite snapshot (`log.txt`)

Recent **Run All Layer Tests** captures (Metal / arm64, ~2992 rows) show:

| Metric | Value |
|--------|--------|
| **Broken (❌)** | **0** |
| **Fatal / NaN (💀)** | **0** |
| Bit-exact (💎) | ~75% of classified rows |
| Heavy drift (🟤) | ~17% — mostly forward parity vs FP32 reference on native-int / low-bit paths |

**Fixes reflected in this run (vs earlier transcripts):**

- **Training matrix** — `File` / `RAM` columns print correctly (no `%!s(MISSING)`); every Dense training row **TrainOK PASS** and **Save/Reload PASS** for all 21 dtypes.
- **Save/Reload** — CNN1/2/3, Dense, Embedding, LSTM, MHA, Residual, RNN, SwiGLU each end with `[Save/Reload <layer>] PASS`.
- **Global manifest** — no hard failures across the full layer sweep.

**Still classified as 🟤 (not ❌):** Dense forward parity rows where CPU uses true integer/low-bit math and the harness compares to a float-shaped reference; CNN backward **H-DRIFT** on Float16/BFloat16/Int4 (GPU vs CPU reference). Treat as tolerance bands — see parity legend above.

---

## Dense forward ASM (Plan 9)

Lucy **Dense → Generic Layer Suite** prints **Go SC · Go MC · ASM SC · ASM MC · GPU SC · GPU MC** and speedup columns:

- **Go/Asm↑** = Go wall time ÷ ASM wall time (**> 1.0** = assembly wins).
- Toggle: `UseAsmForward` on the network/layer; kernels live under `poly/asm/` (see [`asm/README.md`](../poly/asm/README.md)).

**Latest Dense bench (8×1024→512, Metal host, from `log.txt`):**

| Highlight | Go/Asm↑ SC | Go/Asm↑ MC |
|-----------|------------|------------|
| Best single-core | **Uint8** ~**2.46×** | — |
| Best multi-core | — | **Uint4** ~**3.55×** |
| Strong quant MC | — | **Ternary** ~3.21×, **FP4** ~3.25×, **Binary** ~2.78×, **Int8** ~2.72× |
| Float32 | ~1.11× SC, ~1.00× MC (parity) | |
| Float64 | **&lt; 1×** (asm slower on this shape) | ~0.61× MC |

Low-bit and morphed-`uint8` paths benefit most from native integer dots in Plan 9. Float64 SC/MC still favors Go tiled matmul on the current tile sizes — tuning item, not a broken toggle.

**Backward / training:** Plan 9 **asm** (`poly/asm/`) is **Dense forward-only**. Plan 9 **SIMD** (`poly/simd/`) covers **forward + backward** on Dense, SwiGLU, MHA, CNN1–3, RNN, LSTM via `TrainingModeCPUSimd` / `SetSimdForwardRecursive`; see [simd.md](simd.md).

---

## Interpreting a real log (examples)

The following patterns show up in recent `log.txt` captures (Metal adapter, tiled CNN1 suite):

1. **CNN1 generic suite note** — The harness itself reminds you that generic CNN1 tests still include **simulated / PTQ fallback** where a dtype has no strict native path. For a **strict native-only** CPU/GPU/tiling audit, use the **Glitch** `layer_matrix` example (see Glitch docs / examples in-repo).

2. **Float64 on GPU forward** — CPU microseconds vs GPU milliseconds often look like a large “speedup ratio < 1×”; that is frequently **dispatch overhead dominating tiny work**, not a claim that FP64 GPU is slower than CPU math in the large-batch limit.

3. **Wide integer CNN1 backward** — **Int64 / Uint64 / Int32 / Uint32** rows may show **🟤 H-DRIFT** vs float reference in GPU backward parity: the harness compares against an FP32-shaped reference while the native path uses integer semantics — read those rows as **classification / tolerance**, not as “GPU kernel wrong.”

4. **Save/Reload after training** — On the **Dense** suite (May 2026 log), **Save/Reload PASS** for all 21 dtypes after training. Older CNN-only rows or pre-native-save builds may still show FAIL on specific combos; diff against current `persistence.go` (`Native: true` + per-layer `dtype`) before treating as open bugs.

5. **Uint CPU training** — **Uint64 / Uint32** (and sometimes **Uint16**) may show **TrainOK FAIL** on CPU-tiled modes while GPU modes **PASS**: that points at **CPU-side training / loss scaling** for unsigned paths, not at GPU correctness.

6. **Peak performance gap line** — The footer **PEAK PERFORMANCE GAP** (e.g. Dense Forward Float16) is a **headline ratio** from one worst row in the scan table; it is useful for spotting outliers, not as a single global quality score.

---

## Poly package: what the suites actually exercise

High-signal files and areas (not exhaustive):

| Area | Representative files |
|------|------------------------|
| Core types & dispatch | `poly.go`, `forward.go`, `backward.go`, `training.go` |
| Numerical morphing | `weights.go`, `quantization.go`, CNN/ dense / MHA polymorphic `*.go` |
| GPU / WebGPU | `wgpu_context.go`, `wgpu_forward.go`, `wgpu_kernels.go`, `wgpu_shaders.go`, `wgpu_softmax.go` |
| Tiling & tile size | `tile_detection.go`, `*_tiled*.go` paths in dense / CNN / MHA |
| Serialization | `serialization.go`, `persistence.go`, `safetensors.go` |
| Native layer matrix harness | `native_layer_matrix.go`, `native_matrix_builtin_hooks.go` |
| Telemetry | `tanhi.go`, hardware probes in `hardware.go` |

When you add a layer or dtype, extend **both** the Lucy (or Glitch) harness **and** this doc if the log format or tolerance bands change.

---

## Nine-layer Intel bridge (`nine_layer.txt`)

**Menu:** Lucy **[9]** → **[4]** (medium only) or **[5]** (full 90-cell matrix).  
**Guide:** [`accelerators.md`](accelerators.md) — architecture, dtype upload, offload policy.

### What the log exercises

Each cell = one layer type × one dtype (FP32 / FP16 / INT8) × one size tier (`small` / `medium` / `large`):

1. Build Loom network with that `dtype` on the layer.
2. **`SyncToAccel`** once — compile OpenVINO graph + bake `WeightStore` weights (MatMul/Conv/MHA only).
3. **`DispatchLayer`** forward — Loom CPU baseline vs Intel CPU vs Intel NPU.
4. Drift: Loom↔Intel output diff + Intel repeat-forward determinism.

### Timing columns

| Column | Meaning |
|--------|---------|
| **Loom CPU / Intel CPU / Intel NPU** | Median **infer** ms after compile (steady state) |
| **Spd CPU / Spd NPU** | Loom ÷ Intel — **&gt; 1** Intel faster, **&lt; 1** Intel slower |
| **Compile C / Compile N** | One-time `SyncToAccel` ms — **not** in infer column |

Intel can be slower than Loom when tensor work is tiny (NPU ~0.5 ms floor) or when Loom’s CPU path is sub-millisecond (norms, small softmax). That is expected — not a broken weight upload.

### Manifest block (bottom of log)

| Line | Healthy signal (Jun 2026 Fedora run) |
|------|--------------------------------------|
| Intel faster than Loom (CPU) | ~56/90 — wins on medium/large MAC |
| Intel faster than Loom (NPU) | ~36/90 — wins mainly large MAC |
| Intel infer repeat (CPU/NPU) | **90/90 💎 EXACT** — determinism OK |
| Loom↔Intel parity ≤ INDUS | CPU ~23/90, NPU ~61/90 (looser NPU tolerance) |

**Parity buckets:** 💎 EXACT · ✅ INDUS · 🟨 LOWBIT · 🟤 H-DRIFT · ❌ BROKE · 💀 FATAL — same legend as seven-layer tables.

**Known ❌ BROKE rows:** LayerNorm / RMSNorm FP32/FP16 (~1.8 drift) — Intel uses fixed CABI graphs, not Loom `WeightStore`. **INT8 MAC** large-tier drift 3–36 — Loom dequant matmul vs OV f32 + NPU dynamic quant.

### Dtype / integration status (read with the log)

- **Weight upload:** FP32, FP16, INT8 on MatMul / Conv / MHA — ✅ at init (`LayerWeightBytesForAccel`).
- **Forward infer:** activation bytes only per hop — ✅.
- **“Integration done”?** Forward **bridge** yes (experimental); product NPU toggle, all dtypes, norms, backward — **no**. See accelerators.md “Is the integration done?”.

---

## Related commands (developer workflow)

Exact entrypoints move with refactors; prefer:

- `lucy/README.md` — MRBiVS stack and pointers into `poly/`.
- `poly/README.md` — version checklist and capability matrix.
- `welvet/cabi/internal/check/` — C-ABI vs `poly/` export parity scanner (Go); expect **461/461 (100%)** after v0.79 (`LoomSyncInferenceWeights`).

---

## See also

- [bedrock_validation.md](bedrock_validation.md) — v0.79.0 seven-layer suite, MHA/KV, C-ABI  
- [numerical_types.md](numerical_types.md) — DType list and `WeightStore` lifecycle  
- [gpu.md](gpu.md) — WebGPU context and dispatch overview  
- [serialization.md](serialization.md) — Save/load and safetensors  
- [training.md](training.md) — Training modes and loss paths  
