# Testing, validation, and Lucy logs

This page ties together **how we stress `poly/`**, where **artifacts land**, and how to read **parity tables** in captured logs (for example `lucy/lucy_testing_output/log.txt`).

---

## Where logs come from

The **Lucy** tree (`lucy/`) drives broad layer suites: forward/backward parity, training matrices, save/reload checks, and GPU timing tables. A typical full run writes a transcript under:

- `lucy/lucy_testing_output/log.txt`

That file is meant for human review and regression diffing (adapter name, Metal/Vulkan, per-dtype rows, and summary tallies at the end of each section).

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

## Interpreting a real log (examples)

The following patterns show up in recent `log.txt` captures (Metal adapter, tiled CNN1 suite):

1. **CNN1 generic suite note** — The harness itself reminds you that generic CNN1 tests still include **simulated / PTQ fallback** where a dtype has no strict native path. For a **strict native-only** CPU/GPU/tiling audit, use the **Glitch** `layer_matrix` example (see Glitch docs / examples in-repo).

2. **Float64 on GPU forward** — CPU microseconds vs GPU milliseconds often look like a large “speedup ratio < 1×”; that is frequently **dispatch overhead dominating tiny work**, not a claim that FP64 GPU is slower than CPU math in the large-batch limit.

3. **Wide integer CNN1 backward** — **Int64 / Uint64 / Int32 / Uint32** rows may show **🟤 H-DRIFT** vs float reference in GPU backward parity: the harness compares against an FP32-shaped reference while the native path uses integer semantics — read those rows as **classification / tolerance**, not as “GPU kernel wrong.”

4. **Save/Reload after training** — Several low-bit and float16-family rows show **Save/Reload FAIL** while **TrainOK PASS**. That usually means **training dynamics fit** but **round-trip persistence** for that dtype + layer combo still has gaps (format, unpack, or test expectation). Treat as a **tracking list** for serialization and `WeightStore` versioning, not as “training is broken.”

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

## Related commands (developer workflow)

Exact entrypoints move with refactors; prefer:

- `lucy/README.md` — MRBiVS stack and pointers into `poly/`.
- `poly/README.md` — version checklist and capability matrix.
- `welvet/cabi/internal/check/` — C-ABI vs `poly/` export parity scanner (Go).

---

## See also

- [numerical_types.md](numerical_types.md) — DType list and `WeightStore` lifecycle  
- [gpu.md](gpu.md) — WebGPU context and dispatch overview  
- [serialization.md](serialization.md) — Save/load and safetensors  
- [training.md](training.md) — Training modes and loss paths  
