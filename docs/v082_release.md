# v0.82.0 — Snapdragon Bridge (SIMD CPU + Qualcomm/Hexagon NPU)

**Release:** **0.81.0 "Accelerator Bridge"** → **0.82.0 "Snapdragon Bridge"**  
**Checklist:** **117 / 147** (79.6%) on `adjustments` — a **second** vendor NPU (Qualcomm/Hexagon) plus a **SIMD CPU fast-path** advance the Numerical Core, Accelerators, and Ecosystem categories.

Two headline items land on top of the v0.81 Intel bridge:

1. **SIMD CPU forward** — hand-written AVX2/FMA (x86-64) and NEON (ARM64) dot-tile kernels behind `SetSimdForward`, with portable Go loops still the parity reference.
2. **Qualcomm / Hexagon NPU** — the second `poly/accel` vendor plugin, running on **Windows ARM64** through the **QNN AI Engine Direct** SDK. Forward-only, per-layer, experimental — the same maturity bar as Intel.

---

## What shipped

### SIMD CPU fast-path (`poly/simd`)

| Item | Detail |
|------|--------|
| **x86-64** | AVX2/FMA `DotTile` dot-product tiles for dense/matmul-heavy forward paths |
| **ARM64** | NEON `DotTile` (`neon_arm64.go`) — `unsafe.Slice` over `float32` pointers |
| **Toggle** | `SetSimdForward(true/false)` — off falls back to portable Go tiled loops |
| **Parity** | Go loops remain the bit-exact reference; SIMD is an opt-in throughput path |
| **Docs / bench** | [`simd.md`](simd.md); Lucy **[7]** (seven-layer) and **[11]** (transformer decode) |

### Qualcomm / Hexagon NPU plugin (`accel/qualcomm`)

| Item | Detail |
|------|--------|
| **Plugin** | `loom_accel_qualcomm.dll` — QNN AI Engine Direct (QAIRT) behind the vendor-neutral `loom_accel.h` C ABI |
| **Build** | `build_clang.ps1` (`llvm-mingw` `aarch64-w64-mingw32-clang++`); MSVC/GNU ABI notes in `accel/qualcomm/README.md` |
| **Windows loader** | `poly/accel/plugin_qualcomm_windows.go` — `LoadLibraryA` / `GetProcAddress`, `PrepareQualcommRuntime` adjusts `PATH` |
| **Targets** | `ExecQualcommCPU` (Kryo), `ExecQualcommNPU` (Hexagon HTP) |
| **DTypes** | FP32 / FP16 / INT16 / INT8 / INT4 (see `bench_manifest.json`) |
| **Install** | `install_qairt.ps1 -Persist` sets machine-wide `QNN_SDK_ROOT` / `LOOM_QUALCOMM_RUNTIME` |

### Dispatch + robustness

| Item | Detail |
|------|--------|
| **`accel_qualcomm.go`** | `DiscoverQualcommAccel` plugin discovery; per-layer `DispatchLayer` offload |
| **Unique graph names** | `loom_<op>_<dtype>_<seq>` (atomic counter) — fixes `QnnGraph_create` collisions across dtypes in one context |
| **Context reset** | `CompiledGraph::~CompiledGraph()` → `Backend::reset_context()` — frees leaked graphs; fixes `0xc0000005` on long matrix runs |
| **Quiet logging** | QNN log level clamped to ERROR by default (`LOOM_QNN_VERBOSE=1` to restore); Go-side terminal noise filter (`quiet.go`) — full output still written to `snapdragon.txt` |

### Lucy [12] — Snapdragon NPU bridge suite

| Item | Detail |
|------|--------|
| **Menu** | `[12]` — mirrors the Intel `[9]` `nine_layer` suite |
| **Tables** | Timing (Loom / Qualcomm CPU / Qualcomm NPU, speedup) + seven-style drift spectrum |
| **Log** | `lucy/lucy_testing_output/snapdragon.txt` |

### webgpu on Windows ARM64

`go.mod` `replace` directive pointing at a local `openfluke/webgpu` copy works around the MSVC-vs-GNU ABI mismatch when linking `libwgpu_native.a` on Windows ARM64. Documented in `accel/qualcomm/README.md`.

---

## What this release is (and is not)

**You now have:**

- A **second vendor NPU** (Qualcomm/Hexagon) on Windows ARM64 through the same `poly/accel` C ABI as Intel
- A **SIMD CPU fast-path** (AVX2/NEON) that keeps Go loops as the parity reference
- Persistent, all-user QNN environment setup and a reproducible `clang++` build
- **Experimental** label — a rocky-but-real bridge, good for a release, not for prod

**You do not yet claim:**

- Whole-model `.entity` → NPU lowering (offload is per-layer, forward-only)
- Training or backward on the Qualcomm path
- Bit-perfect Loom ↔ Hexagon parity on all layers/dtypes
- An NPU parity suite vs the WebGPU reference (SmolLM-class smoke)
- A JSON network field for `exec: qualcomm-npu` (targets set in code)

---

## Quick start (developers, Windows ARM64)

```powershell
# 1. Install the QNN runtime + persist env for all users
cd accel\qualcomm
./install_qairt.ps1 -Persist

# 2. Build the plugin with llvm-mingw clang++
./build_clang.ps1

# 3. Run the Lucy Snapdragon suite
cd ..\..\lucy
go run .
# -> 12
```

Set `LOOM_QNN_VERBOSE=1` to see full QNN/HTP logs in the terminal (otherwise clamped to ERROR).

---

## Checklist deltas (v0.81 → v0.82)

| Category | v0.81 | v0.82 | Change |
|----------|:-----:|:-----:|--------|
| 1. Numerical Core | 22 / 31 | 23 / 32 | +SIMD CPU forward kernels |
| 3. Accelerators & Distributed | 4 / 18 | 6 / 18 | +Intel + Qualcomm per-layer NPU dispatch |
| 5. Deployment Ecosystem | 25 / 27 | 27 / 27 | +Intel + Qualcomm NPU backends |
| **Grand total** | 112 / 146 | **117 / 147** | **76.7% → 79.6%** |

---

## Next targets (v0.83+)

- **Whole-model `.entity` → NPU** lowering (both vendors), not just per-layer
- **NPU parity suite** vs WebGPU reference (SmolLM-class smoke)
- **AccelPlanner** + JSON `exec` field (`qualcomm-npu` / `intel-npu` per layer)
- **GPU backward** wiring (SwiGLU / MHA) continues from the v0.81 roadmap
- **Google TPU** plugin (`libloom_accel_google.so`) — same ABI

---

## Key source files

| Area | Files |
|------|-------|
| SIMD | `poly/simd/*.go` (`neon_arm64.go`, x86-64 kernels), `SetSimdForward` |
| Accel package | `poly/accel/*.go` (`plugin_qualcomm_windows.go`, `plugin_qualcomm_stub.go`) |
| Qualcomm dispatch | `poly/accel_qualcomm.go`, `poly/forward.go` |
| Qualcomm plugin C++ | `accel/qualcomm/src/` (`qnn_wrapper.*`, `loom_accel_qualcomm.cpp`, `layer_models.*`) |
| Build / install | `accel/qualcomm/build_clang.ps1`, `install_qairt.ps1`, `bench_manifest.json` |
| Lucy suite | `lucy/examples/snapdragon/` |

---

## See also

- [snapdragon_npu.md](snapdragon_npu.md) — Snapdragon bridge deep-dive (achievements + honest gaps)
- [simd.md](simd.md) — SIMD CPU forward path
- [accelerators.md](accelerators.md) — full vendor accel guide (Intel + Qualcomm)
- [v081_release.md](v081_release.md) — previous release (Intel NPU + plugin model)
