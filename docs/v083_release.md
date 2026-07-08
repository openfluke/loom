# v0.83.0 — Apple Bridge (Apple GPU / Metal + BF16)

**Release:** **0.82.0 "Snapdragon Bridge"** → **0.83.0 "Apple Bridge"**  
**Checklist:** **119 / 149** (79.9%) on `adjustments` — a **third** accelerator vendor (Apple GPU via Metal/MPSGraph) advances the Accelerators and Ecosystem categories, plus a **BF16** wire dtype for the shared accel bridge.

One headline item lands on top of the v0.82 Intel + Qualcomm bridges:

1. **Apple GPU / Metal** — the third `poly/accel` vendor plugin, running on **macOS Apple silicon** through Apple's **Metal Performance Shaders Graph**. Forward-only, per-layer, experimental — the same maturity bar as Intel and Qualcomm. No SDK to vendor: Metal ships with macOS.

---

## What shipped

### Apple Metal / MPSGraph plugin (`accel/apple`)

| Item | Detail |
|------|--------|
| **Plugin** | `libloom_accel_apple.dylib` — Metal / MetalPerformanceShaders / MetalPerformanceShadersGraph behind the vendor-neutral `loom_accel.h` C ABI |
| **Build** | CMake (`build.sh` → `build/libloom_accel_apple.dylib`); C++17 + Objective-C++ (`mps_backend.mm` under ARC) |
| **Darwin loader** | `poly/accel/plugin_darwin.go` — `dlopen`/`dlsym`; `apple_stub.go` for non-darwin / no-cgo |
| **Devices** | `ExecAppleCPU` (portable C++ reference, parity anchor) · `ExecAppleGPU` (Metal / MPSGraph, per-op CPU fallback) |
| **GPU ops** | MatMul, MHA-MatMul, ReLU, Sigmoid, Softmax, Add, Multiply on MPSGraph; Conv/GELU/pool/norm fall back to the CPU reference |
| **DTypes** | FP32 / FP16 / **BF16** / INT16 / INT8 / INT4 (see `accel/apple/bench_manifest.json`) |
| **Install** | none — Xcode command-line tools provide the Metal frameworks |

### BF16 wire dtype (shared bridge)

| Item | Detail |
|------|--------|
| **`poly/accel_intel.go`** | Vendor-neutral bridge now packs/unpacks **bfloat16** (top 16 bits of FP32, round-to-nearest-even) for weights and I/O — `float32ToBFloat16Bits` / `bfloat16BitsToFloat32` |
| **Plugin side** | `accel/apple/src/half.hpp` (`float_to_bfloat16`/`bfloat16_to_float`), `shapes.hpp` (`known_dtype` + 2-byte `io_elem_size`), `WireFmt {FP32,FP16,BF16}` in `loom_accel_apple.cpp` |
| **Rationale** | BF16 is the native low-precision type on Apple silicon; each accelerator advertises the dtypes it can handle via its own `bench_manifest.json` |

### Lucy [13] — Apple GPU bridge suite

| Item | Detail |
|------|--------|
| **Menu** | `[13]` — mirrors the Intel `[9]` and Qualcomm `[12]` suites |
| **Tables** | Timing (Loom / Apple CPU / Metal GPU, speedup + compile) + seven-style drift spectrum |
| **Log** | `lucy/lucy_testing_output/apple.txt` |

---

## Numbers (from `apple.txt`, Lucy [13] → [5], 180 cells)

- **Determinism:** 180/180 **💎 EXACT** repeat-forward on **both** Apple CPU and Metal GPU.
- **Parity:** GPU 132/180 ≤ INDUS, CPU 78/180 (GPU carries a looser tolerance; raw drift is near-identical).
- **Speed:** Metal GPU up to **5.4×** faster than Loom CPU on large MatMul/MHA; Apple CPU reference up to **94×** on elementwise (ReLU/GELU/Sigmoid at INT4).
- **Weak spots:** Conv1D/Conv2D + GELU are CPU-reference-only (no MPSGraph path, ~0.24–0.27× Loom CPU); LayerNorm/RMSNorm parity ❌ BROKE (no weight bake); INT8 MatMul drift breaks on the large tier.

---

## What this release is (and is not)

**You now have:**

- A **third accelerator vendor** (Apple GPU) on macOS through the same `poly/accel` C ABI as Intel and Qualcomm
- A **CPU reference + Metal GPU** pair behind one plugin, with transparent per-op fallback
- A **BF16** wire dtype in the shared bridge (Apple-native low precision)
- **Experimental** label — proven plumbing, good for a release, not for prod

**You do not yet claim:**

- MPSGraph Conv / GELU (both run the CPU reference today)
- LayerNorm / RMSNorm weight bake (parity broken)
- Apple Neural Engine (ANE) — Metal only; ANE needs a Core ML path (future)
- Whole-model `.entity` → GPU lowering (offload is per-layer, forward-only)
- Training or backward on the Apple path
- A JSON network field for `exec: apple-gpu` (targets set in code)

---

## Quick start (developers, macOS Apple silicon)

```bash
# 1. Build the plugin (needs Xcode command-line tools)
cd accel/apple
./build.sh

# 2. Run the Lucy Apple suite
cd ../../lucy
CGO_ENABLED=1 go run .
# -> 13
#   [4] medium DispatchLayer suite
#   [5] full 10×6×3 matrix (apple.txt)
#   [0] raw CABI matrix (all 15 layers)
```

`accel.DefaultApplePath()` walks up from cwd for `accel/apple/build/libloom_accel_apple.dylib`, or set `LOOM_ACCEL_APPLE_DYLIB`.

---

## Checklist deltas (v0.82 → v0.83)

| Category | v0.82 | v0.83 | Change |
|----------|:-----:|:-----:|--------|
| 3. Accelerators & Distributed | 6 / 18 | 7 / 19 | +Apple GPU per-layer dispatch |
| 5. Deployment Ecosystem | 27 / 27 | 28 / 28 | +Apple GPU backend |
| **Grand total** | 117 / 147 | **119 / 149** | **79.6% → 79.9%** |

---

## Next targets (v0.84+)

- **MPSGraph Conv / GELU** — move Conv1D/Conv2D/GELU off the CPU reference onto the GPU
- **Norm weight bake** — fix LayerNorm/RMSNorm parity (Loom weights into the reference/graph)
- **ANE via Core ML** — reach the Neural Engine (not a Metal device)
- **Whole-model `.entity` → NPU/GPU** lowering (all vendors), not just per-layer
- **NPU parity suite** vs WebGPU reference (SmolLM-class smoke)
- **AccelPlanner** + JSON `exec` field (`apple-gpu` / `intel-npu` / `qualcomm-npu` per layer)
- **Google TPU** plugin (`libloom_accel_google.so`) — same ABI

---

## Key source files

| Area | Files |
|------|-------|
| Apple plugin C++ | `accel/apple/src/` (`loom_accel_apple.cpp`, `cpu_reference.*`, `mps_backend.mm`, `shapes.hpp`, `half.hpp`) |
| Build | `accel/apple/CMakeLists.txt`, `accel/apple/build.sh`, `accel/apple/bench_manifest.json` |
| Accel package | `poly/accel/plugin_darwin.go`, `poly/accel/apple_stub.go`, `target.go`, `registry.go`, `accel.go` |
| Apple dispatch | `poly/accel_apple.go`, `poly/accel_intel.go` (BF16 + vendor-neutral routing), `poly/forward.go` |
| Lucy suite | `lucy/examples/apple/`, `lucy/examples/apple_menu.go` |

---

## See also

- [apple_metal.md](apple_metal.md) — Apple bridge deep-dive (results + honest gaps)
- [accelerators.md](accelerators.md) — full vendor accel guide (Intel + Qualcomm + Apple)
- [snapdragon_npu.md](snapdragon_npu.md) — Qualcomm/Hexagon bridge
- [v082_release.md](v082_release.md) — previous release (SIMD + Qualcomm NPU)
