# Apple (Metal / MPSGraph) — Loom C ABI plugin

Builds **`libloom_accel_apple.dylib`**, the Apple-silicon bridge Loom loads at runtime
via `poly/accel`. This is the macOS mirror of [`accel/intel`](../intel) (OpenVINO) and
[`accel/qualcomm`](../qualcomm) (QNN) — all three implement the identical vendor-neutral
C ABI in `include/loom_accel.h`.

Unlike OpenVINO or QNN there is **no external SDK to install**: Metal,
MetalPerformanceShadersGraph, and Accelerate ship with macOS.

## Two devices

| Device string | Backend | Role |
|---|---|---|
| `"CPU"` | Portable C++ reference (`cpu_reference.cpp`) | Deterministic **parity anchor** — always available, implements all 15 bench layers |
| `"GPU"` | Metal via **MPSGraph** (`mps_backend.mm`) | Real acceleration for a curated op set; **falls back to the CPU reference** for anything not yet wired |

Apple's **Neural Engine (ANE)** has no public low-level graph API — it is reached
indirectly through Metal / Core ML. This first bridge targets the **Metal GPU**
(the programmable accelerator); a Core ML → ANE path is future work.

### GPU-accelerated ops today

`MatMul`, `MHA-MatMul`, `ReLU`, `Sigmoid`, `Softmax`, `Add`, `Multiply`.

Everything else (`Conv1D/2D`, `DepthwiseConv`, `AvgPool/MaxPool`, `GELU`,
`LayerNorm`, `RMSNorm`) runs on the CPU reference even when the `"GPU"` device is
selected — honest, incremental, same maturity bar as the Intel/Qualcomm plugins.

## Build (macOS)

```bash
cd accel/apple
./build.sh
# → build/libloom_accel_apple.dylib
```

Requires the Xcode command line tools (`clang++`, the macOS SDK) and CMake ≥ 3.16.

## Use from Go (`poly/accel`)

```go
reg, err := poly.DiscoverAppleAccel(accel.AccelConfig{
    AppleSO: accel.DefaultApplePath(), // or LOOM_ACCEL_APPLE_DYLIB
})
if err != nil { /* no plugin — stay on Loom CPU */ }
defer reg.Close()

net.Accel = reg
net.Layers[0].ExecTarget = accel.ExecAppleGPU // or ExecAppleCPU
net.SyncToAccel("medium")
out, _, _ := poly.ForwardPolymorphic(net, input)
```

`accel.DefaultApplePath()` honours `LOOM_ACCEL_APPLE_DYLIB`, then `LOOM_ROOT`, then
walks up from the cwd for `accel/apple/build/libloom_accel_apple.dylib`.

## Layout

| Path | Role |
|---|---|
| `include/loom_accel.h` | Vendor-neutral C ABI (in sync with `poly/accel/include/`) |
| `src/shapes.hpp` | Per-tier shapes + layer/dtype tables (mirrors `bench_manifest.json`) |
| `src/half.hpp` | IEEE half + bfloat16 ⇄ float (matches the Go FP16/BF16 byte layout) |
| `src/cpu_reference.{hpp,cpp}` | Deterministic CPU forward for all 15 layers |
| `src/mps_backend.{hpp,mm}` | Metal / MPSGraph GPU forward for the accelerated subset |
| `src/loom_accel_apple.cpp` | C ABI implementation (compile once, infer many) |
| `bench_manifest.json` | Layer × dtype × size matrix |
| `build.sh` | CMake build → `build/libloom_accel_apple.dylib` |
| `build/libloom_accel_apple.dylib` | Output (gitignored) |

## Numerical types

Each accelerator advertises the dtypes **it** can handle via its own `bench_manifest.json`;
the shared Go bridge (`poly/accel_intel.go`) supports the union, and the plugin rejects
anything its `known_dtype` table doesn't list. Apple advertises **FP32, FP16, BF16, INT16,
INT8, INT4**.

The C ABI hands over **FP32** activations/weights for FP32/INT8/INT16/INT4 (matching
`poly/accel_intel.go`), **FP16** (2-byte IEEE half) for FP16, and **BF16** (2-byte bfloat16,
top 16 bits of FP32, round-to-nearest-even) for BF16 — the native low-precision type on
Apple silicon. The GPU path computes in FP32 regardless; FP16/BF16 only change the wire
byte layout. Full Loom numerical-type parity on the accelerator is **not** claimed — this is
an experimental, forward-only, per-layer bridge.

## Environment

| Variable | Purpose |
|---|---|
| `LOOM_ACCEL_APPLE_DYLIB` | Plugin path (optional — auto-discovered under `accel/apple/build/`) |
| `LOOM_ROOT` | Loom repo root when cwd is outside the tree |
