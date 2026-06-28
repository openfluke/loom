# MatMul stack — Loom weights → Intel NPU demo

Runs a **3-layer MatMul network** (Dense + LINEAR only) through Loom with **dtype-aware weight upload** on every backend:

| Backend | How |
|---------|-----|
| **Loom CPU** | Go `poly` MatMul |
| **Intel CPU** | `SyncToAccel` uploads weights → OpenVINO compile → infer |
| **Intel NPU** | Same weight upload, NPU device plugin |

**Why MatMul only?** Only MatMul / Conv / MHA-MatMul bake Loom weights into the Intel graph. ReLU, LayerNorm, Softmax use **fixed** OpenVINO graphs and ignore Loom weights — not a weight-transfer demo.

## Run

```bash
cd accel/intel
source ./setup_env.sh
./build.sh

cd example
source ../setup_env.sh
CGO_ENABLED=1 go run .
```

## Expected output

```
╔══════════════════════════════════════════════════════════════╗
║  MatMul stack — same Loom weights → Intel CPU / NPU          ║
╚══════════════════════════════════════════════════════════════╝
  Plugin: .../loom/accel/intel/build/libloom_accel_intel.so
  NPU: available
  Network: 3× MatMul (Dense LINEAR), FP32, 16×256 — weights via SyncToAccel

  ┌─────────────┬──────────┬──────────┬───────────┬──────────────┬────────┐
  │ Backend     │ median ms│ p95 ms   │ compile ms│ vs Loom drift│ status │
  ├─────────────┼──────────┼──────────┼───────────┼──────────────┼────────┤
  │ Loom CPU    │    0.45  │    0.62  │      0.00 │ —            │ OK     │
  │ Intel CPU (2.1x) │  0.21  │    0.38  │    110.00 │ 1.2e-02      │ OK     │
  │ Intel NPU (0.4x) │  1.10  │    1.45  │    170.00 │ 1.2e-02      │ OK     │
  └─────────────┴──────────┴──────────┴───────────┴──────────────┴────────┘

  Weights: one network, fixed seed — SyncToAccel uploads per-layer dtype (FP32 native bytes in this demo).
```

- **One network build**, seed `42` — all three backends see identical weights.
- **compile ms** — one-time `SyncToAccel` per backend (3 layers × ~40–60 ms).
- **drift** — Loom vs Intel output (should be small for MatMul-only).
- Timings vary; NPU loses on tiny stacks, wins on large tiers in the layer bench.

## Network

```
Input [16×256]
  → MatMul → MatMul → MatMul
Output [16×256]
```

Medium shapes from `../bench_manifest.json`.

## Env vars

| Variable | Purpose |
|----------|---------|
| `LOOM_ACCEL_INTEL_SO` | Plugin path (auto: `accel/intel/build/`) |
| `LOOM_ROOT` | Loom module root if cwd is outside repo |
| `INTEL_OPENVINO_DIR` | Set by `setup_env.sh` |

**Build output:** one file — `build/libloom_accel_intel.so` (no `.so.1` symlinks). Do not delete it; Loom `dlopen`s that exact name.
