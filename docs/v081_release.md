# v0.81.0 — Accelerator Bridge (Intel NPU + vendor plugin model)

**Release:** **0.80.0 "Native Ship"** → **0.81.0 "Accelerator Bridge"**  
**Checklist:** **112 / 146** (76.7%) on `adjustments` — Intel forward dispatch advances **Accelerators & Distributed** (experimental)

First public **vendor accelerator** path: Loom forwards individual layers through **`poly/accel`** into chaosglue-built plugins, starting with **Intel OpenVINO CPU + NPU** on Linux.

---

## What shipped

### `poly/accel` — vendor-neutral plugin loader

| Item | Detail |
|------|--------|
| **Package** | `poly/accel/` — `Discover`, `Registry`, `Plugin`, `CompiledLayer` |
| **C ABI** | `loom_accel.h` in chaosglue (Loom does not vendor OpenVINO) |
| **Linux** | `dlopen` via CGO (`CGO_ENABLED=1`) |
| **Intel plugin** | `libloom_accel_intel.so` — built from `chaosglue/npu/intel/cabi/` |

### Dispatch integration

| Item | Detail |
|------|--------|
| **`accel_intel.go`** | `DiscoverAccel`, `SyncToAccel`, `DispatchAccelForward`, weight → FP32 bytes |
| **`forward.go`** | `DispatchLayer` calls accel when `layer.ExecTarget.UseAccel()` |
| **`VolumetricLayer`** | `ExecTarget`, `AccelBinding` fields |
| **Init-once** | `SyncToAccel(sizeLabel)` compiles + uploads weights; steady infer reuses handle |

### Lucy [9] — Intel NPU bridge suite

| Item | Detail |
|------|--------|
| **Menu** | `[9]` → `[4]` medium or `[5]` full matrix |
| **Tables** | Timing (Loom / Intel CPU / Intel NPU, speedup) + seven-style drift spectrum |
| **Log** | `lucy_testing_output/nine_layer.txt` |
| **Proof** | 90 cells: Intel infer **💎 EXACT** repeat-forward; Conv2D large **~22×** NPU vs Loom |

### Documentation

| File | Contents |
|------|----------|
| [`accelerators.md`](accelerators.md) | User/developer guide — Intel now, Qualcomm + Google planned |
| chaosglue [`npu/docs/2025-06-26-loom-dispatch-integration-assessment.md`](https://github.com/openfluke/chaosglue/blob/main/npu/docs/2025-06-26-loom-dispatch-integration-assessment.md) | Full benchmark evidence |

---

## What this release is (and is not)

**You now have:**

- A **real dispatch hook** — not a standalone bench binary
- **Intel CPU + NPU** on Linux with documented env + Lucy validation
- A **plugin model** ready for **Qualcomm NPU** and **Google TPU** (same ABI, new `.so`)
- **Experimental** label — appropriate for first wild release

**You do not yet claim:**

- End-user “turn on NPU” without code (`ExecTarget` is manual)
- JSON network field for `exec: intel-npu`
- Training or backward on vendor path
- Bit-perfect Loom ↔ Intel parity on all layers
- Windows or macOS Intel plugin builds
- Qualcomm or Google plugins (roadmap only)

---

## Quick start (developers)

```bash
# 1. Build Intel CABI
cd accel/intel && ./install_openvino.sh && source ./setup_env.sh && ./build.sh

# 2. Optional explicit plugin path
export LOOM_ACCEL_INTEL_SO="$PWD/accel/intel/build/libloom_accel_intel.so"

# 3. Run Lucy validation
cd lucy
CGO_ENABLED=1 go run .
# → 9 → 4
```

Or: `./run_npu_bridge.sh` from `lucy/`.

Monolithic MLP demo: `cd accel/intel/example && CGO_ENABLED=1 go run .`

---

## Future vendors (planned)

| Vendor | Plugin (planned) | SDK / hardware |
|--------|------------------|----------------|
| **Intel** | `libloom_accel_intel.so` | ✅ OpenVINO, Core Ultra NPU |
| **Qualcomm** | `libloom_accel_qcom.so` | QNN / Hexagon, Snapdragon X |
| **Google** | `libloom_accel_google.so` | TPU / PJRT (cloud + edge TBD) |

Loom code path is identical: `DiscoverAccel` → `ExecTarget` → `SyncToAccel` → `ForwardPolymorphic`.

---

## Next targets (v0.82+)

- **AccelPlanner** — auto-select CPU vs Intel CPU vs Intel NPU from shape + layer type
- **JSON `exec` field** — `"intel-npu"` per layer in network JSON
- **Parity** — MatMul bias/layout, norm weight upload, shared INT8 quant
- **Qualcomm CABI** stub in chaosglue `npu/qualcomm/`
- **ASM rollout** (continues from v0.80 roadmap) — Dense backward, SwiGLU, MHA

---

## Key source files

| Area | Files |
|------|-------|
| Accel package | `poly/accel/*.go` |
| Intel dispatch | `poly/accel_intel.go`, `poly/forward.go` |
| Types | `poly/poly.go` (`ExecTarget`, `AccelBinding`, `net.Accel`) |
| Lucy suite | `lucy/examples/nine_layer/` |
| CABI | `accel/intel/include/loom_accel.h`, `accel/intel/src/` |

---

## See also

- [accelerators.md](accelerators.md) — full developer guide
- [v080_release.md](v080_release.md) — previous release (ENTITY + WebGPU)
- [dispatch.md](dispatch.md) — `DispatchLayer` hub
- [gpu.md](gpu.md) — WebGPU (complementary backend)
