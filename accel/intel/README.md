# Intel NPU — Loom C ABI plugin

Builds **`libloom_accel_intel.so`**, the OpenVINO bridge Loom loads at runtime via `poly/accel`.

## Quick start (Linux + Intel Core Ultra NPU)

```bash
cd accel/intel
./install_openvino.sh
source ./setup_env.sh
./build.sh
# → build/libloom_accel_intel.so
```

## Monolithic demo (CPU vs NPU)

```bash
cd accel/intel/example
source ../setup_env.sh
CGO_ENABLED=1 go run .
```

See [`example/README.md`](example/README.md).

## Layout

| Path | Role |
|---|---|
| `include/loom_accel.h` | Vendor-neutral C ABI (sync with `poly/accel/include/`) |
| `src/loom_accel_intel.cpp` | OpenVINO compile + infer |
| `src/layer_models.*` | Layer graphs (small/medium/large shapes) |
| `bench_manifest.json` | Shape/dtype matrix (Lucy menu [9]) |
| `example/` | Go MLP benchmark |
| `build/libloom_accel_intel.so` | Output (gitignored) |

## Environment

| Variable | Purpose |
|---|---|
| `LOOM_ACCEL_INTEL_SO` | Plugin path (optional — auto-discovered under `accel/intel/build/`) |
| `LOOM_ROOT` | Loom repo root when cwd is outside the tree |
| `INTEL_OPENVINO_DIR` | OpenVINO install (from `setup_env.sh`) |
| `INTEL_NPU_LIBDIR` | NPU driver/compiler libs |

## Dependencies

OpenVINO C++ runtime ≥ 2025.4, Intel NPU driver + compiler shim on Linux. See `install_*.sh`.
