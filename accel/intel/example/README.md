# Monolithic MLP — CPU vs NPU demo

Runs a **5-layer feed-forward network** through Loom with the same weights:

| Backend | How |
|---|---|
| **Loom CPU** | Go `poly` kernels |
| **Intel CPU** | OpenVINO via `libloom_accel_intel.so` |
| **Intel NPU** | OpenVINO NPU plugin (Linux + Core Ultra) |

## Prerequisites

Build the plugin (from repo root or `accel/intel/`):

```bash
cd accel/intel
./install_openvino.sh
source ./setup_env.sh
./build.sh
```

**Linux + NPU:** `CGO_ENABLED=1`, Intel NPU driver installed.

**macOS:** Loom CPU-only smoke test (`go run .`); full Intel/NPU path needs Linux.

## Run

```bash
cd accel/intel/example
source ../setup_env.sh
CGO_ENABLED=1 go run .
```

Optional env vars:

- `LOOM_ACCEL_INTEL_SO` — path to plugin (auto-discovered from `accel/intel/build/` if unset)
- `LOOM_ROOT` — Loom module root if not running from inside the repo tree
- `INTEL_OPENVINO_DIR` — OpenVINO install (set by `setup_env.sh`)

## Network

```
Input [16×256]
  → Dense+ReLU → Dense+ReLU → Dense → LayerNorm → Softmax
Output [16×256]
```

Uses **medium** shapes from `../bench_manifest.json` (matches OpenVINO graphs in the plugin).
