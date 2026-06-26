# Vendor accelerators (NPU / TPU)

**Version:** Loom **v0.81.0** — experimental  
**Status:** Intel CPU + NPU on Linux; Qualcomm NPU and Google TPU planned

This document covers the **`poly/accel`** package: how Loom offloads individual layers to vendor silicon through external C ABI plugins, without embedding OpenVINO, QNN, or TPU SDKs inside the Loom module.

---

## Why a separate accel track

WebGPU covers **portable GPU** (Vulkan / Metal / DX12 / browser). Vendor NPUs and TPUs need **vendor SDKs** that do not belong in the core Go module:

| Approach | Loom owns | chaosglue / vendor tree owns |
|---|---|---|
| **WebGPU** | WGSL, `WGPUContext`, buffers | wgpu-native prebuilts |
| **Vendor accel** | `DispatchLayer` hook, tensor bytes, `ExecTarget` | `libloom_accel_intel.so`, OpenVINO, drivers |

One network graph, one `ForwardPolymorphic` loop — per-layer `ExecTarget` picks **Loom CPU**, **Intel CPU**, **Intel NPU**, or (future) **Qualcomm NPU** / **Google TPU**.

---

## Architecture

```
BuildNetworkFromJSON
    → VolumetricNetwork
    → net.Accel = registry          // DiscoverAccel once

Per layer:
    layer.ExecTarget = ExecIntelNPU // or ExecIntelCPU, ExecLoomCPU
    net.SyncToAccel(sizeLabel)      // compile once + upload weights

ForwardPolymorphic
    → DispatchLayer
        → DispatchAccelForward      // if ExecTarget.UseAccel()
        → else DenseForward / CNN / …
```

**C ABI header (vendor-neutral):** [`chaosglue/npu/include/loom_accel.h`](https://github.com/openfluke/chaosglue/blob/main/npu/include/loom_accel.h)

| Symbol | Purpose |
|---|---|
| `loom_accel_plugin_open("CPU"\|"NPU")` | Open device |
| `loom_accel_compile_layer` | Build graph + bake weights |
| `loom_accel_infer` | Steady forward |
| `loom_accel_weight_bytes` | Expected FP32 weight blob size |

---

## Intel (shipped — experimental)

**Plugin:** `libloom_accel_intel.so` (OpenVINO inside)  
**Build:** [`chaosglue/npu/intel/cabi/`](https://github.com/openfluke/chaosglue/tree/main/npu/intel/cabi)

### Requirements

- **Linux** amd64/arm64 (Windows `.dll` planned)
- **`CGO_ENABLED=1`** when building/running Loom
- OpenVINO runtime + Intel NPU driver on **`LD_LIBRARY_PATH`**
- Meteor Lake / Core Ultra class NPU (or CPU-only OpenVINO path)

### Environment

```bash
export LOOM_ACCEL_INTEL_SO=~/git/chaosglue/npu/intel/cabi/build/libloom_accel_intel.so
source ~/git/chaosglue/npu/intel/example/setup_env.sh   # OpenVINO + NPU libs
```

`accel.DefaultIntelPath()` also searches common chaosglue build locations if the env var is unset.

### Application code

```go
reg, err := poly.DiscoverAccel(accel.AccelConfig{
    IntelSO: accel.DefaultIntelPath(),
})
if err != nil { /* no plugin — stay on Loom CPU */ }
defer reg.Close()

net, _ := poly.BuildNetworkFromJSON(spec)
net.Accel = reg

for i := range net.Layers {
    net.Layers[i].ExecTarget = accel.ExecIntelNPU // or ExecIntelCPU
}

if err := net.SyncToAccel("medium"); err != nil { /* compile failed */ }

out, _, _ := poly.ForwardPolymorphic(net, input)
```

### `ExecTarget` values

| Value | Runs on |
|---|---|
| `accel.ExecLoomCPU` | Default — Go poly CPU |
| `accel.ExecIntelCPU` | OpenVINO CPU |
| `accel.ExecIntelNPU` | OpenVINO Intel NPU plugin |

### Weight upload

At `SyncToAccel`, Loom passes `WeightStore.Master` (FP32) into `CompileLayer`:

| Layer | Weights baked into OV graph? |
|---|---|
| MatMul, MHA-MatMul | ✅ when count matches |
| Conv1D, Conv2D | ✅ when count matches |
| ReLU, GELU, Sigmoid, Softmax | ❌ baked constants in CABI |
| LayerNorm, RMSNorm | ❌ (planned) |

### `sizeLabel`

Must match bench manifest tiers used when the OpenVINO graph was authored: **`small`**, **`medium`**, **`large`**. Wrong label → shape mismatch at infer.

### Limitations (v0.81)

- **Forward only** — training/backward use Loom CPU when accel-bound
- **Manual `ExecTarget`** — no JSON `"exec": "intel-npu"` yet (AccelPlanner planned)
- **Numerical parity** — Softmax/Sigmoid INT8 strong; MatMul/norms FP32 often drift (separate graphs)
- **Small tensors** — NPU fixed ~0.5 ms overhead; offload **medium/large** MAC ops only

### Validation — Lucy menu [9]

```bash
cd loom/lucy
CGO_ENABLED=1 go run .
# → 9 → 4   medium DispatchLayer suite
# → 9 → 5   full 90-cell matrix
```

Output: timing table (Loom vs Intel CPU vs Intel NPU, speedup ratios) + seven-style drift spectrum + manifest histogram.

Log: `lucy_testing_output/nine_layer.txt`

Full evidence: [chaosglue integration assessment](https://github.com/openfluke/chaosglue/blob/main/npu/docs/2025-06-26-loom-dispatch-integration-assessment.md)

---

## Qualcomm NPU (planned)

**Target plugin:** `libloom_accel_qcom.so`  
**SDK:** Qualcomm AI Engine Direct / QNN (device-specific)

Same `loom_accel.h` vtable. Loom side unchanged: `DiscoverAccel` will open a second plugin when `AccelConfig` supplies the path. Expected env:

```bash
export LOOM_ACCEL_QCOM_SO=/path/to/libloom_accel_qcom.so
```

Snapdragon X Elite / Hexagon class devices. No implementation in-tree yet — Intel path proves the dispatch model.

---

## Google TPU (planned)

**Target plugin:** `libloom_accel_google.so`  
**SDK:** libtpu / OpenXLA PJRT (deployment TBD)

Same C ABI surface. Useful for cloud TPU pods and future edge TPU silicon. Loom remains a **client** that compiles per-layer subgraphs and ships weights once.

---

## Package layout

```
poly/
├── accel/
│   ├── accel.go           Public types, DefaultIntelPath
│   ├── target.go          ExecTarget enum
│   ├── registry.go        Discover, PluginFor
│   ├── plugin_linux.go    dlopen + C ABI calls (CGO)
│   └── runtime_linux.go   OpenVINO LD_LIBRARY_PATH hints
├── accel_intel.go         SyncToAccel, DispatchAccelForward
└── forward.go             DispatchLayer → DispatchAccelForward
```

---

## Comparison to WebGPU

| | WebGPU | Vendor accel |
|---|---|---|
| **Scope** | Full network GPU path | Per-layer offload |
| **Portability** | Vulkan/Metal/browser | Vendor + OS specific |
| **Build** | Pure Go + wgpu module | **CGO** + external `.so` |
| **Training** | GPU backward supported | Forward only (v0.81) |
| **Best for** | LLM decode, large batches | Fixed-function NPU MAC ops |

Use **both**: WebGPU for general GPU; Intel NPU for Conv/MatMul on Core Ultra when shapes are large enough.

---

## Roadmap

| Milestone | Description |
|---|---|
| **v0.81** ✅ | Intel forward dispatch, Lucy [9], docs |
| **v0.82** | AccelPlanner, JSON `exec` field, MatMul parity |
| **v0.83+** | Qualcomm + Google plugins; backward CPU fallback policy |
| **v1.0** | Vendor accel rows enter formal 1.0 checklist |

---

## See also

- [`dispatch.md`](dispatch.md) — `DispatchLayer` routing
- [`gpu.md`](gpu.md) — WebGPU backend
- [`v081_release.md`](v081_release.md) — release notes
- [`testing_and_validation.md`](testing_and_validation.md) — Lucy log interpretation
