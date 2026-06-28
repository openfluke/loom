# Vendor accelerators (NPU / TPU)

**Version:** Loom **v0.81.0** — experimental  
**Status:** Intel CPU + NPU on Linux; Qualcomm NPU and Google TPU planned

This document covers the **`poly/accel`** package: how Loom offloads individual layers to vendor silicon through external C ABI plugins, without embedding OpenVINO, QNN, or TPU SDKs inside the Loom module.

---

## Why a separate accel track

WebGPU covers **portable GPU** (Vulkan / Metal / DX12 / browser). Vendor NPUs and TPUs need **vendor SDKs** that do not belong in the core Go module:

| Approach | Loom owns | External / vendor tree |
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

**C ABI header (vendor-neutral):** `accel/intel/include/loom_accel.h` (copy in `poly/accel/include/`)

| Symbol | Purpose |
|---|---|
| `loom_accel_plugin_open("CPU"\|"NPU")` | Open device |
| `loom_accel_compile_layer` | Build graph + bake weights |
| `loom_accel_infer` | Steady forward |
| `loom_accel_weight_bytes` | Expected native weight blob size (FP32/INT8: 4×N bytes, FP16: 2×N) |

---

## Intel (shipped — experimental)

**Plugin:** `libloom_accel_intel.so` (OpenVINO inside) — **one shared object** per build (`accel/intel/build/`; no `.so.1` soname)  
**Build:** `accel/intel/` (`./build.sh` after `source setup_env.sh`)

### Requirements

- **Linux** amd64/arm64 (Windows `.dll` planned)
- **`CGO_ENABLED=1`** when building/running Loom
- OpenVINO runtime + Intel NPU driver on **`LD_LIBRARY_PATH`**
- Meteor Lake / Core Ultra class NPU (or CPU-only OpenVINO path)

### Environment

```bash
export LOOM_ACCEL_INTEL_SO=/path/to/libloom_accel_intel.so   # optional
source accel/intel/setup_env.sh
```

`accel.DefaultIntelPath()` walks up from cwd for `accel/intel/build/libloom_accel_intel.so`, or set `LOOM_ROOT`.

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

### Weight upload (C ABI v2)

At **`SyncToAccel`** (init once per layer/device), Loom passes dtype-aware weight bytes via **`LayerWeightBytesForAccel`** into **`loom_accel_compile_layer`**. Weights are baked into the OpenVINO graph as constants — **not** re-sent on each forward.

| Layer dtype | Weight bytes at compile | Notes |
|---|---|---|
| **FP32** | `WeightStore.Master` as little-endian float32 | Direct |
| **FP16** | Native `[]uint16` from `Morph(FP16)` (IEEE half) | No FP32 round-trip |
| **INT8** | Dequantized float32 from `GetActive(INT8)` | Matches Loom CPU matmul; OV INT8 graph is still f32 + NPU dynamic quant |

Steady forward only passes **activation** buffers: `loom_accel_infer(in, out)` — pointer + memcpy, no weight pointer per hop.

| Layer | Loom weights baked into OV graph? |
|---|---|
| MatMul, MHA-MatMul | ✅ when byte count matches |
| Conv1D, Conv2D | ✅ when byte count matches |
| ReLU, GELU, Sigmoid, Softmax | ❌ fixed constants in CABI |
| LayerNorm, RMSNorm | ❌ fixed constants in CABI (planned: weight bake) |

### Multi-numerical support — what we have vs what we don’t

| Scope | Status |
|---|---|
| **Loom CPU** | All **21** `DType` values via `WeightStore.Morph` + `GetActive` |
| **Intel accel (bench)** | **FP32, FP16, INT8** only — matches `bench_manifest.json` / Lucy [9] matrix |
| **Weight upload to Intel** | ✅ FP32 / FP16 / INT8 on MatMul, Conv, MHA-MatMul (June 2026) |
| **Activation I/O to Intel** | FP32 / FP16 native bytes; INT8 activations expanded to f32 bytes for OV graph |
| **Not on Intel yet** | BFloat16, Int4, FP8, native INT8 weight constants, remaining morph dtypes |

So: **proper multi-dtype weight upload for the three Intel bench dtypes** — yes. **Full Loom numerical-type parity on NPU** — no (and INT8 math path still differs from Loom on large MAC ops).

### Is the integration “done”?

**Forward dispatch bridge: shipped, experimental — not production-complete.**

| Done (v0.81) | Not done (roadmap) |
|---|---|
| `DiscoverAccel` + `SyncToAccel` + `DispatchAccelForward` | JSON `"exec": "intel-npu"` / AccelPlanner |
| Intel **CPU + NPU** on Linux, CGO + `libloom_accel_intel.so` | Training / backward on vendor path |
| Init-once compile + dtype-aware weight bake | LayerNorm / RMSNorm weight bake |
| Lucy **[9]** 90-cell matrix + `nine_layer.txt` | Bit-perfect parity all layers/dtypes |
| Auto OpenVINO path discovery from `loom/lucy` cwd | Windows plugin, zero-copy I/O |
| Intel infer **💎 EXACT** repeat-forward (90/90) | Qualcomm / Google plugins |

Treat integration as **proven plumbing** you can build on — not a finished “flip NPU on for any network” product feature.

### Benchmark snapshot — Lucy [9] → [5] (90 cells)

**Host:** Fedora, Core Ultra class NPU, OpenVINO via chaosglue deps.  
**Log:** `lucy/lucy_testing_output/nine_layer.txt`  
**Method:** `SyncToAccel` once per device, median infer ms (compile excluded). **Spd** = Loom ÷ Intel (&lt; 1 = Intel slower).

#### Manifest

| Check | Pass | Fail | Total |
|---|---:|---:|---:|
| Intel faster than Loom (**CPU**) | 56 | 34 | 90 |
| Intel faster than Loom (**NPU**) | 36 | 54 | 90 |
| Loom↔Intel parity ≤ INDUS (CPU) | 23 | 67 | 90 |
| Loom↔Intel parity ≤ INDUS (NPU) | 61 | 29 | 90 |
| Intel infer repeat-forward (CPU) | **90** | 0 | 90 |
| Intel infer repeat-forward (NPU) | **90** | 0 | 90 |

Determinism is production-grade; speed and parity are layer/size dependent.

#### Small tier (batch=4, dim=32 — latency floor)

Intel **NPU loses almost every cell** (~0.3–0.7 ms infer floor vs Loom ~0.01 ms).

| Layer | DType | Loom ms | Intel CPU ms | Intel NPU ms | Spd CPU | Spd NPU |
|---|---|---:|---:|---:|---:|---:|
| MatMul | FP32 | 0.009 | 0.013 | 0.585 | 0.69× | 0.02× |
| Conv1D | FP32 | 0.097 | 0.038 | 0.601 | 2.6× | 0.16× |
| ReLU | FP32 | 0.008 | 0.009 | 0.347 | 0.89× | 0.02× |
| Softmax | FP32 | 0.001 | 0.014 | 0.343 | 0.07× | 0.00× |

#### Medium tier (batch=16, dim=256)

| Layer | DType | Loom ms | Intel CPU ms | Intel NPU ms | Spd CPU | Spd NPU |
|---|---|---:|---:|---:|---:|---:|
| MatMul | FP32 | 0.415 | 0.040 | 0.607 | **10×** | 0.68× |
| MatMul | FP16 | 0.644 | 0.056 | 0.312 | **12×** | **2.1×** |
| Conv1D | FP32 | 0.985 | 0.291 | 0.698 | **3.4×** | 1.4× |
| Conv2D | FP32 | 10.134 | 1.456 | 1.252 | **7.0×** | **8.1×** |
| Conv2D | INT8 | 10.815 | 1.233 | 1.162 | **8.8×** | **9.3×** |
| ReLU | INT8 | 1.277 | 0.027 | 0.612 | **47×** | 2.1× |

#### Large tier (batch=8, dim=1024; Conv2D 48×48)

| Layer | DType | Loom ms | Intel CPU ms | Intel NPU ms | Spd CPU | Spd NPU |
|---|---|---:|---:|---:|---:|---:|
| MatMul | FP32 | 3.520 | 0.220 | 0.701 | **16×** | **5.0×** |
| MatMul | FP16 | 5.803 | 0.268 | 0.666 | **22×** | **8.7×** |
| Conv1D | FP32 | 28.140 | 2.438 | 2.310 | **12×** | **12×** |
| Conv2D | FP32 | 117.441 | 5.355 | 5.683 | **22×** | **21×** |
| GELU | FP32 | 3.322 | 0.074 | 0.615 | **45×** | 5.4× |
| MHA-MatMul | INT8 | 4.277 | 0.196 | 0.678 | **22×** | **6.3×** |

#### When to offload

| Target | Offload | Skip |
|---|---|---|
| **Intel CPU** | medium/large **Conv2D, Conv1D, MatMul, MHA**; large activations | small tensors; norms (parity ❌) |
| **Intel NPU** | **large** MAC (Conv2D ~8–21×); medium+ MatMul **FP16** | **small** tier; medium MatMul FP32 (floor &gt; compute); norms/softmax |
| **Either** | Many steady forwards (compile 7–70 ms once) | One-shot micro-ops |

**Why Intel looks slower on some rows:** infer time = fixed dispatch overhead + math. NPU overhead ≈ **0.5 ms**; Loom small MatMul ≈ **0.01 ms**. Weights-on-init removes re-upload — it does not remove per-hop CGO, pack/unpack, or NPU launch latency.

**Parity highlights:** Softmax/Sigmoid strong; LayerNorm/RMSNorm **❌ BROKE** (~1.8, no Loom weights); INT8 MAC drift 3–36 on large tiers (Loom dequant matmul vs OV f32 + dynamic quant).

### `sizeLabel`

Must match bench manifest tiers used when the OpenVINO graph was authored: **`small`**, **`medium`**, **`large`**. Wrong label → shape mismatch at infer.

### Limitations (v0.81)

- **Forward only** — training/backward use Loom CPU when accel-bound
- **Manual `ExecTarget`** — no JSON `"exec": "intel-npu"` yet (AccelPlanner planned)
- **Three Intel dtypes** — FP32/FP16/INT8 bench path only; not all 21 Loom dtypes
- **Numerical parity** — norms broken; INT8 MAC drift on large tiers; MatMul FP32 medium 🟤 H-DRIFT
- **Small tensors** — NPU ~0.5 ms floor; offload **medium/large** MAC ops only
- **Per-hop copies** — tensor → `[]byte` → OV tensor; not zero-copy yet

### Validation — Lucy menu [9]

```bash
cd loom/lucy
CGO_ENABLED=1 go run .
# OpenVINO paths auto-discovered from chaosglue npu deps (no setup_env.sh required)
# → 9 → 4   medium DispatchLayer suite
# → 9 → 5   full 90-cell matrix
```

Or: `./run_npu_bridge.sh` (sources `accel/intel/setup_env.sh` explicitly).

Output: timing table (Loom vs Intel CPU vs Intel NPU, speedup ratios) + seven-style drift spectrum + manifest histogram.

Log: `lucy_testing_output/nine_layer.txt` — see [`testing_and_validation.md`](testing_and_validation.md#nine-layer-intel-bridge-nine_layertxt).

---

## Welvet C-ABI (non-Go bindings)

C / Flutter / Python callers use **`welvet/cabi`** (`welvet.so` / `welvet.h`) instead of importing `poly` directly. Intel offload mirrors the Go flow:

```c
// Pseudocode — see welvet.h for exact signatures
long accel = LoomDiscoverAccel(NULL);           // optional LOOM_ACCEL_INTEL_SO path
long net   = LoomBuildNetworkFromJSON(spec_json);
LoomNetworkAttachAccel(net, accel);
LoomSetLayerExecTarget(net, layer_idx, ExecIntelNPU);
LoomSyncToAccel(net, "medium");
LoomDispatchAccelForward(net, layer_idx, input_handle);
```

| Export | Maps to `poly` |
|--------|----------------|
| `LoomDiscoverAccel` | `DiscoverAccel` |
| `LoomSyncToAccel` | `VolumetricNetwork.SyncToAccel` |
| `LoomLayerWeightBytesForAccel` | `LayerWeightBytesForAccel` |
| `LoomDispatchAccelForward` | `DispatchAccelForward` |

**Build Linux:** `cd welvet/cabi/internal/build && ./build_linux.sh` → `dist/linux_amd64/welvet.so` (and `arm64`).  
**Parity:** `cd welvet/cabi/internal/check && go run .` → **489/489**.

The Intel plugin (`libloom_accel_intel.so`) is still a **separate** dlopen artifact — Welvet does not link OpenVINO at compile time.

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
| **v0.81** ✅ | Intel forward dispatch, dtype-aware weight upload (FP32/FP16/INT8), Lucy [9], benchmark tables in docs |
| **v0.82** | AccelPlanner, JSON `exec` field, MatMul FP32 parity, norm weight bake |
| **v0.83+** | Native INT8 OV graphs, Qualcomm + Google plugins; backward CPU fallback policy |
| **v1.0** | Vendor accel rows enter formal 1.0 checklist |

---

## See also

- [`dispatch.md`](dispatch.md) — `DispatchLayer` routing
- [`gpu.md`](gpu.md) — WebGPU backend
- [`v081_release.md`](v081_release.md) — release notes
- [`testing_and_validation.md`](testing_and_validation.md) — Lucy log interpretation
