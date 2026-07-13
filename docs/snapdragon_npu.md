# Snapdragon (Hexagon NPU) bridge

**Version:** Loom **v0.82.0 "Snapdragon Bridge"** — **experimental** (same maturity bar as the Intel NPU bridge)
**Status:** Windows on Snapdragon X · Hexagon v73 HTP + Kryo CPU reference · forward-only
**Plugin:** `loom_accel_qualcomm.dll` (Qualcomm **QNN AI Engine Direct** inside)
**Lucy menu:** **[12]** — log: `lucy_testing_output/snapdragon.txt` (see [lucy.md](lucy.md))

> This is the Snapdragon mirror of the Intel path in [`accelerators.md`](accelerators.md).
> It shares the identical vendor-neutral C ABI (`accel/qualcomm/include/loom_accel.h`),
> the same `poly/accel` loader, the same `SyncToAccel` → `DispatchLayer` flow. Only the
> vendor backend differs: QNN/Hexagon instead of OpenVINO/Intel.

---

## TL;DR — this was a lot of work, and it works enough to demo

Getting Loom talking to the Hexagon NPU on Windows/ARM64 required clearing a long
chain of blockers, none of which were "the neural net" itself:

- **ARM64 SIMD** kernels (`neon_arm64.go`) so the Loom CPU baseline runs natively.
- **`webgpu` `go.mod` replace** — the published static archive has an ABI mismatch on
  windows/arm64; both `loom/go.mod` and Lucy `go.mod` must point at a local build.
- **QAIRT SDK acquisition** — ~1.3 GB Qualcomm SDK behind a login, vendored under
  `accel/qualcomm/deps/`, plus the free HTP runtime with signed Hexagon skel/cat.
- **A C++ QNN plugin** built with **llvm-mingw** (no CMake/MSVC on the box) that
  loads `QnnHtp.dll` / `QnnCpu.dll`, builds a single-op graph per layer, quantizes
  weights, finalizes and executes it.
- **Windows cgo loader** (`LoadLibraryExA` + `GetProcAddress`) for the plugin.
- **Machine-wide env** so `QNN_SDK_ROOT` survives a fresh SSH login / any user.
- **Bug hunts:** duplicate QNN graph names failing every dtype after FP32; a graph
  leak that slowed the full matrix and could fault (`0xc0000005`) on Ctrl+C; a wall
  of DSP-transport log spam.

The result: **a real Loom → Hexagon NPU dispatch bridge with deterministic inference
and multi-× speedups on larger layers.** It is **not** production-complete — plenty
of layers/dtypes still fall back or error. Treat it exactly like the Intel bridge:
**proven plumbing to build on, not a "flip the NPU on for any network" feature.**

---

## What actually runs today

Measured from `snapdragon.txt` (Lucy **[12] → [5]**, full 15-layer × 5-dtype ×
3-size matrix, `SyncToAccel` once per device):

| Layer | FP32 | FP16 | INT16 | INT8 | INT4 |
|---|:--:|:--:|:--:|:--:|:--:|
| **MatMul** | ✅ HTP+CPU | ✗ | ✗ | ✗ | ✗ |
| **MHA-MatMul** | ✅ HTP | ✗ | ✗ | ✗ | ✗ |
| **ReLU** | ✅ | ✗ | ✅ | ✅ | ✅ |
| **GELU** | ✅ | ✗ | ✅ | ✅ | ✅ |
| **Sigmoid** | ✅ | ✗ | ✅ | ✅ | ✅ |
| **Softmax** | ✅ | ✗ | ✅ | ✅ | ✅ |
| **Conv1D / Conv2D** | ✗ | ✗ | ✗ | ✗ | ✗ |
| **LayerNorm / RMSNorm** | ✗ | ✗ | ✗ | ✗ | ✗ |

✅ = compiles + infers on Hexagon HTP (and QnnCpu where shown). ✗ = errors at
graph build today (see [Known gaps](#known-gaps--a-lot-of-work-left)).

**Elementwise/activation ops are the strong suit** (FP32 + INT16/INT8/INT4). MatMul
and MHA run FP32 on the NPU. Everything else is still on the to-do pile.

---

## Numbers that matter (from `snapdragon.txt`)

### Determinism — production-grade

Repeat-forward drift (same input twice) is **💎 EXACT on every cell that ran**:

| Check | Pass | Total |
|---|---:|---:|
| QNN infer repeat (CPU) | **54** | 54 |
| QNN infer repeat (NPU) | **54** | 54 |

The NPU gives byte-identical outputs run to run — the single most important
property for using it as a real execution target.

### Loom ↔ Hexagon parity

| Check | ≤ INDUS | Total |
|---|---:|---:|
| Loom ↔ Qualcomm parity (**NPU**) | **45** | 54 |
| Loom ↔ Qualcomm parity (**CPU**) | 33 | 54 |

NPU parity is *tighter* than CPU here because the QnnCpu reference backend is FP32-only
and drifts more on the quantized activation ops. Softmax is near-exact
(`~1e-6`); MatMul/MHA FP32 sit at 🟨 LOWBIT on HTP (H-DRIFT vs the CPU anchor).

### Speed — HTP wins as layers grow

The Hexagon infer floor is ~**0.5 ms** (fixed launch cost), so small tensors lose but
**large tiers win big** (Loom ÷ Hexagon, higher = NPU faster):

| Tier | Layer | DType | Loom ms | Hexagon ms | Speed |
|---|---|---|---:|---:|---:|
| large | MatMul | FP32 | 3.582 | 0.510 | **7.0×** |
| large | ReLU | INT8 | 3.516 | 0.508 | **6.9×** |
| large | Sigmoid | INT4 | 3.598 | 0.531 | **6.8×** |
| large | GELU | INT4 | 3.505 | 0.539 | **6.5×** |
| large | MHA-MatMul | FP32 | 2.187 | 0.525 | **4.2×** |
| medium | MatMul | FP32 | 1.032 | 0.507 | **2.0×** |
| small | MatMul | FP32 | ~0 | 0.510 | loses (floor) |

Manifest totals: Hexagon beat Loom on **25 / 150** cells, mostly the large tier.
`QnnCpu` beat Loom on only **1 / 150** — it's an accuracy anchor, not a speed play.

### One-time compile cost

`SyncToAccel` compiles each graph once. HTP graph prepare is the expensive part:

| Op | HTP compile (ms) |
|---|---:|
| Activation ops (ReLU/GELU/…) | ~17–20 |
| MatMul (small→large) | ~113 → ~187 |
| MHA-MatMul (large) | ~187 |

Compile is paid **once per layer/device**; steady inference is the sub-ms numbers above.

---

## How it plugs into Loom (unchanged dispatch model)

```go
reg, err := poly.DiscoverQualcommAccel(accel.AccelConfig{
    QualcommSO: accel.DefaultQualcommPath(),
})
if err != nil { /* no plugin — stay on Loom CPU */ }
defer reg.Close()

net, _ := poly.BuildNetworkFromJSON(spec)
net.Accel = reg

net.Layers[0].ExecTarget = accel.ExecQualcommNPU // or ExecQualcommCPU
_ = net.SyncToAccel("medium")                    // compile once + upload weights

out, _, _ := poly.ForwardPolymorphic(net, input)
```

### `ExecTarget` values

| Value | Runs on |
|---|---|
| `accel.ExecLoomCPU` | Default — Go poly CPU |
| `accel.ExecQualcommCPU` | QNN `QnnCpu` reference backend (FP32 parity anchor) |
| `accel.ExecQualcommNPU` | QNN `QnnHtp` — Hexagon Tensor Processor |

### Accelerator cores

| Core | Backend | Data types | Quant | Role |
|---|---|---|---|---|
| **HTP** (Hexagon) | `QnnHtp` | INT4/8/16, FP16* | needed | Dedicated NPU — the fast path |
| **CPU** (Kryo) | `QnnCpu` | FP32 | — | Parity anchor; rejects FP16/INT* graph nodes today |
| **GPU** (Adreno) | `QnnGpu` | FP32/FP16/INT* | — | Not wired into the bench yet |

\* FP16 is native HTP silicon but currently errors in our graph build — see gaps.

---

## Known gaps — a lot of work left

This is honestly *experimental*. What's still broken or missing:

| Gap | Detail |
|---|---|
| **Conv1D / Conv2D** | `ERR` on every dtype — op param encoding (stride/pad tensors) not accepted by `graphAddNode` yet |
| **LayerNorm / RMSNorm** | `ERR` — no weight bake + norm op mapping incomplete |
| **FP16 everywhere** | `QnnGraph_create`/`graphAddNode` fails; FP16 tensor path needs fixing despite HTP supporting it |
| **MatMul/MHA quantized** | Only FP32 compiles; INT4/8/16 MatMul error (weight-tensor / quant-encoding) |
| **INT4** | Per-tensor 4-bit weights compile for activations but MatMul INT4 still fails; QNN wants block-wise 4-bit encoding |
| **QnnCpu quant** | CPU reference backend is FP32-only — quantized cells only validate against Loom, not a QNN CPU anchor |
| **Adreno GPU** | `QnnGpu` backend selectable in the wrapper but untested |
| **Training / backward** | Forward-only; backward stays on Loom CPU |
| **JSON `exec` field** | No `"exec": "qualcomm-npu"` yet — `ExecTarget` is set programmatically |
| **Zero-copy I/O** | Per-hop tensor → `[]byte` → QNN tensor copies |

### Environment robustness

- QNN on Windows/Snapdragon usually can't load the signed Hexagon skel and falls back
  to the **user driver** path — this is normal, inference still works
  (`Hexagon NPU: available`). The chatter is filtered in Lucy [12] unless
  `LOOM_QNN_VERBOSE=1`.
- `QNN_SDK_ROOT` must be set **machine-wide** for fresh shells / other users;
  `install_qairt.ps1 -Persist` (elevated) does this.

---

## Build & run

Full setup (SDK download, env persistence, llvm-mingw build, webgpu note) lives in
[`accel/qualcomm/README.md`](../accel/qualcomm/README.md). Short version:

```powershell
cd accel/qualcomm
pwsh -File .\install_qairt.ps1 -Persist   # runtime + SDK detect + machine env (elevated)
. .\setup_env.ps1
.\build_clang.ps1                         # llvm-mingw → build/loom_accel_qualcomm.dll

cd ..\..\lucy_bloom_rivers
go run .                                  # → [12] Snapdragon NPU bridge
#   [4] medium DispatchLayer suite  (fast, ~9 s)
#   [5] full 15×5×3 matrix          (minutes — HTP compiles per graph)
```

Avoid Ctrl+C mid-compile on the full matrix — QNN can fault (`0xc0000005`).

---

## See also

- [`accelerators.md`](accelerators.md) — the accel model + Intel NPU bridge
- [`accel/qualcomm/README.md`](../accel/qualcomm/README.md) — install / build / env / webgpu
- [`numerical_types.md`](numerical_types.md) — Loom's 21 DTypes vs vendor bench dtypes
- [`dispatch.md`](dispatch.md) — `DispatchLayer` routing
- [`windows_arm64.md`](windows_arm64.md) — Windows on ARM build notes
