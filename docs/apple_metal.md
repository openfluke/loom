# Apple (Metal GPU / MPSGraph) bridge

**Version:** Loom **v0.83.0 "Apple Bridge"** — **experimental** (same maturity bar as the Intel & Qualcomm NPU bridges)
**Status:** macOS on Apple silicon · Metal GPU (MPSGraph) + Accelerate/CPU reference · forward-only
**Plugin:** `libloom_accel_apple.dylib` (Apple **Metal Performance Shaders Graph** inside)
**Lucy menu:** **[13]** — log: `lucy/lucy_testing_output/apple.txt`

> This is the Apple mirror of the Intel path in [`accelerators.md`](accelerators.md) and the
> Qualcomm path in [`snapdragon_npu.md`](snapdragon_npu.md). It shares the identical
> vendor-neutral C ABI (`accel/apple/include/loom_accel.h`), the same `poly/accel` loader,
> the same `SyncToAccel` → `DispatchLayer` flow. Only the vendor backend differs: Metal /
> MPSGraph instead of OpenVINO or QNN.

---

## TL;DR

The Apple plugin is the cleanest of the three vendor bridges to stand up, because there is
**no external SDK to vendor** — Metal, MetalPerformanceShaders, and MetalPerformanceShadersGraph
ship with macOS. There's no 1.3 GB download, no signed DSP skel, no `LD_LIBRARY_PATH` dance.

It exposes **two devices** behind the same C ABI:

- **`CPU`** — a portable, deterministic C++ reference forward for all 15 layers. This is the
  **parity anchor** (it plays the role QnnCpu plays for Qualcomm).
- **`GPU`** — a **Metal / MPSGraph** backend for the ops MPSGraph handles well (MatMul, MHA,
  ReLU, Sigmoid, Softmax, Add, Multiply). Anything it doesn't build silently falls back to the
  CPU reference, so **every cell still returns a correct value**.

The result: **a real Loom → Apple GPU dispatch bridge with byte-perfect run-to-run determinism
(180/180 on both devices) and multi-× speedups on large MatMul/MHA on the GPU and on
elementwise ops on the CPU.** It is **not** production-complete — Conv and the norms are still
CPU-reference-only and slower than Loom's own CPU, and the usual low-bit MatMul drift is present.
Treat it exactly like the Intel/Qualcomm bridges: **proven plumbing to build on, not a "flip the
GPU on for any network" feature.**

---

## What actually runs today

Measured from `apple.txt` (Lucy **[13] → [5]**, the 10-layer DispatchLayer matrix × 6 dtypes ×
3 sizes = **180 cells**, `SyncToAccel` once per device). The raw CABI bench (menu **[0]**)
additionally covers DepthwiseConv / AvgPool / MaxPool for the full 15-layer plugin surface.

| Layer | Metal GPU | Apple CPU ref | Notes |
|---|:--:|:--:|---|
| **MatMul** | ✅ MPSGraph | ✅ | GPU wins on large tiers |
| **MHA-MatMul** | ✅ MPSGraph | ✅ | GPU wins on large tiers |
| **ReLU** | ✅ MPSGraph | ✅ | CPU is faster at small/medium |
| **Sigmoid** | ✅ MPSGraph | ✅ | CPU is faster at small/medium |
| **Softmax** | ✅ MPSGraph | ✅ | GPU floor-bound; CPU near-instant |
| **Add / Multiply** | ✅ MPSGraph | ✅ | (raw CABI bench) |
| **GELU** | ⚠️ CPU fallback | ✅ | not built on MPSGraph yet |
| **Conv1D / Conv2D** | ⚠️ CPU fallback | ✅ | CPU reference; slower than Loom CPU |
| **DepthwiseConv / AvgPool / MaxPool** | ⚠️ CPU fallback | ✅ | (raw CABI bench) |
| **LayerNorm / RMSNorm** | ⚠️ CPU fallback | ✅ | parity broken (no Loom weights baked) |

✅ = compiles + infers on that device. ⚠️ CPU fallback = the GPU device transparently runs the
CPU reference (correct result, no GPU speedup).

**Every dtype runs on every layer** — the matrix is **FP32 / FP16 / BF16 / INT16 / INT8 / INT4**.
Compute is FP32 internally; the dtype only changes the wire byte layout (FP16/BF16 are 2 bytes,
the rest hand over FP32 values, matching `poly/accel_intel.go`). **BF16 is the Apple-native
addition** to the bridge — Apple silicon has native bfloat16.

---

## Numbers that matter (from `apple.txt`)

### Determinism — production-grade

Repeat-forward drift (same input twice) is **💎 EXACT on every cell, both devices**:

| Check | Pass | Total |
|---|---:|---:|
| Apple infer repeat (**CPU**) | **180** | 180 |
| Apple infer repeat (**GPU**) | **180** | 180 |

Byte-identical outputs run to run — the single most important property for using either device
as a real execution target.

### Loom ↔ Apple parity

| Check | ≤ INDUS | Total |
|---|---:|---:|
| Loom ↔ Apple parity (**GPU**) | **132** | 180 |
| Loom ↔ Apple parity (**CPU**) | 78 | 180 |

Full GPU bucket spread: 💎 6 · ✅ 126 · 🟨 36 (LOWBIT) · ❌ 12 (BROKE).
Full CPU bucket spread: 💎 6 · ✅ 72 · 🟨 54 · 🟤 27 (H-DRIFT) · ❌ 21.

The `≤ INDUS` count is **higher on the GPU** than the CPU because the GPU target carries a looser
parity tolerance (GPU math legitimately reorders reductions) — the raw drift magnitudes for the
two devices are nearly identical. So "GPU parity looks better" is a tolerance-bucket effect, not
the GPU being more precise than the CPU reference.

**Parity highlights:**
- **Softmax** is effectively exact (`~1e-9` FP32, `~1e-6` FP16/BF16) — the standout.
- **MatMul / MHA / Conv** sit at 🟨 LOWBIT → 🟤 H-DRIFT as tiers grow (FP32/FP16/BF16 accumulation
  order vs Loom's CPU matmul).
- **LayerNorm / RMSNorm** are ❌ BROKE at ~1.8 for FP32/BF16 — the CPU reference fills its own
  constant scale/bias instead of Loom's weights (identical root cause to the Intel bridge norms).
- **INT8** MatMul/Conv/MHA drift is large (3 → 31) and goes ❌ BROKE on the large tier — the
  same Loom-dequant-matmul vs plugin-FP32 mismatch seen on Intel/Qualcomm.

### Speed — GPU for big MAC, CPU for elementwise

Two clear winners depending on op class (Spd = Loom ÷ Apple, higher = Apple faster):

**Metal GPU wins on large MatMul / MHA** (the GPU has a ~0.5 ms launch floor, so it only pays off
once the tensor is big enough):

| Tier | Layer | DType | Loom ms | Metal GPU ms | Speed |
|---|---|---|---:|---:|---:|
| large | MHA-MatMul | FP16 | 3.079 | 0.571 | **5.4×** |
| large | MHA-MatMul | INT8 | 2.096 | 0.399 | **5.3×** |
| large | MatMul | FP16 | 2.613 | 0.562 | **4.6×** |
| large | MatMul | INT8 | 2.322 | 0.571 | **4.1×** |
| large | MatMul | BF16 | 1.603 | 0.598 | **2.7×** |

**Apple CPU reference wins big on elementwise** (tight native loop, no launch cost):

| Tier | Layer | DType | Loom ms | Apple CPU ms | Speed |
|---|---|---|---:|---:|---:|
| medium | ReLU | INT4 | 0.903 | 0.010 | **90×** |
| large | ReLU | INT4 | 1.880 | 0.020 | **94×** |
| large | GELU | INT4 | 2.395 | 0.029 | **83×** |
| large | Sigmoid | INT4 | 2.179 | 0.027 | **81×** |
| medium | GELU | INT8 | 0.869 | 0.017 | **51×** |

**Where both Apple devices lose:**
- **Small tier, GPU** — the ~0.5 ms floor swamps sub-0.01 ms compute (`0.02×` on MatMul/ReLU).
- **Conv1D / Conv2D, every tier** — both devices run the CPU reference conv, which is ~0.24–0.27×
  Loom's own optimized CPU conv. Don't offload Conv yet.
- **Norms** — tiny ops, CPU-reference-only, ~0.3–0.6× (and parity broken anyway).

### One-time compile cost

`SyncToAccel` compiles each layer once. The CPU reference is ~free; the **MPSGraph build** is the
cost on the GPU device — and it is cheap compared to Hexagon/OpenVINO:

| Device | Compile (ms) |
|---|---:|
| Apple CPU reference | ~0.00 |
| Metal GPU (MPSGraph MatMul/act) | ~0.4 – 5.8 |

Compile is paid **once per layer/device**; steady inference is the sub-ms numbers above.

---

## How it plugs into Loom (unchanged dispatch model)

```go
reg, err := poly.DiscoverAppleAccel(accel.AccelConfig{
    AppleSO: accel.DefaultApplePath(),
})
if err != nil { /* no plugin — stay on Loom CPU */ }
defer reg.Close()

net, _ := poly.BuildNetworkFromJSON(spec)
net.Accel = reg

net.Layers[0].ExecTarget = accel.ExecAppleGPU // or ExecAppleCPU
_ = net.SyncToAccel("medium")                 // compile once + upload weights

out, _, _ := poly.ForwardPolymorphic(net, input)
```

### `ExecTarget` values

| Value | Runs on |
|---|---|
| `accel.ExecLoomCPU` | Default — Go poly CPU |
| `accel.ExecAppleCPU` | Apple plugin CPU reference (deterministic parity anchor) |
| `accel.ExecAppleGPU` | Metal / MPSGraph (falls back to the CPU reference per-op when MPSGraph can't build) |

### Accelerator devices

| Device | Backend | Data types | Role |
|---|---|---|---|
| **GPU** | Metal / MPSGraph | FP32/FP16/BF16 + INT* (as fp32 wire) | The fast path for large MatMul / MHA |
| **CPU** | portable C++ reference | all 6 bench dtypes | Parity anchor + fallback for unbuilt GPU ops |
| **ANE** (Neural Engine) | — | — | Not wired — reachable only indirectly via Core ML (future) |

---

## Known gaps — what's left

Honestly *experimental*. What's still missing or weak:

| Gap | Detail |
|---|---|
| **Conv1D / Conv2D on GPU** | Not built on MPSGraph — both devices run the CPU reference, which is slower than Loom's own CPU conv. Biggest speed opportunity left. |
| **GELU on GPU** | Falls back to CPU reference; MPSGraph has the ops, just not wired. |
| **LayerNorm / RMSNorm parity** | ❌ BROKE — CPU reference uses internal constant scale/bias instead of Loom's uploaded weights. Needs weight bake (same fix as Intel). |
| **INT8 MAC drift** | MatMul/Conv/MHA INT8 drift is large and BROKE on large tiers (Loom dequant matmul vs plugin FP32). |
| **FP16/BF16 precision** | Compute is FP32 internally; FP16/BF16 only change the wire layout, so no true half-precision GPU math yet. |
| **ANE** | Apple Neural Engine is not a Metal device; reaching it needs a Core ML path (future plugin mode). |
| **Training / backward** | Forward-only; backward stays on Loom CPU. |
| **JSON `exec` field** | No `"exec": "apple-gpu"` yet — `ExecTarget` is set programmatically. |
| **Zero-copy I/O** | Per-hop tensor → `[]byte` → Metal buffer copies. |

---

## When to offload

| Target | Offload | Skip |
|---|---|---|
| **Apple GPU** | **large** MatMul / MHA-MatMul (2.4–5.4×) | small tensors (0.5 ms floor); Conv (CPU fallback); norms |
| **Apple CPU** | elementwise **ReLU / GELU / Sigmoid** at medium+ (16–94×); small MatMul | large MatMul/MHA (single-threaded ref, 0.1–0.3×); Conv |
| **Either** | many steady forwards (compile paid once) | one-shot micro-ops |

Rule of thumb: **big matmuls → Metal GPU; heavy elementwise → Apple CPU; convolutions → keep on
Loom CPU** until the GPU conv path lands.

---

## Build & run

No SDK download — just Xcode command-line tools (for the Metal frameworks). Full notes in
[`accel/apple/README.md`](../accel/apple/README.md). Short version:

```bash
cd accel/apple
./build.sh                 # CMake → build/libloom_accel_apple.dylib

cd ../../lucy
CGO_ENABLED=1 go run .      # → [13] Apple GPU bridge
#   [4] medium DispatchLayer suite   (fast)
#   [5] full 10×6×3 matrix           (the apple.txt tables above)
#   [0] raw CABI matrix (all 15 layers, direct plugin)
```

`accel.DefaultApplePath()` walks up from cwd for `accel/apple/build/libloom_accel_apple.dylib`,
or set `LOOM_ACCEL_APPLE_DYLIB`.

---

## See also

- [`accelerators.md`](accelerators.md) — the accel model + Intel NPU bridge
- [`snapdragon_npu.md`](snapdragon_npu.md) — the Qualcomm/Hexagon mirror
- [`accel/apple/README.md`](../accel/apple/README.md) — build / dtypes / layout
- [`numerical_types.md`](numerical_types.md) — Loom's 21 DTypes vs vendor bench dtypes
- [`dispatch.md`](dispatch.md) — `DispatchLayer` routing
