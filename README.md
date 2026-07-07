# LOOM: Universal Bit-Perfect Deterministic AI Engine

[![npm version](https://img.shields.io/npm/v/@openfluke/welvet.svg)](https://www.npmjs.com/package/@openfluke/welvet)
[![npm downloads](https://img.shields.io/npm/dm/@openfluke/welvet.svg)](https://www.npmjs.com/package/@openfluke/welvet)
[![PyPI version](https://img.shields.io/pypi/v/welvet.svg)](https://pypi.org/project/welvet/)
[![PyPI downloads](https://img.shields.io/pypi/dm/welvet.svg)](https://pypi.org/project/welvet/)
[![pub package](https://img.shields.io/pub/v/welvet.svg)](https://pub.dev/packages/welvet)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**"The SQLite of AI" — A Polyglot Neural Engine with Bit-Exact Reproducibility**

Loom is a **Deterministic Neural Virtual Machine (DNVM)** engineered for absolute numerical consistency and extreme efficiency. It guarantees **bitwise-identical results** across all platforms, backends, and language bindings, bypassing memory bandwidth bottlenecks through polymorphic dispatch and volumetric 3D modeling.

![Loom Overview](./loom_overview.jpg)

## 🌐 The Polyglot Solution
Loom is designed as a universal runtime that prioritizes portability and sovereignty:

- **True "Copy-Paste" Portability**: Models are language-agnostic. Move weights and logic between Go, Python, C#, and WASM without translation layers.
- **Write Once, Run Everywhere**: A standardized format that performs identically on Browser (WASM/WebGPU), Mobile (iOS/Android), and Desktop (Linux/Windows/macOS).
- **Universal Import**: Direct ingestion from major frameworks—zero vendor lock-in.
- **Active Edge Training**: Full backpropagation enabled on-device. No "frozen brains"; Loom learns from user interaction at the edge.
- **Sovereign & Private**: Zero cloud dependencies. User data and model execution remain 100% local.


## 💎 The Bedrock Philosophy
Loom is a **"Bedrock Edition"** neural engine. Unlike standard frameworks that build on top of high-level abstractions, Loom is designed at the bit-level to bypass the physical memory limitations of consumer hardware. 

- **Cross-Platform Determinism**: 0.0000000000 difference between CPU and GPU, x86 and ARM, native and browser.
- **Universal Precision**: Native support for 21 numerical types (FP64 to 1-bit Binary), allowing Loom to "morph" precision to match specific silicon preferences.
- **Bit-Perfect Identity**: Verified across hundreds of permutations with 0.000000% mathematical divergence.

## 🚀 The Technical Pillars (Final Form)
The project has transitioned to the **Multi-numerical POLYmorphic Volumetric Tiled-tensor Dispatcher (M-POLY-VTD)** core.

- **Step neural mesh**: A living mesh architecture with clock-cycle accurate updates and temporal feedback loops that simulate biological neural firing.
- **DNA Engine**: A hierarchical spatial correlation system that extracts topological "signatures" of models, enabling high-fidelity comparison and "Logic Shift" detection in 3D space.
- **Tween (neural target propagation)**: A robust alternative to backpropagation that uses localized, gap-based Hebbian learning to bridge the difference between actual and idealized activations. We call it **tween** in code (`tween.go`); papers often use *target propagation* or related names.
- **Bit-Packed Persistence**: An idempotent serialization tunnel that achieves up to **98.4% compression**, allowing extreme model sizes to fit in consumer RAM/VRAM.

## 📂 Project Structure
- **[`poly/`](./poly/)**: The current-generation engine core (M-POLY-VTD). Active development.
- **[`lucy/`](./lucy/)**: Interactive harness — Poly Talk, ENTITY Talk, seven-layer CPU suite, HF download.
- **[`planetbridging/`](./planetbridging/)**: Planet → Loom bridging POC (**v0.5.0** complete in-tree; **releases after Loom 0.80**).
- **[`welvet/dart/`](./welvet/dart/)**: Flutter/Dart FFI plugin — [`welvet` on pub.dev](https://pub.dev/packages/welvet) (`flutter pub add welvet`).

## 🛠️ Getting Started
For technical deep-dives into M-POLY-VTD, refer to the **[`docs/`](./docs/)** index ([`docs/index.md`](./docs/index.md)) and benchmarks within the [`poly/`](./poly/) core. Topics include deployment, GPU, layers — plus **donate compute** (LAN TCP, [`docs/donate_compute.md`](./docs/donate_compute.md)) and **TANHI** UDP layer telemetry for the SoulGlitch HUD ([`docs/tanhi.md`](./docs/tanhi.md)).

Loom provides bit-exact reproducibility across:
- **Go** (Native)
- [**TypeScript/Node.js**](https://www.npmjs.com/package/@openfluke/welvet) (@openfluke/welvet)
- **Browser** (WASM + WebGPU)
- [**Python**](https://pypi.org/project/welvet/) (`pip install welvet`)
- [**Flutter/Dart**](https://pub.dev/packages/welvet) (`welvet` FFI plugin — iOS, Android, Linux, macOS, Windows)
- **C#/.NET** (Welvet) - *(In Development)*

## 📊 Versioning & Roadmap
Loom uses a mathematical versioning system derived from a strictly verified checklist in [`poly/README.md`](./poly/README.md) (row counts and completion ratio are maintained there).

### **Current Version: 0.82.0 — CURRENT** (from **0.81.0**)
- **Completion Ratio**: 79.6% (**117 / 147** checklist rows on `adjustments`)
- **Codename**: **0.82.0 "Snapdragon Bridge"** — **SIMD** CPU fast-path (AVX2/FMA on x86-64, NEON on ARM64) and a second experimental NPU vendor: **Qualcomm/Hexagon** (QNN, Windows ARM64, Lucy **[12]**) joining **Intel CPU+NPU** (Lucy **[9]**) via `poly/accel`.
- **Status**: Two vendor NPU plugins shipped (Intel on Linux, Qualcomm on Windows ARM64) — experimental, forward-only per-layer offload; GPU + ENTITY from 0.80 remain production paths. See [`docs/v082_release.md`](./docs/v082_release.md), [`docs/snapdragon_npu.md`](./docs/snapdragon_npu.md), [`docs/simd.md`](./docs/simd.md) and [`docs/accelerators.md`](./docs/accelerators.md).
- **Milestones**:
    - **v0.79.0 "Bedrock Validation"** ✅ — See [`docs/bedrock_validation.md`](./docs/bedrock_validation.md).
    - **v0.80.0 "Native Ship"** ✅ — See [`docs/v080_release.md`](./docs/v080_release.md).
    - **v0.81.0 "Accelerator Bridge"** ✅ — See [`docs/v081_release.md`](./docs/v081_release.md).
- **Next Target — v0.83**: whole-model `.entity` → NPU lowering; NPU parity suite vs WebGPU; GPU backward (SwiGLU/MHA); AccelPlanner + JSON `exec`; Donate Compute live inference.

For a detailed breakdown of the roadmap and version calculation, see [poly/README.md](./poly/README.md#📊-true-version-calculation).

---

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

---

*Loom: Universal precision. Volumetric freedom. Bedrock performance.*

**Made with ❤️ by Openfluke**