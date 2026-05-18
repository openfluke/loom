# LOOM: Universal Bit-Perfect Deterministic AI Engine

[![npm version](https://img.shields.io/npm/v/@openfluke/welvet.svg)](https://www.npmjs.com/package/@openfluke/welvet)
[![npm downloads](https://img.shields.io/npm/dm/@openfluke/welvet.svg)](https://www.npmjs.com/package/@openfluke/welvet)
[![PyPI version](https://img.shields.io/pypi/v/welvet.svg)](https://pypi.org/project/welvet/)
[![PyPI downloads](https://img.shields.io/pypi/dm/welvet.svg)](https://pypi.org/project/welvet/)
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
- **[`poly/`](./poly/)**: The current-generation engine core (M-POLY-VTD). This is where active development happens.
- **[`legacy/`](./legacy/)**: Historical codebase and previous iterations of Loom.

## 🛠️ Getting Started
For technical deep-dives into M-POLY-VTD, refer to the **[`docs/`](./docs/)** index ([`docs/index.md`](./docs/index.md)) and benchmarks within the [`poly/`](./poly/) core. Topics include deployment, GPU, layers — plus **donate compute** (LAN TCP, [`docs/donate_compute.md`](./docs/donate_compute.md)) and **TANHI** UDP layer telemetry for the SoulGlitch HUD ([`docs/tanhi.md`](./docs/tanhi.md)).

Loom provides bit-exact reproducibility across:
- **Go** (Native)
- [**TypeScript/Node.js**](https://www.npmjs.com/package/@openfluke/welvet) (@openfluke/welvet)
- **Browser** (WASM + WebGPU)
- [**Python**](https://pypi.org/project/welvet/) (`pip install welvet`)
- **C#/.NET** (Welvet) - *(In Development)*

## 📊 Versioning & Roadmap
Loom uses a mathematical versioning system derived from a strictly verified checklist in [`poly/README.md`](./poly/README.md) (row counts and completion ratio are maintained there).

### **Current Version: 0.78.0 — CURRENT**
- **Completion Ratio**: 76.1% (108 / 142 checklist rows verified — *ASM CPU* wave)
- **Status**: **0.78.0 "ASM CPU"** adds **Plan 9 dense forward** (`poly/asm`, `UseAsmForward`) on top of **Operation Mesh**: **Donate Compute** (TCP LAN), **TANHI** (UDP layer telemetry for SoulGlitch), **Lucy** (HF pull + compile-on-the-go + layer suites with **Go · ASM · GPU** timers), **native JSON persistence** per layer dtype (not FP32-only checkpoints), **Qwen3-class** ingest, and **tiled-first** forward/backward — still with unified **SC/MC Tiling**, stabilized **step mesh 3D grid**, and **C-ABI parity** for Python and TypeScript.
    - > [!NOTE]
    - > **Numerical Tiling**: Introduces specialized Single-Core (SC) and Multi-Core (MC) profiles, optimizing register pressure vs. high-bandwidth L1/L2 cache throughput.
- **Milestones**:
    - **v0.75.0 "Multi-Core Symphony"** ✅ — Tiling across the dispatcher; stabilized volumetric hopping; `welvet` parity.
    - **v0.76.0 "Operation Mesh"** ✅ — Wire protocols, LM tooling burst, RAM-aware load path, telemetry. See [poly/README.md § v0.76.0](./poly/README.md#v0760--operation-mesh-previous).
    - **v0.78.0 "ASM CPU"** ✅ — Dense forward Plan 9 kernels (21 dtypes), Lucy ASM columns, native save/load per dtype. See [poly/README.md § v0.78.0](./poly/README.md#v0780--asm-cpu-current).
- **Next Target — v0.8.0 "Edge-First"**: Thermal-aware scheduling, UMA pinning, command-buffer graphing for mobile and wearable deployment.
- **Next Steps**: Command Graph Buffering; Thermal-Aware hardware scheduling.

For a detailed breakdown of the roadmap and version calculation, see [poly/README.md](./poly/README.md#📊-true-version-calculation).

---

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

---

*Loom: Universal precision. Volumetric freedom. Bedrock performance.*

**Made with ❤️ by Openfluke**