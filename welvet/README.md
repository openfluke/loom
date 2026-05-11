# Welvet: The Loom SDK & Polyglot Bindings

**Welvet** is the high-level fabric woven by the Loom engine. it provides the polyglot bridges and SDKs required to interface with the **M-POLY-VTD** core from any programming language.

## 📂 Structure
- **[`cabi/`](./cabi/)**: The core C-ABI bridge (Go). This generates the shared libraries (`.dll`, `.so`, `.dylib`) used by other language bindings.
- **[`cabi/`](./cabi/)**: The core C-ABI bridge (Go). This generates the shared libraries (`.dll`, `.so`, `.dylib`) used by all language bindings.
- **[`python/`](./python/)**: Native Python SDK (`welvet`). High-level OOP wrappers for Transformers, Populations, and Volumetric Networks.
- **[`typescript/`](./typescript/)**: Isomorphic TypeScript/Node.js SDK (`@openfluke/welvet`) with WebGPU and WASM support.

## 🛠️ C-ABI Features (v0.75.0)
The C-ABI bridge now achieves **100% Functional Parity** (345+ features) with the Loom engine:
- **Transformer Support**: Full mapping for `TokensToTensor`, `ForwardFull`, and `KV-Cache` management.
- **NEAT Evolution**: Direct access to population-scale genetic mutation and evolutionary `Evolve` cycles.
- **Step mesh controls**: Precise, clock-cycle accurate control over neural mesh propagation with 3D coordinate guarding.
- **Neural Target Propagation**: Direct access to gap-bridging Hebbian learning.
- **DNA Engine**: Exported topological signatures for model comparison and splicing.

## 🚀 Building
To build the shared library:
```bash
go build -o welvet/cabi/welvet.dll -buildmode=c-shared welvet/cabi/main.go
```

*Welvet: Bridging the gap between bit-level performance and polyglot flexibility.*
