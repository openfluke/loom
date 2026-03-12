# Welvet: The Loom SDK & Polyglot Bindings

**Welvet** is the high-level fabric woven by the Loom engine. it provides the polyglot bridges and SDKs required to interface with the **M-POLY-VTD** core from any programming language.

## 📂 Structure
- **[`cabi/`](./cabi/)**: The core C-ABI bridge (Go). This generates the shared libraries (`.dll`, `.so`, `.dylib`) used by other language bindings.
- **[Future] `python/`**: Native Python bindings via ctypes/welvet.
- **[Future] `csharp/`**: .NET bindings for ultra-deterministic AI in Unity/Godot.
- **[Future] `typescript/`**: Node.js and WASM extensions.

## 🛠️ C-ABI Features
The new C-ABI bridge exposes the advanced mechanics of the Loom engine:
- **Handle-Based Management**: Scalable management of `VolumetricNetwork` and `SystolicState` instances.
- **Systolic Mesh Controls**: Precise, clock-cycle accurate control over neural mesh propagation.
- **Neural Target Propagation**: Direct access to gap-bridging Hebbian learning.
- **DNA Engine**: Exported topological signatures for model comparison.

## 🚀 Building
To build the shared library:
```bash
go build -o welvet/cabi/welvet.dll -buildmode=c-shared welvet/cabi/main.go
```

*Welvet: Bridging the gap beTargetProp bit-level performance and polyglot flexibility.*
