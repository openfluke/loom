# Paper 1: Loom — A Polyglot, Zero-Dependency Runtime for Embeddable AI

> **Target Venue:** MLSys, OSDI, or similar systems conference

## Abstract

Modern AI deployment requires navigating a complex dependency stack: Python runtimes, CUDA drivers, C++ libraries, and platform-specific binaries. This creates friction for edge deployment, embedded systems, and polyglot environments. We present **Loom**, a neural network framework written in pure Go that compiles to a single zero-dependency binary and exposes a universal C-ABI, enabling identical model behavior across Python, C#, TypeScript, Rust, WebAssembly (browser), and native applications without runtime dependencies.

---

## 1. Problem Statement

### The AI Deployment Problem

| Challenge | Traditional Stack | Loom Solution |
|-----------|-------------------|---------------|
| **Runtime Dependencies** | Python 3.x, PyTorch/TF, CUDA, cuDNN | None (single binary) |
| **Binary Size** | 500MB+ (PyTorch), 200MB+ (TF Lite) | ~10MB (Loom binary) |
| **Cross-Platform** | Separate builds per platform | One C-ABI, identical behavior |
| **Browser Deployment** | TensorFlow.js (separate implementation) | Same WASM binary, same code |
| **Edge/Embedded** | Complex cross-compilation | `GOOS=linux GOARCH=arm64 go build` |

### Why This Matters

1. **Edge AI** requires small, self-contained binaries
2. **Polyglot backends** (Go services calling Python models) create operational complexity
3. **Browser ML** typically requires separate reimplementation
4. **Embedded systems** cannot run Python interpreters

---

## 2. Technical Approach

### 2.1 Pure Go Implementation

Loom is implemented entirely in Go with zero CGO dependencies for core functionality:

```go
// Build with zero external dependencies
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o loom_linux ./...
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -o loom.exe ./...
CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -o loom_mac ./...
```

This produces identical behavior across all platforms.

### 2.2 Universal C-ABI

A single shared library (`.so`, `.dll`, `.dylib`) exposes all functionality:

```c
// Core API (identical across all languages)
extern void CreateLoomNetwork(const char* jsonConfig);
extern const char* LoomForward(const char* inputJson);
extern const char* LoomTrain(const char* trainingJson);
extern const char* LoomSaveModel(const char* modelId);
extern void LoomLoadModel(const char* jsonData, const char* modelId);
```

**Language Bindings:**

| Language | Binding Method | Example |
|----------|----------------|---------|
| Python | ctypes/cffi | `welvet.forward_simple([0.1, 0.2])` |
| C# | P/Invoke | `LoomForward(inputJson)` |
| TypeScript | WASM imports | `loom.forward([0.1, 0.2])` |
| Rust | FFI | `loom_forward(input.as_ptr())` |
| C/C++ | Direct linking | `LoomForward(input)` |

### 2.3 WebAssembly With Full Training

Unlike inference-only WASM frameworks, Loom supports **complete training in the browser**:

```javascript
// Browser training - identical to server-side
const trainingData = [
    { input: [0.1, 0.2], target: [1.0] },
    { input: [0.8, 0.9], target: [0.0] }
];

const result = loom.train(JSON.stringify({
    epochs: 100,
    learning_rate: 0.01,
    data: trainingData
}));
```

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Loom Core (Pure Go)                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ Dense   │  │ Conv2D  │  │ LSTM    │  │ MHA     │  ... 10+   │
│  │ Layer   │  │ Layer   │  │ Layer   │  │ Layer   │  layers    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │
│  ┌─────────────────────────────────────────────────┐            │
│  │         Generic Tensor Backend (Generics)        │            │
│  │    float32 | float64 | int8 | int16 | int32     │            │
│  └─────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   C-ABI       │    │   WASM        │    │   Native Go   │
│   (.so/.dll)  │    │   (.wasm)     │    │   (binary)    │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Python/C#/    │    │ Browser       │    │ Go Services   │
│ Rust/C++      │    │ Node.js       │    │ CLI Tools     │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

## 4. Experimental Results

### 4.1 Binary Size Comparison

| Framework | Binary/Package Size | Dependencies |
|-----------|---------------------|--------------|
| PyTorch (CPU) | 500MB+ | Python, libstdc++, MKL |
| TensorFlow Lite | 200MB+ | C++ runtime |
| ONNX Runtime | 50MB+ | C++ runtime |
| **Loom** | **~10MB** | **None** |

### 4.2 Multi-Precision Storage

From `tva/test_0_0_7.go` Part 2:

```
Layer Type  | float32  | float64  | int32    | int16    | int8
------------|----------|----------|----------|----------|--------
MHA         | 116.5KB  | 232.1KB  | 116.5KB  | 58.7KB   | 29.9KB
SwiGLU      | 29.9KB   | 58.8KB   | 29.9KB   | 15.4KB   | 8.2KB
Dense       | 970B     | 970B     | 970B     | 962B     | 954B
```

**Result:** 4x storage reduction with int8 for complex layers.

### 4.3 Cross-Platform Consistency

All platforms produce **bit-for-bit identical** results:

```
Platform      | MSE Difference | Model Size
--------------|----------------|------------
Go Native     | 0.000000       | 25.6KB
Python ctypes | 0.000000       | 25.6KB
C# P/Invoke   | 0.000000       | 25.6KB
TypeScript    | 0.000000       | 25.6KB
Browser WASM  | 0.000000       | 25.6KB
```

---

## 5. Code References

| Component | Path | Description |
|-----------|------|-------------|
| C-ABI | [`cabi/main.go`](../cabi/main.go) | C foreign function interface |
| WASM | [`wasm/main.go`](../wasm/main.go) | WebAssembly exports |
| Python | [`python/src/welvet/`](../python/src/welvet/) | Python bindings |
| C# | [`csharp/`](../csharp/) | .NET bindings |
| TypeScript | [`typescript/`](../typescript/) | Node.js/Browser bindings |
| Test Suite | [`tva/test_0_0_7.go`](../tva/test_0_0_7.go) | Comprehensive validation |

---

## 6. How to Reproduce

### Build the C-ABI Library

```bash
cd cabi
./build.sh  # Produces libloom.so / loom.dll / libloom.dylib
```

### Run Cross-Platform Tests

```bash
# Go Native
go run tva/test_0_0_7.go

# Python
cd python && pip install -e . && python examples/all_layers_test.py

# TypeScript
cd typescript && npm install && npm run test

# C#
cd csharp && dotnet run --project examples/
```

### Build WASM

```bash
cd wasm
./build.sh  # Produces main.wasm
# Open example.html in browser
```

---

## 7. Conclusion

Loom demonstrates that a **pure Go neural network framework** can achieve:

1. **Zero-dependency deployment** via single binary
2. **Universal polyglot support** via C-ABI
3. **Browser training** via WASM
4. **4x storage reduction** via native multi-precision
5. **Bit-for-bit cross-platform consistency**

This eliminates the "Python tax" for production AI systems while maintaining research flexibility.

---

**Related Papers:**
- [Paper 2: StepTweenChain Optimizer](research_paper_2_steptween.md)
- [Paper 3: Heterogeneous MoE](research_paper_3_heterogeneous_moe.md)
- [Paper 4: Native Integer Training](research_paper_4_integer_training.md)
- [Paper 5: Spatially-Adaptive Stitching](research_paper_5_arc_stitching.md)
- [Paper 6: Universal Precision & WebGPU](research_paper_6_universal_precision.md)
