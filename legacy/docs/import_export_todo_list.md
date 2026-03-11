# AI Model Format Support Matrix

This document outlines the planned support for various AI model file formats in the **OpenFluke/Loom** engine. It categorizes formats by the engineering effort required to implement a parser **from scratch in Go** (without C-bindings/cgo).

## File Type Summary

| Framework | Extension(s) | Format Type | Go Implementation Effort |
| :--- | :--- | :--- | :--- |
| **Hugging Face** | `.safetensors` | JSON Header + Raw Bytes | 🟢 **Easiest** |
| **TensorFlow.js** | `.json` + `.bin` | JSON + Raw Bytes | 🟢 **Easy** (Note: Generation via Py3.13 broken) |
| **ONNX** | `.onnx` | Protobuf | 🟡 **Medium (Standard)** |
| **TensorFlow Lite** | `.tflite` | FlatBuffers | 🟡 **Medium** |
| **Llama.cpp** | `.gguf` | Custom Binary | 🟡 **Medium** |
| **Core ML (Apple)** | `.mlmodel` | Protobuf | 🟠 **High (Stretch Goal)** |
| **TensorFlow** | `.pb` | Protobuf (GraphDef) | ❌ **Use Converter** |
| **Keras** | `.h5`, `.keras` | HDF5 | ❌ **Use Converter** |
| **PyTorch** | `.pt`, `.pth` | Python Pickle | ❌ **Use Converter** |
| **Scikit-Learn** | `.pkl` | Python Pickle | ❌ **Use Converter** |

---

## Technical Details & Implementation Plan

### 🟢 Tier 1: The "Go-Native" Friendly Formats
*Target: Immediate Support*

**1. `.safetensors` (Hugging Face)**
* **Structure:** An 8-byte length prefix, a JSON header describing tensor shapes/types, followed immediately by the raw byte buffers.
* **Implementation:** Use `encoding/json` for the header and `os.File.Seek` / `io.ReadFull` for the data. No external dependencies required.
* **Role:** The primary "Native" format for saving/loading Loom models.

**2. TensorFlow.js (`model.json` + `group1-shard1of1.bin`)**
* **Structure:** The JSON file defines the graph topology and weights manifest. The `.bin` files contain flat byte arrays.
* **Implementation:** Trivial to parse using Go's standard library. Good for web compatibility.

### 🟡 Tier 2: Schema-Based & Structured Binaries
*Target: Core Compatibility*

**3. `.onnx` (Open Neural Network Exchange)**
* **Structure:** Protocol Buffers (Protobuf).
* **Implementation:** Requires `google.golang.org/protobuf`. We compile the `onnx.proto` definition into Go structs.
* **Role:** The "Universal Adapter." If a user has a model from PyTorch/TF/Scikit, they convert it to ONNX, and we import the ONNX.

**4. `.tflite` (TensorFlow Lite)**
* **Structure:** Google FlatBuffers.
* **Implementation:** Requires compiling the schema with `flatc` to generate Go code.
* **Pros:** Extremely fast access (zero-copy parsing). Good for edge/mobile use cases.

**5. `.gguf` (Llama.cpp / GGUF)**
* **Structure:** Custom binary format (Header -> Key-Value Pairs -> Tensor Info -> Data).
* **Implementation:** Requires writing a custom binary reader using `encoding/binary`.
* **Role:** Essential for supporting Large Language Models (LLMs) locally.

### 🟠 Tier 3: The "Stretch" Goal

**6. Core ML (`.mlmodel`)**
* **Structure:** Protocol Buffers (Protobuf).
* **Status:** **Supported eventually.**
* **Challenge:** While it uses Protobuf (like ONNX), the specification is Apple-specific and massive. Implementing the full spec is high effort, but valuable for macOS/iOS integration later.

---

## ❌ Tier 4: The "Stupid" List (Converter Required)

**Policy:** We will **not** write native parsers for these formats. They contain embedded logic, virtual machine instructions, or file-system-in-a-file complexities that are out of scope for a clean Go runtime.

**The Strategy:** Users must convert these models to **ONNX** or **Safetensors** using Python scripts provided in `tools/converters/`.

**1. PyTorch (`.pt` / `.pth`)**
* **Reason:** These files are serialized Python objects (Pickle). Reading them requires re-implementing a Python Virtual Machine to execute opcodes (`GLOBAL`, `BUILD`).
* **Solution:** `torch.onnx.export(...)`

**2. TensorFlow Native (`.pb` / SavedModel)**
* **Reason:** Contains "Control Flow" operators (`Switch`, `Merge`, `NextIteration`) designed for Google's distributed data centers. Requires a complex VM to execute.
* **Solution:** `tf2onnx`

**3. Keras (`.h5`)**
* **Reason:** Uses HDF5, which is a complex "file system within a file" specification.
* **Solution:** `tf.lite.TFLiteConverter` or `tf2onnx`.