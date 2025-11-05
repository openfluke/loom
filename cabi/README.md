# LOOM C ABI

C Foreign Function Interface (FFI) for the LOOM neural network framework. Allows calling LOOM from C, C++, Rust, Python (ctypes/cffi), and other languages that support C ABI.

## Features

- **All 5 Layer Types**: Dense, Conv2D, Multi-Head Attention, RNN, LSTM fully supported
- **Registry-based Initialization**: Dynamic layer creation via `CallLayerInit()` for any layer type
- **Full Training Support**: Complete forward/backward propagation with automatic gradients
- **Multi-platform support**: Linux, macOS, Windows, Android, iOS
- **Cross-compilation**: Build for multiple architectures from a single machine
- **Handle-based object management**: Safe lifecycle management with sync.Mutex
- **JSON parameter passing**: Simple, language-agnostic API
- **Reflection-based method calling**: Automatic exposure of all Network methods
- **Type conversion**: Automatic Go ↔ JSON type mapping (including custom types)
- **GPU support**: Enable/disable GPU acceleration (Dense, Conv2D, Attention)
- **Model serialization**: Save/load networks as JSON strings
- **Introspection**: List available methods and get object info

## Building

### Quick Start (Current Platform)

```bash
./build.sh
```

This builds for your current OS and architecture, placing output in `compiled/<platform>_<arch>/`.

### Multi-Platform Builds

```bash
# Build for specific platform and architecture
./build_all.sh linux arm64          # Linux ARM64
./build_all.sh macos universal      # macOS Universal Binary
./build_all.sh windows x86_64       # Windows 64-bit
./build_all.sh android arm64        # Android ARM64
./build_all.sh ios xcframework      # iOS XCFramework

# Build all architectures for current platform
./build_all.sh all

# Clean before building
./build_all.sh --clean linux x86_64

# Show help
./build_all.sh --help
```

### Supported Platforms and Architectures

| Platform    | Architectures                           | Output                           | Notes                       |
| ----------- | --------------------------------------- | -------------------------------- | --------------------------- |
| **Linux**   | x86_64, arm64, armv7, x86               | `libloom.so`                     | Native or cross-compile     |
| **macOS**   | x86_64, arm64, universal                | `libloom.dylib`                  | Universal = Fat binary      |
| **Windows** | x86_64, x86, arm64                      | `libloom.dll`                    | Requires mingw-w64          |
| **Android** | arm64, armv7, x86_64, x86               | `libloom.so`                     | Requires Android NDK        |
| **iOS**     | arm64, x86_64_sim, arm64_sim, universal | `libloom.dylib` / `.xcframework` | Requires Xcode (macOS only) |

### Output Structure

All builds are organized in the `compiled/` directory:

```
compiled/
├── linux_x86_64/
│   ├── libloom.so
│   ├── libloom.h
│   └── simple_bench
├── linux_arm64/
│   ├── libloom.so
│   ├── libloom.h
│   └── simple_bench
├── macos_universal/
│   ├── libloom.dylib
│   ├── libloom.h
│   └── simple_bench
├── windows_x86_64/
│   ├── libloom.dll
│   ├── libloom.h
│   └── simple_bench.exe
├── android_arm64/
│   ├── libloom.so
│   ├── libloom.h
│   └── simple_bench
└── ios_xcframework/
    └── LOOM.xcframework/
```

### Cross-Compilation Requirements

**Windows** (from Linux/macOS):

```bash
# Ubuntu/Debian
sudo apt install mingw-w64

# macOS
brew install mingw-w64
```

**Android**:

```bash
# Download Android NDK from https://developer.android.com/ndk/downloads
# Or install via Homebrew on macOS:
brew install --cask android-ndk

# Set environment variable (add to ~/.zshrc for persistence)
export ANDROID_NDK_HOME=/opt/homebrew/share/android-ndk

# Verify installation
ls $ANDROID_NDK_HOME/toolchains/llvm/prebuilt/
```

**Linux ARM64** (cross-compile from macOS/x86_64):

```bash
# macOS
brew install aarch64-unknown-linux-gnu

# Ubuntu/Debian
sudo apt install gcc-aarch64-linux-gnu
```

**Windows ARM64** (native ARM64 on Linux, fallback to x86_64 on macOS):

```bash
# macOS: Homebrew mingw-w64 lacks ARM64 support
#        Script will use x86_64-w64-mingw32-gcc as fallback (produces x86_64 binary)
brew install mingw-w64

# Linux: True ARM64 Windows builds
sudo apt install gcc-mingw-w64-aarch64
```

**iOS**:

- Requires macOS with Xcode installed
- Run: `xcode-select --install`

### Quick Build Scripts (macOS)

For convenience on macOS, use the dedicated build scripts:

```bash
# Native macOS builds
./build_macos.sh              # Current architecture
ARCH=universal ./build_macos.sh  # Universal binary (x86_64 + arm64)

# Cross-compile for other platforms
./build_windows.sh            # Windows x86_64 and x86
./build_windows_arm64.sh      # Windows ARM64 (⚠️ falls back to x86_64 on macOS)
./build_linux_arm64.sh        # Linux ARM64
./build_android.sh            # Android ARM64
./build_ios.sh                # iOS (requires Xcode)
ARCH=universal ./build_ios.sh # iOS XCFramework
```

## C API Reference

### Network Creation

```c
char* Loom_NewNetwork(int inputSize, int gridRows, int gridCols, int layersPerCell, bool useGPU);
```

Creates a new neural network and returns JSON with handle:

```json
{
  "handle": 1,
  "type": "Network",
  "input_size": 784,
  "grid_rows": 2,
  "grid_cols": 1,
  "layers_cell": 1,
  "total_layers": 2,
  "gpu": true,
  "gpu_init_ms": 45
}
```

### Layer Initialization

```c
char* Loom_InitDenseLayer(int inputSize, int outputSize, int activation);
```

Creates a dense layer configuration (returns JSON string):

**Activation types**:

- `0` = Linear
- `1` = ReLU
- `2` = Sigmoid
- `3` = Tanh

```c
char* Loom_SetLayer(int64_t handle, int row, int col, int layer, char* configJSON);
```

Sets layer configuration from JSON.

**Example**:

```c
char* config = Loom_InitDenseLayer(784, 392, 1); // 784→392, ReLU
Loom_SetLayer(handle, 0, 0, 0, config);
Loom_FreeCString(config);
```

### Method Calling

```c
char* Loom_Call(int64_t handle, char* method, char* argsJSON);
```

Dynamically calls any Network method with JSON arguments.

**Examples**:

```c
// Forward pass
char* output = Loom_Call(handle, "ForwardCPU", "[[0.1, 0.2, ...]]");

// Training
char* result = Loom_Call(handle, "Train", "[{\"epochs\": 10, \"lr\": 0.01}]");

// Get batch size
char* size = Loom_Call(handle, "GetBatchSize", "[]");
```

### Introspection

```c
char* Loom_ListMethods(int64_t handle);
```

Returns JSON array of all available methods:

```json
{
  "methods": [
    {
      "name": "ForwardCPU",
      "parameters": ["[][]float32"],
      "returns": ["[][]float32"]
    },
    {
      "name": "Train",
      "parameters": ["nn.TrainingConfig"],
      "returns": ["error"]
    }
  ],
  "count": 24
}
```

```c
char* Loom_GetInfo(int64_t handle);
```

Returns object metadata:

```json
{
  "type": "*nn.Network",
  "kind": "ptr",
  "methods": 24,
  "handle": 1,
  "gpu_enabled": true,
  "grid_rows": 2,
  "grid_cols": 1,
  "layers_per_cell": 1,
  "input_size": 784,
  "batch_size": 32,
  "total_layers": 2
}
```

### Model Persistence

```c
char* Loom_SaveModel(int64_t handle, char* modelID);
```

Serializes network to JSON string.

```c
char* Loom_LoadModel(char* jsonString, char* modelID);
```

Deserializes network from JSON (returns handle info).

### Memory Management

```c
void Loom_Free(int64_t handle);
```

Releases GPU resources and deletes object from handle map.

```c
void Loom_FreeCString(char* p);
```

Frees C strings allocated by LOOM (all return values).

### Version Info

```c
char* Loom_GetVersion();
```

Returns version string.

## Usage Example

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Function declarations
extern char* Loom_NewNetwork(int, int, int, int, bool);
extern char* Loom_InitDenseLayer(int, int, int);
extern char* Loom_SetLayer(int64_t, int, int, int, char*);
extern char* Loom_Call(int64_t, char*, char*);
extern void Loom_Free(int64_t);
extern void Loom_FreeCString(char*);

int main() {
    // Create network
    char* result = Loom_NewNetwork(784, 2, 1, 1, false);
    int64_t handle = extractHandle(result); // Parse JSON
    Loom_FreeCString(result);

    // Initialize layers
    char* layer0 = Loom_InitDenseLayer(784, 392, 1);
    Loom_SetLayer(handle, 0, 0, 0, layer0);
    Loom_FreeCString(layer0);

    char* layer1 = Loom_InitDenseLayer(392, 10, 0);
    Loom_SetLayer(handle, 1, 0, 0, layer1);
    Loom_FreeCString(layer1);

    // Forward pass
    char* input = "[[0.1, 0.2, ...]]"; // 784 values
    char* output = Loom_Call(handle, "ForwardCPU", input);
    printf("Output: %s\n", output);
    Loom_FreeCString(output);

    // Cleanup
    Loom_Free(handle);
    return 0;
}
```

Compile:

```bash
gcc -o my_program my_program.c -L. -lloom -Wl,-rpath,.
```

## Benchmark Results

Run `./simple_bench` to compare CPU vs GPU performance:

```
=== LOOM C ABI Simple Benchmark ===
Version: LOOM C ABI v1.0

Network: 2x1x1 grid, input_size=784
Iterations: 100

--- CPU Test ---
CPU Network created in 2.34 ms (handle: 1)
Layers initialized
CPU Forward: 100 iterations in 45.67 ms (avg: 0.4567 ms/iter)

--- GPU Test ---
GPU Network created in 52.10 ms (handle: 2)
Layers initialized
GPU Forward: 100 iterations in 12.34 ms (avg: 0.1234 ms/iter)

=== Results ===
CPU Avg: 0.4567 ms/iter
GPU Avg: 0.1234 ms/iter
Speedup: 3.70x (GPU faster)
```

## Type Conversion

LOOM automatically converts between JSON and Go types:

| Go Type                    | JSON Type | Example            |
| -------------------------- | --------- | ------------------ |
| `int`, `int32`, `int64`    | Number    | `42`               |
| `float32`, `float64`       | Number    | `3.14`             |
| `bool`                     | Boolean   | `true`             |
| `string`                   | String    | `"hello"`          |
| `[]T`                      | Array     | `[1, 2, 3]`        |
| `map[string]T`             | Object    | `{"key": "value"}` |
| `struct`                   | Object    | `{"field": 123}`   |
| Custom types (`LayerType`) | Number    | `1`                |

## Language Bindings

### Python (ctypes)

```python
import ctypes
import json

loom = ctypes.CDLL('./libloom.so')
loom.Loom_NewNetwork.restype = ctypes.c_char_p
loom.Loom_Call.restype = ctypes.c_char_p
loom.Loom_FreeCString.argtypes = [ctypes.c_char_p]

# Create network
result = loom.Loom_NewNetwork(784, 2, 1, 1, False)
data = json.loads(result.decode('utf-8'))
handle = data['handle']
loom.Loom_FreeCString(result)

# Forward pass
input_json = json.dumps([[0.1] * 784])
output = loom.Loom_Call(handle, b"ForwardCPU", input_json.encode())
print(output.decode('utf-8'))
loom.Loom_FreeCString(output)

# Cleanup
loom.Loom_Free(handle)
```

### Rust (FFI)

```rust
use std::ffi::{CString, CStr};
use std::os::raw::c_char;

#[link(name = "loom")]
extern "C" {
    fn Loom_NewNetwork(input: i32, rows: i32, cols: i32, layers: i32, gpu: bool) -> *mut c_char;
    fn Loom_Call(handle: i64, method: *const c_char, args: *const c_char) -> *mut c_char;
    fn Loom_Free(handle: i64);
    fn Loom_FreeCString(p: *mut c_char);
}

fn main() {
    unsafe {
        let result = Loom_NewNetwork(784, 2, 1, 1, false);
        let result_str = CStr::from_ptr(result).to_str().unwrap();
        println!("{}", result_str);
        Loom_FreeCString(result);
    }
}
```

## Architecture

```
┌─────────────┐
│ C/C++/Rust  │
│   Program   │
└──────┬──────┘
       │ C ABI calls
       ▼
┌─────────────────────────┐
│   libloom.so/dylib      │
│  ┌───────────────────┐  │
│  │ Handle Manager    │  │
│  │  (sync.Mutex)     │  │
│  └─────────┬─────────┘  │
│            │             │
│  ┌─────────▼─────────┐  │
│  │ JSON Converter    │  │
│  │  (reflect)        │  │
│  └─────────┬─────────┘  │
│            │             │
│  ┌─────────▼─────────┐  │
│  │  nn.Network       │  │
│  │  Methods (24+)    │  │
│  └───────────────────┘  │
└─────────────────────────┘
```

## Error Handling

All functions return JSON. Errors are indicated by `{"error": "message"}`:

```c
char* result = Loom_Call(handle, "InvalidMethod", "[]");
// Returns: {"error": "Method not found: InvalidMethod"}
```

Always check for errors before parsing results.

## Thread Safety

- Handle storage is protected by `sync.Mutex`
- Multiple goroutines can safely access different Network objects
- Same Network object should not be used concurrently from multiple threads

## Performance Notes

- **GPU initialization overhead**: ~50ms first call
- **JSON parsing**: Minimal overhead for small payloads
- **Reflection overhead**: ~1-5µs per method call
- **Best for**: Batch operations, not per-sample inference

## License

Same as parent LOOM project.
