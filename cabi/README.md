# LOOM C ABI

C Foreign Function Interface (FFI) for LOOM. Use LOOM transformers and neural networks from **any language** that supports C FFI: Python, C#, Rust, C++, Node.js, etc.

## 🚀 Transformer Inference

Run LLMs (LLaMA, SmolLM, GPT-2, etc.) from any language via C ABI with **streaming support**.

### Quick Start

**1. Build the library:**

```bash
./build.sh
# Creates libloom.so (Linux) / libloom.dylib (macOS) / libloom.dll (Windows)
```

**2. Run the demo:**

```bash
# Download a model first (e.g., SmolLM2-135M-Instruct from HuggingFace)
# Then start the web interface:
python3 web_interface.py ../models/SmolLM2-135M-Instruct 8080

# Open http://localhost:8080/inference.html in your browser
```

### Python Example (Streaming)

```python
import ctypes
import json

# Load library
loom = ctypes.CDLL('./libloom.so')

# Configure function signatures
loom.LoadTokenizerFromBytes.argtypes = [ctypes.c_char_p, ctypes.c_int]
loom.LoadTokenizerFromBytes.restype = ctypes.c_void_p

loom.LoadTransformerFromBytes.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
loom.LoadTransformerFromBytes.restype = ctypes.c_void_p

loom.GenerateNextToken.argtypes = [ctypes.c_char_p, ctypes.c_float]
loom.GenerateNextToken.restype = ctypes.c_void_p

loom.EncodeText.argtypes = [ctypes.c_char_p, ctypes.c_bool]
loom.EncodeText.restype = ctypes.c_void_p

loom.DecodeTokens.argtypes = [ctypes.c_char_p, ctypes.c_bool]
loom.DecodeTokens.restype = ctypes.c_void_p

loom.Loom_FreeCString.argtypes = [ctypes.c_void_p]
loom.Loom_FreeCString.restype = None

# Load tokenizer
with open('models/SmolLM2-135M-Instruct/tokenizer.json', 'rb') as f:
    tok_data = f.read()
result_ptr = loom.LoadTokenizerFromBytes(tok_data, len(tok_data))
result_json = ctypes.string_at(result_ptr).decode('utf-8')
loom.Loom_FreeCString(result_ptr)
result = json.loads(result_json)
print(f"✓ Tokenizer loaded (vocab: {result['vocab_size']})")

# Load transformer
with open('models/SmolLM2-135M-Instruct/config.json', 'rb') as f:
    config = f.read()
with open('models/SmolLM2-135M-Instruct/model.safetensors', 'rb') as f:
    weights = f.read()
result_ptr = loom.LoadTransformerFromBytes(config, len(config), weights, len(weights))
result_json = ctypes.string_at(result_ptr).decode('utf-8')
loom.Loom_FreeCString(result_ptr)
result = json.loads(result_json)
print(f"✓ Model loaded ({result['num_layers']} layers, hidden={result['hidden_size']})")

# Encode prompt
prompt = "Once upon a time"
encode_ptr = loom.EncodeText(prompt.encode('utf-8'), True)
encode_json = ctypes.string_at(encode_ptr).decode('utf-8')
loom.Loom_FreeCString(encode_ptr)
tokens = json.loads(encode_json)['ids']

# Generate tokens one at a time (streaming)
for i in range(50):
    gen_ptr = loom.GenerateNextToken(json.dumps(tokens).encode('utf-8'), 0.7)
    gen_json = ctypes.string_at(gen_ptr).decode('utf-8')
    loom.Loom_FreeCString(gen_ptr)
    gen_result = json.loads(gen_json)

    next_token = gen_result['token']
    tokens.append(next_token)

    # Decode and print token
    decode_ptr = loom.DecodeTokens(json.dumps([next_token]).encode('utf-8'), True)
    decode_json = ctypes.string_at(decode_ptr).decode('utf-8')
    loom.Loom_FreeCString(decode_ptr)
    token_text = json.loads(decode_json)['text']
    print(token_text, end='', flush=True)

    if gen_result.get('is_eos'):
        break

print()  # Newline at end
```

## Transformer API Reference

### Loading

```c
// Load tokenizer from bytes
char* LoadTokenizerFromBytes(char* dataPtr, int dataLen);
// Returns: {"success": true, "vocab_size": 49152, ...}

// Load transformer model
char* LoadTransformerFromBytes(char* configPtr, int configLen,
                               char* weightsPtr, int weightsLen);
// Returns: {"success": true, "num_layers": 30, "hidden_size": 576, ...}
```

### Text Processing

```c
// Encode text to token IDs
char* EncodeText(char* textPtr, bool addSpecialTokens);
// Returns: {"success": true, "ids": [123, 456, ...]}

// Decode token IDs to text
char* DecodeTokens(char* idsJSON, bool skipSpecialTokens);
// Returns: {"success": true, "text": "decoded text"}
```

### Generation

```c
// Generate single next token (for streaming)
char* GenerateNextToken(char* idsJSON, float temperature);
// Returns: {"success": true, "token": 789, "is_eos": false}

// Generate full text at once
char* GenerateText(char* promptPtr, int maxTokens, float temperature);
// Returns: {"success": true, "generated_text": "...", "num_tokens": 50}
```

### Memory Management

```c
// Free C strings returned by LOOM functions
void Loom_FreeCString(char* ptr);
```

**⚠️ Important:** All functions return JSON strings allocated with `malloc()`. You **must** call `Loom_FreeCString()` on every returned pointer to avoid memory leaks.

**⚠️ Use `c_void_p` in Python:** When using ctypes, declare return types as `ctypes.c_void_p` (not `c_char_p`) to avoid Python's automatic string conversion which corrupts the pointer.

## Language Examples

### C#

```csharp
[DllImport("libloom.so")]
private static extern IntPtr LoadTokenizerFromBytes(byte[] data, int len);

[DllImport("libloom.so")]
private static extern void Loom_FreeCString(IntPtr ptr);

// Usage
byte[] tokData = File.ReadAllBytes("tokenizer.json");
IntPtr resultPtr = LoadTokenizerFromBytes(tokData, tokData.Length);
string resultJson = Marshal.PtrToStringAnsi(resultPtr);
Loom_FreeCString(resultPtr);
```

### Rust

```rust
#[link(name = "loom")]
extern "C" {
    fn LoadTokenizerFromBytes(data: *const u8, len: i32) -> *mut c_char;
    fn Loom_FreeCString(ptr: *mut c_char);
}

unsafe {
    let tok_data = std::fs::read("tokenizer.json")?;
    let result_ptr = LoadTokenizerFromBytes(tok_data.as_ptr(), tok_data.len() as i32);
    let result_cstr = CStr::from_ptr(result_ptr);
    let result_json = result_cstr.to_str()?;
    Loom_FreeCString(result_ptr);
}
```

### Node.js (ffi-napi)

```javascript
const ffi = require("ffi-napi");
const loom = ffi.Library("./libloom.so", {
  LoadTokenizerFromBytes: ["string", ["pointer", "int"]],
  Loom_FreeCString: ["void", ["pointer"]],
});

const tokData = fs.readFileSync("tokenizer.json");
const resultPtr = loom.LoadTokenizerFromBytes(tokData, tokData.length);
const resultJson = JSON.parse(resultPtr);
loom.Loom_FreeCString(resultPtr);
```

## Building

**✅ All build scripts now include `transformer.go`** - All platforms will have transformer inference support.

### Current Platform

```bash
./build.sh
```

### Multi-Platform

```bash
./build_all.sh linux arm64          # Linux ARM64
./build_all.sh macos universal      # macOS Universal Binary
./build_all.sh windows x86_64       # Windows 64-bit
./build_all.sh android arm64        # Android ARM64
./build_all.sh ios xcframework      # iOS XCFramework

# Build all available platforms at once
./build_all.sh --clean all
```

### Verify Transformer Functions

After building, verify transformer functions are included:

```bash
# macOS
nm -gU compiled/macos_universal/libloom.dylib | grep LoadTokenizer

# Linux
nm -D compiled/linux_x86_64/libloom.so | grep LoadTokenizer

# Windows (on Windows)
dumpbin /exports compiled/windows_x86_64/libloom.dll | findstr LoadTokenizer
```

You should see: `LoadTokenizerFromBytes`, `LoadTransformerFromBytes`, `EncodeText`, `DecodeTokens`, `GenerateText`, `GenerateNextToken`

### Supported Platforms

| Platform | Architectures                | Output          | Notes                   |
| -------- | ---------------------------- | --------------- | ----------------------- |
| Linux    | x86_64, arm64, armv7, x86    | `libloom.so`    | Native or cross-compile |
| macOS    | x86_64, arm64, universal     | `libloom.dylib` | Universal = fat binary  |
| Windows  | x86_64, x86, arm64           | `libloom.dll`   | Requires mingw-w64      |
| Android  | arm64, armv7, x86_64, x86    | `libloom.so`    | Requires Android NDK    |
| iOS      | arm64, x86_64_sim, arm64_sim | `libloom.dylib` | Requires Xcode          |

Output goes to `compiled/<platform>_<arch>/`.

## Neural Network API (Legacy)

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

````c
## Neural Network API (Legacy)

The C ABI also exposes the original neural network training API. See the full docs in the old README or use `Loom_ListMethods()` to discover available functions.

### Basic Neural Network Example

```c
#include <stdio.h>
#include <stdlib.h>

extern char* Loom_NewNetwork(int, int, int, int, bool);
extern char* Loom_InitDenseLayer(int, int, int);
extern char* Loom_SetLayer(int64_t, int, int, int, char*);
extern char* Loom_Call(int64_t, char*, char*);
extern void Loom_Free(int64_t);
extern void Loom_FreeCString(char*);

int main() {
    // Create 2-layer network (784 → 392 → 10)
    char* result = Loom_NewNetwork(784, 2, 1, 1, false);
    int64_t handle = 1;  // Parse from JSON
    Loom_FreeCString(result);

    // Layer 1: 784 → 392, ReLU
    char* layer0 = Loom_InitDenseLayer(784, 392, 1);
    Loom_SetLayer(handle, 0, 0, 0, layer0);
    Loom_FreeCString(layer0);

    // Layer 2: 392 → 10, Linear
    char* layer1 = Loom_InitDenseLayer(392, 10, 0);
    Loom_SetLayer(handle, 1, 0, 0, layer1);
    Loom_FreeCString(layer1);

    // Forward pass
    char* output = Loom_Call(handle, "ForwardCPU", "[[0.1, ...]]");
    printf("Output: %s\n", output);
    Loom_FreeCString(output);

    Loom_Free(handle);
    return 0;
}
````

Compile:

```bash
gcc -o mnist mnist.c -L. -lloom -Wl,-rpath,.
```

## Files

- `transformer.go` - Transformer inference C exports
- `main.go` - Neural network C exports (legacy)
- `web_interface.py` - Python web server with streaming inference
- `inference.html` - Browser UI for text generation
- `build.sh` - Simple build script
- `build_all.sh` - Multi-platform build system
- `test_transformer.sh` - Setup verification

## Architecture

```
┌─────────────────────────┐
│  Your Application       │
│  (Python/C#/Rust/etc)   │
└───────────┬─────────────┘
            │ C FFI
            ▼
┌─────────────────────────┐
│   libloom.so/.dylib     │
│ ┌─────────────────────┐ │
│ │ Transformer Engine  │ │
│ │  • SmolLM2-135M     │ │
│ │  • Token-by-token   │ │
│ │  • BPE tokenizer    │ │
│ └─────────────────────┘ │
│ ┌─────────────────────┐ │
│ │ Neural Network API  │ │
│ │  • Training         │ │
│ │  • Inference        │ │
│ │  • GPU support      │ │
│ └─────────────────────┘ │
└─────────────────────────┘
```

## Performance Notes

- **Memory**: SmolLM2-135M uses ~500MB RAM
- **Speed**: ~10-50 tokens/sec on CPU (depends on model size)
- **Streaming**: Token-by-token generation for real-time UX
- **GPU**: Not yet implemented for transformers (CPU only)

## Troubleshooting

**Python: "munmap_chunk(): invalid pointer"**

- Use `ctypes.c_void_p` for return types, not `c_char_p`
- Python's automatic conversion corrupts the pointer

**Library not found:**

```bash
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH  # Linux
export DYLD_LIBRARY_PATH=.:$DYLD_LIBRARY_PATH  # macOS
```

**Cross-compilation fails:**

- Install required toolchains (see Building section)
- Check `$ANDROID_NDK_HOME` for Android builds

## License

MIT (same as parent LOOM project)

---

**Ready to use transformers from your favorite language? Start with `web_interface.py` as a reference implementation!** 🚀

````

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
````

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

## Transformer Inference API

### Loading Functions

```c
// Load tokenizer from bytes
char* LoadTokenizerFromBytes(char* dataPtr, int dataLen);

// Load transformer model from config and weights
char* LoadTransformerFromBytes(char* configPtr, int configLen,
                               char* weightsPtr, int weightsLen);
```

### Text Processing

```c
// Encode text to token IDs
char* EncodeText(char* textPtr, bool addSpecialTokens);

// Decode token IDs to text
char* DecodeTokens(char* idsJSON, bool skipSpecialTokens);
```

### Generation

```c
// Generate text from prompt
char* GenerateText(char* promptPtr, int maxTokens, float temperature);

// Generate single next token
char* GenerateNextToken(char* idsJSON, float temperature);
```

### Quick Start - Transformer Inference

**1. Serve model files:**

```bash
# Method 1: Go HTTP server
cd cmd/serve_model_bytes
./serve_model_bytes -model ../../models/SmolLM2-135M-Instruct -port 8080

# Method 2: Python web interface (uses C ABI directly)
./web_interface.py ../models/SmolLM2-135M-Instruct 8080
```

**2. Open web interface:**

```bash
# Open inference.html in your browser
open http://localhost:8080/inference.html
```

**Example Python usage:**

```python
#!/usr/bin/env python3
import ctypes
import json

# Load shared library
loom = ctypes.CDLL('./libloom.so')

# Configure function signatures
loom.LoadTokenizerFromBytes.argtypes = [ctypes.c_char_p, ctypes.c_int]
loom.LoadTokenizerFromBytes.restype = ctypes.c_char_p
loom.GenerateText.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_float]
loom.GenerateText.restype = ctypes.c_char_p

# Load tokenizer
with open('models/SmolLM2-135M-Instruct/tokenizer.json', 'rb') as f:
    tok_data = f.read()
result_ptr = loom.LoadTokenizerFromBytes(tok_data, len(tok_data))
result = json.loads(ctypes.string_at(result_ptr).decode('utf-8'))
print(f"Tokenizer: vocab_size={result['vocab_size']}")

# Load transformer
with open('models/SmolLM2-135M-Instruct/config.json', 'rb') as f:
    config = f.read()
with open('models/SmolLM2-135M-Instruct/model.safetensors', 'rb') as f:
    weights = f.read()
result_ptr = loom.LoadTransformerFromBytes(config, len(config), weights, len(weights))
result = json.loads(ctypes.string_at(result_ptr).decode('utf-8'))
print(f"Model: {result['num_layers']} layers, hidden_size={result['hidden_size']}")

# Generate text
result_ptr = loom.GenerateText(b"Once upon a time", 50, 0.7)
result = json.loads(ctypes.string_at(result_ptr).decode('utf-8'))
print(f"Generated: {result['generated_text']}")
```

## Neural Network API

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
