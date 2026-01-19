# Loom Codebase Guide

This is the central documentation index for Loom, a **Deterministic Neural Virtual Machine (DNVM)** — a portable execution environment for neural networks with bitwise-identical results across all platforms.

## Package Overview

| Package | Description | Documentation |
|---------|-------------|---------------|
| **nn** | Core neural network implementation | [nn/overview.md](./nn/overview.md) |
| **cabi** | C ABI for FFI (Python, C#, C++) | [cabi.md](#c-abi-package) |
| **wasm** | WebAssembly bindings for JavaScript | [wasm.md](#wasm-package) |
| **tokenizer** | BPE tokenizer for LLMs | [tokenizer.md](#tokenizer-package) |
| **pods** | Compute pods for modular operations | [pods.md](#pods-package) |
| **detector** | Hardware capability detection | [detector.md](#detector-package) |

---

## Directory Structure

```
loom/
├── nn/                    # Core neural network package
│   ├── types.go           # Network, LayerConfig, Tensor
│   ├── forward.go         # Forward propagation
│   ├── backward.go        # Backward propagation
│   ├── training.go        # Training loops
│   ├── tween.go           # Neural Tweening
│   ├── serialization.go   # Model save/load
│   └── ...                # 53 Go files total
│
├── cabi/                  # C ABI for cross-language FFI
│   ├── main.go            # C-exported functions
│   ├── build_*.sh         # Build scripts per platform
│   ├── universal_test.c   # C test suite
│   └── compiled/          # Pre-built binaries
│
├── wasm/                  # WebAssembly module
│   ├── main.go            # JS exports via syscall/js
│   ├── main.wasm          # Compiled WASM binary
│   └── *.html             # Demo applications
│
├── tokenizer/             # BPE tokenizer
│   └── bpe.go             # HuggingFace-compatible BPE
│
├── pods/                  # Modular compute pods
│   ├── core.go            # Pod interface
│   ├── ml_gemm.go         # Matrix multiplication pod
│   └── ...                # Domain-specific pods
│
├── detector/              # Hardware detection
│   └── detector.go        # GPU/CPU capability detection
│
├── tva/                   # Tests, Validation, Analysis
│   ├── test_0_0_7.go      # Comprehensive test suite
│   ├── examples/          # Example scripts
│   └── gpu/               # GPU verification
│
├── python/                # Python bindings (welvet)
├── csharp/                # C# bindings (Welvet)
├── typescript/            # TypeScript bindings
│
└── docs/                  # Documentation
    ├── nn/                # NN package docs
    └── *.md               # Research papers, assessments
```

---

## C ABI Package

**Location:** `/cabi/`

The C ABI provides Foreign Function Interface support for calling Loom from any language that can load shared libraries.

### Exported Functions

| Function | Description |
|----------|-------------|
| `CreateLoomNetwork` | Create network from JSON config |
| `LoomForward` | Forward pass |
| `LoomBackward` | Backward pass |
| `LoomTrain` | Training with config |
| `LoomSaveModel` | Serialize to JSON string |
| `LoomLoadModel` | Load from JSON string |

### Step-Based Functions

| Function | Description |
|----------|-------------|
| `LoomInitStepState` | Create step state |
| `LoomSetInput` | Set input tensor |
| `LoomStepForward` | Execute forward step |
| `LoomStepBackward` | Execute backward step |
| `LoomApplyGradients` | Update weights |

### Tween Functions

| Function | Description |
|----------|-------------|
| `LoomCreateTweenState` | Create tween state |
| `LoomTweenStep` | Bidirectional training step |
| `LoomFreeTweenState` | Cleanup tween state |

### Optimizer Functions

| Function | Description |
|----------|-------------|
| `LoomApplyGradientsAdamW` | AdamW optimizer |
| `LoomApplyGradientsRMSprop` | RMSprop optimizer |
| `LoomApplyGradientsSGDMomentum` | SGD with momentum |

### Build Scripts

```bash
# Build for current platform
./build.sh

# Platform-specific builds
./build_linux.sh
./build_macos.sh
./build_windows.sh
./build_linux_arm64.sh
./build_windows_arm64.sh
./build_android.sh
./build_ios.sh

# Package all platforms
./package_release.sh
```

### Usage Example (C)

```c
#include <stdio.h>

// Declare imports
extern char* CreateLoomNetwork(char* jsonConfig);
extern char* LoomForward(float* inputs, int length);
extern void FreeLoomString(char* str);

int main() {
    // Create network
    char* result = CreateLoomNetwork("{...}");
    printf("Network: %s\n", result);
    FreeLoomString(result);
    
    // Forward pass
    float input[1024] = {1.0, 2.0, ...};
    char* output = LoomForward(input, 1024);
    printf("Output: %s\n", output);
    FreeLoomString(output);
    
    return 0;
}
```

---

## WASM Package

**Location:** `/wasm/`

WebAssembly module for running Loom in browsers and Node.js.

### Exported Functions

| Function | JavaScript Usage |
|----------|-----------------|
| `CreateNetwork` | `loom.CreateNetwork(config)` |
| `LoadNetworkFromString` | `loom.LoadNetworkFromString(json, id)` |
| `CreateTweenState` | `loom.CreateTweenState(network)` |
| `CreateAdaptationTracker` | `loom.CreateAdaptationTracker(window, total)` |
| `GraftNetworks` | `loom.GraftNetworks(net1, net2)` |

### Network Methods

When you create a network, it exposes all `Network` methods:

```javascript
const net = loom.CreateNetwork(config);

// Training
net.ForwardCPU(input);
net.BackwardCPU(gradOutput);
net.ApplyGradients(learningRate);

// Serialization
const json = net.SaveModelToString("model_id");

// Introspection
const methods = net.GetMethods();
const info = net.GetNetworkInfo();
```

### StepState Wrapper

```javascript
const state = net.InitStepState(inputSize);
state.SetInput(inputData);
const output = state.StepForward();
state.StepBackward(gradOutput);
```

### TweenState Wrapper

```javascript
const ts = loom.CreateTweenState(net);
const loss = ts.TweenStep(input, targetClass, outputSize, learningRate);
console.log("Loss:", loss);
```

### Building WASM

```bash
cd wasm
./build_wasm.sh
# Produces main.wasm (~9 MB)

# Serve demos
./serve.sh
# Open http://localhost:8080
```

### Demo Pages

| Page | Description |
|------|-------------|
| `index.html` | Basic network demo |
| `0_0_7.html` | v0.0.7 feature showcase |
| `adaptation_demo.html` | Task adaptation demo |
| `optimizer.html` | Optimizer comparison |
| `step_example.html` | Step-based execution |

---

## Tokenizer Package

**Location:** `/tokenizer/`

BPE tokenizer compatible with HuggingFace tokenizer.json format.

### Structures

```go
type Tokenizer struct {
    Vocab         map[string]int
    ReverseVocab  map[int]string
    Merges        []MergePair
    SpecialTokens map[string]int
    AddedTokens   map[string]int
    PreTokenizer  *PreTokenizer
    BOSToken      string
    EOSToken      string
    PADToken      string
}

type MergePair struct {
    First  string
    Second string
    Rank   int
}
```

### Loading

```go
// From file
tok, err := tokenizer.LoadFromFile("tokenizer.json")

// From bytes
tok, err := tokenizer.LoadFromBytes(data)
```

### Encoding

```go
// Basic encoding
ids := tok.Encode("Hello, world!", true)  // Add special tokens

// With offsets
ids, offsets := tok.EncodeWithOffsets("Hello, world!")
```

### Decoding

```go
text := tok.Decode(ids, true)  // Skip special tokens
```

### Helper Methods

```go
size := tok.VocabSize()
id, ok := tok.TokenToID("hello")
token, ok := tok.IDToToken(123)
```

---

## Pods Package

**Location:** `/pods/`

Modular compute pods for domain-specific operations.

### Pod Interface

```go
type Pod interface {
    Name() string
    Run(ctx *ExecContext, in any) (out any, err error)
}
```

### Execution Context

```go
type ExecContext struct {
    Ctx      context.Context
    UseGPU   bool
    Report   *detector.Report
    GPU      GPUHooks
    TempPool *Pool
    Now      time.Time
}
```

### Available Pods

| Pod | File | Description |
|-----|------|-------------|
| ML GEMM | `ml_gemm.go` | Matrix multiplication |
| Softmax/Norm | `ml_softmax_norm.go` | Normalization ops |
| A* Navigation | `nav_astar.go` | Pathfinding |
| Reduce | `primitives_reduce.go` | Sum, max, min |
| Scan | `primitives_scan.go` | Prefix sum |
| Culling | `render_culling.go` | Frustum culling |
| Audio STFT | `audio_stft.go` | Spectrogram |
| Video YUV | `video_yuv.go` | Color conversion |
| Varint | `compress_varint.go` | Compression |

### Usage

```go
ctx := pods.NewContext(nil)
ctx.UseGPU = true

gemm := NewGEMMPod()
result, err := gemm.Run(ctx, GEMMInput{A: matA, B: matB})
```

---

## Detector Package

**Location:** `/detector/`

Hardware capability detection for runtime optimization.

### Report Structure

```go
type Report struct {
    HasGPU        bool
    GPUVendor     string
    GPUDevice     string
    MaxWorkgroup  int
    MaxBufferSize int64
    Features      []string
}
```

### Usage

```go
report := detector.Detect()
if report.HasGPU {
    fmt.Printf("GPU: %s %s\n", report.GPUVendor, report.GPUDevice)
}
```

---

## Cross-Language Support

### Python (welvet)

```python
import welvet

# Load model
net = welvet.load_model_from_string(json_data, "model_id")

# Forward
output = net.forward_cpu(input_data)

# Training
ts = welvet.create_tween_state(net)
loss = ts.tween_step(input, target, size, lr)
```

### C# (Welvet)

```csharp
using Welvet;

// Load model
var net = NativeMethods.LoomLoadModel(jsonData, "model_id");

// Forward
var output = NativeMethods.LoomForward(input, input.Length);

// Training
var tsHandle = NativeMethods.LoomCreateTweenState(1);
var loss = NativeMethods.LoomTweenStep(tsHandle, input, len, target, size, lr);
```

### TypeScript

```typescript
import { Loom } from './loom';

const loom = await Loom.init();
const net = loom.CreateNetwork(config);

const output = net.ForwardCPU(input);
const loss = ts.TweenStep(input, target, size, lr);
```

---

## Testing & Examples

### Test Suite

```bash
# Run comprehensive tests
cd tva
go run test_0_0_7.go
```

### Examples

```bash
# Run specific example
cd tva/examples
go run mnist_demo.go
go run transformer_demo.go
go run tween_benchmark.go
```

### GPU Verification

```bash
cd tva/gpu
go run main.go -layer Dense -depth shallow
go run verification/main.go -layer Dense -size small
```

---

## Documentation Index

### NN Package

- [Overview](./nn/overview.md) - Architecture and file structure
- [Layers](./nn/layers.md) - All 16 layer types
- [Training](./nn/training.md) - Forward/backward, training loops
- [Optimizers](./nn/optimizers.md) - SGD, AdamW, RMSprop + schedulers
- [Serialization](./nn/serialization.md) - Model persistence
- [Tween](./nn/tween.md) - Neural Tweening algorithm
- [Introspection](./nn/introspection.md) - Telemetry and observers
- [Quick Reference](./nn/quick_reference.md) - Cheat sheet

### Research Papers

- [Polyglot Runtime](./research_paper_1_polyglot_runtime.md)
- [StepTween](./research_paper_2_steptween.md)
- [Heterogeneous MoE](./research_paper_3_heterogeneous_moe.md)
- [Integer Training](./research_paper_4_integer_training.md)
- [ARC Stitching](./research_paper_5_arc_stitching.md)

### Assessments

- [Loom vs Other Frameworks](./loom_assessment_comparison.md)
- [Neural Tween Analysis](./neural_tween_analysis.md)
- [Step Tween Assessment](./step_tween_assessment.md)
