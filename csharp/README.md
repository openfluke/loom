# Welvet - LOOM C# / .NET Bindings

**Wrapper for Embedding Loom Via External (C-ABI) Toolchain**

High-performance neural network library with **transformer inference** for .NET via C-ABI bindings. Zero runtime dependencies—just add the NuGet package and go.

[![NuGet](https://img.shields.io/nuget/v/Welvet.svg)](https://www.nuget.org/packages/Welvet/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../LICENSE)

## Framework Comparison

| Feature Category | Feature | **Loom/Welvet** | **ML.NET** | **TensorFlow.NET** | **ONNX Runtime** | **Accord.NET** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Core** | **Runtime Dependency** | **None** (Native) | .NET Native | TF C++ | ONNX C++ | .NET Native |
| | **Auto-Differentiation** | ⚠️ Hybrid/Manual | ⚠️ Limited | ✅ Full | ❌ | ❌ |
| **Loading** | **Safetensors** | ✅ **Native** | ❌ | ❌ | ❌ | ❌ |
| | **Structure Inference** | ✅ **Auto-Detect** | ❌ | ❌ | ❌ | ❌ |
| **Training** | **Full Training** | ✅ **Complete** | ⚠️ Limited | ✅ | ❌ Inference Only | ✅ |
| | **Neural Tweening** | ✅ **Hybrid Engine** | ❌ | ❌ | ❌ | ❌ |
| | **LR Schedulers** | ✅ **7 Types** | ⚠️ | ✅ | ❌ | ⚠️ |
| **Layer Support** | **Dense (MLP)** | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **Conv1D/2D** | ✅ **Native** | ✅ | ✅ | ✅ | ✅ |
| | **RNN / LSTM** | ✅ **Full Gate** | ⚠️ | ✅ | ✅ | ⚠️ |
| | **Transformer (MHA)** | ✅ (Explicit) | ❌ | ✅ | ✅ | ❌ |
| | **Parallel / MoE** | ✅ **Structure** | ❌ | ❌ (Manual) | ❌ | ❌ |
| | **Stitch Layers** | ✅ **Native** | ❌ | ❌ | ❌ | ❌ |
| **Advanced** | **Step-Based Forward** | ✅ **Unique** | ❌ | ❌ | ❌ | ❌ |
| | **K-Means / Stats** | ✅ **Parallel** | ✅ | ❌ | ❌ | ✅ |
| | **Cross-Lang ABI** | ✅ **Universal** | ❌ | ❌ | ⚠️ | ❌ |
| **Platform** | **Streaming LLM** | ✅ | ❌ | ✅ | ✅ | ❌ |

For detailed analysis, see [`docs/loom_assessment_comparison.md`](../docs/loom_assessment_comparison.md).

## 🌍 Cross-Ecosystem Compatibility

Models trained in C# can be loaded instantly in Python, Go, or TypeScript. **Bit-for-bit identical results** across all platforms.

| Platform | Package | Install |
|:---------|:--------|:--------|
| **C#/.NET** | [NuGet](https://www.nuget.org/packages/Welvet) | `dotnet add package Welvet` |
| **Python** | [PyPI](https://pypi.org/project/welvet/) | `pip install welvet` |
| **TypeScript/Node** | [NPM](https://www.npmjs.com/package/@openfluke/welvet) | `npm install @openfluke/welvet` |
| **Go** | [GitHub](https://github.com/openfluke/loom) | `go get github.com/openfluke/loom` |

### Supported Platforms

Welvet includes pre-compiled binaries for:
- **Linux**: x86_64, ARM64
- **Windows**: x86_64, ARM64
- **macOS**: Apple Silicon (M1/M2/M3), Intel, Universal
- **Android**: ARM64 (via MAUI)
- **iOS**: ARM64; simulators (via MAUI with XCFramework)

## Installation

```bash
dotnet add package Welvet
```

Or via NuGet Package Manager:
```
Install-Package Welvet
```

## Quick Start

### 🎉 Simple API (Recommended)

The simple API provides **cross-platform consistency** with Python, TypeScript, and Go:

```csharp
using System.Text.Json;
using Welvet;

// Create network from JSON configuration
var config = @"{
    ""batch_size"": 1,
    ""grid_rows"": 1,
    ""grid_cols"": 1,
    ""layers_per_cell"": 2,
    ""layers"": [
        {""type"": ""dense"", ""input_height"": 4, ""output_height"": 8, ""activation"": ""relu""},
        {""type"": ""dense"", ""input_height"": 8, ""output_height"": 2, ""activation"": ""sigmoid""}
    ]
}";

var result = Network.CreateFromJson(config);
Console.WriteLine("✅ Network created!");

// Training data
var batches = new[] {
    new { Input = new[] { 0f, 0f, 1f, 1f }, Target = new[] { 1f, 0f } },
    new { Input = new[] { 1f, 1f, 0f, 0f }, Target = new[] { 0f, 1f } }
};

var trainConfig = new { Epochs = 100, LearningRate = 0.1f, LossType = "mse" };
var trainResult = NativeMethods.LoomTrain(
    JsonSerializer.Serialize(batches),
    JsonSerializer.Serialize(trainConfig)
);
Console.WriteLine("✅ Training complete!");

// Forward pass
float[] input = { 0f, 0f, 1f, 1f };
var outputPtr = NativeMethods.LoomForward(input, input.Length);
var output = JsonSerializer.Deserialize<float[]>(NativeMethods.PtrToStringAndFree(outputPtr));
Console.WriteLine($"Output: [{output[0]:F3}, {output[1]:F3}]");  // [0.95, 0.05]

// Save/Load - works across ALL platforms!
var modelPtr = NativeMethods.LoomSaveModel("my_model");
var modelJson = NativeMethods.PtrToStringAndFree(modelPtr);
Console.WriteLine($"✓ Model saved ({modelJson.Length} bytes)");

// Load in Python, TypeScript, or Go with identical results
NativeMethods.LoomLoadModel(modelJson, "my_model");
```

**Simple API Functions:**

| Function | Description |
|:---------|:------------|
| `CreateLoomNetwork(config)` | Create network from JSON |
| `LoomForward(inputs, length)` | Forward pass |
| `LoomBackward(gradients, length)` | Backward pass |
| `LoomUpdateWeights(learningRate)` | Update weights |
| `LoomTrain(batchesJSON, configJSON)` | Train network |
| `LoomSaveModel(modelID)` | Save to JSON string |
| `LoomLoadModel(jsonString, modelID)` | Load from JSON |
| `LoomGetNetworkInfo()` | Get network info |
| `LoomEvaluateNetwork(inputsJSON, expectedJSON)` | Evaluate with metrics |

### ⚡ Stepping API - Fine-Grained Execution Control

Execute networks one step at a time for online learning and stateful processing:

```csharp
using Welvet;

// Create network
Network.CreateFromJson(config);

// Initialize stepping state
using var stepState = new StepState(inputSize: 4);

// Training loop - update weights after EACH step
for (int step = 0; step < 100000; step++)
{
    stepState.SetInput(new float[] { 0.1f, 0.2f, 0.1f, 0.3f });
    stepState.StepForward();
    var output = stepState.GetOutput();
    
    // Calculate gradients
    var gradients = output.Select((o, i) => o - target[i]).ToArray();
    
    // Backward pass and update
    stepState.StepBackward(gradients);
    NativeMethods.LoomApplyGradients(0.01f);
}
```

### 🧬 Neural Tweening API - Real-Time Adaptation

Neural tweening enables networks to adapt to changing goals in real-time without full backpropagation:

```csharp
using Welvet;

Network.CreateFromJson(config);

// Create TweenState with chain rule (StepTweenChain mode)
using var tween = new TweenState(useChainRule: true);

// Continuously adapt to targets
foreach (var (observation, targetClass) in trainingStream)
{
    float gap = tween.Step(observation, targetClass, outputSize: 4, learningRate: 0.02f);
    Console.WriteLine($"Adaptation gap: {gap:F4}");
}
```

### 📊 AdaptationTracker - Benchmark Task Switching

Track accuracy across task changes for benchmarking real-time adaptation:

```csharp
using Welvet;

using var tracker = new AdaptationTracker(windowDurationMs: 1000, totalDurationMs: 10000);
tracker.SetModelInfo("Dense-5L", "StepTweenChain");

// Schedule task changes
tracker.ScheduleTaskChange(3333, taskId: 1, taskName: "AVOID");
tracker.ScheduleTaskChange(6666, taskId: 0, taskName: "CHASE");
tracker.Start("CHASE", taskId: 0);

// Run test loop
while (timeElapsed < 10000)
{
    int currentTask = tracker.GetCurrentTask();
    // ... run network ...
    tracker.RecordOutput(isCorrect: correct);
}

var results = tracker.GetResults();
Console.WriteLine($"Avg accuracy: {results.RootElement.GetProperty("avg_accuracy").GetDouble():F1}%");
```

### 🔗 Network Grafting - Architecture Fusion

Combine multiple trained networks into a single parallel super-network:

```csharp
using Welvet;

long h1 = NativeMethods.LoomCreateNetworkHandle(config);
long h2 = NativeMethods.LoomCreateNetworkHandle(config);

var handlesJson = JsonSerializer.Serialize(new[] { h1, h2 });
var resultPtr = NativeMethods.LoomGraftNetworks(handlesJson, "concat");
var result = NativeMethods.PtrToStringAndFree(resultPtr);
// Result: {"success": true, "num_branches": 2, "combine_mode": "concat"}
```

### 🧠 Statistical Tools

Built-in K-Means clustering and correlation analysis:

```csharp
using Welvet;

// K-Means Clustering
var data = new[] { new[] { 1f, 1f }, new[] { 1.1f, 1.1f }, new[] { 5f, 5f }, new[] { 5.1f, 5.1f } };
var kmeansPtr = NativeMethods.LoomKMeansCluster(JsonSerializer.Serialize(data), k: 2, maxIter: 100);
var kmeans = JsonSerializer.Deserialize<JsonElement>(NativeMethods.PtrToStringAndFree(kmeansPtr));
Console.WriteLine($"Centroids: {kmeans.GetProperty("centroids")}");

// Correlation Matrix
var matrix = new[] { new[] { 1f, 2f, 3f }, new[] { 4f, 5f, 6f } };
var corrPtr = NativeMethods.LoomComputeCorrelation(JsonSerializer.Serialize(matrix));
```

### 🚀 Transformer Inference (LLMs)

Run LLaMA, SmolLM, GPT-2, and other transformers with **streaming support**:

```csharp
using Welvet;

// Load model
Transformer.LoadTokenizer("models/SmolLM2-135M-Instruct/tokenizer.json");
Transformer.LoadModelFromDirectory("models/SmolLM2-135M-Instruct");

// Stream generation - tokens appear in real-time!
foreach (var token in Transformer.GenerateStream("The capital of France is", maxTokens: 50))
{
    Console.Write(token);
}

// Or generate all at once
var text = Transformer.Generate("Once upon a time", maxTokens: 50, temperature: 0.7f);
```

## Complete Test Suite

The `UniversalTest.cs` example demonstrates all framework capabilities:

```bash
cd examples
dotnet run --project UniversalTest.csproj
```

**Test Coverage (80/80 tests passing):**
- ✅ 12 Layer Types × 6 Data Types (72 tests)
- ✅ Network Grafting
- ✅ K-Means Clustering & Correlation Analysis
- ✅ Optimizers (SGD, AdamW, RMSprop)
- ✅ Ensemble Features
- ✅ Observer Pattern (Adaptation Tracking)
- ✅ Introspection API
- ✅ Step & Tween API
- ✅ Advanced Layers (Embedding, Residual)

## Layer Types

| Layer | Type String | Description |
|:------|:------------|:------------|
| Dense | `dense` | Fully connected layer |
| LSTM | `lstm` | Long Short-Term Memory |
| RNN | `rnn` | Recurrent Neural Network |
| Conv2D | `conv2d` | 2D Convolution |
| Conv1D | `conv1d` | 1D Convolution |
| Multi-Head Attention | `multi_head_attention` | Transformer attention |
| LayerNorm | `layer_norm` | Layer normalization |
| RMSNorm | `rms_norm` | RMS normalization |
| SwiGLU | `swiglu` | SwiGLU activation layer |
| Softmax | `softmax` | Softmax classification |
| Embedding | `embedding` | Token embedding |
| Parallel | `parallel` | Branching with combine modes |
| Sequential | `sequential` | Grouped sub-layers |

**Parallel Combine Modes:** `add`, `concat`, `multiply`, `average`, `grid_scatter`, `filter`

**Activation Functions:** `relu`, `sigmoid`, `tanh`, `softmax`, `gelu`, `swish`, `mish`, `leaky_relu`, `elu`, `selu`, `linear`

## 🌐 Cross-Platform Model Sharing

The same JSON model works across **all platforms** - train once, deploy anywhere:

```csharp
// C# - Save model
var modelJson = NativeMethods.PtrToStringAndFree(NativeMethods.LoomSaveModel("model"));
File.WriteAllText("model.json", modelJson);
```

```python
# Python - Load same model
import welvet
with open("model.json") as f:
    modelJson = f.read()
welvet.load_model_simple(modelJson, "model")
output = welvet.forward_simple([0.1, 0.2, 0.3, 0.4])  # Identical results!
```

```typescript
// TypeScript - Load same model
import { loadLoomNetwork } from "@openfluke/welvet";
const modelJson = fs.readFileSync("model.json", "utf-8");
const network = loadLoomNetwork(modelJson, "model");
const output = network.ForwardCPU(JSON.stringify([[0.1, 0.2, 0.3, 0.4]]));
// Bit-for-bit identical!
```

```go
// Go - Load same model
import "github.com/openfluke/loom/nn"
modelJson, _ := os.ReadFile("model.json")
network, _ := nn.LoadModelFromString(string(modelJson), "model")
output, _ := network.ForwardCPU([]float32{0.1, 0.2, 0.3, 0.4})
```

## Building from Source

```bash
git clone https://github.com/openfluke/loom.git
cd loom/csharp

# Build
dotnet build -c Release

# Run examples
cd examples
dotnet run --project UniversalTest.csproj
```

## Documentation

- [Main LOOM README](../README.md) - Framework overview
- [Python Bindings](../python/README.md) - Python API reference
- [TypeScript Bindings](../typescript/README.md) - TypeScript/WASM API
- [Assessment Comparison](../docs/loom_assessment_comparison.md) - Detailed framework comparison

## License

Apache License 2.0 - see [LICENSE](../LICENSE) for details.

## Links

- **GitHub**: [github.com/openfluke/loom](https://github.com/openfluke/loom)
- **NuGet**: [Welvet](https://www.nuget.org/packages/Welvet)
- **PyPI**: [welvet](https://pypi.org/project/welvet/)
- **NPM**: [@openfluke/welvet](https://www.npmjs.com/package/@openfluke/welvet)
