# Welvet

> .NET bindings for the LOOM neural network framework via C-ABI

**Welvet** (Wrapper for Embedding Loom Via External Toolchain) provides a complete neural network API for .NET applications. Built on the LOOM C-ABI, it delivers high-performance deep learning with GPU acceleration support.

[![NuGet](https://img.shields.io/nuget/v/Welvet.svg)](https://www.nuget.org/packages/Welvet/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../LICENSE)

## ✨ Features

- 🎉 **NEW: Simple API** - Streamlined functions matching Python/TypeScript/C - `CreateLoomNetwork`, `LoomForward`, `LoomTrain`, `LoomEvaluateNetwork`
- 🤖 **Transformer Inference** - Run LLMs like SmolLM2-135M with streaming generation
- 🚀 **Native Performance** - Direct P/Invoke to C library, zero overhead
- 🧠 **8 Layer Types (All CPU)** - Dense, Conv2D, Multi-Head Attention, LayerNorm, RNN, LSTM, Softmax (10 variants), Parallel (4 combine modes)
- ✅ **Full CPU Implementation** - Every layer works with complete forward/backward passes - tested and reliable!
- ✅ **Cross-Platform Consistency** - Same API and results as Python, TypeScript, C, WASM
- 💾 **One-Line Model Loading** - Load complete models with `LoomLoadModel()`
- 🎯 **Full Training Support** - Forward, backward, weight updates
- 🌐 **Cross-Platform** - Works on Linux, macOS, Windows (x64, ARM64)
- 📘 **Strongly Typed** - Full C# API with IntelliSense support
- 🎨 **Multiple Activations** - ReLU, Sigmoid, Tanh, Softplus, LeakyReLU
- ⚠️ **GPU Note** - GPU/WebGPU code exists but is untested; all demos use reliable CPU execution

## 📦 Installation

```bash
dotnet add package Welvet
```

Or via NuGet Package Manager:

```
Install-Package Welvet
```

## 🚀 Quick Start

### 🎉 NEW: Simple API (Recommended)

The simple API provides cross-platform consistency with Python, TypeScript, and C:

```csharp
using System;
using System.Text.Json;
using Welvet.Examples;

// Create network from JSON
var config = @"{
    ""batch_size"": 1,
    ""grid_rows"": 1,
    ""grid_cols"": 3,
    ""layers_per_cell"": 1,
    ""layers"": [
        {""type"": ""dense"", ""input_size"": 8, ""output_size"": 16, ""activation"": ""relu""},
        {
            ""type"": ""parallel"",
            ""combine_mode"": ""grid_scatter"",
            ""grid_output_rows"": 3,
            ""grid_output_cols"": 1,
            ""grid_output_layers"": 1,
            ""grid_positions"": [
                {""branch_index"": 0, ""target_row"": 0, ""target_col"": 0, ""target_layer"": 0},
                {""branch_index"": 1, ""target_row"": 1, ""target_col"": 0, ""target_layer"": 0},
                {""branch_index"": 2, ""target_row"": 2, ""target_col"": 0, ""target_layer"": 0}
            ],
            ""branches"": [
                {
                    ""type"": ""parallel"",
                    ""combine_mode"": ""add"",
                    ""branches"": [
                        {""type"": ""dense"", ""input_size"": 16, ""output_size"": 8, ""activation"": ""relu""},
                        {""type"": ""dense"", ""input_size"": 16, ""output_size"": 8, ""activation"": ""gelu""}
                    ]
                },
                {""type"": ""lstm"", ""input_size"": 16, ""hidden_size"": 8, ""seq_length"": 1},
                {""type"": ""rnn"", ""input_size"": 16, ""hidden_size"": 8, ""seq_length"": 1}
            ]
        },
        {""type"": ""dense"", ""input_size"": 24, ""output_size"": 2, ""activation"": ""sigmoid""}
    ]
}";

var resultPtr = GridScatterDemo.CreateLoomNetwork(config);
var result = GridScatterDemo.PtrToStringAndFree(resultPtr);
Console.WriteLine("✅ Network created!");

// Training
var batches = new[] {
    new { Input = new[] { 0.2f, 0.2f, 0.2f, 0.2f, 0.8f, 0.8f, 0.8f, 0.8f }, Target = new[] { 1.0f, 0.0f } },
    new { Input = new[] { 0.9f, 0.9f, 0.9f, 0.9f, 0.1f, 0.1f, 0.1f, 0.1f }, Target = new[] { 0.0f, 1.0f } },
    new { Input = new[] { 0.7f, 0.7f, 0.7f, 0.7f, 0.3f, 0.3f, 0.3f, 0.3f }, Target = new[] { 0.0f, 1.0f } },
    new { Input = new[] { 0.3f, 0.3f, 0.3f, 0.3f, 0.7f, 0.7f, 0.7f, 0.7f }, Target = new[] { 1.0f, 0.0f } }
};

string batchesJson = JsonSerializer.Serialize(batches);
string trainingConfig = @"{
    ""Epochs"": 800,
    ""LearningRate"": 0.15,
    ""UseGPU"": false,
    ""PrintEveryBatch"": 0,
    ""GradientClip"": 1.0,
    ""LossType"": ""mse"",
    ""Verbose"": false
}";

var trainResultPtr = GridScatterDemo.LoomTrain(batchesJson, trainingConfig);
Console.WriteLine("✅ Training complete!");

// Forward pass
float[] input = new[] { 0.2f, 0.2f, 0.2f, 0.2f, 0.8f, 0.8f, 0.8f, 0.8f };
var outputPtr = GridScatterDemo.LoomForward(input, input.Length);
var outputJson = GridScatterDemo.PtrToStringAndFree(outputPtr);
var output = JsonSerializer.Deserialize<float[]>(outputJson);
Console.WriteLine($"Output: [{output[0]:F3}, {output[1]:F3}]");  // [0.950, 0.050]

// Evaluate
var evalInputs = new[] {
    new[] { 0.2f, 0.2f, 0.2f, 0.2f, 0.8f, 0.8f, 0.8f, 0.8f },
    // ... more samples
};
var expected = new[] { 0.0, 1.0, 1.0, 0.0 };

var evalPtr = GridScatterDemo.LoomEvaluateNetwork(
    JsonSerializer.Serialize(evalInputs),
    JsonSerializer.Serialize(expected)
);
var metricsJson = GridScatterDemo.PtrToStringAndFree(evalPtr);
// Parse and display metrics: Quality Score: 100/100, Avg Deviation: 0.00%

// Save/Load
var modelJsonPtr = GridScatterDemo.LoomSaveModel("my_model");
var modelJson = GridScatterDemo.PtrToStringAndFree(modelJsonPtr);
Console.WriteLine($"✓ Model saved ({modelJson.Length} bytes)");

var loadPtr = GridScatterDemo.LoomLoadModel(modelJson, "my_model");
Console.WriteLine("✓ Model loaded - predictions match!");
```

**Simple API Functions:**

- `CreateLoomNetwork(jsonConfig)` - Create from JSON
- `LoomForward(inputs, length)` - Forward pass
- `LoomBackward(gradients, length)` - Backward pass
- `LoomUpdateWeights(learningRate)` - Update weights
- `LoomTrain(batchesJSON, configJSON)` - Train network
- `LoomSaveModel(modelID)` - Save to JSON string
- `LoomLoadModel(jsonString, modelID)` - Load from JSON
- `LoomGetNetworkInfo()` - Get network information
- `LoomEvaluateNetwork(inputsJSON, expectedJSON)` - Evaluate with metrics
- `FreeLoomString(ptr)` - Free C strings

**Cross-Platform Results:**

- ✅ Same training: 99.5% improvement, 100/100 quality score
- ✅ Same save/load: 0.00 prediction difference
- ✅ Same evaluation: Identical deviation metrics

See `examples/GridScatterDemo.cs` for complete working example.

## 📚 API Reference

### Transformer Class (NEW!)

#### Static Methods

```csharp
// Load tokenizer from file or bytes
TokenizerLoadResult LoadTokenizer(string tokenizerPath)
TokenizerLoadResult LoadTokenizerFromBytes(byte[] data)

// Load model from directory, files, or bytes
TransformerLoadResult LoadModelFromDirectory(string modelDir)
TransformerLoadResult LoadModel(string configPath, string weightsPath)
TransformerLoadResult LoadModelFromBytes(byte[] configData, byte[] weightsData)

// Encode text to token IDs
EncodeResult Encode(string text, bool addSpecialTokens = true)

// Decode token IDs to text
DecodeResult Decode(int[] tokenIds, bool skipSpecialTokens = true)

// Generate text (blocking, all tokens at once)
GenerateResult Generate(string prompt, int maxTokens = 50, float temperature = 0.7f)

// Generate text (streaming, token-by-token)
IEnumerable<string> GenerateStream(string prompt, int maxTokens = 50, float temperature = 0.7f)
```

#### Example: Streaming Generation

```csharp
using Welvet;

// Load model
Transformer.LoadTokenizer("models/SmolLM2-135M-Instruct/tokenizer.json");
var result = Transformer.LoadModelFromDirectory("models/SmolLM2-135M-Instruct");

if (!result.Success)
{
    Console.WriteLine($"Error: {result.Error}");
    return;
}

Console.WriteLine($"Model loaded: {result.NumLayers} layers, {result.HiddenSize} hidden size");

// Stream generation
foreach (var token in Transformer.GenerateStream("The capital of France is", maxTokens: 50))
{
    Console.Write(token);  // Prints token-by-token in real-time
}
```

### Network Class

#### Static Methods

```csharp
// Create new network
Network Create(int inputSize, int gridRows = 2, int gridCols = 2,
               int layersPerCell = 3, bool useGpu = false)

// Load complete model (ONE LINE!)
Network LoadFromString(string modelJson, string modelId = "loaded_model")

// Get library version
string GetVersion()
```

#### Instance Methods

```csharp
// Save model to JSON
string SaveToString(string modelId = "saved_model")

// Forward pass
float[] Forward(float[] input)

// Backward pass
float[] Backward(float[] gradOutput)

// Update weights
void UpdateWeights(float learningRate)

// Zero gradients
void ZeroGradients()

// Get network info
Dictionary<string, object> GetInfo()

// Dispose (free native resources)
void Dispose()
```

### Activation Enum

```csharp
public enum Activation
{
    ScaledReLU = 0,  // ReLU with 1.1x scaling (default)
    ReLU = 0,        // Alias for ScaledReLU
    Sigmoid = 1,     // Logistic sigmoid
    Tanh = 2,        // Hyperbolic tangent
    Softplus = 3,    // Smooth ReLU
    LeakyReLU = 4,   // ReLU with 0.1x negative slope
    Linear = 3       // No activation (alias for Softplus)
}
```

## 🎯 Complete Example

```csharp
using Welvet;
using System;
using System.IO;
using System.Linq;

class Program
{
    static void Main()
    {
        Console.WriteLine("🧠 WELVET Example\n");

        // Load model from JSON
        string modelJson = File.ReadAllText("test.json");
        using var network = Network.LoadFromString(modelJson, "example");

        Console.WriteLine($"✅ Model loaded! (handle: {network.Handle})");
        Console.WriteLine($"✅ LOOM version: {Network.GetVersion()}\n");

        // Create input
        float[] input = Enumerable.Repeat(0.8f, 16)
            .Concat(Enumerable.Repeat(0.2f, 16))
            .ToArray();

        // Forward pass
        Console.WriteLine("▶️ Running inference...");
        float[] output = network.Forward(input);

        Console.WriteLine($"Output: [{string.Join(", ", output.Select(x => $"{x:F6}"))}]");

        // Training
        Console.WriteLine("\n🎯 Training for 10 epochs...");
        float[] target = new float[] { 0.5f, 0.5f };

        for (int epoch = 0; epoch < 10; epoch++)
        {
            float[] currentOutput = network.Forward(input);

            // Compute loss (MSE)
            float loss = currentOutput.Zip(target, (o, t) => (o - t) * (o - t))
                .Average();

            // Backward pass
            float[] gradOutput = currentOutput.Zip(target, (o, t) => (o - t) * 2 / target.Length)
                .ToArray();
            network.Backward(gradOutput);

            // Update weights
            network.UpdateWeights(0.05f);

            if (epoch == 0 || epoch == 9)
                Console.WriteLine($"Epoch {epoch + 1}: Loss = {loss:F6}");
        }

        Console.WriteLine("\n✅ Training complete!");

        // Save model
        string savedJson = network.SaveToString("trained_model");
        File.WriteAllText("trained.json", savedJson);
        Console.WriteLine("✅ Model saved to trained.json");
    }
}
```

## 🤖 Transformer Examples

### Test All Functions

Run `TransformerTest.cs` to test loading, encoding, decoding, and streaming:

```bash
dotnet run --project TransformerTest.cs ../../models/SmolLM2-135M-Instruct
```

### Web Interface with Streaming

Run `TransformerWebInterface.cs` for a beautiful web UI with real-time streaming:

```bash
dotnet run --project TransformerWebInterface.cs ../../models/SmolLM2-135M-Instruct 8080
```

Then open http://localhost:8080/inference.html in your browser to see tokens stream in real-time!

**Features:**

- Server-Sent Events (SSE) streaming
- Beautiful token-by-token display
- Adjustable temperature and max tokens
- CORS enabled for external clients
- Same HTML UI as Python/Go versions

## 🌐 Cross-Platform Model Loading

The same JSON model file works across **all platforms**:

```csharp
// C#/.NET
using var network = Network.LoadFromString(modelJson, "model_id");
```

```python
# Python
import welvet
network = welvet.load_model_from_string(model_json, "model_id")
```

```javascript
// JavaScript/WASM
import { initLoom } from "@openfluke/welvet";
const loom = await initLoom();
const network = loom.LoadModelFromString(modelJSON, "model_id");
```

```go
// Go
network, _ := nn.LoadModelFromString(modelJSON, "model_id")
```

See `examples/all_layers_validation.go` in the main repo for a complete cross-platform test!

## 🔧 Building from Source

```bash
# Clone the repo
git clone https://github.com/openfluke/loom.git
cd loom/csharp

# Build the library
dotnet build -c Release

# Run tests (if you create them)
dotnet test

# Pack for NuGet
dotnet pack -c Release
```

## 📦 Publishing to NuGet

```bash
# Build and pack
dotnet pack -c Release

# Publish to NuGet
dotnet nuget push bin/Release/Welvet.0.0.2.nupkg \
    --api-key YOUR_API_KEY \
    --source https://api.nuget.org/v3/index.json
```

## 📖 Documentation

- [Main LOOM README](../README.md) - Framework overview
- [Python Bindings](../python/README.md) - Python API reference
- [TypeScript Bindings](../typescript/README.md) - TypeScript/WASM API
- [Examples](../examples/README.md) - Code examples and demos

## 🤝 Contributing

Contributions welcome! Please see the [main repository](https://github.com/openfluke/loom) for guidelines.

## 📄 License

Apache License 2.0 - see [LICENSE](../LICENSE) for details.

## 🙏 Acknowledgments

Built on the LOOM neural network framework by OpenFluke.
