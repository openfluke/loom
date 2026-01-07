# Understanding Model Serialization

This guide explains how Loom saves and loads neural network models—what actually gets stored, how the formats work, and how to use serialization for different deployment scenarios.

---

## What Gets Saved?

When you save a model, you're capturing:

1. **Architecture**: The structure of the network (grid size, layer types, configuration)
2. **Weights**: The learned parameters (millions of floating point numbers)
3. **Metadata**: Version info, model ID, creation time

```
Saved Model File
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Architecture                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Grid: 2×3, LayersPerCell: 2                               │  │
│  │                                                           │  │
│  │ Layer[0,0,0]: Dense, 1024→512, ReLU                       │  │
│  │ Layer[0,0,1]: Dense, 512→256, ReLU                        │  │
│  │ Layer[0,1,0]: Attention, heads=8, dim=256                 │  │
│  │ ...                                                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Weights (encoded)                                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Format: Base64                                            │  │
│  │ Data: "eyJ0eXBlIjoiZmxvYXQzMi1hcnJheSIsImxl..."           │  │
│  │       (millions of numbers compressed to text)            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Metadata                                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ ID: "my-classifier-v1"                                    │  │
│  │ Type: "modelhost/bundle"                                  │  │
│  │ Version: 1                                                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Bundle Format

Loom uses a "bundle" format that can contain multiple models. This is useful for:
- Encoder-decoder pairs
- Ensemble models
- Different versions of the same model

```json
{
  "type": "modelhost/bundle",
  "version": 1,
  "models": [
    {
      "id": "encoder",
      "cfg": { ... architecture ... },
      "weights": { ... encoded weights ... }
    },
    {
      "id": "decoder",
      "cfg": { ... architecture ... },
      "weights": { ... encoded weights ... }
    }
  ]
}
```

Even single models use this format (just with one entry in the `models` array). This keeps the format consistent.

---

## How Weights Are Encoded

The challenging part is encoding millions of floating-point numbers efficiently. Here's what happens:

### Step 1: Binary Conversion

Float32 values are converted to their binary representation:

```
Float32: 3.14159...

In memory (IEEE 754):
    Sign:     0 (positive)
    Exponent: 10000000 (128, meaning 2^1)
    Mantissa: 10010010000111111011011
    
Binary bytes: [0x40, 0x49, 0x0F, 0xDB]
```

### Step 2: Base64 Encoding

Binary data is converted to text using Base64 (only uses safe ASCII characters):

```
Binary bytes:  [0x40, 0x49, 0x0F, 0xDB, ...]
                         ↓
Base64 string: "QEkP2w=="   (roughly 33% larger than binary)
```

### Why Base64?

JSON can't directly contain binary data (binary bytes might include special characters). Base64 ensures the weights are safe to embed in JSON:

```
Direct binary in JSON: BROKEN
    {"weights": "AB\x00CD\xFF..."}
              ↑              ↑
         Null byte       Invalid UTF-8
         breaks JSON     breaks JSON

Base64 in JSON: SAFE
    {"weights": "QEkP2w/rABC..."}
                 ↑
    Only letters, numbers, +, /, =
```

---

## File-Based Serialization

### Saving a Single Model

```go
err := network.SaveModel("model.json", "my-classifier")
```

What happens:
1. Network traverses all layers
2. For each layer: records type, size, activation
3. For each weight matrix: converts to bytes, then Base64
4. Writes JSON to file

```
Network in memory                    model.json on disk
┌─────────────────┐                 ┌─────────────────────┐
│ Grid: 2×2       │────────────────▶│ {                   │
│ Layers: [...]   │   SaveModel()   │   "type": "...",    │
│ Weights: [...]  │                 │   "models": [...]   │
└─────────────────┘                 │ }                   │
                                    └─────────────────────┘
```

### Loading a Model

```go
network, err := nn.LoadModel("model.json", "my-classifier")
```

What happens:
1. Reads JSON from file
2. Parses architecture configuration
3. Creates empty network with correct structure
4. Decodes Base64 weights back to floats
5. Populates network with weights

```
model.json                          Network in memory
┌─────────────────────┐            ┌─────────────────┐
│ {                   │            │ Grid: 2×2       │
│   "models": [{      │──────────▶ │ Layers: [Dense, │
│     "cfg": {...},   │ LoadModel()│          Softmax]│
│     "weights": "..."│            │ Weights: [1.2,  │
│   }]                │            │          -0.5,  │
│ }                   │            │          ...]   │
└─────────────────────┘            └─────────────────┘
```

---

## String-Based Serialization

Sometimes you don't have a file system—for example, in WebAssembly or when sending models over a network.

### Saving to String

```go
jsonString, err := network.SaveModelToString("my-classifier")
// jsonString is now a complete JSON representation
```

The output is exactly the same as file-based, but returned as a string instead of written to disk.

### Loading from String

```go
network, err := nn.LoadModelFromString(jsonString, "my-classifier")
```

### Use Cases

```
File-based:                         String-based:
┌─────────────┐                     ┌─────────────┐
│ Desktop app │                     │ Browser     │
│ Server      │                     │ WebAssembly │
│ CLI tools   │                     │ REST API    │
│ Notebooks   │                     │ Database    │
└─────────────┘                     │ Serverless  │
                                    │ Mobile      │
                                    └─────────────┘

File-based works when you have                String-based works when
disk access and want persistence.             you need to move models
                                              around without files.
```

---

## Multi-Precision Weights

Not all weights need to be 32-bit floats. Loom supports multiple precisions:

### Precision Options

```
Precision     Size    Range                 Use case
─────────────────────────────────────────────────────────────────
float64       8 bytes ±10^308               High-precision research
float32       4 bytes ±10^38                Standard training
float16       2 bytes ±65504                Inference, GPU
int32         4 bytes ±2 billion            Integer networks
int16         2 bytes ±32767                Quantized models
int8          1 byte  ±127                  Edge devices, mobile
```

### How It Works

```
Original weights (float32):     [0.523, -0.127, 0.891, ...]
                                   ↓
                              Quantize to int8
                                   ↓
Quantized (int8):              [67, -16, 114, ...]
                                + scale factor: 0.00785
                                + zero point: 0
                                   ↓
                              Encode to file
                                   ↓
On disk: compact int8 representation (4× smaller!)
                                   ↓
                              Decode on load
                                   ↓
Restored float32:              [0.526, -0.126, 0.895, ...]
                                (small precision loss)
```

### Size Comparison

```
1 million weights:

float64:    8 MB
float32:    4 MB  ← Standard
float16:    2 MB  ← GPU inference
int8:       1 MB  ← Edge/mobile

For a 7B parameter model:
    float32: 28 GB
    int8:    7 GB   ← Actually deployable on consumer hardware!
```

---

## Loading External Models: SafeTensors

HuggingFace models are often stored in "SafeTensors" format. Loom can load these directly.

### What is SafeTensors?

SafeTensors is a simple format for storing tensors:

```
SafeTensors File Structure:

┌───────────────────────────────────────────────────────┐
│ Header (JSON)                                         │
│ ┌───────────────────────────────────────────────────┐ │
│ │ {                                                 │ │
│ │   "model.embed_tokens.weight": {                  │ │
│ │     "dtype": "F16",                               │ │
│ │     "shape": [151552, 576],                       │ │
│ │     "data_offsets": [0, 174635008]                │ │
│ │   },                                              │ │
│ │   "model.layers.0.self_attn.q_proj.weight": {     │ │
│ │     "dtype": "F16",                               │ │
│ │     "shape": [576, 576],                          │ │
│ │     "data_offsets": [174635008, 174967424]        │ │
│ │   },                                              │ │
│ │   ...                                             │ │
│ │ }                                                 │ │
│ └───────────────────────────────────────────────────┘ │
├───────────────────────────────────────────────────────┤
│ Binary Data                                           │
│ ┌───────────────────────────────────────────────────┐ │
│ │ [raw bytes for embed_tokens.weight]               │ │
│ │ [raw bytes for q_proj.weight]                     │ │
│ │ [raw bytes for k_proj.weight]                     │ │
│ │ ...                                               │ │
│ └───────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
```

### Loading SafeTensors

```go
tensors, err := nn.LoadSafeTensors("model.safetensors")

// Access individual tensors by name
embeddingWeights := tensors["model.embed_tokens.weight"]
```

### Data Type Handling

SafeTensors may store weights in float16 or bfloat16. Loom automatically converts:

```
File contains:     float16 (2 bytes each)
                        ↓
                   Auto-convert
                        ↓
In memory:         float32 (4 bytes each)

You don't need to worry about the conversion!
```

---

## Generic Model Loading

What if you have an unknown model format? Loom can auto-detect:

```go
network, detected, err := nn.LoadGenericFromBytes(weightsData, configData)
```

### The Detection Process

```
Input: mystery safetensors file
              │
              ▼
┌─────────────────────────────────────────────┐
│ 1. Parse safetensors header                 │
│    • Extract tensor names and shapes        │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│ 2. Analyze tensor names                     │
│    • "model.layers.0.self_attn.q_proj"     │
│      → Looks like attention                 │
│    • "model.embed_tokens"                   │
│      → Looks like embedding                 │
│    • "model.layers.0.mlp.gate_proj"        │
│      → Looks like SwiGLU                    │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│ 3. Build network architecture               │
│    • Create layers matching detected types  │
│    • Wire them together appropriately       │
│    • Load weights into correct layers       │
└─────────────────────────────────────────────┘
                      │
                      ▼
           Ready-to-use Network!
```

### The `detected` Return Value

The function returns detected tensor info for inspection:

```go
for _, t := range detected {
    fmt.Printf("%s: %v (%s)\n", t.Name, t.Shape, t.Type)
}

// Output:
// model.embed_tokens.weight: [151552, 576] (Embedding)
// model.layers.0.self_attn.q_proj.weight: [576, 576] (Attention)
// model.layers.0.self_attn.k_proj.weight: [192, 576] (Attention)
// ...
```

---

## Transformer-Specific Loading

For Llama-style transformers, there's a specialized loader:

```go
network, err := nn.LoadTransformerFromSafetensors("./models/llama-7b/")
```

### What It Understands

```
Llama Architecture Pattern:

model.embed_tokens.weight           → Embedding layer
model.norm.weight                   → Final RMSNorm
lm_head.weight                      → Output projection

For each of N layers:
    model.layers.{i}.input_layernorm.weight    → Pre-attention norm
    model.layers.{i}.self_attn.q_proj.weight   → Query projection
    model.layers.{i}.self_attn.k_proj.weight   → Key projection
    model.layers.{i}.self_attn.v_proj.weight   → Value projection
    model.layers.{i}.self_attn.o_proj.weight   → Output projection
    model.layers.{i}.post_attention_layernorm.weight  → Pre-MLP norm
    model.layers.{i}.mlp.gate_proj.weight      → SwiGLU gate
    model.layers.{i}.mlp.up_proj.weight        → SwiGLU up
    model.layers.{i}.mlp.down_proj.weight      → SwiGLU down
```

### Supported Models

- Llama, Llama 2, Llama 3
- Mistral
- Qwen2.5
- TinyLlama
- SmolLM
- Any model using the Llama architecture

---

## Cross-Platform Deployment

One of Loom's strengths is that saved models work across all platforms:

```
Model saved in Go
        │
        ▼
   model.json
        │
        ├──────────────────────────────────────────┐
        │                    │                     │
        ▼                    ▼                     ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│     Go      │      │   Browser   │      │   Python    │
│   Native    │      │    WASM     │      │   welvet    │
└─────────────┘      └─────────────┘      └─────────────┘
        │                    │                     │
        ▼                    ▼                     ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│    C/C++    │      │  TypeScript │      │     C#      │
│   via CABI  │      │   bindings  │      │   Welvet    │
└─────────────┘      └─────────────┘      └─────────────┘
```

### Loading in Different Languages

**Go (native)**:
```go
network, _ := nn.LoadModel("model.json", "my_model")
output, _ := network.ForwardCPU(input)
```

**JavaScript (WASM)**:
```javascript
const json = await fetch('model.json').then(r => r.text());
const network = loom.LoadNetworkFromString(json, "my_model");
const output = network.ForwardCPU(inputArray);
```

**Python (welvet)**:
```python
with open("model.json") as f:
    json_str = f.read()
network = welvet.load_model_from_string(json_str, "my_model")
output = network.forward_cpu(input_array)
```

**C (CABI)**:
```c
char* json = read_file("model.json");
Network* net = LoomLoadModel(json, "my_model");
float* output = LoomForward(net, input, input_len);
```

---

## Practical Tips

### Versioning Models

Include version in the model ID:
```go
network.SaveModel("checkpoints/model_v2.1.0.json", "classifier_v2.1.0")
```

### Checkpointing During Training

Save periodically to recover from crashes:
```go
for epoch := 0; epoch < 1000; epoch++ {
    // ... training ...
    
    if epoch % 100 == 0 {
        network.SaveModel(
            fmt.Sprintf("checkpoints/epoch_%04d.json", epoch),
            "training_checkpoint",
        )
    }
}
```

### Validating Loaded Models

After loading, verify the model works:
```go
loaded, err := nn.LoadModel("model.json", "my_model")
if err != nil {
    return err
}

// Test with known input
testInput := make([]float32, 1024)
output, _ := loaded.ForwardCPU(testInput)

// Check output is reasonable (not NaN, not all zeros)
if math.IsNaN(float64(output[0])) {
    return errors.New("model produces NaN")
}
```

---

## Summary

Serialization captures the complete state of a trained model:
- **Architecture**: Layer types, sizes, configurations
- **Weights**: Millions of learned parameters
- **Format**: JSON with Base64-encoded binary weights

Key operations:
- `SaveModel` / `LoadModel` - File-based
- `SaveModelToString` / `LoadModelFromString` - String-based
- `LoadSafeTensors` - HuggingFace format
- `LoadGenericFromBytes` - Auto-detect format
- `LoadTransformerFromSafetensors` - Llama-style models

The same model file works across Go, WASM, Python, C#, and C.
