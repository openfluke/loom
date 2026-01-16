# GPU-Accelerated Neural Network Layers

This guide covers all GPU-accelerated layer types in Loom, showing how to build neural networks that leverage GPU acceleration for both training and inference.

---

## Enabling GPU Acceleration

GPU acceleration is enabled per-network with two steps:

```go
network, _ := nn.BuildNetworkFromJSON(config)
network.GPU = true                    // Enable GPU mode
err := network.WeightsToGPU()         // Transfer weights to GPU memory
if err != nil {
    log.Fatal("GPU not available:", err)
}

// Forward automatically routes to GPU
output, duration := network.Forward(input)

// When done, release GPU resources
network.ReleaseGPUWeights()
```

All layer types below support both CPU and GPU execution with automatic parity checking.

### Supported Layers Status

| Layer | Forward | Backward | Notes |
|:------|:-------:|:--------:|:------|
| **Dense** | ✅ **Stable** | ✅ **Stable** | Best speedup (up to 20x). |
| **Conv2D** | ✅ **Stable** | ✅ **Stable** | Good for large batches/kernels. |
| **Conv1D** | ✅ **Stable** | ⚠️ **Experimental** | Accuracy under review. |
| **RNN / LSTM** | ✅ **Stable** | ⚠️ **Experimental** | Verified parity, BPTT limited. |
| **SwiGLU** | ✅ **Stable** | ⚠️ **Experimental** | Works perfectly. |
| **Norms** | ✅ **Stable** | ⚠️ **Experimental** | LayerNorm and RMSNorm supported. |
| **MHA** | ✅ **Stable** | ⚠️ **Experimental** | Multi-Head Attention supported. |

---

## Dense Layer

The Dense (fully-connected) layer is the fundamental building block. Every input connects to every output through learned weights.

### What It Does

```
Inputs (2048)                    Outputs (2048)
    │                                  │
    ├── w₀₀, w₀₁, ... ──────────────▶ o₀
    ├── w₁₀, w₁₁, ... ──────────────▶ o₁
    │        ...                       ...
    └── w₂₀₄₇,₀, ... ──────────────▶ o₂₀₄₇

Total: 2048 × 2048 = 4,194,304 weights
```

### JSON Configuration

```json
{
  "id": "dense_network",
  "batch_size": 1,
  "grid_rows": 1,
  "grid_cols": 1,
  "layers_per_cell": 5,
  "layers": [
    {"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
    {"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
    {"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
    {"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
    {"type": "dense", "activation": "sigmoid", "input_height": 2048, "output_height": 2}
  ]
}
```

### Go Code Example

```go
// Create a dense layer programmatically
layer := nn.InitDenseLayer(2048, 2048, nn.ActivationLeakyReLU)

// Or use nn.NewNetwork and SetLayer
network := nn.NewNetwork(2048, 1, 1, 3)
network.SetLayer(0, 0, 0, nn.InitDenseLayer(2048, 1024, nn.ActivationLeakyReLU))
network.SetLayer(0, 0, 1, nn.InitDenseLayer(1024, 512, nn.ActivationLeakyReLU))
network.SetLayer(0, 0, 2, nn.InitDenseLayer(512, 10, nn.ActivationSigmoid))
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `input_height` | Number of input features |
| `output_height` | Number of output features |
| `activation` | Activation function: `relu`, `leaky_relu`, `sigmoid`, `tanh`, `linear` |

---

## LayerNorm Layer

Layer Normalization normalizes activations across the feature dimension, stabilizing training by preventing value drift.

### What It Does

```
For each sample in a batch:
1. Compute mean: μ = mean(features)
2. Compute variance: σ² = var(features)
3. Normalize: x̂ = (x - μ) / √(σ² + ε)
4. Scale and shift: y = γ × x̂ + β

Where γ (gamma) and β (beta) are learnable parameters.
```

### JSON Configuration

```json
{
  "layers": [
    {"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
    {"type": "layer_norm", "norm_size": 2048, "epsilon": 1e-5},
    {"type": "layer_norm", "norm_size": 2048, "epsilon": 1e-5},
    {"type": "layer_norm", "norm_size": 2048, "epsilon": 1e-5},
    {"type": "dense", "activation": "sigmoid", "input_height": 2048, "output_height": 2}
  ]
}
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `norm_size` | Size of the feature dimension to normalize |
| `epsilon` | Small constant for numerical stability (typically `1e-5`) |

---

## RMSNorm Layer

RMS Normalization is a simplified version of LayerNorm used in modern LLMs like Llama. It only uses the root-mean-square (no mean subtraction).

### What It Does

```
rms = √(mean(x²) + ε)
output = (x / rms) × γ

Simpler than LayerNorm:
- No mean computation
- No beta parameter (just gamma)
- Slightly faster, works well empirically
```

### JSON Configuration

```json
{
  "layers": [
    {"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
    {"type": "rms_norm", "norm_size": 2048, "epsilon": 1e-5},
    {"type": "rms_norm", "norm_size": 2048, "epsilon": 1e-5},
    {"type": "rms_norm", "norm_size": 2048, "epsilon": 1e-5},
    {"type": "dense", "activation": "sigmoid", "input_height": 2048, "output_height": 2}
  ]
}
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `norm_size` | Size of the feature dimension to normalize |
| `epsilon` | Small constant for numerical stability (typically `1e-5` or `1e-6`) |

---

## Softmax Layer

Softmax converts arbitrary values into a probability distribution that sums to 1.

### What It Does

```
Input logits:  [2.0, 1.0, 0.1]
                  │
                  ▼ exp(each value)
             [7.39, 2.72, 1.11]
                  │
                  ▼ divide by sum (11.22)
Output probs: [0.66, 0.24, 0.10]
              ─────────────────
               sums to 1.0 ✓
```

### JSON Configuration

```json
{
  "layers": [
    {"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
    {"type": "softmax", "temperature": 1.0},
    {"type": "softmax", "temperature": 1.0},
    {"type": "softmax", "temperature": 1.0},
    {"type": "dense", "activation": "sigmoid", "input_height": 2048, "output_height": 2}
  ]
}
```

### Go Code Example

```go
// Standard softmax
layer := nn.InitSoftmaxLayer()

// Temperature-scaled (lower = sharper, higher = smoother)
layer := nn.InitTemperatureSoftmaxLayer(0.5)

// Grid softmax for multi-agent (each row sums to 1)
layer := nn.InitGridSoftmaxLayer(4, 8)  // 4 agents, 8 actions each

// Masked softmax (for legal moves in games)
layer := nn.InitMaskedSoftmaxLayer(10)
layer.Mask = []bool{true, true, false, true, ...}  // false = illegal
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `temperature` | Controls sharpness: 0.1=confident, 1.0=normal, 5.0=smooth |

---

## Conv1D Layer

1D Convolution slides a kernel over sequential data, detecting local patterns.

### What It Does

```
Input: [batch][channels][sequence]

Kernel (3 elements) slides across sequence:
  [a, b, c] slides over [x₀, x₁, x₂, x₃, x₄, x₅, ...]
  
  Position 0: a×x₀ + b×x₁ + c×x₂ → output[0]
  Position 1: a×x₁ + b×x₂ + c×x₃ → output[1]
  ...

Output: [batch][filters][output_length]
```

### JSON Configuration

```json
{
  "layers": [
    {"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
    {"type": "conv1d", "conv1d_in_channels": 64, "conv1d_filters": 64, 
     "conv1d_kernel_size": 3, "conv1d_stride": 1, "conv1d_padding": 1},
    {"type": "conv1d", "conv1d_in_channels": 64, "conv1d_filters": 64, 
     "conv1d_kernel_size": 3, "conv1d_stride": 1, "conv1d_padding": 1},
    {"type": "conv1d", "conv1d_in_channels": 64, "conv1d_filters": 64, 
     "conv1d_kernel_size": 3, "conv1d_stride": 1, "conv1d_padding": 1},
    {"type": "dense", "activation": "sigmoid", "input_height": 2048, "output_height": 2}
  ]
}
```

The input size 2048 = 32 sequence × 64 channels. With padding=1 and stride=1, output size stays 2048.

### Go Code Example

```go
// Conv1D: 32 seq length, 64 input channels, kernel=3, stride=1, padding=1, 64 filters
layer := nn.InitConv1DLayer(32, 64, 3, 1, 1, 64, nn.ActivationReLU)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `conv1d_in_channels` | Number of input channels |
| `conv1d_filters` | Number of output filters |
| `conv1d_kernel_size` | Size of the convolution kernel |
| `conv1d_stride` | Step size for kernel movement |
| `conv1d_padding` | Zero-padding added to input edges |

---

## Conv2D Layer

2D Convolution slides a kernel over spatial data (images), detecting local patterns like edges and textures.

### What It Does

```
Input: [batch][channels][height][width]

3×3 Kernel slides across 2D image:
┌───┬───┬───┐
│ a │ b │ c │    Convolves at each spatial position
├───┼───┼───┤    to produce one output value
│ d │ e │ f │
├───┼───┼───┤
│ g │ h │ i │
└───┴───┴───┘

Output: [batch][filters][out_height][out_width]
```

### JSON Configuration

```json
{
  "layers": [
    {"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
    {"type": "conv2d", "input_channels": 8, "filters": 8, "kernel_size": 3, 
     "stride": 1, "padding": 1, "input_height": 16, "input_width": 16},
    {"type": "conv2d", "input_channels": 8, "filters": 8, "kernel_size": 3, 
     "stride": 1, "padding": 1, "input_height": 16, "input_width": 16},
    {"type": "conv2d", "input_channels": 8, "filters": 8, "kernel_size": 3, 
     "stride": 1, "padding": 1, "input_height": 16, "input_width": 16},
    {"type": "dense", "activation": "sigmoid", "input_height": 2048, "output_height": 2}
  ]
}
```

The input 2048 = 16×16×8 (height × width × channels). With padding=1, stride=1, kernel=3, output stays at 16×16×8=2048.

### Go Code Example

```go
// Conv2D: 16×16 image, 8 input channels, 3×3 kernel, stride 1, padding 1, 8 filters
layer := nn.InitConv2DLayer(16, 16, 8, 3, 1, 1, 8, nn.ActivationReLU)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `input_height`, `input_width` | Spatial dimensions of input |
| `input_channels` | Number of input channels |
| `filters` | Number of output filters/channels |
| `kernel_size` | Size of the square kernel (e.g., 3 for 3×3) |
| `stride` | Step size for kernel movement |
| `padding` | Zero-padding added to input edges |

---

## SwiGLU Layer

SwiGLU is a gated activation used in modern LLMs (Llama, Mistral, etc.). It combines three projections with a gating mechanism.

### What It Does

```
SwiGLU(x) = down_proj(silu(gate_proj(x)) × up_proj(x))

Where:
- gate_proj: Linear projection to intermediate size
- up_proj: Another linear projection to intermediate size
- silu(x) = x × sigmoid(x) (Swish activation)
- down_proj: Project back to input size
```

### JSON Configuration

```json
{
  "layers": [
    {"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
    {"type": "swiglu", "input_height": 2048, "output_height": 2048},
    {"type": "swiglu", "input_height": 2048, "output_height": 2048},
    {"type": "swiglu", "input_height": 2048, "output_height": 2048},
    {"type": "dense", "activation": "sigmoid", "input_height": 2048, "output_height": 2}
  ]
}
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `input_height` | Input feature size |
| `output_height` | Output feature size (typically same as input) |

---

## RNN Layer

Recurrent Neural Networks process sequences by maintaining a hidden state that carries information through time.

### What It Does

```
Sequence: [x₀, x₁, x₂, x₃, ...]

     x₀        x₁        x₂        x₃
      │         │         │         │
      ▼         ▼         ▼         ▼
   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
h₀→│ RNN  │→│ RNN  │→│ RNN  │→│ RNN  │→h₄
   │ Cell │ │ Cell │ │ Cell │ │ Cell │
   └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
      │         │         │         │
      ▼         ▼         ▼         ▼
     y₀        y₁        y₂        y₃

Hidden state h carries context forward through time.
Same weights used at every step (weight sharing).
```

### JSON Configuration

```json
{
  "layers": [
    {"type": "dense", "activation": "leaky_relu", "input_height": 512, "output_height": 512},
    {"type": "rnn", "input_size": 64, "hidden_size": 64, "seq_length": 8},
    {"type": "dense", "activation": "sigmoid", "input_height": 512, "output_height": 2}
  ]
}
```

Input size 512 = 8 sequence × 64 features. Output is also 8 × 64 = 512.

### Go Code Example

```go
// RNN: 64 input features, 64 hidden size, batch size 1, sequence length 8
layer := nn.InitRNNLayer(64, 64, 1, 8)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `input_size` | Size of input features at each time step |
| `hidden_size` | Size of the hidden state |
| `seq_length` | Length of input sequences |

---

## Complete GPU Test Example

This example creates a network with each layer type and tests CPU vs GPU parity:

```go
package main

import (
    "fmt"
    "math/rand"
    
    "github.com/openfluke/loom/nn"
)

func main() {
    // Define network with Dense layers
    config := `{
        "id": "gpu_test",
        "batch_size": 1,
        "grid_rows": 1,
        "grid_cols": 1,
        "layers_per_cell": 3,
        "layers": [
            {"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
            {"type": "layer_norm", "norm_size": 2048, "epsilon": 1e-5},
            {"type": "dense", "activation": "sigmoid", "input_height": 2048, "output_height": 10}
        ]
    }`
    
    network, _ := nn.BuildNetworkFromJSON(config)
    network.BatchSize = 1
    network.InitializeWeights()
    
    // Create random input
    input := make([]float32, 2048)
    for i := range input {
        input[i] = rand.Float32()*2 - 1
    }
    
    // CPU forward pass
    network.GPU = false
    cpuOutput, cpuTime := network.Forward(input)
    fmt.Printf("CPU: %v\n", cpuTime)
    
    // GPU forward pass
    network.GPU = true
    network.WeightsToGPU()
    gpuOutput, gpuTime := network.Forward(input)  // Same API, uses GPU
    fmt.Printf("GPU: %v (%.2fx speedup)\n", gpuTime, float64(cpuTime)/float64(gpuTime))
    
    // Verify parity
    maxError := 0.0
    for i := range cpuOutput {
        if diff := abs(cpuOutput[i] - gpuOutput[i]); diff > maxError {
            maxError = diff
        }
    }
    fmt.Printf("Max error: %e\n", maxError)
    
    network.ReleaseGPUWeights()
}

func abs(x float32) float64 {
    if x < 0 { return float64(-x) }
    return float64(x)
}
```

---

## Summary

| Layer | Use Case | GPU Benefit |
|-------|----------|-------------|
| Dense | General transformations | High (matrix multiply) |
| LayerNorm | Training stability | Medium |
| RMSNorm | LLM normalization | Medium |
| Softmax | Probabilities | Medium |
| Conv1D | Sequence patterns | High |
| Conv2D | Image patterns | Very High |
| SwiGLU | LLM activations | High |
| RNN | Sequential memory | Medium |

All layers support backward pass for training with gradient computation on GPU.
