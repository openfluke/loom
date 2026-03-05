# FP4 Weight Quantization (NVFP4 E2M1)

Loom supports ultra-low precision weight quantization for deploying large models on consumer hardware. Specifically, it implements the **NVFP4 E2M1** (NVIDIA 4-bit Floating Point, 2 exponent bits, 1 mantissa bit) format for `Dense` and `SwiGLU` layers. 

This technique reduces the VRAM and RAM footprint of model weights by approximately 81% compared to FP32, enabling models like SmolLM2-135M to run entirely within a few hundred megabytes of memory.

---

## What is NVFP4 E2M1?

FP4 is a 4-bit floating-point format. The E2M1 variant allocates:
- **1 Sign bit**
- **2 Exponent bits** (with a bias of 1)
- **1 Mantissa bit** (implicit leading 1)

This gives it a highly non-linear, dynamic range that is much better suited for neural network weights than 4-bit integers (INT4). 

### Micro-Scaling (Grouped Quantization)

To make FP4 work without catastrophic accuracy loss, Loom uses **micro-scaling**. 
Weights are grouped into blocks (typically of size 16), and each block shares a single FP32 scale factor.

```
Original (FP32)    [16 floats]   (64 bytes)
                      │
                      ▼
Quantized (FP4)    [16 nibbles]  (8 bytes)
                   [1 Scale]     (4 bytes)
                      │
Total              12 bytes per 16 weights (0.75 bytes / weight, ~81% saving)
```

The scale is calculated as the maximum absolute value in the block divided by the maximum representable value in FP4.

---

## How to use FP4 Quantization

### 1. Model Loading
You load your model exactly as you normally would (e.g., using `LoadTransformerFromSafetensors`).

```go
network, err := loadTensors("model.safetensors")
```

### 2. Building FP4 Weights
Convert the standard FP32 weights of the Dense and SwiGLU layers into the compressed FP4 format.

```go
// 1. Convert applicable layers to FP4
fp4Weights := network.BuildFP4Weights()

// 2. Clear original Float32 weights to free RAM
for i := range network.Layers {
    network.Layers[i].ClearDenseWeights()
}
```

This returns a `map[int]*FP4LayerWeights` mapping the layer index to its packed FP4 data and scales.

### 3. Execution (GPU or CPU)

**GPU Inference (Recommended)**
Loom provides specialized WebGPU compute shaders that perform matrix multiplication directly on the packed 4-bit data, dequantizing on-the-fly in GPU registers.

```go
network.GPU = true
network.GPUInferenceOnly = true

// Mount the FP4 weights instead of the normal F32 weights
err := network.WeightsFP4ToGPU(fp4Weights)

// Run inference using the normal Forward or ForwardFP4GPU methods
output, duration := network.ForwardFP4GPU(input)
```

**CPU Inference (Fallback)**
If a GPU is unavailable, Loom can still perform inference on the CPU using the compressed weights. It unpacks the weights row-by-row into a temporary float32 buffer during the forward pass.

```go
network.GPU = false
output, err := network.ForwardFP4CPU(input, fp4Weights)
```

---

## Internal Details (`nvfp4.go`)

### Packing (`PackFP4(weights []float32) PackedWeights`)
The packing process iterates over the FP32 weights in chunks of 16:
1. Finds the local absolute maximum.
2. Calculates the chunk's `Scale = max_val / 6.0`.
3. Divides every weight in the chunk by `Scale`.
4. Maps the scaled float to the nearest valid FP4 E2M1 value using a bitwise lookup table.
5. Packs two 4-bit nibbles into a single byte.

### Unpacking (`UnpackFP4(packed PackedWeights, out []float32)`)
To reconstruct the weights:
1. Each byte is split into two nibbles.
2. The nibbles are mapped back to their corresponding FP32 values using a precomputed lookup table.
3. The resulting floats are multiplied by the chunk's shared `Scale`.

*(Note: The reconstructed weights will not be perfectly identical to the original FP32 weights due to the quantization loss, but the accuracy degradation is usually negligible for well-trained LLMs).*

---

## Supported Layers

Currently, FP4 quantization is exclusively supported for:
- **Dense Layer** (`gpu.FP4DenseLayer`)
- **SwiGLU Layer** (`gpu.FP4SwiGLULayer`)

Other layers like Convolution, Multi-Head Attention, and LayerNorm remain in FP32 because they either contain too few weights to justify the quantization cost or are highly sensitive to precision loss.

---

## Under the Hood: WebGPU WGSL Shaders

The actual magic of Loom's FP4 implementation happens inside the custom WebGPU compute shaders (`dense_fp4.go`, `swiglu_fp4.go`).

Unlike traditional quantization approaches that decompress weights back into a floating-point VRAM buffer *before* multiplication, Loom keeps the weights packed in VRAM.

1. **Register-Level Unpacking**: 
   The shader reads a single 32-bit integer from VRAM, which contains **8** packed FP4 weights (4 bytes).
   The shader extracts each nibble using bitwise operations (`>>` and `& 0xF`) and converts it back to an `f32` inside the GPU's ultra-fast local registers.
2. **WGSL Implementation**:
   Because WGSL does not have an elegant built-in way to decode arbitrary 4-bit floating point formats, Loom writes a lookup table (or bit manipulation logic) directly into the shader code. The weights are multiplied with the incoming activation, scaled, and accumulated.
3. **Optimized SwiGLU**:
   The `FP4SwiGLULayer` fuses the `Gate`, `Up`, and `Down` projections of an LLM feedforward block into a single pipeline, executing the non-linear SiLU activation and element-wise multiplication on-chip to avoid unnecessary VRAM round-trips.

---

## GPU Residual Connections

Loom supports GPU-side residual connections (skip-connections) to keep activations completely on the GPU between Transformer blocks.

For FP4 layers, this is seamlessly integrated:
```go
network.EnableGPUResiduals = true
```
When this is enabled, the FP4 layers inject their output directly into a shared GPU residual buffer (using an atomic or in-place addition shader) instead of downloading the activations back to the CPU.

---

## VRAM Tracking

With models pushing the VRAM limits of consumer GPUs, tracking exact memory allocation is critical when using quantization.

Loom includes a built-in VRAM Tracker (`gpu.TrackVRAM`) that hooks directly into the WebGPU buffer creation pipeline:

```go
// Add the following in your code to print a profile of the highest VRAM consumers:
gpu.PrintVRAMUsage()
```

This will output a profile showing total tracked VRAM, the number of FP4 layers versus FP32 layers, and an ordered list of the largest individual buffers (e.g., `GPU LM head`, `KCacheBuffer`, `L0_GateData`).

Example output:
```text
--- VRAM Usage Profile ---
Layers: 120 total (30 FP4)
--------------------------
GPU LM head                   :   108.00 MB
KCacheBuffer                  :     0.50 MB
L0_GateData                   :     2.00 MB
...
Total Tracked VRAM: 350.52 MB
--------------------------
```

---

## Using the Built-in Tools

Loom provides two out-of-the-box tools in the `tva/` directory to showcase FP4 quantization on Large Language Models like SmolLM2 and Qwen:
- **`fp4_quicktalk`**: An interactive CLI application for chatting with FP4-compressed LLMs. Type `vram` inside the chat to test the VRAM tracker.
- **`fp4_test`**: A performance and determinism benchmarking tool that verifies the FP4 WebGPU compute shaders perfectly match the Go CPU implementation.

See the [Examples Guide](./examples.md) for more information on how to run them.
