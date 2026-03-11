# Paper 6: Bridging the Precision Gap: Ubiquitous WebGPU Acceleration for Universal Safetensors

> **Target Venue:** SysML, OSDI, GPGPU

## Abstract

Traditional deep learning frameworks suffer from "Kernel Proliferation," requiring separate implementations for every hardware backend and numerical precision combination. We present **Loom**, a Go-native framework that utilizes **WebGPU** as a universal compute abstraction. Loom supports the full spectrum of **13 Safetensors DTypes** (including FP4/E2M1, Int64, and Uint8) with **bit-deterministic parity (0.000000 diff)** across CPU and GPU. Our results demonstrate that 4-bit quantization (FP4) can achieve **99.4% quality** on spatial tasks like MNIST while reducing model size by **87%**.

---

## Key Results (MNIST CNN)

| DType | Quality Score | Avg Dev | File Size | RAM Usage |
|-------|---------------|---------|-----------|-----------|
| **F32** | 100.00% | 0.0000% | 2.92 MB | 5.86 MB |
| **BF16**| 100.00% | 0.0009% | 1.46 MB | 4.40 MB |
| **F4 (FP4)** | **99.40%** | **0.6029%** | **374 KB** | **3.30 MB** |
| **I8**  | 99.61% | 0.3855% | 747 KB | 3.67 MB |

---

## Architecture: The "Universal Translator"

```
[ safetensors file ]
      │
      ▼
[ LoadSafetensors ] ➔ ➔ ➔ [ Universal Loader ]
      │                         │
      │ (13 dtypes supported)   │ (Auto-conversion)
      ▼                         ▼
[ Host RAM (F32) ] <────> [ WebGPU Buffers (F32) ]
                                │
                                ▼
                      [ Universal Kernels ]
                      (Forward / Backward)
```

Loom abstracts the underlying data storage (`DType`) from the compute precision. This allows for high-precision training and low-precision deployment without manual kernel tuning.

---

## Technical Approach

1.  **Low-Bit Fidelity**: We implemented the **E2M1 (FP4)** format, utilizing a specialized "Shift-and-Scale" strategy to map trained weights into the format's dynamic range.
2.  **WebGPU Abstraction**: By leveraging `wgpu-native`, Loom provides a unified shader-based execution path that ensures bit-exact behavior across Vulkan, DX12, and Metal.
3.  **Automatic Resizing**: The framework handles byte-packing/unpacking for 4-bit formats (2 values per byte) while maintaining tensor alignment.

```go
// From nn/safetensors.go
func fp4ToFloat32(fp4 uint8) float32 {
    sign := uint32((fp4 >> 3) & 0x1)
    exponent := uint32((fp4 >> 1) & 0x3)
    mantissa := uint32(fp4 & 0x1)
    // Decode Normal/Subnormal/Zero...
    // ...
    return float32frombits(f32bits)
}
```

---

## Code References

| Component | Path |
|-----------|------|
| Multi-Precision Loader | [`nn/safetensors.go`](../nn/safetensors.go) |
| FP4 Encoder | [`nn/safetensors_save.go`](../nn/safetensors_save.go) |
| Universal Demo | [`tva/demo/mnist/main.go`](../tva/demo/mnist/main.go) |

---

## How to Reproduce

```bash
# Run the MNIST numerical type benchmark
go run tva/demo/mnist/main.go
# Observe the Comparison Summary Table at the end
```

---

**Related:** [Paper 1](research_paper_1_polyglot_runtime.md) | [Paper 2](research_paper_2_steptween.md) | [Paper 3](research_paper_3_heterogeneous_moe.md) | [Paper 4](research_paper_4_integer_training.md) | [Paper 5](research_paper_5_arc_stitching.md)
