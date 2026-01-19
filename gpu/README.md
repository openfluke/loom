# Loom GPU Package

This package provides WebGPU acceleration for Loom neural network layers.
It is designed to be a drop-in replacement for CPU layers where applicable, utilizing an optimized "Virtual Machine" architecture similar to Paragon v3.

## Architecture

The GPU implementation uses a **One Pipeline Per Layer** approach with explicit resource binding.
This avoids the complexity of generating a single monolithic shader for the entire network, while still maintaining high performance through separate dispatch.

### Performance Notes

You may observe that for small configurations (e.g., shallow networks, small widths, Batch Size 1), the GPU implementation is **slower** than the CPU.
This is expected behavior due to:
1. **Latency vs Throughput**: GPUs are throughput devices. The overhead of submitting commands and transferring data (PCIe latency) dominates for small payloads. CPU has direct memory access (~nanoseconds), whereas GPU roundtrip is ~2-5ms.
2. **Setup Cost**: Creating buffers and pipelines ("Mounting") takes significant time (hundreds of ms or seconds for shader compilation). This assumes the network is long-lived.

For larger batch sizes or deeper/wider networks, the parallel compute capabilities of the GPU will outscale the CPU.

## Usage

```go
import "github.com/openfluke/loom/gpu"

// Create specs
specs := []gpu.DenseLayerSpec{...}

// Create sequence
seq := gpu.NewDenseSequence(specs)
err := seq.Build() // Must call before Forward

// Execute
out, err := seq.Forward(input)

// Cleanup
seq.Cleanup()
```

## Pipelining

For streaming applications, use `ForwardPipelined`. This submits command buffers incrementally, potentially allowing the CPU to prepare the next batch while the GPU processes the current layers (Overlap).
