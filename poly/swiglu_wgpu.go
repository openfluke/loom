package poly

import (
	"fmt"
	"github.com/openfluke/webgpu/wgpu"
)

// SwiGLUForwardWGPU handles the GPU-accelerated forward pass for a SwiGLU layer.
func SwiGLUForwardWGPU[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if layer.Network.GPUContext == nil {
		return SwiGLUForwardTiled(layer, input) // Fallback
	}
	ctx := layer.Network.GPUContext
	inputSize, intermediateSize := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inputSize

	preAct = NewTensor[T](seqLen, intermediateSize)
	postAct = NewTensor[T](seqLen, inputSize)

	// Determine tile sizes
	scTile, mcTile := ctx.GPUTileSize*4, int(ctx.Limits.MaxComputeInvocationsPerWorkgroup)
	if scTile < 64 { scTile = 64 }
	if mcTile > 256 { mcTile = 256 }
	tileSize := scTile
	if layer.Network.EnableMultiCoreTiling {
		tileSize = mcTile
	}

	// 1. Prepare Buffers
	// Upload weights and biases if not already resident
	if !layer.IsGPUResident {
		if err := layer.SyncToGPU(); err != nil {
			fmt.Printf("GPU Sync Error: %v\n", err)
			return SwiGLUForwardTiled(layer, input)
		}
	}

	// Upload input
	inputBuf := ctx.GetActivationBuffer("swiglu_input", uint64(len(input.Data)*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	ctx.Queue.WriteBuffer(inputBuf, 0, wgpu.ToBytes(ConvertTensor[T, float32](input).Data))

	// Intermediate and Output buffers
	interOut := ctx.GetActivationBuffer("swiglu_inter", uint64(seqLen*intermediateSize*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	mlpOut := ctx.GetActivationBuffer("swiglu_out", uint64(seqLen*inputSize*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)

	// 2. Dispatch Kernels
	gW, _ := layer.WeightStore.GPUWeights[DType(100)].(*wgpu.Buffer)
	uW, _ := layer.WeightStore.GPUWeights[DType(101)].(*wgpu.Buffer)
	dW, _ := layer.WeightStore.GPUWeights[DType(102)].(*wgpu.Buffer)

	if gW == nil || uW == nil || dW == nil {
		fmt.Printf("GPU Weights missing for SwiGLU\n")
		return SwiGLUForwardTiled(layer, input)
	}

	// Gate & Up
	gB, _ := layer.WeightStore.GPUWeights[DType(110)].(*wgpu.Buffer)
	uB, _ := layer.WeightStore.GPUWeights[DType(111)].(*wgpu.Buffer)
	if gB == nil { gB = ctx.BlankBuffer }
	if uB == nil { uB = ctx.BlankBuffer }

	if layer.DType == DTypeFP4 || layer.DType == DTypeInt4 || layer.DType == DTypeUint4 {
		sg := layer.WeightStore.GPUScales[DType(100)]
		su := layer.WeightStore.GPUScales[DType(101)]
		if sg == nil || su == nil {
			fmt.Printf("GPU Scales missing for SwiGLU Q4\n")
			return SwiGLUForwardTiled(layer, input)
		}
		if err := ctx.DispatchSwiGLUQ4(seqLen, inputSize, intermediateSize, inputBuf, sg, gW, su, uW, gB, uB, interOut, tileSize); err != nil {
			fmt.Printf("GPU Dispatch Error (Fwd Q4): %v\n", err)
			return SwiGLUForwardTiled(layer, input)
		}
	} else {
		if err := ctx.DispatchSwiGLU(seqLen, inputSize, intermediateSize, inputBuf, gW, uW, gB, uB, interOut, tileSize); err != nil {
			fmt.Printf("GPU Dispatch Error (Fwd): %v\n", err)
			return SwiGLUForwardTiled(layer, input)
		}
	}

	// Down Projection
	dB, _ := layer.WeightStore.GPUWeights[DType(112)].(*wgpu.Buffer)
	if dB == nil { dB = ctx.BlankBuffer }

	// Use DispatchDenseTiled or DispatchDense with bias support
	var act uint32 = 99 // ActivationLinear is -1 (Linear), use 99 for WGPU linear kernel
	if err := ctx.DispatchDenseTiled(tileSize, seqLen, intermediateSize, inputSize, act, 1.0, interOut, dW, dB, mlpOut); err != nil {
		fmt.Printf("GPU Dispatch Error (Down): %v\n", err)
		return SwiGLUForwardTiled(layer, input)
	}

	// 3. Download results
	preData, err := ctx.ReadBuffer(interOut)
	if err != nil { return SwiGLUForwardTiled(layer, input) }
	postData, err := ctx.ReadBuffer(mlpOut)
	if err != nil { return SwiGLUForwardTiled(layer, input) }

	for i := range preAct.Data { preAct.Data[i] = T(preData[i]) }
	for i := range postAct.Data { postAct.Data[i] = T(postData[i]) }

	return preAct, postAct
}

// SwiGLUBackwardWGPU handles the GPU-accelerated backward pass for a SwiGLU layer.
func SwiGLUBackwardWGPU[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	// FALLBACK to CPU for now if architecture is unclear.
	// We need gateIn and upIn separately for the ShaderSwiGLUBackward, but preAct only stores [silu(gate) * up].
	return SwiGLUBackwardTiled(layer, gradOutput, input, preAct)
}
