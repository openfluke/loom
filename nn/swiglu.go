package nn

import (
	"math"
)

// =============================================================================
// Generic SwiGLU Implementation
// =============================================================================

// SwiGLUForward performs SwiGLU gated activation for any numeric type.
// SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
func SwiGLUForward[T Numeric](
	input, gateWeights, upWeights, downWeights, gateBias, upBias, downBias *Tensor[T],
	inputSize, intermediateSize, seqLen int,
) (output *Tensor[T]) {
	output = NewTensor[T](seqLen * inputSize)
	gateOut := make([]float64, seqLen*intermediateSize)
	upOut := make([]float64, seqLen*intermediateSize)

	// 1. Gate projection
	for s := 0; s < seqLen; s++ {
		for i := 0; i < intermediateSize; i++ {
			sum := float64(gateBias.Data[i])
			for j := 0; j < inputSize; j++ {
				sum += float64(input.Data[s*inputSize+j]) * float64(gateWeights.Data[j*intermediateSize+i])
			}
			gateOut[s*intermediateSize+i] = sum
		}
	}

	// 2. Up projection
	for s := 0; s < seqLen; s++ {
		for i := 0; i < intermediateSize; i++ {
			sum := float64(upBias.Data[i])
			for j := 0; j < inputSize; j++ {
				sum += float64(input.Data[s*inputSize+j]) * float64(upWeights.Data[j*intermediateSize+i])
			}
			upOut[s*intermediateSize+i] = sum
		}
	}

	// 3. Apply SiLU to gate
	for i := 0; i < seqLen*intermediateSize; i++ {
		x := gateOut[i]
		sigmoid := 1.0 / (1.0 + math.Exp(-x))
		gateOut[i] = x * sigmoid
	}

	// 4. Element-wise multiply
	for i := 0; i < seqLen*intermediateSize; i++ {
		gateOut[i] = gateOut[i] * upOut[i]
	}

	// 5. Down projection
	for s := 0; s < seqLen; s++ {
		for i := 0; i < inputSize; i++ {
			sum := float64(downBias.Data[i])
			for j := 0; j < intermediateSize; j++ {
				sum += gateOut[s*intermediateSize+j] * float64(downWeights.Data[j*inputSize+i])
			}
			output.Data[s*inputSize+i] = T(sum)
		}
	}

	return output
}

// =============================================================================
// Backward-compatible float32 function
// =============================================================================

// SwiGLUForwardCPU performs SwiGLU gated activation on CPU
func SwiGLUForwardCPU(input []float32, config *LayerConfig, batchSize int) ([]float32, []float32) {
	inputSize := config.InputHeight
	intermediateSize := config.OutputHeight
	seqLen := len(input) / inputSize

	inputT := NewTensorFromSlice(input, len(input))
	gateWT := NewTensorFromSlice(config.GateWeights, len(config.GateWeights))
	upWT := NewTensorFromSlice(config.UpWeights, len(config.UpWeights))
	downWT := NewTensorFromSlice(config.DownWeights, len(config.DownWeights))
	gateBT := NewTensorFromSlice(config.GateBias, len(config.GateBias))
	upBT := NewTensorFromSlice(config.UpBias, len(config.UpBias))
	downBT := NewTensorFromSlice(config.DownBias, len(config.DownBias))

	result := SwiGLUForward(inputT, gateWT, upWT, downWT, gateBT, upBT, downBT, inputSize, intermediateSize, seqLen)
	return result.Data, result.Data
}

