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
		gateOut[i] = gateOut[i] * upOut[i] // gateOut now holds the intermediate activation
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

// SwiGLUBackward computes gradients for SwiGLU.
func SwiGLUBackward[T Numeric](
	input, gradOutput *Tensor[T],
	gateWeights, upWeights, downWeights, gateBias, upBias, downBias *Tensor[T],
	inputSize, intermediateSize, seqLen int,
) (gradInput, gradGateW, gradUpW, gradDownW, gradGateB, gradUpB, gradDownB *Tensor[T]) {
	
	gradInput = NewTensor[T](seqLen * inputSize)
	gradGateW = NewTensor[T](inputSize * intermediateSize)
	gradUpW = NewTensor[T](inputSize * intermediateSize)
	gradDownW = NewTensor[T](intermediateSize * inputSize)
	gradGateB = NewTensor[T](intermediateSize)
	gradUpB = NewTensor[T](intermediateSize)
	gradDownB = NewTensor[T](inputSize)

	// Recompute forward pass intermediates needed for backward
	// This inefficient recomputation avoids storing huge state, but costs compute.
	gatePreAct := make([]float64, seqLen*intermediateSize)
	upPreAct := make([]float64, seqLen*intermediateSize)
	siluGate := make([]float64, seqLen*intermediateSize)
	intermediateAct := make([]float64, seqLen*intermediateSize) // silu(gate) * up

	// Forward Recomputation
	for s := 0; s < seqLen; s++ {
		for i := 0; i < intermediateSize; i++ {
			// Gate
			sumGate := float64(gateBias.Data[i])
			for j := 0; j < inputSize; j++ {
				sumGate += float64(input.Data[s*inputSize+j]) * float64(gateWeights.Data[j*intermediateSize+i])
			}
			gatePreAct[s*intermediateSize+i] = sumGate
			
			// Up
			sumUp := float64(upBias.Data[i])
			for j := 0; j < inputSize; j++ {
				sumUp += float64(input.Data[s*inputSize+j]) * float64(upWeights.Data[j*intermediateSize+i])
			}
			upPreAct[s*intermediateSize+i] = sumUp
			
			// Calculate activation
			x := sumGate
			sig := 1.0 / (1.0 + math.Exp(-x))
			silu := x * sig
			siluGate[s*intermediateSize+i] = silu
			intermediateAct[s*intermediateSize+i] = silu * sumUp
		}
	}

	// Backward Pass
	gradIntermediate := make([]float64, seqLen*intermediateSize)

	for s := 0; s < seqLen; s++ {
		// Backprop through Down Projection
		for i := 0; i < inputSize; i++ {
			grad := float64(gradOutput.Data[s*inputSize+i])
			gradDownB.Data[i] += T(grad)
			
			for j := 0; j < intermediateSize; j++ {
				// grad w.r.t downWeights: intermediateAct[j] * grad
				gradDownW.Data[j*inputSize+i] += T(intermediateAct[s*intermediateSize+j] * grad)
				
				// grad w.r.t intermediate: downWeights[j, i] * grad
				gradIntermediate[s*intermediateSize+j] += grad * float64(downWeights.Data[j*inputSize+i])
			}
		}

		// Backprop through Activation (Element-wise)
		for i := 0; i < intermediateSize; i++ {
			gradInter := gradIntermediate[s*intermediateSize+i]
			
			// d/d(UpPreAct) = gradInter * siluGate[i]
			dUp := gradInter * siluGate[s*intermediateSize+i]
			
			// d/d(GatePreAct) = gradInter * UpPreAct[i] * d/dx(silu(x))
			// d/dx(x*sig(x)) = sig(x) + x*sig(x)*(1-sig(x)) = sig(x) * (1 + x*(1-sig(x)))
			x := gatePreAct[s*intermediateSize+i]
			sig := 1.0 / (1.0 + math.Exp(-x))
			dSilu := sig * (1.0 + x*(1.0-sig))
			
			dGate := gradInter * upPreAct[s*intermediateSize+i] * dSilu
			
			// Accumulate Bias gradients
			gradUpB.Data[i] += T(dUp)
			gradGateB.Data[i] += T(dGate)
			
			// Backprop to weights and input
			for j := 0; j < inputSize; j++ {
				inVal := float64(input.Data[s*inputSize+j])
				
				// Weights
				gradUpW.Data[j*intermediateSize+i] += T(inVal * dUp)
				gradGateW.Data[j*intermediateSize+i] += T(inVal * dGate)
				
				// Input
				gradInput.Data[s*inputSize+j] += T(dUp*float64(upWeights.Data[j*intermediateSize+i]) + 
												  dGate*float64(gateWeights.Data[j*intermediateSize+i]))
			}
		}
	}

	return gradInput, gradGateW, gradUpW, gradDownW, gradGateB, gradUpB, gradDownB
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
