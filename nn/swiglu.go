package nn

import (
	"math"
)

// SwiGLUForwardCPU performs SwiGLU gated activation on CPU
// SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
func SwiGLUForwardCPU(input []float32, config *LayerConfig, batchSize int) ([]float32, []float32) {
	inputSize := config.InputHeight         // Hidden size
	intermediateSize := config.OutputHeight // Intermediate size
	seqLen := len(input) / inputSize

	// Output size is back to inputSize (hidden)
	output := make([]float32, seqLen*inputSize)

	// Intermediate activations for gate and up projections
	gateOut := make([]float32, seqLen*intermediateSize)
	upOut := make([]float32, seqLen*intermediateSize)

	// 1. Gate projection: gate_proj(x)
	// GateWeights is transposed: [inputSize][intermediateSize] in column-major order
	for s := 0; s < seqLen; s++ {
		for i := 0; i < intermediateSize; i++ {
			sum := config.GateBias[i]
			for j := 0; j < inputSize; j++ {
				// Access transposed weights: [j][i] = [j*intermediateSize + i]
				sum += input[s*inputSize+j] * config.GateWeights[j*intermediateSize+i]
			}
			gateOut[s*intermediateSize+i] = sum
		}
	}

	// 2. Up projection: up_proj(x)
	// UpWeights is transposed: [inputSize][intermediateSize] in column-major order
	for s := 0; s < seqLen; s++ {
		for i := 0; i < intermediateSize; i++ {
			sum := config.UpBias[i]
			for j := 0; j < inputSize; j++ {
				// Access transposed weights: [j][i] = [j*intermediateSize + i]
				sum += input[s*inputSize+j] * config.UpWeights[j*intermediateSize+i]
			}
			upOut[s*intermediateSize+i] = sum
		}
	}

	// 3. Apply SiLU to gate: silu(gate_proj(x)) = x * sigmoid(x)
	for i := 0; i < seqLen*intermediateSize; i++ {
		x := gateOut[i]
		sigmoid := 1.0 / (1.0 + float32(math.Exp(float64(-x))))
		gateOut[i] = x * sigmoid // SiLU activation
	}

	// 4. Element-wise multiply: silu(gate_proj(x)) * up_proj(x)
	for i := 0; i < seqLen*intermediateSize; i++ {
		gateOut[i] = gateOut[i] * upOut[i]
	}

	// 5. Down projection: down_proj(gated_output)
	// DownWeights is transposed: [intermediateSize][inputSize] in column-major order
	for s := 0; s < seqLen; s++ {
		for i := 0; i < inputSize; i++ {
			sum := config.DownBias[i]
			for j := 0; j < intermediateSize; j++ {
				// Access transposed weights: [j][i] = [j*inputSize + i]
				sum += gateOut[s*intermediateSize+j] * config.DownWeights[j*inputSize+i]
			}
			output[s*inputSize+i] = sum
		}
	}

	// Notify observer if present
	if config.Observer != nil {
		notifyObserver(config, "forward", -1, input, output, 0)
	}

	// Return output as both pre and post activation (no activation at output)
	return output, output
}
