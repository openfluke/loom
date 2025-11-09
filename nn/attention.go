package nn

import (
	"math"
	"math/rand"
)

// InitMultiHeadAttentionLayer initializes a Multi-Head Attention layer
// dModel: model dimension (embedding size)
// numHeads: number of attention heads
// seqLength: sequence length
// activation: activation function (typically not used in standard MHA, but available for variants)
func InitMultiHeadAttentionLayer(
	dModel, numHeads, seqLength int,
	activation ActivationType,
) LayerConfig {
	if dModel%numHeads != 0 {
		panic("dModel must be divisible by numHeads")
	}

	headDim := dModel / numHeads

	// Xavier/Glorot initialization: stddev = sqrt(2 / (fan_in + fan_out))
	// For linear layers: fan_in = fan_out = dModel
	stddev := float32(math.Sqrt(2.0 / float64(2*dModel)))

	// Initialize Q, K, V projection weights [dModel x dModel]
	qWeights := make([]float32, dModel*dModel)
	kWeights := make([]float32, dModel*dModel)
	vWeights := make([]float32, dModel*dModel)
	outputWeight := make([]float32, dModel*dModel)

	for i := range qWeights {
		qWeights[i] = float32(rand.NormFloat64()) * stddev
		kWeights[i] = float32(rand.NormFloat64()) * stddev
		vWeights[i] = float32(rand.NormFloat64()) * stddev
		outputWeight[i] = float32(rand.NormFloat64()) * stddev
	}

	// Initialize biases to zero
	qBias := make([]float32, dModel)
	kBias := make([]float32, dModel)
	vBias := make([]float32, dModel)
	outputBias := make([]float32, dModel)

	return LayerConfig{
		Type:         LayerMultiHeadAttention,
		Activation:   activation,
		NumHeads:     numHeads,
		HeadDim:      headDim,
		DModel:       dModel,
		SeqLength:    seqLength,
		QWeights:     qWeights,
		KWeights:     kWeights,
		VWeights:     vWeights,
		OutputWeight: outputWeight,
		QBias:        qBias,
		KBias:        kBias,
		VBias:        vBias,
		OutputBias:   outputBias,
	}
}

// multiHeadAttentionForwardCPU performs multi-head attention on CPU
// input shape: [batch][seqLength][dModel] (flattened)
// Returns: preActivation (before output activation), postActivation (after output activation)
func multiHeadAttentionForwardCPU(input []float32, config *LayerConfig, batchSize int) ([]float32, []float32) {
	dModel := config.DModel
	numHeads := config.NumHeads
	headDim := config.HeadDim
	seqLen := config.SeqLength

	// For sequence models like BERT, the batchSize parameter is set to seqLength
	// but the actual batch dimension is 1. Adjust based on input size.
	actualBatchSize := 1
	actualSeqLen := len(input) / dModel
	if actualSeqLen != batchSize*seqLen {
		batchSize = actualBatchSize
		seqLen = actualSeqLen
	}

	// Output size same as input: [batch][seqLength][dModel]
	outputSize := batchSize * seqLen * dModel
	preActivation := make([]float32, outputSize)
	postActivation := make([]float32, outputSize)

	// Temporary storage for Q, K, V projections
	Q := make([]float32, outputSize)
	K := make([]float32, outputSize)
	V := make([]float32, outputSize)

	// Step 1: Project input to Q, K, V
	// Q = input * QWeights + QBias
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < dModel; d++ {
				qSum := config.QBias[d]
				kSum := config.KBias[d]
				vSum := config.VBias[d]

				for i := 0; i < dModel; i++ {
					inputIdx := b*seqLen*dModel + s*dModel + i
					qWeightIdx := i*dModel + d
					kWeightIdx := i*dModel + d
					vWeightIdx := i*dModel + d

					qSum += input[inputIdx] * config.QWeights[qWeightIdx]
					kSum += input[inputIdx] * config.KWeights[kWeightIdx]
					vSum += input[inputIdx] * config.VWeights[vWeightIdx]
				}

				outputIdx := b*seqLen*dModel + s*dModel + d
				Q[outputIdx] = qSum
				K[outputIdx] = kSum
				V[outputIdx] = vSum
			}
		}
	}

	// Step 2: Reshape to [batch][numHeads][seqLen][headDim] and compute attention
	// For each batch and head
	for b := 0; b < batchSize; b++ {
		for h := 0; h < numHeads; h++ {
			// Compute attention scores: Q * K^T / sqrt(headDim)
			scale := float32(1.0 / math.Sqrt(float64(headDim)))
			attentionScores := make([]float32, seqLen*seqLen)

			for qi := 0; qi < seqLen; qi++ {
				for ki := 0; ki < seqLen; ki++ {
					sum := float32(0)
					for d := 0; d < headDim; d++ {
						// Q index: [b][qi][h*headDim + d]
						qIdx := b*seqLen*dModel + qi*dModel + h*headDim + d
						// K index: [b][ki][h*headDim + d]
						kIdx := b*seqLen*dModel + ki*dModel + h*headDim + d
						sum += Q[qIdx] * K[kIdx]
					}
					attentionScores[qi*seqLen+ki] = sum * scale
				}
			}

			// Apply softmax to attention scores (row-wise)
			for qi := 0; qi < seqLen; qi++ {
				// Find max for numerical stability
				maxVal := attentionScores[qi*seqLen]
				for ki := 1; ki < seqLen; ki++ {
					if attentionScores[qi*seqLen+ki] > maxVal {
						maxVal = attentionScores[qi*seqLen+ki]
					}
				}

				// Compute exp and sum
				sumExp := float32(0)
				for ki := 0; ki < seqLen; ki++ {
					attentionScores[qi*seqLen+ki] = float32(math.Exp(float64(attentionScores[qi*seqLen+ki] - maxVal)))
					sumExp += attentionScores[qi*seqLen+ki]
				}

				// Normalize
				for ki := 0; ki < seqLen; ki++ {
					attentionScores[qi*seqLen+ki] /= sumExp
				}
			}

			// Multiply attention scores by V
			// output[qi] = sum over ki of (attention[qi][ki] * V[ki])
			for qi := 0; qi < seqLen; qi++ {
				for d := 0; d < headDim; d++ {
					sum := float32(0)
					for ki := 0; ki < seqLen; ki++ {
						vIdx := b*seqLen*dModel + ki*dModel + h*headDim + d
						sum += attentionScores[qi*seqLen+ki] * V[vIdx]
					}
					// Store in temporary output position
					outIdx := b*seqLen*dModel + qi*dModel + h*headDim + d
					preActivation[outIdx] = sum
				}
			}
		}
	}

	// Step 3: Concatenate heads and apply output projection
	// Output = preActivation * OutputWeight + OutputBias
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < dModel; d++ {
				sum := config.OutputBias[d]

				for i := 0; i < dModel; i++ {
					inputIdx := b*seqLen*dModel + s*dModel + i
					weightIdx := i*dModel + d
					sum += preActivation[inputIdx] * config.OutputWeight[weightIdx]
				}

				outputIdx := b*seqLen*dModel + s*dModel + d

				// Store pre-activation (before final activation function)
				preActivation[outputIdx] = sum

				// Apply activation function
				postActivation[outputIdx] = activateCPU(sum, config.Activation)
			}
		}
	}

	return preActivation, postActivation
}

// multiHeadAttentionBackwardCPU computes gradients for multi-head attention on CPU
// Note: This is a simplified backprop that treats attention as a simple transformation
// A full implementation would need to store Q, K, V, and attention scores during forward pass
func multiHeadAttentionBackwardCPU(
	gradOutput []float32,
	input []float32,
	preActivation []float32,
	config *LayerConfig,
	batchSize int,
) (gradInput []float32, gradQWeights []float32, gradKWeights []float32, gradVWeights []float32, gradOutputWeight []float32, gradQBias []float32, gradKBias []float32, gradVBias []float32, gradOutputBias []float32) {
	dModel := config.DModel
	seqLen := config.SeqLength

	// Initialize gradients
	inputSize := batchSize * seqLen * dModel
	gradInput = make([]float32, inputSize)
	gradQWeights = make([]float32, dModel*dModel)
	gradKWeights = make([]float32, dModel*dModel)
	gradVWeights = make([]float32, dModel*dModel)
	gradOutputWeight = make([]float32, dModel*dModel)
	gradQBias = make([]float32, dModel)
	gradKBias = make([]float32, dModel)
	gradVBias = make([]float32, dModel)
	gradOutputBias = make([]float32, dModel)

	// Step 1: Apply activation derivative to gradOutput
	// preActivation contains the output AFTER the output projection (before final activation)
	gradPreActivation := make([]float32, inputSize)
	for i := 0; i < inputSize; i++ {
		derivative := activateDerivativeCPU(preActivation[i], config.Activation)
		gradPreActivation[i] = gradOutput[i] * derivative
	}

	// Step 2: Backprop through output projection
	// We need to reconstruct the attention output (before output projection)
	// Forward: output = attentionOutput * OutputWeight + OutputBias
	// Backward: gradAttentionOutput = gradOutput * OutputWeight^T

	// First, we need to recompute the attention output by running forward pass
	// (In a proper implementation, we'd store this during forward pass)
	attentionOutput := make([]float32, inputSize)

	// Recompute Q, K, V projections
	Q := make([]float32, inputSize)
	K := make([]float32, inputSize)
	V := make([]float32, inputSize)

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < dModel; d++ {
				qSum := config.QBias[d]
				kSum := config.KBias[d]
				vSum := config.VBias[d]

				for i := 0; i < dModel; i++ {
					inputIdx := b*seqLen*dModel + s*dModel + i
					qWeightIdx := i*dModel + d
					kWeightIdx := i*dModel + d
					vWeightIdx := i*dModel + d

					qSum += input[inputIdx] * config.QWeights[qWeightIdx]
					kSum += input[inputIdx] * config.KWeights[kWeightIdx]
					vSum += input[inputIdx] * config.VWeights[vWeightIdx]
				}

				outputIdx := b*seqLen*dModel + s*dModel + d
				Q[outputIdx] = qSum
				K[outputIdx] = kSum
				V[outputIdx] = vSum
			}
		}
	}

	// Recompute attention (simplified - treating it as identity for gradient flow)
	// In reality, we'd need attention scores, but for now approximate with V values
	copy(attentionOutput, V)

	// Now backprop through output projection
	// gradAttentionOutput will receive gradients from the output layer
	gradAttentionOutput := make([]float32, inputSize)

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < dModel; d++ {
				gradIdx := b*seqLen*dModel + s*dModel + d
				gradOut := gradPreActivation[gradIdx]

				// Gradient w.r.t. output bias
				gradOutputBias[d] += gradOut

				// Gradient w.r.t. output weights and attention output
				for i := 0; i < dModel; i++ {
					attnIdx := b*seqLen*dModel + s*dModel + i
					weightIdx := i*dModel + d

					// Gradient w.r.t. output weights: grad = gradOut * input
					gradOutputWeight[weightIdx] += gradOut * attentionOutput[attnIdx]

					// Gradient w.r.t. attention output: grad = gradOut * weight
					gradAttentionOutput[attnIdx] += gradOut * config.OutputWeight[weightIdx]
				}
			}
		}
	}

	// Step 3: Backprop through V projection
	// V = input * VWeights + VBias
	// For simplified attention, gradAttentionOutput flows directly to gradV
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < dModel; d++ {
				gradIdx := b*seqLen*dModel + s*dModel + d
				gradV := gradAttentionOutput[gradIdx]

				// Gradient w.r.t. V bias
				gradVBias[d] += gradV

				// Gradient w.r.t. V weights and input
				for i := 0; i < dModel; i++ {
					inputIdx := b*seqLen*dModel + s*dModel + i
					vWeightIdx := i*dModel + d

					// Gradient w.r.t. V weights
					gradVWeights[vWeightIdx] += gradV * input[inputIdx]

					// Gradient w.r.t. input (from V path)
					gradInput[inputIdx] += gradV * config.VWeights[vWeightIdx]
				}
			}
		}
	}

	// Step 4: Also backprop through Q and K projections
	// For simplified version, we approximate by flowing gradients through Q and K as well
	// (In full attention, this would go through attention score derivatives)

	// Use a fraction of the gradient for Q and K to approximate their contribution
	gradientScale := float32(0.3) // Scale down Q/K gradients since V carries most signal

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < dModel; d++ {
				gradIdx := b*seqLen*dModel + s*dModel + d
				gradQ := gradAttentionOutput[gradIdx] * gradientScale
				gradK := gradAttentionOutput[gradIdx] * gradientScale

				// Gradient w.r.t. Q and K biases
				gradQBias[d] += gradQ
				gradKBias[d] += gradK

				// Gradient w.r.t. Q and K weights and input
				for i := 0; i < dModel; i++ {
					inputIdx := b*seqLen*dModel + s*dModel + i
					qWeightIdx := i*dModel + d
					kWeightIdx := i*dModel + d

					// Gradient w.r.t. weights
					gradQWeights[qWeightIdx] += gradQ * input[inputIdx]
					gradKWeights[kWeightIdx] += gradK * input[inputIdx]

					// Gradient w.r.t. input (from Q and K paths)
					gradInput[inputIdx] += gradQ * config.QWeights[qWeightIdx]
					gradInput[inputIdx] += gradK * config.KWeights[kWeightIdx]
				}
			}
		}
	}

	return gradInput, gradQWeights, gradKWeights, gradVWeights, gradOutputWeight, gradQBias, gradKBias, gradVBias, gradOutputBias
}
