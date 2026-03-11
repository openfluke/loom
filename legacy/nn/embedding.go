package nn

import (
	"math"
	"math/rand"
)

// =============================================================================
// Generic Embedding Implementation
// =============================================================================

// EmbeddingForward performs embedding lookup for any numeric type.
// Input: token indices as int values (stored in T, will be cast to int)
// Output: [seqLen, embeddingDim]
func EmbeddingForward[T Numeric](
	tokenIDs *Tensor[T],
	weights *Tensor[T],
	vocabSize, embeddingDim int,
) *Tensor[T] {
	seqLen := len(tokenIDs.Data)
	output := NewTensor[T](seqLen * embeddingDim)

	for i := 0; i < seqLen; i++ {
		// Cast token ID to int
		tokenID := int(tokenIDs.Data[i])
		
		// Bounds check
		if tokenID < 0 || tokenID >= vocabSize {
			continue // Skip invalid tokens, leave as zero
		}

		// Copy embedding vector
		for j := 0; j < embeddingDim; j++ {
			idx := tokenID*embeddingDim + j
			if idx < len(weights.Data) {
				output.Data[i*embeddingDim+j] = weights.Data[idx]
			}
		}
	}

	return output
}

// EmbeddingBackward computes gradients for embedding lookup.
// Only the embeddings for tokens that were looked up get gradients.
func EmbeddingBackward[T Numeric](
	gradOutput, tokenIDs *Tensor[T],
	vocabSize, embeddingDim int,
) *Tensor[T] {
	gradWeights := NewTensor[T](vocabSize * embeddingDim)
	seqLen := len(tokenIDs.Data)

	for i := 0; i < seqLen; i++ {
		tokenID := int(tokenIDs.Data[i])
		
		if tokenID < 0 || tokenID >= vocabSize {
			continue
		}

		// Accumulate gradient for this embedding
		for j := 0; j < embeddingDim; j++ {
			gradIdx := tokenID*embeddingDim + j
			outIdx := i*embeddingDim + j
			if gradIdx < len(gradWeights.Data) && outIdx < len(gradOutput.Data) {
				gradWeights.Data[gradIdx] += gradOutput.Data[outIdx]
			}
		}
	}

	return gradWeights
}

// =============================================================================
// Backward-compatible float32 functions
// =============================================================================

// InitEmbeddingLayer initializes an Embedding layer with random weights
func InitEmbeddingLayer(vocabSize, embeddingDim int) LayerConfig {
	// Initialize embedding weights (uniform distribution)
	weights := make([]float32, vocabSize*embeddingDim)
	scale := float32(1.0 / math.Sqrt(float64(embeddingDim)))
	
	for i := range weights {
		weights[i] = (float32(rand.Float64())*2 - 1) * scale
	}

	return LayerConfig{
		Type:             LayerEmbedding,
		VocabSize:        vocabSize,
		EmbeddingDim:     embeddingDim,
		EmbeddingWeights: weights,
	}
}

// embeddingForwardCPU performs embedding lookup on CPU
func embeddingForwardCPU(tokenIDs []float32, config *LayerConfig) []float32 {
	tokenT := NewTensorFromSlice(tokenIDs, len(tokenIDs))
	weightsT := NewTensorFromSlice(config.EmbeddingWeights, len(config.EmbeddingWeights))

	output := EmbeddingForward(tokenT, weightsT, config.VocabSize, config.EmbeddingDim)
	return output.Data
}

// embeddingBackwardCPU computes gradients for embedding on CPU
func embeddingBackwardCPU(
	gradOutput, tokenIDs []float32,
	config *LayerConfig,
) []float32 {
	gradOutputT := NewTensorFromSlice(gradOutput, len(gradOutput))
	tokenT := NewTensorFromSlice(tokenIDs, len(tokenIDs))

	gradWeights := EmbeddingBackward(gradOutputT, tokenT, config.VocabSize, config.EmbeddingDim)
	return gradWeights.Data
}
