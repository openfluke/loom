package nn

import (
	"math"
)

// applyRotaryEmbedding applies RoPE (Rotary Position Embedding) to Q and K
// This is CRITICAL for transformer models to understand token positions
// Supports Grouped Query Attention (GQA) where Q and K have different head counts
func applyRotaryEmbedding(q, k []float32, seqLen, numQHeads, numKVHeads, headDim int, ropeTheta float64) {
	// Compute frequency bands
	freqs := make([]float64, headDim/2)
	for i := 0; i < headDim/2; i++ {
		freqs[i] = 1.0 / math.Pow(ropeTheta, float64(2*i)/float64(headDim))
	}

	qHeadSize := numQHeads * headDim
	kHeadSize := numKVHeads * headDim

	// Apply rotation to each position
	for pos := 0; pos < seqLen; pos++ {
		// Apply to Q (all query heads)
		for head := 0; head < numQHeads; head++ {
			qOffset := pos*qHeadSize + head*headDim
			applyRotation(q, qOffset, headDim, freqs, pos)
		}

		// Apply to K (KV heads for GQA)
		for head := 0; head < numKVHeads; head++ {
			kOffset := pos*kHeadSize + head*headDim
			applyRotation(k, kOffset, headDim, freqs, pos)
		}
	}
}

// applyRotation applies the actual rotation to a vector
func applyRotation(vec []float32, offset, headDim int, freqs []float64, pos int) {
	for i := 0; i < headDim/2; i++ {
		// Get the pair of values to rotate
		idx1 := offset + 2*i
		idx2 := offset + 2*i + 1

		x := float64(vec[idx1])
		y := float64(vec[idx2])

		// Compute rotation angle
		theta := freqs[i] * float64(pos)
		cosTheta := math.Cos(theta)
		sinTheta := math.Sin(theta)

		// Apply 2D rotation
		vec[idx1] = float32(x*cosTheta - y*sinTheta)
		vec[idx2] = float32(x*sinTheta + y*cosTheta)
	}
}
