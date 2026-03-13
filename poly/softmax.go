package poly

import (
	"math"
	"math/rand"
)

// SoftmaxForwardPolymorphic performs a differentiable Softmax forward pass with ALL variants.
func SoftmaxForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	temp := layer.Temperature
	if temp == 0 { temp = 1.0 }

	n := len(input.Data)
	preAct = input
	postAct = NewTensor[T](input.Shape...)

	// Determine logic based on variant
	switch layer.SoftmaxType {
	case SoftmaxStandard, SoftmaxTemperature, SoftmaxGumbel, SoftmaxMasked, SoftmaxSparse, SoftmaxEntmax:
		// These act on the whole vector or a single group
		logits := GetLogits[T](input.Data, temp, layer.DType)
		
		var probs []float32
		switch layer.SoftmaxType {
		case SoftmaxStandard, SoftmaxTemperature:
			probs = Softmax(logits)
		case SoftmaxGumbel:
			noisy := make([]float32, len(logits))
			for i, v := range logits {
				u := rand.Float32()
				if u < 1e-10 { u = 1e-10 }
				gumbel := -float32(math.Log(-math.Log(float64(u))))
				noisy[i] = v + gumbel
			}
			probs = Softmax(noisy)
		case SoftmaxMasked:
			masked := make([]float32, len(logits))
			for i := 0; i < len(logits); i++ {
				if i < len(layer.Mask) && !layer.Mask[i] {
					masked[i] = -1e9
				} else {
					masked[i] = logits[i]
				}
			}
			probs = Softmax(masked)
		case SoftmaxSparse:
			probs = SoftmaxSparseHelper(logits)
		case SoftmaxEntmax:
			alpha := layer.EntmaxAlpha
			if alpha == 0 { alpha = 1.5 }
			probs = SoftmaxEntmaxHelper(logits, float32(alpha))
		}
		
		for i := 0; i < n; i++ {
			postAct.Data[i] = T(probs[i])
		}

	case SoftmaxGrid, SoftmaxHierarchical:
		rows := layer.SoftmaxRows
		cols := layer.SoftmaxCols
		if layer.SoftmaxType == SoftmaxHierarchical && len(layer.HierarchyLevels) > 0 {
			// Hierarchical as defined in nn: treat as grid with last level
			cols = layer.HierarchyLevels[len(layer.HierarchyLevels)-1]
			rows = n / cols
		}
		if rows == 0 || cols == 0 {
			rows = 1
			cols = n
		}

		for r := 0; r < rows; r++ {
			start := r * cols
			end := start + cols
			if end > n { end = n }
			
			logits := GetLogits[T](input.Data[start:end], temp, layer.DType)
			probs := Softmax(logits)
			
			for i := start; i < end; i++ {
				postAct.Data[i] = T(probs[i-start])
			}
		}

	default:
		// Fallback to standard
		logits := GetLogits[T](input.Data, temp, layer.DType)
		probs := Softmax(logits)
		for i := 0; i < n; i++ {
			postAct.Data[i] = T(probs[i])
		}
	}

	return preAct, postAct
}

// GetLogits extracts float32 logits from any tensor type with temperature scaling
func GetLogits[T Numeric](data []T, temp float64, dtype DType) []float32 {
	logits := make([]float32, len(data))
	t32 := float32(temp)
	
	// Specialized fast-paths for common types
	switch dtype {
	case DTypeFloat32:
		for i := range data { logits[i] = float32(data[i]) / t32 }
	case DTypeFloat64:
		for i := range data { logits[i] = float32(data[i]) / t32 }
	case DTypeInt8, DTypeUint8, DTypeInt16, DTypeUint16, DTypeInt32, DTypeUint32:
		for i := range data { logits[i] = float32(data[i]) / t32 }
	default:
		// Simulation fallback
		for i := range data { logits[i] = float32(data[i]) / t32 }
	}
	return logits
}

// SoftmaxBackwardPolymorphic computes gradients for ALL Softmax variants.
func SoftmaxBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, postAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	n := len(input.Data)
	gradInput = NewTensor[T](input.Shape...)
	gradWeights = nil

	rows := 1
	cols := n

	if layer.SoftmaxType == SoftmaxGrid || layer.SoftmaxType == SoftmaxHierarchical {
		rows = layer.SoftmaxRows
		cols = layer.SoftmaxCols
		if layer.SoftmaxType == SoftmaxHierarchical && len(layer.HierarchyLevels) > 0 {
			cols = layer.HierarchyLevels[len(layer.HierarchyLevels)-1]
			rows = n / cols
		}
	}
	
	if rows == 0 || cols == 0 {
		rows = 1
		cols = n
	}

	for r := 0; r < rows; r++ {
		start := r * cols
		end := start + cols
		if end > n { end = n }

		probs := make([]float32, end-start)
		grads := make([]float32, end-start)
		
		for i := start; i < end; i++ {
			// In VTD, the probabilities are in the post-activation state.
			// The caller (Backward) usually passes preAct, but for activation-layers,
			// preAct IS the output in some conventions, or we assume it's cached.
			probs[i-start] = float32(postAct.Data[i])
			grads[i-start] = float32(gradOutput.Data[i])
		}

		if layer.SoftmaxType == SoftmaxMasked {
			for i := 0; i < len(grads); i++ {
				if i+start < len(layer.Mask) && !layer.Mask[i+start] {
					grads[i] = 0
				}
			}
		}

		gradLogits := SoftmaxBackward(grads, probs)
		for i := start; i < end; i++ {
			gradInput.Data[i] = T(gradLogits[i-start])
		}
	}

	return gradInput, gradWeights
}

// Softmax is a helper for Softmax math
func Softmax(logits []float32) []float32 {
	if len(logits) == 0 { return nil }
	maxLogit := logits[0]
	for _, v := range logits { if v > maxLogit { maxLogit = v } }
	expValues := make([]float32, len(logits))
	var sumExp float32
	for i, v := range logits {
		expValues[i] = float32(math.Exp(float64(v - maxLogit)))
		sumExp += expValues[i]
	}
	for i := range expValues { expValues[i] /= sumExp }
	return expValues
}

// SoftmaxBackward is a helper for Softmax Jacobian
func SoftmaxBackward(gradOutput, softmaxOutput []float32) []float32 {
	n := len(gradOutput)
	gradLogits := make([]float32, n)
	var dotProd float32
	for i := 0; i < n; i++ { dotProd += gradOutput[i] * softmaxOutput[i] }
	for j := 0; j < n; j++ { gradLogits[j] = softmaxOutput[j] * (gradOutput[j] - dotProd) }
	return gradLogits
}

// SoftmaxSparseHelper implements sparsemax
func SoftmaxSparseHelper(logits []float32) []float32 {
	n := len(logits)
	sorted := make([]float32, n)
	copy(sorted, logits)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if sorted[i] < sorted[j] { sorted[i], sorted[j] = sorted[j], sorted[i] }
		}
	}
	cumSum := float32(0)
	k := 0
	for i := 0; i < n; i++ {
		cumSum += sorted[i]
		if sorted[i] - (cumSum-1.0)/float32(i+1) > 0 { k = i + 1 } else { break }
	}
	tau := float32(0)
	if k > 0 {
		cumSum = 0
		for i := 0; i < k; i++ { cumSum += sorted[i] }
		tau = (cumSum - 1.0) / float32(k)
	}
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		result[i] = float32(math.Max(0, float64(logits[i]-tau)))
	}
	return result
}

// SoftmaxEntmaxHelper implements entmax-1.5 approximation
func SoftmaxEntmaxHelper(logits []float32, alpha float32) []float32 {
	if alpha <= 1.0 { return Softmax(logits) }
	if alpha >= 2.0 { return SoftmaxSparseHelper(logits) }
	weight := alpha - 1.0
	s1 := Softmax(logits)
	s2 := SoftmaxSparseHelper(logits)
	res := make([]float32, len(logits))
	var sum float32
	for i := range res {
		res[i] = (1-weight)*s1[i] + weight*s2[i]
		sum += res[i]
	}
	for i := range res { res[i] /= sum }
	return res
}
