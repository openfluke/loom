package poly

import (
	"math"
)

// LayerNormForwardPolymorphic performs layer normalization for any numeric type.
func LayerNormForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize, normSize := input.Shape[0], layer.OutputHeight
	if normSize == 0 {
		normSize = len(input.Data) / batchSize
	}

	preAct = NewTensor[T](batchSize, 2)
	postAct = NewTensor[T](input.Shape...)

	epsilon := 1e-5

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)
	gamma, beta := wData[0:normSize], wData[normSize:2*normSize]

	for b := 0; b < batchSize; b++ {
		start := b * normSize
		var sum, sumSq float64
		for i := 0; i < normSize; i++ {
			val := float64(input.Data[start+i])
			sum += val
			sumSq += val * val
		}
		mean := sum / float64(normSize)
		variance := (sumSq / float64(normSize)) - (mean * mean)
		std := math.Sqrt(variance + epsilon)

		preAct.Data[b*2] = T(mean)
		preAct.Data[b*2+1] = T(variance)

		for i := 0; i < normSize; i++ {
			normed := (float64(input.Data[start+i]) - mean) / std
			postAct.Data[start+i] = T(normed*float64(gamma[i]) + float64(beta[i]))
		}
	}
	return preAct, postAct
}

// LayerNormBackwardPolymorphic calculates gradients for LayerNorm.
func LayerNormBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize, normSize := input.Shape[0], layer.OutputHeight
	if normSize == 0 {
		normSize = len(input.Data) / batchSize
	}

	gradInput = NewTensor[T](input.Shape...)
	gradWeights = NewTensor[T](len(layer.WeightStore.Master))
	epsilon := 1e-5

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	for b := 0; b < batchSize; b++ {
		start := b * normSize
		mean, variance := float64(preAct.Data[b*2]), float64(preAct.Data[b*2+1])
		std := math.Sqrt(variance + epsilon)
		invStd := 1.0 / std

		var sum_dy, sum_dy_xhat float64
		for i := 0; i < normSize; i++ {
			idx := start + i
			dy := float64(gradOutput.Data[idx])
			xhat := (float64(input.Data[idx]) - mean) * invStd

			gradWeights.Data[i] += T(dy * xhat)
			gradWeights.Data[normSize+i] += T(dy)

			g := float64(wData[i])
			dy_xhat := dy * g
			sum_dy += dy_xhat
			sum_dy_xhat += dy_xhat * xhat
		}

		for i := 0; i < normSize; i++ {
			idx := start + i
			xhat := (float64(input.Data[idx]) - mean) * invStd
			g := float64(wData[i])

			dy_xhat := float64(gradOutput.Data[idx]) * g
			dx := invStd * (dy_xhat - (sum_dy / float64(normSize)) - (xhat * sum_dy_xhat / float64(normSize)))
			gradInput.Data[idx] = T(dx)
		}
	}

	return gradInput, gradWeights
}

// RMSNormForwardPolymorphic performs RMS normalization.
func RMSNormForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize, normSize := input.Shape[0], layer.OutputHeight
	if normSize == 0 {
		normSize = len(input.Data) / batchSize
	}

	preAct = NewTensor[T](batchSize)
	postAct = NewTensor[T](input.Shape...)
	epsilon := 1e-6
	if layer.RMSNormEps > 0 {
		epsilon = layer.RMSNormEps
	}

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	for b := 0; b < batchSize; b++ {
		start := b * normSize
		var sumSq float64
		for i := 0; i < normSize; i++ {
			val := float64(input.Data[start+i])
			sumSq += val * val
		}
		rms := math.Sqrt(sumSq/float64(normSize) + epsilon)
		preAct.Data[b] = T(rms)

		for i := 0; i < normSize; i++ {
			normed := float64(input.Data[start+i]) / rms
			g := float64(wData[i])
			postAct.Data[start+i] = T(normed * g)
		}
	}
	return preAct, postAct
}

// RMSNormBackwardPolymorphic calculates gradients for RMSNorm.
func RMSNormBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize, normSize := input.Shape[0], layer.OutputHeight
	if normSize == 0 {
		normSize = len(input.Data) / batchSize
	}

	gradInput = NewTensor[T](input.Shape...)
	gradWeights = NewTensor[T](len(layer.WeightStore.Master))

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	for b := 0; b < batchSize; b++ {
		start := b * normSize
		rms := float64(preAct.Data[b])
		invRMS := 1.0 / rms
		invRMS3 := 1.0 / (rms * rms * rms)

		var sum_dxhat_x float64
		for i := 0; i < normSize; i++ {
			idx := start + i
			dy := float64(gradOutput.Data[idx])
			x := float64(input.Data[idx])
			xhat := x * invRMS

			gradWeights.Data[i] += T(dy * xhat)
			g := float64(wData[i])
			sum_dxhat_x += dy * g * x
		}

		term2 := sum_dxhat_x * invRMS3 / float64(normSize)
		for i := 0; i < normSize; i++ {
			idx := start + i
			dy := float64(gradOutput.Data[idx])
			x := float64(input.Data[idx])
			g := float64(wData[i])

			dx := (dy * g * invRMS) - (x * term2)
			gradInput.Data[idx] = T(dx)
		}
	}

	return gradInput, gradWeights
}
