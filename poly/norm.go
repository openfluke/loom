package poly

import (
	"math"
)

// LayerNormForwardPolymorphic performs layer normalization for any numeric type.
func LayerNormForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize, normSize := input.Shape[0], layer.OutputHeight // normSize is hidden_size
	if normSize == 0 { normSize = len(input.Data) / batchSize }

	if layer.UseTiling && layer.TileSize > 0 {
		return LayerNormForwardTiled(layer, input)
	}
	
	preAct = NewTensor[T](batchSize, 2) // [mean, variance] per sample
	postAct = NewTensor[T](input.Shape...)

	epsilon := 1e-5
	
	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }
	
	// Fast-paths for Float32/Float64
	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			gamma, beta := rawW[0:normSize], rawW[normSize:2*normSize]
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
				std := float64(math.Sqrt(float64(variance) + epsilon))
				
				preAct.Data[b*2] = T(mean)
				preAct.Data[b*2+1] = T(variance)
				
				for i := 0; i < normSize; i++ {
					normed := (float64(input.Data[start+i]) - mean) / std
					postAct.Data[start+i] = T(normed*float64(gamma[i]) + float64(beta[i]))
				}
			}
			return preAct, postAct
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			gamma, beta := rawW[0:normSize], rawW[normSize:2*normSize]
			for b := 0; b < batchSize; b++ {
				start := b * normSize
				var sum, sumSq float32
				for i := 0; i < normSize; i++ {
					val := float32(input.Data[start+i])
					sum += val
					sumSq += val * val
				}
				mean := sum / float32(normSize)
				variance := (sumSq / float32(normSize)) - (mean * mean)
				std := float32(math.Sqrt(float64(variance) + epsilon))
				
				preAct.Data[b*2] = T(mean)
				preAct.Data[b*2+1] = T(variance)
				
				for i := 0; i < normSize; i++ {
					normed := (float32(input.Data[start+i]) - mean) / std
					postAct.Data[start+i] = T(normed*float32(gamma[i]) + float32(beta[i]))
				}
			}
			return preAct, postAct
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			gamma, beta := rawW[0:normSize], rawW[normSize:2*normSize]
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
				std := float64(math.Sqrt(float64(variance) + epsilon))
				
				preAct.Data[b*2] = T(mean)
				preAct.Data[b*2+1] = T(variance)
				
				for i := 0; i < normSize; i++ {
					normed := (float64(input.Data[start+i]) - mean) / std
					postAct.Data[start+i] = T(normed*float64(gamma[i]) + float64(beta[i]))
				}
			}
			return preAct, postAct
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			gamma, beta := rawW[0:normSize], rawW[normSize:2*normSize]
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
				std := float64(math.Sqrt(float64(variance) + epsilon))
				
				preAct.Data[b*2] = T(mean)
				preAct.Data[b*2+1] = T(variance)
				
				for i := 0; i < normSize; i++ {
					normed := (float64(input.Data[start+i]) - mean) / std
					postAct.Data[start+i] = T(normed*float64(gamma[i]) + float64(beta[i]))
				}
			}
			return preAct, postAct
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			gamma, beta := rawW[0:normSize], rawW[normSize:2*normSize]
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
				std := float64(math.Sqrt(float64(variance) + epsilon))
				
				preAct.Data[b*2] = T(mean)
				preAct.Data[b*2+1] = T(variance)
				
				for i := 0; i < normSize; i++ {
					normed := (float64(input.Data[start+i]) - mean) / std
					postAct.Data[start+i] = T(normed*float64(gamma[i]) + float64(beta[i]))
				}
			}
			return preAct, postAct
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			gamma, beta := rawW[0:normSize], rawW[normSize:2*normSize]
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
				std := float64(math.Sqrt(float64(variance) + epsilon))
				
				preAct.Data[b*2] = T(mean)
				preAct.Data[b*2+1] = T(variance)
				
				for i := 0; i < normSize; i++ {
					normed := (float64(input.Data[start+i]) - mean) / std
					postAct.Data[start+i] = T(normed*float64(gamma[i]) + float64(beta[i]))
				}
			}
			return preAct, postAct
		}
	}

	// Universal fallback
	scale := layer.WeightStore.Scale
	if scale == 0 { scale = 1.0 }
	wData := CastWeights[float32](weights)
	gamma, beta := wData[0:normSize], wData[normSize:2*normSize]
	
	for b := 0; b < batchSize; b++ {
		start := b * normSize
		var sum, sumSq float32
		for i := 0; i < normSize; i++ {
			val := float32(input.Data[start+i])
			sum += val
			sumSq += val * val
		}
		mean := sum / float32(normSize)
		variance := (sumSq / float32(normSize)) - (mean * mean)
		std := float32(math.Sqrt(float64(variance) + epsilon))
		
		preAct.Data[b*2] = T(mean)
		preAct.Data[b*2+1] = T(variance)
		
		for i := 0; i < normSize; i++ {
			normed := (float32(input.Data[start+i]) - mean) / std
			g := SimulatePrecision(gamma[i], layer.DType, scale)
			bt := SimulatePrecision(beta[i], layer.DType, scale)
			postAct.Data[start+i] = T(normed*g + bt)
		}
	}

	return preAct, postAct
}

// LayerNormBackwardPolymorphic calculates gradients for LayerNorm.
func LayerNormBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize, normSize := input.Shape[0], layer.OutputHeight
	if normSize == 0 { normSize = len(input.Data) / batchSize }
	
	if layer.UseTiling && layer.TileSize > 0 {
		return LayerNormBackwardTiled(layer, gradOutput, input, preAct)
	}
	
	gradInput = NewTensor[T](input.Shape...)
	gradWeights = NewTensor[T](len(layer.WeightStore.Master))
	
	epsilon := 1e-5
	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }
	
	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			for b := 0; b < batchSize; b++ {
				start := b * normSize
				mean, variance := float64(preAct.Data[b*2]), float64(preAct.Data[b*2+1])
				std := float64(math.Sqrt(float64(variance) + epsilon))
				invStd := 1.0 / std
				
				var sum_dy, sum_dy_xhat float64
				for i := 0; i < normSize; i++ {
					idx := start + i
					dy := float64(gradOutput.Data[idx])
					xhat := (float64(input.Data[idx]) - mean) * invStd
					
					// Weight grads
					gradWeights.Data[i] += T(dy * xhat) // gamma grad
					gradWeights.Data[normSize+i] += T(dy) // beta grad
					
					g := float64(rawW[i])
					
					dy_xhat := dy * g
					sum_dy += dy_xhat
					sum_dy_xhat += dy_xhat * xhat
				}
				
				for i := 0; i < normSize; i++ {
					idx := start + i
					xhat := (float64(input.Data[idx]) - mean) * invStd
					g := float64(rawW[i])
					
					dy_xhat := float64(gradOutput.Data[idx]) * g
					dx := invStd * (dy_xhat - (sum_dy/float64(normSize)) - (xhat * sum_dy_xhat / float64(normSize)))
					gradInput.Data[idx] = T(dx)
				}
			}
			return gradInput, gradWeights
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			for b := 0; b < batchSize; b++ {
				start := b * normSize
				mean, variance := float32(preAct.Data[b*2]), float32(preAct.Data[b*2+1])
				std := float32(math.Sqrt(float64(variance) + epsilon))
				invStd := 1.0 / std
				
				var sum_dy, sum_dy_xhat float32
				for i := 0; i < normSize; i++ {
					idx := start + i
					dy := float32(gradOutput.Data[idx])
					xhat := (float32(input.Data[idx]) - mean) * invStd
					
					// Weight grads
					gradWeights.Data[i] += T(dy * xhat) // gamma grad
					gradWeights.Data[normSize+i] += T(dy) // beta grad
					
					g := float32(rawW[i])
					
					dy_xhat := dy * g
					sum_dy += dy_xhat
					sum_dy_xhat += dy_xhat * xhat
				}
				
				for i := 0; i < normSize; i++ {
					idx := start + i
					xhat := (float32(input.Data[idx]) - mean) * invStd
					g := float32(rawW[i])
					
					dy_xhat := float32(gradOutput.Data[idx]) * g
					dx := invStd * (dy_xhat - (sum_dy/float32(normSize)) - (xhat * sum_dy_xhat / float32(normSize)))
					gradInput.Data[idx] = T(dx)
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			for b := 0; b < batchSize; b++ {
				start := b * normSize
				mean, variance := float64(preAct.Data[b*2]), float64(preAct.Data[b*2+1])
				std := float64(math.Sqrt(float64(variance) + epsilon))
				invStd := 1.0 / std
				
				var sum_dy, sum_dy_xhat float64
				for i := 0; i < normSize; i++ {
					idx := start + i
					dy := float64(gradOutput.Data[idx])
					xhat := (float64(input.Data[idx]) - mean) * invStd
					
					// Weight grads
					gradWeights.Data[i] += T(dy * xhat) // gamma grad
					gradWeights.Data[normSize+i] += T(dy) // beta grad
					
					g := float64(rawW[i])
					
					dy_xhat := dy * g
					sum_dy += dy_xhat
					sum_dy_xhat += dy_xhat * xhat
				}
				
				for i := 0; i < normSize; i++ {
					idx := start + i
					xhat := (float64(input.Data[idx]) - mean) * invStd
					g := float64(rawW[i])
					
					dy_xhat := float64(gradOutput.Data[idx]) * g
					dx := invStd * (dy_xhat - (sum_dy/float64(normSize)) - (xhat * sum_dy_xhat / float64(normSize)))
					gradInput.Data[idx] = T(dx)
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			for b := 0; b < batchSize; b++ {
				start := b * normSize
				mean, variance := float64(preAct.Data[b*2]), float64(preAct.Data[b*2+1])
				std := float64(math.Sqrt(float64(variance) + epsilon))
				invStd := 1.0 / std
				
				var sum_dy, sum_dy_xhat float64
				for i := 0; i < normSize; i++ {
					idx := start + i
					dy := float64(gradOutput.Data[idx])
					xhat := (float64(input.Data[idx]) - mean) * invStd
					
					// Weight grads
					gradWeights.Data[i] += T(dy * xhat) // gamma grad
					gradWeights.Data[normSize+i] += T(dy) // beta grad
					
					g := float64(rawW[i])
					
					dy_xhat := dy * g
					sum_dy += dy_xhat
					sum_dy_xhat += dy_xhat * xhat
				}
				
				for i := 0; i < normSize; i++ {
					idx := start + i
					xhat := (float64(input.Data[idx]) - mean) * invStd
					g := float64(rawW[i])
					
					dy_xhat := float64(gradOutput.Data[idx]) * g
					dx := invStd * (dy_xhat - (sum_dy/float64(normSize)) - (xhat * sum_dy_xhat / float64(normSize)))
					gradInput.Data[idx] = T(dx)
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			for b := 0; b < batchSize; b++ {
				start := b * normSize
				mean, variance := float64(preAct.Data[b*2]), float64(preAct.Data[b*2+1])
				std := float64(math.Sqrt(float64(variance) + epsilon))
				invStd := 1.0 / std
				
				var sum_dy, sum_dy_xhat float64
				for i := 0; i < normSize; i++ {
					idx := start + i
					dy := float64(gradOutput.Data[idx])
					xhat := (float64(input.Data[idx]) - mean) * invStd
					
					// Weight grads
					gradWeights.Data[i] += T(dy * xhat) // gamma grad
					gradWeights.Data[normSize+i] += T(dy) // beta grad
					
					g := float64(rawW[i])
					
					dy_xhat := dy * g
					sum_dy += dy_xhat
					sum_dy_xhat += dy_xhat * xhat
				}
				
				for i := 0; i < normSize; i++ {
					idx := start + i
					xhat := (float64(input.Data[idx]) - mean) * invStd
					g := float64(rawW[i])
					
					dy_xhat := float64(gradOutput.Data[idx]) * g
					dx := invStd * (dy_xhat - (sum_dy/float64(normSize)) - (xhat * sum_dy_xhat / float64(normSize)))
					gradInput.Data[idx] = T(dx)
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			for b := 0; b < batchSize; b++ {
				start := b * normSize
				mean, variance := float64(preAct.Data[b*2]), float64(preAct.Data[b*2+1])
				std := float64(math.Sqrt(float64(variance) + epsilon))
				invStd := 1.0 / std
				
				var sum_dy, sum_dy_xhat float64
				for i := 0; i < normSize; i++ {
					idx := start + i
					dy := float64(gradOutput.Data[idx])
					xhat := (float64(input.Data[idx]) - mean) * invStd
					
					// Weight grads
					gradWeights.Data[i] += T(dy * xhat) // gamma grad
					gradWeights.Data[normSize+i] += T(dy) // beta grad
					
					g := float64(rawW[i])
					
					dy_xhat := dy * g
					sum_dy += dy_xhat
					sum_dy_xhat += dy_xhat * xhat
				}
				
				for i := 0; i < normSize; i++ {
					idx := start + i
					xhat := (float64(input.Data[idx]) - mean) * invStd
					g := float64(rawW[i])
					
					dy_xhat := float64(gradOutput.Data[idx]) * g
					dx := invStd * (dy_xhat - (sum_dy/float64(normSize)) - (xhat * sum_dy_xhat / float64(normSize)))
					gradInput.Data[idx] = T(dx)
				}
			}
			return gradInput, gradWeights
		}
	}

	// Standard grad logic
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
			
			// Weight grads
			gradWeights.Data[i] += T(dy * xhat) // gamma grad
			gradWeights.Data[normSize+i] += T(dy) // beta grad
			
			wData := CastWeights[float32](weights)
			g := float64(SimulatePrecision(wData[i], layer.DType, layer.WeightStore.Scale))
			
			dy_xhat := dy * g
			sum_dy += dy_xhat
			sum_dy_xhat += dy_xhat * xhat
		}
		
		for i := 0; i < normSize; i++ {
			idx := start + i
			xhat := (float64(input.Data[idx]) - mean) * invStd
			
			wData := CastWeights[float32](weights)
			g := float64(SimulatePrecision(wData[i], layer.DType, layer.WeightStore.Scale))
			
			dy_xhat := float64(gradOutput.Data[idx]) * g
			dx := invStd * (dy_xhat - (sum_dy/float64(normSize)) - (xhat * sum_dy_xhat / float64(normSize)))
			gradInput.Data[idx] = T(dx)
		}
	}
	
	return gradInput, gradWeights
}

// RMSNormForwardPolymorphic performs RMS normalization.
func RMSNormForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize, normSize := input.Shape[0], layer.OutputHeight
	if normSize == 0 { normSize = len(input.Data) / batchSize }

	if layer.UseTiling && layer.TileSize > 0 {
		return RMSNormForwardTiled(layer, input)
	}
	
	preAct = NewTensor[T](batchSize) // [rms] per sample
	postAct = NewTensor[T](input.Shape...)
	epsilon := 1e-6
	
	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }
	
	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			for b := 0; b < batchSize; b++ {
				start := b * normSize
				var sumSq float64
				for i := 0; i < normSize; i++ {
					val := float64(input.Data[start+i])
					sumSq += val * val
				}
				rms := float64(math.Sqrt(float64(sumSq)/float64(normSize) + epsilon))
				preAct.Data[b] = T(rms)
				
				for i := 0; i < normSize; i++ {
					normed := float64(input.Data[start+i]) / rms
					g := float64(rawW[i])
					postAct.Data[start+i] = T(normed * g)
				}
			}
			return preAct, postAct
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			for b := 0; b < batchSize; b++ {
				start := b * normSize
				var sumSq float32
				for i := 0; i < normSize; i++ {
					val := float32(input.Data[start+i])
					sumSq += val * val
				}
				rms := float32(math.Sqrt(float64(sumSq)/float64(normSize) + epsilon))
				preAct.Data[b] = T(rms)
				
				for i := 0; i < normSize; i++ {
					normed := float32(input.Data[start+i]) / rms
					g := float32(rawW[i])
					postAct.Data[start+i] = T(normed * g)
				}
			}
			return preAct, postAct
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			for b := 0; b < batchSize; b++ {
				start := b * normSize
				var sumSq float64
				for i := 0; i < normSize; i++ {
					val := float64(input.Data[start+i])
					sumSq += val * val
				}
				rms := float64(math.Sqrt(float64(sumSq)/float64(normSize) + epsilon))
				preAct.Data[b] = T(rms)
				
				for i := 0; i < normSize; i++ {
					normed := float64(input.Data[start+i]) / rms
					g := float64(rawW[i])
					postAct.Data[start+i] = T(normed * g)
				}
			}
			return preAct, postAct
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			for b := 0; b < batchSize; b++ {
				start := b * normSize
				var sumSq float64
				for i := 0; i < normSize; i++ {
					val := float64(input.Data[start+i])
					sumSq += val * val
				}
				rms := float64(math.Sqrt(float64(sumSq)/float64(normSize) + epsilon))
				preAct.Data[b] = T(rms)
				
				for i := 0; i < normSize; i++ {
					normed := float64(input.Data[start+i]) / rms
					g := float64(rawW[i])
					postAct.Data[start+i] = T(normed * g)
				}
			}
			return preAct, postAct
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			for b := 0; b < batchSize; b++ {
				start := b * normSize
				var sumSq float64
				for i := 0; i < normSize; i++ {
					val := float64(input.Data[start+i])
					sumSq += val * val
				}
				rms := float64(math.Sqrt(float64(sumSq)/float64(normSize) + epsilon))
				preAct.Data[b] = T(rms)
				
				for i := 0; i < normSize; i++ {
					normed := float64(input.Data[start+i]) / rms
					g := float64(rawW[i])
					postAct.Data[start+i] = T(normed * g)
				}
			}
			return preAct, postAct
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			for b := 0; b < batchSize; b++ {
				start := b * normSize
				var sumSq float64
				for i := 0; i < normSize; i++ {
					val := float64(input.Data[start+i])
					sumSq += val * val
				}
				rms := float64(math.Sqrt(float64(sumSq)/float64(normSize) + epsilon))
				preAct.Data[b] = T(rms)
				
				for i := 0; i < normSize; i++ {
					normed := float64(input.Data[start+i]) / rms
					g := float64(rawW[i])
					postAct.Data[start+i] = T(normed * g)
				}
			}
			return preAct, postAct
		}
	}
	
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
			wData := CastWeights[float32](weights)
			g := float64(SimulatePrecision(wData[i], layer.DType, layer.WeightStore.Scale))
			postAct.Data[start+i] = T(normed * g)
		}
	}
	return preAct, postAct
}

// RMSNormBackwardPolymorphic calculates gradients for RMSNorm.
func RMSNormBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize, normSize := input.Shape[0], layer.OutputHeight
	if normSize == 0 { normSize = len(input.Data) / batchSize }
	
	if layer.UseTiling && layer.TileSize > 0 {
		return RMSNormBackwardTiled(layer, gradOutput, input, preAct)
	}
	
	gradInput = NewTensor[T](input.Shape...)
	gradWeights = NewTensor[T](len(layer.WeightStore.Master))
	
	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }
	
	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
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
					
					g := float64(rawW[i])
					sum_dxhat_x += dy * g * x
				}
				
				term2 := sum_dxhat_x * invRMS3 / float64(normSize)
				for i := 0; i < normSize; i++ {
					idx := start + i
					dy := float64(gradOutput.Data[idx])
					x := float64(input.Data[idx])
					g := float64(rawW[i])
					
					dx := (dy * g * invRMS) - (x * term2)
					gradInput.Data[idx] = T(dx)
				}
			}
			return gradInput, gradWeights
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			for b := 0; b < batchSize; b++ {
				start := b * normSize
				rms := float32(preAct.Data[b])
				invRMS := 1.0 / rms
				invRMS3 := 1.0 / (rms * rms * rms)
				
				var sum_dxhat_x float32
				for i := 0; i < normSize; i++ {
					idx := start + i
					dy := float32(gradOutput.Data[idx])
					x := float32(input.Data[idx])
					xhat := x * invRMS
					
					gradWeights.Data[i] += T(dy * xhat)
					
					g := float32(rawW[i])
					sum_dxhat_x += dy * g * x
				}
				
				term2 := sum_dxhat_x * invRMS3 / float32(normSize)
				for i := 0; i < normSize; i++ {
					idx := start + i
					dy := float32(gradOutput.Data[idx])
					x := float32(input.Data[idx])
					g := float32(rawW[i])
					
					dx := (dy * g * invRMS) - (x * term2)
					gradInput.Data[idx] = T(dx)
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
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
					
					g := float64(rawW[i])
					sum_dxhat_x += dy * g * x
				}
				
				term2 := sum_dxhat_x * invRMS3 / float64(normSize)
				for i := 0; i < normSize; i++ {
					idx := start + i
					dy := float64(gradOutput.Data[idx])
					x := float64(input.Data[idx])
					g := float64(rawW[i])
					
					dx := (dy * g * invRMS) - (x * term2)
					gradInput.Data[idx] = T(dx)
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
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
					
					g := float64(rawW[i])
					sum_dxhat_x += dy * g * x
				}
				
				term2 := sum_dxhat_x * invRMS3 / float64(normSize)
				for i := 0; i < normSize; i++ {
					idx := start + i
					dy := float64(gradOutput.Data[idx])
					x := float64(input.Data[idx])
					g := float64(rawW[i])
					
					dx := (dy * g * invRMS) - (x * term2)
					gradInput.Data[idx] = T(dx)
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
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
					
					g := float64(rawW[i])
					sum_dxhat_x += dy * g * x
				}
				
				term2 := sum_dxhat_x * invRMS3 / float64(normSize)
				for i := 0; i < normSize; i++ {
					idx := start + i
					dy := float64(gradOutput.Data[idx])
					x := float64(input.Data[idx])
					g := float64(rawW[i])
					
					dx := (dy * g * invRMS) - (x * term2)
					gradInput.Data[idx] = T(dx)
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
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
					
					g := float64(rawW[i])
					sum_dxhat_x += dy * g * x
				}
				
				term2 := sum_dxhat_x * invRMS3 / float64(normSize)
				for i := 0; i < normSize; i++ {
					idx := start + i
					dy := float64(gradOutput.Data[idx])
					x := float64(input.Data[idx])
					g := float64(rawW[i])
					
					dx := (dy * g * invRMS) - (x * term2)
					gradInput.Data[idx] = T(dx)
				}
			}
			return gradInput, gradWeights
		}
	}

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
			
			wData := CastWeights[float32](weights)
			g := float64(SimulatePrecision(wData[i], layer.DType, layer.WeightStore.Scale))
			sum_dxhat_x += dy * g * x
		}
		
		term2 := sum_dxhat_x * invRMS3 / float64(normSize)
		for i := 0; i < normSize; i++ {
			idx := start + i
			dy := float64(gradOutput.Data[idx])
			x := float64(input.Data[idx])
			
			wData := CastWeights[float32](weights)
			g := float64(SimulatePrecision(wData[i], layer.DType, layer.WeightStore.Scale))
			
			dx := (dy * g * invRMS) - (x * term2)
			gradInput.Data[idx] = T(dx)
		}
	}
	return gradInput, gradWeights
}

// LayerNormForwardTiled performs a tiled forward pass for LayerNorm.
func LayerNormForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize, normSize := input.Shape[0], layer.OutputHeight
	if normSize == 0 { normSize = len(input.Data) / batchSize }
	tileSize := layer.TileSize
	if tileSize <= 0 { tileSize = 256 }

	preAct = NewTensor[T](batchSize, 2)
	postAct = NewTensor[T](input.Shape...)
	epsilon := 1e-5

	weights := layer.WeightStore.GetActive(layer.DType)
	wScale := layer.WeightStore.Scale
	if wScale == 0 { wScale = 1.0 }

	for b := 0; b < batchSize; b++ {
		start := b * normSize
		var globalSum, globalSumSq float64

		// 1. Tiled Sum/SumSq calculation
		for t := 0; t < normSize; t += tileSize {
			end := t + tileSize
			if end > normSize { end = normSize }
			for i := t; i < end; i++ {
				val := float64(input.Data[start+i])
				globalSum += val
				globalSumSq += val * val
			}
		}

		mean := globalSum / float64(normSize)
		variance := (globalSumSq / float64(normSize)) - (mean * mean)
		std := math.Sqrt(variance + epsilon)
		invStd := 1.0 / std

		preAct.Data[b*2] = T(mean)
		preAct.Data[b*2+1] = T(variance)

		// 2. Tiled Normalization and weighting
		for t := 0; t < normSize; t += tileSize {
			end := t + tileSize
			if end > normSize { end = normSize }
			for i := t; i < end; i++ {
				normed := (float64(input.Data[start+i]) - mean) * invStd
				g, beta := float64(1.0), float64(0.0)
				if weights != nil {
					switch w := weights.(type) {
					case []float32:
						g = float64(w[i])
						beta = float64(w[normSize+i])
					case []float64:
						g = w[i]
						beta = w[normSize+i]
					case []int8:
						g = float64(w[i]) * float64(wScale)
						beta = float64(w[normSize+i]) * float64(wScale)
					}
				}
				postAct.Data[start+i] = T(normed*g + beta)
			}
		}
	}
	return preAct, postAct
}

// LayerNormBackwardTiled performs a tiled backward pass for LayerNorm.
func LayerNormBackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize, normSize := input.Shape[0], layer.OutputHeight
	if normSize == 0 { normSize = len(input.Data) / batchSize }
	tileSize := layer.TileSize
	if tileSize <= 0 { tileSize = 256 }

	gradInput = NewTensor[T](input.Shape...)
	gradWeights = NewTensor[T](len(layer.WeightStore.Master))
	epsilon := 1e-5

	weights := layer.WeightStore.GetActive(layer.DType)
	wScale := layer.WeightStore.Scale
	if wScale == 0 { wScale = 1.0 }

	for b := 0; b < batchSize; b++ {
		start := b * normSize
		mean, variance := float64(preAct.Data[b*2]), float64(preAct.Data[b*2+1])
		std := math.Sqrt(variance + epsilon)
		invStd := 1.0 / std

		var sumDY, sumDYXhat float64

		// 1. First tiled pass for sumDY and sumDYXhat
		for t := 0; t < normSize; t += tileSize {
			end := t + tileSize
			if end > normSize { end = normSize }
			for i := t; i < end; i++ {
				idx := start + i
				dy := float64(gradOutput.Data[idx])
				xhat := (float64(input.Data[idx]) - mean) * invStd

				g := float64(1.0)
				if weights != nil {
					switch w := weights.(type) {
					case []float32: g = float64(w[i])
					case []float64: g = w[i]
					case []int8: g = float64(w[i]) * float64(wScale)
					}
				}

				// Accumulate weight grads (beta is at normSize + i)
				gradWeights.Data[i] += T(dy * xhat)
				gradWeights.Data[normSize+i] += T(dy)

				dyG := dy * g
				sumDY += dyG
				sumDYXhat += dyG * xhat
			}
		}

		// 2. Second tiled pass for gradInput
		for t := 0; t < normSize; t += tileSize {
			end := t + tileSize
			if end > normSize { end = normSize }
			for i := t; i < end; i++ {
				idx := start + i
				xhat := (float64(input.Data[idx]) - mean) * invStd
				
				g := float64(1.0)
				if weights != nil {
					switch w := weights.(type) {
					case []float32: g = float64(w[i])
					case []float64: g = w[i]
					case []int8: g = float64(w[i]) * float64(wScale)
					}
				}
				
				dyG := float64(gradOutput.Data[idx]) * g
				dx := invStd * (dyG - (sumDY/float64(normSize)) - (xhat * sumDYXhat / float64(normSize)))
				gradInput.Data[idx] = T(dx)
			}
		}
	}
	return gradInput, gradWeights
}

// RMSNormForwardTiled performs a tiled forward pass for RMSNorm.
func RMSNormForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize, normSize := input.Shape[0], layer.OutputHeight
	if normSize == 0 { normSize = len(input.Data) / batchSize }
	tileSize := layer.TileSize
	if tileSize <= 0 { tileSize = 256 }

	preAct = NewTensor[T](batchSize)
	postAct = NewTensor[T](input.Shape...)
	epsilon := 1e-6

	weights := layer.WeightStore.GetActive(layer.DType)
	wScale := layer.WeightStore.Scale
	if wScale == 0 { wScale = 1.0 }

	for b := 0; b < batchSize; b++ {
		start := b * normSize
		var globalSumSq float64

		for t := 0; t < normSize; t += tileSize {
			end := t + tileSize
			if end > normSize { end = normSize }
			for i := t; i < end; i++ {
				val := float64(input.Data[start+i])
				globalSumSq += val * val
			}
		}

		rms := math.Sqrt(globalSumSq/float64(normSize) + epsilon)
		invRMS := 1.0 / rms
		preAct.Data[b] = T(rms)

		for t := 0; t < normSize; t += tileSize {
			end := t + tileSize
			if end > normSize { end = normSize }
			for i := t; i < end; i++ {
				normed := float64(input.Data[start+i]) * invRMS
				g := float64(1.0)
				if weights != nil {
					switch w := weights.(type) {
					case []float32: g = float64(w[i])
					case []float64: g = w[i]
					case []int8: g = float64(w[i]) * float64(wScale)
					}
				}
				postAct.Data[start+i] = T(normed * g)
			}
		}
	}
	return preAct, postAct
}

// RMSNormBackwardTiled performs a tiled backward pass for RMSNorm.
func RMSNormBackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize, normSize := input.Shape[0], layer.OutputHeight
	if normSize == 0 { normSize = len(input.Data) / batchSize }
	tileSize := layer.TileSize
	if tileSize <= 0 { tileSize = 256 }

	gradInput = NewTensor[T](input.Shape...)
	gradWeights = NewTensor[T](len(layer.WeightStore.Master))

	weights := layer.WeightStore.GetActive(layer.DType)
	wScale := layer.WeightStore.Scale
	if wScale == 0 { wScale = 1.0 }

	for b := 0; b < batchSize; b++ {
		start := b * normSize
		rms := float64(preAct.Data[b])
		invRMS := 1.0 / rms
		invRMS3 := 1.0 / (rms * rms * rms)

		var sumDXhatX float64

		for t := 0; t < normSize; t += tileSize {
			end := t + tileSize
			if end > normSize { end = normSize }
			for i := t; i < end; i++ {
				idx := start + i
				dy := float64(gradOutput.Data[idx])
				x := float64(input.Data[idx])
				xhat := x * invRMS

				g := float64(1.0)
				if weights != nil {
					switch w := weights.(type) {
					case []float32: g = float64(w[i])
					case []float64: g = w[i]
					case []int8: g = float64(w[i]) * float64(wScale)
					}
				}

				gradWeights.Data[i] += T(dy * xhat)
				sumDXhatX += dy * g * x
			}
		}

		term2 := sumDXhatX * invRMS3 / float64(normSize)
		for t := 0; t < normSize; t += tileSize {
			end := t + tileSize
			if end > normSize { end = normSize }
			for i := t; i < end; i++ {
				idx := start + i
				dy := float64(gradOutput.Data[idx])
				x := float64(input.Data[idx])
				g := float64(1.0)
				if weights != nil {
					switch w := weights.(type) {
					case []float32: g = float64(w[i])
					case []float64: g = w[i]
					case []int8: g = float64(w[i]) * float64(wScale)
					}
				}
				dx := (dy * g * invRMS) - (x * term2)
				gradInput.Data[idx] = T(dx)
			}
		}
	}
	return gradInput, gradWeights
}
