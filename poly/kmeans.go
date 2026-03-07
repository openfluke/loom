package poly

// KMeansForwardPolymorphic performs a differentiable K-Means clustering forward pass.
func KMeansForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	numClusters := layer.NumClusters
	temp := layer.KMeansTemperature
	if temp == 0 { temp = 1.0 }
	
	// Input is treated as a single feature vector for this coordinate
	featureDim := len(input.Data)
	
	// Weights are cluster centers [NumClusters, FeatureDim]
	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }

	logits := make([]float32, numClusters)

	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			for k := 0; k < numClusters; k++ {
				var sqDist float32
				offset := k * featureDim
				for d := 0; d < featureDim; d++ {
					diff := float32(input.Data[d]) - float32(rawW[offset+d])
					sqDist += diff * diff
				}
				logits[k] = -sqDist / (2 * float32(temp * temp))
			}
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			for k := 0; k < numClusters; k++ {
				var sqDist float32
				offset := k * featureDim
				for d := 0; d < featureDim; d++ {
					diff := float32(input.Data[d]) - float32(rawW[offset+d])
					sqDist += diff * diff
				}
				logits[k] = -sqDist / (2 * float32(temp * temp))
			}
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			for k := 0; k < numClusters; k++ {
				var sqDist float32
				offset := k * featureDim
				for d := 0; d < featureDim; d++ {
					diff := float32(input.Data[d]) - float32(rawW[offset+d])
					sqDist += diff * diff
				}
				logits[k] = -sqDist / (2 * float32(temp * temp))
			}
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			for k := 0; k < numClusters; k++ {
				var sqDist float32
				offset := k * featureDim
				for d := 0; d < featureDim; d++ {
					diff := float32(input.Data[d]) - float32(rawW[offset+d])
					sqDist += diff * diff
				}
				logits[k] = -sqDist / (2 * float32(temp * temp))
			}
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			for k := 0; k < numClusters; k++ {
				var sqDist float32
				offset := k * featureDim
				for d := 0; d < featureDim; d++ {
					diff := float32(input.Data[d]) - float32(rawW[offset+d])
					sqDist += diff * diff
				}
				logits[k] = -sqDist / (2 * float32(temp * temp))
			}
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			for k := 0; k < numClusters; k++ {
				var sqDist float32
				offset := k * featureDim
				for d := 0; d < featureDim; d++ {
					diff := float32(input.Data[d]) - float32(rawW[offset+d])
					sqDist += diff * diff
				}
				logits[k] = -sqDist / (2 * float32(temp * temp))
			}
		}
	default:
		// Universal fallback
		scaleW := layer.WeightStore.Scale
		if scaleW == 0 { scaleW = 1.0 }
		wData := CastWeights[float32](weights)
		for k := 0; k < numClusters; k++ {
			var sqDist float32
			offset := k * featureDim
			for d := 0; d < featureDim; d++ {
				wVal := SimulatePrecision(wData[offset+d], layer.DType, scaleW)
				diff := float32(input.Data[d]) - wVal
				sqDist += diff * diff
			}
			logits[k] = -sqDist / (2 * float32(temp * temp))
		}
	}

	// Softmax to get assignments
	assignments := Softmax(logits)
	
	if layer.KMeansOutputMode == "features" {
		// Output mode: weighted sum of cluster centers (reconstruction)
		preAct = NewTensor[T](featureDim)
		postAct = NewTensor[T](featureDim)
		
		// Fallback for reconstruction logic (using float32 intermediate)
		wData := CastWeights[float32](weights)
		for k := 0; k < numClusters; k++ {
			a := assignments[k]
			offset := k * featureDim
			for d := 0; d < featureDim; d++ {
				val := a * wData[offset+d]
				preAct.Data[d] += T(val)
			}
		}
		for d := 0; d < featureDim; d++ {
			postAct.Data[d] = Activate(preAct.Data[d], layer.Activation)
		}
	} else {
		// Output mode: probabilities (K outputs)
		preAct = NewTensor[T](numClusters)
		postAct = NewTensor[T](numClusters)
		for k := 0; k < numClusters; k++ {
			preAct.Data[k] = T(assignments[k])
			postAct.Data[k] = Activate(preAct.Data[k], layer.Activation)
		}
	}

	return preAct, postAct
}

// KMeansBackwardPolymorphic computes gradients for cluster centers and propagates to input.
func KMeansBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	numClusters := layer.NumClusters
	featureDim := len(input.Data)
	temp := layer.KMeansTemperature
	if temp == 0 { temp = 1.0 }
	tempSq := float32(temp * temp)
	
	gradInput = NewTensor[T](featureDim)
	gradWeights = NewTensor[T](numClusters * featureDim)
	
	// Re-compute assignments for backward logic
	assignments := make([]float32, numClusters)
	if layer.KMeansOutputMode != "features" {
		for k := 0; k < numClusters; k++ {
			assignments[k] = float32(preAct.Data[k])
		}
	} else {
		// Recompute if needed
	}

	gradLogits := SoftmaxBackward(ConvertTensor[T, float32](gradOutput).Data, assignments)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }
	wData := CastWeights[float32](weights)

	for k := 0; k < numClusters; k++ {
		offset := k * featureDim
		for d := 0; d < featureDim; d++ {
			diff := float32(input.Data[d]) - wData[offset+d]
			
			// dL/dc_k = gradLogits[k] * (x - c_k) / tempSq
			gradWeights.Data[offset+d] += T(gradLogits[k] * diff / tempSq)
			
			// dL/dx = - Sum gradLogits[k] * (x - c_k) / tempSq
			gradInput.Data[d] -= T(gradLogits[k] * diff / tempSq)
		}
	}

	return gradInput, gradWeights
}
