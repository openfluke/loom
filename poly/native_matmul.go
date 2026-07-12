package poly

// native_matmul.go — per-element native dtype MAC (no bulk GetActive / []float32 decode).

// IsTrueNativeDType reports dtypes with fully integer forward/backward/update.
func IsTrueNativeDType(dtype DType) bool {
	return IsDenseTrueNativeDType(dtype)
}

func useLayerTrueNative(layer *VolumetricLayer) bool {
	return layer != nil &&
		layer.Network != nil &&
		layer.Network.UseExactDType &&
		layer.WeightStore != nil &&
		layerSupportsNativeExact(layer.Type) &&
		IsTrueNativeDType(layer.DType)
}

// nativeDotRow computes one output neuron via storage-dtype MAC rules.
func nativeDotRow(layer *VolumetricLayer, input []float32, rowOff, inSz int) float32 {
	return denseNativeDotForward(layer, input, rowOff, inSz)
}

func nativeBiasAt(layer *VolumetricLayer, biasIdx int) float32 {
	return nativeWeightValueF32(layer.WeightStore, layer.DType, biasIdx)
}

func nativeWeightValueF32(ws *WeightStore, dtype DType, idx int) float32 {
	if ws == nil {
		return 0
	}
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	native := ws.GetNative(dtype)
	if native == nil {
		if ws.Master != nil && idx < len(ws.Master) {
			return ws.Master[idx]
		}
		return 0
	}

	switch dtype {
	case DTypeFloat64:
		w := native.([]float64)
		if idx >= len(w) {
			return 0
		}
		return float32(w[idx])
	case DTypeFloat32:
		if ws.Master != nil && idx < len(ws.Master) {
			return ws.Master[idx]
		}
		return 0
	case DTypeFloat16:
		w := native.([]uint16)
		if idx >= len(w) {
			return 0
		}
		return float16ToFloat32(w[idx])
	case DTypeBFloat16:
		w := native.([]uint16)
		if idx >= len(w) {
			return 0
		}
		return bfloat16ToFloat32(w[idx])
	case DTypeFP8E4M3:
		w := native.([]uint8)
		if idx >= len(w) {
			return 0
		}
		return e4m3ToFloat32(w[idx]) * scale
	case DTypeFP8E5M2:
		w := native.([]uint8)
		if idx >= len(w) {
			return 0
		}
		return e5m2ToFloat32(w[idx]) * scale
	case DTypeInt64:
		w := native.([]int64)
		if idx >= len(w) {
			return 0
		}
		return float32(w[idx]) * scale
	case DTypeUint64:
		w := native.([]uint64)
		if idx >= len(w) {
			return 0
		}
		return float32(w[idx]) * scale
	case DTypeInt32:
		w := native.([]int32)
		if idx >= len(w) {
			return 0
		}
		return float32(w[idx]) * scale
	case DTypeUint32:
		w := native.([]uint32)
		if idx >= len(w) {
			return 0
		}
		return float32(w[idx]) * scale
	case DTypeInt16:
		w := native.([]int16)
		if idx >= len(w) {
			return 0
		}
		return float32(w[idx]) * scale
	case DTypeUint16:
		w := native.([]uint16)
		if idx >= len(w) {
			return 0
		}
		return float32(w[idx]) * scale
	case DTypeFP4:
		w := native.([]uint8)
		if idx >= len(w) {
			return 0
		}
		return fp4CodeToFloat32(w[idx], scale)
	default:
		codes, ok := nativeU8WeightsView(native)
		if !ok || idx >= len(codes) {
			return 0
		}
		wv := denseNativeSignedU8Weight(dtype, codes[idx], scale)
		return float32(wv) * scale
	}
}

func nativeGradW(layer *VolumetricLayer, inputVal float32, gradPre float64) float64 {
	return denseNativeGradWTerm(layer, inputVal, gradPre)
}

func nativeGradX(layer *VolumetricLayer, weightIdx int, gradPre float64) float64 {
	return denseNativeGradXTerm(layer, weightIdx, gradPre)
}

func markLayerNativeWeightsUpdated(layer *VolumetricLayer, ws *WeightStore, dtype DType, raw []uint8) {
	if layer.ExactDense == nil {
		layer.ExactDense = &DenseExactCache{}
	}
	layer.ExactDense.WeightsUpdated = true
	ws.Versions[dtype] = raw
	ws.Master = nil
	ws.GPUWeights = make(map[DType]any)
	if ws.CPUPacked != nil {
		delete(ws.CPUPacked, dtype)
	}
}

func publishInt8Weights(ws *WeightStore, dtype DType, weights []int8) {
	raw := make([]uint8, len(weights))
	for i, v := range weights {
		raw[i] = uint8(v)
	}
	ws.Versions[dtype] = raw
}
