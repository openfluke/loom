package poly

// tween_native.go — native / true-native weight access for Tween (target propagation).

func tweenUsesNativeExact(n *VolumetricNetwork) bool {
	return n != nil && n.UseExactDType
}

func tweenHasWeights(layer *VolumetricLayer) bool {
	if layer == nil || layer.WeightStore == nil {
		return false
	}
	if len(layer.WeightStore.Master) > 0 {
		return true
	}
	return layer.WeightStore.GetNative(layer.DType) != nil
}

func tweenWeightCount(layer *VolumetricLayer) int {
	if layer == nil || layer.WeightStore == nil {
		return 0
	}
	if len(layer.WeightStore.Master) > 0 {
		return len(layer.WeightStore.Master)
	}
	native := layer.WeightStore.GetNative(layer.DType)
	if native == nil {
		return 0
	}
	switch layer.DType {
	case DTypeFloat64:
		return len(native.([]float64))
	case DTypeFloat16, DTypeBFloat16:
		return len(native.([]uint16))
	case DTypeFP8E4M3, DTypeFP8E5M2, DTypeUint8, DTypeUint4, DTypeUint2:
		return len(native.([]uint8))
	case DTypeInt64:
		return len(native.([]int64))
	case DTypeUint64:
		return len(native.([]uint64))
	case DTypeInt32:
		return len(native.([]int32))
	case DTypeUint32:
		return len(native.([]uint32))
	case DTypeInt16:
		return len(native.([]int16))
	case DTypeUint16:
		return len(native.([]uint16))
	default:
		return len(native.([]int8))
	}
}

// TweenWeightF32 reads one weight in float32 MAC space (exported for tests).
func TweenWeightF32(layer *VolumetricLayer, idx int) float32 {
	return tweenWeightF32(layer, idx)
}

// TweenWeightCount reports native or Master weight length (exported for tests).
func TweenWeightCount(layer *VolumetricLayer) int {
	return tweenWeightCount(layer)
}

// tweenWeightF32 reads one weight in float32 MAC space (native rules when UseExactDType).
func tweenWeightF32(layer *VolumetricLayer, idx int) float32 {
	if layer == nil || layer.WeightStore == nil {
		return 0
	}
	if layer.Network != nil && layer.Network.UseExactDType {
		return nativeWeightValueF32(layer.WeightStore, layer.DType, idx)
	}
	if idx < len(layer.WeightStore.Master) {
		return layer.WeightStore.Master[idx]
	}
	return 0
}

// applyTweenWeightDelta applies direct weight additions (gap / Hebbian updates).
func applyTweenWeightDelta(layer *VolumetricLayer, deltas []float32) {
	if layer == nil || layer.WeightStore == nil || len(deltas) == 0 {
		return
	}
	limit := tweenWeightCount(layer)
	if limit <= 0 || limit > len(deltas) {
		limit = len(deltas)
	}
	if layer.Network != nil && layer.Network.UseExactDType {
		grad := NewTensor[float32](limit)
		for i := 0; i < limit; i++ {
			grad.Data[i] = -deltas[i]
		}
		ApplyRecursiveGradients(layer, grad, 1.0, 0.0)
		return
	}
	ws := layer.WeightStore
	for i := 0; i < limit && i < len(ws.Master); i++ {
		ws.Master[i] += deltas[i]
	}
	if layer.DType != DTypeFloat32 {
		ws.Morph(layer.DType)
	}
}

func applyTweenGapsLayerwiseNative[T Numeric](n *VolumetricNetwork, s *TweenState[T], lr float32) {
	for i := 0; i < s.TotalLayers; i++ {
		budget := s.LinkBudgets[i]

		l := &n.Layers[i]
		if l.IsDisabled || l.WeightStore == nil {
			continue
		}
		if budget < 0.2 {
			continue
		}

		layerRate := lr * (0.5 + budget*0.5)

		input := s.ForwardActs[i]
		actual := s.ForwardActs[i+1]
		target := s.BackwardTargets[i+1]

		if input == nil || actual == nil || target == nil {
			continue
		}

		outSize := l.OutputHeight
		inSize := l.InputHeight
		gap := make([]float32, outSize)
		for j := 0; j < outSize && j < len(actual.Data) && j < len(target.Data); j++ {
			gap[j] = float32(target.Data[j]) - float32(actual.Data[j])
		}

		wCount := tweenWeightCount(l)
		if wCount == 0 {
			continue
		}
		deltas := make([]float32, wCount)

		switch l.Type {
		case LayerDense:
			velSize := (inSize * outSize) + outSize
			if s.WeightVel[i] == nil || len(s.WeightVel[i]) != velSize {
				s.WeightVel[i] = make([]float32, velSize)
			}
			mom := s.Config.Momentum

			for out := 0; out < outSize && out < len(gap); out++ {
				for in := 0; in < inSize && in < len(input.Data); in++ {
					wIdx := in*outSize + out
					if wIdx < wCount {
						delta := layerRate * float32(input.Data[in]) * gap[out]
						s.WeightVel[i][wIdx] = mom*s.WeightVel[i][wIdx] + (1-mom)*delta
						deltas[wIdx] = s.WeightVel[i][wIdx]
					}
				}
				bIdx := (inSize * outSize) + out
				if bIdx < wCount {
					delta := layerRate * gap[out]
					s.WeightVel[i][bIdx] = mom*s.WeightVel[i][bIdx] + (1-mom)*delta
					deltas[bIdx] = s.WeightVel[i][bIdx]
				}
			}
		case LayerRNN:
			ihSize := outSize * inSize
			hhSize := outSize * outSize
			seqLen := len(input.Data) / inSize
			for si := 0; si < seqLen; si++ {
				for out := 0; out < outSize; out++ {
					g := gap[si*outSize+out]
					bIdx := ihSize + hhSize + out
					if bIdx < wCount {
						deltas[bIdx] += layerRate * g
					}
					for in := 0; in < inSize; in++ {
						wIdx := out*inSize + in
						if wIdx < ihSize {
							deltas[wIdx] += layerRate * float32(input.Data[si*inSize+in]) * g
						}
					}
					if si > 0 {
						for hp := 0; hp < outSize; hp++ {
							wIdx := ihSize + out*outSize + hp
							if wIdx < ihSize+hhSize {
								deltas[wIdx] += layerRate * float32(actual.Data[(si-1)*outSize+hp]) * g * 0.5
							}
						}
					}
				}
			}
		case LayerLSTM:
			ihSize := outSize * inSize
			hhSize := outSize * outSize
			bSize := outSize
			gateSize := ihSize + hhSize + bSize
			seqLen := len(input.Data) / inSize
			for si := 0; si < seqLen; si++ {
				for g := 0; g < 4; g++ {
					gateOffset := g * gateSize
					for out := 0; out < outSize; out++ {
						localGap := gap[si*outSize+out]
						bIdx := gateOffset + ihSize + hhSize + out
						if bIdx < wCount {
							deltas[bIdx] += layerRate * localGap * 0.25
						}
						for in := 0; in < inSize; in++ {
							wIdx := gateOffset + out*inSize + in
							if wIdx < gateOffset+ihSize {
								deltas[wIdx] += layerRate * float32(input.Data[si*inSize+in]) * localGap * 0.25
							}
						}
						if si > 0 {
							for hp := 0; hp < outSize; hp++ {
								wIdx := gateOffset + ihSize + out*outSize + hp
								if wIdx < gateOffset+ihSize+hhSize {
									deltas[wIdx] += layerRate * float32(actual.Data[(si-1)*outSize+hp]) * localGap * 0.1
								}
							}
						}
					}
				}
			}
		case LayerMultiHeadAttention:
			seqLen := len(input.Data) / inSize
			for si := 0; si < seqLen; si++ {
				for out := 0; out < outSize && out < len(gap)/seqLen; out++ {
					g := gap[si*outSize+out]
					for in := 0; in < inSize && in < len(input.Data)/seqLen; in++ {
						wIdx := in*outSize + out
						if wIdx < wCount {
							deltas[wIdx] += layerRate * float32(input.Data[si*inSize+in]) * g * 0.5
						}
					}
				}
			}
		case LayerSwiGLU:
			intermediateSize := l.OutputHeight
			downWStart := 2 * inSize * intermediateSize
			seqLen := len(input.Data) / inSize
			for si := 0; si < seqLen; si++ {
				for out := 0; out < inSize && out < len(gap)/seqLen; out++ {
					g := gap[si*inSize+out]
					for in := 0; in < intermediateSize; in++ {
						wIdx := downWStart + in*inSize + out
						if wIdx < wCount {
							deltas[wIdx] += layerRate * g * 0.1
						}
					}
				}
			}
		case LayerLayerNorm, LayerRMSNorm:
			if outSize <= wCount {
				for out := 0; out < outSize && out < len(gap); out++ {
					deltas[out] += layerRate * gap[out] * 0.01
				}
			}
		default:
			continue
		}

		applyTweenWeightDelta(l, deltas)
	}
}
