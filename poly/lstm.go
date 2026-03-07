package poly

import (
	"math"
)

// LSTMForwardPolymorphic performs a forward pass through a polymorphic LSTM layer.
// preAct stores [iSum, fSum, gSum, oSum, cCurr] (5 * hiddenSize)
func LSTMForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize, inputSize, hiddenSize, seqLength := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	preAct = NewTensor[T](batchSize, seqLength, 5*hiddenSize) 
	postAct = NewTensor[T](batchSize, seqLength, hiddenSize) 

	scale := layer.WeightStore.Scale
	if scale == 0 { scale = 1.0 }

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }

	ihSize, hhSize, bSize := hiddenSize*inputSize, hiddenSize*hiddenSize, hiddenSize
	gateSize := ihSize + hhSize + bSize

	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			wI, wF, wG, wO := rawW[0:gateSize], rawW[gateSize:2*gateSize], rawW[2*gateSize:3*gateSize], rawW[3*gateSize:4*gateSize]
			for b := 0; b < batchSize; b++ {
				hPrev, cPrev := make([]float64, hiddenSize), make([]float64, hiddenSize)
				for t := 0; t < seqLength; t++ {
					it, pIdx := b*seqLength*inputSize+t*inputSize, b*seqLength*5*hiddenSize+t*5*hiddenSize
					for h := 0; h < hiddenSize; h++ {
						iS, fS, gS, oS := float64(wI[ihSize+hhSize+h]), float64(wF[ihSize+hhSize+h]), float64(wG[ihSize+hhSize+h]), float64(wO[ihSize+hhSize+h])
						for i := 0; i < inputSize; i++ {
							x := float64(input.Data[it+i])
							iS += x * float64(wI[h*inputSize+i]); fS += x * float64(wF[h*inputSize+i]); gS += x * float64(wG[h*inputSize+i]); oS += x * float64(wO[h*inputSize+i])
						}
						for hp := 0; hp < hiddenSize; hp++ {
							hv := float64(hPrev[hp])
							iS += hv * float64(wI[ihSize+h*hiddenSize+hp]); fS += hv * float64(wF[ihSize+h*hiddenSize+hp]); gS += hv * float64(wG[ihSize+h*hiddenSize+hp]); oS += hv * float64(wO[ihSize+h*hiddenSize+hp])
						}
						iG, fG, oG := 1.0/(1.0+float64(math.Exp(-float64(iS)))), 1.0/(1.0+float64(math.Exp(-float64(fS)))), 1.0/(1.0+float64(math.Exp(-float64(oS))))
						gG := float64(math.Tanh(float64(gS)))
						cC := fG*float64(cPrev[h]) + iG*gG
						hC := oG * float64(math.Tanh(cC))
						postAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(hC)
						preAct.Data[pIdx+h], preAct.Data[pIdx+hiddenSize+h], preAct.Data[pIdx+2*hiddenSize+h], preAct.Data[pIdx+3*hiddenSize+h], preAct.Data[pIdx+4*hiddenSize+h] = T(iS), T(fS), T(gS), T(oS), T(cC)
						hPrev[h], cPrev[h] = float64(hC), float64(cC)
					}
				}
			}
			return preAct, postAct
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			wI, wF, wG, wO := rawW[0:gateSize], rawW[gateSize:2*gateSize], rawW[2*gateSize:3*gateSize], rawW[3*gateSize:4*gateSize]
			for b := 0; b < batchSize; b++ {
				hPrev, cPrev := make([]float32, hiddenSize), make([]float32, hiddenSize)
				for t := 0; t < seqLength; t++ {
					it, pIdx := b*seqLength*inputSize+t*inputSize, b*seqLength*5*hiddenSize+t*5*hiddenSize
					for h := 0; h < hiddenSize; h++ {
						iS, fS, gS, oS := float32(wI[ihSize+hhSize+h]), float32(wF[ihSize+hhSize+h]), float32(wG[ihSize+hhSize+h]), float32(wO[ihSize+hhSize+h])
						for i := 0; i < inputSize; i++ {
							x := float32(input.Data[it+i])
							iS += x * float32(wI[h*inputSize+i]); fS += x * float32(wF[h*inputSize+i]); gS += x * float32(wG[h*inputSize+i]); oS += x * float32(wO[h*inputSize+i])
						}
						for hp := 0; hp < hiddenSize; hp++ {
							hv := float32(hPrev[hp])
							iS += hv * float32(wI[ihSize+h*hiddenSize+hp]); fS += hv * float32(wF[ihSize+h*hiddenSize+hp]); gS += hv * float32(wG[ihSize+h*hiddenSize+hp]); oS += hv * float32(wO[ihSize+h*hiddenSize+hp])
						}
						iG, fG, oG := 1.0/(1.0+float64(math.Exp(-float64(iS)))), 1.0/(1.0+float64(math.Exp(-float64(fS)))), 1.0/(1.0+float64(math.Exp(-float64(oS))))
						gG := float64(math.Tanh(float64(gS)))
						cC := fG*float64(cPrev[h]) + iG*gG
						hC := oG * float64(math.Tanh(cC))
						postAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(hC)
						preAct.Data[pIdx+h], preAct.Data[pIdx+hiddenSize+h], preAct.Data[pIdx+2*hiddenSize+h], preAct.Data[pIdx+3*hiddenSize+h], preAct.Data[pIdx+4*hiddenSize+h] = T(iS), T(fS), T(gS), T(oS), T(cC)
						hPrev[h], cPrev[h] = float32(hC), float32(cC)
					}
				}
			}
			return preAct, postAct
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			wI, wF, wG, wO := rawW[0:gateSize], rawW[gateSize:2*gateSize], rawW[2*gateSize:3*gateSize], rawW[3*gateSize:4*gateSize]
			for b := 0; b < batchSize; b++ {
				hPrev, cPrev := make([]int64, hiddenSize), make([]int64, hiddenSize)
				for t := 0; t < seqLength; t++ {
					it, pIdx := b*seqLength*inputSize+t*inputSize, b*seqLength*5*hiddenSize+t*5*hiddenSize
					for h := 0; h < hiddenSize; h++ {
						iS, fS, gS, oS := int64(wI[ihSize+hhSize+h]), int64(wF[ihSize+hhSize+h]), int64(wG[ihSize+hhSize+h]), int64(wO[ihSize+hhSize+h])
						for i := 0; i < inputSize; i++ {
							x := int64(input.Data[it+i])
							iS += x * int64(wI[h*inputSize+i]); fS += x * int64(wF[h*inputSize+i]); gS += x * int64(wG[h*inputSize+i]); oS += x * int64(wO[h*inputSize+i])
						}
						for hp := 0; hp < hiddenSize; hp++ {
							hv := int64(hPrev[hp])
							iS += hv * int64(wI[ihSize+h*hiddenSize+hp]); fS += hv * int64(wF[ihSize+h*hiddenSize+hp]); gS += hv * int64(wG[ihSize+h*hiddenSize+hp]); oS += hv * int64(wO[ihSize+h*hiddenSize+hp])
						}
						iG, fG, oG := 1.0/(1.0+float64(math.Exp(-float64(iS)))), 1.0/(1.0+float64(math.Exp(-float64(fS)))), 1.0/(1.0+float64(math.Exp(-float64(oS))))
						gG := float64(math.Tanh(float64(gS)))
						cC := fG*float64(cPrev[h]) + iG*gG
						hC := oG * float64(math.Tanh(cC))
						postAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(hC)
						preAct.Data[pIdx+h], preAct.Data[pIdx+hiddenSize+h], preAct.Data[pIdx+2*hiddenSize+h], preAct.Data[pIdx+3*hiddenSize+h], preAct.Data[pIdx+4*hiddenSize+h] = T(iS), T(fS), T(gS), T(oS), T(cC)
						hPrev[h], cPrev[h] = int64(hC), int64(cC)
					}
				}
			}
			return preAct, postAct
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			wI, wF, wG, wO := rawW[0:gateSize], rawW[gateSize:2*gateSize], rawW[2*gateSize:3*gateSize], rawW[3*gateSize:4*gateSize]
			for b := 0; b < batchSize; b++ {
				hPrev, cPrev := make([]int32, hiddenSize), make([]int32, hiddenSize)
				for t := 0; t < seqLength; t++ {
					it, pIdx := b*seqLength*inputSize+t*inputSize, b*seqLength*5*hiddenSize+t*5*hiddenSize
					for h := 0; h < hiddenSize; h++ {
						iS, fS, gS, oS := int32(wI[ihSize+hhSize+h]), int32(wF[ihSize+hhSize+h]), int32(wG[ihSize+hhSize+h]), int32(wO[ihSize+hhSize+h])
						for i := 0; i < inputSize; i++ {
							x := int32(input.Data[it+i])
							iS += x * int32(wI[h*inputSize+i]); fS += x * int32(wF[h*inputSize+i]); gS += x * int32(wG[h*inputSize+i]); oS += x * int32(wO[h*inputSize+i])
						}
						for hp := 0; hp < hiddenSize; hp++ {
							hv := int32(hPrev[hp])
							iS += hv * int32(wI[ihSize+h*hiddenSize+hp]); fS += hv * int32(wF[ihSize+h*hiddenSize+hp]); gS += hv * int32(wG[ihSize+h*hiddenSize+hp]); oS += hv * int32(wO[ihSize+h*hiddenSize+hp])
						}
						iG, fG, oG := 1.0/(1.0+float64(math.Exp(-float64(iS)))), 1.0/(1.0+float64(math.Exp(-float64(fS)))), 1.0/(1.0+float64(math.Exp(-float64(oS))))
						gG := float64(math.Tanh(float64(gS)))
						cC := fG*float64(cPrev[h]) + iG*gG
						hC := oG * float64(math.Tanh(cC))
						postAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(hC)
						preAct.Data[pIdx+h], preAct.Data[pIdx+hiddenSize+h], preAct.Data[pIdx+2*hiddenSize+h], preAct.Data[pIdx+3*hiddenSize+h], preAct.Data[pIdx+4*hiddenSize+h] = T(iS), T(fS), T(gS), T(oS), T(cC)
						hPrev[h], cPrev[h] = int32(hC), int32(cC)
					}
				}
			}
			return preAct, postAct
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			wI, wF, wG, wO := rawW[0:gateSize], rawW[gateSize:2*gateSize], rawW[2*gateSize:3*gateSize], rawW[3*gateSize:4*gateSize]
			for b := 0; b < batchSize; b++ {
				hPrev, cPrev := make([]int32, hiddenSize), make([]int32, hiddenSize)
				for t := 0; t < seqLength; t++ {
					it, pIdx := b*seqLength*inputSize+t*inputSize, b*seqLength*5*hiddenSize+t*5*hiddenSize
					for h := 0; h < hiddenSize; h++ {
						iS, fS, gS, oS := int32(wI[ihSize+hhSize+h]), int32(wF[ihSize+hhSize+h]), int32(wG[ihSize+hhSize+h]), int32(wO[ihSize+hhSize+h])
						for i := 0; i < inputSize; i++ {
							x := int32(input.Data[it+i])
							iS += x * int32(wI[h*inputSize+i]); fS += x * int32(wF[h*inputSize+i]); gS += x * int32(wG[h*inputSize+i]); oS += x * int32(wO[h*inputSize+i])
						}
						for hp := 0; hp < hiddenSize; hp++ {
							hv := int32(hPrev[hp])
							iS += hv * int32(wI[ihSize+h*hiddenSize+hp]); fS += hv * int32(wF[ihSize+h*hiddenSize+hp]); gS += hv * int32(wG[ihSize+h*hiddenSize+hp]); oS += hv * int32(wO[ihSize+h*hiddenSize+hp])
						}
						iG, fG, oG := 1.0/(1.0+float64(math.Exp(-float64(iS)))), 1.0/(1.0+float64(math.Exp(-float64(fS)))), 1.0/(1.0+float64(math.Exp(-float64(oS))))
						gG := float64(math.Tanh(float64(gS)))
						cC := fG*float64(cPrev[h]) + iG*gG
						hC := oG * float64(math.Tanh(cC))
						postAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(hC)
						preAct.Data[pIdx+h], preAct.Data[pIdx+hiddenSize+h], preAct.Data[pIdx+2*hiddenSize+h], preAct.Data[pIdx+3*hiddenSize+h], preAct.Data[pIdx+4*hiddenSize+h] = T(iS), T(fS), T(gS), T(oS), T(cC)
						hPrev[h], cPrev[h] = int32(hC), int32(cC)
					}
				}
			}
			return preAct, postAct
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			wI, wF, wG, wO := rawW[0:gateSize], rawW[gateSize:2*gateSize], rawW[2*gateSize:3*gateSize], rawW[3*gateSize:4*gateSize]
			for b := 0; b < batchSize; b++ {
				hPrev, cPrev := make([]int32, hiddenSize), make([]int32, hiddenSize)
				for t := 0; t < seqLength; t++ {
					it, pIdx := b*seqLength*inputSize+t*inputSize, b*seqLength*5*hiddenSize+t*5*hiddenSize
					for h := 0; h < hiddenSize; h++ {
						iS, fS, gS, oS := int32(wI[ihSize+hhSize+h]), int32(wF[ihSize+hhSize+h]), int32(wG[ihSize+hhSize+h]), int32(wO[ihSize+hhSize+h])
						for i := 0; i < inputSize; i++ {
							x := int32(input.Data[it+i])
							iS += x * int32(wI[h*inputSize+i]); fS += x * int32(wF[h*inputSize+i]); gS += x * int32(wG[h*inputSize+i]); oS += x * int32(wO[h*inputSize+i])
						}
						for hp := 0; hp < hiddenSize; hp++ {
							hv := int32(hPrev[hp])
							iS += hv * int32(wI[ihSize+h*hiddenSize+hp]); fS += hv * int32(wF[ihSize+h*hiddenSize+hp]); gS += hv * int32(wG[ihSize+h*hiddenSize+hp]); oS += hv * int32(wO[ihSize+h*hiddenSize+hp])
						}
						iG, fG, oG := 1.0/(1.0+float64(math.Exp(-float64(iS)))), 1.0/(1.0+float64(math.Exp(-float64(fS)))), 1.0/(1.0+float64(math.Exp(-float64(oS))))
						gG := float64(math.Tanh(float64(gS)))
						cC := fG*float64(cPrev[h]) + iG*gG
						hC := oG * float64(math.Tanh(cC))
						postAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(hC)
						preAct.Data[pIdx+h], preAct.Data[pIdx+hiddenSize+h], preAct.Data[pIdx+2*hiddenSize+h], preAct.Data[pIdx+3*hiddenSize+h], preAct.Data[pIdx+4*hiddenSize+h] = T(iS), T(fS), T(gS), T(oS), T(cC)
						hPrev[h], cPrev[h] = int32(hC), int32(cC)
					}
				}
			}
			return preAct, postAct
		}
	}

	// UNIVERSAL POLYMORPHIC FALLTHROUGH
	wData := CastWeights[T](weights)
	wI, wF, wG, wO := wData[0:gateSize], wData[gateSize:2*gateSize], wData[2*gateSize:3*gateSize], wData[3*gateSize:4*gateSize]

	for b := 0; b < batchSize; b++ {
		hPrev := make([]float32, hiddenSize)
		cPrev := make([]float32, hiddenSize)
		for t := 0; t < seqLength; t++ {
			it := b*seqLength*inputSize + t*inputSize
			for h := 0; h < hiddenSize; h++ {
				iSum := SimulatePrecision(float32(wI[ihSize+hhSize+h]), layer.DType, scale)
				for i := 0; i < inputSize; i++ { iSum += float32(input.Data[it+i]) * SimulatePrecision(float32(wI[h*inputSize+i]), layer.DType, scale) }
				for hp := 0; hp < hiddenSize; hp++ { iSum += hPrev[hp] * SimulatePrecision(float32(wI[ihSize+h*hiddenSize+hp]), layer.DType, scale) }
				iG := 1.0 / (1.0 + float32(math.Exp(-float64(iSum))))

				fSum := SimulatePrecision(float32(wF[ihSize+hhSize+h]), layer.DType, scale)
				for i := 0; i < inputSize; i++ { fSum += float32(input.Data[it+i]) * SimulatePrecision(float32(wF[h*inputSize+i]), layer.DType, scale) }
				for hp := 0; hp < hiddenSize; hp++ { fSum += hPrev[hp] * SimulatePrecision(float32(wF[ihSize+h*hiddenSize+hp]), layer.DType, scale) }
				fG := 1.0 / (1.0 + float32(math.Exp(-float64(fSum))))

				gSum := SimulatePrecision(float32(wG[ihSize+hhSize+h]), layer.DType, scale)
				for i := 0; i < inputSize; i++ { gSum += float32(input.Data[it+i]) * SimulatePrecision(float32(wG[h*inputSize+i]), layer.DType, scale) }
				for hp := 0; hp < hiddenSize; hp++ { gSum += hPrev[hp] * SimulatePrecision(float32(wG[ihSize+h*hiddenSize+hp]), layer.DType, scale) }
				gG := float32(math.Tanh(float64(gSum)))

				oSum := SimulatePrecision(float32(wO[ihSize+hhSize+h]), layer.DType, scale)
				for i := 0; i < inputSize; i++ { oSum += float32(input.Data[it+i]) * SimulatePrecision(float32(wO[h*inputSize+i]), layer.DType, scale) }
				for hp := 0; hp < hiddenSize; hp++ { oSum += hPrev[hp] * SimulatePrecision(float32(wO[ihSize+h*hiddenSize+hp]), layer.DType, scale) }
				oG := 1.0 / (1.0 + float32(math.Exp(-float64(oSum))))

				cC := fG*cPrev[h] + iG*gG
				cPrev[h] = cC
				hC := oG * float32(math.Tanh(float64(cC)))
				postAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(hC)
				hPrev[h] = hC

				pIdx := b*seqLength*5*hiddenSize + t*5*hiddenSize
				preAct.Data[pIdx+h], preAct.Data[pIdx+hiddenSize+h], preAct.Data[pIdx+2*hiddenSize+h], preAct.Data[pIdx+3*hiddenSize+h], preAct.Data[pIdx+4*hiddenSize+h] = T(iSum), T(fSum), T(gSum), T(oSum), T(cC)
			}
		}
	}

	return preAct, postAct
}

// LSTMBackwardPolymorphic calculates gradients for the LSTM layer using BPTT.
func LSTMBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize, inputSize, hiddenSize, seqLength := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	gradInput = NewTensor[T](batchSize, seqLength, inputSize)
	gradWeights = NewTensor[T](len(layer.WeightStore.Master))

	ihSize, hhSize, bSize := hiddenSize*inputSize, hiddenSize*hiddenSize, hiddenSize
	gateSize := ihSize + hhSize + bSize

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }
	
	scale := layer.WeightStore.Scale
	if scale == 0 { scale = 1.0 }

	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			wI, wF, wG, wO := rawW[0:gateSize], rawW[gateSize:2*gateSize], rawW[2*gateSize:3*gateSize], rawW[3*gateSize:4*gateSize]
			gBI, gBF, gBG, gBO := make([]float64, bSize), make([]float64, bSize), make([]float64, bSize), make([]float64, bSize)
			gIHI, gIHF, gIHG, gIHO := make([]float64, ihSize), make([]float64, ihSize), make([]float64, ihSize), make([]float64, ihSize)
			gHHI, gHHF, gHHG, gHHO := make([]float64, hhSize), make([]float64, hhSize), make([]float64, hhSize), make([]float64, hhSize)
			for b := 0; b < batchSize; b++ {
				gradH, gradC := make([]float64, hiddenSize), make([]float64, hiddenSize)
				for t := seqLength - 1; t >= 0; t-- {
					nextGradH, nextGradC := make([]float64, hiddenSize), make([]float64, hiddenSize)
					it, pIdx := b*seqLength*inputSize+t*inputSize, b*seqLength*5*hiddenSize+t*5*hiddenSize
					for h := 0; h < hiddenSize; h++ {
						dh := gradH[h] + float64(gradOutput.Data[b*seqLength*hiddenSize+t*hiddenSize+h])
						iS, fS, gS, oS, cC := float64(preAct.Data[pIdx+h]), float64(preAct.Data[pIdx+hiddenSize+h]), float64(preAct.Data[pIdx+2*hiddenSize+h]), float64(preAct.Data[pIdx+3*hiddenSize+h]), float64(preAct.Data[pIdx+4*hiddenSize+h])
						iG, fG, oG := 1.0/(1.0+math.Exp(-iS)), 1.0/(1.0+math.Exp(-fS)), 1.0/(1.0+math.Exp(-oS))
						gG, cT := math.Tanh(gS), math.Tanh(cC)
						cP := 0.0; if t > 0 { cP = float64(preAct.Data[pIdx-5*hiddenSize+4*hiddenSize+h]) }
						dc := gradC[h] + dh*oG*(1.0-cT*cT)
						doP, dfP, diP, dgP := dh*cT*oG*(1.0-oG), dc*cP*fG*(1.0-fG), dc*gG*iG*(1.0-iG), dc*iG*(1.0-gG*gG)
						gBI[h] += float64(diP); gBF[h] += float64(dfP); gBG[h] += float64(dgP); gBO[h] += float64(doP)
						for i := 0; i < inputSize; i++ {
							x := float64(input.Data[it+i])
							gIHI[h*inputSize+i] += float64(diP * x); gIHF[h*inputSize+i] += float64(dfP * x); gIHG[h*inputSize+i] += float64(dgP * x); gIHO[h*inputSize+i] += float64(doP * x)
							gradInput.Data[it+i] += T(float64(wI[h*inputSize+i])*diP + float64(wF[h*inputSize+i])*dfP + float64(wG[h*inputSize+i])*dgP + float64(wO[h*inputSize+i])*doP)
						}
						for hp := 0; hp < hiddenSize; hp++ {
							hvP := 0.0; if t > 0 {
								pP := pIdx - 5*hiddenSize
								oGP := 1.0 / (1.0 + math.Exp(-float64(preAct.Data[pP+3*hiddenSize+hp])))
								hvP = oGP * math.Tanh(float64(preAct.Data[pP+4*hiddenSize+hp]))
							}
							gHHI[h*hiddenSize+hp] += float64(diP * hvP); gHHF[h*hiddenSize+hp] += float64(dfP * hvP); gHHG[h*hiddenSize+hp] += float64(dgP * hvP); gHHO[h*hiddenSize+hp] += float64(doP * hvP)
							nextGradH[hp] += float64(wI[ihSize+h*hiddenSize+hp])*diP + float64(wF[ihSize+h*hiddenSize+hp])*dfP + float64(wG[ihSize+h*hiddenSize+hp])*dgP + float64(wO[ihSize+h*hiddenSize+hp])*doP
						}
						nextGradC[h] = dc * fG
					}
					gradH, gradC = nextGradH, nextGradC
				}
			}
			for h := 0; h < hiddenSize; h++ {
				for i := 0; i < inputSize; i++ {
					gradWeights.Data[h*inputSize+i], gradWeights.Data[gateSize+h*inputSize+i], gradWeights.Data[2*gateSize+h*inputSize+i], gradWeights.Data[3*gateSize+h*inputSize+i] = T(gIHI[h*inputSize+i]), T(gIHF[h*inputSize+i]), T(gIHG[h*inputSize+i]), T(gIHO[h*inputSize+i])
				}
				for hp := 0; hp < hiddenSize; hp++ {
					gradWeights.Data[ihSize+h*hiddenSize+hp], gradWeights.Data[gateSize+ihSize+h*hiddenSize+hp], gradWeights.Data[2*gateSize+ihSize+h*hiddenSize+hp], gradWeights.Data[3*gateSize+ihSize+h*hiddenSize+hp] = T(gHHI[h*hiddenSize+hp]), T(gHHF[h*hiddenSize+hp]), T(gHHG[h*hiddenSize+hp]), T(gHHO[h*hiddenSize+hp])
				}
				gradWeights.Data[ihSize+hhSize+h], gradWeights.Data[gateSize+ihSize+hhSize+h], gradWeights.Data[2*gateSize+ihSize+hhSize+h], gradWeights.Data[3*gateSize+ihSize+hhSize+h] = T(gBI[h]), T(gBF[h]), T(gBG[h]), T(gBO[h])
			}
			return gradInput, gradWeights
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			wI, wF, wG, wO := rawW[0:gateSize], rawW[gateSize:2*gateSize], rawW[2*gateSize:3*gateSize], rawW[3*gateSize:4*gateSize]
			gBI, gBF, gBG, gBO := make([]float32, bSize), make([]float32, bSize), make([]float32, bSize), make([]float32, bSize)
			gIHI, gIHF, gIHG, gIHO := make([]float32, ihSize), make([]float32, ihSize), make([]float32, ihSize), make([]float32, ihSize)
			gHHI, gHHF, gHHG, gHHO := make([]float32, hhSize), make([]float32, hhSize), make([]float32, hhSize), make([]float32, hhSize)
			for b := 0; b < batchSize; b++ {
				gradH, gradC := make([]float64, hiddenSize), make([]float64, hiddenSize)
				for t := seqLength - 1; t >= 0; t-- {
					nextGradH, nextGradC := make([]float64, hiddenSize), make([]float64, hiddenSize)
					it, pIdx := b*seqLength*inputSize+t*inputSize, b*seqLength*5*hiddenSize+t*5*hiddenSize
					for h := 0; h < hiddenSize; h++ {
						dh := gradH[h] + float64(gradOutput.Data[b*seqLength*hiddenSize+t*hiddenSize+h])
						iS, fS, gS, oS, cC := float64(preAct.Data[pIdx+h]), float64(preAct.Data[pIdx+hiddenSize+h]), float64(preAct.Data[pIdx+2*hiddenSize+h]), float64(preAct.Data[pIdx+3*hiddenSize+h]), float64(preAct.Data[pIdx+4*hiddenSize+h])
						iG, fG, oG := 1.0/(1.0+math.Exp(-iS)), 1.0/(1.0+math.Exp(-fS)), 1.0/(1.0+math.Exp(-oS))
						gG, cT := math.Tanh(gS), math.Tanh(cC)
						cP := 0.0; if t > 0 { cP = float64(preAct.Data[pIdx-5*hiddenSize+4*hiddenSize+h]) }
						dc := gradC[h] + dh*oG*(1.0-cT*cT)
						doP, dfP, diP, dgP := dh*cT*oG*(1.0-oG), dc*cP*fG*(1.0-fG), dc*gG*iG*(1.0-iG), dc*iG*(1.0-gG*gG)
						gBI[h] += float32(diP); gBF[h] += float32(dfP); gBG[h] += float32(dgP); gBO[h] += float32(doP)
						for i := 0; i < inputSize; i++ {
							x := float64(input.Data[it+i])
							gIHI[h*inputSize+i] += float32(diP * x); gIHF[h*inputSize+i] += float32(dfP * x); gIHG[h*inputSize+i] += float32(dgP * x); gIHO[h*inputSize+i] += float32(doP * x)
							gradInput.Data[it+i] += T(float64(wI[h*inputSize+i])*diP + float64(wF[h*inputSize+i])*dfP + float64(wG[h*inputSize+i])*dgP + float64(wO[h*inputSize+i])*doP)
						}
						for hp := 0; hp < hiddenSize; hp++ {
							hvP := 0.0; if t > 0 {
								pP := pIdx - 5*hiddenSize
								oGP := 1.0 / (1.0 + math.Exp(-float64(preAct.Data[pP+3*hiddenSize+hp])))
								hvP = oGP * math.Tanh(float64(preAct.Data[pP+4*hiddenSize+hp]))
							}
							gHHI[h*hiddenSize+hp] += float32(diP * hvP); gHHF[h*hiddenSize+hp] += float32(dfP * hvP); gHHG[h*hiddenSize+hp] += float32(dgP * hvP); gHHO[h*hiddenSize+hp] += float32(doP * hvP)
							nextGradH[hp] += float64(wI[ihSize+h*hiddenSize+hp])*diP + float64(wF[ihSize+h*hiddenSize+hp])*dfP + float64(wG[ihSize+h*hiddenSize+hp])*dgP + float64(wO[ihSize+h*hiddenSize+hp])*doP
						}
						nextGradC[h] = dc * fG
					}
					gradH, gradC = nextGradH, nextGradC
				}
			}
			for h := 0; h < hiddenSize; h++ {
				for i := 0; i < inputSize; i++ {
					gradWeights.Data[h*inputSize+i], gradWeights.Data[gateSize+h*inputSize+i], gradWeights.Data[2*gateSize+h*inputSize+i], gradWeights.Data[3*gateSize+h*inputSize+i] = T(gIHI[h*inputSize+i]), T(gIHF[h*inputSize+i]), T(gIHG[h*inputSize+i]), T(gIHO[h*inputSize+i])
				}
				for hp := 0; hp < hiddenSize; hp++ {
					gradWeights.Data[ihSize+h*hiddenSize+hp], gradWeights.Data[gateSize+ihSize+h*hiddenSize+hp], gradWeights.Data[2*gateSize+ihSize+h*hiddenSize+hp], gradWeights.Data[3*gateSize+ihSize+h*hiddenSize+hp] = T(gHHI[h*hiddenSize+hp]), T(gHHF[h*hiddenSize+hp]), T(gHHG[h*hiddenSize+hp]), T(gHHO[h*hiddenSize+hp])
				}
				gradWeights.Data[ihSize+hhSize+h], gradWeights.Data[gateSize+ihSize+hhSize+h], gradWeights.Data[2*gateSize+ihSize+hhSize+h], gradWeights.Data[3*gateSize+ihSize+hhSize+h] = T(gBI[h]), T(gBF[h]), T(gBG[h]), T(gBO[h])
			}
			return gradInput, gradWeights
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			wI, wF, wG, wO := rawW[0:gateSize], rawW[gateSize:2*gateSize], rawW[2*gateSize:3*gateSize], rawW[3*gateSize:4*gateSize]
			gBI, gBF, gBG, gBO := make([]int64, bSize), make([]int64, bSize), make([]int64, bSize), make([]int64, bSize)
			gIHI, gIHF, gIHG, gIHO := make([]int64, ihSize), make([]int64, ihSize), make([]int64, ihSize), make([]int64, ihSize)
			gHHI, gHHF, gHHG, gHHO := make([]int64, hhSize), make([]int64, hhSize), make([]int64, hhSize), make([]int64, hhSize)
			for b := 0; b < batchSize; b++ {
				gradH, gradC := make([]float64, hiddenSize), make([]float64, hiddenSize)
				for t := seqLength - 1; t >= 0; t-- {
					nextGradH, nextGradC := make([]float64, hiddenSize), make([]float64, hiddenSize)
					it, pIdx := b*seqLength*inputSize+t*inputSize, b*seqLength*5*hiddenSize+t*5*hiddenSize
					for h := 0; h < hiddenSize; h++ {
						dh := gradH[h] + float64(gradOutput.Data[b*seqLength*hiddenSize+t*hiddenSize+h])
						iS, fS, gS, oS, cC := float64(preAct.Data[pIdx+h]), float64(preAct.Data[pIdx+hiddenSize+h]), float64(preAct.Data[pIdx+2*hiddenSize+h]), float64(preAct.Data[pIdx+3*hiddenSize+h]), float64(preAct.Data[pIdx+4*hiddenSize+h])
						iG, fG, oG := 1.0/(1.0+math.Exp(-iS)), 1.0/(1.0+math.Exp(-fS)), 1.0/(1.0+math.Exp(-oS))
						gG, cT := math.Tanh(gS), math.Tanh(cC)
						cP := 0.0; if t > 0 { cP = float64(preAct.Data[pIdx-5*hiddenSize+4*hiddenSize+h]) }
						dc := gradC[h] + dh*oG*(1.0-cT*cT)
						doP, dfP, diP, dgP := dh*cT*oG*(1.0-oG), dc*cP*fG*(1.0-fG), dc*gG*iG*(1.0-iG), dc*iG*(1.0-gG*gG)
						gBI[h] += int64(diP); gBF[h] += int64(dfP); gBG[h] += int64(dgP); gBO[h] += int64(doP)
						for i := 0; i < inputSize; i++ {
							x := float64(input.Data[it+i])
							gIHI[h*inputSize+i] += int64(diP * x); gIHF[h*inputSize+i] += int64(dfP * x); gIHG[h*inputSize+i] += int64(dgP * x); gIHO[h*inputSize+i] += int64(doP * x)
							gradInput.Data[it+i] += T(float64(wI[h*inputSize+i])*diP + float64(wF[h*inputSize+i])*dfP + float64(wG[h*inputSize+i])*dgP + float64(wO[h*inputSize+i])*doP)
						}
						for hp := 0; hp < hiddenSize; hp++ {
							hvP := 0.0; if t > 0 {
								pP := pIdx - 5*hiddenSize
								oGP := 1.0 / (1.0 + math.Exp(-float64(preAct.Data[pP+3*hiddenSize+hp])))
								hvP = oGP * math.Tanh(float64(preAct.Data[pP+4*hiddenSize+hp]))
							}
							gHHI[h*hiddenSize+hp] += int64(diP * hvP); gHHF[h*hiddenSize+hp] += int64(dfP * hvP); gHHG[h*hiddenSize+hp] += int64(dgP * hvP); gHHO[h*hiddenSize+hp] += int64(doP * hvP)
							nextGradH[hp] += float64(wI[ihSize+h*hiddenSize+hp])*diP + float64(wF[ihSize+h*hiddenSize+hp])*dfP + float64(wG[ihSize+h*hiddenSize+hp])*dgP + float64(wO[ihSize+h*hiddenSize+hp])*doP
						}
						nextGradC[h] = dc * fG
					}
					gradH, gradC = nextGradH, nextGradC
				}
			}
			for h := 0; h < hiddenSize; h++ {
				for i := 0; i < inputSize; i++ {
					gradWeights.Data[h*inputSize+i], gradWeights.Data[gateSize+h*inputSize+i], gradWeights.Data[2*gateSize+h*inputSize+i], gradWeights.Data[3*gateSize+h*inputSize+i] = T(gIHI[h*inputSize+i]), T(gIHF[h*inputSize+i]), T(gIHG[h*inputSize+i]), T(gIHO[h*inputSize+i])
				}
				for hp := 0; hp < hiddenSize; hp++ {
					gradWeights.Data[ihSize+h*hiddenSize+hp], gradWeights.Data[gateSize+ihSize+h*hiddenSize+hp], gradWeights.Data[2*gateSize+ihSize+h*hiddenSize+hp], gradWeights.Data[3*gateSize+ihSize+h*hiddenSize+hp] = T(gHHI[h*hiddenSize+hp]), T(gHHF[h*hiddenSize+hp]), T(gHHG[h*hiddenSize+hp]), T(gHHO[h*hiddenSize+hp])
				}
				gradWeights.Data[ihSize+hhSize+h], gradWeights.Data[gateSize+ihSize+hhSize+h], gradWeights.Data[2*gateSize+ihSize+hhSize+h], gradWeights.Data[3*gateSize+ihSize+hhSize+h] = T(gBI[h]), T(gBF[h]), T(gBG[h]), T(gBO[h])
			}
			return gradInput, gradWeights
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			wI, wF, wG, wO := rawW[0:gateSize], rawW[gateSize:2*gateSize], rawW[2*gateSize:3*gateSize], rawW[3*gateSize:4*gateSize]
			gBI, gBF, gBG, gBO := make([]int32, bSize), make([]int32, bSize), make([]int32, bSize), make([]int32, bSize)
			gIHI, gIHF, gIHG, gIHO := make([]int32, ihSize), make([]int32, ihSize), make([]int32, ihSize), make([]int32, ihSize)
			gHHI, gHHF, gHHG, gHHO := make([]int32, hhSize), make([]int32, hhSize), make([]int32, hhSize), make([]int32, hhSize)
			for b := 0; b < batchSize; b++ {
				gradH, gradC := make([]float64, hiddenSize), make([]float64, hiddenSize)
				for t := seqLength - 1; t >= 0; t-- {
					nextGradH, nextGradC := make([]float64, hiddenSize), make([]float64, hiddenSize)
					it, pIdx := b*seqLength*inputSize+t*inputSize, b*seqLength*5*hiddenSize+t*5*hiddenSize
					for h := 0; h < hiddenSize; h++ {
						dh := gradH[h] + float64(gradOutput.Data[b*seqLength*hiddenSize+t*hiddenSize+h])
						iS, fS, gS, oS, cC := float64(preAct.Data[pIdx+h]), float64(preAct.Data[pIdx+hiddenSize+h]), float64(preAct.Data[pIdx+2*hiddenSize+h]), float64(preAct.Data[pIdx+3*hiddenSize+h]), float64(preAct.Data[pIdx+4*hiddenSize+h])
						iG, fG, oG := 1.0/(1.0+math.Exp(-iS)), 1.0/(1.0+math.Exp(-fS)), 1.0/(1.0+math.Exp(-oS))
						gG, cT := math.Tanh(gS), math.Tanh(cC)
						cP := 0.0; if t > 0 { cP = float64(preAct.Data[pIdx-5*hiddenSize+4*hiddenSize+h]) }
						dc := gradC[h] + dh*oG*(1.0-cT*cT)
						doP, dfP, diP, dgP := dh*cT*oG*(1.0-oG), dc*cP*fG*(1.0-fG), dc*gG*iG*(1.0-iG), dc*iG*(1.0-gG*gG)
						gBI[h] += int32(diP); gBF[h] += int32(dfP); gBG[h] += int32(dgP); gBO[h] += int32(doP)
						for i := 0; i < inputSize; i++ {
							x := float64(input.Data[it+i])
							gIHI[h*inputSize+i] += int32(diP * x); gIHF[h*inputSize+i] += int32(dfP * x); gIHG[h*inputSize+i] += int32(dgP * x); gIHO[h*inputSize+i] += int32(doP * x)
							gradInput.Data[it+i] += T(float64(wI[h*inputSize+i])*diP + float64(wF[h*inputSize+i])*dfP + float64(wG[h*inputSize+i])*dgP + float64(wO[h*inputSize+i])*doP)
						}
						for hp := 0; hp < hiddenSize; hp++ {
							hvP := 0.0; if t > 0 {
								pP := pIdx - 5*hiddenSize
								oGP := 1.0 / (1.0 + math.Exp(-float64(preAct.Data[pP+3*hiddenSize+hp])))
								hvP = oGP * math.Tanh(float64(preAct.Data[pP+4*hiddenSize+hp]))
							}
							gHHI[h*hiddenSize+hp] += int32(diP * hvP); gHHF[h*hiddenSize+hp] += int32(dfP * hvP); gHHG[h*hiddenSize+hp] += int32(dgP * hvP); gHHO[h*hiddenSize+hp] += int32(doP * hvP)
							nextGradH[hp] += float64(wI[ihSize+h*hiddenSize+hp])*diP + float64(wF[ihSize+h*hiddenSize+hp])*dfP + float64(wG[ihSize+h*hiddenSize+hp])*dgP + float64(wO[ihSize+h*hiddenSize+hp])*doP
						}
						nextGradC[h] = dc * fG
					}
					gradH, gradC = nextGradH, nextGradC
				}
			}
			for h := 0; h < hiddenSize; h++ {
				for i := 0; i < inputSize; i++ {
					gradWeights.Data[h*inputSize+i], gradWeights.Data[gateSize+h*inputSize+i], gradWeights.Data[2*gateSize+h*inputSize+i], gradWeights.Data[3*gateSize+h*inputSize+i] = T(gIHI[h*inputSize+i]), T(gIHF[h*inputSize+i]), T(gIHG[h*inputSize+i]), T(gIHO[h*inputSize+i])
				}
				for hp := 0; hp < hiddenSize; hp++ {
					gradWeights.Data[ihSize+h*hiddenSize+hp], gradWeights.Data[gateSize+ihSize+h*hiddenSize+hp], gradWeights.Data[2*gateSize+ihSize+h*hiddenSize+hp], gradWeights.Data[3*gateSize+ihSize+h*hiddenSize+hp] = T(gHHI[h*hiddenSize+hp]), T(gHHF[h*hiddenSize+hp]), T(gHHG[h*hiddenSize+hp]), T(gHHO[h*hiddenSize+hp])
				}
				gradWeights.Data[ihSize+hhSize+h], gradWeights.Data[gateSize+ihSize+hhSize+h], gradWeights.Data[2*gateSize+ihSize+hhSize+h], gradWeights.Data[3*gateSize+ihSize+hhSize+h] = T(gBI[h]), T(gBF[h]), T(gBG[h]), T(gBO[h])
			}
			return gradInput, gradWeights
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			wI, wF, wG, wO := rawW[0:gateSize], rawW[gateSize:2*gateSize], rawW[2*gateSize:3*gateSize], rawW[3*gateSize:4*gateSize]
			gBI, gBF, gBG, gBO := make([]int32, bSize), make([]int32, bSize), make([]int32, bSize), make([]int32, bSize)
			gIHI, gIHF, gIHG, gIHO := make([]int32, ihSize), make([]int32, ihSize), make([]int32, ihSize), make([]int32, ihSize)
			gHHI, gHHF, gHHG, gHHO := make([]int32, hhSize), make([]int32, hhSize), make([]int32, hhSize), make([]int32, hhSize)
			for b := 0; b < batchSize; b++ {
				gradH, gradC := make([]float64, hiddenSize), make([]float64, hiddenSize)
				for t := seqLength - 1; t >= 0; t-- {
					nextGradH, nextGradC := make([]float64, hiddenSize), make([]float64, hiddenSize)
					it, pIdx := b*seqLength*inputSize+t*inputSize, b*seqLength*5*hiddenSize+t*5*hiddenSize
					for h := 0; h < hiddenSize; h++ {
						dh := gradH[h] + float64(gradOutput.Data[b*seqLength*hiddenSize+t*hiddenSize+h])
						iS, fS, gS, oS, cC := float64(preAct.Data[pIdx+h]), float64(preAct.Data[pIdx+hiddenSize+h]), float64(preAct.Data[pIdx+2*hiddenSize+h]), float64(preAct.Data[pIdx+3*hiddenSize+h]), float64(preAct.Data[pIdx+4*hiddenSize+h])
						iG, fG, oG := 1.0/(1.0+math.Exp(-iS)), 1.0/(1.0+math.Exp(-fS)), 1.0/(1.0+math.Exp(-oS))
						gG, cT := math.Tanh(gS), math.Tanh(cC)
						cP := 0.0; if t > 0 { cP = float64(preAct.Data[pIdx-5*hiddenSize+4*hiddenSize+h]) }
						dc := gradC[h] + dh*oG*(1.0-cT*cT)
						doP, dfP, diP, dgP := dh*cT*oG*(1.0-oG), dc*cP*fG*(1.0-fG), dc*gG*iG*(1.0-iG), dc*iG*(1.0-gG*gG)
						gBI[h] += int32(diP); gBF[h] += int32(dfP); gBG[h] += int32(dgP); gBO[h] += int32(doP)
						for i := 0; i < inputSize; i++ {
							x := float64(input.Data[it+i])
							gIHI[h*inputSize+i] += int32(diP * x); gIHF[h*inputSize+i] += int32(dfP * x); gIHG[h*inputSize+i] += int32(dgP * x); gIHO[h*inputSize+i] += int32(doP * x)
							gradInput.Data[it+i] += T(float64(wI[h*inputSize+i])*diP + float64(wF[h*inputSize+i])*dfP + float64(wG[h*inputSize+i])*dgP + float64(wO[h*inputSize+i])*doP)
						}
						for hp := 0; hp < hiddenSize; hp++ {
							hvP := 0.0; if t > 0 {
								pP := pIdx - 5*hiddenSize
								oGP := 1.0 / (1.0 + math.Exp(-float64(preAct.Data[pP+3*hiddenSize+hp])))
								hvP = oGP * math.Tanh(float64(preAct.Data[pP+4*hiddenSize+hp]))
							}
							gHHI[h*hiddenSize+hp] += int32(diP * hvP); gHHF[h*hiddenSize+hp] += int32(dfP * hvP); gHHG[h*hiddenSize+hp] += int32(dgP * hvP); gHHO[h*hiddenSize+hp] += int32(doP * hvP)
							nextGradH[hp] += float64(wI[ihSize+h*hiddenSize+hp])*diP + float64(wF[ihSize+h*hiddenSize+hp])*dfP + float64(wG[ihSize+h*hiddenSize+hp])*dgP + float64(wO[ihSize+h*hiddenSize+hp])*doP
						}
						nextGradC[h] = dc * fG
					}
					gradH, gradC = nextGradH, nextGradC
				}
			}
			for h := 0; h < hiddenSize; h++ {
				for i := 0; i < inputSize; i++ {
					gradWeights.Data[h*inputSize+i], gradWeights.Data[gateSize+h*inputSize+i], gradWeights.Data[2*gateSize+h*inputSize+i], gradWeights.Data[3*gateSize+h*inputSize+i] = T(gIHI[h*inputSize+i]), T(gIHF[h*inputSize+i]), T(gIHG[h*inputSize+i]), T(gIHO[h*inputSize+i])
				}
				for hp := 0; hp < hiddenSize; hp++ {
					gradWeights.Data[ihSize+h*hiddenSize+hp], gradWeights.Data[gateSize+ihSize+h*hiddenSize+hp], gradWeights.Data[2*gateSize+ihSize+h*hiddenSize+hp], gradWeights.Data[3*gateSize+ihSize+h*hiddenSize+hp] = T(gHHI[h*hiddenSize+hp]), T(gHHF[h*hiddenSize+hp]), T(gHHG[h*hiddenSize+hp]), T(gHHO[h*hiddenSize+hp])
				}
				gradWeights.Data[ihSize+hhSize+h], gradWeights.Data[gateSize+ihSize+hhSize+h], gradWeights.Data[2*gateSize+ihSize+hhSize+h], gradWeights.Data[3*gateSize+ihSize+hhSize+h] = T(gBI[h]), T(gBF[h]), T(gBG[h]), T(gBO[h])
			}
			return gradInput, gradWeights
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			wI, wF, wG, wO := rawW[0:gateSize], rawW[gateSize:2*gateSize], rawW[2*gateSize:3*gateSize], rawW[3*gateSize:4*gateSize]
			gBI, gBF, gBG, gBO := make([]int32, bSize), make([]int32, bSize), make([]int32, bSize), make([]int32, bSize)
			gIHI, gIHF, gIHG, gIHO := make([]int32, ihSize), make([]int32, ihSize), make([]int32, ihSize), make([]int32, ihSize)
			gHHI, gHHF, gHHG, gHHO := make([]int32, hhSize), make([]int32, hhSize), make([]int32, hhSize), make([]int32, hhSize)
			for b := 0; b < batchSize; b++ {
				gradH, gradC := make([]float64, hiddenSize), make([]float64, hiddenSize)
				for t := seqLength - 1; t >= 0; t-- {
					nextGradH, nextGradC := make([]float64, hiddenSize), make([]float64, hiddenSize)
					it, pIdx := b*seqLength*inputSize+t*inputSize, b*seqLength*5*hiddenSize+t*5*hiddenSize
					for h := 0; h < hiddenSize; h++ {
						dh := gradH[h] + float64(gradOutput.Data[b*seqLength*hiddenSize+t*hiddenSize+h])
						iS, fS, gS, oS, cC := float64(preAct.Data[pIdx+h]), float64(preAct.Data[pIdx+hiddenSize+h]), float64(preAct.Data[pIdx+2*hiddenSize+h]), float64(preAct.Data[pIdx+3*hiddenSize+h]), float64(preAct.Data[pIdx+4*hiddenSize+h])
						iG, fG, oG := 1.0/(1.0+math.Exp(-iS)), 1.0/(1.0+math.Exp(-fS)), 1.0/(1.0+math.Exp(-oS))
						gG, cT := math.Tanh(gS), math.Tanh(cC)
						cP := 0.0; if t > 0 { cP = float64(preAct.Data[pIdx-5*hiddenSize+4*hiddenSize+h]) }
						dc := gradC[h] + dh*oG*(1.0-cT*cT)
						doP, dfP, diP, dgP := dh*cT*oG*(1.0-oG), dc*cP*fG*(1.0-fG), dc*gG*iG*(1.0-iG), dc*iG*(1.0-gG*gG)
						gBI[h] += int32(diP); gBF[h] += int32(dfP); gBG[h] += int32(dgP); gBO[h] += int32(doP)
						for i := 0; i < inputSize; i++ {
							x := float64(input.Data[it+i])
							gIHI[h*inputSize+i] += int32(diP * x); gIHF[h*inputSize+i] += int32(dfP * x); gIHG[h*inputSize+i] += int32(dgP * x); gIHO[h*inputSize+i] += int32(doP * x)
							gradInput.Data[it+i] += T(float64(wI[h*inputSize+i])*diP + float64(wF[h*inputSize+i])*dfP + float64(wG[h*inputSize+i])*dgP + float64(wO[h*inputSize+i])*doP)
						}
						for hp := 0; hp < hiddenSize; hp++ {
							hvP := 0.0; if t > 0 {
								pP := pIdx - 5*hiddenSize
								oGP := 1.0 / (1.0 + math.Exp(-float64(preAct.Data[pP+3*hiddenSize+hp])))
								hvP = oGP * math.Tanh(float64(preAct.Data[pP+4*hiddenSize+hp]))
							}
							gHHI[h*hiddenSize+hp] += int32(diP * hvP); gHHF[h*hiddenSize+hp] += int32(dfP * hvP); gHHG[h*hiddenSize+hp] += int32(dgP * hvP); gHHO[h*hiddenSize+hp] += int32(doP * hvP)
							nextGradH[hp] += float64(wI[ihSize+h*hiddenSize+hp])*diP + float64(wF[ihSize+h*hiddenSize+hp])*dfP + float64(wG[ihSize+h*hiddenSize+hp])*dgP + float64(wO[ihSize+h*hiddenSize+hp])*doP
						}
						nextGradC[h] = dc * fG
					}
					gradH, gradC = nextGradH, nextGradC
				}
			}
			for h := 0; h < hiddenSize; h++ {
				for i := 0; i < inputSize; i++ {
					gradWeights.Data[h*inputSize+i], gradWeights.Data[gateSize+h*inputSize+i], gradWeights.Data[2*gateSize+h*inputSize+i], gradWeights.Data[3*gateSize+h*inputSize+i] = T(gIHI[h*inputSize+i]), T(gIHF[h*inputSize+i]), T(gIHG[h*inputSize+i]), T(gIHO[h*inputSize+i])
				}
				for hp := 0; hp < hiddenSize; hp++ {
					gradWeights.Data[ihSize+h*hiddenSize+hp], gradWeights.Data[gateSize+ihSize+h*hiddenSize+hp], gradWeights.Data[2*gateSize+ihSize+h*hiddenSize+hp], gradWeights.Data[3*gateSize+ihSize+h*hiddenSize+hp] = T(gHHI[h*hiddenSize+hp]), T(gHHF[h*hiddenSize+hp]), T(gHHG[h*hiddenSize+hp]), T(gHHO[h*hiddenSize+hp])
				}
				gradWeights.Data[ihSize+hhSize+h], gradWeights.Data[gateSize+ihSize+hhSize+h], gradWeights.Data[2*gateSize+ihSize+hhSize+h], gradWeights.Data[3*gateSize+ihSize+hhSize+h] = T(gBI[h]), T(gBF[h]), T(gBG[h]), T(gBO[h])
			}
			return gradInput, gradWeights
		}
	}

	// UNIVERSAL POLYMORPHIC FALLTHROUGH (Simulation logic for low-bit/QAT)
	wData := CastWeights[T](weights)
	wI, wF, wG, wO := wData[0:gateSize], wData[gateSize:2*gateSize], wData[2*gateSize:3*gateSize], wData[3*gateSize:4*gateSize]
	gBI, gBF, gBG, gBO := make([]float32, bSize), make([]float32, bSize), make([]float32, bSize), make([]float32, bSize)
	gIHI, gIHF, gIHG, gIHO := make([]float32, ihSize), make([]float32, ihSize), make([]float32, ihSize), make([]float32, ihSize)
	gHHI, gHHF, gHHG, gHHO := make([]float32, hhSize), make([]float32, hhSize), make([]float32, hhSize), make([]float32, hhSize)
	for b := 0; b < batchSize; b++ {
		gradH, gradC := make([]float32, hiddenSize), make([]float32, hiddenSize)
		for t := seqLength - 1; t >= 0; t-- {
			nextGradH, nextGradC := make([]float32, hiddenSize), make([]float32, hiddenSize)
			it, pIdx := b*seqLength*inputSize+t*inputSize, b*seqLength*5*hiddenSize+t*5*hiddenSize
			for h := 0; h < hiddenSize; h++ {
				dh := gradH[h] + float32(gradOutput.Data[b*seqLength*hiddenSize+t*hiddenSize+h])
				iS, fS, gS, oS, cC := float32(preAct.Data[pIdx+h]), float32(preAct.Data[pIdx+hiddenSize+h]), float32(preAct.Data[pIdx+2*hiddenSize+h]), float32(preAct.Data[pIdx+3*hiddenSize+h]), float32(preAct.Data[pIdx+4*hiddenSize+h])
				iG, fG, oG := 1.0/(1.0+float32(math.Exp(-float64(iS)))), 1.0/(1.0+float32(math.Exp(-float64(fS)))), 1.0/(1.0+float32(math.Exp(-float64(oS))))
				gG, cT := float32(math.Tanh(float64(gS))), float32(math.Tanh(float64(cC)))
				cP := float32(0); if t > 0 { cP = float32(preAct.Data[pIdx-5*hiddenSize+4*hiddenSize+h]) }
				dc := gradC[h] + dh*oG*(1.0-cT*cT)
				doP, dfP, diP, dgP := dh*cT*oG*(1.0-oG), dc*cP*fG*(1.0-fG), dc*gG*iG*(1.0-iG), dc*iG*(1.0-gG*gG)
				gBI[h] += diP; gBF[h] += dfP; gBG[h] += dgP; gBO[h] += doP 
				for i := 0; i < inputSize; i++ {
					x := float32(input.Data[it+i])
					sWI := SimulatePrecision(float32(wI[h*inputSize+i]), layer.DType, scale)
					sWF := SimulatePrecision(float32(wF[h*inputSize+i]), layer.DType, scale)
					sWG := SimulatePrecision(float32(wG[h*inputSize+i]), layer.DType, scale)
					sWO := SimulatePrecision(float32(wO[h*inputSize+i]), layer.DType, scale)
					gIHI[h*inputSize+i] += diP * x; gIHF[h*inputSize+i] += dfP * x; gIHG[h*inputSize+i] += dgP * x; gIHO[h*inputSize+i] += doP * x
					gradInput.Data[it+i] += T(sWI*diP + sWF*dfP + sWG*dgP + sWO*doP)
				}
				for hp := 0; hp < hiddenSize; hp++ {
					hvP := float32(0); if t > 0 {
						pP := pIdx - 5*hiddenSize
						oGP := 1.0 / (1.0 + float32(math.Exp(-float64(float32(preAct.Data[pP+3*hiddenSize+hp])))))
						hvP = oGP * float32(math.Tanh(float64(float32(preAct.Data[pP+4*hiddenSize+hp]))))
					}
					sWHi := SimulatePrecision(float32(wI[ihSize+h*hiddenSize+hp]), layer.DType, scale)
					sWHf := SimulatePrecision(float32(wF[ihSize+h*hiddenSize+hp]), layer.DType, scale)
					sWHg := SimulatePrecision(float32(wG[ihSize+h*hiddenSize+hp]), layer.DType, scale)
					sWHo := SimulatePrecision(float32(wO[ihSize+h*hiddenSize+hp]), layer.DType, scale)
					gHHI[h*hiddenSize+hp] += diP * hvP; gHHF[h*hiddenSize+hp] += dfP * hvP; gHHG[h*hiddenSize+hp] += dgP * hvP; gHHO[h*hiddenSize+hp] += doP * hvP
					nextGradH[hp] += sWHi*diP + sWHf*dfP + sWHg*dgP + sWHo*doP
				}
				nextGradC[h] = dc * fG
			}
			gradH, gradC = nextGradH, nextGradC
		}
	}
	for h := 0; h < hiddenSize; h++ {
		for i := 0; i < inputSize; i++ {
			gradWeights.Data[h*inputSize+i], gradWeights.Data[gateSize+h*inputSize+i], gradWeights.Data[2*gateSize+h*inputSize+i], gradWeights.Data[3*gateSize+h*inputSize+i] = T(gIHI[h*inputSize+i]), T(gIHF[h*inputSize+i]), T(gIHG[h*inputSize+i]), T(gIHO[h*inputSize+i])
		}
		for hp := 0; hp < hiddenSize; hp++ {
			gradWeights.Data[ihSize+h*hiddenSize+hp], gradWeights.Data[gateSize+ihSize+h*hiddenSize+hp], gradWeights.Data[2*gateSize+ihSize+h*hiddenSize+hp], gradWeights.Data[3*gateSize+ihSize+h*hiddenSize+hp] = T(gHHI[h*hiddenSize+hp]), T(gHHF[h*hiddenSize+hp]), T(gHHG[h*hiddenSize+hp]), T(gHHO[h*hiddenSize+hp])
		}
		gradWeights.Data[ihSize+hhSize+h], gradWeights.Data[gateSize+ihSize+hhSize+h], gradWeights.Data[2*gateSize+ihSize+hhSize+h], gradWeights.Data[3*gateSize+ihSize+hhSize+h] = T(gBI[h]), T(gBF[h]), T(gBG[h]), T(gBO[h])
	}
	return gradInput, gradWeights
}

func double(v float32) float64 { return float64(v) }
