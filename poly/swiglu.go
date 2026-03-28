package poly

import (
	"math"
	"runtime"
	"sync"
)

// SwiGLUForwardPolymorphic performs a forward pass through a SwiGLU layer.
func SwiGLUForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if layer.UseGPU {
		return SwiGLUForwardWGPU(layer, input)
	}
	if layer.UseTiling {
		return SwiGLUForwardTiled(layer, input)
	}
	inputSize, intermediateSize := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inputSize
	
	// preAct stores [siluGate * up] intermediates for backward pass
	preAct = NewTensor[T](seqLen, intermediateSize)
	postAct = NewTensor[T](seqLen, inputSize)
	
	
	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }
	
	gateWStart := 0
	upWStart := inputSize * intermediateSize
	downWStart := 2 * inputSize * intermediateSize
	
	gateBStart := 2 * inputSize * intermediateSize + intermediateSize * inputSize
	upBStart := gateBStart + intermediateSize
	downBStart := upBStart + intermediateSize

	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			gateW := rawW[gateWStart : gateWStart + inputSize * intermediateSize]
			upW   := rawW[upWStart : upWStart + inputSize * intermediateSize]
			downW := rawW[downWStart : downWStart + intermediateSize * inputSize]
			
			gateB := rawW[gateBStart : gateBStart + intermediateSize]
			upB   := rawW[upBStart : upBStart + intermediateSize]
			downB := rawW[downBStart : downBStart + inputSize]

			for s := 0; s < seqLen; s++ {
				for i := 0; i < intermediateSize; i++ {
					var sumGate, sumUp float64
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						sumGate += inVal * float64(gateW[i*inputSize+j])
						sumUp   += inVal * float64(upW[i*inputSize+j])
					}
					sumGate += float64(gateB[i])
					sumUp   += float64(upB[i])
					
					sig := 1.0 / (1.0 + math.Exp(-float64(sumGate)))
					silu := float64(float64(sumGate) * sig)
					preAct.Data[s*intermediateSize+i] = T(silu * sumUp)
				}
			}
			
			for s := 0; s < seqLen; s++ {
				for i := 0; i < inputSize; i++ {
					var sum float64
					for j := 0; j < intermediateSize; j++ {
						sum += float64(preAct.Data[s*intermediateSize+j]) * float64(downW[i*intermediateSize+j])
					}
					sum += float64(downB[i])
					postAct.Data[s*inputSize+i] = T(sum)
				}
			}
			return preAct, postAct
		}
	case DTypeFloat32, DTypeFloat16, DTypeBFloat16:
		if rawW, ok := weights.([]float32); ok {
			gateW := rawW[gateWStart : gateWStart + inputSize * intermediateSize]
			upW   := rawW[upWStart : upWStart + inputSize * intermediateSize]
			downW := rawW[downWStart : downWStart + intermediateSize * inputSize]
			
			gateB := rawW[gateBStart : gateBStart + intermediateSize]
			upB   := rawW[upBStart : upBStart + intermediateSize]
			downB := rawW[downBStart : downBStart + inputSize]

			for s := 0; s < seqLen; s++ {
				for i := 0; i < intermediateSize; i++ {
					var sumGate, sumUp float32
					for j := 0; j < inputSize; j++ {
						inVal := float32(input.Data[s*inputSize+j])
						sumGate += inVal * float32(gateW[i*inputSize+j])
						sumUp   += inVal * float32(upW[i*inputSize+j])
					}
					sumGate += float32(gateB[i])
					sumUp   += float32(upB[i])
					
					sig := 1.0 / (1.0 + math.Exp(-float64(sumGate)))
					silu := float32(float64(sumGate) * sig)
					preAct.Data[s*intermediateSize+i] = T(silu * sumUp)
				}
			}
			
			for s := 0; s < seqLen; s++ {
				for i := 0; i < inputSize; i++ {
					var sum float32
					for j := 0; j < intermediateSize; j++ {
						sum += float32(preAct.Data[s*intermediateSize+j]) * float32(downW[i*intermediateSize+j])
					}
					sum += float32(downB[i])
					postAct.Data[s*inputSize+i] = T(sum)
				}
			}
			return preAct, postAct
		}
	case DTypeInt64, DTypeUint64, DTypeInt32, DTypeUint32, DTypeInt16, DTypeUint16, DTypeInt8, DTypeUint8, DTypeFP8E4M3, DTypeFP8E5M2, DTypeInt4, DTypeUint4, DTypeFP4, DTypeInt2, DTypeUint2, DTypeTernary, DTypeBinary:
		scale := float64(layer.WeightStore.Scale)
		if scale == 0 { scale = 1.0 }
		
		wData := CastWeights[float64](weights)
		gateW := wData[gateWStart : gateWStart + inputSize * intermediateSize]
		upW   := wData[upWStart : upWStart + inputSize * intermediateSize]
		downW := wData[downWStart : downWStart + intermediateSize * inputSize]
		
		gateB := wData[gateBStart : gateBStart + intermediateSize]
		upB   := wData[upBStart : upBStart + intermediateSize]
		downB := wData[downBStart : downBStart + inputSize]

		for s := 0; s < seqLen; s++ {
			for i := 0; i < intermediateSize; i++ {
				var sumGate, sumUp float64
				for j := 0; j < inputSize; j++ {
					inVal := float64(input.Data[s*inputSize+j])
					sumGate += inVal * gateW[i*inputSize+j] * scale
					sumUp   += inVal * upW[i*inputSize+j] * scale
				}
				sumGate += gateB[i] * scale
				sumUp   += upB[i] * scale
				
				sig := 1.0 / (1.0 + math.Exp(-sumGate))
				silu := sumGate * sig
				preAct.Data[s*intermediateSize+i] = T(silu * sumUp)
			}
		}
		
		for s := 0; s < seqLen; s++ {
			for i := 0; i < inputSize; i++ {
				var sum float64
				for j := 0; j < intermediateSize; j++ {
					sum += float64(preAct.Data[s*intermediateSize+j]) * downW[i*intermediateSize+j] * scale
				}
				sum += downB[i] * scale
				postAct.Data[s*inputSize+i] = T(sum)
			}
		}
		return preAct, postAct
	}

	scale := float32(layer.WeightStore.Scale)
	if scale == 0 { scale = 1.0 }
	wData := CastWeights[float32](weights)
	
	gateW := wData[gateWStart : gateWStart + inputSize * intermediateSize]
	upW   := wData[upWStart : upWStart + inputSize * intermediateSize]
	downW := wData[downWStart : downWStart + intermediateSize * inputSize]
	
	gateB := wData[gateBStart : gateBStart + intermediateSize]
	upB   := wData[upBStart : upBStart + intermediateSize]
	downB := wData[downBStart : downBStart + inputSize]

	for s := 0; s < seqLen; s++ {
		for i := 0; i < intermediateSize; i++ {
			var sumGate, sumUp float32
			for j := 0; j < inputSize; j++ {
				inVal := float32(input.Data[s*inputSize+j])
				sumGate += inVal * SimulatePrecision(gateW[i*inputSize+j], layer.DType, scale)
				sumUp   += inVal * SimulatePrecision(upW[i*inputSize+j], layer.DType, scale)
			}
			sumGate += SimulatePrecision(gateB[i], layer.DType, scale)
			sumUp   += SimulatePrecision(upB[i], layer.DType, scale)
			
			sig := float32(1.0 / (1.0 + math.Exp(-float64(sumGate))))
			silu := sumGate * sig
			preAct.Data[s*intermediateSize+i] = T(silu * sumUp)
		}
	}
	
	for s := 0; s < seqLen; s++ {
		for i := 0; i < inputSize; i++ {
			var sum float32
			for j := 0; j < intermediateSize; j++ {
				sum += float32(preAct.Data[s*intermediateSize+j]) * SimulatePrecision(downW[i*intermediateSize+j], layer.DType, scale)
			}
			sum += SimulatePrecision(downB[i], layer.DType, scale)
			postAct.Data[s*inputSize+i] = T(sum)
		}
	}

	return preAct, postAct
}

// SwiGLUBackwardPolymorphic calculates gradients for the SwiGLU layer.
func SwiGLUBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if layer.UseGPU {
		return SwiGLUBackwardWGPU(layer, gradOutput, input, preAct)
	}
	if layer.UseTiling {
		return SwiGLUBackwardTiled(layer, gradOutput, input, preAct)
	}

	inputSize, intermediateSize := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inputSize
	
	gradInput = NewTensor[T](input.Shape...)
	gradWeights = NewTensor[T](len(layer.WeightStore.Master))
	
	

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }
	
	gateWStart := 0
	upWStart := inputSize * intermediateSize
	downWStart := 2 * inputSize * intermediateSize
	
	gateBStart := 2 * inputSize * intermediateSize + intermediateSize * inputSize
	upBStart := gateBStart + intermediateSize
	downBStart := upBStart + intermediateSize

	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			gateW := rawW[gateWStart : gateWStart + inputSize * intermediateSize]
			upW   := rawW[upWStart : upWStart + inputSize * intermediateSize]
			downW := rawW[downWStart : downWStart + intermediateSize * inputSize]
			
			gateB := rawW[gateBStart : gateBStart + intermediateSize]
			upB   := rawW[upBStart : upBStart + intermediateSize]

			for s := 0; s < seqLen; s++ {
				gradInterRow := make([]float64, intermediateSize)
				for i := 0; i < inputSize; i++ {
					dy := float64(gradOutput.Data[s*inputSize+i])
					gradWeights.Data[downBStart+i] += T(dy)
					for j := 0; j < intermediateSize; j++ {
						act := float64(preAct.Data[s*intermediateSize+j])
						gradWeights.Data[downWStart + j*inputSize+i] += T(act * dy)
						gradInterRow[j] += dy * float64(downW[j*inputSize+i])
					}
				}
				
				for i := 0; i < intermediateSize; i++ {
					var sumGate, sumUp float64
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						sumGate += inVal * float64(gateW[i*inputSize+j])
						sumUp   += inVal * float64(upW[i*inputSize+j])
					}
					sumGate += float64(gateB[i])
					sumUp   += float64(upB[i])
					
					x := float64(sumGate)
					sig := 1.0 / (1.0 + math.Exp(-x))
					silu := x * sig
					dSilu := sig * (1.0 + x*(1.0-sig))
					
					gi := gradInterRow[i]
					dUp := gi * silu
					dGate := gi * float64(sumUp) * dSilu
					
					gradWeights.Data[upBStart+i] += T(dUp)
					gradWeights.Data[gateBStart+i] += T(dGate)
					
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						gradWeights.Data[upWStart + j*intermediateSize+i] += T(inVal * dUp)
						gradWeights.Data[gateWStart + j*intermediateSize+i] += T(inVal * dGate)
						
						gradInput.Data[s*inputSize+j] += T(dUp * float64(upW[j*intermediateSize+i]) + dGate * float64(gateW[j*intermediateSize+i]))
					}
				}
			}
			return gradInput, gradWeights
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			gateW := rawW[gateWStart : gateWStart + inputSize * intermediateSize]
			upW   := rawW[upWStart : upWStart + inputSize * intermediateSize]
			downW := rawW[downWStart : downWStart + intermediateSize * inputSize]
			
			gateB := rawW[gateBStart : gateBStart + intermediateSize]
			upB   := rawW[upBStart : upBStart + intermediateSize]

			for s := 0; s < seqLen; s++ {
				gradInterRow := make([]float64, intermediateSize)
				for i := 0; i < inputSize; i++ {
					dy := float64(gradOutput.Data[s*inputSize+i])
					gradWeights.Data[downBStart+i] += T(dy)
					for j := 0; j < intermediateSize; j++ {
						act := float64(preAct.Data[s*intermediateSize+j])
						gradWeights.Data[downWStart + j*inputSize+i] += T(act * dy)
						gradInterRow[j] += dy * float64(downW[j*inputSize+i])
					}
				}
				
				for i := 0; i < intermediateSize; i++ {
					var sumGate, sumUp float32
					for j := 0; j < inputSize; j++ {
						inVal := float32(input.Data[s*inputSize+j])
						sumGate += inVal * float32(gateW[i*inputSize+j])
						sumUp   += inVal * float32(upW[i*inputSize+j])
					}
					sumGate += float32(gateB[i])
					sumUp   += float32(upB[i])
					
					x := float64(sumGate)
					sig := 1.0 / (1.0 + math.Exp(-x))
					silu := x * sig
					dSilu := sig * (1.0 + x*(1.0-sig))
					
					gi := gradInterRow[i]
					dUp := gi * silu
					dGate := gi * float64(sumUp) * dSilu
					
					gradWeights.Data[upBStart+i] += T(dUp)
					gradWeights.Data[gateBStart+i] += T(dGate)
					
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						gradWeights.Data[upWStart + j*intermediateSize+i] += T(inVal * dUp)
						gradWeights.Data[gateWStart + j*intermediateSize+i] += T(inVal * dGate)
						
						gradInput.Data[s*inputSize+j] += T(dUp * float64(upW[j*intermediateSize+i]) + dGate * float64(gateW[j*intermediateSize+i]))
					}
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt64, DTypeUint64, DTypeInt32, DTypeUint32, DTypeInt16, DTypeUint16, DTypeInt8, DTypeUint8, DTypeFP8E4M3, DTypeFP8E5M2, DTypeInt4, DTypeUint4, DTypeFP4, DTypeInt2, DTypeUint2, DTypeTernary, DTypeBinary:
		scale := float64(layer.WeightStore.Scale)
		if scale == 0 { scale = 1.0 }
		
		wData := CastWeights[float64](weights)
		gateW := wData[gateWStart : gateWStart + inputSize * intermediateSize]
		upW   := wData[upWStart : upWStart + inputSize * intermediateSize]
		downW := wData[downWStart : downWStart + intermediateSize * inputSize]
		
		gateB := wData[gateBStart : gateBStart + intermediateSize]
		upB   := wData[upBStart : upBStart + intermediateSize]

		for s := 0; s < seqLen; s++ {
			gradInterRow := make([]float64, intermediateSize)
			for i := 0; i < inputSize; i++ {
				dy := float64(gradOutput.Data[s*inputSize+i])
				gradWeights.Data[downBStart+i] += T(dy)
				for j := 0; j < intermediateSize; j++ {
					act := float64(preAct.Data[s*intermediateSize+j])
					gradWeights.Data[downWStart + i*intermediateSize+j] += T(act * dy)
					gradInterRow[j] += dy * downW[i*intermediateSize+j] * scale
				}
			}
			
			for i := 0; i < intermediateSize; i++ {
				var sumGate, sumUp float64
				for j := 0; j < inputSize; j++ {
					inVal := float64(input.Data[s*inputSize+j])
					sumGate += inVal * gateW[i*inputSize+j] * scale
					sumUp   += inVal * upW[i*inputSize+j] * scale
				}
				sumGate += gateB[i]
				sumUp   += upB[i]
				
				x := sumGate
				sig := 1.0 / (1.0 + math.Exp(-x))
				silu := x * sig
				dSilu := sig * (1.0 + x*(1.0-sig))
				
				gi := gradInterRow[i]
				dUp := gi * silu
				dGate := gi * sumUp * dSilu
				
				gradWeights.Data[upBStart+i] += T(dUp)
				gradWeights.Data[gateBStart+i] += T(dGate)
				
				for j := 0; j < inputSize; j++ {
					inVal := float64(input.Data[s*inputSize+j])
					gradWeights.Data[upWStart + i*inputSize+j] += T(inVal * dUp)
					gradWeights.Data[gateWStart + i*inputSize+j] += T(inVal * dGate)
					
					gradInput.Data[s*inputSize+j] += T(dUp * upW[i*inputSize+j] * scale + dGate * gateW[i*inputSize+j] * scale)
				}
			}
		}
		return gradInput, gradWeights
	}

	scale := layer.WeightStore.Scale
	if scale == 0 { scale = 1.0 }
	wData := CastWeights[float32](weights)

	gateW := wData[gateWStart : gateWStart + inputSize * intermediateSize]
	upW   := wData[upWStart : upWStart + inputSize * intermediateSize]
	downW := wData[downWStart : downWStart + intermediateSize * inputSize]
	
	gateB := wData[gateBStart : gateBStart + intermediateSize]
	upB   := wData[upBStart : upBStart + intermediateSize]

	for s := 0; s < seqLen; s++ {
		gradInterRow := make([]float64, intermediateSize)
		for i := 0; i < inputSize; i++ {
			dy := float64(gradOutput.Data[s*inputSize+i])
			gradWeights.Data[downBStart+i] += T(dy)
			for j := 0; j < intermediateSize; j++ {
				act := float64(preAct.Data[s*intermediateSize+j])
				gradWeights.Data[downWStart + i*intermediateSize+j] += T(act * dy)
				gradInterRow[j] += dy * float64(SimulatePrecision(downW[i*intermediateSize+j], layer.DType, scale))
			}
		}
		
		for i := 0; i < intermediateSize; i++ {
			var sumGate, sumUp float32
			for j := 0; j < inputSize; j++ {
				inVal := float32(input.Data[s*inputSize+j])
				sumGate += inVal * SimulatePrecision(gateW[i*inputSize+j], layer.DType, scale)
				sumUp   += inVal * SimulatePrecision(upW[i*inputSize+j], layer.DType, scale)
			}
			sumGate += SimulatePrecision(gateB[i], layer.DType, scale)
			sumUp   += SimulatePrecision(upB[i], layer.DType, scale)
			
			x := float64(sumGate)
			sig := 1.0 / (1.0 + math.Exp(-x))
			silu := x * sig
			dSilu := sig * (1.0 + x*(1.0-sig))
			
			gi := gradInterRow[i]
			dUp := gi * silu
			dGate := gi * float64(sumUp) * dSilu
			
			gradWeights.Data[upBStart+i] += T(dUp)
			gradWeights.Data[gateBStart+i] += T(dGate)
			
			for j := 0; j < inputSize; j++ {
				inVal := float64(input.Data[s*inputSize+j])
				gradWeights.Data[upWStart + i*inputSize+j] += T(inVal * dUp)
				gradWeights.Data[gateWStart + i*inputSize+j] += T(inVal * dGate)
				
				gradInput.Data[s*inputSize+j] += T(dUp * float64(SimulatePrecision(upW[i*inputSize+j], layer.DType, scale)) + 
												  dGate * float64(SimulatePrecision(gateW[i*inputSize+j], layer.DType, scale)))
			}
		}
	}

	return gradInput, gradWeights
}

// SwiGLUForwardTiled performs an optimized, tiled forward pass for SwiGLU.
// Branches to the multi-core parallel path when layer.EnableMultiCoreTiling is set.
func SwiGLUForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if layer.EnableMultiCoreTiling {
		return swigluForwardTiledParallel(layer, input)
	}
	inputSize, intermediateSize := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inputSize
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 { tileSize = 32 }

	preAct = NewTensor[T](seqLen, intermediateSize)
	postAct = NewTensor[T](seqLen, inputSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }
	
	gateWStart := 0
	upWStart := inputSize * intermediateSize
	downWStart := 2 * inputSize * intermediateSize
	gateBStart := 2 * inputSize * intermediateSize + intermediateSize * inputSize
	upBStart := gateBStart + intermediateSize
	downBStart := upBStart + intermediateSize

	scaleW := layer.WeightStore.Scale
	if scaleW == 0 { scaleW = 1.0 }

	// Tiled Gate & Up Projections
	swigluTiledProjectGateUp(input.Data, weights, gateWStart, upWStart, gateBStart, upBStart, preAct.Data, inputSize, intermediateSize, seqLen, layer.DType, scaleW, tileSize)

	// Tiled Down Projection
	swigluTiledProjectDown(preAct.Data, weights, downWStart, downBStart, postAct.Data, intermediateSize, inputSize, seqLen, layer.DType, scaleW, tileSize)

	return preAct, postAct
}

func swigluTiledProjectGateUp[TIn Numeric, TOut Numeric](input []TIn, weights any, gWStart, uWStart, gBStart, uBStart int, output []TOut, inDim, outDim, seqLen int, dtype DType, scale float32, tileSize int) {
	// Specialized fast-paths
	switch dtype {
	case DTypeFloat32, DTypeFloat16, DTypeBFloat16:
		if wData, ok := weights.([]float32); ok {
			swigluTiledProjectGateUpFloat32(input, wData, gWStart, uWStart, gBStart, uBStart, output, inDim, outDim, seqLen, tileSize)
			return
		}
	case DTypeInt8, DTypeUint8, DTypeFP8E4M3, DTypeFP8E5M2, DTypeInt4, DTypeUint4, DTypeFP4:
		if wData, ok := weights.([]int8); ok {
			// Biases are stored as float32 in the master weights, so we need to cast and scale them.
			// This assumes biases are stored contiguously after weights.
			// The original `weights` variable is `any`, so we need to cast it to `[]float32` to access biases.
			masterWeights := CastWeights[float32](weights)
			if masterWeights == nil { return }
			gB := masterWeights[gBStart:]
			uB := masterWeights[uBStart:]
			swigluTiledProjectGateUpInt8(input, wData, gWStart, uWStart, gB, uB, output, inDim, outDim, seqLen, scale, tileSize)
			return
		}
	}

	wData := CastWeights[float32](weights)
	if wData == nil { return }
	gW := wData[gWStart:]
	uW := wData[uWStart:]
	gB := wData[gBStart:]
	uB := wData[uBStart:]

	// Temporary buffers for partial sums to maintain tiling efficiency
	// Using a buffer that can handle many steps if needed
	sumG := make([]float32, outDim)
	sumU := make([]float32, outDim)

	for s := 0; s < seqLen; s++ {
		// Initialize sums with raw bias (NOT simulated)
		for o := 0; o < outDim; o++ {
			sumG[o] = gB[o]
			sumU[o] = uB[o]
		}

		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim { iEnd = inDim }

			for oTile := 0; oTile < outDim; oTile += tileSize {
				oEnd := oTile + tileSize
				if oEnd > outDim { oEnd = outDim }

				for o := oTile; o < oEnd; o++ {
					sG := sumG[o]
					sU := sumU[o]
					for i := iTile; i < iEnd; i++ {
						inVal := float32(input[s*inDim+i])
						sG += inVal * SimulatePrecision(gW[o*inDim+i], dtype, scale)
						sU += inVal * SimulatePrecision(uW[o*inDim+i], dtype, scale)
					}
					sumG[o] = sG
					sumU[o] = sU
				}
			}
		}

		// Apply SiLU and gating
		for o := 0; o < outDim; o++ {
			sig := 1.0 / (1.0 + math.Exp(-float64(sumG[o])))
			silu := float32(float64(sumG[o]) * sig)
			output[s*outDim+o] = TOut(silu * sumU[o])
		}
	}
}

func swigluTiledProjectDown[TIn Numeric, TOut Numeric](input []TIn, weights any, dWStart, dBStart int, output []TOut, inDim, outDim, seqLen int, dtype DType, scale float32, tileSize int) {
	// Reusing mhaTiledProject logic (Dense-like)
	mhaTiledProject(input, weights, dWStart, dBStart, output, inDim, outDim, seqLen, dtype, scale, tileSize)
}

func swigluTiledProjectGateUpFloat32[TIn Numeric, TOut Numeric](input []TIn, wData []float32, gWStart, uWStart, gBStart, uBStart int, output []TOut, inDim, outDim, seqLen int, tileSize int) {
	gW := wData[gWStart:]
	uW := wData[uWStart:]
	gB := wData[gBStart:]
	uB := wData[uBStart:]

	sumG := make([]float32, outDim)
	sumU := make([]float32, outDim)

	// Local buffer for input tile
	inTileBuf := make([]float32, tileSize)

	for s := 0; s < seqLen; s++ {
		for o := 0; o < outDim; o++ {
			sumG[o] = gB[o]
			sumU[o] = uB[o]
		}
		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim { iEnd = inDim }
			currentITileSize := iEnd - iTile

			// Load input tile once
			for i := 0; i < currentITileSize; i++ {
				inTileBuf[i] = float32(input[s*inDim+iTile+i])
			}

			for oTile := 0; oTile < outDim; oTile += tileSize {
				oEnd := oTile + tileSize
				if oEnd > outDim { oEnd = outDim }

				for o := oTile; o < oEnd; o++ {
					sG, sU := sumG[o], sumU[o]
					rowOff := o * inDim + iTile
					// Unrolled dot product for both Gate and Up
					i := 0
					for ; i <= currentITileSize-4; i += 4 {
						iv0, iv1, iv2, iv3 := inTileBuf[i], inTileBuf[i+1], inTileBuf[i+2], inTileBuf[i+3]
						sG += iv0 * gW[rowOff+i]
						sG += iv1 * gW[rowOff+i+1]
						sG += iv2 * gW[rowOff+i+2]
						sG += iv3 * gW[rowOff+i+3]
						
						sU += iv0 * uW[rowOff+i]
						sU += iv1 * uW[rowOff+i+1]
						sU += iv2 * uW[rowOff+i+2]
						sU += iv3 * uW[rowOff+i+3]
					}
					for ; i < currentITileSize; i++ {
						iv := inTileBuf[i]
						sG += iv * gW[rowOff+i]
						sU += iv * uW[rowOff+i]
					}
					sumG[o], sumU[o] = sG, sU
				}
			}
		}
		for o := 0; o < outDim; o++ {
			sig := 1.0 / (1.0 + math.Exp(-float64(sumG[o])))
			output[s*outDim+o] = TOut(float32(float64(sumG[o]) * sig) * sumU[o])
		}
	}
}

func swigluTiledProjectGateUpInt8[TIn Numeric, TOut Numeric](input []TIn, wData []int8, gWStart, uWStart int, gB, uB []float32, output []TOut, inDim, outDim, seqLen int, scale float32, tileSize int) {
	gW := wData[gWStart:]
	uW := wData[uWStart:]
	
	sumG := make([]float32, outDim)
	sumU := make([]float32, outDim)

	// Local buffer for input tile
	inTileBuf := make([]float32, tileSize)

	for s := 0; s < seqLen; s++ {
		for o := 0; o < outDim; o++ {
			sumG[o] = gB[o]
			sumU[o] = uB[o]
		}
		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim { iEnd = inDim }
			currentITileSize := iEnd - iTile

			// Load input tile once
			for i := 0; i < currentITileSize; i++ {
				inTileBuf[i] = float32(input[s*inDim+iTile+i])
			}

			for oTile := 0; oTile < outDim; oTile += tileSize {
				oEnd := oTile + tileSize
				if oEnd > outDim { oEnd = outDim }
				for o := oTile; o < oEnd; o++ {
					sG, sU := sumG[o], sumU[o]
					rowOff := o * inDim + iTile
					// Unrolled dot product for both Gate and Up
					i := 0
					for ; i <= currentITileSize-4; i += 4 {
						iv0, iv1, iv2, iv3 := inTileBuf[i], inTileBuf[i+1], inTileBuf[i+2], inTileBuf[i+3]
						sG += iv0 * (float32(gW[rowOff+i]) * scale)
						sG += iv1 * (float32(gW[rowOff+i+1]) * scale)
						sG += iv2 * (float32(gW[rowOff+i+2]) * scale)
						sG += iv3 * (float32(gW[rowOff+i+3]) * scale)
						
						sU += iv0 * (float32(uW[rowOff+i]) * scale)
						sU += iv1 * (float32(uW[rowOff+i+1]) * scale)
						sU += iv2 * (float32(uW[rowOff+i+2]) * scale)
						sU += iv3 * (float32(uW[rowOff+i+3]) * scale)
					}
					for ; i < currentITileSize; i++ {
						iv := inTileBuf[i]
						sG += iv * (float32(gW[rowOff+i]) * scale)
						sU += iv * (float32(uW[rowOff+i]) * scale)
					}
					sumG[o], sumU[o] = sG, sU
				}
			}
		}
		for o := 0; o < outDim; o++ {
			sig := 1.0 / (1.0 + math.Exp(-float64(sumG[o])))
			output[s*outDim+o] = TOut(float32(float64(sumG[o]) * sig) * sumU[o])
		}
	}
}

// SwiGLUBackwardTiled calculates gradients for SwiGLU using a tiled approach.
func SwiGLUBackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if layer.EnableMultiCoreTiling {
		return swigluBackwardTiledParallel(layer, gradOutput, input, preAct)
	}

	inputSize, intermediateSize := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inputSize
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 { tileSize = 32 }

	gradInput = NewTensor[T](input.Shape...)
	gradWeights = NewTensor[T](len(layer.WeightStore.Master))

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }

	gateWStart := 0
	upWStart := inputSize * intermediateSize
	downWStart := 2 * inputSize * intermediateSize
	gateBStart := 2 * inputSize * intermediateSize + intermediateSize * inputSize
	upBStart := gateBStart + intermediateSize
	downBStart := upBStart + intermediateSize

	scaleW := layer.WeightStore.Scale
	if scaleW == 0 { scaleW = 1.0 }

	// Temporary buffer for intermediate gradients dL/d(siluGate * up)
	gradInter := NewTensor[T](seqLen, intermediateSize)

	// 1. Tiled Down Projection Backward: dL/d(preAct) and dL/d(downW/B)
	mhaTiledProjectBackward[T, T](layer, gradOutput, preAct, nil, nil, gradInter, gradWeights, intermediateSize, inputSize, seqLen, downWStart, downBStart, tileSize)

	// 2. Tiled Gate & Up Projection Backward: dL/dx and dL/d(gateW/B, upW/B)
	swigluTiledProjectGateUpBackward(gradInter.Data, input.Data, weights, gateWStart, upWStart, gateBStart, upBStart, gradInput.Data, gradWeights.Data, inputSize, intermediateSize, seqLen, layer.DType, scaleW, tileSize)

	return gradInput, gradWeights
}

func swigluTiledProjectGateUpBackward[TIn Numeric, TGrad Numeric](gradInter []TGrad, input []TIn, weights any, gateWStart, upWStart, gateBStart, upBStart int, gradInput []TGrad, gradWeights []TGrad, inDim, outDim, seqLen int, dtype DType, scale float32, tileSize int) {
	wData := CastWeights[float32](weights)
	if wData == nil { return }
	gW := wData[gateWStart:]
	uW := wData[upWStart:]
	gB := wData[gateBStart:]
	uB := wData[upBStart:]

	// Buffers for partial sums
	sumSigG := make([]float32, outDim)
	sumUp := make([]float32, outDim)

	for s := 0; s < seqLen; s++ {
		// First pass: compute intermediate activations for this row
		for o := 0; o < outDim; o++ {
			sumSigG[o] = gB[o]
			sumUp[o] = uB[o]
		}
		for i := 0; i < inDim; i++ {
			inVal := float32(input[s*inDim+i])
			for o := 0; o < outDim; o++ {
				sumSigG[o] += inVal * SimulatePrecision(gW[o*inDim+i], dtype, scale)
				sumUp[o] += inVal * SimulatePrecision(uW[o*inDim+i], dtype, scale)
			}
		}

		// Second pass: compute dUp and dGate and backprop to weights and input
		for oTile := 0; oTile < outDim; oTile += tileSize {
			oEnd := oTile + tileSize
			if oEnd > outDim { oEnd = outDim }

			for o := oTile; o < oEnd; o++ {
				gi := float64(gradInter[s*outDim+o])
				
				x := float64(sumSigG[o])
				sig := 1.0 / (1.0 + math.Exp(-x))
				silu := x * sig
				dSilu := sig * (1.0 + x*(1.0-sig))
				
				dUp := gi * silu
				dGate := gi * float64(sumUp[o]) * dSilu
				
				gradWeights[upBStart+o] += TGrad(dUp)
				gradWeights[gateBStart+o] += TGrad(dGate)

				for iTile := 0; iTile < inDim; iTile += tileSize {
					iEnd := iTile + tileSize
					if iEnd > inDim { iEnd = inDim }
					
					for i := iTile; i < iEnd; i++ {
						inVal := float64(input[s*inDim+i])
						gradWeights[upWStart + o*inDim+i] += TGrad(inVal * dUp)
						gradWeights[gateWStart + o*inDim+i] += TGrad(inVal * dGate)
						
						gradInput[s*inDim+i] += TGrad(dUp * float64(SimulatePrecision(uW[o*inDim+i], dtype, scale)) + 
														dGate * float64(SimulatePrecision(gW[o*inDim+i], dtype, scale)))
					}
				}
			}
		}
	}
}

// ── SwiGLU multi-core parallel tiled forward ──────────────────────────────────
// swigluForwardTiledParallel parallelises over sequence positions.
// Gate/Up and Down projections each run with one goroutine per seqLen tile.

func swigluForwardTiledParallel[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	inputSize, intermediateSize := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inputSize
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	preAct = NewTensor[T](seqLen, intermediateSize)
	postAct = NewTensor[T](seqLen, inputSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}

	gateWStart := 0
	upWStart := inputSize * intermediateSize
	downWStart := 2 * inputSize * intermediateSize
	gateBStart := 2*inputSize*intermediateSize + intermediateSize*inputSize
	upBStart := gateBStart + intermediateSize
	downBStart := upBStart + intermediateSize

	scaleW := layer.WeightStore.Scale
	if scaleW == 0 {
		scaleW = 1.0
	}

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	// Phase 1: Gate & Up projections, parallelised over sequence tiles.
	for sTile := 0; sTile < seqLen; sTile += tileSize {
		sem <- struct{}{}
		wg.Add(1)
		go func(sTile int) {
			defer func() { <-sem; wg.Done() }()
			sEnd := sTile + tileSize
			if sEnd > seqLen {
				sEnd = seqLen
			}
			swigluTiledProjectGateUp(input.Data[sTile*inputSize:sEnd*inputSize],
				weights, gateWStart, upWStart, gateBStart, upBStart,
				preAct.Data[sTile*intermediateSize:sEnd*intermediateSize],
				inputSize, intermediateSize, sEnd-sTile, layer.DType, scaleW, tileSize)
		}(sTile)
	}
	wg.Wait()

	// Phase 2: Down projection, parallelised over sequence tiles.
	for sTile := 0; sTile < seqLen; sTile += tileSize {
		sem <- struct{}{}
		wg.Add(1)
		go func(sTile int) {
			defer func() { <-sem; wg.Done() }()
			sEnd := sTile + tileSize
			if sEnd > seqLen {
				sEnd = seqLen
			}
			swigluTiledProjectDown(preAct.Data[sTile*intermediateSize:sEnd*intermediateSize],
				weights, downWStart, downBStart,
				postAct.Data[sTile*inputSize:sEnd*inputSize],
				intermediateSize, inputSize, sEnd-sTile, layer.DType, scaleW, tileSize)
		}(sTile)
	}
	wg.Wait()

	return preAct, postAct
}

// swigluBackwardTiledParallel computes SwiGLU backward in parallel over seqLen tiles.
// Each goroutine accumulates gradWeights into a local buffer to avoid races;
// after the WaitGroup the local buffers are merged into gradWeights.Data.
func swigluBackwardTiledParallel[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	inputSize, intermediateSize := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inputSize
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	gradInput = NewTensor[T](input.Shape...)
	gradWeights = NewTensor[T](len(layer.WeightStore.Master))

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}

	gateWStart := 0
	upWStart := inputSize * intermediateSize
	downWStart := 2 * inputSize * intermediateSize
	gateBStart := 2*inputSize*intermediateSize + intermediateSize*inputSize
	upBStart := gateBStart + intermediateSize
	downBStart := upBStart + intermediateSize

	scaleW := layer.WeightStore.Scale
	if scaleW == 0 {
		scaleW = 1.0
	}

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	numTiles := (seqLen + tileSize - 1) / tileSize
	localGradWSlices := make([][]T, numTiles)
	for k := 0; k < numTiles; k++ {
		localGradWSlices[k] = make([]T, len(gradWeights.Data))
	}

	for k := 0; k < numTiles; k++ {
		sTile := k * tileSize
		sem <- struct{}{}
		wg.Add(1)
		go func(k, sTile int) {
			defer func() { <-sem; wg.Done() }()
			sEnd := sTile + tileSize
			if sEnd > seqLen {
				sEnd = seqLen
			}
			localGW := localGradWSlices[k]
			localGI := gradInput.Data // gradInput[s,...] unique per s — safe to write directly

			// Temporary gradInter for this tile
			tileLen := sEnd - sTile
			gradInterTile := make([]T, tileLen*intermediateSize)

			// Down projection backward for this tile's rows
			for s := sTile; s < sEnd; s++ {
				ls := s - sTile
				mhaTiledProjectBackward[T, T](layer, gradOutput, preAct, nil, nil,
					&Tensor[T]{Data: gradInterTile[ls*intermediateSize : (ls+1)*intermediateSize], Shape: []int{1, intermediateSize}},
					&Tensor[T]{Data: localGW, Shape: []int{len(localGW)}},
					intermediateSize, inputSize, 1,
					downWStart, downBStart, tileSize)
				// Above call processes row s of gradOutput but uses index 0 for the 1-row sub-tensors.
				// We need the actual row — use a slice view approach instead.
				_ = ls
			}

			// Re-do using full-slice views for the tile
			gradOutputTile := &Tensor[T]{Data: gradOutput.Data[sTile*inputSize : sEnd*inputSize], Shape: []int{tileLen, inputSize}}
			preActTile := &Tensor[T]{Data: preAct.Data[sTile*intermediateSize : sEnd*intermediateSize], Shape: []int{tileLen, intermediateSize}}
			gradInterFull := &Tensor[T]{Data: gradInterTile, Shape: []int{tileLen, intermediateSize}}
			localGWTensor := &Tensor[T]{Data: localGW, Shape: []int{len(localGW)}}

			// Reset gradInterTile and redo properly
			for ii := range gradInterTile {
				gradInterTile[ii] = 0
			}
			mhaTiledProjectBackward[T, T](layer, gradOutputTile, preActTile, nil, nil, gradInterFull, localGWTensor, intermediateSize, inputSize, tileLen, downWStart, downBStart, tileSize)

			// Gate & Up projection backward for this tile
			inputTile := input.Data[sTile*inputSize : sEnd*inputSize]
			gradInputTile := localGI[sTile*inputSize : sEnd*inputSize]
			swigluTiledProjectGateUpBackward(gradInterTile, inputTile, weights, gateWStart, upWStart, gateBStart, upBStart, gradInputTile, localGW, inputSize, intermediateSize, tileLen, layer.DType, scaleW, tileSize)
		}(k, sTile)
	}
	wg.Wait()

	// Merge localGradW slices into gradWeights.Data
	for k := 0; k < numTiles; k++ {
		for i, v := range localGradWSlices[k] {
			gradWeights.Data[i] += v
		}
	}

	return gradInput, gradWeights
}
