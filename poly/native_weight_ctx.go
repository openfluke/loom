package poly

import "math"

// nativeWeightCtx resolves WeightStore native slices once per forward/backward pass.
// Avoids per-element GetNative + interface type assertions in hot MHA loops.
type nativeWeightCtx struct {
	layer     *VolumetricLayer
	dt        DType
	scale     float32
	gradWTerm func(inputVal float32, gradPre float64) float64
	gradXAt   func(weightIdx int, gradPre float64) float64
	weightF64 func(idx int) float64
	dotRow    func(input []float32, rowOff, inSz int) float32
	biasF64   func(idx int) float64
}

func newNativeWeightCtx(layer *VolumetricLayer) nativeWeightCtx {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	ctx := nativeWeightCtx{
		layer: layer,
		dt:    layer.DType,
		scale: scale,
		gradWTerm: func(inputVal float32, gradPre float64) float64 {
			return denseNativeGradWTerm(layer, inputVal, gradPre)
		},
	}
	native := ws.GetNative(layer.DType)

	switch layer.DType {
	case DTypeFloat64:
		w := native.([]float64)
		ctx.gradXAt = func(idx int, gp float64) float64 { return w[idx] * gp }
		ctx.weightF64 = func(idx int) float64 { return w[idx] }
		ctx.biasF64 = ctx.weightF64
		ctx.dotRow = func(input []float32, rowOff, inSz int) float32 {
			var sum float64
			for i := 0; i < inSz; i++ {
				sum += w[rowOff+i] * float64(input[i])
			}
			return float32(sum)
		}
	case DTypeFloat32:
		w := ws.Master
		ctx.gradXAt = func(idx int, gp float64) float64 { return float64(w[idx]) * gp }
		ctx.weightF64 = func(idx int) float64 { return float64(w[idx]) }
		ctx.biasF64 = ctx.weightF64
		ctx.dotRow = func(input []float32, rowOff, inSz int) float32 {
			var sum float32
			for i := 0; i < inSz; i++ {
				sum += w[rowOff+i] * input[i]
			}
			return sum
		}
	case DTypeFloat16:
		w := native.([]uint16)
		ctx.gradXAt = func(idx int, gp float64) float64 {
			return float64(float16ToFloat32(w[idx])) * gp
		}
		ctx.weightF64 = func(idx int) float64 { return float64(float16ToFloat32(w[idx])) }
		ctx.biasF64 = ctx.weightF64
		ctx.dotRow = func(input []float32, rowOff, inSz int) float32 {
			sum := float32(0)
			for i := 0; i < inSz; i++ {
				wv := float16ToFloat32(w[rowOff+i])
				prod := float16ToFloat32(float32ToFloat16(wv * input[i]))
				sum = float16ToFloat32(float32ToFloat16(sum + prod))
			}
			return sum
		}
	case DTypeBFloat16:
		w := native.([]uint16)
		ctx.gradXAt = func(idx int, gp float64) float64 {
			return float64(bfloat16ToFloat32(w[idx])) * gp
		}
		ctx.weightF64 = func(idx int) float64 { return float64(bfloat16ToFloat32(w[idx])) }
		ctx.biasF64 = ctx.weightF64
		ctx.dotRow = func(input []float32, rowOff, inSz int) float32 {
			sum := float32(0)
			for i := 0; i < inSz; i++ {
				wv := bfloat16ToFloat32(w[rowOff+i])
				prod := bfloat16ToFloat32(float32ToBFloat16(wv * input[i]))
				sum = bfloat16ToFloat32(float32ToBFloat16(sum + prod))
			}
			return sum
		}
	case DTypeFP8E4M3:
		w := native.([]uint8)
		ctx.gradXAt = func(idx int, gp float64) float64 {
			return float64(e4m3ToFloat32(w[idx])*scale) * gp
		}
		ctx.weightF64 = func(idx int) float64 { return float64(e4m3ToFloat32(w[idx]) * scale) }
		ctx.biasF64 = ctx.weightF64
		ctx.dotRow = func(input []float32, rowOff, inSz int) float32 {
			sum := float32(0)
			for i := 0; i < inSz; i++ {
				wv := e4m3ToFloat32(w[rowOff+i]) * scale
				p := e4m3ToFloat32(float32ToE4M3(wv*input[i]/scale)) * scale
				sum = e4m3ToFloat32(float32ToE4M3(sum/scale+p/scale)) * scale
			}
			return sum
		}
	case DTypeFP8E5M2:
		w := native.([]uint8)
		ctx.gradXAt = func(idx int, gp float64) float64 {
			return float64(e5m2ToFloat32(w[idx])*scale) * gp
		}
		ctx.weightF64 = func(idx int) float64 { return float64(e5m2ToFloat32(w[idx]) * scale) }
		ctx.biasF64 = ctx.weightF64
		ctx.dotRow = func(input []float32, rowOff, inSz int) float32 {
			sum := float32(0)
			for i := 0; i < inSz; i++ {
				wv := e5m2ToFloat32(w[rowOff+i]) * scale
				p := e5m2ToFloat32(float32ToE5M2(wv*input[i]/scale)) * scale
				sum = e5m2ToFloat32(float32ToE5M2(sum/scale+p/scale)) * scale
			}
			return sum
		}
	case DTypeInt64:
		w := native.([]int64)
		ctx.gradXAt = func(idx int, gp float64) float64 {
			return float64(w[idx]) * float64(scale) * gp
		}
		ctx.weightF64 = func(idx int) float64 { return float64(w[idx]) * float64(scale) }
		ctx.biasF64 = ctx.weightF64
		ctx.dotRow = func(input []float32, rowOff, inSz int) float32 {
			var acc int64
			for i := 0; i < inSz; i++ {
				acc += w[rowOff+i] * int64(math.Round(float64(input[i])/float64(scale)))
			}
			return float32(acc) * scale * scale
		}
	case DTypeUint64:
		w := native.([]uint64)
		ctx.gradXAt = func(idx int, gp float64) float64 {
			return float64(w[idx]) * float64(scale) * gp
		}
		ctx.weightF64 = func(idx int) float64 { return float64(w[idx]) * float64(scale) }
		ctx.biasF64 = ctx.weightF64
		ctx.dotRow = dotRowUintScaled(w, scale)
	case DTypeInt32:
		w := native.([]int32)
		ctx.gradXAt = func(idx int, gp float64) float64 {
			return float64(w[idx]) * float64(scale) * gp
		}
		ctx.weightF64 = func(idx int) float64 { return float64(w[idx]) * float64(scale) }
		ctx.biasF64 = ctx.weightF64
		ctx.dotRow = dotRowInt32Scaled(w, scale)
	case DTypeUint32:
		w := native.([]uint32)
		ctx.gradXAt = func(idx int, gp float64) float64 {
			return float64(w[idx]) * float64(scale) * gp
		}
		ctx.weightF64 = func(idx int) float64 { return float64(w[idx]) * float64(scale) }
		ctx.biasF64 = ctx.weightF64
		ctx.dotRow = dotRowUint32Scaled(w, scale)
	case DTypeInt16:
		w := native.([]int16)
		ctx.gradXAt = func(idx int, gp float64) float64 {
			return float64(w[idx]) * float64(scale) * gp
		}
		ctx.weightF64 = func(idx int) float64 { return float64(w[idx]) * float64(scale) }
		ctx.biasF64 = ctx.weightF64
		ctx.dotRow = dotRowInt16Scaled(w, scale)
	case DTypeUint16:
		w := native.([]uint16)
		ctx.gradXAt = func(idx int, gp float64) float64 {
			return float64(w[idx]) * float64(scale) * gp
		}
		ctx.weightF64 = func(idx int) float64 { return float64(w[idx]) * float64(scale) }
		ctx.biasF64 = ctx.weightF64
		ctx.dotRow = dotRowUint16Scaled(w, scale)
	case DTypeFP4:
		w := native.([]uint8)
		ctx.gradXAt = func(idx int, gp float64) float64 {
			return float64(fp4CodeToFloat32(w[idx], scale)) * gp
		}
		ctx.weightF64 = func(idx int) float64 { return float64(fp4CodeToFloat32(w[idx], scale)) }
		ctx.biasF64 = ctx.weightF64
		ctx.dotRow = func(input []float32, rowOff, inSz int) float32 {
			var acc float64
			for i := 0; i < inSz; i++ {
				wv := fp4CodeToFloat32(w[rowOff+i], scale)
				acc += float64(wv) * float64(input[i])
			}
			return float32(acc)
		}
	default:
		codes, ok := nativeU8WeightsView(native)
		if !ok {
			ctx.gradXAt = func(int, float64) float64 { return 0 }
			ctx.weightF64 = func(int) float64 { return 0 }
			ctx.biasF64 = ctx.weightF64
			ctx.dotRow = func([]float32, int, int) float32 { return 0 }
			return ctx
		}
		dt := layer.DType
		ctx.gradXAt = func(idx int, gp float64) float64 {
			wv := denseNativeSignedU8Weight(dt, codes[idx], scale)
			return float64(wv) * float64(scale) * gp
		}
		ctx.weightF64 = func(idx int) float64 {
			wv := denseNativeSignedU8Weight(dt, codes[idx], scale)
			return float64(wv) * float64(scale)
		}
		ctx.biasF64 = ctx.weightF64
		ctx.dotRow = func(input []float32, rowOff, inSz int) float32 {
			var acc int64
			for i := 0; i < inSz; i++ {
				wv := denseNativeSignedU8Weight(dt, codes[rowOff+i], scale)
				xq := int64(math.Round(float64(input[i]) / float64(scale)))
				acc += int64(wv) * xq
			}
			return float32(acc) * scale * scale
		}
	}
	return ctx
}

func dotRowInt32Scaled(w []int32, scale float32) func([]float32, int, int) float32 {
	return func(input []float32, rowOff, inSz int) float32 {
		var acc int64
		for i := 0; i < inSz; i++ {
			acc += int64(w[rowOff+i]) * int64(math.Round(float64(input[i])/float64(scale)))
		}
		return float32(acc) * scale * scale
	}
}

func dotRowInt16Scaled(w []int16, scale float32) func([]float32, int, int) float32 {
	return func(input []float32, rowOff, inSz int) float32 {
		var acc int64
		for i := 0; i < inSz; i++ {
			acc += int64(w[rowOff+i]) * int64(math.Round(float64(input[i])/float64(scale)))
		}
		return float32(acc) * scale * scale
	}
}

func dotRowUintScaled(w []uint64, scale float32) func([]float32, int, int) float32 {
	return func(input []float32, rowOff, inSz int) float32 {
		var acc uint64
		for i := 0; i < inSz; i++ {
			xq := uint64(math.Max(0, math.Round(float64(input[i])/float64(scale))))
			acc += w[rowOff+i] * xq
		}
		return float32(acc) * scale * scale
	}
}

func dotRowUint32Scaled(w []uint32, scale float32) func([]float32, int, int) float32 {
	return func(input []float32, rowOff, inSz int) float32 {
		var acc int64
		for i := 0; i < inSz; i++ {
			xq := int64(math.Max(0, math.Round(float64(input[i])/float64(scale))))
			acc += int64(w[rowOff+i]) * xq
		}
		return float32(acc) * scale * scale
	}
}

func dotRowUint16Scaled(w []uint16, scale float32) func([]float32, int, int) float32 {
	return func(input []float32, rowOff, inSz int) float32 {
		var acc int64
		for i := 0; i < inSz; i++ {
			xq := int64(math.Max(0, math.Round(float64(input[i])/float64(scale))))
			acc += int64(w[rowOff+i]) * xq
		}
		return float32(acc) * scale * scale
	}
}

func mhaProjectRowCtx(ctx *nativeWeightCtx, inRow []float32, wStart, bStart, outDim, inDim int, out []float64) {
	for o := 0; o < outDim; o++ {
		sum := ctx.biasF64(bStart + o)
		sum += float64(ctx.dotRow(inRow, wStart+o*inDim, inDim))
		out[o] = sum
	}
}

// materializeF32Weights decodes native storage into a contiguous float32 buffer once per pass.
func (ctx *nativeWeightCtx) materializeF32Weights(count int) []float32 {
	out := make([]float32, count)
	for i := 0; i < count; i++ {
		out[i] = float32(ctx.weightF64(i))
	}
	return out
}
