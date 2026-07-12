package poly

import (
	"math"
	"math/rand"

	"github.com/openfluke/loom/poly/simd"
)

// integer_native.go — shared int8 matmul / backward / stochastic update for all layers.

func int8DotRow(weights []int8, input []int8, rowOff, inSz int) int8 {
	acc := int8DotRowAcc(weights, input, rowOff, inSz)
	return clampI8(acc >> 8)
}

func int8DotRowAcc(weights []int8, input []int8, rowOff, inSz int) int32 {
	if simd.Int8DotSimdActive() && inSz >= 8 {
		return simd.DotI8Tile(weights, input, rowOff, 0, inSz, 0)
	}
	var acc int32
	for i := 0; i < inSz; i++ {
		acc += int32(weights[rowOff+i]) * int32(input[i])
	}
	return acc
}

func int8AccumWeightGrad(gradW []int32, weights []int8, input []int8, gradOut int32, rowOff, inSz int) {
	if simd.Int8DotSimdActive() && inSz >= 8 {
		simd.SaxpyI8ScaleI32Acc(gradW, rowOff, input, gradOut, inSz)
		return
	}
	for i := 0; i < inSz; i++ {
		gradW[rowOff+i] += int32(input[i]) * gradOut
	}
}

func int8AccumInputGrad(gradIn []int32, weights []int8, gradOut int32, rowOff, inSz int) {
	for i := 0; i < inSz; i++ {
		gradIn[i] += (int32(weights[rowOff+i]) * gradOut) >> 8
	}
}

// gradF64ToI32 maps a float-space gradient into int8 MAC/update units (plain round, no forced sign).
func gradF64ToI32(v float64, scale float32) int32 {
	if v == 0 {
		return 0
	}
	if scale == 0 {
		scale = 1
	}
	g := int64(math.Round(v / float64(scale)))
	if g > 32767 {
		g = 32767
	}
	if g < -32768 {
		g = -32768
	}
	return int32(g)
}

func applyStochasticInt8Update(weights []int8, gradWeights []int32, lrShift uint) {
	mask := int32((1 << lrShift) - 1)
	for i := range weights {
		scaledGrad := gradWeights[i] >> lrShift
		if (gradWeights[i] & mask) > rand.Int31n(1<<lrShift) {
			scaledGrad++
		}
		next := int32(weights[i]) - scaledGrad
		weights[i] = clampI8(next)
	}
}

func int8LinearForward(weights []int8, input []int8, batch, inSz, outSz int, pre, post []int8, act ActivationType) {
	TrueInt8DenseForward(weights, input, batch, inSz, outSz, pre, post, act)
}

func int8LinearBackward(
	weights []int8,
	input []int8,
	gradOut []int8,
	batch, inSz, outSz int,
	lrShift uint,
	act ActivationType,
	preI8 []int8,
) ([]int8, []int8) {
	return TrueInt8DenseBackwardUpdate(weights, input, gradOut, batch, inSz, outSz, lrShift, act, preI8)
}
