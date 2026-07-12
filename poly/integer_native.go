package poly

import "math/rand"

// integer_native.go — shared int8 matmul / backward / stochastic update for all layers.

func int8DotRow(weights []int8, input []int8, rowOff, inSz int) int8 {
	var acc int32
	for i := 0; i < inSz; i++ {
		acc += int32(weights[rowOff+i]) * int32(input[i])
	}
	return clampI8(acc >> 8)
}

func int8DotRowAcc(weights []int8, input []int8, rowOff, inSz int) int32 {
	var acc int32
	for i := 0; i < inSz; i++ {
		acc += int32(weights[rowOff+i]) * int32(input[i])
	}
	return acc
}

func int8AccumWeightGrad(gradW []int32, weights []int8, input []int8, gradOut int32, rowOff, inSz int) {
	for i := 0; i < inSz; i++ {
		gradW[rowOff+i] += int32(input[i]) * gradOut
	}
}

func int8AccumInputGrad(gradIn []int32, weights []int8, gradOut int32, rowOff, inSz int) {
	for i := 0; i < inSz; i++ {
		gradIn[i] += (int32(weights[rowOff+i]) * gradOut) >> 8
	}
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
