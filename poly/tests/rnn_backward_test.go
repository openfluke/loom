package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestRNNBackwardSimdMatchesTiled(t *testing.T) {
	if !poly.Plan9SimdEnabled() {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}

	inputSize, hiddenSize, seqLen := 64, 64, 8
	net := newRNNTestNet(inputSize, hiddenSize, seqLen)
	l := net.GetLayer(0, 0, 0, 0)
	input := poly.NewTensor[float32](4, seqLen, inputSize)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%17)
	}

	net.SetSimdForward(false)
	preT, _ := poly.RNNForwardPolymorphic(l, input)
	gradOut := poly.NewTensor[float32](4, seqLen, hiddenSize)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.02 * float32((i%7)+1)
	}
	dxT, dwT := poly.RNNBackwardPolymorphic(l, gradOut, input, preT)

	net.SetSimdForward(true)
	preS, _ := poly.RNNForwardPolymorphic(l, input)
	dxS, dwS := poly.RNNBackwardPolymorphic(l, gradOut, input, preS)

	assertMaxDiffF32(t, dxT.Data, dxS.Data, 1e-4, "dX tiled vs simd")
	assertMaxDiffF32(t, dwT.Data, dwS.Data, 1e-4, "dW tiled vs simd")
}

func TestRNNBackwardSCMatchesMC(t *testing.T) {
	inputSize, hiddenSize, seqLen := 64, 64, 8
	netSC := newRNNTestNet(inputSize, hiddenSize, seqLen)
	netMC := newRNNTestNet(inputSize, hiddenSize, seqLen)
	netMC.GetLayer(0, 0, 0, 0).EnableMultiCoreTiling = true
	lSC := netSC.GetLayer(0, 0, 0, 0)
	lMC := netMC.GetLayer(0, 0, 0, 0)

	input := poly.NewTensor[float32](4, seqLen, inputSize)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%13)
	}

	preSC, _ := poly.RNNForwardPolymorphic(lSC, input)
	preMC, _ := poly.RNNForwardPolymorphic(lMC, input)
	gradOut := poly.NewTensor[float32](4, seqLen, hiddenSize)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.02 * float32((i%7)+1)
	}

	dxSC, dwSC := poly.RNNBackwardPolymorphic(lSC, gradOut, input, preSC)
	dxMC, dwMC := poly.RNNBackwardPolymorphic(lMC, gradOut, input, preMC)

	assertMaxDiffF32(t, dxSC.Data, dxMC.Data, 1e-5, "dX SC vs MC")
	assertMaxDiffF32(t, dwSC.Data, dwMC.Data, 1e-5, "dW SC vs MC")
}
