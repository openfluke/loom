package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestCNN1BackwardSimdMatchesTiled(t *testing.T) {
	if !poly.Plan9SimdEnabled() {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}

	inC, filters, seqLen, kSize := 16, 16, 16, 3
	net := newCNN1TestNet(inC, filters, seqLen, kSize)
	l := net.GetLayer(0, 0, 0, 0)
	input := poly.NewTensor[float32](4, inC, seqLen)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%17)
	}

	net.SetSimdForward(false)
	preT, _ := poly.CNN1ForwardPolymorphic(l, input)
	gradOut := poly.NewTensor[float32](4, filters, seqLen)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.02 * float32((i%7)+1)
	}
	dxT, dwT := poly.CNN1BackwardPolymorphic(l, gradOut, input, preT)

	net.SetSimdForward(true)
	preS, _ := poly.CNN1ForwardPolymorphic(l, input)
	dxS, dwS := poly.CNN1BackwardPolymorphic(l, gradOut, input, preS)

	assertMaxDiffF32(t, dxT.Data, dxS.Data, 1e-4, "dX tiled vs simd")
	assertMaxDiffF32(t, dwT.Data, dwS.Data, 1e-4, "dW tiled vs simd")
}

func TestCNN1BackwardSCMatchesMC(t *testing.T) {
	inC, filters, seqLen, kSize := 16, 16, 16, 3
	netSC := newCNN1TestNet(inC, filters, seqLen, kSize)
	netMC := newCNN1TestNet(inC, filters, seqLen, kSize)
	netMC.GetLayer(0, 0, 0, 0).EnableMultiCoreTiling = true
	lSC := netSC.GetLayer(0, 0, 0, 0)
	lMC := netMC.GetLayer(0, 0, 0, 0)

	input := poly.NewTensor[float32](4, inC, seqLen)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%13)
	}

	preSC, _ := poly.CNN1ForwardPolymorphic(lSC, input)
	preMC, _ := poly.CNN1ForwardPolymorphic(lMC, input)
	gradOut := poly.NewTensor[float32](4, filters, seqLen)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.02 * float32((i%7)+1)
	}

	dxSC, dwSC := poly.CNN1BackwardPolymorphic(lSC, gradOut, input, preSC)
	dxMC, dwMC := poly.CNN1BackwardPolymorphic(lMC, gradOut, input, preMC)

	assertMaxDiffF32(t, dxSC.Data, dxMC.Data, 1e-5, "dX SC vs MC")
	assertMaxDiffF32(t, dwSC.Data, dwMC.Data, 1e-5, "dW SC vs MC")
}
