package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestCNN3BackwardSimdMatchesTiled(t *testing.T) {
	if !poly.Plan9SimdEnabled() {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}

	inC, filters, spatial, kSize := 8, 8, 8, 3
	net := newCNN3TestNet(inC, filters, spatial, kSize)
	l := net.GetLayer(0, 0, 0, 0)
	input := poly.NewTensor[float32](4, inC, spatial, spatial, spatial)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%17)
	}

	net.SetSimdForward(false)
	preT, _ := poly.CNN3ForwardPolymorphic(l, input)
	gradOut := poly.NewTensor[float32](4, filters, spatial, spatial, spatial)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.02 * float32((i%7)+1)
	}
	dxT, dwT := poly.CNN3BackwardPolymorphic(l, gradOut, input, preT)

	net.SetSimdForward(true)
	preS, _ := poly.CNN3ForwardPolymorphic(l, input)
	dxS, dwS := poly.CNN3BackwardPolymorphic(l, gradOut, input, preS)

	assertMaxDiffF32(t, dxT.Data, dxS.Data, 1e-4, "dX tiled vs simd")
	assertMaxDiffF32(t, dwT.Data, dwS.Data, 1e-4, "dW tiled vs simd")
}

func TestCNN3BackwardSCMatchesMC(t *testing.T) {
	inC, filters, spatial, kSize := 8, 8, 8, 3
	netSC := newCNN3TestNet(inC, filters, spatial, kSize)
	netMC := newCNN3TestNet(inC, filters, spatial, kSize)
	netMC.GetLayer(0, 0, 0, 0).EnableMultiCoreTiling = true
	lSC := netSC.GetLayer(0, 0, 0, 0)
	lMC := netMC.GetLayer(0, 0, 0, 0)

	input := poly.NewTensor[float32](4, inC, spatial, spatial, spatial)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%13)
	}

	preSC, _ := poly.CNN3ForwardPolymorphic(lSC, input)
	preMC, _ := poly.CNN3ForwardPolymorphic(lMC, input)
	gradOut := poly.NewTensor[float32](4, filters, spatial, spatial, spatial)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.02 * float32((i%7)+1)
	}

	dxSC, dwSC := poly.CNN3BackwardPolymorphic(lSC, gradOut, input, preSC)
	dxMC, dwMC := poly.CNN3BackwardPolymorphic(lMC, gradOut, input, preMC)

	assertMaxDiffF32(t, dxSC.Data, dxMC.Data, 1e-5, "dX SC vs MC")
	assertMaxDiffF32(t, dwSC.Data, dwMC.Data, 1e-5, "dW SC vs MC")
}
