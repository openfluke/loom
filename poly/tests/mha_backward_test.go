package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestMHABackwardSimdMatchesTiled(t *testing.T) {
	if !poly.Plan9SimdEnabled() {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}

	net := newMHATestLayer(64, 4, 8)
	l := net.GetLayer(0, 0, 0, 0)
	input := poly.NewTensor[float32](4, 8, 64)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%13)
	}

	net.SetSimdForward(false)
	preT, _ := poly.MHAForwardPolymorphic(l, input)
	gradOut := poly.NewTensor[float32](4, 8, 64)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.02 * float32((i%7)+1)
	}
	dxT, dwT := poly.MHABackwardPolymorphic(l, gradOut, input, preT)

	net.SetSimdForward(true)
	preS, _ := poly.MHAForwardPolymorphic(l, input)
	dxS, dwS := poly.MHABackwardPolymorphic(l, gradOut, input, preS)

	assertMaxDiffF32(t, dxT.Data, dxS.Data, 1e-3, "dX tiled vs simd")
	assertMaxDiffF32(t, dwT.Data, dwS.Data, 1e-3, "dW tiled vs simd")
}
