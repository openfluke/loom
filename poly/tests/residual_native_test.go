package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestResidualNativeExactSimdMatchesScalar(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerResidual) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = poly.LayerResidual
	l.DType = poly.DTypeFloat32
	l.InputHeight = 64
	l.OutputHeight = 64
	net.UseExactDType = true

	input := poly.NewTensor[float32](4, 64)
	skip := poly.NewTensor[float32](4, 64)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%17)
		skip.Data[i] = 0.02 * float32((i%11)+1)
	}
	gradOut := poly.NewTensor[float32](4, 64)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.03 * float32((i%7)+1)
	}

	net.SetSimdForward(false)
	preS, postS := poly.ResidualForwardPolymorphic(l, input, skip)
	net.SetSimdForward(true)
	preV, postV := poly.ResidualForwardPolymorphic(l, input, skip)

	assertMaxDiffF32(t, postS.Data, postV.Data, 0, "native residual fwd")
	_ = preS
	_ = preV

	dxS, _ := poly.ResidualBackwardPolymorphic(l, gradOut, input, preS)
	net.SetSimdForward(false)
	dxS2, _ := poly.ResidualBackwardPolymorphic(l, gradOut, input, preS)
	net.SetSimdForward(true)
	dxV, _ := poly.ResidualBackwardPolymorphic(l, gradOut, input, preV)
	assertMaxDiffF32(t, dxS.Data, dxV.Data, 0, "native residual bwd dX")
	assertMaxDiffF32(t, dxS2.Data, dxV.Data, 0, "native residual bwd dX scalar vs simd")
}
