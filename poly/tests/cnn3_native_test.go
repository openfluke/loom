package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestCNN3NativeExactInt8ForwardShape(t *testing.T) {
	net := newCNN3TestNet(4, 4, 6, 3)
	net.UseExactDType = true
	l := net.GetLayer(0, 0, 0, 0)
	l.DType = poly.DTypeInt8
	l.WeightStore.Scale = 0.01
	l.WeightStore.ForceMorph(poly.DTypeInt8)

	input := poly.NewTensor[float32](2, 4, 6, 6, 6)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%11)
	}

	pre, post := poly.CNN3ForwardPolymorphic(l, input)
	if len(pre.Shape) != 5 || pre.Shape[2] != 6 {
		t.Fatalf("expected 5D output with depth=6, got shape %v", pre.Shape)
	}
	if len(post.Data) != len(pre.Data) {
		t.Fatal("postAct size mismatch")
	}

	gradOut := poly.NewTensor[float32](2, 4, 6, 6, 6)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.02 * float32((i%5)+1)
	}
	dx, _ := poly.CNN3BackwardPolymorphic(l, gradOut, input, pre)
	if len(dx.Shape) != 5 || dx.Shape[1] != 4 || dx.Shape[2] != 6 {
		t.Fatalf("expected 5D grad input, got shape %v", dx.Shape)
	}
}

func TestCNN3NativeExactSimdMatchesScalarInt8(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerCNN3) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}

	input := poly.NewTensor[float32](2, 4, 6, 6, 6)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%13)
	}
	gradOut := poly.NewTensor[float32](2, 4, 6, 6, 6)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.02 * float32((i%7)+1)
	}

	netScalar := newCNN3NativeInt8Net()
	netSimd := newCNN3NativeInt8Net()

	netScalar.SetSimdForward(false)
	preS, postS := poly.CNN3ForwardPolymorphic(netScalar.GetLayer(0, 0, 0, 0), input)

	netSimd.SetSimdForward(true)
	preV, postV := poly.CNN3ForwardPolymorphic(netSimd.GetLayer(0, 0, 0, 0), input)

	assertMaxDiffF32(t, preS.Data, preV.Data, 0, "native int8 fwd pre")
	assertMaxDiffF32(t, postS.Data, postV.Data, 0, "native int8 fwd post")

	dxS, _ := poly.CNN3BackwardPolymorphic(netScalar.GetLayer(0, 0, 0, 0), gradOut, input, preS)
	dxV, _ := poly.CNN3BackwardPolymorphic(netSimd.GetLayer(0, 0, 0, 0), gradOut, input, preV)
	assertMaxDiffF32(t, dxS.Data, dxV.Data, 0, "native int8 bwd dX")
}

func newCNN3NativeInt8Net() *poly.VolumetricNetwork {
	net := newCNN3TestNet(4, 4, 6, 3)
	net.UseExactDType = true
	l := net.GetLayer(0, 0, 0, 0)
	l.DType = poly.DTypeInt8
	l.WeightStore.Scale = 0.01
	l.WeightStore.ForceMorph(poly.DTypeInt8)
	return net
}
