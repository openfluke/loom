package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func newRNNNativeInt8Net() *poly.VolumetricNetwork {
	net := newRNNTestNet(8, 8, 4)
	net.UseExactDType = true
	l := net.GetLayer(0, 0, 0, 0)
	l.DType = poly.DTypeInt8
	l.WeightStore.Scale = 0.01
	l.WeightStore.ForceMorph(poly.DTypeInt8)
	return net
}

func TestRNNNativeExactInt8ForwardShape(t *testing.T) {
	net := newRNNNativeInt8Net()
	l := net.GetLayer(0, 0, 0, 0)
	input := poly.NewTensor[float32](2, 4, 8)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%11)
	}
	pre, post := poly.RNNForwardPolymorphic(l, input)
	if len(pre.Shape) != 3 || pre.Shape[2] != 8 {
		t.Fatalf("expected [batch,seq,hid], got %v", pre.Shape)
	}
	if len(post.Data) != len(pre.Data) {
		t.Fatal("postAct size mismatch")
	}
	gradOut := poly.NewTensor[float32](2, 4, 8)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.02 * float32((i%5)+1)
	}
	dx, _ := poly.RNNBackwardPolymorphic(l, gradOut, input, pre)
	if len(dx.Shape) != 3 || dx.Shape[2] != 8 {
		t.Fatalf("expected grad input [batch,seq,in], got %v", dx.Shape)
	}
}

func TestRNNNativeExactSimdMatchesScalarInt8(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerRNN) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	input := poly.NewTensor[float32](2, 4, 8)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%13)
	}
	gradOut := poly.NewTensor[float32](2, 4, 8)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.02 * float32((i%7)+1)
	}

	netScalar := newRNNNativeInt8Net()
	netSimd := newRNNNativeInt8Net()

	netScalar.SetSimdForward(false)
	preS, postS := poly.RNNForwardPolymorphic(netScalar.GetLayer(0, 0, 0, 0), input)
	netSimd.SetSimdForward(true)
	preV, postV := poly.RNNForwardPolymorphic(netSimd.GetLayer(0, 0, 0, 0), input)

	assertMaxDiffF32(t, preS.Data, preV.Data, 0, "native int8 fwd pre")
	assertMaxDiffF32(t, postS.Data, postV.Data, 0, "native int8 fwd post")

	dxS, _ := poly.RNNBackwardPolymorphic(netScalar.GetLayer(0, 0, 0, 0), gradOut, input, preS)
	dxV, _ := poly.RNNBackwardPolymorphic(netSimd.GetLayer(0, 0, 0, 0), gradOut, input, preV)
	assertMaxDiffF32(t, dxS.Data, dxV.Data, 0, "native int8 bwd dX")
}
