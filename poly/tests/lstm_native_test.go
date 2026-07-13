package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func newLSTMNativeInt8Net() *poly.VolumetricNetwork {
	net := newLSTMTestNet(8, 8, 4)
	net.UseExactDType = true
	l := net.GetLayer(0, 0, 0, 0)
	l.DType = poly.DTypeInt8
	l.WeightStore.Scale = 0.01
	l.WeightStore.ForceMorph(poly.DTypeInt8)
	return net
}

func TestLSTMNativeExactInt8ForwardShape(t *testing.T) {
	net := newLSTMNativeInt8Net()
	l := net.GetLayer(0, 0, 0, 0)
	input := poly.NewTensor[float32](2, 4, 8)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%11)
	}
	pre, post := poly.LSTMForwardPolymorphic(l, input)
	if len(pre.Shape) != 3 || pre.Shape[2] != 40 {
		t.Fatalf("expected [batch,seq,5*hid], got %v", pre.Shape)
	}
	if len(post.Shape) != 3 || post.Shape[2] != 8 {
		t.Fatalf("expected [batch,seq,hid], got %v", post.Shape)
	}
	gradOut := poly.NewTensor[float32](2, 4, 8)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.02 * float32((i%5)+1)
	}
	dx, _ := poly.LSTMBackwardPolymorphic(l, gradOut, input, pre)
	if len(dx.Shape) != 3 || dx.Shape[2] != 8 {
		t.Fatalf("expected grad input [batch,seq,in], got %v", dx.Shape)
	}
}

func TestLSTMNativeExactSimdMatchesScalarInt8(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerLSTM) {
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

	netScalar := newLSTMNativeInt8Net()
	netSimd := newLSTMNativeInt8Net()

	netScalar.SetSimdForward(false)
	preS, postS := poly.LSTMForwardPolymorphic(netScalar.GetLayer(0, 0, 0, 0), input)
	netSimd.SetSimdForward(true)
	preV, postV := poly.LSTMForwardPolymorphic(netSimd.GetLayer(0, 0, 0, 0), input)

	assertMaxDiffF32(t, preS.Data, preV.Data, 0, "native int8 fwd pre")
	assertMaxDiffF32(t, postS.Data, postV.Data, 0, "native int8 fwd post")

	dxS, _ := poly.LSTMBackwardPolymorphic(netScalar.GetLayer(0, 0, 0, 0), gradOut, input, preS)
	dxV, _ := poly.LSTMBackwardPolymorphic(netSimd.GetLayer(0, 0, 0, 0), gradOut, input, preV)
	assertMaxDiffF32(t, dxS.Data, dxV.Data, 0, "native int8 bwd dX")
}
