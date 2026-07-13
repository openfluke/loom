package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func newEmbeddingNativeInt8Net() *poly.VolumetricNetwork {
	json := `{"id":"test","depth":1,"rows":1,"cols":1,"layers_per_cell":1,"layers":[` +
		`{"z":0,"y":0,"x":0,"l":0,"type":"EMBEDDING","dtype":"Int8","vocab_size":32,"embedding_dim":16}` +
		`]}`
	net, err := poly.BuildNetworkFromJSON([]byte(json))
	if err != nil {
		panic(err)
	}
	net.UseExactDType = true
	l := &net.Layers[0]
	l.Network = net
	l.DType = poly.DTypeInt8
	l.WeightStore.Scale = 0.01
	l.WeightStore.ForceMorph(poly.DTypeInt8)
	return net
}

func TestEmbeddingNativeExactSimdMatchesScalarInt8(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerEmbedding) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	input := poly.NewTensor[float32](8, 1)
	for i := range input.Data {
		input.Data[i] = float32(i % 32)
	}
	gradOut := poly.NewTensor[float32](8, 16)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.02 * float32((i%5)+1)
	}

	netScalar := newEmbeddingNativeInt8Net()
	netSimd := newEmbeddingNativeInt8Net()
	lS := &netScalar.Layers[0]
	lV := &netSimd.Layers[0]
	copy(lV.WeightStore.Master, lS.WeightStore.Master)
	lV.WeightStore.ForceMorph(poly.DTypeInt8)

	netScalar.SetSimdForward(false)
	preS, postS := poly.EmbeddingForwardPolymorphic(lS, input)
	netSimd.SetSimdForward(true)
	preV, postV := poly.EmbeddingForwardPolymorphic(lV, input)

	assertMaxDiffF32(t, preS.Data, preV.Data, 0, "native int8 fwd pre")
	assertMaxDiffF32(t, postS.Data, postV.Data, 0, "native int8 fwd post")

	_, gwS := poly.EmbeddingBackwardPolymorphic(lS, gradOut, input, preS)
	_, gwV := poly.EmbeddingBackwardPolymorphic(lV, gradOut, input, preV)
	assertMaxDiffF32(t, gwS.Data, gwV.Data, 0, "native int8 bwd dW")
}
