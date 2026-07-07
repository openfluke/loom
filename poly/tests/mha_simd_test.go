package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func newMHATestLayer(dModel, heads, seq int) *poly.VolumetricNetwork {
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = poly.LayerMultiHeadAttention
	l.DModel = dModel
	l.NumHeads = heads
	l.SeqLength = seq
	l.HeadDim = dModel / heads
	l.DType = poly.DTypeFloat32
	qDim := heads * l.HeadDim
	kvDim := qDim
	wCount := qDim*dModel + kvDim*dModel + kvDim*dModel + dModel*qDim +
		qDim + kvDim + kvDim + dModel
	l.WeightStore = poly.NewWeightStore(wCount)
	for i := range l.WeightStore.Master {
		l.WeightStore.Master[i] = 0.001 * float32((i%19)+1)
	}
	return net
}

func TestMHASimdMatchesTiledFloat32(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerMultiHeadAttention) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}

	net := newMHATestLayer(64, 4, 8)
	l := net.GetLayer(0, 0, 0, 0)
	input := poly.NewTensor[float32](4, 8, 64)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%13)
	}

	net.SetSimdForward(false)
	l.UseSimdForward = false
	preT, postT := poly.MHAForwardPolymorphic(l, input)

	net.SetSimdForward(true)
	preS, postS := poly.MHAForwardPolymorphic(l, input)

	var maxPre, maxPost float32
	for i := range preT.Data {
		d := preT.Data[i] - preS.Data[i]
		if d < 0 {
			d = -d
		}
		if d > maxPre {
			maxPre = d
		}
	}
	for i := range postT.Data {
		d := postT.Data[i] - postS.Data[i]
		if d < 0 {
			d = -d
		}
		if d > maxPost {
			maxPost = d
		}
	}
	if maxPre > 1e-3 || maxPost > 1e-3 {
		t.Fatalf("tiled vs simd max diff pre=%g post=%g want <= 1e-3", maxPre, maxPost)
	}
}

func TestMHASimdFallsBackOnTinyLayers(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerMultiHeadAttention) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	net := newMHATestLayer(4, 2, 2)
	l := net.GetLayer(0, 0, 0, 0)
	input := poly.NewTensor[float32](2, 2, 4)
	net.SetSimdForward(false)
	preT, postT := poly.MHAForwardPolymorphic(l, input)
	net.SetSimdForward(true)
	preS, postS := poly.MHAForwardPolymorphic(l, input)
	for i := range preT.Data {
		if preT.Data[i] != preS.Data[i] || postT.Data[i] != postS.Data[i] {
			t.Fatalf("tiny MHA fallback mismatch at %d", i)
		}
	}
}

func TestMHABuildPopulatesSimdTileSizes(t *testing.T) {
	spec := []byte(`{
		"depth":1,"rows":1,"cols":1,"layers_per_cell":1,
		"layers":[{"z":0,"y":0,"x":0,"l":0,"type":"MHA","dtype":"FLOAT32",
			"d_model":64,"num_heads":4,"seq_length":8}]
	}`)
	net, err := poly.BuildNetworkFromJSON(spec)
	if err != nil {
		t.Fatal(err)
	}
	l := net.GetLayer(0, 0, 0, 0)
	if l.CPUSimdTileSizes == nil {
		t.Fatal("CPUSimdTileSizes nil after build")
	}
	scalar := l.GetCPUTileSize(poly.DTypeFloat32)
	simd := l.GetCPUSimdTileSize(poly.DTypeFloat32)
	if scalar <= 0 || simd <= 0 {
		t.Fatalf("scalar=%d simd=%d", scalar, simd)
	}
	if simd < 32 {
		t.Fatalf("simd tile %d capped below d_model/2 (head_dim=16); want >= 32", simd)
	}
	t.Logf("mha scalar=%d simd=%d", scalar, simd)
}
