package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func newSwiGLUTestNet(inDim, interDim, seqLen int) *poly.VolumetricNetwork {
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = poly.LayerSwiGLU
	l.Activation = poly.ActivationSilu
	l.InputHeight = inDim
	l.OutputHeight = interDim
	l.DType = poly.DTypeFloat32
	wSize := inDim * interDim
	total := 3*wSize + 2*interDim + inDim
	l.WeightStore = poly.NewWeightStore(total)
	for i := range l.WeightStore.Master {
		l.WeightStore.Master[i] = 0.001 * float32((i%17)+1)
	}
	l.EnableMultiCoreTiling = false
	return net
}

func TestSwiGLUSimdMatchesTiledFloat32(t *testing.T) {
	if !poly.Plan9SimdEnabled() {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}

	inDim, interDim, seqLen := 64, 128, 4
	net := newSwiGLUTestNet(inDim, interDim, seqLen)
	input := poly.NewTensor[float32](seqLen, inDim)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%11)
	}

	net.SetSimdForward(false)
	preT, postT := poly.SwiGLUForwardPolymorphic(net.GetLayer(0, 0, 0, 0), input)

	net.SetSimdForward(true)
	preS, postS := poly.SwiGLUForwardPolymorphic(net.GetLayer(0, 0, 0, 0), input)

	if len(preT.Data) != len(preS.Data) || len(postT.Data) != len(postS.Data) {
		t.Fatalf("shape mismatch pre=%d/%d post=%d/%d", len(preT.Data), len(preS.Data), len(postT.Data), len(postS.Data))
	}
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
	if maxPre > 1e-4 || maxPost > 1e-4 {
		t.Fatalf("tiled vs simd max diff pre=%g post=%g want <= 1e-4", maxPre, maxPost)
	}
}

func TestSwiGLUSimdFallsBackOnTinyLayers(t *testing.T) {
	if !poly.Plan9SimdEnabled() {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	net := newSwiGLUTestNet(4, 4, 2)
	net.SetSimdForward(true)
	input := poly.NewTensor[float32](2, 4)
	for i := range input.Data {
		input.Data[i] = 0.1
	}
	l := net.GetLayer(0, 0, 0, 0)
	net.SetSimdForward(false)
	preT, postT := poly.SwiGLUForwardPolymorphic(l, input)
	net.SetSimdForward(true)
	preS, postS := poly.SwiGLUForwardPolymorphic(l, input)
	for i := range preT.Data {
		if preT.Data[i] != preS.Data[i] || postT.Data[i] != postS.Data[i] {
			t.Fatalf("tiny layer fallback mismatch at %d", i)
		}
	}
}

func TestSwiGLUBuildPopulatesSimdTileSizes(t *testing.T) {
	spec := []byte(`{
		"depth":1,"rows":1,"cols":1,"layers_per_cell":1,
		"layers":[{"z":0,"y":0,"x":0,"l":0,"type":"SWIGLU","dtype":"FLOAT32",
			"input_height":64,"output_height":128}]
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
	t.Logf("swiglu scalar=%d simd=%d", scalar, simd)
}
