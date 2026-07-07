package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func newCNN2TestNet(inC, filters, spatial, kSize int) *poly.VolumetricNetwork {
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = poly.LayerCNN2
	l.InputChannels = inC
	l.Filters = filters
	l.InputHeight = spatial
	l.InputWidth = spatial
	l.OutputHeight = spatial
	l.OutputWidth = spatial
	l.KernelSize = kSize
	l.Stride = 1
	l.Padding = 1
	l.Activation = poly.ActivationReLU
	l.DType = poly.DTypeFloat32
	wCount := filters * inC * kSize * kSize
	l.WeightStore = poly.NewWeightStore(wCount)
	for i := range l.WeightStore.Master {
		l.WeightStore.Master[i] = 0.001 * float32((i%23)+1)
	}
	l.EnableMultiCoreTiling = false
	return net
}

func TestCNN2SimdMatchesTiledFloat32(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerCNN2) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}

	inC, filters, spatial, kSize := 16, 16, 16, 3
	net := newCNN2TestNet(inC, filters, spatial, kSize)
	input := poly.NewTensor[float32](4, inC, spatial, spatial)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%17)
	}

	net.SetSimdForward(false)
	preT, postT := poly.CNN2ForwardPolymorphic(net.GetLayer(0, 0, 0, 0), input)

	net.SetSimdForward(true)
	preS, postS := poly.CNN2ForwardPolymorphic(net.GetLayer(0, 0, 0, 0), input)

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

func TestCNN2SimdFallsBackOnTinyLayers(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerCNN2) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	net := newCNN2TestNet(1, 1, 4, 1)
	l := net.GetLayer(0, 0, 0, 0)
	input := poly.NewTensor[float32](2, 1, 4, 4)
	net.SetSimdForward(false)
	preT, postT := poly.CNN2ForwardPolymorphic(l, input)
	net.SetSimdForward(true)
	preS, postS := poly.CNN2ForwardPolymorphic(l, input)
	for i := range preT.Data {
		if preT.Data[i] != preS.Data[i] || postT.Data[i] != postS.Data[i] {
			t.Fatalf("tiny CNN2 fallback mismatch at %d", i)
		}
	}
}

func TestCNN2BuildPopulatesSimdTileSizes(t *testing.T) {
	spec := []byte(`{
		"depth":1,"rows":1,"cols":1,"layers_per_cell":1,
		"layers":[{"z":0,"y":0,"x":0,"l":0,"type":"CNN2","dtype":"FLOAT32",
			"input_channels":16,"filters":16,"input_height":16,"input_width":16,
			"output_height":16,"output_width":16,"kernel_size":3,"stride":1,"padding":1}]
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
	t.Logf("cnn2 scalar=%d simd=%d", scalar, simd)
}
