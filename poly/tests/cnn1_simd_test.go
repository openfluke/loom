package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func newCNN1TestNet(inC, filters, seqLen, kSize int) *poly.VolumetricNetwork {
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = poly.LayerCNN1
	l.InputChannels = inC
	l.Filters = filters
	l.InputHeight = seqLen
	l.OutputHeight = seqLen
	l.KernelSize = kSize
	l.Stride = 1
	l.Padding = 1
	l.Activation = poly.ActivationReLU
	l.DType = poly.DTypeFloat32
	wCount := filters * inC * kSize
	l.WeightStore = poly.NewWeightStore(wCount)
	for i := range l.WeightStore.Master {
		l.WeightStore.Master[i] = 0.001 * float32((i%23)+1)
	}
	l.EnableMultiCoreTiling = false
	return net
}

func TestCNN1SimdMatchesTiledFloat32(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerCNN1) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}

	inC, filters, seqLen, kSize := 16, 16, 16, 3
	net := newCNN1TestNet(inC, filters, seqLen, kSize)
	input := poly.NewTensor[float32](4, inC, seqLen)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%17)
	}

	net.SetSimdForward(false)
	preT, postT := poly.CNN1ForwardPolymorphic(net.GetLayer(0, 0, 0, 0), input)

	net.SetSimdForward(true)
	preS, postS := poly.CNN1ForwardPolymorphic(net.GetLayer(0, 0, 0, 0), input)

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

func TestCNN1SimdFallsBackOnTinyLayers(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerCNN1) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	net := newCNN1TestNet(1, 1, 4, 1)
	l := net.GetLayer(0, 0, 0, 0)
	input := poly.NewTensor[float32](2, 1, 4)
	net.SetSimdForward(false)
	preT, postT := poly.CNN1ForwardPolymorphic(l, input)
	net.SetSimdForward(true)
	preS, postS := poly.CNN1ForwardPolymorphic(l, input)
	for i := range preT.Data {
		if preT.Data[i] != preS.Data[i] || postT.Data[i] != postS.Data[i] {
			t.Fatalf("tiny CNN1 fallback mismatch at %d", i)
		}
	}
}

func TestCNN1BuildPopulatesSimdTileSizes(t *testing.T) {
	spec := []byte(`{
		"depth":1,"rows":1,"cols":1,"layers_per_cell":1,
		"layers":[{"z":0,"y":0,"x":0,"l":0,"type":"CNN1","dtype":"FLOAT32",
			"input_channels":16,"filters":16,"input_height":16,"output_height":16,
			"kernel_size":3,"stride":1,"padding":1}]
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
	t.Logf("cnn1 scalar=%d simd=%d", scalar, simd)
}

func TestCNN1SimdMatchesTiledNarrowChannels(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerCNN1) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	// 2×2×2 grid shape: inC=8, kernelVol=24 at k=3
	net := newCNN1TestNet(8, 8, 8, 3)
	input := poly.NewTensor[float32](4, 8, 8)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%13)
	}
	net.SetSimdForward(false)
	preT, postT := poly.CNN1ForwardPolymorphic(net.GetLayer(0, 0, 0, 0), input)
	net.SetSimdForward(true)
	preS, postS := poly.CNN1ForwardPolymorphic(net.GetLayer(0, 0, 0, 0), input)
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
		t.Fatalf("narrow channel tiled vs simd max diff pre=%g post=%g", maxPre, maxPost)
	}
}
