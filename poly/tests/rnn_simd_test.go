package poly_test

import (
	"testing"
	"time"

	"github.com/openfluke/loom/poly"
)

func newRNNTestNet(inputSize, hiddenSize, seqLen int) *poly.VolumetricNetwork {
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = poly.LayerRNN
	l.InputHeight = inputSize
	l.OutputHeight = hiddenSize
	l.SeqLength = seqLen
	l.Activation = poly.ActivationTanh
	l.DType = poly.DTypeFloat32
	wCount := hiddenSize*inputSize + hiddenSize*hiddenSize + hiddenSize
	l.WeightStore = poly.NewWeightStore(wCount)
	for i := range l.WeightStore.Master {
		l.WeightStore.Master[i] = 0.001 * float32((i%23)+1)
	}
	l.EnableMultiCoreTiling = false
	return net
}

func TestRNNSimdMatchesTiledFloat32(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerRNN) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}

	inputSize, hiddenSize, seqLen := 64, 64, 8
	net := newRNNTestNet(inputSize, hiddenSize, seqLen)
	input := poly.NewTensor[float32](4, seqLen, inputSize)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%17)
	}

	net.SetSimdForward(false)
	preT, postT := poly.RNNForwardPolymorphic(net.GetLayer(0, 0, 0, 0), input)

	net.SetSimdForward(true)
	preS, postS := poly.RNNForwardPolymorphic(net.GetLayer(0, 0, 0, 0), input)

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

// Explicit SIMD is honored even on narrow hidden dims; result must match tiled.
func TestRNNSimdMatchesTiledOnTinyLayers(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerRNN) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	net := newRNNTestNet(4, 4, 1)
	l := net.GetLayer(0, 0, 0, 0)
	input := poly.NewTensor[float32](2, 1, 4)
	for i := range input.Data {
		input.Data[i] = 0.02 * float32((i%5)+1)
	}
	net.SetSimdForward(false)
	preT, postT := poly.RNNForwardPolymorphic(l, input)
	net.SetSimdForward(true)
	preS, postS := poly.RNNForwardPolymorphic(l, input)
	for i := range preT.Data {
		if absF32rnn(preT.Data[i]-preS.Data[i]) > 1e-5 || absF32rnn(postT.Data[i]-postS.Data[i]) > 1e-5 {
			t.Fatalf("tiny RNN simd vs tiled mismatch at %d", i)
		}
	}
}

func absF32rnn(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// Demonstrates the auto tile-sizing adapts to ANY user-defined RNN width
// (not the seven_layer test dims) and shows the SIMD crossover with size.
func TestRNNSimdCrossoverByWidth(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerRNN) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	seqLen, batch := 8, 8
	for _, dim := range []int{4, 16, 64, 256, 512, 1024} {
		net := newRNNTestNet(dim, dim, seqLen)
		l := net.GetLayer(0, 0, 0, 0)
		l.EnableMultiCoreTiling = true
		input := poly.NewTensor[float32](batch, seqLen, dim)
		for i := range input.Data {
			input.Data[i] = 0.01 * float32(i%13)
		}
		simdTile := l.GetCPUSimdTileSize(poly.DTypeFloat32)
		scalarTile := l.GetCPUTileSize(poly.DTypeFloat32)

		warm := func(simd bool) {
			net.SetSimdForward(simd)
			for i := 0; i < 3; i++ {
				poly.RNNForwardPolymorphic(l, input)
			}
		}
		timeIt := func(simd bool) time.Duration {
			net.SetSimdForward(simd)
			iters := 50
			t0 := time.Now()
			for i := 0; i < iters; i++ {
				poly.RNNForwardPolymorphic(l, input)
			}
			return time.Since(t0) / time.Duration(iters)
		}
		warm(false)
		tiled := timeIt(false)
		warm(true)
		sd := timeIt(true)
		ratio := float64(tiled) / float64(sd)
		t.Logf("dim=%-5d scalarTile=%-4d simdTile=%-4d tiled=%-10v simd=%-10v speedup=%.2fx",
			dim, scalarTile, simdTile, tiled, sd, ratio)
	}
}

func TestRNNBuildPopulatesSimdTileSizes(t *testing.T) {
	spec := []byte(`{
		"depth":1,"rows":1,"cols":1,"layers_per_cell":1,
		"layers":[{"z":0,"y":0,"x":0,"l":0,"type":"RNN","dtype":"FLOAT32",
			"input_height":32,"output_height":32,"seq_length":8}]
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
	t.Logf("rnn scalar=%d simd=%d", scalar, simd)
}
