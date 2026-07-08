package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestDenseSimdMatchesTiledFloat32(t *testing.T) {
	if !poly.Plan9SimdEnabled() {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}

	net := poly.NewVolumetricNetwork(1, 1, 1, 3)
	dim := 256
	batch := 4
	for z := 0; z < 1; z++ {
		for y := 0; y < 1; y++ {
			for x := 0; x < 1; x++ {
				for i := 0; i < 3; i++ {
					l := net.GetLayer(z, y, x, i)
					l.Type = poly.LayerDense
					l.InputHeight = dim
					l.OutputHeight = dim
					l.Activation = poly.ActivationReLU
					l.DType = poly.DTypeFloat32
					l.TileSize = 32
					l.EnableMultiCoreTiling = true
					wCount := dim * dim
					l.WeightStore = poly.NewWeightStore(wCount)
					for j := range l.WeightStore.Master {
						l.WeightStore.Master[j] = 0.001 * float32((j%17)+1)
					}
				}
			}
		}
	}

	input := poly.NewTensor[float32](batch, dim)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%11)
	}

	net.SetSimdForward(false)
	outTiled, _, _ := poly.ForwardPolymorphic(net, input)

	net.SetSimdForward(true)
	outSimd, _, _ := poly.ForwardPolymorphic(net, input)

	if len(outTiled.Data) != len(outSimd.Data) {
		t.Fatalf("length mismatch tiled=%d simd=%d", len(outTiled.Data), len(outSimd.Data))
	}
	var maxDiff float32
	for i := range outTiled.Data {
		d := outTiled.Data[i] - outSimd.Data[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 1e-5 {
		t.Fatalf("tiled vs simd max diff %g want <= 1e-5", maxDiff)
	}
}

// Explicit SIMD is honored even on narrow layers (no silent fallback to tiled);
// the SIMD dot must still match the tiled result.
func TestDenseSimdMatchesTiledOnTinyLayers(t *testing.T) {
	if !poly.Plan9SimdEnabled() {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = poly.LayerDense
	l.InputHeight = 4
	l.OutputHeight = 4
	l.DType = poly.DTypeFloat32
	l.WeightStore = poly.NewWeightStore(16)
	for j := range l.WeightStore.Master {
		l.WeightStore.Master[j] = 0.01 * float32((j%7)+1)
	}
	input := poly.NewTensor[float32](4, 4)
	for i := range input.Data {
		input.Data[i] = 0.02 * float32((i%5)+1)
	}

	net.SetSimdForward(true)
	outSimd, _, _ := poly.ForwardPolymorphic(net, input)
	net.SetSimdForward(false)
	outTiled, _, _ := poly.ForwardPolymorphic(net, input)

	for i := range outTiled.Data {
		d := outTiled.Data[i] - outSimd.Data[i]
		if d < 0 {
			d = -d
		}
		if d > 1e-5 {
			t.Fatalf("tiny layer simd vs tiled mismatch at %d: %g vs %g", i, outTiled.Data[i], outSimd.Data[i])
		}
	}
}
