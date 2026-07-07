package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestBuildNetworkPopulatesPerDtypeTileSizes(t *testing.T) {
	spec := []byte(`{
		"depth":1,"rows":1,"cols":1,"layers_per_cell":1,
		"layers":[{"z":0,"y":0,"x":0,"l":0,"type":"DENSE","dtype":"FLOAT32",
			"input_height":64,"output_height":64}]
	}`)
	net, err := poly.BuildNetworkFromJSON(spec)
	if err != nil {
		t.Fatal(err)
	}
	l := net.GetLayer(0, 0, 0, 0)
	if l.CPUTileSizes == nil {
		t.Fatal("CPUTileSizes nil after BuildNetworkFromJSON")
	}
	if l.CPUSimdTileSizes == nil {
		t.Fatal("CPUSimdTileSizes nil after BuildNetworkFromJSON")
	}
	for _, dt := range []poly.DType{poly.DTypeFloat32, poly.DTypeInt8, poly.DTypeUint64} {
		scalar := l.GetCPUTileSize(dt)
		simd := l.GetCPUSimdTileSize(dt)
		if scalar <= 0 {
			t.Fatalf("scalar tile for %v = %d", dt, scalar)
		}
		if simd <= 0 {
			t.Fatalf("simd tile for %v = %d", dt, simd)
		}
		t.Logf("%v scalar=%d simd=%d", dt, scalar, simd)
	}
}

func TestMorphScaleForStackDepthUint(t *testing.T) {
	got := poly.MasterWeightScaleForStackDepth(poly.DTypeUint64, 189, 32)
	if got >= 1.0 || got <= 0 {
		t.Fatalf("expected (0,1) master scale, got %g", got)
	}
}

func TestEnsureRuntimeTileSizesLazy(t *testing.T) {
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = poly.LayerDense
	l.InputHeight = 32
	l.OutputHeight = 32
	l.CPUTileSizes = nil
	l.EnsureRuntimeTileSizes()
	if l.CPUTileSizes == nil || l.CPUSimdTileSizes == nil {
		t.Fatal("EnsureRuntimeTileSizes did not populate maps")
	}
}
