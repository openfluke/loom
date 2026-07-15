package poly_test

import (
	"math"
	"testing"

	. "github.com/openfluke/loom/poly"
	"github.com/openfluke/loom/poly/simd"
)

func TestGemvQ4_0PackedMatchesDequant(t *testing.T) {
	outRows, inCols := 64, 128
	w := make([]float32, outRows*inCols)
	in := make([]float32, inCols)
	bias := make([]float32, outRows)
	for i := range w {
		w[i] = float32((i%17)-8) * 0.01
	}
	for i := range in {
		in[i] = float32((i%9)-4) * 0.05
	}
	for i := range bias {
		bias[i] = float32(i%3) * 0.001
	}
	scales, packed := PackQ4_0GPU(w)
	deq := DequantizeQ4_0GPUPacked(scales, packed)

	want := make([]float64, outRows)
	got := make([]float64, outRows)
	for o := 0; o < outRows; o++ {
		sum := float64(bias[o])
		for i := 0; i < inCols; i++ {
			sum += float64(in[i]) * float64(deq[o*inCols+i])
		}
		want[o] = sum
	}
	GemvQ4_0Packed(scales, packed, in, bias, got, outRows, inCols)

	for o := 0; o < outRows; o++ {
		if math.Abs(want[o]-got[o]) > 1e-4 {
			t.Fatalf("row %d: want %g got %g", o, want[o], got[o])
		}
	}
}

func TestGemvQ4_0PackedSIMDMatchesScalar(t *testing.T) {
	if !simd.SimdEnabled() {
		t.Skip("SIMD not linked for this arch")
	}
	outRows, inCols := 96, 256
	w := make([]float32, outRows*inCols)
	in := make([]float32, inCols)
	bias := make([]float32, outRows)
	for i := range w {
		w[i] = float32((i%17)-8) * 0.01
	}
	for i := range in {
		in[i] = float32((i%9)-4) * 0.05
	}
	for i := range bias {
		bias[i] = float32(i%3) * 0.001
	}
	scales, packed := PackQ4_0GPU(w)

	scalar := make([]float64, outRows)
	got := make([]float64, outRows)
	GemvQ4_0PackedParallel(scales, packed, in, bias, scalar, outRows, inCols, false)
	GemvQ4_0PackedParallel(scales, packed, in, bias, got, outRows, inCols, true)

	for o := 0; o < outRows; o++ {
		if math.Abs(scalar[o]-got[o]) > 1e-4 {
			t.Fatalf("row %d: scalar %g simd %g", o, scalar[o], got[o])
		}
	}
}

func TestMaterializeSkippedWhenPackedQ4CPU(t *testing.T) {
	l := &VolumetricLayer{
		Type:    LayerSwiGLU,
		Network: &VolumetricNetwork{UsePackedQ4CPU: true},
		WeightStore: &WeightStore{
			Q4_0Scales: map[DType][]float32{DType(100): {1}},
			Q4_0Packed: map[DType][]uint32{DType(100): {0}},
			Master:     []float32{1, 2, 3},
		},
		DType: DTypeInt4,
	}
	before := append([]float32(nil), l.WeightStore.Master...)
	l.MaterializeQ4_0ForCPU()
	if l.DType != DTypeInt4 {
		t.Fatalf("DType became %v, want Int4", l.DType)
	}
	if len(l.WeightStore.Master) != len(before) {
		t.Fatalf("Master resized under UsePackedQ4CPU")
	}
}
