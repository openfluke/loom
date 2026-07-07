package poly_test

import (
	"testing"

	. "github.com/openfluke/loom/poly"
)

// TestBitNetTernaryDenseSimdMatchesScalar drives a packed-ternary Dense layer
// through the exported forward API with the AVX2 MAD kernel off then on, and
// requires bit-identical output (both paths do exact integer accumulation).
// Cols=70 is deliberately not a multiple of 32 to exercise row padding.
func TestBitNetTernaryDenseSimdMatchesScalar(t *testing.T) {
	rows, cols := 40, 70
	weights := deterministicWeights(rows * cols)

	build := func() *VolumetricLayer {
		net := NewVolumetricNetwork(1, 1, 1, 1)
		l := net.GetLayer(0, 0, 0, 0)
		l.Type = LayerDense
		l.Activation = ActivationLinear
		l.DType = DTypeTernary
		l.InputHeight = cols
		l.OutputHeight = rows
		l.WeightStore = NewWeightStore(cols * rows)
		copy(l.WeightStore.Master, weights)
		net.UseExactDType = true
		return l
	}
	input := NewTensorFromSlice(deterministicWeights(2*cols), 2, cols)

	SetBitNetTernarySimdForward(false)
	_, outScalar := DenseForwardPolymorphic(build(), input)

	SetBitNetTernarySimdForward(true)
	defer SetBitNetTernarySimdForward(false)
	if !BitNetTernarySimdActive() {
		t.Skip("AVX2 BitNet ternary SIMD not available on this GOARCH")
	}
	_, outSimd := DenseForwardPolymorphic(build(), input)

	if diff := maxAbsDiffSlice(outScalar.Data, outSimd.Data); diff != 0 {
		t.Fatalf("SIMD vs scalar diff = %g, want 0", diff)
	}
}
