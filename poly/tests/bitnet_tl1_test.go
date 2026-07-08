package poly_test

import (
	"testing"

	. "github.com/openfluke/loom/poly"
	"github.com/openfluke/loom/poly/simd"
)

// TestBitNetTL1MatVecMatchesScalar requires TL1 SIMD forward to match the scalar
// packed ternary path across several shapes (including odd column counts).
func TestBitNetTL1MatVecMatchesScalar(t *testing.T) {
	if !simd.BitNetTL1Available() {
		t.Skip("TL1 kernel not available on this GOARCH")
	}
	shapes := [][2]int{
		{40, 70}, {8, 64}, {33, 65}, {16, 128}, {64, 2560}, {128, 6912}, {7, 191},
	}
	for _, s := range shapes {
		rows, cols := s[0], s[1]
		weights := deterministicWeights(rows * cols)
		input := NewTensorFromSlice(deterministicWeights(2*cols), 2, cols)

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

		SetBitNetTernarySimdForward(false)
		_, scalar := DenseForwardPolymorphic(build(), input)

		SetBitNetTernarySimdForward(true)
		SetBitNetTL1Forward(true)
		defer SetBitNetTL1Forward(false)
		_, tl1 := DenseForwardPolymorphic(build(), input)
		SetBitNetTernarySimdForward(false)

		if diff := maxAbsDiffSlice(scalar.Data, tl1.Data); diff != 0 {
			t.Fatalf("rows=%d cols=%d: tl1 vs scalar diff = %g, want 0", rows, cols, diff)
		}
	}
}

func BenchmarkBitNetTL1DownProj(b *testing.B) { benchBitNetTL1Dense(b, 2560, 6912) }
func BenchmarkBitNetTL1GateProj(b *testing.B) { benchBitNetTL1Dense(b, 6912, 2560) }

func benchBitNetTL1Dense(b *testing.B, rows, cols int) {
	if !simd.BitNetTL1Available() {
		b.Skip("TL1 not available")
	}
	net := NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = LayerDense
	l.Activation = ActivationLinear
	l.DType = DTypeTernary
	l.InputHeight = cols
	l.OutputHeight = rows
	l.WeightStore = NewWeightStore(cols * rows)
	copy(l.WeightStore.Master, deterministicWeights(rows*cols))
	net.UseExactDType = true
	SetBitNetTernarySimdForward(true)
	SetBitNetTL1Forward(true)
	defer func() {
		SetBitNetTL1Forward(false)
		SetBitNetTernarySimdForward(false)
	}()

	input := NewTensorFromSlice(deterministicWeights(cols), 1, cols)
	_, _ = DenseForwardPolymorphic(l, input)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = DenseForwardPolymorphic(l, input)
	}
}
