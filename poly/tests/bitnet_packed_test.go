package poly_test

import (
	"math/rand"
	"testing"

	. "github.com/openfluke/loom/poly"
	"github.com/openfluke/loom/poly/simd"
)

// TestBitNetPackedMatVecMatchesScalar sweeps several row/col shapes (including
// column counts that are NOT multiples of 64, to exercise row padding) and
// requires the packed-2-bit SIMD forward to be bit-identical to the scalar
// packed path. Integer math is exact, so "close" is not acceptable — it must match.
func TestBitNetPackedMatVecMatchesScalar(t *testing.T) {
	if !simd.BitNetPackedAvailable() {
		t.Skip("packed ternary kernel not available on this GOARCH")
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
		_, packed := DenseForwardPolymorphic(build(), input)
		SetBitNetTernarySimdForward(false)

		if diff := maxAbsDiffSlice(scalar.Data, packed.Data); diff != 0 {
			t.Fatalf("rows=%d cols=%d: packed vs scalar diff = %g, want 0", rows, cols, diff)
		}
	}
}

func benchBitNetPackedDense(b *testing.B, rows, cols int) {
	rng := rand.New(rand.NewSource(7))
	raw := make([]float32, rows*cols)
	for i := range raw {
		raw[i] = float32(rng.Intn(3) - 1)
	}
	net := NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = LayerDense
	l.Activation = ActivationLinear
	l.DType = DTypeTernary
	l.InputHeight = cols
	l.OutputHeight = rows
	l.WeightStore = NewWeightStore(cols * rows)
	copy(l.WeightStore.Master, raw)
	net.UseExactDType = true
	SetBitNetTernarySimdForward(true)
	SetBitNetTL1Forward(false)
	defer SetBitNetTernarySimdForward(false)

	input := NewTensorFromSlice(deterministicWeights(cols), 1, cols)
	_, _ = DenseForwardPolymorphic(l, input) // warm (build packed once)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = DenseForwardPolymorphic(l, input)
	}
}

func BenchmarkBitNetPackedDownProj(b *testing.B) { benchBitNetPackedDense(b, 2560, 6912) }
func BenchmarkBitNetPackedGateProj(b *testing.B) { benchBitNetPackedDense(b, 6912, 2560) }
