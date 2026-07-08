package poly_test

import (
	"testing"

	. "github.com/openfluke/loom/poly"
)

// microsoft/bitnet-b1.58-2B-4T style projection dims. seqLen=1 is the decode
// (single-token) case that dominates chat latency. Compares the packed-ternary
// CPU matvec with the NEON/AVX2 MAD kernel off vs on.
func benchBitNetDense(b *testing.B, rows, cols int, simdOn bool) {
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

	SetBitNetTernarySimdForward(simdOn)
	defer SetBitNetTernarySimdForward(false)
	if simdOn && !BitNetTernarySimdActive() {
		b.Skip("BitNet ternary SIMD not available on this GOARCH")
	}

	input := NewTensorFromSlice(deterministicWeights(cols), 1, cols)
	// Warm up so the one-time packed matrix + code unpack is not timed.
	_, _ = DenseForwardPolymorphic(l, input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = DenseForwardPolymorphic(l, input)
	}
}

func BenchmarkBitNetDenseDownScalar(b *testing.B) { benchBitNetDense(b, 2560, 6912, false) }
func BenchmarkBitNetDenseDownSimd(b *testing.B)   { benchBitNetDense(b, 2560, 6912, true) }
func BenchmarkBitNetDenseGateScalar(b *testing.B) { benchBitNetDense(b, 6912, 2560, false) }
func BenchmarkBitNetDenseGateSimd(b *testing.B)   { benchBitNetDense(b, 6912, 2560, true) }
