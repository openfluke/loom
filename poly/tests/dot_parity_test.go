package poly_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/openfluke/loom/poly/simd"
)

// TestDotKernelParitySnapshot prints a stable checksum of simd.DotTile across many
// shapes so we can confirm a kernel rewrite is bit-identical to the prior kernel.
func TestDotKernelParitySnapshot(t *testing.T) {
	rng := rand.New(rand.NewSource(12345))
	var bits uint64
	total := 0.0
	for _, n := range []int{1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 63, 64, 127, 576, 1536, 2048, 4099} {
		x := make([]float32, n)
		w := make([]float32, n)
		for i := 0; i < n; i++ {
			x[i] = float32(rng.NormFloat64())
			w[i] = float32(rng.NormFloat64())
		}
		prev := rng.NormFloat64() * 3.0
		got := simd.DotTile(x, w, 0, n, prev)
		bits ^= math.Float64bits(got)
		total += got
	}
	fmt.Printf("PARITY bits=%#016x total=%.17g simd=%v\n", bits, total, simd.SimdEnabled())
}

func benchDot(b *testing.B, n int) {
	rng := rand.New(rand.NewSource(7))
	x := make([]float32, n)
	w := make([]float32, n)
	for i := 0; i < n; i++ {
		x[i] = float32(rng.NormFloat64())
		w[i] = float32(rng.NormFloat64())
	}
	b.SetBytes(int64(n * 8))
	b.ResetTimer()
	var acc float64
	for i := 0; i < b.N; i++ {
		acc = simd.DotTile(x, w, 0, n, acc)
	}
	dotSink = acc
}

var dotSink float64

func BenchmarkDot576(b *testing.B)  { benchDot(b, 576) }
func BenchmarkDot1536(b *testing.B) { benchDot(b, 1536) }
func BenchmarkDot2048(b *testing.B) { benchDot(b, 2048) }
