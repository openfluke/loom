package simd_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/openfluke/loom/poly/simd"
)

func saxpyRef(acc []float64, alpha float64, x []float32, n int) {
	for i := 0; i < n; i++ {
		acc[i] += alpha * float64(x[i])
	}
}

func TestSaxpyF32AccF64MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, n := range []int{0, 1, 3, 4, 7, 8, 15, 16, 31, 64, 128, 256, 576} {
		for trial := 0; trial < 40; trial++ {
			want := make([]float64, n)
			got := make([]float64, n)
			x := make([]float32, n)
			for i := 0; i < n; i++ {
				want[i] = rng.NormFloat64()
				got[i] = want[i]
				x[i] = float32(rng.NormFloat64())
			}
			alpha := rng.NormFloat64() * 0.5
			saxpyRef(want, alpha, x, n)
			simd.SaxpyF32AccF64(got, alpha, x, n)
			for i := 0; i < n; i++ {
				if d := math.Abs(got[i] - want[i]); d > 1e-12 {
					t.Fatalf("n=%d trial=%d i=%d diff=%g", n, trial, i, d)
				}
			}
		}
	}
}
