package simd

import (
	"math"
	"math/rand"
	"testing"
)

// avx2OrderReference reproduces the exact operation order of the amd64 AVX2
// kernel (dotF32AccF64Avx2) in portable Go: two 4-lane float64 accumulators over
// 8-wide blocks, the same horizontal reduction tree, prev added after the
// vector reduce, then a sequential scalar tail. DotTile on every arch must be
// bit-identical to this.
func avx2OrderReference(x, w []float32, prev float64) float64 {
	n := len(x)
	var y0, y1 [4]float64
	i := 0
	for ; i+8 <= n; i += 8 {
		for j := 0; j < 4; j++ {
			y0[j] += float64(x[i+j]) * float64(w[i+j])
		}
		for j := 0; j < 4; j++ {
			y1[j] += float64(x[i+4+j]) * float64(w[i+4+j])
		}
	}
	var s [4]float64
	for j := 0; j < 4; j++ {
		s[j] = y0[j] + y1[j]
	}
	acc := (s[0] + s[2]) + (s[1] + s[3])
	acc += prev
	for ; i < n; i++ {
		acc += float64(x[i]) * float64(w[i])
	}
	return acc
}

func TestDotTileMatchesAVX2Order(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	sizes := []int{0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 257, 576, 1000, 1536}
	for _, n := range sizes {
		for trial := 0; trial < 100; trial++ {
			x := make([]float32, n)
			w := make([]float32, n)
			for i := 0; i < n; i++ {
				x[i] = float32(rng.NormFloat64())
				w[i] = float32(rng.NormFloat64())
			}
			prev := rng.NormFloat64() * 3
			want := avx2OrderReference(x, w, prev)
			var got float64
			if n == 0 {
				got = DotTile(x, w, 0, 0, prev)
			} else {
				got = DotTile(x, w, 0, n, prev)
			}
			if math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("n=%d trial=%d: DotTile=%v (%016x) want %v (%016x)",
					n, trial, got, math.Float64bits(got), want, math.Float64bits(want))
			}
		}
	}
}

// scalarDotF64 is the pre-SIMD reference (plain float64 loop) used to size the
// NEON win in BenchmarkDotTile*.
func scalarDotF64(x, w []float32, prev float64) float64 {
	sum := prev
	for i := range x {
		sum += float64(x[i]) * float64(w[i])
	}
	return sum
}

var benchSink float64

func benchVec(n int) ([]float32, []float32) {
	x := make([]float32, n)
	w := make([]float32, n)
	for i := range x {
		x[i] = float32(i%7) * 0.5
		w[i] = float32(i%5) * 0.25
	}
	return x, w
}

func BenchmarkDotTileSimd1536(b *testing.B) {
	x, w := benchVec(1536)
	for i := 0; i < b.N; i++ {
		benchSink = DotTile(x, w, 0, 1536, 0)
	}
}

func BenchmarkDotTileScalar1536(b *testing.B) {
	x, w := benchVec(1536)
	for i := 0; i < b.N; i++ {
		benchSink = scalarDotF64(x, w, 0)
	}
}

func TestDotTileSubrange(t *testing.T) {
	// DotTile must honor i0/i1 and carry prev correctly across sub-tiles.
	x := make([]float32, 100)
	w := make([]float32, 100)
	for i := range x {
		x[i] = float32(i) * 0.01
		w[i] = float32(100-i) * 0.02
	}
	// Full range in one call vs two contiguous sub-ranges carrying prev.
	full := DotTile(x, w, 0, 100, 0)
	part := DotTile(x, w, 0, 37, 0)
	part = DotTile(x, w, 37, 100, part)
	// These need not be bit-identical (different grouping), but must be close.
	if math.Abs(full-part) > 1e-9*math.Abs(full)+1e-9 {
		t.Fatalf("subrange mismatch: full=%v part=%v", full, part)
	}
}
