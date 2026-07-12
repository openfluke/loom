package poly_test

import (
	"math/rand"
	"testing"

	"github.com/openfluke/loom/poly/simd"
)

func dotI8ScalarRef(a, b []int8, aOff, bOff, n int, prev int32) int32 {
	sum := prev
	for i := 0; i < n; i++ {
		sum += int32(a[aOff+i]) * int32(b[bOff+i])
	}
	return sum
}

func TestDotI8TileMatchesScalar(t *testing.T) {
	simd.SetInt8DotSimdForward(true)
	defer simd.SetInt8DotSimdForward(false)

	rng := rand.New(rand.NewSource(42))
	for _, n := range []int{0, 1, 7, 8, 15, 16, 31, 32, 64, 127} {
		a := make([]int8, n)
		b := make([]int8, n)
		for i := range a {
			a[i] = int8(rng.Intn(255) - 128)
			b[i] = int8(rng.Intn(255) - 128)
		}
		prev := int32(rng.Intn(1000) - 500)
		got := simd.DotI8Tile(a, b, 0, 0, n, prev)
		want := dotI8ScalarRef(a, b, 0, 0, n, prev)
		if got != want {
			t.Fatalf("n=%d prev=%d: simd=%d scalar=%d", n, prev, got, want)
		}
	}
}

func TestDotI8TileOffsets(t *testing.T) {
	simd.SetInt8DotSimdForward(true)
	defer simd.SetInt8DotSimdForward(false)

	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	b := []int8{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}
	got := simd.DotI8Tile(a, b, 2, 3, 6, 100)
	want := dotI8ScalarRef(a, b, 2, 3, 6, 100)
	if got != want {
		t.Fatalf("offset dot: simd=%d scalar=%d", got, want)
	}
}
