package poly_test

import (
	"math/rand"
	"testing"

	"github.com/openfluke/loom/poly/simd"
)

func saxpyI8ScalarRef(gradW []int32, rowOff int, input []int8, scale int32, n int) {
	for i := 0; i < n; i++ {
		gradW[rowOff+i] += int32(input[i]) * scale
	}
}

func TestSaxpyI8ScaleI32AccMatchesScalar(t *testing.T) {
	simd.SetInt8DotSimdForward(true)
	defer simd.SetInt8DotSimdForward(false)

	rng := rand.New(rand.NewSource(99))
	for _, n := range []int{0, 1, 7, 8, 15, 16, 31, 64} {
		gradW := make([]int32, n+4)
		want := make([]int32, n+4)
		input := make([]int8, n)
		for i := range input {
			input[i] = int8(rng.Intn(255) - 128)
			gradW[i+2] = int32(rng.Intn(1000) - 500)
			want[i+2] = gradW[i+2]
		}
		scale := int32(rng.Intn(20) - 10)
		simd.SaxpyI8ScaleI32Acc(gradW, 2, input, scale, n)
		saxpyI8ScalarRef(want, 2, input, scale, n)
		for i := 0; i < n; i++ {
			if gradW[i+2] != want[i+2] {
				t.Fatalf("n=%d i=%d got=%d want=%d", n, i, gradW[i+2], want[i+2])
			}
		}
	}
}
