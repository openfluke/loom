package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly/simd"
)

func TestDotU8TileMatchesScalar(t *testing.T) {
	simd.SetInt8DotSimdForward(true)
	t.Cleanup(func() { simd.SetInt8DotSimdForward(false) })

	a := make([]uint8, 32)
	b := make([]uint8, 32)
	for i := range a {
		a[i] = uint8((i*7 + 3) % 200)
		b[i] = uint8((i*11 + 5) % 180)
	}
	want := int32(0)
	for i := 0; i < 32; i++ {
		want += int32(a[i]) * int32(b[i])
	}
	got := simd.DotU8Tile(a, b, 0, 0, 32, 0)
	if got != want {
		t.Fatalf("DotU8Tile = %d want %d", got, want)
	}
}

func TestSaxpyI8ShiftedInputGradMatchesScalar(t *testing.T) {
	simd.SetInt8DotSimdForward(true)
	t.Cleanup(func() { simd.SetInt8DotSimdForward(false) })

	weights := make([]int8, 16)
	inputGrad := make([]int32, 16)
	want := make([]int32, 16)
	for i := range weights {
		weights[i] = int8((i*5)%127 - 63)
		want[i] = int32(i * 3)
		inputGrad[i] = int32(i * 3)
	}
	gradOut := int32(42)
	for i := 0; i < 16; i++ {
		want[i] += (int32(weights[i]) * gradOut) >> 8
	}
	simd.SaxpyI8ShiftedInputGradAcc(inputGrad, weights, 0, gradOut, 16)
	for i := range want {
		if inputGrad[i] != want[i] {
			t.Fatalf("grad[%d] = %d want %d", i, inputGrad[i], want[i])
		}
	}
}
