package matmul_test

import (
	"testing"

	"github.com/openfluke/loom/poly/asm/matmul"
)

func TestForwardTiledF32MatchesInputTiling(t *testing.T) {
	const batch, inDim, outDim = 2, 8, 3
	tileSize := 3
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4}
	weights := []float32{
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0,
	}
	want := make([]float32, batch*outDim)
	matmul.ForwardTiledF32(want, input, weights, batch, inDim, outDim, false, 32)

	got := make([]float32, batch*outDim)
	matmul.ForwardTiledF32(got, input, weights, batch, inDim, outDim, false, tileSize)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("index %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestForwardTiledInt8MC(t *testing.T) {
	const batch, inDim, outDim = 2, 4, 3
	input := []int8{1, 2, 3, 4, 0, -1, 2, 1}
	weights := []int8{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	got := make([]int8, batch*outDim)
	matmul.ForwardTiledI8(got, input, weights, batch, inDim, outDim, true, 2)
	if got[0] != 1 || got[4] != -1 {
		t.Fatalf("got %v", got)
	}
}
