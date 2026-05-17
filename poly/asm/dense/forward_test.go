package dense

import (
	"testing"

	"github.com/openfluke/loom/poly/asm"
	"github.com/openfluke/loom/poly/asm/dot"
	"github.com/openfluke/loom/poly/asm/matmul"
)

func TestForwardF32MatchesDot(t *testing.T) {
	const batch, inDim, outDim = 2, 4, 3
	input := []float32{1, 2, 3, 4, 0.5, 1, 1.5, 2}
	weights := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	want := make([]float32, batch*outDim)
	matmul.ForwardGEMVF32(want, input, weights, batch, inDim, outDim, false, 32)

	got := make([]float32, batch*outDim)
	Forward(got, input, weights, batch, inDim, outDim, false, 32)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("index %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestForwardInt8(t *testing.T) {
	const batch, inDim, outDim = 2, 4, 3
	input := []int8{1, 2, 3, 4, 0, -1, 2, 1}
	weights := []int8{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	got := make([]int8, batch*outDim)
	Forward(got, input, weights, batch, inDim, outDim, false, 32)
	if got[0] != 1 || got[4] != -1 {
		t.Fatalf("got %v", got)
	}
}

func TestDotF32Asm(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	w := []float32{1, 0, 0, 0}
	if !asm.Enabled() {
		t.Skip("no assembly on this platform")
	}
	if dot.F32(x, w, 4) != 1 {
		t.Fatal("asm dot mismatch")
	}
}
