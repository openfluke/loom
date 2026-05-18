package dot_test

import (
	"testing"

	"github.com/openfluke/loom/poly/asm/dot"
)

func TestF32TileAccF64MatchesGo(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	w := []float32{0.5, -1, 2, 0.25}
	want := dot.F32TileAccF64Go(x, w, len(x))
	got := dot.F32TileAccF64(x, w, len(x))
	if got != want {
		t.Fatalf("got %v want %v", got, want)
	}
}
