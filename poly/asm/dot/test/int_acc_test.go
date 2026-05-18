package dot_test

import (
	"testing"

	"github.com/openfluke/loom/poly/asm/dot"
)

func TestIntTileAccF64MatchesGo(t *testing.T) {
	cases := []struct {
		name string
		run  func(x, w []int8, n int) float64
		goFn func(x, w []int8, n int) float64
	}{
		{"i8", dot.I8TileAccF64, dot.I8TileAccF64Go},
	}
	x := []int8{1, -2, 3, 4}
	w := []int8{2, 1, -1, 2}
	for _, c := range cases {
		got := c.run(x, w, len(x))
		want := c.goFn(x, w, len(x))
		if got != want {
			t.Fatalf("%s: got %v want %v", c.name, got, want)
		}
	}
}

func TestI32TileAccF64(t *testing.T) {
	x := []int32{1000, -2000, 3, 4}
	w := []int32{2, 1, -1, 2}
	got := dot.I32TileAccF64(x, w, len(x))
	want := dot.I32TileAccF64Go(x, w, len(x))
	if got != want {
		t.Fatalf("got %v want %v", got, want)
	}
}
