package dot_test

import (
	"testing"

	"github.com/openfluke/loom/poly/asm/dot"
)

func TestI8TileNativeI64MatchesGo(t *testing.T) {
	x := []int8{1, -2, 3, 4}
	w := []int8{2, 1, -1, 2}
	got := dot.I8TileNativeI64(x, w, len(x))
	want := dot.I8TileNativeI64Go(x, w, len(x))
	if got != want {
		t.Fatalf("got %d want %d", got, want)
	}
}

func TestNibblePackedRowNativeI64(t *testing.T) {
	x := []uint8{1, 2, 3, 4}
	packed := []uint32{0x21004321}
	got := dot.NibblePackedRowNativeI64(x, packed, 0, 4)
	want := dot.NibblePackedRowNativeI64Go(x, packed, 0, 4)
	if got != want {
		t.Fatalf("got %d want %d", got, want)
	}
}
