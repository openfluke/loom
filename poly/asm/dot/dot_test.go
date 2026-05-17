package dot

import (
	"testing"

	"github.com/openfluke/loom/poly/asm"
)

func TestF32Go(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	w := []float32{1, 0, 0, 0}
	if F32Go(x, w, 4) != 1 {
		t.Fatal("go dot mismatch")
	}
}

func TestF32Asm(t *testing.T) {
	if !asm.Enabled() {
		t.Skip("no assembly on this platform")
	}
	x := []float32{1, 2, 3, 4}
	w := []float32{1, 0, 0, 0}
	if F32(x, w, 4) != 1 {
		t.Fatal("asm dot mismatch")
	}
}
