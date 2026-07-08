package poly_test

import (
	"testing"

	. "github.com/openfluke/loom/poly"
	"github.com/openfluke/loom/poly/simd"
)

// TestBitNetPackedPathEngaged confirms that with SIMD on (arm64), a ternary Dense
// forward builds the packed-2-bit weights and does NOT build the 4x-larger byte
// Codes array — i.e. the packed kernel is the one running.
func TestBitNetPackedPathEngaged(t *testing.T) {
	if !simd.BitNetPackedAvailable() {
		t.Skip("packed kernel not available on this GOARCH")
	}
	const rows, cols = 128, 2560
	net := NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = LayerDense
	l.Activation = ActivationLinear
	l.DType = DTypeTernary
	l.InputHeight = cols
	l.OutputHeight = rows
	l.WeightStore = NewWeightStore(cols * rows)
	copy(l.WeightStore.Master, deterministicWeights(rows*cols))
	net.UseExactDType = true

	SetBitNetTernarySimdForward(true)
	defer SetBitNetTernarySimdForward(false)

	input := NewTensorFromSlice(deterministicWeights(cols), 1, cols)
	_, _ = DenseForwardPolymorphic(l, input)

	m, ok := l.WeightStore.GetBitNetTernaryMatrix(0, rows, cols)
	if !ok {
		t.Fatal("no packed ternary matrix")
	}
	if len(m.PackedStride) == 0 || m.PackedBlocks == 0 {
		t.Fatalf("packed weights not built: stride=%d blocks=%d", len(m.PackedStride), m.PackedBlocks)
	}
	if len(m.Codes) != 0 {
		t.Fatalf("byte Codes array was built (%d bytes) — packed path not taken", len(m.Codes))
	}
	// Packed uses ~2 bits/weight; the old Codes path used 1 byte/weight (4x more).
	t.Logf("packed bytes=%d (%.2f bits/weight) vs byte-code would be %d",
		len(m.PackedStride), float64(len(m.PackedStride))*8/float64(rows*cols), rows*((cols+31)/32)*32)
}
