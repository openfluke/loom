package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestGetActiveWideIntAppliesScale(t *testing.T) {
	ws := &poly.WeightStore{
		Master:   []float32{0.5, -0.25, 0.75},
		Scale:    0.01,
		Versions: make(map[poly.DType]any),
	}
	ws.Morph(poly.DTypeInt32)
	active := ws.GetActive(poly.DTypeInt32)
	f32, ok := active.([]float32)
	if !ok {
		t.Fatalf("GetActive(Int32) = %T, want []float32", active)
	}
	if len(f32) != 3 {
		t.Fatalf("len = %d, want 3", len(f32))
	}
	// Morph rounds to int codes then GetActive must re-apply scale.
	if f32[0] != 0.5 || f32[1] != -0.25 || f32[2] != 0.75 {
		t.Fatalf("dequantized = %v, want [0.5 -0.25 0.75]", f32)
	}
}
