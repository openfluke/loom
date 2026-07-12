package sevenlayer

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

// All Plan-9-SIMD layer types: fwd+bwd SC·MC·SIMD parity on Float32 (1×1×1).
// Dense/SwiGLU/MHA/CNN1/CNN2/CNN3 backward use saxpy+DotTile .s; other layers tiled until SIMD bwd lands.
func TestSimdParityAllLayers_Float32_1x1(t *testing.T) {
	if !poly.Plan9SimdEnabled() {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	tc := allDTypes[1] // Float32
	g := GridSpec{Depth: 1, Rows: 1, Cols: 1}

	cases := []struct {
		name string
		s    LayerSuite
	}{
		{"Dense", denseSuiteForGrid(g)},
		{"SwiGLU", swigluSuiteForGrid(g)},
		{"MHA", mhaSuiteForGrid(g)},
		{"CNN1", cnn1SuiteForGrid(g)},
		{"CNN2", cnn2SuiteForGrid(g)},
		{"RNN", rnnSuiteForGrid(g)},
		{"LSTM", lstmSuiteForGrid(g)},
	}

	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			if !poly.Plan9SimdForwardForLayer(c.s.PrimaryType) {
				t.Skip("layer type has no SIMD forward on this arch")
			}
			assertSCMCSimdParity(t, c.s, tc)
		})
	}
}
