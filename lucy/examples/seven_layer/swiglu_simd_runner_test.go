package sevenlayer

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestSwiGLUSimdParityFloat32_1x1(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerSwiGLU) {
		t.Skip("no Plan 9 SIMD")
	}
	g := GridSpec{Depth: 1, Rows: 1, Cols: 1}
	s := swigluSuiteForGrid(g)
	assertSCMCSimdParity(t, s, allDTypes[1])
}

func TestSwiGLUSimdParityAllGrids_Float32(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerSwiGLU) {
		t.Skip("no Plan 9 SIMD")
	}
	tc := allDTypes[1]
	for _, g := range StandardGrids {
		g := g
		t.Run(g.String(), func(t *testing.T) {
			s := swigluSuiteForGrid(g)
			assertSCMCSimdParity(t, s, tc)
		})
	}
}
