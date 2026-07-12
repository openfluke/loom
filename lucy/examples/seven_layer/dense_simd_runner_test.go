package sevenlayer

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestDenseSimdParityFloat32_1x1(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerDense) {
		t.Skip("no Plan 9 SIMD")
	}
	g := GridSpec{Depth: 1, Rows: 1, Cols: 1}
	s := denseSuiteForGrid(g)
	assertSCMCSimdParity(t, s, allDTypes[1])
}

func TestDenseSimdParityAllDTypes_1x1(t *testing.T) {
	if testing.Short() {
		t.Skip("slow")
	}
	g := GridSpec{Depth: 1, Rows: 1, Cols: 1}
	s := denseSuiteForGrid(g)
	for _, tc := range allDTypes {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			assertSCMCSimdParity(t, s, tc)
		})
	}
}

func TestDenseSimdParityAllGrids_Float32(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerDense) {
		t.Skip("no Plan 9 SIMD")
	}
	tc := allDTypes[1]
	for _, g := range StandardGrids {
		g := g
		t.Run(g.String(), func(t *testing.T) {
			s := denseSuiteForGrid(g)
			assertSCMCSimdParity(t, s, tc)
		})
	}
}
