package sevenlayer

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

// TestEmbeddingMultiCellForward reproduces the seven-layer Embedding grid crash:
// origin cell (EMBEDDING + 6 DENSE) chained into non-origin DENSE cells must have
// matching cross-cell widths, otherwise the forward indexes past the activation.
func TestEmbeddingMultiCellForward(t *testing.T) {
	grids := []GridSpec{
		{Depth: 2, Rows: 2, Cols: 2},
		{Depth: 3, Rows: 3, Cols: 3},
	}
	for _, g := range grids {
		suite := embeddingSuite(g)
		for _, dt := range []string{"float32", "float64"} {
			net, err := poly.BuildNetworkFromJSON(suite.BuildJSON(dt))
			if err != nil {
				t.Fatalf("grid %s dtype %s: build: %v", g, dt, err)
			}
			input := suite.MakeInput()
			// sinTarget runs ForwardPolymorphic — the exact call that panicked.
			tgt := suite.MakeTarget(net, input)
			if tgt == nil || len(tgt.Data) == 0 {
				t.Fatalf("grid %s dtype %s: empty target", g, dt)
			}
		}
	}
}

func TestEmbeddingNativeMultiCellForward(t *testing.T) {
	grids := []GridSpec{
		{Depth: 2, Rows: 2, Cols: 2},
		{Depth: 3, Rows: 3, Cols: 3},
	}
	for _, g := range grids {
		suite := buildEmbeddingNativeSuite(g)
		for _, dt := range []string{"float32", "float64"} {
			net, err := poly.BuildNetworkFromJSON(suite.BuildJSON(dt))
			if err != nil {
				t.Fatalf("grid %s dtype %s: build: %v", g, dt, err)
			}
			input := suite.MakeInput()
			tgt := suite.MakeTarget(net, input)
			if tgt == nil || len(tgt.Data) == 0 {
				t.Fatalf("grid %s dtype %s: empty target", g, dt)
			}
		}
	}
}
