package sevenlayer

import (
	"fmt"
	"strings"
)

// GridSpec is the volumetric cell lattice (depth × rows × cols).
type GridSpec struct {
	Depth, Rows, Cols int
}

func (g GridSpec) Cells() int { return g.Depth * g.Rows * g.Cols }

func (g GridSpec) String() string {
	return fmt.Sprintf("%d×%d×%d", g.Depth, g.Rows, g.Cols)
}

// StackLayers is total layers in forward order (cells × layers_per_cell).
func (g GridSpec) StackLayers() int { return g.Cells() * sevenLayersPerCell }

// StandardGrids runs 1³, 2³, and 3³ without full-size tensors on larger lattices.
var StandardGrids = []GridSpec{
	{Depth: 1, Rows: 1, Cols: 1},
	{Depth: 2, Rows: 2, Cols: 2},
	{Depth: 3, Rows: 3, Cols: 3},
}

// CNN1Grids runs 1³, 2³, and 3³ (channel widths sized for k=3 SIMD crossover).
var CNN1Grids = StandardGrids

// CNN2Grids runs 1³, 2³, and 3³ (channel widths sized for k=3 SIMD crossover).
var CNN2Grids = StandardGrids

var ConvGrids = []GridSpec{
	{Depth: 1, Rows: 1, Cols: 1},
	{Depth: 2, Rows: 2, Cols: 2},
}

// CNN3Grids: 3D conv only on 1³ (56+ layers on 2³ is impractical for menu [7]).
var CNN3Grids = []GridSpec{
	{Depth: 1, Rows: 1, Cols: 1},
}

func trainEpochsForGrid(g GridSpec) int {
	switch g.Cells() {
	case 1:
		return 50
	case 8:
		return 12
	default:
		return 6
	}
}

func benchItersForGrid(g GridSpec) int {
	switch g.Cells() {
	case 1:
		return 25
	case 8:
		return 10
	default:
		return 5
	}
}

func gridCheckpointSuffix(g GridSpec) string {
	return fmt.Sprintf("_%dd%dr%dc", g.Depth, g.Rows, g.Cols)
}

func runGrids(grids []GridSpec, build func(GridSpec) LayerSuite) bool {
	ok := true
	for _, g := range grids {
		fmt.Printf("\n  ▷ Grid %s — %d cells × %d layers/cell = %d forward stack\n",
			g, g.Cells(), sevenLayersPerCell, g.StackLayers())
		if !RunLayerSuite(build(g)) {
			ok = false
		}
	}
	return ok
}

func runAllGrids(build func(GridSpec) LayerSuite) bool {
	return runGrids(StandardGrids, build)
}

func writeNetworkHeader(b *strings.Builder, id string, g GridSpec) {
	fmt.Fprintf(b,
		`{"id":"%s","depth":%d,"rows":%d,"cols":%d,"layers_per_cell":%d,"layers":[`,
		id, g.Depth, g.Rows, g.Cols, sevenLayersPerCell,
	)
}

func forEachGridCell(g GridSpec, fn func(z, y, x int)) {
	for z := 0; z < g.Depth; z++ {
		for y := 0; y < g.Rows; y++ {
			for x := 0; x < g.Cols; x++ {
				fn(z, y, x)
			}
		}
	}
}

func appendLayerJSON(b *strings.Builder, first *bool, layerJSON string) {
	if !*first {
		b.WriteByte(',')
	}
	*first = false
	b.WriteString(layerJSON)
}

// isStackOrigin is the first cell in z→y→x forward order (only valid place for Embedding input).
func isStackOrigin(z, y, x int) bool { return z == 0 && y == 0 && x == 0 }
