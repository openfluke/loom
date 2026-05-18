//go:build amd64 || arm64

package matmul

import (
	"github.com/openfluke/loom/poly/asm"
	"github.com/openfluke/loom/poly/asm/dot"
)

func init() {
	if asm.Enabled() {
		dotTileF32 = dotTileF32Asm
		dotTileF64 = dotTileF64Asm
	}
}

func dotTileF32Asm(inRow, wRow []float32, i0, i1 int, prev float64) float64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.F32TileAccF64(inRow[i0:i1], wRow[i0:i1], n)
}

func dotTileF64Asm(inRow, wRow []float64, i0, i1 int, prev float64) float64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.F64(inRow[i0:i1], wRow[i0:i1], n)
}
