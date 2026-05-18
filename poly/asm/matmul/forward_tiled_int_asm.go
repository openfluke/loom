//go:build amd64 || arm64

package matmul

import (
	"github.com/openfluke/loom/poly/asm"
	"github.com/openfluke/loom/poly/asm/dot"
)

func init() {
	if !asm.Enabled() {
		return
	}
	dotTileI8 = dotTileI8Asm
	dotTileI16 = dotTileI16Asm
	dotTileI32 = dotTileI32Asm
	dotTileI64 = dotTileI64Asm
	dotTileU8 = dotTileU8Asm
	dotTileU16 = dotTileU16Asm
	dotTileU32 = dotTileU32Asm
	dotTileU64 = dotTileU64Asm
}

func dotTileI8Asm(inRow, wRow []int8, i0, i1 int, prev float64) float64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.I8TileAccF64(inRow[i0:i1], wRow[i0:i1], n)
}

func dotTileI16Asm(inRow, wRow []int16, i0, i1 int, prev float64) float64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.I16TileAccF64(inRow[i0:i1], wRow[i0:i1], n)
}

func dotTileI32Asm(inRow, wRow []int32, i0, i1 int, prev float64) float64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.I32TileAccF64(inRow[i0:i1], wRow[i0:i1], n)
}

func dotTileI64Asm(inRow, wRow []int64, i0, i1 int, prev float64) float64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.I64TileAccF64(inRow[i0:i1], wRow[i0:i1], n)
}

func dotTileU8Asm(inRow, wRow []uint8, i0, i1 int, prev float64) float64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.U8TileAccF64(inRow[i0:i1], wRow[i0:i1], n)
}

func dotTileU16Asm(inRow, wRow []uint16, i0, i1 int, prev float64) float64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.U16TileAccF64(inRow[i0:i1], wRow[i0:i1], n)
}

func dotTileU32Asm(inRow, wRow []uint32, i0, i1 int, prev float64) float64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.U32TileAccF64(inRow[i0:i1], wRow[i0:i1], n)
}

func dotTileU64Asm(inRow, wRow []uint64, i0, i1 int, prev float64) float64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.U64TileAccF64(inRow[i0:i1], wRow[i0:i1], n)
}
