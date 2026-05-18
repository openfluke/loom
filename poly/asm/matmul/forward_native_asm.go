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
	dotTileU8Native = dotTileU8NativeAsm
	dotTileU8SignedNative = dotTileU8SignedNativeAsm
	dotTileI8Native = dotTileI8NativeAsm
	dotTileI16Native = dotTileI16NativeAsm
	dotTileI32Native = dotTileI32NativeAsm
	dotTileU16Native = dotTileU16NativeAsm
	dotTileU32Native = dotTileU32NativeAsm
	dotTileI64Native = dotTileI64NativeAsm
	dotTileU64Native = dotTileU64NativeAsm
}

func dotTileU8NativeAsm(inRow, wRow []uint8, i0, i1 int, prev int64) int64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.U8TileNativeI64(inRow[i0:i1], wRow[i0:i1], n)
}

func dotTileU8SignedNativeAsm(inRow, wRow []uint8, i0, i1 int, prev int64) int64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.U8BytesTileNativeI64(inRow[i0:i1], wRow[i0:i1], n)
}

func dotTileI8NativeAsm(inRow, wRow []int8, i0, i1 int, prev int64) int64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.I8TileNativeI64(inRow[i0:i1], wRow[i0:i1], n)
}

func dotTileI16NativeAsm(inRow, wRow []int16, i0, i1 int, prev int64) int64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.I16TileNativeI64(inRow[i0:i1], wRow[i0:i1], n)
}

func dotTileI32NativeAsm(inRow, wRow []int32, i0, i1 int, prev int64) int64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.I32TileNativeI64(inRow[i0:i1], wRow[i0:i1], n)
}

func dotTileU16NativeAsm(inRow, wRow []uint16, i0, i1 int, prev int64) int64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.U16TileNativeI64(inRow[i0:i1], wRow[i0:i1], n)
}

func dotTileU32NativeAsm(inRow, wRow []uint32, i0, i1 int, prev int64) int64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.U32TileNativeI64(inRow[i0:i1], wRow[i0:i1], n)
}

func dotTileI64NativeAsm(inRow, wRow []int64, i0, i1 int, prev int64) int64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.I64TileNativeI64(inRow[i0:i1], wRow[i0:i1], n)
}

func dotTileU64NativeAsm(inRow, wRow []uint64, i0, i1 int, prev int64) int64 {
	n := i1 - i0
	if n <= 0 {
		return prev
	}
	return prev + dot.U64TileNativeI64(inRow[i0:i1], wRow[i0:i1], n)
}
