//go:build amd64 || arm64

package dot

import (
	"unsafe"

	"github.com/openfluke/loom/poly/asm"
)

//go:noescape
func dotI8NativeI64(x, w *int8, n int) int64

//go:noescape
func dotI16NativeI64(x, w *int16, n int) int64

//go:noescape
func dotI32NativeI64(x, w *int32, n int) int64

//go:noescape
func dotU8NativeI64(x, w *uint8, n int) int64

//go:noescape
func dotU16NativeI64(x, w *uint16, n int) int64

//go:noescape
func dotU32NativeI64(x, w *uint32, n int) int64

//go:noescape
func dotI64NativeI64(x, w *int64, n int) int64

//go:noescape
func dotU64NativeI64(x, w *uint64, n int) int64

func i8NativeI64(x []int8, w []int8, n int) int64 {
	if asm.Enabled() {
		return dotI8NativeI64(&x[0], &w[0], n)
	}
	return I8TileNativeI64Go(x, w, n)
}

func i16NativeI64(x []int16, w []int16, n int) int64 {
	if asm.Enabled() {
		return dotI16NativeI64(&x[0], &w[0], n)
	}
	return I16TileNativeI64Go(x, w, n)
}

func i32NativeI64(x []int32, w []int32, n int) int64 {
	if asm.Enabled() {
		return dotI32NativeI64(&x[0], &w[0], n)
	}
	return I32TileNativeI64Go(x, w, n)
}

func u8NativeI64(x []uint8, w []uint8, n int) int64 {
	if asm.Enabled() {
		return dotU8NativeI64(&x[0], &w[0], n)
	}
	return U8TileNativeI64Go(x, w, n)
}

func u16NativeI64(x []uint16, w []uint16, n int) int64 {
	if asm.Enabled() {
		return dotU16NativeI64(&x[0], &w[0], n)
	}
	return U16TileNativeI64Go(x, w, n)
}

func u32NativeI64(x []uint32, w []uint32, n int) int64 {
	if asm.Enabled() {
		return dotU32NativeI64(&x[0], &w[0], n)
	}
	return U32TileNativeI64Go(x, w, n)
}

func i64NativeI64(x []int64, w []int64, n int) int64 {
	if asm.Enabled() {
		return dotI64NativeI64(&x[0], &w[0], n)
	}
	return I64TileNativeI64Go(x, w, n)
}

func u64NativeI64(x []uint64, w []uint64, n int) int64 {
	if asm.Enabled() {
		return dotU64NativeI64(&x[0], &w[0], n)
	}
	return U64TileNativeI64Go(x, w, n)
}

func u8NativeI64Bytes(x, w []uint8, n int) int64 {
	// int8 quant values live in uint8 morph storage — same bits as []int8.
	return i8NativeI64(
		*(*[]int8)(unsafe.Pointer(&x)),
		*(*[]int8)(unsafe.Pointer(&w)),
		n,
	)
}
