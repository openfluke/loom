//go:build amd64 || arm64

package dot

import "github.com/openfluke/loom/poly/asm"

//go:noescape
func dotI8AccF64(x, w *int8, n int) float64

//go:noescape
func dotI16AccF64(x, w *int16, n int) float64

//go:noescape
func dotI32AccF64(x, w *int32, n int) float64

//go:noescape
func dotI64AccF64(x, w *int64, n int) float64

//go:noescape
func dotU8AccF64(x, w *uint8, n int) float64

//go:noescape
func dotU16AccF64(x, w *uint16, n int) float64

//go:noescape
func dotU32AccF64(x, w *uint32, n int) float64

//go:noescape
func dotU64AccF64(x, w *uint64, n int) float64

func i8AccF64(x, w []int8, n int) float64 {
	if asm.Enabled() {
		return dotI8AccF64(&x[0], &w[0], n)
	}
	return I8TileAccF64Go(x, w, n)
}

func i16AccF64(x, w []int16, n int) float64 {
	if asm.Enabled() {
		return dotI16AccF64(&x[0], &w[0], n)
	}
	return I16TileAccF64Go(x, w, n)
}

func i32AccF64(x, w []int32, n int) float64 {
	if asm.Enabled() {
		return dotI32AccF64(&x[0], &w[0], n)
	}
	return I32TileAccF64Go(x, w, n)
}

func i64AccF64(x, w []int64, n int) float64 {
	if asm.Enabled() {
		return dotI64AccF64(&x[0], &w[0], n)
	}
	return I64TileAccF64Go(x, w, n)
}

func u8AccF64(x, w []uint8, n int) float64 {
	if asm.Enabled() {
		return dotU8AccF64(&x[0], &w[0], n)
	}
	return U8TileAccF64Go(x, w, n)
}

func u16AccF64(x, w []uint16, n int) float64 {
	if asm.Enabled() {
		return dotU16AccF64(&x[0], &w[0], n)
	}
	return U16TileAccF64Go(x, w, n)
}

func u32AccF64(x, w []uint32, n int) float64 {
	if asm.Enabled() {
		return dotU32AccF64(&x[0], &w[0], n)
	}
	return U32TileAccF64Go(x, w, n)
}

func u64AccF64(x, w []uint64, n int) float64 {
	if asm.Enabled() {
		return dotU64AccF64(&x[0], &w[0], n)
	}
	return U64TileAccF64Go(x, w, n)
}
