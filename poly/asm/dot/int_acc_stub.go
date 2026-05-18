//go:build !amd64 && !arm64

package dot

func i8AccF64(x, w []int8, n int) float64   { return I8TileAccF64Go(x, w, n) }
func i16AccF64(x, w []int16, n int) float64 { return I16TileAccF64Go(x, w, n) }
func i32AccF64(x, w []int32, n int) float64 { return I32TileAccF64Go(x, w, n) }
func i64AccF64(x, w []int64, n int) float64 { return I64TileAccF64Go(x, w, n) }
func u8AccF64(x, w []uint8, n int) float64   { return U8TileAccF64Go(x, w, n) }
func u16AccF64(x, w []uint16, n int) float64 { return U16TileAccF64Go(x, w, n) }
func u32AccF64(x, w []uint32, n int) float64 { return U32TileAccF64Go(x, w, n) }
func u64AccF64(x, w []uint64, n int) float64 { return U64TileAccF64Go(x, w, n) }
