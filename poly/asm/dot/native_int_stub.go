//go:build !amd64 && !arm64

package dot

func i8NativeI64(x []int8, w []int8, n int) int64     { return I8TileNativeI64Go(x, w, n) }
func i16NativeI64(x []int16, w []int16, n int) int64   { return I16TileNativeI64Go(x, w, n) }
func i32NativeI64(x []int32, w []int32, n int) int64   { return I32TileNativeI64Go(x, w, n) }
func u8NativeI64(x []uint8, w []uint8, n int) int64     { return U8TileNativeI64Go(x, w, n) }
func u16NativeI64(x []uint16, w []uint16, n int) int64   { return U16TileNativeI64Go(x, w, n) }
func u32NativeI64(x []uint32, w []uint32, n int) int64   { return U32TileNativeI64Go(x, w, n) }
func u8NativeI64Bytes(x, w []uint8, n int) int64        { return U8BytesTileNativeI64Go(x, w, n) }
func i64NativeI64(x []int64, w []int64, n int) int64   { return I64TileNativeI64Go(x, w, n) }
func u64NativeI64(x []uint64, w []uint64, n int) int64 { return U64TileNativeI64Go(x, w, n) }
