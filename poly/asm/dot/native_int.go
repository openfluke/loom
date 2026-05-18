package dot

// Native integer tile dots: element-type multiply, int64 accumulate (no FP conversion).

func I8TileNativeI64(x, w []int8, n int) int64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return i8NativeI64(x, w, n)
}

func I8TileNativeI64Go(x, w []int8, n int) int64 {
	var sum int64
	for i := 0; i < n; i++ {
		sum += int64(x[i]) * int64(w[i])
	}
	return sum
}

func I16TileNativeI64(x, w []int16, n int) int64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return i16NativeI64(x, w, n)
}

func I16TileNativeI64Go(x, w []int16, n int) int64 {
	var sum int64
	for i := 0; i < n; i++ {
		sum += int64(x[i]) * int64(w[i])
	}
	return sum
}

func I32TileNativeI64(x, w []int32, n int) int64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return i32NativeI64(x, w, n)
}

func I32TileNativeI64Go(x, w []int32, n int) int64 {
	var sum int64
	for i := 0; i < n; i++ {
		sum += int64(x[i]) * int64(w[i])
	}
	return sum
}

func U8TileNativeI64(x, w []uint8, n int) int64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return u8NativeI64(x, w, n)
}

func U8TileNativeI64Go(x, w []uint8, n int) int64 {
	var sum int64
	for i := 0; i < n; i++ {
		sum += int64(x[i]) * int64(w[i])
	}
	return sum
}

func U16TileNativeI64(x, w []uint16, n int) int64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return u16NativeI64(x, w, n)
}

func U16TileNativeI64Go(x, w []uint16, n int) int64 {
	var sum int64
	for i := 0; i < n; i++ {
		sum += int64(x[i]) * int64(w[i])
	}
	return sum
}

func U32TileNativeI64(x, w []uint32, n int) int64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return u32NativeI64(x, w, n)
}

func U32TileNativeI64Go(x, w []uint32, n int) int64 {
	var sum int64
	for i := 0; i < n; i++ {
		sum += int64(x[i]) * int64(w[i])
	}
	return sum
}

func I64TileNativeI64(x, w []int64, n int) int64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return i64NativeI64(x, w, n)
}

func I64TileNativeI64Go(x, w []int64, n int) int64 {
	var sum int64
	for i := 0; i < n; i++ {
		sum += x[i] * w[i]
	}
	return sum
}

func U64TileNativeI64(x, w []uint64, n int) int64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return u64NativeI64(x, w, n)
}

func U64TileNativeI64Go(x, w []uint64, n int) int64 {
	var sum int64
	for i := 0; i < n; i++ {
		sum += int64(x[i]) * int64(w[i])
	}
	return sum
}

// U8BytesTileNativeI64 dots two []uint8 lanes (int8 bit patterns in byte storage).
func U8BytesTileNativeI64(x, w []uint8, n int) int64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return u8NativeI64Bytes(x, w, n)
}

func U8BytesTileNativeI64Go(x, w []uint8, n int) int64 {
	var sum int64
	for i := 0; i < n; i++ {
		sum += int64(int8(x[i])) * int64(int8(w[i]))
	}
	return sum
}
