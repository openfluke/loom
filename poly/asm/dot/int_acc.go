package dot

// Integer tile dots accumulate in float64 (matches poly dense tiled forward).

func I8TileAccF64(x, w []int8, n int) float64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return i8AccF64(x, w, n)
}

func I8TileAccF64Go(x, w []int8, n int) float64 {
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(x[i]) * float64(w[i])
	}
	return sum
}

func I16TileAccF64(x, w []int16, n int) float64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return i16AccF64(x, w, n)
}

func I16TileAccF64Go(x, w []int16, n int) float64 {
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(x[i]) * float64(w[i])
	}
	return sum
}

func I32TileAccF64(x, w []int32, n int) float64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return i32AccF64(x, w, n)
}

func I32TileAccF64Go(x, w []int32, n int) float64 {
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(x[i]) * float64(w[i])
	}
	return sum
}

func I64TileAccF64(x, w []int64, n int) float64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return i64AccF64(x, w, n)
}

func I64TileAccF64Go(x, w []int64, n int) float64 {
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(x[i]) * float64(w[i])
	}
	return sum
}

func U8TileAccF64(x, w []uint8, n int) float64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return u8AccF64(x, w, n)
}

func U8TileAccF64Go(x, w []uint8, n int) float64 {
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(x[i]) * float64(w[i])
	}
	return sum
}

func U16TileAccF64(x, w []uint16, n int) float64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return u16AccF64(x, w, n)
}

func U16TileAccF64Go(x, w []uint16, n int) float64 {
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(x[i]) * float64(w[i])
	}
	return sum
}

func U32TileAccF64(x, w []uint32, n int) float64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return u32AccF64(x, w, n)
}

func U32TileAccF64Go(x, w []uint32, n int) float64 {
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(x[i]) * float64(w[i])
	}
	return sum
}

func U64TileAccF64(x, w []uint64, n int) float64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return u64AccF64(x, w, n)
}

func U64TileAccF64Go(x, w []uint64, n int) float64 {
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(x[i]) * float64(w[i])
	}
	return sum
}
