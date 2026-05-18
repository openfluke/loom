package dot

// F32 computes sum(x[i]*w[i]) for i in [0,n).
func F32(x, w []float32, n int) float32 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return f32(x, w, n)
}

// F32Go is the portable float32 dot product.
func F32Go(x, w []float32, n int) float32 {
	var sum float32
	for i := 0; i < n; i++ {
		sum += x[i] * w[i]
	}
	return sum
}
