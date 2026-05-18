package dot

// F32TileAccF64 returns sum(x[i]*w[i]) for i in [0,n), accumulating in float64.
// Matches poly.denseForwardTiled* inner loops for float32 activations/weights.
func F32TileAccF64(x, w []float32, n int) float64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return f32AccF64(x, w, n)
}

// F32TileAccF64Go is the portable implementation.
func F32TileAccF64Go(x, w []float32, n int) float64 {
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(x[i]) * float64(w[i])
	}
	return sum
}
