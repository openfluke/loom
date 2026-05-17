package dot

// F64 computes sum(x[i]*w[i]) for i in [0,n).
func F64(x, w []float64, n int) float64 {
	if n <= 0 || len(x) < n || len(w) < n {
		return 0
	}
	return f64(x, w, n)
}

// F64Go is the portable float64 dot product.
func F64Go(x, w []float64, n int) float64 {
	var sum float64
	for i := 0; i < n; i++ {
		sum += x[i] * w[i]
	}
	return sum
}
