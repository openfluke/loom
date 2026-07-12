package simd

// SaxpyF32AccF64 computes acc[i] += alpha * float64(x[i]) for i in [0,n).
// Dense backward dW/dX: scaled row accumulation into float64 gradient buffers.
func SaxpyF32AccF64(acc []float64, alpha float64, x []float32, n int) {
	if n <= 0 || len(acc) < n || len(x) < n {
		return
	}
	if simdEnabled() {
		saxpyF32AccF64Simd(&acc[0], alpha, &x[0], n)
		return
	}
	saxpyF32AccF64Go(acc, alpha, x, n)
}

func saxpyF32AccF64Go(acc []float64, alpha float64, x []float32, n int) {
	for i := 0; i < n; i++ {
		acc[i] += alpha * float64(x[i])
	}
}
