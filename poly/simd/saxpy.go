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

// SaxpyF32AccF64InStride computes acc[i] += alpha * float64(x[i*xStride]) for i in [0,n).
// SwiGLU down-projection dX: weight columns are strided in the packed weight blob.
func SaxpyF32AccF64InStride(acc []float64, alpha float64, x []float32, xStride, n int) {
	if n <= 0 || xStride <= 0 {
		return
	}
	if xStride == 1 && len(acc) >= n && len(x) >= n {
		SaxpyF32AccF64(acc, alpha, x, n)
		return
	}
	for i := 0; i < n; i++ {
		xi := i * xStride
		if xi >= len(x) || i >= len(acc) {
			return
		}
		acc[i] += alpha * float64(x[xi])
	}
}
