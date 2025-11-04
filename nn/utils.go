package nn

import (
	"math"
)

// MaxAbsDiff calculates the maximum absolute difference between two slices
func MaxAbsDiff(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	m := 0.0
	for i := 0; i < n; i++ {
		d := math.Abs(float64(a[i] - b[i]))
		if d > m {
			m = d
		}
	}
	return m
}

// Min returns the minimum value in a slice
func Min(v []float32) float32 {
	if len(v) == 0 {
		return 0
	}
	m := v[0]
	for _, x := range v {
		if x < m {
			m = x
		}
	}
	return m
}

// Max returns the maximum value in a slice
func Max(v []float32) float32 {
	if len(v) == 0 {
		return 0
	}
	m := v[0]
	for _, x := range v {
		if x > m {
			m = x
		}
	}
	return m
}

// Mean returns the mean value of a slice
func Mean(v []float32) float32 {
	if len(v) == 0 {
		return 0
	}
	sum := float32(0)
	for _, x := range v {
		sum += x
	}
	return sum / float32(len(v))
}
