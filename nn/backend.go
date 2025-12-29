package nn

import (
	"math"
)

// Backend defines the interface for tensor operations.
// This abstraction allows swapping implementations (CPU, GPU, quantized)
// without changing layer code.
type Backend[T Numeric] interface {
	// MatMul performs matrix multiplication: result = a @ b
	// a shape: [M, K], b shape: [K, N] -> result shape: [M, N]
	MatMul(a, b *Tensor[T]) *Tensor[T]

	// MatMulAdd performs matrix multiplication with bias: result = a @ b + c
	MatMulAdd(a, b, c *Tensor[T]) *Tensor[T]

	// Add performs element-wise addition: result = a + b
	Add(a, b *Tensor[T]) *Tensor[T]

	// Scale multiplies all elements by a scalar: result = t * factor
	Scale(t *Tensor[T], factor T) *Tensor[T]

	// Activate applies activation function element-wise
	Activate(t *Tensor[T], actType ActivationType) *Tensor[T]

	// ActivateDerivative computes activation derivative for backprop
	ActivateDerivative(preAct *Tensor[T], actType ActivationType) *Tensor[T]

	// OuterProduct computes outer product: result[i,j] = a[i] * b[j]
	// Essential for NeuralTween weight updates
	OuterProduct(a, b *Tensor[T]) *Tensor[T]

	// Sum computes the sum of all elements
	Sum(t *Tensor[T]) T

	// Mean computes the mean of all elements
	Mean(t *Tensor[T]) T

	// Sqrt computes element-wise square root
	Sqrt(t *Tensor[T]) *Tensor[T]
}

// =============================================================================
// CPUBackend Implementation
// =============================================================================

// CPUBackend provides CPU-based tensor operations.
type CPUBackend[T Numeric] struct{}

// NewCPUBackend creates a new CPU backend.
func NewCPUBackend[T Numeric]() *CPUBackend[T] {
	return &CPUBackend[T]{}
}

// MatMul performs matrix multiplication: result = a @ b
func (b *CPUBackend[T]) MatMul(a, mat *Tensor[T]) *Tensor[T] {
	// Assume a is [M, K] and mat is [K, N]
	if len(a.Shape) != 2 || len(mat.Shape) != 2 {
		// Fallback: treat as vectors or flat matrices
		return b.matMulFlat(a, mat)
	}

	M, K := a.Shape[0], a.Shape[1]
	K2, N := mat.Shape[0], mat.Shape[1]

	if K != K2 {
		// Shape mismatch, return empty tensor
		return NewTensor[T](0)
	}

	result := NewTensor[T](M * N)
	result.Shape = []int{M, N}

	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := float64(0)
			for k := 0; k < K; k++ {
				sum += float64(a.Data[i*K+k]) * float64(mat.Data[k*N+j])
			}
			result.Data[i*N+j] = T(sum)
		}
	}

	return result
}

// matMulFlat handles flat tensor multiplication (vector, batched, etc.)
func (b *CPUBackend[T]) matMulFlat(a, mat *Tensor[T]) *Tensor[T] {
	// Simple dot product
	minLen := len(a.Data)
	if len(mat.Data) < minLen {
		minLen = len(mat.Data)
	}

	result := NewTensor[T](1)
	sum := float64(0)
	for i := 0; i < minLen; i++ {
		sum += float64(a.Data[i]) * float64(mat.Data[i])
	}
	result.Data[0] = T(sum)
	return result
}

// MatMulAdd performs matrix multiplication with bias: result = a @ b + c
func (b *CPUBackend[T]) MatMulAdd(a, mat, c *Tensor[T]) *Tensor[T] {
	result := b.MatMul(a, mat)
	return b.Add(result, c)
}

// Add performs element-wise addition: result = a + b
func (b *CPUBackend[T]) Add(a, other *Tensor[T]) *Tensor[T] {
	maxLen := len(a.Data)
	if len(other.Data) > maxLen {
		maxLen = len(other.Data)
	}

	result := NewTensor[T](maxLen)
	for i := 0; i < maxLen; i++ {
		var av, ov T
		if i < len(a.Data) {
			av = a.Data[i]
		}
		if i < len(other.Data) {
			ov = other.Data[i]
		}
		result.Data[i] = T(float64(av) + float64(ov))
	}
	return result
}

// Scale multiplies all elements by a scalar: result = t * factor
func (b *CPUBackend[T]) Scale(t *Tensor[T], factor T) *Tensor[T] {
	result := NewTensor[T](len(t.Data))
	for i, v := range t.Data {
		result.Data[i] = T(float64(v) * float64(factor))
	}
	return result
}

// Activate applies activation function element-wise
func (b *CPUBackend[T]) Activate(t *Tensor[T], actType ActivationType) *Tensor[T] {
	result := NewTensor[T](len(t.Data))
	for i, v := range t.Data {
		result.Data[i] = Activate(v, actType)
	}
	return result
}

// ActivateDerivative computes activation derivative for backprop
func (b *CPUBackend[T]) ActivateDerivative(preAct *Tensor[T], actType ActivationType) *Tensor[T] {
	result := NewTensor[T](len(preAct.Data))
	for i, v := range preAct.Data {
		result.Data[i] = ActivateDerivative(v, actType)
	}
	return result
}

// OuterProduct computes outer product: result[i,j] = a[i] * b[j]
func (b *CPUBackend[T]) OuterProduct(a, other *Tensor[T]) *Tensor[T] {
	M := len(a.Data)
	N := len(other.Data)
	result := NewTensor[T](M * N)
	result.Shape = []int{M, N}

	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			result.Data[i*N+j] = T(float64(a.Data[i]) * float64(other.Data[j]))
		}
	}
	return result
}

// Sum computes the sum of all elements
func (b *CPUBackend[T]) Sum(t *Tensor[T]) T {
	sum := float64(0)
	for _, v := range t.Data {
		sum += float64(v)
	}
	return T(sum)
}

// Mean computes the mean of all elements
func (b *CPUBackend[T]) Mean(t *Tensor[T]) T {
	if len(t.Data) == 0 {
		return T(0)
	}
	sum := b.Sum(t)
	return T(float64(sum) / float64(len(t.Data)))
}

// Sqrt computes element-wise square root
func (b *CPUBackend[T]) Sqrt(t *Tensor[T]) *Tensor[T] {
	result := NewTensor[T](len(t.Data))
	for i, v := range t.Data {
		result.Data[i] = T(math.Sqrt(float64(v)))
	}
	return result
}

// =============================================================================
// Helper Functions for Type Conversion
// =============================================================================

// ConvertTensorFloat32ToT converts a float32 tensor to any Numeric type.
func ConvertTensorFloat32ToT[T Numeric](src *Tensor[float32]) *Tensor[T] {
	result := NewTensor[T](len(src.Data))
	result.Shape = src.Shape
	result.Strides = src.Strides
	for i, v := range src.Data {
		result.Data[i] = T(v)
	}
	return result
}

// ConvertTensorTToFloat32 converts any Numeric tensor to float32.
func ConvertTensorTToFloat32[T Numeric](src *Tensor[T]) *Tensor[float32] {
	result := NewTensor[float32](len(src.Data))
	result.Shape = src.Shape
	result.Strides = src.Strides
	for i, v := range src.Data {
		result.Data[i] = float32(v)
	}
	return result
}

// ConvertSliceFloat32ToT converts a float32 slice to any Numeric type.
func ConvertSliceFloat32ToT[T Numeric](src []float32) []T {
	result := make([]T, len(src))
	for i, v := range src {
		result[i] = T(v)
	}
	return result
}

// ConvertSliceTToFloat32 converts any Numeric slice to float32.
func ConvertSliceTToFloat32[T Numeric](src []T) []float32 {
	result := make([]float32, len(src))
	for i, v := range src {
		result[i] = float32(v)
	}
	return result
}

