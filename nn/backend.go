package nn

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

// CPUBackend provides CPU-based tensor operations.
// This is the default implementation for compatibility.
type CPUBackend[T Numeric] struct{}

// NewCPUBackend creates a new CPU backend.
func NewCPUBackend[T Numeric]() *CPUBackend[T] {
	return &CPUBackend[T]{}
}

// Note: Full implementations of Backend methods are in backend_cpu.go
// This file defines the interface and basic structure.
