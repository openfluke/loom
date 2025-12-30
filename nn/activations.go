package nn

import (
	"math"
)

// Activate applies the activation function for any numeric type.
// This is the generic implementation that supports all Numeric types.
func Activate[T Numeric](v T, activation ActivationType) T {
	// Convert to float64 for math operations, then back to T
	vf := float64(v)
	var result float64

	switch activation {
	case ActivationScaledReLU: // case 0
		vf = vf * 1.1
		if vf < 0 {
			vf = 0
		}
		result = vf
	case ActivationSigmoid: // case 1
		result = 1.0 / (1.0 + math.Exp(-vf))
	case ActivationTanh: // case 2
		result = math.Tanh(vf)
	case ActivationSoftplus: // case 3
		result = math.Log(1.0 + math.Exp(vf))
	case ActivationLeakyReLU: // case 4
		if vf < 0 {
			vf = vf * 0.1
		}
		result = vf
	default:
		result = vf
	}

	// For integer types, scale fractional activations to preserve dynamic range
	// Sigmoid/Tanh output [0,1], which truncates to 0 for integers.
	// We scale by 100 to match the test/usage convention for fixed point.
	if IsIntegerType[T]() {
		switch activation {
		case ActivationSigmoid, ActivationTanh, ActivationSoftplus:
			result *= 100.0
		}
	}

	return T(result)
}

// ActivateDerivative computes the derivative of the activation function for any numeric type.
// Note: This computes the derivative with respect to the PRE-activation value.
func ActivateDerivative[T Numeric](preActivation T, activation ActivationType) T {
	vf := float64(preActivation)
	var result float64

	switch activation {
	case ActivationScaledReLU: // case 0
		// d/dv (max(0, 1.1*v)) = 1.1 if v > 0, else 0
		if vf > 0 {
			result = 1.1
		} else {
			result = 0
		}
	case ActivationSigmoid: // case 1
		// d/dv (1/(1+e^-v)) = sigmoid(v) * (1 - sigmoid(v))
		sig := 1.0 / (1.0 + math.Exp(-vf))
		result = sig * (1.0 - sig)
	case ActivationTanh: // case 2
		// d/dv tanh(v) = 1 - tanh^2(v)
		t := math.Tanh(vf)
		result = 1.0 - t*t
	case ActivationSoftplus: // case 3
		// d/dv log(1 + e^v) = e^v / (1 + e^v) = sigmoid(v)
		result = 1.0 / (1.0 + math.Exp(-vf))
	case ActivationLeakyReLU: // case 4
		// d/dv (v if v >= 0, else 0.1*v) = 1 if v >= 0, else 0.1
		if vf >= 0 {
			result = 1.0
		} else {
			result = 0.1
		}
	default:
		result = 1.0
	}

	// For integer types, derivatives of fractional functions are < 1 and truncate to 0.
	// We scale them to allow gradient flow (Pseudo-quantization aware training).
	if IsIntegerType[T]() {
		// Constant scaling factor of 100 matching the activation scaling
		scale := 100.0
		switch activation {
		case ActivationSigmoid, ActivationTanh, ActivationSoftplus:
			result *= scale
		case ActivationLeakyReLU:
			// For LeakyReLU, the derivative is 0.1 for negative values.
			// T(0.1) is 0. We scale it to 10 (0.1 * 100).
			// Positive part is 1.0 -> 100.
			result *= scale
		case ActivationScaledReLU:
			// ScaledReLU has slope 1.1 -> 110
			result *= scale
		}
	}

	return T(result)
}

// ActivateTensor applies activation function to all elements of a tensor.
func ActivateTensor[T Numeric](t *Tensor[T], activation ActivationType) *Tensor[T] {
	result := t.Clone()
	for i := range result.Data {
		result.Data[i] = Activate(result.Data[i], activation)
	}
	return result
}

// ActivateDerivativeTensor computes activation derivatives for all elements.
func ActivateDerivativeTensor[T Numeric](preAct *Tensor[T], activation ActivationType) *Tensor[T] {
	result := NewTensor[T](preAct.Shape...)
	for i := range preAct.Data {
		result.Data[i] = ActivateDerivative(preAct.Data[i], activation)
	}
	return result
}

// =============================================================================
// Backward-compatible float32 functions (used by existing layer implementations)
// =============================================================================

// activateCPU applies the activation function on CPU - matches grid_demo exactly
// Deprecated: Use Activate[float32] for new code.
func activateCPU(v float32, activation ActivationType) float32 {
	return Activate(v, activation)
}

// activateDerivativeCPU computes the derivative of the activation function
// Note: This computes the derivative with respect to the PRE-activation value
// Deprecated: Use ActivateDerivative[float32] for new code.
func activateDerivativeCPU(preActivation float32, activation ActivationType) float32 {
	return ActivateDerivative(preActivation, activation)
}

