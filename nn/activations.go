package nn

import (
	"math"
)

// Activate applies the activation function for any numeric type.
// This is the generic implementation that supports all Numeric types.
func Activate[T Numeric](v T, activation ActivationType) T {
	// Convert to float64 for math operations, then back to T
	vf := float64(v)

	// For integer types, we assume input is scaled by 100*100=10000 (from MatMul)
	// We need to normalize it back to ~1.0 range for activation functions
	if IsIntegerType[T]() {
		vf /= 10000.0
	}

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
			result = 0.1 * vf
		} else {
			result = vf
		}
	case ActivationGELU: // case 5
		// Gaussian Error Linear Unit (GELU) approximation:
		// 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
		const (
			sqrt2Pi = 0.7978845608
			coeff   = 0.044715
		)
		x3 := vf * vf * vf
		inner := sqrt2Pi * (vf + coeff*x3)
		result = 0.5 * vf * (1.0 + math.Tanh(inner))
	default: // Linear / Unknown
		result = vf
	}

	// For integer types, rescale output to x100 range
	if IsIntegerType[T]() {
		// Target range x100
		scale := 100.0
		result *= scale
	}

	return T(result)
}

// ActivateDerivative computes the derivative of the activation function for any numeric type.
// Note: This computes the derivative with respect to the PRE-activation value.
func ActivateDerivative[T Numeric](preActivation T, activation ActivationType) T {
	vf := float64(preActivation)

	// For integer types, normalize pre-activation input (x10000 -> x1)
	if IsIntegerType[T]() {
		vf /= 10000.0
	}

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
	case ActivationGELU: // case 5
		// d/dx GELU(x) ~ 0.5 * (1 + tanh(s * (x + c*x^3))) +
		//               0.5 * x * sech^2(s * (x + c*x^3)) * s * (1 + 3*c*x^2)
		const (
			sqrt2Pi = 0.7978845608
			coeff   = 0.044715
		)
		x2 := vf * vf
		x3 := x2 * vf
		inner := sqrt2Pi * (vf + coeff*x3)
		tanhInner := math.Tanh(inner)

		sech2 := 1.0 - tanhInner*tanhInner
		result = 0.5*(1.0+tanhInner) + 0.5*vf*sech2*sqrt2Pi*(1.0+3.0*coeff*x2)
	default:
		result = 1.0
	}

	// For integer types, scale derivative by 100 to propagate gradients
	if IsIntegerType[T]() {
		result *= 100.0
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
