package nn

import (
	"math"
)

// activateCPU applies the activation function on CPU - matches grid_demo exactly
func activateCPU(v float32, activation ActivationType) float32 {
	switch activation {
	case ActivationScaledReLU: // case 0
		v = v * 1.1
		if v < 0 {
			v = 0
		}
		return v
	case ActivationSigmoid: // case 1
		return 1.0 / (1.0 + float32(math.Exp(float64(-v))))
	case ActivationTanh: // case 2
		return float32(math.Tanh(float64(v)))
	case ActivationSoftplus: // case 3
		return float32(math.Log(1.0 + math.Exp(float64(v))))
	case ActivationLeakyReLU: // case 4
		if v < 0 {
			v = v * 0.1
		}
		return v
	default:
		return v
	}
}

// activateDerivativeCPU computes the derivative of the activation function
// Note: This computes the derivative with respect to the PRE-activation value
func activateDerivativeCPU(preActivation float32, activation ActivationType) float32 {
	switch activation {
	case ActivationScaledReLU: // case 0
		// d/dv (max(0, 1.1*v)) = 1.1 if v > 0, else 0
		if preActivation > 0 {
			return 1.1
		}
		return 0
	case ActivationSigmoid: // case 1
		// d/dv (1/(1+e^-v)) = sigmoid(v) * (1 - sigmoid(v))
		sig := 1.0 / (1.0 + float32(math.Exp(float64(-preActivation))))
		return sig * (1.0 - sig)
	case ActivationTanh: // case 2
		// d/dv tanh(v) = 1 - tanh^2(v)
		t := float32(math.Tanh(float64(preActivation)))
		return 1.0 - t*t
	case ActivationSoftplus: // case 3
		// d/dv log(1 + e^v) = e^v / (1 + e^v) = sigmoid(v)
		return 1.0 / (1.0 + float32(math.Exp(float64(-preActivation))))
	case ActivationLeakyReLU: // case 4
		// d/dv (v if v >= 0, else 0.1*v) = 1 if v >= 0, else 0.1
		if preActivation >= 0 {
			return 1.0
		}
		return 0.1
	default:
		return 1.0
	}
}
