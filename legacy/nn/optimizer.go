package nn

import (
	"fmt"
	"math"
)

// Optimizer interface defines the contract for all optimizers
type Optimizer interface {
	// Step applies gradients to network weights
	Step(network *Network, learningRate float32)

	// Reset clears optimizer state (momentum, etc.)
	Reset()

	// GetState returns optimizer state for serialization
	GetState() map[string]interface{}

	// LoadState restores optimizer state from serialization
	LoadState(state map[string]interface{}) error

	// Name returns the optimizer name
	Name() string
}

// ============================================================================
// SGD Optimizer (Stochastic Gradient Descent with optional momentum)
// ============================================================================

type SGDOptimizer struct {
	momentum   float32
	velocities map[string][]float32 // Momentum buffers
	dampening  float32
	nesterov   bool
}

func NewSGDOptimizer() *SGDOptimizer {
	return &SGDOptimizer{
		momentum:   0.0,
		velocities: make(map[string][]float32),
		dampening:  0.0,
		nesterov:   false,
	}
}

func NewSGDOptimizerWithMomentum(momentum, dampening float32, nesterov bool) *SGDOptimizer {
	return &SGDOptimizer{
		momentum:   momentum,
		velocities: make(map[string][]float32),
		dampening:  dampening,
		nesterov:   nesterov,
	}
}

func (opt *SGDOptimizer) Step(network *Network, learningRate float32) {
	// Simple SGD without momentum
	if opt.momentum == 0.0 {
		opt.stepSimple(network, learningRate)
		return
	}

	// SGD with momentum
	opt.stepWithMomentum(network, learningRate)
}

func (opt *SGDOptimizer) stepSimple(network *Network, learningRate float32) {
	// Update weights: w = w - lr * grad
	for i := 0; i < network.TotalLayers(); i++ {
		layer := &network.Layers[i]

		// Update Kernel weights
		if len(layer.Kernel) > 0 && len(network.kernelGradients[i]) == len(layer.Kernel) {
			for j := range layer.Kernel {
				layer.Kernel[j] -= learningRate * network.kernelGradients[i][j]
			}
		}

		// Update Bias
		if len(layer.Bias) > 0 && len(network.biasGradients[i]) == len(layer.Bias) {
			for j := range layer.Bias {
				layer.Bias[j] -= learningRate * network.biasGradients[i][j]
			}
		}
	}
}

func (opt *SGDOptimizer) stepWithMomentum(network *Network, learningRate float32) {
	for i := 0; i < network.TotalLayers(); i++ {
		layer := &network.Layers[i]

		// Update Kernel weights with momentum
		if len(layer.Kernel) > 0 && len(network.kernelGradients[i]) == len(layer.Kernel) {
			key := fmt.Sprintf("kernel_%d", i)

			// Initialize velocity if needed
			if opt.velocities[key] == nil {
				opt.velocities[key] = make([]float32, len(layer.Kernel))
			}

			// Update with momentum: v = momentum * v + (1 - dampening) * grad
			//                       w = w - lr * v (or w - lr * (grad + momentum * v) for Nesterov)
			for j := range layer.Kernel {
				grad := network.kernelGradients[i][j]
				opt.velocities[key][j] = opt.momentum*opt.velocities[key][j] + (1-opt.dampening)*grad

				if opt.nesterov {
					layer.Kernel[j] -= learningRate * (grad + opt.momentum*opt.velocities[key][j])
				} else {
					layer.Kernel[j] -= learningRate * opt.velocities[key][j]
				}
			}
		}

		// Update Bias with momentum
		if len(layer.Bias) > 0 && len(network.biasGradients[i]) == len(layer.Bias) {
			key := fmt.Sprintf("bias_%d", i)

			if opt.velocities[key] == nil {
				opt.velocities[key] = make([]float32, len(layer.Bias))
			}

			for j := range layer.Bias {
				grad := network.biasGradients[i][j]
				opt.velocities[key][j] = opt.momentum*opt.velocities[key][j] + (1-opt.dampening)*grad

				if opt.nesterov {
					layer.Bias[j] -= learningRate * (grad + opt.momentum*opt.velocities[key][j])
				} else {
					layer.Bias[j] -= learningRate * opt.velocities[key][j]
				}
			}
		}
	}
}

func (opt *SGDOptimizer) Reset() {
	opt.velocities = make(map[string][]float32)
}

func (opt *SGDOptimizer) GetState() map[string]interface{} {
	return map[string]interface{}{
		"type":      "sgd",
		"momentum":  opt.momentum,
		"dampening": opt.dampening,
		"nesterov":  opt.nesterov,
	}
}

func (opt *SGDOptimizer) LoadState(state map[string]interface{}) error {
	if t, ok := state["type"].(string); !ok || t != "sgd" {
		return fmt.Errorf("invalid optimizer type: expected sgd, got %v", state["type"])
	}

	if m, ok := state["momentum"].(float64); ok {
		opt.momentum = float32(m)
	}
	if d, ok := state["dampening"].(float64); ok {
		opt.dampening = float32(d)
	}
	if n, ok := state["nesterov"].(bool); ok {
		opt.nesterov = n
	}

	return nil
}

func (opt *SGDOptimizer) Name() string {
	if opt.momentum > 0 {
		if opt.nesterov {
			return "SGD (Nesterov momentum)"
		}
		return "SGD (momentum)"
	}
	return "SGD"
}

// ============================================================================
// AdamW Optimizer (Adam with decoupled weight decay)
// ============================================================================

type AdamWOptimizer struct {
	beta1       float32
	beta2       float32
	epsilon     float32
	weightDecay float32
	step        int

	// First moment estimates (momentum)
	m map[string][]float32

	// Second moment estimates (variance)
	v map[string][]float32
}

func NewAdamWOptimizer(beta1, beta2, epsilon, weightDecay float32) *AdamWOptimizer {
	return &AdamWOptimizer{
		beta1:       beta1,
		beta2:       beta2,
		epsilon:     epsilon,
		weightDecay: weightDecay,
		step:        0,
		m:           make(map[string][]float32),
		v:           make(map[string][]float32),
	}
}

func NewAdamWOptimizerDefault() *AdamWOptimizer {
	return NewAdamWOptimizer(0.9, 0.999, 1e-8, 0.01)
}

func (opt *AdamWOptimizer) Step(network *Network, learningRate float32) {
	opt.step++

	// Bias correction factors
	biasCorrection1 := 1.0 - float32(math.Pow(float64(opt.beta1), float64(opt.step)))
	biasCorrection2 := 1.0 - float32(math.Pow(float64(opt.beta2), float64(opt.step)))

	for i := 0; i < network.TotalLayers(); i++ {
		layer := &network.Layers[i]

		// Update Kernel weights with AdamW
		if len(layer.Kernel) > 0 && len(network.kernelGradients[i]) == len(layer.Kernel) {
			key := fmt.Sprintf("kernel_%d", i)

			// Initialize moments if needed
			if opt.m[key] == nil {
				opt.m[key] = make([]float32, len(layer.Kernel))
				opt.v[key] = make([]float32, len(layer.Kernel))
			}

			// AdamW update
			for j := range layer.Kernel {
				grad := network.kernelGradients[i][j]

				// Update biased first moment estimate
				opt.m[key][j] = opt.beta1*opt.m[key][j] + (1-opt.beta1)*grad

				// Update biased second moment estimate
				opt.v[key][j] = opt.beta2*opt.v[key][j] + (1-opt.beta2)*grad*grad

				// Compute bias-corrected moments
				mHat := opt.m[key][j] / biasCorrection1
				vHat := opt.v[key][j] / biasCorrection2

				// Update weights with AdamW (decoupled weight decay)
				layer.Kernel[j] -= learningRate * (mHat/(float32(math.Sqrt(float64(vHat)))+opt.epsilon) + opt.weightDecay*layer.Kernel[j])
			}
		}

		// Update Bias with AdamW
		if len(layer.Bias) > 0 && len(network.biasGradients[i]) == len(layer.Bias) {
			key := fmt.Sprintf("bias_%d", i)

			if opt.m[key] == nil {
				opt.m[key] = make([]float32, len(layer.Bias))
				opt.v[key] = make([]float32, len(layer.Bias))
			}

			for j := range layer.Bias {
				grad := network.biasGradients[i][j]

				opt.m[key][j] = opt.beta1*opt.m[key][j] + (1-opt.beta1)*grad
				opt.v[key][j] = opt.beta2*opt.v[key][j] + (1-opt.beta2)*grad*grad

				mHat := opt.m[key][j] / biasCorrection1
				vHat := opt.v[key][j] / biasCorrection2

				layer.Bias[j] -= learningRate * (mHat / (float32(math.Sqrt(float64(vHat))) + opt.epsilon))
			}
		}
	}
}

func (opt *AdamWOptimizer) Reset() {
	opt.step = 0
	opt.m = make(map[string][]float32)
	opt.v = make(map[string][]float32)
}

func (opt *AdamWOptimizer) GetState() map[string]interface{} {
	return map[string]interface{}{
		"type":         "adamw",
		"beta1":        opt.beta1,
		"beta2":        opt.beta2,
		"epsilon":      opt.epsilon,
		"weight_decay": opt.weightDecay,
		"step":         opt.step,
	}
}

func (opt *AdamWOptimizer) LoadState(state map[string]interface{}) error {
	if t, ok := state["type"].(string); !ok || t != "adamw" {
		return fmt.Errorf("invalid optimizer type: expected adamw, got %v", state["type"])
	}

	if b1, ok := state["beta1"].(float64); ok {
		opt.beta1 = float32(b1)
	}
	if b2, ok := state["beta2"].(float64); ok {
		opt.beta2 = float32(b2)
	}
	if eps, ok := state["epsilon"].(float64); ok {
		opt.epsilon = float32(eps)
	}
	if wd, ok := state["weight_decay"].(float64); ok {
		opt.weightDecay = float32(wd)
	}
	if s, ok := state["step"].(float64); ok {
		opt.step = int(s)
	}

	return nil
}

func (opt *AdamWOptimizer) Name() string {
	return "AdamW"
}

// ============================================================================
// RMSprop Optimizer
// ============================================================================

type RMSpropOptimizer struct {
	alpha    float32 // Decay rate
	epsilon  float32
	momentum float32

	// Running average of squared gradients
	v map[string][]float32

	// Momentum buffer (if momentum > 0)
	buf map[string][]float32
}

func NewRMSpropOptimizer(alpha, epsilon, momentum float32) *RMSpropOptimizer {
	return &RMSpropOptimizer{
		alpha:    alpha,
		epsilon:  epsilon,
		momentum: momentum,
		v:        make(map[string][]float32),
		buf:      make(map[string][]float32),
	}
}

func NewRMSpropOptimizerDefault() *RMSpropOptimizer {
	return NewRMSpropOptimizer(0.99, 1e-8, 0.0)
}

func (opt *RMSpropOptimizer) Step(network *Network, learningRate float32) {
	for i := 0; i < network.TotalLayers(); i++ {
		layer := &network.Layers[i]

		// Update Kernel weights with RMSprop
		if len(layer.Kernel) > 0 && len(network.kernelGradients[i]) == len(layer.Kernel) {
			key := fmt.Sprintf("kernel_%d", i)

			// Initialize running average if needed
			if opt.v[key] == nil {
				opt.v[key] = make([]float32, len(layer.Kernel))

				if opt.momentum > 0 {
					opt.buf[key] = make([]float32, len(layer.Kernel))
				}
			}

			// RMSprop update
			for j := range layer.Kernel {
				grad := network.kernelGradients[i][j]

				// Update running average: v = alpha * v + (1 - alpha) * grad^2
				opt.v[key][j] = opt.alpha*opt.v[key][j] + (1-opt.alpha)*grad*grad

				if opt.momentum > 0 {
					// With momentum: buf = momentum * buf + grad / sqrt(v + eps)
					opt.buf[key][j] = opt.momentum*opt.buf[key][j] + grad/float32(math.Sqrt(float64(opt.v[key][j]+opt.epsilon)))
					layer.Kernel[j] -= learningRate * opt.buf[key][j]
				} else {
					// Without momentum: w = w - lr * grad / sqrt(v + eps)
					layer.Kernel[j] -= learningRate * grad / float32(math.Sqrt(float64(opt.v[key][j]+opt.epsilon)))
				}
			}
		}

		// Update Bias with RMSprop
		if len(layer.Bias) > 0 && len(network.biasGradients[i]) == len(layer.Bias) {
			key := fmt.Sprintf("bias_%d", i)

			if opt.v[key] == nil {
				opt.v[key] = make([]float32, len(layer.Bias))

				if opt.momentum > 0 {
					opt.buf[key] = make([]float32, len(layer.Bias))
				}
			}

			for j := range layer.Bias {
				grad := network.biasGradients[i][j]

				opt.v[key][j] = opt.alpha*opt.v[key][j] + (1-opt.alpha)*grad*grad

				if opt.momentum > 0 {
					opt.buf[key][j] = opt.momentum*opt.buf[key][j] + grad/float32(math.Sqrt(float64(opt.v[key][j]+opt.epsilon)))
					layer.Bias[j] -= learningRate * opt.buf[key][j]
				} else {
					layer.Bias[j] -= learningRate * grad / float32(math.Sqrt(float64(opt.v[key][j]+opt.epsilon)))
				}
			}
		}
	}
}

func (opt *RMSpropOptimizer) Reset() {
	opt.v = make(map[string][]float32)
	opt.buf = make(map[string][]float32)
}

func (opt *RMSpropOptimizer) GetState() map[string]interface{} {
	return map[string]interface{}{
		"type":     "rmsprop",
		"alpha":    opt.alpha,
		"epsilon":  opt.epsilon,
		"momentum": opt.momentum,
	}
}

func (opt *RMSpropOptimizer) LoadState(state map[string]interface{}) error {
	if t, ok := state["type"].(string); !ok || t != "rmsprop" {
		return fmt.Errorf("invalid optimizer type: expected rmsprop, got %v", state["type"])
	}

	if a, ok := state["alpha"].(float64); ok {
		opt.alpha = float32(a)
	}
	if eps, ok := state["epsilon"].(float64); ok {
		opt.epsilon = float32(eps)
	}
	if m, ok := state["momentum"].(float64); ok {
		opt.momentum = float32(m)
	}

	return nil
}

func (opt *RMSpropOptimizer) Name() string {
	if opt.momentum > 0 {
		return "RMSprop (momentum)"
	}
	return "RMSprop"
}
