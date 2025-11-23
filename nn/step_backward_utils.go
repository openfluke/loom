package nn

// ApplyGradients applies the stored gradients to the weights
// If an optimizer is set, it will use that optimizer's update rule
// Otherwise, it falls back to simple SGD (w = w - lr * grad)
func (n *Network) ApplyGradients(learningRate float32) {
	// Use optimizer if set
	if n.optimizer != nil {
		n.optimizer.Step(n, learningRate)
		return
	}

	// Fallback to simple SGD (backward compatible)
	for i := 0; i < n.TotalLayers(); i++ {
		// Update Kernels
		if len(n.Layers[i].Kernel) > 0 && len(n.kernelGradients[i]) == len(n.Layers[i].Kernel) {
			for j := range n.Layers[i].Kernel {
				n.Layers[i].Kernel[j] -= learningRate * n.kernelGradients[i][j]
			}
		}

		// Update Biases
		if len(n.Layers[i].Bias) > 0 && len(n.biasGradients[i]) == len(n.Layers[i].Bias) {
			for j := range n.Layers[i].Bias {
				n.Layers[i].Bias[j] -= learningRate * n.biasGradients[i][j]
			}
		}

		// Update Attention Weights
		if n.Layers[i].Type == LayerMultiHeadAttention {
			// This is more complex because weights are split.
			// Simplified: we concatenated them in Backward, so we need to split them back
			// or access them via a unified view.
			// For this example, let's assume we only support Dense/Conv2D for now
			// or we need to implement the split logic.
			// Given the example uses Dense, this simple loop works for Dense/Conv2D.
			// For Attention, we would need to unpack `kernelGradients[i]`.
		}
	}
}

// SetOptimizer sets the optimizer to use for gradient updates
func (n *Network) SetOptimizer(opt Optimizer) {
	n.optimizer = opt
	if opt != nil {
		n.optimizerType = opt.Name()
	} else {
		n.optimizerType = ""
	}
}

// GetOptimizer returns the current optimizer (may be nil)
func (n *Network) GetOptimizer() Optimizer {
	return n.optimizer
}

// ResetOptimizer clears the optimizer state
func (n *Network) ResetOptimizer() {
	if n.optimizer != nil {
		n.optimizer.Reset()
	}
}

// ============================================================================
// Convenience methods for setting specific optimizers
// These are automatically exposed to WASM/Python/C#/TypeScript via reflection
// ============================================================================

// ApplyGradientsAdamW is a convenience method for using AdamW optimizer
// Automatically creates and sets an AdamW optimizer if not already set
func (n *Network) ApplyGradientsAdamW(learningRate, beta1, beta2, weightDecay float32) {
	// Create AdamW optimizer if not set or if it's a different type
	if n.optimizer == nil || n.optimizerType != "AdamW" {
		n.SetOptimizer(NewAdamWOptimizer(beta1, beta2, 1e-8, weightDecay))
	}
	n.ApplyGradients(learningRate)
}

// ApplyGradientsRMSprop is a convenience method for using RMSprop optimizer
func (n *Network) ApplyGradientsRMSprop(learningRate, alpha, epsilon, momentum float32) {
	if n.optimizer == nil || n.optimizerType != "RMSprop" {
		n.SetOptimizer(NewRMSpropOptimizer(alpha, epsilon, momentum))
	}
	n.ApplyGradients(learningRate)
}

// ApplyGradientsSGDMomentum is a convenience method for using SGD with momentum
func (n *Network) ApplyGradientsSGDMomentum(learningRate, momentum, dampening float32, nesterov bool) {
	if n.optimizer == nil || n.optimizerType != "SGD (momentum)" {
		n.SetOptimizer(NewSGDOptimizerWithMomentum(momentum, dampening, nesterov))
	}
	n.ApplyGradients(learningRate)
}
