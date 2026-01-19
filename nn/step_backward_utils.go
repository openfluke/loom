package nn

// ApplyGradients applies the stored gradients to the weights
// If an optimizer is set, it will use that optimizer's update rule
// Otherwise, it falls back to simple SGD (w = w - lr * grad)
func (n *Network) ApplyGradients(learningRate float32) {
	// GPU path: apply gradients directly on GPU
	if n.GPU && n.gpuMounted {
		n.applyGradientsGPU(learningRate)
		return
	}

	// Use optimizer if set
	if n.optimizer != nil {
		n.optimizer.Step(n, learningRate)
		return
	}

	// Fallback to simple SGD (backward compatible)
	for i := 0; i < n.TotalLayers(); i++ {
		layer := &n.Layers[i]

		// 1. Update Kernels
		if len(layer.Kernel) > 0 && len(n.kernelGradients[i]) == len(layer.Kernel) {
			for j := range layer.Kernel {
				layer.Kernel[j] -= learningRate * n.kernelGradients[i][j]
			}
		}

		// 2. Update Biases
		if len(layer.Bias) > 0 && len(n.biasGradients[i]) == len(layer.Bias) {
			for j := range layer.Bias {
				layer.Bias[j] -= learningRate * n.biasGradients[i][j]
			}
		}

		// 3. Recurse into sub-networks and nested layers
		if layer.Type == LayerKMeans && layer.SubNetwork != nil {
			layer.SubNetwork.ApplyGradients(learningRate)
		}

		if layer.Type == LayerSequential || layer.Type == LayerParallel {
			for j := range layer.ParallelBranches {
				applyGradientsToConfig(&layer.ParallelBranches[j], learningRate)
			}
		}
	}
}

// applyGradientsToConfig recursively applies gradients to nested layer configurations.
func applyGradientsToConfig(cfg *LayerConfig, lr float32) {
	// 1. Recurse into KMeans Sub-Network
	if cfg.Type == LayerKMeans && cfg.SubNetwork != nil {
		cfg.SubNetwork.ApplyGradients(lr)
	}

	// 2. Recurse into Sequential/Parallel branches
	if cfg.Type == LayerSequential || cfg.Type == LayerParallel {
		for i := range cfg.ParallelBranches {
			applyGradientsToConfig(&cfg.ParallelBranches[i], lr)
		}
		// Also update filter gate if present
		if cfg.FilterGateConfig != nil {
			applyGradientsToConfig(cfg.FilterGateConfig, lr)
		}
	}

	// 3. Fallback: Update kernels/biases directly if they have them
	// This is for nested layers that aren't top-level network layers.
	// Since we don't have a flattened gradient pool for nested configs yet,
	// we assume they might handle it via their own backward/ApplyGradients logic
	// (like SubNetwork) or we'd need to extend this to support flattened sub-grads.
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
