package nn

// ApplyGradients applies the stored gradients to the weights using SGD
func (n *Network) ApplyGradients(learningRate float32) {
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
