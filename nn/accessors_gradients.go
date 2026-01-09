package nn

// GetKernelGradients returns gradients for a specific layer index
func (n *Network) GetKernelGradients(layerIdx int) []float32 {
	// Check bounds
	if layerIdx < 0 || layerIdx >= len(n.kernelGradients) {
		return nil
	}
	// It is a concrete type [][]float32
	return n.kernelGradients[layerIdx]
}

// GetBiasGradients returns bias gradients for a specific layer index
func (n *Network) GetBiasGradients(layerIdx int) []float32 {
	if layerIdx < 0 || layerIdx >= len(n.biasGradients) {
		return nil
	}
	return n.biasGradients[layerIdx]
}
