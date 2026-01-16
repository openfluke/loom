package nn

import (
	"fmt"
	"math"
)

// =============================================================================
// DIFFERENTIABLE K-MEANS CLUSTERING LAYER
// =============================================================================
//
// This layer acts like a "learnable softmax" that automatically clusters inputs.
// The cluster centers are learnable parameters updated through backpropagation.
//
// FORWARD PASS:
//   1. Transform input through attached sub-network (feature extraction)
//   2. Compute similarity/distance from features to each cluster center
//   3. Apply softmax to get soft cluster assignments (probabilities)
//   4. Output: K probabilities indicating cluster membership
//
// BACKWARD PASS:
//   1. Receive gradient w.r.t. output cluster probabilities
//   2. Compute gradient w.r.t. cluster centers (stored for ApplyGradients)
//   3. Propagate gradient back through attached sub-network
//
// The cluster assignments EMERGE from training - the network learns to
// organize inputs into meaningful clusters based on the training signal.
// =============================================================================

// ForwardKMeansCPU performs the forward pass for a differentiable K-Means layer
func ForwardKMeansCPU(input []float32, config *LayerConfig) ([]float32, error) {
	if config.SubNetwork == nil {
		// Fallback for legacy initialization (if any), though InitKMeansLayer should handle this.
		return nil, fmt.Errorf("KMeans layer requires a SubNetwork for feature transformation")
	}

	// DEBUG: Verify SubNetwork layer type
	if net, ok := config.SubNetwork.(*Network); ok {
		layer := net.GetLayer(0, 0, 0)
		if layer != nil && layer.Type == LayerDense && config.AttachedLayer != nil && config.AttachedLayer.Type != LayerDense {
			fmt.Printf("DEBUG: SubNetwork Layer Type mismatch! Got Dense(0), expected %d\n", config.AttachedLayer.Type)
		}
	}

	// Step 1: Transform input through attached sub-network to get features
	// Input MUST be float32 for CPU pass
	features, _ := config.SubNetwork.ForwardCPU(input)

	// We don't error check the second return value (duration)
	// If features is nil or empty, something went wrong, but subNet should have panic/logged or returns empty
	if len(features) == 0 {
		return nil, fmt.Errorf("sub-network forward returned empty features")
	}

	// Store features in config for backward pass (to be picked up by forward.go)
	config.PreActivations = features

	featureDim := len(features)

	// Initialize or resize cluster centers if needed (lazy initialization)
	// This allows the layer to adapt to the actual feature dimension
	if len(config.ClusterCenters) == 0 || config.ClusterDim != featureDim {
		config.ClusterDim = featureDim
		config.ClusterCenters = make([]float32, config.NumClusters*featureDim)
		// Initialize centers spread around feature space
		for k := 0; k < config.NumClusters; k++ {
			centerOffset := k * featureDim
			for d := 0; d < featureDim; d++ {
				// Initialize with small random offsets from input features
				spread := float32(k-config.NumClusters/2) * 0.2
				config.ClusterCenters[centerOffset+d] = features[d]*(1+spread*0.1) + (randomFloat()*2-1)*0.1
			}
		}
		// Allocate gradient storage
		config.ClusterGradients = make([]float32, config.NumClusters*featureDim)
	}

	// Step 2: Compute negative squared distances (for softmax - closer = higher)
	// Using squared Euclidean distance: d² = Σ(x_i - c_i)²
	logits := make([]float32, config.NumClusters)
	for k := 0; k < config.NumClusters; k++ {
		centerOffset := k * config.ClusterDim
		var sqDist float32
		for d := 0; d < config.ClusterDim; d++ {
			diff := features[d] - config.ClusterCenters[centerOffset+d]
			sqDist += diff * diff
		}
		// Negative distance so closer centers get higher logits
		temp := config.KMeansTemperature
		if temp == 0 {
			temp = 1.0
		}
		logits[k] = -sqDist / (2 * temp * temp) // Gaussian-style soft assignment
	}

	// Step 3: Apply softmax to get probabilities
	assignments := softmaxSimple(logits)

	// Step 4: Output based on mode
	if config.KMeansOutputMode == "features" {
		// Weighted sum of cluster centers (soft clustering output)
		output := make([]float32, config.ClusterDim)
		for k := 0; k < config.NumClusters; k++ {
			weight := assignments[k]
			centerOffset := k * config.ClusterDim
			for d := 0; d < config.ClusterDim; d++ {
				output[d] += weight * config.ClusterCenters[centerOffset+d]
			}
		}
		return output, nil
	}

	// Default: return cluster probabilities
	return assignments, nil
}

// BackwardKMeansCPU computes gradients for cluster centers and propagates to attached layer
func BackwardKMeansCPU(gradOutput []float32, config *LayerConfig, input []float32, features []float32, assignments []float32, learningRate float32) ([]float32, error) {
	if config.SubNetwork == nil {
		return nil, fmt.Errorf("KMeans layer requires a SubNetwork")
	}

	if len(features) == 0 || len(config.ClusterCenters) == 0 {
		return make([]float32, len(input)), nil
	}

	temp := config.KMeansTemperature
	if temp == 0 {
		temp = 1.0
	}
	tempSq := temp * temp

	// Initialize gradient storage if needed
	if len(config.ClusterGradients) != len(config.ClusterCenters) {
		config.ClusterGradients = make([]float32, len(config.ClusterCenters))
	}

	// Gradient w.r.t. features (for backprop to attached layer)
	gradFeatures := make([]float32, config.ClusterDim)

	if config.KMeansOutputMode == "features" {
		// Output mode: weighted sum of cluster centers
		gradAssignments := make([]float32, config.NumClusters)
		for k := 0; k < config.NumClusters; k++ {
			centerOffset := k * config.ClusterDim
			for d := 0; d < config.ClusterDim; d++ {
				gradAssignments[k] += config.ClusterCenters[centerOffset+d] * gradOutput[d]
			}
		}

		gradLogits := softmaxBackwardSimple(gradAssignments, assignments)

		for k := 0; k < config.NumClusters; k++ {
			centerOffset := k * config.ClusterDim
			weight := assignments[k]

			for d := 0; d < config.ClusterDim; d++ {
				config.ClusterGradients[centerOffset+d] += weight * gradOutput[d]
				diff := features[d] - config.ClusterCenters[centerOffset+d]
				config.ClusterGradients[centerOffset+d] += gradLogits[k] * diff / tempSq

				gradFeatures[d] += weight * gradOutput[d]
				gradFeatures[d] -= gradLogits[k] * diff / tempSq
			}
		}
	} else {
		// Output mode: probabilities
		gradLogits := softmaxBackwardSimple(gradOutput, assignments)

		for k := 0; k < config.NumClusters; k++ {
			centerOffset := k * config.ClusterDim
			for d := 0; d < config.ClusterDim; d++ {
				diff := features[d] - config.ClusterCenters[centerOffset+d]
				config.ClusterGradients[centerOffset+d] += gradLogits[k] * diff / tempSq
				gradFeatures[d] -= gradLogits[k] * diff / tempSq
			}
		}
	}

	// Apply gradients to cluster centers directly (in-place update)
	for i := range config.ClusterCenters {
		config.ClusterCenters[i] -= learningRate * config.ClusterGradients[i]
		config.ClusterGradients[i] = 0 // Reset
	}

	// Backpropagate gradient through sub-network
	gradInput, _ := config.SubNetwork.BackwardCPU(gradFeatures)

	// Apply updates to sub-network layers (SGD approximation)
	if net, ok := config.SubNetwork.(*Network); ok {
		// Iterate over layers to apply gradients
		// Access grid for Layer 0,0,0
		layerConfig := net.GetLayer(0, 0, 0)
		layerIdx := 0 // For single layer network in sub-network

		if layerIdx < len(net.kernelGradients) && net.kernelGradients[layerIdx] != nil {
			kGrad := net.kernelGradients[layerIdx]

			// Dense/Conv uses Kernel/Bias
			if len(layerConfig.Kernel) > 0 && len(kGrad) > 0 {
				for i := range layerConfig.Kernel {
					if i < len(kGrad) {
						layerConfig.Kernel[i] -= learningRate * kGrad[i]
					}
				}
			}
			// Norm uses Gamma
			if len(layerConfig.Gamma) > 0 && len(kGrad) > 0 {
				for i := range layerConfig.Gamma {
					if i < len(kGrad) {
						layerConfig.Gamma[i] -= learningRate * kGrad[i]
					}
				}
			}
		}

		if layerIdx < len(net.biasGradients) && net.biasGradients[layerIdx] != nil {
			bGrad := net.biasGradients[layerIdx]

			// Bias
			if len(layerConfig.Bias) > 0 && len(bGrad) > 0 {
				for i := range layerConfig.Bias {
					if i < len(bGrad) {
						layerConfig.Bias[i] -= learningRate * bGrad[i]
					}
				}
			}
			// Norm Beta
			if len(layerConfig.Beta) > 0 && len(bGrad) > 0 {
				for i := range layerConfig.Beta {
					if i < len(bGrad) {
						layerConfig.Beta[i] -= learningRate * bGrad[i]
					}
				}
			}
		}
	}

	return gradInput, nil
}

// softmaxBackwardSimple computes gradient through softmax
// Given ∂L/∂output, returns ∂L/∂logits
func softmaxBackwardSimple(gradOutput []float32, softmaxOutput []float32) []float32 {
	n := len(gradOutput)
	gradLogits := make([]float32, n)

	// Jacobian of softmax: ∂p_i/∂z_j = p_i * (δ_ij - p_j)
	// So: ∂L/∂z_j = Σ_i ∂L/∂p_i * p_i * (δ_ij - p_j)
	//             = p_j * (∂L/∂p_j - Σ_i ∂L/∂p_i * p_i)

	// Compute dot product: Σ_i ∂L/∂p_i * p_i
	var dotProd float32
	for i := 0; i < n; i++ {
		dotProd += gradOutput[i] * softmaxOutput[i]
	}

	// Compute gradient
	for j := 0; j < n; j++ {
		gradLogits[j] = softmaxOutput[j] * (gradOutput[j] - dotProd)
	}

	return gradLogits
}

// Helper function to forward through any layer type (Legacy/Utility)
// NOTE: Main forward pass now uses SubNetwork, skipping this.
// Kept for compatibility if used elsewhere, though it duplicates logic.
func forwardLayerHelper(input []float32, config *LayerConfig) ([]float32, error) {
	switch config.Type {
	case LayerDense:
		preAct, postAct := denseForwardCPU(input, config, 1)
		config.PreActivations = preAct
		return postAct, nil
	case LayerConv1D:
		preAct, postAct := conv1DForwardCPU(input, config, 1)
		config.PreActivations = preAct
		return postAct, nil
	case LayerConv2D:
		preAct, postAct := conv2DForwardCPU(input, config, 1)
		config.PreActivations = preAct
		return postAct, nil
	case LayerLSTM:
		output, states := lstmForwardCPU(config, input, 1, config.SeqLength, config.RNNInputSize, config.HiddenSize)
		// Serialize states for backward pass
		var serialized []float32
		serialized = append(serialized, states["hidden"]...)
		serialized = append(serialized, states["cell"]...)
		serialized = append(serialized, states["i_gate"]...)
		serialized = append(serialized, states["f_gate"]...)
		serialized = append(serialized, states["g_gate"]...)
		serialized = append(serialized, states["o_gate"]...)
		serialized = append(serialized, states["c_tanh"]...)
		config.PreActivations = serialized
		return output, nil
	case LayerRNN:
		output, hiddenStates := rnnForwardCPU(config, input, 1, config.SeqLength, config.RNNInputSize, config.HiddenSize)
		config.PreActivations = hiddenStates
		return output, nil
	case LayerSwiGLU:
		_, postAct := SwiGLUForwardCPU(input, config, 1)
		config.PreActivations = postAct
		return postAct, nil
	case LayerMultiHeadAttention:
		output, concatenated := MultiHeadAttentionForwardCPU(input, config, 1)
		config.PreActivations = concatenated
		return output, nil
	case LayerNorm:
		output := make([]float32, len(input))
		layerNormForwardCPU(input, output, config, 1)
		config.PreActivations = input
		return output, nil
	case LayerRMSNorm:
		output := make([]float32, len(input))
		rmsNormForwardCPU(input, output, config, 1)
		config.PreActivations = input
		return output, nil
	case LayerSequential:
		output, intermediates, err := sequentialForwardCPU(input, config.ParallelBranches, 1)
		if err != nil {
			return nil, err
		}

		// Serialize intermediates: [len1, data1..., len2, data2...]
		var serialized []float32
		for _, chunk := range intermediates {
			serialized = append(serialized, float32(len(chunk)))
			serialized = append(serialized, chunk...)
		}
		config.PreActivations = serialized
		return output, nil
	case LayerEmbedding:
		return nil, fmt.Errorf("embedding layer not supported as attached layer (requires integer indices)")
	default:
		return nil, fmt.Errorf("unsupported attached layer type: %d", config.Type)
	}
}

// Helper to update standard kernel/bias weights (Legacy/Utility)
func applyGradients(config *LayerConfig, gradKernel, gradBias []float32, learningRate float32) {
	for i := range config.Kernel {
		if i < len(gradKernel) {
			config.Kernel[i] -= learningRate * gradKernel[i]
		}
	}
	for i := range config.Bias {
		if i < len(gradBias) {
			config.Bias[i] -= learningRate * gradBias[i]
		}
	}
}

// Helper for Norm layers (Legacy/Utility)
func applyGradientsGammaBeta(config *LayerConfig, gradGamma, gradBeta []float32, learningRate float32) {
	for i := range config.Gamma {
		if i < len(gradGamma) {
			config.Gamma[i] -= learningRate * gradGamma[i]
		}
	}
	for i := range config.Beta {
		if i < len(gradBeta) {
			config.Beta[i] -= learningRate * gradBeta[i]
		}
	}
}

// Distance metric helpers
func euclideanDistance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

func manhattanDistance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += float32(math.Abs(float64(a[i] - b[i])))
	}
	return sum
}

func cosineDistance(a, b []float32) float32 {
	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 1.0
	}
	similarity := dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
	return 1.0 - similarity
}

// Simple softmax implementation
func softmaxSimple(logits []float32) []float32 {
	maxLogit := logits[0]
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}

	expValues := make([]float32, len(logits))
	var sumExp float32
	for i, v := range logits {
		expValues[i] = float32(math.Exp(float64(v - maxLogit)))
		sumExp += expValues[i]
	}

	for i := range expValues {
		expValues[i] /= sumExp
	}

	return expValues
}

// InitKMeansLayer creates a differentiable K-Means clustering layer
//
// numClusters: number of cluster centers (K)
// attachedLayer: sub-network for feature extraction (Dense, Conv, RNN, etc.)
// outputMode: "probabilities" (K values) or "features" (weighted center sum)
func InitKMeansLayer(numClusters int, attachedLayer LayerConfig, outputMode string) LayerConfig {
	// Determine feature dimension from attached layer output
	featureDim := attachedLayer.OutputHeight
	if featureDim == 0 {
		featureDim = attachedLayer.HiddenSize // For RNN/LSTM
	}
	if featureDim == 0 {
		featureDim = 1 // Will be resized on first forward
	}

	if outputMode == "" {
		outputMode = "probabilities"
	}

	// Create a sub-network for the attached layer
	// Input size is determined by the input to the KMeans layer (which will be known at runtime)
	// But we need to initialize the network structure.
	// For now, assume input dimension matches attached layer's input height.
	inputDim := attachedLayer.InputHeight
	if inputDim == 0 {
		inputDim = attachedLayer.RNNInputSize
	}
	if inputDim == 0 {
		inputDim = 1 // Fallback
	}

	// Initialize a new Network to hold the attached layer
	// We use a small network with just this layer
	subNet := NewNetwork(inputDim, 1, 1, 1) // Batch size 1, 1 layer
	subNet.BatchSize = 1
	// We need to set the layer at 0,0,0
	subNet.SetLayer(0, 0, 0, attachedLayer)
	subNet.InitializeWeights() // Ensure weights are ready

	return LayerConfig{
		Type:              LayerKMeans,
		NumClusters:       numClusters,
		ClusterCenters:    nil, // Lazy initialization on first forward
		ClusterGradients:  nil, // Lazy initialization
		ClusterDim:        featureDim,
		AttachedLayer:     nil, // We use SubNetwork now
		SubNetwork:        subNet,
		DistanceMetric:    "euclidean",
		KMeansTemperature: 1.0,
		KMeansOutputMode:  outputMode,
	}
}
