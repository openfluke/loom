package nn

import (
	"math"
	"math/rand"
)

// NeuralTween implements a novel training approach inspired by:
// - Flash ActionScript tweening (gradual shape morphing)
// - Bidirectional BFS (search from both input and output sides)
// - Network link budgeting (signal loss estimation)
// - Optimal transport theory (finding efficient transformation paths)
//
// Instead of backpropagation, this approach:
// 1. FORWARD: Push input through, capture activations at each layer
// 2. BACKWARD ESTIMATE: From expected output, estimate what each layer SHOULD output
// 3. MEET IN MIDDLE: Like bidirectional BFS, find the "gap" at each layer
// 4. LINK BUDGET: Measure information preservation quality
// 5. TWEEN: Directly adjust weights to close the gap (no gradients needed)

// TweenState holds state for the neural tweening process
type TweenState struct {
	// Forward activations (what each layer actually outputs)
	ForwardActivations [][]float32

	// Backward targets (what each layer SHOULD output, estimated from output)
	BackwardTargets [][]float32

	// Link budgets - measure of information flow quality at each layer
	// 1.0 = perfect signal, 0.0 = complete loss
	LinkBudgets []float32

	// Gap magnitude at each layer (difference between forward and backward)
	LayerGaps []float32

	// Momentum for weight updates (dampens oscillation)
	WeightMomentum [][]float32
	BiasMomentum   [][]float32

	// Best weights found (for early stopping)
	BestScore   float64
	BestWeights [][][]float32 // [layer][weights]
	BestBiases  [][][]float32

	TotalLayers int
	TweenSteps  int
	LossHistory []float32
}

// NewTweenState creates a new tween state for a network
func NewTweenState(n *Network) *TweenState {
	totalLayers := n.TotalLayers()
	ts := &TweenState{
		ForwardActivations: make([][]float32, totalLayers+1),
		BackwardTargets:    make([][]float32, totalLayers+1),
		LinkBudgets:        make([]float32, totalLayers),
		LayerGaps:          make([]float32, totalLayers),
		WeightMomentum:     make([][]float32, totalLayers),
		BiasMomentum:       make([][]float32, totalLayers),
		BestWeights:        make([][][]float32, totalLayers),
		BestBiases:         make([][][]float32, totalLayers),
		BestScore:          0,
		TotalLayers:        totalLayers,
		TweenSteps:         0,
		LossHistory:        []float32{},
	}

	// Initialize momentum arrays based on layer sizes
	for i := 0; i < totalLayers; i++ {
		row := i / (n.GridCols * n.LayersPerCell)
		col := (i / n.LayersPerCell) % n.GridCols
		layer := i % n.LayersPerCell
		cfg := n.GetLayer(row, col, layer)
		if cfg != nil && len(cfg.Kernel) > 0 {
			ts.WeightMomentum[i] = make([]float32, len(cfg.Kernel))
			ts.BiasMomentum[i] = make([]float32, len(cfg.Bias))
		}
	}

	return ts
}

// AnalyzeForward captures activations at each layer during forward pass
func (ts *TweenState) AnalyzeForward(n *Network, input []float32) []float32 {
	output, _ := n.ForwardCPU(input)

	// Capture all activations from the network
	activations := n.Activations()
	for i := range activations {
		if i < len(ts.ForwardActivations) && len(activations[i]) > 0 {
			ts.ForwardActivations[i] = make([]float32, len(activations[i]))
			copy(ts.ForwardActivations[i], activations[i])
		}
	}

	return output
}

// EstimateBackwardTargets propagates expected output backward through layers
// This is NOT gradient computation - it's estimating what each layer SHOULD produce
// For deeper networks, we blend backward estimate with forward activations
func (ts *TweenState) EstimateBackwardTargets(n *Network, expectedClass int, outputSize int) {
	// Start with target output (one-hot for classification)
	targetOutput := make([]float32, outputSize)
	targetOutput[expectedClass] = 1.0
	ts.BackwardTargets[ts.TotalLayers] = targetOutput

	// Work backward through layers, estimating what each layer should produce
	// This uses the inverse weight relationship (like optimal transport)
	currentTarget := targetOutput

	for layerIdx := ts.TotalLayers - 1; layerIdx >= 0; layerIdx-- {
		row := layerIdx / (n.GridCols * n.LayersPerCell)
		col := (layerIdx / n.LayersPerCell) % n.GridCols
		layer := layerIdx % n.LayersPerCell

		cfg := n.GetLayer(row, col, layer)
		if cfg == nil || cfg.Type != LayerDense {
			ts.BackwardTargets[layerIdx] = currentTarget
			continue
		}

		// For dense layer: estimate input that would produce target output
		// Using pseudo-inverse concept: if output = W*input + b
		// Then estimated_input ≈ W_transpose * (output - b) scaled appropriately
		inputSize := cfg.InputHeight
		if inputSize <= 0 {
			inputSize = len(ts.ForwardActivations[layerIdx])
		}
		if inputSize <= 0 {
			ts.BackwardTargets[layerIdx] = currentTarget
			continue
		}

		estimatedInput := make([]float32, inputSize)

		// Transpose multiplication: for each input neuron, sum weighted contributions
		for i := 0; i < inputSize; i++ {
			sum := float32(0)
			weightSum := float32(0)

			for j := 0; j < len(currentTarget); j++ {
				if j < len(cfg.Bias) {
					// Remove bias influence
					adjusted := currentTarget[j] - cfg.Bias[j]*0.5
					// Weight index: input i -> output j
					wIdx := i*cfg.OutputHeight + j
					if wIdx < len(cfg.Kernel) {
						w := cfg.Kernel[wIdx]
						sum += adjusted * w
						weightSum += w * w
					}
				}
			}

			// Normalize by weight magnitude
			if weightSum > 0.001 {
				estimatedInput[i] = sum / float32(math.Sqrt(float64(weightSum)))
			} else {
				// Fall back to forward activation if weights are tiny
				if layerIdx < len(ts.ForwardActivations) && i < len(ts.ForwardActivations[layerIdx]) {
					estimatedInput[i] = ts.ForwardActivations[layerIdx][i]
				}
			}

			// Apply inverse activation (rough estimate)
			if cfg.Activation == ActivationTanh {
				estimatedInput[i] = clamp(estimatedInput[i], -0.99, 0.99)
			} else if cfg.Activation == ActivationSigmoid {
				estimatedInput[i] = clamp(estimatedInput[i], 0.01, 0.99)
			}
		}

		// DEPTH-AWARE BLENDING: blend backward estimate with forward activation
		// Deeper layers (closer to output) trust backward estimate more
		// Shallower layers blend more with actual forward activations
		// This prevents accumulated error from corrupting early layers
		depthRatio := float32(layerIdx+1) / float32(ts.TotalLayers) // 0 at input, 1 at output
		backwardWeight := 0.3 + depthRatio*0.7                      // 30-100% backward, rest forward

		if layerIdx < len(ts.ForwardActivations) && len(ts.ForwardActivations[layerIdx]) > 0 {
			for i := 0; i < len(estimatedInput) && i < len(ts.ForwardActivations[layerIdx]); i++ {
				forwardVal := ts.ForwardActivations[layerIdx][i]
				backwardVal := estimatedInput[i]
				// Blend: tween between forward (what is) and backward (what should be)
				estimatedInput[i] = forwardVal*(1-backwardWeight) + backwardVal*backwardWeight
			}
		}

		ts.BackwardTargets[layerIdx] = estimatedInput
		currentTarget = estimatedInput
	}
}

// CalculateLinkBudgets measures information flow quality at each layer
// Like WiFi link budget: how much signal is preserved vs lost
func (ts *TweenState) CalculateLinkBudgets(n *Network) {
	for i := 0; i < ts.TotalLayers; i++ {
		forward := ts.ForwardActivations[i+1] // Output of layer i
		backward := ts.BackwardTargets[i+1]   // What layer i should output

		if len(forward) == 0 || len(backward) == 0 {
			ts.LinkBudgets[i] = 1.0
			ts.LayerGaps[i] = 0
			continue
		}

		// Calculate alignment between forward and backward estimates
		// High alignment = good link, low alignment = signal loss
		dotProduct := float32(0)
		forwardMag := float32(0)
		backwardMag := float32(0)
		gapSum := float32(0)

		minLen := len(forward)
		if len(backward) < minLen {
			minLen = len(backward)
		}

		for j := 0; j < minLen; j++ {
			dotProduct += forward[j] * backward[j]
			forwardMag += forward[j] * forward[j]
			backwardMag += backward[j] * backward[j]
			gap := forward[j] - backward[j]
			gapSum += gap * gap
		}

		// Cosine similarity as link quality
		if forwardMag > 0.001 && backwardMag > 0.001 {
			cosine := dotProduct / (float32(math.Sqrt(float64(forwardMag))) * float32(math.Sqrt(float64(backwardMag))))
			ts.LinkBudgets[i] = (cosine + 1) / 2 // Map [-1,1] to [0,1]
		} else {
			ts.LinkBudgets[i] = 0.5
		}

		// Gap magnitude (MSE)
		ts.LayerGaps[i] = gapSum / float32(minLen)
	}
}

// TweenWeights directly adjusts weights to close the gap at each layer
// This is the core "tweening" - morphing weights toward better configuration
// Uses momentum to dampen oscillations
func (ts *TweenState) TweenWeights(n *Network, tweenRate float32) {
	momentum := float32(0.8) // Dampening factor

	for layerIdx := 0; layerIdx < ts.TotalLayers; layerIdx++ {
		row := layerIdx / (n.GridCols * n.LayersPerCell)
		col := (layerIdx / n.LayersPerCell) % n.GridCols
		layer := layerIdx % n.LayersPerCell

		cfg := n.GetLayer(row, col, layer)
		if cfg == nil || cfg.Type != LayerDense || len(cfg.Kernel) == 0 {
			continue
		}

		input := ts.ForwardActivations[layerIdx]
		targetOutput := ts.BackwardTargets[layerIdx+1]
		actualOutput := ts.ForwardActivations[layerIdx+1]

		if len(input) == 0 || len(targetOutput) == 0 || len(actualOutput) == 0 {
			continue
		}

		// Link budget influences how aggressively we tween
		budget := ts.LinkBudgets[layerIdx]
		adjustRate := tweenRate * (1.1 - budget) * 0.5 // Reduced rate with momentum

		// For each weight, estimate how to change it to move output toward target
		for j := 0; j < cfg.OutputHeight && j < len(targetOutput) && j < len(actualOutput); j++ {
			outputGap := targetOutput[j] - actualOutput[j]

			// Adjust bias with momentum
			if j < len(cfg.Bias) && layerIdx < len(ts.BiasMomentum) && j < len(ts.BiasMomentum[layerIdx]) {
				delta := adjustRate * outputGap * 0.1
				ts.BiasMomentum[layerIdx][j] = momentum*ts.BiasMomentum[layerIdx][j] + (1-momentum)*delta
				cfg.Bias[j] += ts.BiasMomentum[layerIdx][j]
			}

			// Adjust weights with momentum
			for i := 0; i < cfg.InputHeight && i < len(input); i++ {
				wIdx := i*cfg.OutputHeight + j
				if wIdx >= len(cfg.Kernel) {
					continue
				}

				inputAct := input[i]
				if math.Abs(float64(inputAct)) > 0.01 {
					delta := adjustRate * inputAct * outputGap * 0.01

					// Apply momentum
					if layerIdx < len(ts.WeightMomentum) && wIdx < len(ts.WeightMomentum[layerIdx]) {
						ts.WeightMomentum[layerIdx][wIdx] = momentum*ts.WeightMomentum[layerIdx][wIdx] + (1-momentum)*delta
						cfg.Kernel[wIdx] += ts.WeightMomentum[layerIdx][wIdx]
					} else {
						cfg.Kernel[wIdx] += delta
					}
				}
			}
		}

		n.SetLayer(row, col, layer, *cfg)
	}
}

// TweenStep performs one complete tween iteration
func (ts *TweenState) TweenStep(n *Network, input []float32, expectedClass int, outputSize int, tweenRate float32) float32 {
	// 1. Forward pass - capture what network actually does
	output := ts.AnalyzeForward(n, input)

	// 2. Backward estimation - figure out what each layer SHOULD do
	ts.EstimateBackwardTargets(n, expectedClass, outputSize)

	// 3. Calculate link budgets - measure information flow quality
	ts.CalculateLinkBudgets(n)

	// 4. Tween weights - morph toward better configuration
	ts.TweenWeights(n, tweenRate)

	ts.TweenSteps++

	// Calculate loss for tracking
	loss := float32(0)
	if expectedClass < len(output) {
		for i, v := range output {
			target := float32(0)
			if i == expectedClass {
				target = 1.0
			}
			diff := v - target
			loss += diff * diff
		}
	}

	return loss
}

// Train runs the tweening process over multiple epochs with early stopping
func (ts *TweenState) Train(n *Network, inputs [][]float32, expected []float64, epochs int, tweenRate float32,
	callback func(epoch int, avgLoss float32, metrics *DeviationMetrics)) {

	outputSize := 2 // Binary classification default
	// Detect output size from network
	lastLayer := n.GetLayer(
		(ts.TotalLayers-1)/(n.GridCols*n.LayersPerCell),
		((ts.TotalLayers-1)/n.LayersPerCell)%n.GridCols,
		(ts.TotalLayers-1)%n.LayersPerCell,
	)
	if lastLayer != nil && lastLayer.OutputHeight > 0 {
		outputSize = lastLayer.OutputHeight
	}

	for epoch := 0; epoch < epochs; epoch++ {
		epochLoss := float32(0)

		// Shuffle samples each epoch
		indices := rand.Perm(len(inputs))

		for _, idx := range indices {
			loss := ts.TweenStep(n, inputs[idx], int(expected[idx]), outputSize, tweenRate)
			epochLoss += loss
		}

		avgLoss := epochLoss / float32(len(inputs))
		ts.LossHistory = append(ts.LossHistory, avgLoss)

		// Evaluate and check for best score
		if epoch%5 == 0 || epoch == epochs-1 {
			metrics, _ := n.EvaluateNetwork(inputs, expected)

			// Save best weights
			if metrics.Score > ts.BestScore {
				ts.BestScore = metrics.Score
				ts.SaveBest(n)
			}

			if callback != nil {
				callback(epoch+1, avgLoss, metrics)
			}

			// Early stopping if score is high enough
			if metrics.Score >= 95 {
				ts.RestoreBest(n) // Ensure we have best weights
				return
			}
		}
	}

	// Restore best weights at end
	if ts.BestScore > 0 {
		ts.RestoreBest(n)
	}
}

// SaveBest saves current weights as the best found so far
func (ts *TweenState) SaveBest(n *Network) {
	for i := 0; i < ts.TotalLayers; i++ {
		row := i / (n.GridCols * n.LayersPerCell)
		col := (i / n.LayersPerCell) % n.GridCols
		layer := i % n.LayersPerCell
		cfg := n.GetLayer(row, col, layer)
		if cfg != nil && len(cfg.Kernel) > 0 {
			ts.BestWeights[i] = make([][]float32, 1)
			ts.BestWeights[i][0] = make([]float32, len(cfg.Kernel))
			copy(ts.BestWeights[i][0], cfg.Kernel)

			ts.BestBiases[i] = make([][]float32, 1)
			ts.BestBiases[i][0] = make([]float32, len(cfg.Bias))
			copy(ts.BestBiases[i][0], cfg.Bias)
		}
	}
}

// RestoreBest restores the best weights found during training
func (ts *TweenState) RestoreBest(n *Network) {
	for i := 0; i < ts.TotalLayers; i++ {
		if len(ts.BestWeights[i]) == 0 || len(ts.BestWeights[i][0]) == 0 {
			continue
		}
		row := i / (n.GridCols * n.LayersPerCell)
		col := (i / n.LayersPerCell) % n.GridCols
		layer := i % n.LayersPerCell
		cfg := n.GetLayer(row, col, layer)
		if cfg != nil && len(cfg.Kernel) > 0 {
			copy(cfg.Kernel, ts.BestWeights[i][0])
			if len(ts.BestBiases[i]) > 0 && len(ts.BestBiases[i][0]) > 0 {
				copy(cfg.Bias, ts.BestBiases[i][0])
			}
			n.SetLayer(row, col, layer, *cfg)
		}
	}
}

// TrainLayerwise trains layers greedily from output to input
// This is like greedy layer-wise pretraining but using tweening
// Each layer is trained to close its gap before moving to the next
func (ts *TweenState) TrainLayerwise(n *Network, inputs [][]float32, expected []float64,
	epochsPerLayer int, tweenRate float32,
	callback func(layer int, epoch int, score float64)) {

	outputSize := 2
	lastLayer := n.GetLayer(
		(ts.TotalLayers-1)/(n.GridCols*n.LayersPerCell),
		((ts.TotalLayers-1)/n.LayersPerCell)%n.GridCols,
		(ts.TotalLayers-1)%n.LayersPerCell,
	)
	if lastLayer != nil && lastLayer.OutputHeight > 0 {
		outputSize = lastLayer.OutputHeight
	}

	// Train from output layer backward to input
	for targetLayer := ts.TotalLayers - 1; targetLayer >= 0; targetLayer-- {
		for epoch := 0; epoch < epochsPerLayer; epoch++ {
			indices := rand.Perm(len(inputs))

			for _, idx := range indices {
				// Forward pass
				ts.AnalyzeForward(n, inputs[idx])

				// Backward estimation
				ts.EstimateBackwardTargets(n, int(expected[idx]), outputSize)

				// Calculate link budgets
				ts.CalculateLinkBudgets(n)

				// Only tween the target layer (and those after it that are already trained)
				ts.TweenSingleLayer(n, targetLayer, tweenRate)
			}

			ts.TweenSteps++
		}

		// Report progress after each layer
		if callback != nil {
			metrics, _ := n.EvaluateNetwork(inputs, expected)
			callback(targetLayer, epochsPerLayer, metrics.Score)
		}
	}
}

// TweenSingleLayer tweens only one specific layer
func (ts *TweenState) TweenSingleLayer(n *Network, layerIdx int, tweenRate float32) {
	row := layerIdx / (n.GridCols * n.LayersPerCell)
	col := (layerIdx / n.LayersPerCell) % n.GridCols
	layer := layerIdx % n.LayersPerCell

	cfg := n.GetLayer(row, col, layer)
	if cfg == nil || cfg.Type != LayerDense || len(cfg.Kernel) == 0 {
		return
	}

	input := ts.ForwardActivations[layerIdx]
	targetOutput := ts.BackwardTargets[layerIdx+1]
	actualOutput := ts.ForwardActivations[layerIdx+1]

	if len(input) == 0 || len(targetOutput) == 0 || len(actualOutput) == 0 {
		return
	}

	budget := ts.LinkBudgets[layerIdx]
	adjustRate := tweenRate * (1.5 - budget) // Stronger adjustment for single layer

	for j := 0; j < cfg.OutputHeight && j < len(targetOutput) && j < len(actualOutput); j++ {
		outputGap := targetOutput[j] - actualOutput[j]

		if j < len(cfg.Bias) {
			cfg.Bias[j] += adjustRate * outputGap * 0.2
		}

		for i := 0; i < cfg.InputHeight && i < len(input); i++ {
			wIdx := i*cfg.OutputHeight + j
			if wIdx >= len(cfg.Kernel) {
				continue
			}

			inputAct := input[i]
			if math.Abs(float64(inputAct)) > 0.01 {
				delta := adjustRate * inputAct * outputGap * 0.02
				cfg.Kernel[wIdx] += delta
			}
		}
	}

	n.SetLayer(row, col, layer, *cfg)
}

// GetBudgetSummary returns summary statistics for link budgets
func (ts *TweenState) GetBudgetSummary() (avg, min, max float32) {
	if len(ts.LinkBudgets) == 0 {
		return 0.5, 0.5, 0.5
	}

	min, max = ts.LinkBudgets[0], ts.LinkBudgets[0]
	sum := float32(0)

	for _, b := range ts.LinkBudgets {
		sum += b
		if b < min {
			min = b
		}
		if b > max {
			max = b
		}
	}

	avg = sum / float32(len(ts.LinkBudgets))
	return
}

// GetGapSummary returns summary of layer gaps
func (ts *TweenState) GetGapSummary() (avg, max float32) {
	if len(ts.LayerGaps) == 0 {
		return 0, 0
	}

	sum := float32(0)
	for _, g := range ts.LayerGaps {
		sum += g
		if g > max {
			max = g
		}
	}
	avg = sum / float32(len(ts.LayerGaps))
	return
}

// CalculateLinkBudgets as method for external access
func (ts *TweenState) CalculateLinkBudgetsFromSample(n *Network, input []float32) {
	ts.AnalyzeForward(n, input)
	// For initial budget calculation without targets, just analyze variance
	for i := 0; i < ts.TotalLayers; i++ {
		if i+1 < len(ts.ForwardActivations) && len(ts.ForwardActivations[i+1]) > 0 {
			act := ts.ForwardActivations[i+1]
			mean := float32(0)
			for _, v := range act {
				mean += v
			}
			mean /= float32(len(act))

			variance := float32(0)
			for _, v := range act {
				diff := v - mean
				variance += diff * diff
			}
			variance /= float32(len(act))

			// Higher variance = more information preserved
			ts.LinkBudgets[i] = float32(math.Min(1.0, float64(variance)*2+0.3))
		} else {
			ts.LinkBudgets[i] = 0.5
		}
	}
}

// Helper functions
func clamp(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
