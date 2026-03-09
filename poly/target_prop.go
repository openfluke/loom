package poly

import (
	"math"
)

/*
NEURAL Target Propagation (Bidirectional Target Propagation)
--------------------------------------------------
This technique bridges the gap beTargetProp the actual activations produced
during the forward pass and the idealized targets needed for correct output.

Key Concepts:
- ForwardActs: What the layers produced (Top-Down).
- BackwardTargets: What the layers SHOULD have produced (Bottom-Up).
- Chaining: When enabled, targets are derived from gradients (Act + Grad).
- Link Budget: Measures the fidelity of signal preservation across the mesh.
*/

// TargetPropConfig holds tunable parameters for Neural Target Propagation.
type TargetPropConfig struct {
	BatchSize        int
	UseChainRule     bool    // If true, targets = Act + Grad * Scale
	GradientScale    float32 // Scaling factor for chaining
	DepthScaleFactor float32 // Gradient boosting for deeper layers
	Momentum         float32
	LearningRate     float32

	// Clamping for stability
	ActivationClamp float32
}

// DefaultTargetPropConfig returns standard settings for the TargetProp engine.
func DefaultTargetPropConfig() *TargetPropConfig {
	return &TargetPropConfig{
		BatchSize:        1,
		UseChainRule:     true,
		GradientScale:    0.1,
		DepthScaleFactor: 1.1,
		Momentum:         0.9,
		LearningRate:     0.01,
		ActivationClamp:  10.0,
	}
}

// TargetPropState tracks the bidirectional signal flow.
type TargetPropState[T Numeric] struct {
	ForwardActs     []*Tensor[T]
	PreActs         []*Tensor[T] // Internal pre-activation states for weight-bearing layers
	BackwardTargets []*Tensor[T]

	// Chain Rule storage
	Gradients []*Tensor[float32]

	// Diagnostics
	LinkBudgets []float32
	Gaps        []float32

	Config      *TargetPropConfig
	TotalLayers int
}

// NewTargetPropState initializes a state for the given volumetric network.
func NewTargetPropState[T Numeric](n *VolumetricNetwork, config *TargetPropConfig) *TargetPropState[T] {
	if config == nil {
		config = DefaultTargetPropConfig()
	}
	total := len(n.Layers)
	return &TargetPropState[T]{
		ForwardActs:     make([]*Tensor[T], total+1),
		PreActs:         make([]*Tensor[T], total+1),
		BackwardTargets: make([]*Tensor[T], total+1),
		Gradients:       make([]*Tensor[float32], total+1),
		LinkBudgets:     make([]float32, total),
		Gaps:            make([]float32, total),
		Config:          config,
		TotalLayers:     total,
	}
}

// TargetPropForward executes a standard forward pass but captures ALL activations.
func TargetPropForward[T Numeric](n *VolumetricNetwork, s *TargetPropState[T], input *Tensor[T]) *Tensor[T] {
	s.ForwardActs[0] = input.Clone()

	current := input
	for i := range n.Layers {
		l := &n.Layers[i]
		pre, post := DispatchLayer(l, current, nil)
		s.PreActs[i+1] = pre
		s.ForwardActs[i+1] = post
		current = post
	}
	return current
}

// TargetPropBackward generates targets or gradients from the output back to the input.
func TargetPropBackward[T Numeric](n *VolumetricNetwork, s *TargetPropState[T], target *Tensor[T]) {
	if s.Config.UseChainRule {
		TargetPropBackwardChainRule(n, s, target)
	} else {
		TargetPropBackwardTargetProp(n, s, target)
	}
}

// TargetPropBackwardChainRule uses standard gradients to shift targets.
func TargetPropBackwardChainRule[T Numeric](n *VolumetricNetwork, s *TargetPropState[T], target *Tensor[T]) {
	outputIdx := s.TotalLayers
	actualOutput := s.ForwardActs[outputIdx]
	if actualOutput == nil {
		return
	}
	s.BackwardTargets[outputIdx] = target.Clone()

	// Initial Error Gradient
	grad := NewTensor[float32](target.Shape...)
	for i := range grad.Data {
		grad.Data[i] = float32(target.Data[i]) - float32(actualOutput.Data[i])
	}
	s.Gradients[outputIdx] = grad

	currentGrad := grad
	for i := s.TotalLayers - 1; i >= 0; i-- {
		l := &n.Layers[i]
		input := s.ForwardActs[i]
		preAct := s.PreActs[i+1]
		if preAct == nil {
			preAct = s.ForwardActs[i+1] // Fallback for pure activation layers
		}
		if input == nil {
			continue
		}

		gIn, _ := DispatchLayerBackward(l, ConvertTensor[float32, T](currentGrad), input, nil, preAct)
		f32GradIn := ConvertTensor[T, float32](gIn)
		s.Gradients[i] = f32GradIn
		currentGrad = f32GradIn

		// Target = Actual + Grad * Scale
		targetT := NewTensor[T](input.Shape...)

		for j := range targetT.Data {
			val := float32(input.Data[j]) + f32GradIn.Data[j]*s.Config.GradientScale
			targetT.Data[j] = T(val)
		}
		s.BackwardTargets[i] = targetT
	}
}

// TargetPropBackwardTargetProp uses true Target Propagation (without derivatives).
func TargetPropBackwardTargetProp[T Numeric](n *VolumetricNetwork, s *TargetPropState[T], target *Tensor[T]) {
	outputIdx := s.TotalLayers
	s.BackwardTargets[outputIdx] = target.Clone()

	for i := s.TotalLayers - 1; i >= 0; i-- {
		l := &n.Layers[i]

		// Mesh-Aware: The target for this layer's output might have been
		// propagated from a layer further down the grid.
		currentTarget := s.BackwardTargets[i+1]
		if currentTarget == nil {
			continue
		}

		input := s.ForwardActs[i]
		if input == nil {
			continue
		}

		estimatedTarget := NewTensor[T](input.Shape...)

		// True Target Propagation logic based on layer type
		switch l.Type {
		case LayerDense:
			outSize := l.OutputHeight
			inSize := l.InputHeight
			if l.WeightStore != nil && len(l.WeightStore.Master) > 0 {
				weights := l.WeightStore.Master
				for in := 0; in < inSize; in++ {
					importance := float32(0)
					totalWeight := float32(0)
					for out := 0; out < outSize && out < len(currentTarget.Data); out++ {
						wIdx := in*outSize + out
						if wIdx < len(weights) {
							w := weights[wIdx]
							importance += w * float32(currentTarget.Data[out])
							if w < 0 {
								totalWeight -= w
							} else {
								totalWeight += w
							}
						}
					}
					// Only assign target if enough weight connects this neuron
					if totalWeight > 0.01 {
						estimatedTarget.Data[in] = T(importance / totalWeight)
					}
				}
			}
		case LayerRNN:
			// For RNN, calculate input target using wIH
			inSize := l.InputChannels
			hiddenSize := l.OutputHeight
			if l.WeightStore != nil && len(l.WeightStore.Master) > 0 {
				weights := l.WeightStore.Master
				ihSize := hiddenSize * inSize
				wIH := weights[0:ihSize]

				seqLen := len(input.Data) / inSize
				for s := 0; s < seqLen; s++ {
					for in := 0; in < inSize; in++ {
						importance := float32(0)
						totalWeight := float32(0)
						for h := 0; h < hiddenSize; h++ {
							wIdx := h*inSize + in
							if wIdx < len(wIH) {
								w := wIH[wIdx]
								importance += w * float32(currentTarget.Data[s*hiddenSize+h])
								if w < 0 {
									totalWeight -= w
								} else {
									totalWeight += w
								}
							}
						}
						if totalWeight > 0.01 {
							estimatedTarget.Data[s*inSize+in] = T(importance / totalWeight)
						}
					}
				}
			}
		case LayerLSTM:
			// For LSTM, calculate target by aggregating signals through all 4 gates
			inSize := l.InputChannels
			hiddenSize := l.OutputHeight
			if l.WeightStore != nil && len(l.WeightStore.Master) > 0 {
				weights := l.WeightStore.Master
				// In LSTMForwardPolymorphic, each gate weight size is ihSize + hhSize + bSize
				// gateWeightCount = ihSize + hhSize + bSize
				ihSize := hiddenSize * inSize
				hhSize := hiddenSize * hiddenSize
				bSize := hiddenSize
				gateWeightCount := ihSize + hhSize + bSize

				// Map the 4 gates (using only Input-to-Hidden for back-targeting simplicity)
				gates := [][]float32{
					weights[0:ihSize], // Input (IH part)
					weights[gateWeightCount : gateWeightCount+ihSize],     // Forget (IH part)
					weights[2*gateWeightCount : 2*gateWeightCount+ihSize], // Cell (IH part)
					weights[3*gateWeightCount : 3*gateWeightCount+ihSize], // Output (IH part)
				}

				seqLen := len(input.Data) / inSize
				for s := 0; s < seqLen; s++ {
					for in := 0; in < inSize; in++ {
						importance := float32(0)
						totalWeight := float32(0)
						for g := 0; g < 4; g++ {
							wIH := gates[g][0:ihSize]
							for h := 0; h < hiddenSize; h++ {
								wIdx := h*inSize + in
								if wIdx < len(wIH) {
									w := wIH[wIdx]
									importance += w * float32(currentTarget.Data[s*hiddenSize+h])
									if w < 0 {
										totalWeight -= w
									} else {
										totalWeight += w
									}
								}
							}
						}
						if totalWeight > 0.01 {
							estimatedTarget.Data[s*inSize+in] = T(importance / totalWeight)
						}
					}
				}
			}
		case LayerMultiHeadAttention:
			// Approximation: Input should move towards a weighted combination of targets
			// based on Output projection weights (most direct connection).
			dModel := l.OutputHeight
			if l.WeightStore != nil && len(l.WeightStore.Master) > 0 {
				weights := l.WeightStore.Master
				seqLen := len(input.Data) / dModel
				for s := 0; s < seqLen; s++ {
					for in := 0; in < dModel; in++ {
						importance := float32(0)
						totalWeight := float32(0)
						for out := 0; out < dModel && out < len(currentTarget.Data)/seqLen; out++ {
							// Output weights: [dModel, dModel]
							wIdx := in*dModel + out
							if wIdx < len(weights) {
								w := weights[wIdx]
								importance += w * float32(currentTarget.Data[s*dModel+out])
								if w < 0 {
									totalWeight -= w
								} else {
									totalWeight += w
								}
							}
						}
						if totalWeight > 0.01 {
							estimatedTarget.Data[s*dModel+in] = T(importance / totalWeight)
						}
					}
				}
			}
		case LayerSwiGLU:
			// Approximation: Propagate target through the Down projection matrix
			inSize := l.InputHeight
			intermediateSize := l.OutputHeight
			if l.WeightStore != nil && len(l.WeightStore.Master) > 0 {
				weights := l.WeightStore.Master
				// In SwiGLU, Down weights start after Gate and Up weights
				// Gate (in * int), Up (in * int), Down (int * in)
				downWStart := 2 * inSize * intermediateSize

				seqLen := len(input.Data) / inSize
				for s := 0; s < seqLen; s++ {
					for in := 0; in < intermediateSize; in++ { // Propagate back to preAct
						importance := float32(0)
						totalWeight := float32(0)
						for out := 0; out < inSize && out < len(currentTarget.Data)/seqLen; out++ {
							wIdx := downWStart + in*inSize + out
							if wIdx < len(weights) {
								w := weights[wIdx]
								importance += w * float32(currentTarget.Data[s*inSize+out])
								if w < 0 {
									totalWeight -= w
								} else {
									totalWeight += w
								}
							}
						}
						// Simplification: Not full SwiGLU inverse, just mapping down_proj inverse
						if totalWeight > 0.01 {
							// For actual SwiGLU in TargetProp, we'd need a more complex mapping
							// matching the intermediate size. For now, pass activation.
						}
					}
					// For SwiGLU layer input itself, it's easier to pass input as target
					// and rely on Gap Updates directly.
				}
			}
			copy(estimatedTarget.Data, input.Data)
		default:
			// Fallback: Just pass the activation as target if no explicit rule
			// (e.g., Norm layers, Activations, Softmax)
			copy(estimatedTarget.Data, input.Data)
		}

		// Find where this input came from in the mesh
		sourceIdx := -1 // -1 means global system input
		if l.IsRemoteLink {
			sourceIdx = n.GetIndex(l.TargetZ, l.TargetY, l.TargetX, l.TargetL)
		} else if i > 0 {
			sourceIdx = i - 1
		}

		// Store the estimated target as the target for the source layer's output
		s.BackwardTargets[sourceIdx+1] = estimatedTarget
	}
}

// CalculateLinkBudgets diagnostic: Measures how much informaton is preserved (Cosine Similarity).
func (s *TargetPropState[T]) CalculateLinkBudgets() {
	for i := 0; i < s.TotalLayers; i++ {
		fwd := s.ForwardActs[i+1]
		bwd := s.BackwardTargets[i+1]
		if fwd == nil || bwd == nil {
			continue
		}

		dot, fMag, bMag := 0.0, 0.0, 0.0
		gap := 0.0
		for j := range fwd.Data {
			fv := float64(fwd.Data[j])
			bv := float64(bwd.Data[j])
			dot += fv * bv
			fMag += fv * fv
			bMag += bv * bv
			diff := fv - bv
			gap += diff * diff
		}

		if fMag > 0 && bMag > 0 {
			cosine := dot / (math.Sqrt(fMag) * math.Sqrt(bMag))
			s.LinkBudgets[i] = float32((cosine + 1) / 2)
		}
		s.Gaps[i] = float32(math.Sqrt(gap / float64(len(fwd.Data))))
	}
}

// ApplyTargetPropGaps assigns weight updates based on configuration.
func ApplyTargetPropGaps[T Numeric](n *VolumetricNetwork, s *TargetPropState[T], lr float32) {
	if s.Config.UseChainRule {
		applyChainRuleGradients(n, s, lr)
	} else {
		applyTargetPropGapsTargetProp(n, s, lr)
	}
}

// applyChainRuleGradients recursively updates the model's weight stores using the captured gradients.
func applyChainRuleGradients[T Numeric](n *VolumetricNetwork, s *TargetPropState[T], lr float32) {
	for i := 0; i < s.TotalLayers; i++ {
		// 1. MESH FIDELITY GATING
		budget := s.LinkBudgets[i]
		if budget < 0.2 {
			continue // Prevent updating dead layers
		}

		// 2. DYNAMIC RATE SCALING
		layerRate := lr * (0.5 + budget*0.5)

		l := &n.Layers[i]
		grad := s.Gradients[i+1] // Grad of the output of this layer
		if grad == nil {
			continue
		}

		// 3. APPLY SCALED RATE
		ApplyRecursiveGradients(l, grad, layerRate)
	}
}

// applyTargetPropGapsTargetProp updates weights using the Local Error Signal (Target - Actual).
func applyTargetPropGapsTargetProp[T Numeric](n *VolumetricNetwork, s *TargetPropState[T], lr float32) {
	for i := 0; i < s.TotalLayers; i++ {
		// 1. LINK BUDGET GATING
		budget := s.LinkBudgets[i]

		// If the signal is completely destroyed here, don't update!
		// (You might want to add IgnoreThreshold to your TargetPropConfig)
		if budget < 0.2 {
			continue
		}

		// 2. DYNAMIC RATE SCALING
		// Good signal = higher learning rate. Bad signal = cautious learning rate.
		layerRate := lr * (0.5 + budget*0.5) // Adjust based on your preferred scaling

		l := &n.Layers[i]
		input := s.ForwardActs[i]
		actual := s.ForwardActs[i+1]
		target := s.BackwardTargets[i+1]

		if input == nil || actual == nil || target == nil || l.WeightStore == nil {
			continue
		}

		outSize := l.OutputHeight
		inSize := l.InputHeight
		gap := make([]float32, outSize)
		for j := 0; j < outSize && j < len(actual.Data) && j < len(target.Data); j++ {
			gap[j] = float32(target.Data[j]) - float32(actual.Data[j])
		}

		switch l.Type {
		case LayerDense:
			weights := l.WeightStore.Master
			for out := 0; out < outSize && out < len(gap); out++ {
				for in := 0; in < inSize && in < len(input.Data); in++ {
					wIdx := in*outSize + out
					if wIdx < len(weights) {
						// 3. USE THE LAYER RATE HERE
						delta := layerRate * float32(input.Data[in]) * gap[out]
						weights[wIdx] += delta
					}
				}
			}
		case LayerRNN:
			weights := l.WeightStore.Master
			ihSize := outSize * inSize
			hhSize := outSize * outSize
			wIH, wHH, bH := weights[0:ihSize], weights[ihSize:ihSize+hhSize], weights[ihSize+hhSize:]

			seqLen := len(input.Data) / inSize
			for s := 0; s < seqLen; s++ {
				for out := 0; out < outSize; out++ {
					g := gap[s*outSize+out]
					// Update Bias
					if out < len(bH) {
						bH[out] += layerRate * g
					}
					// Update IH weights
					for in := 0; in < inSize; in++ {
						wIdx := out*inSize + in
						if wIdx < len(wIH) {
							wIH[wIdx] += layerRate * float32(input.Data[s*inSize+in]) * g
						}
					}
					// Update HH weights (Hebbian correlation between gap and previous state)
					if s > 0 {
						for hp := 0; hp < outSize; hp++ {
							wIdx := out*outSize + hp
							if wIdx < len(wHH) {
								wHH[wIdx] += layerRate * float32(actual.Data[(s-1)*outSize+hp]) * g * 0.5
							}
						}
					}
				}
			}
		case LayerLSTM:
			weights := l.WeightStore.Master
			ihSize := outSize * inSize
			hhSize := outSize * outSize
			bSize := outSize
			gateSize := ihSize + hhSize + bSize

			seqLen := len(input.Data) / inSize
			for s := 0; s < seqLen; s++ {
				for g := 0; g < 4; g++ {
					gateOffset := g * gateSize
					wIH := weights[gateOffset : gateOffset+ihSize]
					wHH := weights[gateOffset+ihSize : gateOffset+ihSize+hhSize]
					bH := weights[gateOffset+ihSize+hhSize : gateOffset+gateSize]

					for out := 0; out < outSize; out++ {
						localGap := gap[s*outSize+out]
						// Update Bias
						if out < len(bH) {
							bH[out] += layerRate * localGap * 0.25
						}
						// Update IH
						for in := 0; in < inSize; in++ {
							wIdx := out*inSize + in
							if wIdx < len(wIH) {
								wIH[wIdx] += layerRate * float32(input.Data[s*inSize+in]) * localGap * 0.25
							}
						}
						// Update HH
						if s > 0 {
							for hp := 0; hp < outSize; hp++ {
								wIdx := out*outSize + hp
								if wIdx < len(wHH) {
									wHH[wIdx] += layerRate * float32(actual.Data[(s-1)*outSize+hp]) * localGap * 0.1
								}
							}
						}
					}
				}
			}
		case LayerMultiHeadAttention:
			// Simplified TargetProp Update for output projection
			weights := l.WeightStore.Master
			seqLen := len(input.Data) / inSize // inSize here acts as dModel
			for s := 0; s < seqLen; s++ {
				for out := 0; out < outSize && out < len(gap)/seqLen; out++ {
					g := gap[s*outSize+out]
					for in := 0; in < inSize && in < len(input.Data)/seqLen; in++ {
						wIdx := in*outSize + out
						if wIdx < len(weights) { // Assume OutputWeight is at start
							delta := layerRate * float32(input.Data[s*inSize+in]) * g
							weights[wIdx] += delta * 0.5
						}
					}
				}
			}
		case LayerSwiGLU:
			// Approximate localized learning for SwiGLU gating
			weights := l.WeightStore.Master
			intermediateSize := l.OutputHeight
			downWStart := 2 * inSize * intermediateSize
			seqLen := len(input.Data) / inSize
			for s := 0; s < seqLen; s++ {
				for out := 0; out < inSize && out < len(gap)/seqLen; out++ {
					g := gap[s*inSize+out]
					for in := 0; in < intermediateSize; in++ { // Propagated from preAct
						wIdx := downWStart + in*inSize + out
						if wIdx < len(weights) {
							// For true localized learning, we'd need preAct
							// Simplification: just direct weight gap adjustment based on in/out parity
							weights[wIdx] += layerRate * g * 0.1
						}
					}
				}
			}
		case LayerLayerNorm, LayerRMSNorm:
			// No weights, but shift Gamma based on GAP
			if l.WeightStore != nil && len(l.WeightStore.Master) >= outSize {
				gamma := l.WeightStore.Master[:outSize]
				for out := 0; out < outSize && out < len(gap); out++ {
					// Gamma controls scaling. If gap is positive, we need more output, increase gamma.
					gamma[out] += layerRate * gap[out] * 0.01
				}
			}
		}
	}
}
