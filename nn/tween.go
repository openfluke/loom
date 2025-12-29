package nn

import (
	"fmt"
	"math"
	"math/rand"
)

// =============================================================================
// Generic TweenState Implementation
// =============================================================================

// GenericTweenState holds bidirectional analysis state for any numeric type.
type GenericTweenState[T Numeric] struct {
	// Forward pass: what each layer ACTUALLY produces (top-down)
	ForwardActs []*Tensor[T]

	// Backward pass: what each layer SHOULD produce (bottom-up from expected)
	BackwardTargets []*Tensor[T]

	// Link budget per layer: how much information is preserved (0-1)
	LinkBudgets []float32

	// Gap at each layer: magnitude of difference between forward and backward
	Gaps []float32

	// Momentum for stable updates
	WeightVel []*Tensor[T]
	BiasVel   []*Tensor[T]

	// Config holds all tunable parameters
	Config *TweenConfig

	TotalLayers int
	TweenSteps  int
}

// NewGenericTweenState creates a generic tween state.
func NewGenericTweenState[T Numeric](totalLayers int, config *TweenConfig) *GenericTweenState[T] {
	if config == nil {
		config = DefaultTweenConfig(totalLayers)
	}
	return &GenericTweenState[T]{
		ForwardActs:     make([]*Tensor[T], totalLayers+1),
		BackwardTargets: make([]*Tensor[T], totalLayers+1),
		LinkBudgets:     make([]float32, totalLayers),
		Gaps:            make([]float32, totalLayers),
		WeightVel:       make([]*Tensor[T], totalLayers),
		BiasVel:         make([]*Tensor[T], totalLayers),
		Config:          config,
		TotalLayers:     totalLayers,
		TweenSteps:      0,
	}
}

// SetForwardActivation sets the forward activation for a layer.
func (ts *GenericTweenState[T]) SetForwardActivation(layerIdx int, activation *Tensor[T]) {
	if layerIdx >= 0 && layerIdx < len(ts.ForwardActs) {
		ts.ForwardActs[layerIdx] = activation.Clone()
	}
}

// GetGap returns the gap at a specific layer.
func (ts *GenericTweenState[T]) GetGap(layerIdx int) float32 {
	if layerIdx >= 0 && layerIdx < len(ts.Gaps) {
		return ts.Gaps[layerIdx]
	}
	return 0
}

// ComputeGaps calculates the gap between forward and backward targets for each layer.
func (ts *GenericTweenState[T]) ComputeGaps() {
	for i := 0; i < ts.TotalLayers; i++ {
		if ts.ForwardActs[i] == nil || ts.BackwardTargets[i] == nil {
			continue
		}
		fwd := ts.ForwardActs[i]
		tgt := ts.BackwardTargets[i]
		minLen := len(fwd.Data)
		if len(tgt.Data) < minLen {
			minLen = len(tgt.Data)
		}

		sumSq := 0.0
		for j := 0; j < minLen; j++ {
			diff := float64(fwd.Data[j]) - float64(tgt.Data[j])
			sumSq += diff * diff
		}
		ts.Gaps[i] = float32(math.Sqrt(sumSq / float64(minLen+1)))
	}
}

// =============================================================================
// Original float32 Implementation
// =============================================================================


// NeuralTween - True Bidirectional Approach
//
// Concept:
// - FORWARD (from TOP): Untrained network produces activations at each layer
// - BACKWARD (from BOTTOM): Expected output propagates upward, estimating targets
// - MEET IN MIDDLE: Compare what IS vs what SHOULD BE at each layer
// - LINK BUDGET: Information is LOST going through layers (attenuation)
// - TWEEN: Adjust weights so the RIGHT information survives
//
// Key insight: We DON'T compute gradients. We directly see the GAP at each
// layer between forward (actual) and backward (target) and tween toward it.

// TweenConfig holds all tunable parameters for NeuralTween
type TweenConfig struct {
	// === TUNABLE LEARNING ===
	FrontierEnabled   bool    // Default: true
	FrontierMin       int     // Minimum layer index to oscillate to
	FrontierThreshold float32 // Default: 0.55
	FrontierNoise     float32 // Default: 0.0

	IgnoreThreshold float32 // Default: 0.2
	DenseRate       float32 // Default: 1.0
	RNNRate         float32 // Default: 0.5
	LSTMRate        float32 // Default: 0.5
	AttentionRate   float32 // Default: 0.2
	NormRate        float32 // Default: 0.1
	SwiGLURate      float32 // Default: 0.2
	Conv2DRate      float32 // Default: 0.1

	// === MOMENTUM & UPDATE SCALING ===
	Momentum             float32 // Default: 0.9
	BiasRateMultiplier   float32 // Default: 0.1
	WeightRateMultiplier float32 // Default: 0.01

	// === BACKWARD PASS CLAMPING ===
	TanhClampMin    float32 // Default: -0.95
	TanhClampMax    float32 // Default: 0.95
	SigmoidClampMin float32 // Default: 0.05
	SigmoidClampMax float32 // Default: 0.95
	ReLUClampMax    float32 // Default: 10.0

	// === TRAINING BEHAVIOR ===
	EarlyStopThreshold float64 // Default: 95.0
	EvalFrequency      int     // Default: 5
	LinkBudgetScale    float32 // Default: 0.5

	// === DYNAMIC LAYER PRUNING ===
	PruneEnabled   bool    // Default: false
	PruneThreshold float32 // Default: 0.1
	PrunePatience  int     // Default: 10

	// === BATCH TRAINING ===
	BatchSize int // Default: 1

	// === CHAIN RULE SUPPORT ===
	UseChainRule     bool    // Default: true
	DepthScaleFactor float32 // Default: 1.2

	// === NEW CONFIGURABLE CONSTANTS ===
	GradientScale        float32 // Default: 0.1 (Base scale for gradients)
	TotalWeightThreshold float32 // Default: 0.01 (Threshold for weight importance)
	ReLUSlope            float32 // Default: 1.1 (Slope for positive activation)
	LeakyReLUSlope       float32 // Default: 0.1 (Slope for negative activation)
	DerivativeEpsilon    float32 // Default: 0.01 (Stability epsilon)
	LSTMGateScale        float32 // Default: 0.25 (Scaling for LSTM gates)
	AttentionRoutingRate float32 // Default: 0.2 (Scaling for attention routing)
	AttentionBiasRate    float32 // Default: 0.05 (Scaling for attention bias)
	NormBetaRate         float32 // Default: 0.1 (Scaling for Norm Beta)
	NormGammaRate        float32 // Default: 0.01 (Scaling for Norm Gamma)

	// === EXPLOSION DETECTION ===
	ExplosionDetection bool // Default: false (must be enabled explicitly)
}

// DefaultTweenConfig returns the standard configuration
func DefaultTweenConfig(totalLayers int) *TweenConfig {
	return &TweenConfig{
		FrontierEnabled:      true,
		FrontierMin:          totalLayers - 1,
		FrontierThreshold:    0.55,
		FrontierNoise:        0.0,
		IgnoreThreshold:      0.2,
		DenseRate:            1.0,
		RNNRate:              0.5,
		LSTMRate:             0.5,
		AttentionRate:        0.2,
		NormRate:             0.1,
		SwiGLURate:           0.2,
		Conv2DRate:           0.1,
		Momentum:             0.9,
		BiasRateMultiplier:   0.1,
		WeightRateMultiplier: 0.01,
		TanhClampMin:         -0.95,
		TanhClampMax:         0.95,
		SigmoidClampMin:      0.05,
		SigmoidClampMax:      0.95,
		ReLUClampMax:         10.0,
		EarlyStopThreshold:   95.0,
		EvalFrequency:        5,
		LinkBudgetScale:      0.5,
		PruneEnabled:         false,
		PruneThreshold:       0.1,
		PrunePatience:        10,
		BatchSize:            1,
		UseChainRule:         true,
		DepthScaleFactor:     1.2,
		GradientScale:        0.1,
		TotalWeightThreshold: 0.01,
		ReLUSlope:            1.1,
		LeakyReLUSlope:       0.1,
		DerivativeEpsilon:    0.01,
		LSTMGateScale:        0.25,
		AttentionRoutingRate: 0.2,
		AttentionBiasRate:    0.05,
		NormBetaRate:         0.1,
		NormGammaRate:        0.01,
		ExplosionDetection:   false, // Disabled by default
	}
}

// TweenState holds bidirectional analysis state
type TweenState struct {
	// Forward pass: what each layer ACTUALLY produces (top-down)
	ForwardActs [][]float32

	// Backward pass: what each layer SHOULD produce (bottom-up from expected)
	BackwardTargets [][]float32

	// Link budget per layer: how much information is preserved (0-1)
	// Low budget = high attenuation = need more careful tweening
	LinkBudgets []float32

	// Gap at each layer: magnitude of difference between forward and backward
	Gaps []float32

	// Momentum for stable updates
	WeightVel [][]float32
	BiasVel   [][]float32

	// Best state tracking
	BestScore   float64
	BestWeights [][][]float32
	BestBiases  [][][]float32

	// Config holds all tunable parameters
	Config *TweenConfig

	TotalLayers int
	TweenSteps  int
	LossHistory []float32

	// Tracks consecutive epochs a layer has been dead for Pruning
	DeadEpochs []int

	// === BATCH TRAINING ===
	BatchGaps  [][]float32 // Accumulated gaps per layer [layer][output]
	BatchCount int         // Current samples in batch

	// === VISUALIZATION & DEBUGGING ===
	Verbose bool // If true, print training progress to console

	// Link budget history: [epoch][layer] = budget value (for heatmap visualization)
	LinkBudgetHistory [][]float32

	// Gap history: [epoch][layer] = gap value (for tracking convergence per layer)
	GapHistory [][]float32

	// Depth barrier: cumulative signal preservation from input to each layer
	// DepthBarrier[i] = product of LinkBudgets[0..i], shows how much original info survives
	DepthBarrier []float32

	// Depth barrier history: [epoch] = overall depth barrier (product of all budgets)
	DepthBarrierHistory []float32

	// Per-epoch metrics for plotting
	EpochMetrics []TweenEpochMetrics

	// === CHAIN RULE SUPPORT ===
	ChainGradients [][]float32 // Gradient at each layer, computed via chain rule

	// === GRADIENT EXPLOSION DETECTION ===
	PrevAvgGap     float32 // Previous epoch's average gap
	GapGrowthRate  float32 // Rate of gap growth (current/previous)
	ExplosionCount int     // Consecutive epochs with explosion detected
	AdaptiveRate   float32 // Current adaptive learning rate multiplier (0-1)
	BaselineGap    float32 // Baseline gap from first few epochs
	GapSamples     int     // Number of samples for baseline calculation
}

// TweenEpochMetrics captures detailed per-epoch information for visualization
type TweenEpochMetrics struct {
	Epoch           int
	AvgLoss         float32
	Score           float64
	AvgLinkBudget   float32
	MinLinkBudget   float32
	MaxLinkBudget   float32
	AvgGap          float32
	MaxGap          float32
	DepthBarrier    float32 // Overall info preservation (product of all budgets)
	BottleneckLayer int     // Layer with lowest link budget
}

// NewTweenState creates tween state with tunable defaults.
// Pass nil for config to use defaults.
func NewTweenState(n *Network, config *TweenConfig) *TweenState {
	total := n.TotalLayers()
	if config == nil {
		config = DefaultTweenConfig(total)
	} else {
		// Ensure FrontierMin is valid if user provides a config
		if config.FrontierMin < 0 || config.FrontierMin >= total {
			config.FrontierMin = total - 1
		}
	}

	ts := &TweenState{
		Config:          config,
		ForwardActs:     make([][]float32, total+1),
		BackwardTargets: make([][]float32, total+1),
		LinkBudgets:     make([]float32, total),
		Gaps:            make([]float32, total),
		WeightVel:       make([][]float32, total),
		BiasVel:         make([][]float32, total),
		BestWeights:     make([][][]float32, total),
		BestBiases:      make([][][]float32, total),
		TotalLayers:     total,
		// State
		DeadEpochs: make([]int, total),
		BatchGaps:  make([][]float32, total),
		BatchCount: 0,
		// Visualization
		Verbose:             false,
		LinkBudgetHistory:   make([][]float32, 0),
		GapHistory:          make([][]float32, 0),
		DepthBarrier:        make([]float32, total),
		DepthBarrierHistory: make([]float32, 0),
		EpochMetrics:        make([]TweenEpochMetrics, 0),
		// Chain rule
		ChainGradients: make([][]float32, total+1),
	}

	// Init momentum
	for i := 0; i < total; i++ {
		cfg := ts.getLayerCfg(n, i)
		if cfg != nil && len(cfg.Kernel) > 0 {
			ts.WeightVel[i] = make([]float32, len(cfg.Kernel))
			ts.BiasVel[i] = make([]float32, len(cfg.Bias))
			ts.LinkBudgets[i] = 1.0
		}
	}
	return ts
}

func (ts *TweenState) getLayerCfg(n *Network, i int) *LayerConfig {
	row := i / (n.GridCols * n.LayersPerCell)
	col := (i / n.LayersPerCell) % n.GridCols
	layer := i % n.LayersPerCell
	return n.GetLayer(row, col, layer)
}

// ForwardPass: Push input through untrained network, capture ALL activations
func (ts *TweenState) ForwardPass(n *Network, input []float32) []float32 {
	output, _ := n.ForwardCPU(input)
	acts := n.Activations()
	for i, a := range acts {
		if i < len(ts.ForwardActs) {
			ts.ForwardActs[i] = a
		}
	}
	return output
}

// BackwardPass: From expected output, estimate what each layer SHOULD produce
// This is the "bottom-up" pass - propagating the target upward through layers
func (ts *TweenState) BackwardPass(n *Network, targetClass int, outputSize int) {
	// Start at output: we know exactly what it SHOULD be
	target := make([]float32, outputSize)
	target[targetClass] = 1.0
	ts.BackwardTargets[ts.TotalLayers] = target

	// Propagate upward: for each layer, estimate what input would produce target output
	currentTarget := target

	for i := ts.TotalLayers - 1; i >= 0; i-- {
		cfg := ts.getLayerCfg(n, i)
		if cfg == nil || cfg.Type != LayerDense {
			ts.BackwardTargets[i] = currentTarget
			continue
		}

		// Estimate input that produces currentTarget
		// Simple approach: weighted average based on which outputs need to be high
		inputSize := cfg.InputHeight
		if inputSize <= 0 {
			inputSize = len(ts.ForwardActs[i])
		}

		estimated := make([]float32, inputSize)

		// For each input neuron, calculate its "importance" to the target
		for in := 0; in < inputSize; in++ {
			importance := float32(0)
			totalWeight := float32(0)

			for out := 0; out < len(currentTarget); out++ {
				wIdx := in*cfg.OutputHeight + out
				if wIdx < len(cfg.Kernel) {
					w := cfg.Kernel[wIdx]
					// How much does this input contribute to needed outputs?
					importance += w * currentTarget[out]
					totalWeight += float32(math.Abs(float64(w)))
				}
			}

			// Normalize by total weight magnitude
			if totalWeight > ts.Config.TotalWeightThreshold {
				estimated[in] = importance / float32(totalWeight)
			}

			// Clamp to valid activation range using configurable bounds
			if cfg.Activation == ActivationTanh {
				estimated[in] = clamp(estimated[in], ts.Config.TanhClampMin, ts.Config.TanhClampMax)
			} else if cfg.Activation == ActivationSigmoid {
				estimated[in] = clamp(estimated[in], ts.Config.SigmoidClampMin, ts.Config.SigmoidClampMax)
			} else if cfg.Activation == ActivationScaledReLU || cfg.Activation == ActivationLeakyReLU {
				estimated[in] = clamp(estimated[in], 0, ts.Config.ReLUClampMax)
			}
		}

		ts.BackwardTargets[i] = estimated
		currentTarget = estimated
	}
}

// BackwardPassChainRule: Proper chain rule gradient propagation
// Unlike BackwardPass which uses heuristics, this properly applies:
// 1. Output error gradient (target - actual)
// 2. Activation function derivatives at each layer
// 3. Transpose weight multiplication to propagate gradients
// 4. Depth scaling to combat vanishing gradients
func (ts *TweenState) BackwardPassChainRule(n *Network, targetClass int, outputSize int) {
	// Start at output: compute error gradient (target - actual)
	actual := ts.ForwardActs[ts.TotalLayers]
	if len(actual) == 0 {
		return
	}

	// Create output gradient: target - actual (for MSE derivative)
	outputGrad := make([]float32, outputSize)
	for i := 0; i < outputSize; i++ {
		if i == targetClass {
			outputGrad[i] = 1.0 - actual[i]
		} else if i < len(actual) {
			outputGrad[i] = 0.0 - actual[i]
		}
	}

	// Store the output gradient
	ts.ChainGradients[ts.TotalLayers] = outputGrad
	ts.BackwardTargets[ts.TotalLayers] = make([]float32, outputSize)
	for i := 0; i < outputSize; i++ {
		if i == targetClass {
			ts.BackwardTargets[ts.TotalLayers][i] = 1.0
		}
	}

	// Propagate gradients backward using chain rule
	currentGrad := outputGrad

	for i := ts.TotalLayers - 1; i >= 0; i-- {
		cfg := ts.getLayerCfg(n, i)
		if cfg == nil {
			ts.ChainGradients[i] = currentGrad
			ts.BackwardTargets[i] = ts.ForwardActs[i]
			continue
		}

		// Get pre-activation values if available (for derivative computation)
		// We reconstruct from the forward activations
		layerOutput := ts.ForwardActs[i+1]
		layerInput := ts.ForwardActs[i]

		if len(layerOutput) == 0 || len(layerInput) == 0 {
			ts.ChainGradients[i] = currentGrad
			ts.BackwardTargets[i] = layerInput
			continue
		}

		// Compute depth scale: deeper layers (closer to input) get higher scale
		// Use safe scaling to prevent explosion in very deep networks
		depthFromOutput := float32(ts.TotalLayers - 1 - i)
		depthScale := safeDepthScale(depthFromOutput, ts.Config.DepthScaleFactor, 100.0)

		switch cfg.Type {
		case LayerDense:
			currentGrad = ts.chainRuleBackwardDense(cfg, layerInput, layerOutput, currentGrad, depthScale, i)
		case LayerConv2D:
			currentGrad = ts.chainRuleBackwardConv2D(cfg, layerInput, layerOutput, currentGrad, depthScale)
		case LayerMultiHeadAttention:
			currentGrad = ts.chainRuleBackwardAttention(cfg, layerInput, layerOutput, currentGrad, depthScale)
		case LayerRNN:
			currentGrad = ts.chainRuleBackwardRNN(cfg, layerInput, layerOutput, currentGrad, depthScale)
		case LayerLSTM:
			currentGrad = ts.chainRuleBackwardLSTM(cfg, layerInput, layerOutput, currentGrad, depthScale)
		case LayerNorm, LayerRMSNorm:
			currentGrad = ts.chainRuleBackwardNorm(cfg, layerInput, layerOutput, currentGrad, depthScale)
		case LayerSwiGLU:
			currentGrad = ts.chainRuleBackwardSwiGLU(cfg, layerInput, layerOutput, currentGrad, depthScale)
		default:
			// Pass gradient through for non-trainable layers (Softmax, Residual, etc.)
			if len(currentGrad) != len(layerInput) {
				// Dimension mismatch - create gradient matching input size
				newGrad := make([]float32, len(layerInput))
				minLen := len(currentGrad)
				if len(layerInput) < minLen {
					minLen = len(layerInput)
				}
				for j := 0; j < minLen; j++ {
					newGrad[j] = currentGrad[j]
				}
				currentGrad = newGrad
			}
		}

		// Sanitize gradients: replace NaN/Inf with 0, clip to reasonable range
		const maxGradMagnitude float32 = 10.0
		for j := range currentGrad {
			currentGrad[j] = safeGrad(currentGrad[j], maxGradMagnitude)
		}

		// Store gradient for this layer
		ts.ChainGradients[i] = currentGrad

		// Compute backward target: what the layer input SHOULD be to reduce error
		// This is the forward activation PLUS the scaled gradient (direction to improve)
		backwardTarget := make([]float32, len(layerInput))
		for j := 0; j < len(layerInput); j++ {
			if j < len(currentGrad) {
				// Target = current + gradient direction (scaled for stability)
				backwardTarget[j] = layerInput[j] + currentGrad[j]*ts.Config.GradientScale
			} else {
				backwardTarget[j] = layerInput[j]
			}
		}

		// Clamp to valid activation range and ensure no NaN/Inf
		for j := 0; j < len(backwardTarget); j++ {
			// First sanitize NaN/Inf
			backwardTarget[j] = safeValue(backwardTarget[j], layerInput[j])
			// Then clamp to activation range
			if cfg.Activation == ActivationTanh {
				backwardTarget[j] = clamp(backwardTarget[j], ts.Config.TanhClampMin, ts.Config.TanhClampMax)
			} else if cfg.Activation == ActivationSigmoid {
				backwardTarget[j] = clamp(backwardTarget[j], ts.Config.SigmoidClampMin, ts.Config.SigmoidClampMax)
			} else if cfg.Activation == ActivationScaledReLU || cfg.Activation == ActivationLeakyReLU {
				backwardTarget[j] = clamp(backwardTarget[j], -ts.Config.ReLUClampMax, ts.Config.ReLUClampMax)
			}
		}

		ts.BackwardTargets[i] = backwardTarget
	}
}

// chainRuleBackwardDense: Proper chain rule for Dense layers
// Returns gradient w.r.t. input: grad_input = W^T * (grad_output ⊙ activation'(pre_activation))
func (ts *TweenState) chainRuleBackwardDense(cfg *LayerConfig, input, output, gradOutput []float32, depthScale float32, layerIdx int) []float32 {
	inputSize := cfg.InputHeight
	if inputSize <= 0 {
		inputSize = len(input)
	}
	outputSize := cfg.OutputHeight
	if outputSize <= 0 {
		outputSize = len(output)
	}

	// Maximum gradient magnitude to prevent explosion
	const maxLocalGrad float32 = 10.0

	// Apply activation derivative to output gradient with clipping
	localGrad := make([]float32, len(gradOutput))
	for j := 0; j < len(gradOutput) && j < len(output); j++ {
		actDeriv := ts.activateDerivativeFromOutput(output[j], cfg.Activation)
		localGrad[j] = safeGrad(gradOutput[j]*actDeriv*depthScale, maxLocalGrad)
	}

	// Propagate gradient to input: grad_input = W^T * local_grad
	// Use sqrt-based normalization to control magnitude without killing signal
	gradInput := make([]float32, inputSize)

	// Softer normalization: use sqrt(outputSize) instead of outputSize
	// This preserves more gradient signal while still controlling growth
	normFactor := float32(1.0)
	if outputSize > 1 {
		normFactor = 1.0 / float32(math.Sqrt(float64(outputSize)))
	}

	for in := 0; in < inputSize; in++ {
		gradSum := float32(0)
		for out := 0; out < outputSize && out < len(localGrad); out++ {
			wIdx := in*outputSize + out
			if wIdx < len(cfg.Kernel) {
				gradSum += cfg.Kernel[wIdx] * localGrad[out]
			}
		}
		// Apply sqrt-normalization and clip the final gradient
		gradInput[in] = safeGrad(gradSum*normFactor, maxLocalGrad)
	}

	return gradInput
}

// chainRuleBackwardConv2D: Chain rule for Conv2D layers
func (ts *TweenState) chainRuleBackwardConv2D(cfg *LayerConfig, input, output, gradOutput []float32, depthScale float32) []float32 {
	// Apply activation derivative
	localGrad := make([]float32, len(gradOutput))
	for j := 0; j < len(gradOutput) && j < len(output); j++ {
		actDeriv := ts.activateDerivativeFromOutput(output[j], cfg.Activation)
		localGrad[j] = gradOutput[j] * actDeriv * depthScale
	}

	// For Conv2D, we need to do "transposed convolution"
	// Simplified: average gradients weighted by filter importance
	numFilters := cfg.Filters
	if numFilters == 0 {
		numFilters = len(cfg.Bias)
	}
	if numFilters == 0 {
		return localGrad
	}

	// Compute average gradient per filter
	gradInput := make([]float32, len(input))
	gapsPerFilter := len(localGrad) / numFilters
	if gapsPerFilter == 0 {
		gapsPerFilter = 1
	}

	// Distribute gradients back to input positions
	kernelPerFilter := len(cfg.Kernel) / numFilters
	inChannels := cfg.InputChannels
	if inChannels == 0 {
		inChannels = 1
	}

	for f := 0; f < numFilters; f++ {
		// Average gradient for this filter
		filterGrad := float32(0)
		for j := f * gapsPerFilter; j < (f+1)*gapsPerFilter && j < len(localGrad); j++ {
			filterGrad += localGrad[j]
		}
		if gapsPerFilter > 0 {
			filterGrad /= float32(gapsPerFilter)
		}

		// Distribute to input via kernel weights
		for k := 0; k < kernelPerFilter; k++ {
			kIdx := f*kernelPerFilter + k
			if kIdx < len(cfg.Kernel) {
				// Map kernel position back to input position
				inIdx := k % len(input)
				gradInput[inIdx] += cfg.Kernel[kIdx] * filterGrad
			}
		}
	}

	return gradInput
}

// chainRuleBackwardAttention: Patched to backpropagate through Q/K
func (ts *TweenState) chainRuleBackwardAttention(cfg *LayerConfig, input, output, gradOutput []float32, depthScale float32) []float32 {
	// 1. Initial Gradient Scaling
	localGrad := make([]float32, len(gradOutput))
	for j := range gradOutput {
		localGrad[j] = gradOutput[j] * depthScale
	}

	dModel := cfg.DModel
	if dModel == 0 {
		dModel = len(output)
	}

	// 2. Backprop through Output Projection
	// gradAttn = OutputWeight^T * localGrad
	gradAttn := make([]float32, dModel)
	for i := 0; i < dModel; i++ {
		sum := float32(0)
		for j := 0; j < len(localGrad); j++ {
			wIdx := i*len(localGrad) + j
			if wIdx < len(cfg.OutputWeight) {
				sum += cfg.OutputWeight[wIdx] * localGrad[j]
			}
		}
		gradAttn[i] = sum
	}

	// 3. Backprop through V, Q, and K
	// Previously, we only did V. Now we sum contributions from all three.
	gradInput := make([]float32, len(input))

	// Contribution from V (Content path) - Strongest signal
	for i := 0; i < len(input); i++ {
		sum := float32(0)
		for j := 0; j < len(gradAttn); j++ {
			wIdx := i*len(gradAttn) + j
			if wIdx < len(cfg.VWeights) {
				sum += cfg.VWeights[wIdx] * gradAttn[j]
			}
		}
		gradInput[i] += sum
	}

	// Contribution from Q/K (Routing path) - Weaker, but essential for deep learning
	// Heuristic approximation: The routing error flows back proportional to
	// how much the Q/K weights were adjusted in the update step.
	// This avoids the expensive N^2 softmax derivative but preserves signal flow.
	routingScale := float32(0.3) // Routing carries less gradient info than content

	for i := 0; i < len(input); i++ {
		sum := float32(0)
		for j := 0; j < len(gradAttn); j++ {
			qIdx := i*len(gradAttn) + j
			if qIdx < len(cfg.QWeights) {
				// If Q/K are misaligned, input should change to fix it
				sum += (cfg.QWeights[qIdx] + cfg.KWeights[qIdx]) * gradAttn[j]
			}
		}
		gradInput[i] += sum * routingScale
	}

	return gradInput
}

// chainRuleBackwardRNN: Chain rule for RNN layers
func (ts *TweenState) chainRuleBackwardRNN(cfg *LayerConfig, input, output, gradOutput []float32, depthScale float32) []float32 {
	localGrad := make([]float32, len(gradOutput))
	for j := 0; j < len(gradOutput) && j < len(output); j++ {
		actDeriv := ts.activateDerivativeFromOutput(output[j], ActivationTanh) // RNN typically uses tanh
		localGrad[j] = gradOutput[j] * actDeriv * depthScale
	}

	// Backprop through input-to-hidden weights
	gradInput := make([]float32, len(input))
	hiddenSize := cfg.HiddenSize
	inputSize := cfg.RNNInputSize
	if inputSize == 0 {
		inputSize = len(input)
	}

	for i := 0; i < inputSize && i < len(input); i++ {
		sum := float32(0)
		for h := 0; h < hiddenSize && h < len(localGrad); h++ {
			wIdx := h*inputSize + i
			if wIdx < len(cfg.WeightIH) {
				sum += cfg.WeightIH[wIdx] * localGrad[h]
			}
		}
		gradInput[i] = sum
	}

	return gradInput
}

// chainRuleBackwardLSTM: Chain rule for LSTM layers
func (ts *TweenState) chainRuleBackwardLSTM(cfg *LayerConfig, input, output, gradOutput []float32, depthScale float32) []float32 {
	localGrad := make([]float32, len(gradOutput))
	for j := 0; j < len(gradOutput) && j < len(output); j++ {
		// LSTM output goes through tanh
		actDeriv := ts.activateDerivativeFromOutput(output[j], ActivationTanh)
		localGrad[j] = gradOutput[j] * actDeriv * depthScale
	}

	// Backprop through all 4 gates (simplified - average contribution)
	gradInput := make([]float32, len(input))
	inputSize := cfg.RNNInputSize
	if inputSize == 0 {
		inputSize = len(input)
	}
	hiddenSize := cfg.HiddenSize

	for i := 0; i < inputSize && i < len(input); i++ {
		sum := float32(0)
		for h := 0; h < hiddenSize && h < len(localGrad); h++ {
			wIdx := h*inputSize + i
			// Average contribution from all 4 gates
			if wIdx < len(cfg.WeightIH_i) {
				sum += cfg.WeightIH_i[wIdx] * localGrad[h] * ts.Config.LSTMGateScale
			}
			if wIdx < len(cfg.WeightIH_f) {
				sum += cfg.WeightIH_f[wIdx] * localGrad[h] * ts.Config.LSTMGateScale
			}
			if wIdx < len(cfg.WeightIH_g) {
				sum += cfg.WeightIH_g[wIdx] * localGrad[h] * ts.Config.LSTMGateScale
			}
			if wIdx < len(cfg.WeightIH_o) {
				sum += cfg.WeightIH_o[wIdx] * localGrad[h] * ts.Config.LSTMGateScale
			}
		}
		gradInput[i] = sum
	}

	return gradInput
}

// chainRuleBackwardNorm: Chain rule for LayerNorm/RMSNorm
func (ts *TweenState) chainRuleBackwardNorm(cfg *LayerConfig, input, output, gradOutput []float32, depthScale float32) []float32 {
	// Norm layers are nearly linear, gradient passes through scaled by gamma
	gradInput := make([]float32, len(input))
	for j := 0; j < len(input) && j < len(gradOutput); j++ {
		gamma := float32(1.0)
		if j < len(cfg.Gamma) {
			gamma = cfg.Gamma[j]
		}
		// Gamma applies to gradient as linear scale
		gradInput[j] = gradOutput[j] * gamma * depthScale
	}
	return gradInput
}

// chainRuleBackwardSwiGLU: Chain rule for SwiGLU layers
func (ts *TweenState) chainRuleBackwardSwiGLU(cfg *LayerConfig, input, output, gradOutput []float32, depthScale float32) []float32 {
	localGrad := make([]float32, len(gradOutput))
	for j := 0; j < len(gradOutput); j++ {
		localGrad[j] = gradOutput[j] * depthScale
	}

	// Backprop through down projection
	gradInput := make([]float32, len(input))
	for i := 0; i < len(input); i++ {
		sum := float32(0)
		for j := 0; j < len(localGrad); j++ {
			wIdx := i*len(localGrad) + j
			if wIdx < len(cfg.DownWeights) {
				sum += cfg.DownWeights[wIdx] * localGrad[j]
			}
		}
		gradInput[i] = sum
	}

	return gradInput
}

// activateDerivativeFromOutput computes activation derivative from the output value
// This is more stable than trying to invert the activation
func (ts *TweenState) activateDerivativeFromOutput(output float32, activation ActivationType) float32 {
	switch activation {
	case ActivationScaledReLU:
		if output > 0 {
			return ts.Config.ReLUSlope
		}
		return 0.0
	case ActivationSigmoid:
		// For sigmoid: output = sigmoid(x), derivative = output * (1 - output)
		return output*(1.0-output) + ts.Config.DerivativeEpsilon
	case ActivationTanh:
		// For tanh: output = tanh(x), derivative = 1 - output^2
		return (1.0 - output*output) + ts.Config.DerivativeEpsilon
	case ActivationSoftplus:
		// Softplus derivative is sigmoid
		return 1.0/(1.0+float32(math.Exp(-float64(output)))) + ts.Config.DerivativeEpsilon
	case ActivationLeakyReLU:
		if output >= 0 {
			return 1.0
		}
		return ts.Config.LeakyReLUSlope
	default:
		return 1.0
	}
}

// clipGrad clips a gradient value to prevent saturation/explosion
// This prevents the weight updates from becoming too large
func clipGrad(grad, maxMagnitude float32) float32 {
	if grad > maxMagnitude {
		return maxMagnitude
	}
	if grad < -maxMagnitude {
		return -maxMagnitude
	}
	return grad
}

// isNaN32 checks if a float32 is NaN
func isNaN32(f float32) bool {
	return f != f
}

// isInf32 checks if a float32 is infinite
func isInf32(f float32) bool {
	return f > math.MaxFloat32 || f < -math.MaxFloat32
}

// safeGrad ensures a gradient is finite and clipped
// Returns 0 for NaN/Inf values, otherwise clips to maxMagnitude
func safeGrad(grad, maxMagnitude float32) float32 {
	if isNaN32(grad) || isInf32(grad) {
		return 0
	}
	return clipGrad(grad, maxMagnitude)
}

// safeValue ensures a value is finite, returning fallback if not
func safeValue(v, fallback float32) float32 {
	if isNaN32(v) || isInf32(v) {
		return fallback
	}
	return v
}

// safeDepthScale computes depth scaling with a maximum cap to prevent explosion
// For deep networks (>10 layers), this prevents the exponential growth from causing overflow
func safeDepthScale(depthFromOutput float32, scaleFactor float32, maxScale float32) float32 {
	if depthFromOutput <= 0 {
		return 1.0
	}
	// Cap the depth contribution to prevent exponential explosion
	cappedDepth := depthFromOutput
	if cappedDepth > 10 {
		cappedDepth = 10 + float32(math.Log(float64(depthFromOutput-9))) // Logarithmic growth after depth 10
	}
	scale := float32(math.Pow(float64(scaleFactor), float64(cappedDepth)))
	if scale > maxScale {
		return maxScale
	}
	if isNaN32(scale) || isInf32(scale) {
		return 1.0
	}
	return scale
}

// CalculateLinkBudgets: Measure information preservation at each layer
// High budget = good signal flow, Low budget = high attenuation
func (ts *TweenState) CalculateLinkBudgets() {
	for i := 0; i < ts.TotalLayers; i++ {
		fwd := ts.ForwardActs[i+1]
		bwd := ts.BackwardTargets[i+1]

		if len(fwd) == 0 || len(bwd) == 0 {
			ts.LinkBudgets[i] = 0.5
			ts.Gaps[i] = 0
			continue
		}

		// Calculate alignment (cosine similarity) between forward and backward
		dot, fwdMag, bwdMag := float32(0), float32(0), float32(0)
		gapSum := float32(0)
		minLen := len(fwd)
		if len(bwd) < minLen {
			minLen = len(bwd)
		}

		for j := 0; j < minLen; j++ {
			dot += fwd[j] * bwd[j]
			fwdMag += fwd[j] * fwd[j]
			bwdMag += bwd[j] * bwd[j]
			gap := fwd[j] - bwd[j]
			gapSum += gap * gap
		}

		// Link budget = cosine similarity (how well aligned are forward and target)
		if fwdMag > 0.001 && bwdMag > 0.001 {
			cosine := dot / (float32(math.Sqrt(float64(fwdMag * bwdMag))))
			ts.LinkBudgets[i] = (cosine + 1) / 2 // Map to 0-1
		} else {
			ts.LinkBudgets[i] = 0.5
		}

		ts.Gaps[i] = gapSum / float32(minLen)
	}

	// Calculate depth barrier: cumulative signal preservation
	// DepthBarrier[i] = how much of the original input signal survives to layer i
	cumulative := float32(1.0)
	for i := 0; i < ts.TotalLayers; i++ {
		cumulative *= ts.LinkBudgets[i]
		if i < len(ts.DepthBarrier) {
			ts.DepthBarrier[i] = cumulative
		}
	}
}

// TweenWeights: Adjust weights to close the gap at each layer
// Supports ALL layer types: Dense, Conv2D, Attention, LSTM, LayerNorm, SwiGLU
func (ts *TweenState) TweenWeights(n *Network, rate float32) {
	mom := ts.Config.Momentum

	// Tweening logic
	for i := 0; i < ts.TotalLayers; i++ {
		budget := ts.LinkBudgets[i]

		// DEAD LAYER IGNORING: If budget is too low, don't update!
		// A low budget means the signal is garbage, so updating just adds noise.
		if budget < ts.Config.IgnoreThreshold {
			continue
		}

		cfg := ts.getLayerCfg(n, i)
		if cfg == nil {
			continue
		}

		input := ts.ForwardActs[i]
		actual := ts.ForwardActs[i+1]
		target := ts.BackwardTargets[i+1]

		if len(input) == 0 || len(actual) == 0 || len(target) == 0 {
			continue
		}

		// Scale rate by link budget using configurable scale
		layerRate := rate * (ts.Config.LinkBudgetScale + budget*ts.Config.LinkBudgetScale)

		// Calculate output gap
		minOut := len(actual)
		if len(target) < minOut {
			minOut = len(target)
		}
		outputGaps := make([]float32, minOut)
		for j := 0; j < minOut; j++ {
			outputGaps[j] = target[j] - actual[j]
		}

		// Tween based on layer type - use configurable rates!
		switch cfg.Type {
		case LayerDense:
			ts.tweenDense(cfg, input, outputGaps, layerRate*ts.Config.DenseRate, mom, i)
		case LayerConv2D:
			ts.tweenConv2D(cfg, input, outputGaps, layerRate*ts.Config.Conv2DRate, mom)
		case LayerMultiHeadAttention:
			ts.tweenAttention(cfg, input, outputGaps, layerRate*ts.Config.AttentionRate, mom)
		case LayerRNN:
			ts.tweenRNN(cfg, input, outputGaps, layerRate*ts.Config.RNNRate, mom)
		case LayerLSTM:
			ts.tweenLSTM(cfg, input, outputGaps, layerRate*ts.Config.LSTMRate, mom)
		case LayerNorm, LayerRMSNorm:
			ts.tweenNorm(cfg, outputGaps, layerRate*ts.Config.NormRate)
		case LayerSwiGLU:
			ts.tweenSwiGLU(cfg, input, outputGaps, layerRate*ts.Config.SwiGLURate, mom)
			// LayerSoftmax, LayerResidual, LayerParallel - no trainable weights
		}

		row := i / (n.GridCols * n.LayersPerCell)
		col := (i / n.LayersPerCell) % n.GridCols
		layer := i % n.LayersPerCell
		n.SetLayer(row, col, layer, *cfg)
	}
}

// TweenWeightsChainRule: Use chain rule gradients directly for weight updates
// This is the proper gradient-based approach: dW = input^T * output_gradient
func (ts *TweenState) TweenWeightsChainRule(n *Network, rate float32) {
	mom := ts.Config.Momentum

	for i := 0; i < ts.TotalLayers; i++ {
		cfg := ts.getLayerCfg(n, i)
		if cfg == nil {
			continue
		}

		input := ts.ForwardActs[i]
		output := ts.ForwardActs[i+1]

		// Get the chain gradient for this layer's OUTPUT
		var outputGrad []float32
		if i+1 < len(ts.ChainGradients) && len(ts.ChainGradients[i+1]) > 0 {
			outputGrad = ts.ChainGradients[i+1]
		} else {
			// Fallback to gap-based
			target := ts.BackwardTargets[i+1]
			if len(target) > 0 && len(output) > 0 {
				outputGrad = make([]float32, len(output))
				for j := 0; j < len(output) && j < len(target); j++ {
					outputGrad[j] = target[j] - output[j]
				}
			}
		}

		if len(input) == 0 || len(outputGrad) == 0 {
			continue
		}

		// Compute depth scale: deeper layers (closer to input) get higher scale
		// Use safe scaling to prevent explosion in very deep networks
		depthFromOutput := float32(ts.TotalLayers - 1 - i)
		depthScale := safeDepthScale(depthFromOutput, ts.Config.DepthScaleFactor, 100.0)

		// Apply updates based on layer type
		switch cfg.Type {
		case LayerDense:
			ts.chainRuleUpdateDense(cfg, input, output, outputGrad, rate*ts.Config.DenseRate*depthScale, mom, i)
		case LayerConv2D:
			ts.chainRuleUpdateConv2D(cfg, input, output, outputGrad, rate*ts.Config.Conv2DRate*depthScale, mom)
		case LayerMultiHeadAttention:
			ts.chainRuleUpdateAttention(cfg, input, output, outputGrad, rate*ts.Config.AttentionRate*depthScale, mom)
		case LayerRNN:
			ts.chainRuleUpdateRNN(cfg, input, output, outputGrad, rate*ts.Config.RNNRate*depthScale, mom)
		case LayerLSTM:
			ts.chainRuleUpdateLSTM(cfg, input, output, outputGrad, rate*ts.Config.LSTMRate*depthScale, mom)
		case LayerNorm, LayerRMSNorm:
			ts.chainRuleUpdateNorm(cfg, input, output, outputGrad, rate*ts.Config.NormRate*depthScale)
		case LayerSwiGLU:
			ts.chainRuleUpdateSwiGLU(cfg, input, output, outputGrad, rate*ts.Config.SwiGLURate*depthScale, mom)
		}

		row := i / (n.GridCols * n.LayersPerCell)
		col := (i / n.LayersPerCell) % n.GridCols
		layer := i % n.LayersPerCell
		n.SetLayer(row, col, layer, *cfg)
	}
}

// chainRuleUpdateDense: Apply proper gradient update to Dense layer
// dW[in][out] = input[in] * output_grad[out] (outer product)
// dB[out] = output_grad[out]
func (ts *TweenState) chainRuleUpdateDense(cfg *LayerConfig, input, output, outputGrad []float32, rate, mom float32, layerIdx int) {
	inputSize := cfg.InputHeight
	if inputSize <= 0 {
		inputSize = len(input)
	}
	outputSize := cfg.OutputHeight
	if outputSize <= 0 {
		outputSize = len(output)
	}

	// Apply activation derivative to output gradient
	localGrad := make([]float32, len(outputGrad))
	for j := 0; j < len(outputGrad) && j < len(output); j++ {
		actDeriv := ts.activateDerivativeFromOutput(output[j], cfg.Activation)
		localGrad[j] = outputGrad[j] * actDeriv
	}

	// Update biases: dB = local_grad (with NaN protection)
	for out := 0; out < outputSize && out < len(localGrad) && out < len(cfg.Bias); out++ {
		delta := safeGrad(rate*localGrad[out], 1.0)
		cfg.Bias[out] = safeValue(cfg.Bias[out]+delta, cfg.Bias[out])
	}

	// Update weights: dW[in][out] = input[in] * local_grad[out]
	for in := 0; in < inputSize && in < len(input); in++ {
		for out := 0; out < outputSize && out < len(localGrad); out++ {
			wIdx := in*outputSize + out
			if wIdx >= len(cfg.Kernel) {
				continue
			}

			// Proper gradient: input * output_gradient (with NaN protection)
			delta := safeGrad(rate*input[in]*localGrad[out], 1.0)

			// Apply momentum
			if layerIdx < len(ts.WeightVel) && wIdx < len(ts.WeightVel[layerIdx]) {
				ts.WeightVel[layerIdx][wIdx] = mom*ts.WeightVel[layerIdx][wIdx] + (1-mom)*delta
				newWeight := cfg.Kernel[wIdx] + ts.WeightVel[layerIdx][wIdx]
				cfg.Kernel[wIdx] = safeValue(newWeight, cfg.Kernel[wIdx])
			} else {
				newWeight := cfg.Kernel[wIdx] + delta
				cfg.Kernel[wIdx] = safeValue(newWeight, cfg.Kernel[wIdx])
			}
		}
	}
}

// chainRuleUpdateConv2D: Apply gradient update to Conv2D layer
func (ts *TweenState) chainRuleUpdateConv2D(cfg *LayerConfig, input, output, outputGrad []float32, rate, mom float32) {
	if len(cfg.Kernel) == 0 {
		return
	}

	numFilters := cfg.Filters
	if numFilters == 0 {
		numFilters = len(cfg.Bias)
	}
	if numFilters == 0 {
		return
	}

	// Compute gradient per filter
	gapsPerFilter := len(outputGrad) / numFilters
	if gapsPerFilter == 0 {
		gapsPerFilter = 1
	}

	kernelPerFilter := len(cfg.Kernel) / numFilters
	if kernelPerFilter == 0 {
		return
	}

	for f := 0; f < numFilters; f++ {
		// Sum gradient for this filter
		filterGrad := float32(0)
		for j := f * gapsPerFilter; j < (f+1)*gapsPerFilter && j < len(outputGrad); j++ {
			filterGrad += outputGrad[j]
		}
		filterGrad /= float32(gapsPerFilter)

		// Update kernel weights
		for k := 0; k < kernelPerFilter; k++ {
			kIdx := f*kernelPerFilter + k
			if kIdx < len(cfg.Kernel) {
				inIdx := k % len(input)
				if inIdx < len(input) {
					cfg.Kernel[kIdx] += rate * input[inIdx] * filterGrad
				}
			}
		}

		// Update bias
		if f < len(cfg.Bias) {
			cfg.Bias[f] += rate * filterGrad
		}
	}
}

// chainRuleUpdateAttention: Patched to include "Relevance Routing" gradients
// Instead of using a global average, we now compute specific gradients per head.
// This allows the Chain Rule to learn Q and K routing weights effectively.
func (ts *TweenState) chainRuleUpdateAttention(cfg *LayerConfig, input, output, outputGrad []float32, rate, mom float32) {
	dModel := cfg.DModel
	if dModel == 0 {
		dModel = len(output)
	}
	numHeads := cfg.NumHeads
	if numHeads == 0 {
		numHeads = 2
	}
	headDim := dModel / numHeads

	// Gradient clipping to prevent saturation
	const maxGrad float32 = 0.5

	// 1. Update Output Projection (The Mixer)
	for i := 0; i < dModel && i < len(input); i++ {
		for j := 0; j < len(outputGrad); j++ {
			wIdx := i*len(outputGrad) + j
			if wIdx < len(cfg.OutputWeight) {
				grad := clipGrad(rate*input[i]*outputGrad[j], maxGrad)
				cfg.OutputWeight[wIdx] += grad
			}
		}
	}
	// Update Output Bias
	for j := 0; j < len(outputGrad) && j < len(cfg.OutputBias); j++ {
		cfg.OutputBias[j] += clipGrad(rate*outputGrad[j], maxGrad)
	}

	// 2. Compute "Head Gradients"
	headGrads := make([]float32, dModel)
	for i := 0; i < dModel; i++ {
		sum := float32(0)
		for j := 0; j < len(outputGrad); j++ {
			wIdx := i*len(outputGrad) + j
			if wIdx < len(cfg.OutputWeight) {
				sum += cfg.OutputWeight[wIdx] * outputGrad[j]
			}
		}
		headGrads[i] = sum
	}

	// 3. Update Heads (Q, K, V)
	for h := 0; h < numHeads; h++ {
		headStart := h * headDim
		headEnd := headStart + headDim

		// The gradient signal specific to this head
		thisHeadGrad := headGrads[headStart:headEnd]

		// --- V Update (Content) ---
		vStart := h * headDim * dModel
		for i := vStart; i < vStart+(headDim*dModel); i++ {
			inIdx := (i - vStart) % dModel
			gradIdx := (i - vStart) / dModel
			if inIdx < len(input) && gradIdx < len(thisHeadGrad) {
				grad := clipGrad(rate*input[inIdx]*thisHeadGrad[gradIdx], maxGrad)
				cfg.VWeights[i] += grad
			}
		}

		// --- Q & K Update (Routing) ---
		// Calculate alignment score
		dot := float32(0)
		limit := len(input)
		if limit > headDim {
			limit = headDim
		}

		for k := 0; k < limit && k < len(thisHeadGrad); k++ {
			dot += input[k] * thisHeadGrad[k]
		}

		// Scale factor
		routingRate := clipGrad(dot*rate*0.1, maxGrad)

		qStart := h * headDim * dModel
		kStart := h * headDim * dModel

		// Update Q and K
		for i := 0; i < (headDim * dModel); i++ {
			qIdx := qStart + i
			kIdx := kStart + i
			inIdx := i % dModel

			if qIdx < len(cfg.QWeights) && kIdx < len(cfg.KWeights) && inIdx < len(input) {
				val := input[inIdx] * routingRate
				cfg.QWeights[qIdx] += val
				cfg.KWeights[kIdx] += val
			}
		}
	}

	// Update Biases
	avgGrad := avgSlice(outputGrad)
	tweenBiasSlice(cfg.QBias, avgGrad, rate*0.05)
	tweenBiasSlice(cfg.KBias, avgGrad, rate*0.05)
	tweenBiasSlice(cfg.VBias, avgGrad, rate*0.05)
}

// chainRuleUpdateRNN: Apply gradient update to RNN layer
func (ts *TweenState) chainRuleUpdateRNN(cfg *LayerConfig, input, output, outputGrad []float32, rate, mom float32) {
	hiddenSize := cfg.HiddenSize
	inputSize := cfg.RNNInputSize
	if inputSize == 0 {
		inputSize = len(input)
	}

	// Apply activation derivative (RNN uses tanh)
	localGrad := make([]float32, len(outputGrad))
	for j := 0; j < len(outputGrad) && j < len(output); j++ {
		actDeriv := ts.activateDerivativeFromOutput(output[j], ActivationTanh)
		localGrad[j] = outputGrad[j] * actDeriv
	}

	// Update input-to-hidden weights
	for h := 0; h < hiddenSize && h < len(localGrad); h++ {
		for i := 0; i < inputSize && i < len(input); i++ {
			wIdx := h*inputSize + i
			if wIdx < len(cfg.WeightIH) {
				cfg.WeightIH[wIdx] += rate * input[i] * localGrad[h]
			}
		}
		if h < len(cfg.BiasH) {
			cfg.BiasH[h] += rate * localGrad[h]
		}
	}
}

// chainRuleUpdateLSTM: Apply gradient update to LSTM layer
// LSTM: h_t = o_t * tanh(c_t), so output gate gets strongest gradient
func (ts *TweenState) chainRuleUpdateLSTM(cfg *LayerConfig, input, output, outputGrad []float32, rate, mom float32) {
	hiddenSize := cfg.HiddenSize
	inputSize := cfg.RNNInputSize
	if inputSize == 0 {
		inputSize = len(input)
	}

	// Gradient clipping threshold to prevent saturation
	const maxGrad float32 = 0.5

	// Apply activation derivative with clipping
	localGrad := make([]float32, len(outputGrad))
	for j := 0; j < len(outputGrad) && j < len(output); j++ {
		actDeriv := ts.activateDerivativeFromOutput(output[j], ActivationTanh)
		localGrad[j] = clipGrad(outputGrad[j]*actDeriv, maxGrad)
	}

	// LSTM gate gradients: output gate (o) gets most gradient since it directly affects h
	// forget gate (f) controls memory retention, input gate (i) and cell gate (g) add new info
	for h := 0; h < hiddenSize && h < len(localGrad); h++ {
		for i := 0; i < inputSize && i < len(input); i++ {
			wIdx := h*inputSize + i
			baseGrad := clipGrad(rate*input[i]*localGrad[h], maxGrad)

			// Output gate: strongest (directly multiplies output)
			if wIdx < len(cfg.WeightIH_o) {
				cfg.WeightIH_o[wIdx] += baseGrad * 1.0
			}
			// Forget gate: important for memory
			if wIdx < len(cfg.WeightIH_f) {
				cfg.WeightIH_f[wIdx] += baseGrad * 0.8
			}
			// Input gate: controls new information
			if wIdx < len(cfg.WeightIH_i) {
				cfg.WeightIH_i[wIdx] += baseGrad * 0.6
			}
			// Cell gate: candidate cell values
			if wIdx < len(cfg.WeightIH_g) {
				cfg.WeightIH_g[wIdx] += baseGrad * 0.6
			}
		}

		biasBaseGrad := clipGrad(rate*localGrad[h], maxGrad)
		if h < len(cfg.BiasH_o) {
			cfg.BiasH_o[h] += biasBaseGrad * 1.0
		}
		if h < len(cfg.BiasH_f) {
			cfg.BiasH_f[h] += biasBaseGrad * 0.8
		}
		if h < len(cfg.BiasH_i) {
			cfg.BiasH_i[h] += biasBaseGrad * 0.6
		}
		if h < len(cfg.BiasH_g) {
			cfg.BiasH_g[h] += biasBaseGrad * 0.6
		}
	}
}

// chainRuleUpdateNorm: Apply gradient update to LayerNorm/RMSNorm
// For LayerNorm: y = gamma * (x - mean) / std + beta
// dGamma = sum(grad * normalized_x), dBeta = sum(grad)
func (ts *TweenState) chainRuleUpdateNorm(cfg *LayerConfig, input, output, outputGrad []float32, rate float32) {
	// Gamma gradient: output_grad * normalized_input (stronger rate)
	for j := 0; j < len(outputGrad) && j < len(cfg.Gamma); j++ {
		if j < len(input) {
			// Use input (pre-norm) values for gradient correlation
			cfg.Gamma[j] += rate * outputGrad[j] * 1.0 // Full gradient strength
		}
	}
	if cfg.Type == LayerNorm {
		// Beta gets full gradient (it's the shift parameter)
		for j := 0; j < len(outputGrad) && j < len(cfg.Beta); j++ {
			cfg.Beta[j] += rate * outputGrad[j] * 1.0 // Full gradient strength
		}
	}
}

// chainRuleUpdateSwiGLU: Apply gradient update to SwiGLU layer
// SwiGLU: output = down_proj(silu(gate_proj(x)) * up_proj(x))
// All three projections need gradients
func (ts *TweenState) chainRuleUpdateSwiGLU(cfg *LayerConfig, input, output, outputGrad []float32, rate, mom float32) {
	// Gradient clipping threshold to prevent saturation
	const maxGrad float32 = 0.5

	// Update down projection (most direct path to output)
	for i := 0; i < len(input); i++ {
		for j := 0; j < len(outputGrad); j++ {
			wIdx := i*len(outputGrad) + j
			if wIdx < len(cfg.DownWeights) {
				grad := clipGrad(rate*input[i]*outputGrad[j], maxGrad)
				cfg.DownWeights[wIdx] += grad
			}
		}
	}

	// Update down biases
	for j := range cfg.DownBias {
		if j < len(outputGrad) {
			cfg.DownBias[j] += clipGrad(rate*outputGrad[j], maxGrad)
		}
	}

	// Gate and Up projections: both contribute equally to the multiplicative gate
	avgGrad := clipGrad(avgSlice(outputGrad), maxGrad)
	for i := 0; i < len(cfg.GateWeights); i++ {
		inIdx := i % len(input)
		if inIdx < len(input) {
			// Stronger gradient for gate/up (they work together)
			cfg.GateWeights[i] += clipGrad(rate*input[inIdx]*avgGrad*0.8, maxGrad)
		}
	}
	for i := 0; i < len(cfg.UpWeights); i++ {
		inIdx := i % len(input)
		if inIdx < len(input) {
			cfg.UpWeights[i] += clipGrad(rate*input[inIdx]*avgGrad*0.8, maxGrad)
		}
	}

	// Update gate/up biases
	for j := range cfg.GateBias {
		cfg.GateBias[j] += clipGrad(rate*avgGrad*0.5, maxGrad)
	}
	for j := range cfg.UpBias {
		cfg.UpBias[j] += clipGrad(rate*avgGrad*0.5, maxGrad)
	}
}

// tweenDense handles Dense/Fully-connected layers
func (ts *TweenState) tweenDense(cfg *LayerConfig, input, gaps []float32, rate, mom float32, layerIdx int) {
	for out := 0; out < len(gaps) && out < cfg.OutputHeight; out++ {
		gap := gaps[out]

		// Update bias using configurable multiplier
		if out < len(cfg.Bias) {
			cfg.Bias[out] += rate * gap * ts.Config.BiasRateMultiplier
		}

		// Update weights: Hebbian with gap using configurable multiplier
		for in := 0; in < len(input) && in < cfg.InputHeight; in++ {
			wIdx := in*cfg.OutputHeight + out
			if wIdx >= len(cfg.Kernel) {
				continue
			}
			delta := rate * input[in] * gap * ts.Config.WeightRateMultiplier
			if layerIdx < len(ts.WeightVel) && wIdx < len(ts.WeightVel[layerIdx]) {
				ts.WeightVel[layerIdx][wIdx] = mom*ts.WeightVel[layerIdx][wIdx] + (1-mom)*delta
				cfg.Kernel[wIdx] += ts.WeightVel[layerIdx][wIdx]
			} else {
				cfg.Kernel[wIdx] += delta
			}
		}
	}
}

// tweenConv2D handles 2D Convolutional layers
func (ts *TweenState) tweenConv2D(cfg *LayerConfig, input, gaps []float32, rate, mom float32) {
	if len(cfg.Kernel) == 0 {
		return
	}

	// Average gap across spatial dimensions to get per-filter signal
	numFilters := cfg.Filters
	if numFilters == 0 {
		numFilters = len(cfg.Bias)
	}
	if numFilters == 0 {
		return
	}

	perFilterGap := make([]float32, numFilters)
	gapsPerFilter := len(gaps) / numFilters
	if gapsPerFilter == 0 {
		gapsPerFilter = 1
	}

	for f := 0; f < numFilters; f++ {
		sum := float32(0)
		count := 0
		for j := f * gapsPerFilter; j < (f+1)*gapsPerFilter && j < len(gaps); j++ {
			sum += gaps[j]
			count++
		}
		if count > 0 {
			perFilterGap[f] = sum / float32(count)
		}
	}

	// Update filter weights based on gap
	kernelPerFilter := len(cfg.Kernel) / numFilters
	for f := 0; f < numFilters; f++ {
		gap := perFilterGap[f] * rate * ts.Config.WeightRateMultiplier
		for k := 0; k < kernelPerFilter; k++ {
			idx := f*kernelPerFilter + k
			if idx < len(cfg.Kernel) {
				cfg.Kernel[idx] += gap
			}
		}
		if f < len(cfg.Bias) {
			cfg.Bias[f] += perFilterGap[f] * rate * ts.Config.BiasRateMultiplier
		}
	}
}

// tweenAttention handles Multi-Head Attention with Relevance Routing
// MHA flow: input → Q,K,V projections → attention → output projection → output
func (ts *TweenState) tweenAttention(cfg *LayerConfig, input, gaps []float32, rate, mom float32) {
	numHeads := cfg.NumHeads
	dModel := cfg.DModel

	// Safety defaults
	if numHeads == 0 {
		numHeads = 2
	}
	if dModel == 0 {
		dModel = len(gaps)
	}

	headDim := dModel / numHeads
	if headDim == 0 {
		headDim = 1
	}

	// === STEP 1: Backprop Gap through Output Matrix ===
	// We need to know the specific gap for EACH HEAD.
	// The OutputWeight matrix mixes the heads together. We need to un-mix the error.

	headGaps := make([]float32, dModel) // Concatenated gaps for all heads

	// Transpose Multiply: headGaps = gaps * OutputWeight^T
	// This tells us: "What should the output of Head X have been?"
	for i := 0; i < dModel; i++ {
		sum := float32(0)
		// Optimization: iterate only relevant weights
		for j := 0; j < len(gaps); j++ {
			wIdx := i*len(gaps) + j
			if wIdx < len(cfg.OutputWeight) {
				sum += gaps[j] * cfg.OutputWeight[wIdx]
			}
		}
		headGaps[i] = sum
	}

	// Tween Output Weight (The Mixer)
	// Simple Hebbian: Weight += Rate * OutputGap * HeadInput
	// We approximate HeadInput with the current input to the layer (Input)
	// purely for the sake of momentum direction.
	avgGap := avgSlice(gaps)
	tweenWeightSlice(cfg.OutputWeight, gaps, rate*0.1) // Lower rate for output projection
	tweenBiasSlice(cfg.OutputBias, avgGap, rate*0.1)

	// === STEP 2: Tween Each Head ===
	for h := 0; h < numHeads; h++ {
		headStart := h * headDim
		headEnd := headStart + headDim

		// This is the Gap specific to THIS head
		thisHeadGap := headGaps[headStart:headEnd]

		// Calculate "Relevance" of the input
		// Does the input vector look like it could help fix this head's gap?
		// We approximate this by comparing Input direction vs Gap direction.
		inputMag := float32(0)
		gapMag := float32(0)
		dot := float32(0)

		limit := len(input)
		if limit > headDim {
			limit = headDim
		} // Heuristic: match dims

		for i := 0; i < limit; i++ {
			dot += input[i] * thisHeadGap[i]
			inputMag += input[i] * input[i]
			gapMag += thisHeadGap[i] * thisHeadGap[i]
		}

		// Alignment Score (-1.0 to 1.0)
		alignment := float32(0)
		if inputMag > 0 && gapMag > 0 {
			alignment = dot / (float32(math.Sqrt(float64(inputMag))) * float32(math.Sqrt(float64(gapMag))))
		}

		// === ACTION STRATEGY ===

		// 1. UPDATE V (Content)
		// V's job is to map Input -> Output.
		// We simply push V weights to map Input -> HeadGap.
		vStart := h * headDim * dModel
		vEnd := vStart + (headDim * dModel)

		if vEnd <= len(cfg.VWeights) {
			// Iterate weights for this head
			for i := vStart; i < vEnd; i++ {
				// Map flat index back to input index
				inIdx := (i - vStart) % dModel
				if inIdx < len(input) {
					// Standard Tween: Move weight to close the gap
					// V_new = V_old + Rate * Gap * Input
					gapIdx := (i - vStart) / dModel
					if gapIdx < len(thisHeadGap) {
						cfg.VWeights[i] += rate * thisHeadGap[gapIdx] * input[inIdx]
					}
				}
			}
		}

		// 2. UPDATE Q and K (Routing)
		// This is the "Relevance" Logic.
		// If Alignment > 0: This input is GOOD. We want to attend to it.
		//    -> Make Q and K more similar (maximize dot product).
		// If Alignment < 0: This input is BAD. We want to ignore it.
		//    -> Make Q and K more different.

		qStart := h * headDim * dModel
		kStart := h * headDim * dModel // Assuming shared Q/K structure for now

		routingRate := rate * alignment * 0.2 // Scale by how relevant the input is

		if qStart < len(cfg.QWeights) && kStart < len(cfg.KWeights) {
			for i := 0; i < (headDim * dModel); i++ {
				qIdx := qStart + i
				kIdx := kStart + i
				inIdx := i % dModel

				if qIdx < len(cfg.QWeights) && kIdx < len(cfg.KWeights) && inIdx < len(input) {
					// If alignment is positive (Good input), pull Q and K towards input
					// This increases the chance they will match each other
					val := input[inIdx] * routingRate

					cfg.QWeights[qIdx] += val
					cfg.KWeights[kIdx] += val
				}
			}
		}
	}

	// Tween biases for Q/K/V
	tweenBiasSlice(cfg.QBias, avgGap, rate*0.05)
	tweenBiasSlice(cfg.KBias, avgGap, rate*0.05)
	tweenBiasSlice(cfg.VBias, avgGap, rate*0.1)
}

// tweenRNN handles Recurrent Neural Network layers
func (ts *TweenState) tweenRNN(cfg *LayerConfig, input, gaps []float32, rate, mom float32) {
	avgGap := avgSlice(gaps)
	scaledRate := rate * avgGap * 0.1 // Increased from 0.001!

	tweenWeightSlice(cfg.WeightIH, input, scaledRate)
	tweenWeightSlice(cfg.WeightHH, input, scaledRate)
	tweenBiasSlice(cfg.BiasH, avgGap, rate*0.1)
}

// tweenLSTM handles Long Short-Term Memory layers
func (ts *TweenState) tweenLSTM(cfg *LayerConfig, input, gaps []float32, rate, mom float32) {
	avgGap := avgSlice(gaps)
	scaledRate := rate * avgGap * 0.1 // Increased from 0.001!

	// Input gate
	tweenWeightSlice(cfg.WeightIH_i, input, scaledRate)
	tweenWeightSlice(cfg.WeightHH_i, input, scaledRate)
	tweenBiasSlice(cfg.BiasH_i, avgGap, rate*0.1)

	// Forget gate
	tweenWeightSlice(cfg.WeightIH_f, input, scaledRate)
	tweenWeightSlice(cfg.WeightHH_f, input, scaledRate)
	tweenBiasSlice(cfg.BiasH_f, avgGap, rate*0.1)

	// Cell gate
	tweenWeightSlice(cfg.WeightIH_g, input, scaledRate)
	tweenWeightSlice(cfg.WeightHH_g, input, scaledRate)
	tweenBiasSlice(cfg.BiasH_g, avgGap, rate*0.1)

	// Output gate
	tweenWeightSlice(cfg.WeightIH_o, input, scaledRate)
	tweenWeightSlice(cfg.WeightHH_o, input, scaledRate)
	tweenBiasSlice(cfg.BiasH_o, avgGap, rate*0.1)
}

// tweenNorm handles LayerNorm and RMSNorm
func (ts *TweenState) tweenNorm(cfg *LayerConfig, gaps []float32, rate float32) {
	avgGap := avgSlice(gaps)

	// Tween gamma (scale)
	for i := range cfg.Gamma {
		cfg.Gamma[i] += avgGap * rate * 0.01
	}

	// Tween beta (shift) - only for LayerNorm, not RMSNorm
	if cfg.Type == LayerNorm {
		for i := range cfg.Beta {
			cfg.Beta[i] += avgGap * rate * 0.1
		}
	}
}

// tweenSwiGLU handles SwiGLU gated activation layers
func (ts *TweenState) tweenSwiGLU(cfg *LayerConfig, input, gaps []float32, rate, mom float32) {
	avgGap := avgSlice(gaps)
	scaledRate := rate * avgGap * 0.001

	tweenWeightSlice(cfg.GateWeights, input, scaledRate)
	tweenWeightSlice(cfg.UpWeights, input, scaledRate)
	tweenWeightSlice(cfg.DownWeights, gaps, scaledRate)

	tweenBiasSlice(cfg.GateBias, avgGap, rate*0.1)
	tweenBiasSlice(cfg.UpBias, avgGap, rate*0.1)
	tweenBiasSlice(cfg.DownBias, avgGap, rate*0.1)
}

// Helper: tween a weight slice using input correlation
func tweenWeightSlice(weights, signal []float32, rate float32) {
	if len(weights) == 0 || len(signal) == 0 {
		return
	}
	for i := range weights {
		sigIdx := i % len(signal)
		weights[i] += rate * signal[sigIdx]
	}
}

// Helper: tween a bias slice
func tweenBiasSlice(bias []float32, gap, rate float32) {
	for i := range bias {
		bias[i] += gap * rate
	}
}

// Helper: average of slice
func avgSlice(s []float32) float32 {
	if len(s) == 0 {
		return 0
	}
	sum := float32(0)
	for _, v := range s {
		sum += v
	}
	return sum / float32(len(s))
}

// TweenStep: One complete bidirectional iteration with explosion detection
func (ts *TweenState) TweenStep(n *Network, input []float32, targetClass int, outputSize int, rate float32) float32 {
	// 1. Forward: push through untrained network
	output := ts.ForwardPass(n, input)

	// 2. Backward: propagate expected output upward (chain rule or legacy)
	if ts.Config.UseChainRule {
		ts.BackwardPassChainRule(n, targetClass, outputSize)
	} else {
		ts.BackwardPass(n, targetClass, outputSize)
	}

	// 3. Calculate link budgets (information preservation)
	ts.CalculateLinkBudgets()

	// 4. Calculate current average gap for explosion detection
	avgGap := float32(0)
	for _, g := range ts.Gaps {
		avgGap += g
	}
	if len(ts.Gaps) > 0 {
		avgGap /= float32(len(ts.Gaps))
	}

	// 5. Detect gradient explosion and compute adaptive rate
	effectiveRate := rate * ts.getAdaptiveRateMultiplier(avgGap)

	// 6. Tween weights to close gaps (use chain rule or legacy) with effective rate
	if ts.Config.UseChainRule {
		ts.TweenWeightsChainRule(n, effectiveRate)
	} else {
		ts.TweenWeights(n, effectiveRate)
	}

	ts.TweenSteps++

	// Loss for tracking
	loss := float32(0)
	for i, v := range output {
		t := float32(0)
		if i == targetClass {
			t = 1.0
		}
		loss += (v - t) * (v - t)
	}
	return loss
}

// getAdaptiveRateMultiplier computes a learning rate multiplier based on gap behavior
// Returns a value between 0.01 and 1.0 to dampen learning when gaps explode
func (ts *TweenState) getAdaptiveRateMultiplier(currentGap float32) float32 {
	// If explosion detection is disabled, always return 1.0 (no rate adjustment)
	if !ts.Config.ExplosionDetection {
		return 1.0
	}

	// Initialize adaptive rate if not set
	if ts.AdaptiveRate == 0 {
		ts.AdaptiveRate = 1.0
	}

	// Build baseline from first few epochs
	if ts.GapSamples < 10 {
		ts.BaselineGap = (ts.BaselineGap*float32(ts.GapSamples) + currentGap) / float32(ts.GapSamples+1)
		ts.GapSamples++
		ts.PrevAvgGap = currentGap
		return ts.AdaptiveRate
	}

	// Calculate gap growth rate
	if ts.PrevAvgGap > 0.0001 {
		ts.GapGrowthRate = currentGap / ts.PrevAvgGap
	} else {
		ts.GapGrowthRate = 1.0
	}

	// Explosion detection thresholds
	const (
		growthThreshold  float32 = 1.5  // Gap grew by 50%+
		explodeThreshold float32 = 10.0 // Gap is 10x baseline = definitely exploding
		dampenFactor     float32 = 0.7  // Reduce rate by 30% when explosion detected
		recoverFactor    float32 = 1.05 // Increase rate by 5% when stable
		minRate          float32 = 0.01 // Minimum rate multiplier
		maxRate          float32 = 1.0  // Maximum rate multiplier
	)

	// Check for explosion
	isExploding := ts.GapGrowthRate > growthThreshold ||
		(ts.BaselineGap > 0 && currentGap > ts.BaselineGap*explodeThreshold)

	if isExploding {
		ts.ExplosionCount++
		// Dampen more aggressively with consecutive explosions
		dampen := dampenFactor
		if ts.ExplosionCount > 5 {
			dampen = 0.5 // More aggressive dampening
		}
		if ts.ExplosionCount > 10 {
			dampen = 0.3 // Very aggressive dampening
		}
		ts.AdaptiveRate *= dampen
		if ts.AdaptiveRate < minRate {
			ts.AdaptiveRate = minRate
		}
	} else {
		// Stable - slowly recover rate
		ts.ExplosionCount = 0
		ts.AdaptiveRate *= recoverFactor
		if ts.AdaptiveRate > maxRate {
			ts.AdaptiveRate = maxRate
		}
	}

	ts.PrevAvgGap = currentGap
	return ts.AdaptiveRate
}

// ResetBatch clears accumulated batch gaps
func (ts *TweenState) ResetBatch() {
	for i := range ts.BatchGaps {
		ts.BatchGaps[i] = nil
	}
	ts.BatchCount = 0
}

// TweenStepAccumulate: Accumulates gaps without applying weight updates
// Call TweenBatchApply when batch is complete to apply averaged updates
func (ts *TweenState) TweenStepAccumulate(n *Network, input []float32, targetClass int, outputSize int) float32 {
	// 1. Forward: push through network
	output := ts.ForwardPass(n, input)

	// 2. Backward: propagate expected output upward (chain rule or legacy)
	if ts.Config.UseChainRule {
		ts.BackwardPassChainRule(n, targetClass, outputSize)
	} else {
		ts.BackwardPass(n, targetClass, outputSize)
	}

	// 3. Calculate link budgets
	ts.CalculateLinkBudgets()

	// 4. Accumulate gaps per layer (don't apply weights yet)
	for i := 0; i < ts.TotalLayers; i++ {
		actual := ts.ForwardActs[i+1]
		target := ts.BackwardTargets[i+1]

		if len(actual) == 0 || len(target) == 0 {
			continue
		}

		minOut := len(actual)
		if len(target) < minOut {
			minOut = len(target)
		}

		// Initialize batch gaps slice if needed
		if ts.BatchGaps[i] == nil {
			ts.BatchGaps[i] = make([]float32, minOut)
		}

		// Accumulate gaps
		for j := 0; j < minOut && j < len(ts.BatchGaps[i]); j++ {
			ts.BatchGaps[i][j] += target[j] - actual[j]
		}
	}

	ts.BatchCount++
	ts.TweenSteps++

	// Loss for tracking
	loss := float32(0)
	for i, v := range output {
		t := float32(0)
		if i == targetClass {
			t = 1.0
		}
		loss += (v - t) * (v - t)
	}
	return loss
}

// TweenBatchApply: Applies accumulated batch gaps as averaged weight updates
func (ts *TweenState) TweenBatchApply(n *Network, rate float32) {
	if ts.BatchCount == 0 {
		return
	}

	mom := ts.Config.Momentum
	batchScale := 1.0 / float32(ts.BatchCount)

	for i := 0; i < ts.TotalLayers; i++ {
		budget := ts.LinkBudgets[i]
		if budget < ts.Config.IgnoreThreshold {
			continue
		}

		cfg := ts.getLayerCfg(n, i)
		if cfg == nil {
			continue
		}

		input := ts.ForwardActs[i]
		output := ts.ForwardActs[i+1]
		if len(input) == 0 || len(ts.BatchGaps[i]) == 0 {
			continue
		}

		// Average the accumulated gaps
		avgGaps := make([]float32, len(ts.BatchGaps[i]))
		for j := range ts.BatchGaps[i] {
			avgGaps[j] = ts.BatchGaps[i][j] * batchScale
		}

		// Compute depth scale for chain rule
		// Use safe scaling to prevent explosion in very deep networks
		depthFromOutput := float32(ts.TotalLayers - 1 - i)
		depthScale := safeDepthScale(depthFromOutput, ts.Config.DepthScaleFactor, 100.0)

		// Use chain rule or legacy tweening
		if ts.Config.UseChainRule {
			// Use averaged gaps as output gradient for chain rule update
			switch cfg.Type {
			case LayerDense:
				ts.chainRuleUpdateDense(cfg, input, output, avgGaps, rate*ts.Config.DenseRate*depthScale, mom, i)
			case LayerConv2D:
				ts.chainRuleUpdateConv2D(cfg, input, output, avgGaps, rate*ts.Config.Conv2DRate*depthScale, mom)
			case LayerMultiHeadAttention:
				ts.chainRuleUpdateAttention(cfg, input, output, avgGaps, rate*ts.Config.AttentionRate*depthScale, mom)
			case LayerRNN:
				ts.chainRuleUpdateRNN(cfg, input, output, avgGaps, rate*ts.Config.RNNRate*depthScale, mom)
			case LayerLSTM:
				ts.chainRuleUpdateLSTM(cfg, input, output, avgGaps, rate*ts.Config.LSTMRate*depthScale, mom)
			case LayerNorm, LayerRMSNorm:
				ts.chainRuleUpdateNorm(cfg, input, output, avgGaps, rate*ts.Config.NormRate*depthScale)
			case LayerSwiGLU:
				ts.chainRuleUpdateSwiGLU(cfg, input, output, avgGaps, rate*ts.Config.SwiGLURate*depthScale, mom)
			}
		} else {
			// Scale rate by link budget
			layerRate := rate * (ts.Config.LinkBudgetScale + budget*ts.Config.LinkBudgetScale)

			// Apply legacy tweening based on layer type
			switch cfg.Type {
			case LayerDense:
				ts.tweenDense(cfg, input, avgGaps, layerRate*ts.Config.DenseRate, mom, i)
			case LayerConv2D:
				ts.tweenConv2D(cfg, input, avgGaps, layerRate*ts.Config.Conv2DRate, mom)
			case LayerMultiHeadAttention:
				ts.tweenAttention(cfg, input, avgGaps, layerRate*ts.Config.AttentionRate, mom)
			case LayerRNN:
				ts.tweenRNN(cfg, input, avgGaps, layerRate*ts.Config.RNNRate, mom)
			case LayerLSTM:
				ts.tweenLSTM(cfg, input, avgGaps, layerRate*ts.Config.LSTMRate, mom)
			case LayerNorm, LayerRMSNorm:
				ts.tweenNorm(cfg, avgGaps, layerRate*ts.Config.NormRate)
			case LayerSwiGLU:
				ts.tweenSwiGLU(cfg, input, avgGaps, layerRate*ts.Config.SwiGLURate, mom)
			}
		}

		row := i / (n.GridCols * n.LayersPerCell)
		col := (i / n.LayersPerCell) % n.GridCols
		layer := i % n.LayersPerCell
		n.SetLayer(row, col, layer, *cfg)
	}

	// Reset batch state
	ts.ResetBatch()
}

// TweenBatch: Convenience function for batch training (non-stepping mode)
// Processes all samples in batch, accumulates gaps, then applies averaged update
func (ts *TweenState) TweenBatch(n *Network, inputs [][]float32, targetClasses []int, outputSize int, rate float32) float32 {
	if len(inputs) == 0 {
		return 0
	}

	ts.ResetBatch()
	totalLoss := float32(0)

	for i := range inputs {
		loss := ts.TweenStepAccumulate(n, inputs[i], targetClasses[i], outputSize)
		totalLoss += loss
	}

	ts.TweenBatchApply(n, rate)
	return totalLoss / float32(len(inputs))
}

// Train with early stopping
func (ts *TweenState) Train(n *Network, inputs [][]float32, expected []float64, epochs int, rate float32,
	callback func(epoch int, avgLoss float32, metrics *DeviationMetrics)) {

	outSize := 2
	lastCfg := ts.getLayerCfg(n, ts.TotalLayers-1)
	if lastCfg != nil && lastCfg.OutputHeight > 0 {
		outSize = lastCfg.OutputHeight
	}

	for epoch := 0; epoch < epochs; epoch++ {
		epochLoss := float32(0)
		for _, idx := range rand.Perm(len(inputs)) {
			epochLoss += ts.TweenStep(n, inputs[idx], int(expected[idx]), outSize, rate)
		}
		avgLoss := epochLoss / float32(len(inputs))
		ts.LossHistory = append(ts.LossHistory, avgLoss)

		// PRUNING: Check for dead layers and amputate if necessary
		ts.checkAndPruneLayers(n)

		// Record history for visualization (every epoch)
		ts.recordEpochHistory(epoch)

		if epoch%ts.Config.EvalFrequency == 0 || epoch == epochs-1 {
			metrics, _ := n.EvaluateNetwork(inputs, expected)
			if metrics.Score > ts.BestScore {
				ts.BestScore = metrics.Score
				ts.SaveBest(n)
			}

			// Record detailed epoch metrics
			ts.recordEpochMetrics(epoch+1, avgLoss, metrics)

			// Verbose output
			if ts.Verbose {
				ts.printVerboseProgress(epoch+1, epochs, avgLoss, metrics)
			}

			if callback != nil {
				callback(epoch+1, avgLoss, metrics)
			}
			if metrics.Score >= ts.Config.EarlyStopThreshold {
				if ts.Verbose {
					fmt.Printf("\n🎯 Early stop! Score %.2f%% >= threshold %.2f%%\n", metrics.Score, ts.Config.EarlyStopThreshold)
				}
				ts.RestoreBest(n)
				return
			}
		}
	}
	if ts.BestScore > 0 {
		ts.RestoreBest(n)
	}
}

// recordEpochHistory captures link budgets and gaps for heatmap visualization
func (ts *TweenState) recordEpochHistory(epoch int) {
	// Copy current link budgets
	budgets := make([]float32, len(ts.LinkBudgets))
	copy(budgets, ts.LinkBudgets)
	ts.LinkBudgetHistory = append(ts.LinkBudgetHistory, budgets)

	// Copy current gaps
	gaps := make([]float32, len(ts.Gaps))
	copy(gaps, ts.Gaps)
	ts.GapHistory = append(ts.GapHistory, gaps)

	// Record overall depth barrier (product of all budgets)
	depthBarrier := float32(1.0)
	for _, b := range ts.LinkBudgets {
		depthBarrier *= b
	}
	ts.DepthBarrierHistory = append(ts.DepthBarrierHistory, depthBarrier)
}

// recordEpochMetrics captures detailed metrics for analysis
func (ts *TweenState) recordEpochMetrics(epoch int, avgLoss float32, metrics *DeviationMetrics) {
	avgBudget, minBudget, maxBudget := ts.GetBudgetSummary()
	avgGap, maxGap := ts.GetGapSummary()

	// Find bottleneck layer (lowest link budget)
	bottleneck := 0
	minB := float32(1.0)
	for i, b := range ts.LinkBudgets {
		if b < minB {
			minB = b
			bottleneck = i
		}
	}

	// Overall depth barrier
	depthBarrier := float32(1.0)
	for _, b := range ts.LinkBudgets {
		depthBarrier *= b
	}

	ts.EpochMetrics = append(ts.EpochMetrics, TweenEpochMetrics{
		Epoch:           epoch,
		AvgLoss:         avgLoss,
		Score:           metrics.Score,
		AvgLinkBudget:   avgBudget,
		MinLinkBudget:   minBudget,
		MaxLinkBudget:   maxBudget,
		AvgGap:          avgGap,
		MaxGap:          maxGap,
		DepthBarrier:    depthBarrier,
		BottleneckLayer: bottleneck,
	})
}

// printVerboseProgress prints detailed training progress
func (ts *TweenState) printVerboseProgress(epoch, totalEpochs int, avgLoss float32, metrics *DeviationMetrics) {
	avgBudget, minBudget, _ := ts.GetBudgetSummary()
	avgGap, _ := ts.GetGapSummary()

	// Find bottleneck
	bottleneck := 0
	minB := float32(1.0)
	for i, b := range ts.LinkBudgets {
		if b < minB {
			minB = b
			bottleneck = i
		}
	}

	// Overall depth barrier
	depthBarrier := float32(1.0)
	for _, b := range ts.LinkBudgets {
		depthBarrier *= b
	}

	// Print with visual indicators
	fmt.Printf("Epoch %4d/%d | Loss: %.4f | Score: %5.1f%% | ",
		epoch, totalEpochs, avgLoss, metrics.Score)
	fmt.Printf("LinkBudget: %.3f (min %.3f @L%d) | ", avgBudget, minBudget, bottleneck)
	fmt.Printf("Gap: %.4f | DepthBarrier: %.4f", avgGap, depthBarrier)

	// Visual heatmap bar for link budgets
	fmt.Print(" [")
	for _, b := range ts.LinkBudgets {
		if b < ts.Config.IgnoreThreshold {
			fmt.Print("X") // Ignored (Dead)
		} else if b > 0.8 {
			fmt.Print("█") // High budget
		} else if b > 0.6 {
			fmt.Print("▓")
		} else if b > 0.4 {
			fmt.Print("▒")
		} else if b > 0.2 {
			fmt.Print("░")
		} else {
			fmt.Print("·") // Low budget (bottleneck)
		}
	}
	fmt.Println("]")
}

func (ts *TweenState) SaveBest(n *Network) {
	for i := 0; i < ts.TotalLayers; i++ {
		cfg := ts.getLayerCfg(n, i)
		if cfg != nil && len(cfg.Kernel) > 0 {
			ts.BestWeights[i] = [][]float32{make([]float32, len(cfg.Kernel))}
			copy(ts.BestWeights[i][0], cfg.Kernel)
			ts.BestBiases[i] = [][]float32{make([]float32, len(cfg.Bias))}
			copy(ts.BestBiases[i][0], cfg.Bias)
		}
	}
}

func (ts *TweenState) RestoreBest(n *Network) {
	for i := 0; i < ts.TotalLayers; i++ {
		if len(ts.BestWeights[i]) == 0 {
			continue
		}
		cfg := ts.getLayerCfg(n, i)
		if cfg != nil && len(cfg.Kernel) > 0 {
			copy(cfg.Kernel, ts.BestWeights[i][0])
			copy(cfg.Bias, ts.BestBiases[i][0])
			row := i / (n.GridCols * n.LayersPerCell)
			col := (i / n.LayersPerCell) % n.GridCols
			layer := i % n.LayersPerCell
			n.SetLayer(row, col, layer, *cfg)
		}
	}
}

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
	return sum / float32(len(ts.LinkBudgets)), min, max
}

func (ts *TweenState) GetGapSummary() (avg, max float32) {
	if len(ts.Gaps) == 0 {
		return 0, 0
	}
	sum := float32(0)
	for _, g := range ts.Gaps {
		sum += g
		if g > max {
			max = g
		}
	}
	return sum / float32(len(ts.Gaps)), max
}

func (ts *TweenState) CalculateLinkBudgetsFromSample(n *Network, input []float32) {
	ts.ForwardPass(n, input)
	ts.CalculateLinkBudgets()
}

func clamp(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// checkAndPruneLayers detects, toggles off, and eventually deletes dead layers
func (ts *TweenState) checkAndPruneLayers(n *Network) {
	if !ts.Config.PruneEnabled {
		return
	}

	// SAFETY: Physical pruning breaks Grid (Row/Col) mapping by changing LayersPerCell stride
	// Only allow for 1x1 stacks
	if n.GridRows > 1 || n.GridCols > 1 {
		return
	}

	// 1. Identify candidate layers (Update DeadEpochs for all layers)
	for i := 1; i < ts.TotalLayers-1; i++ {
		// If disabled, track how long it's been disabled (using DeadEpochs)
		if n.Layers[i].IsDisabled {
			ts.DeadEpochs[i]++

			// CHECK FOR PHYSICAL REMOVAL (The "Decide to delete" phase)
			// If it's been disabled for another patience cycle, chop it out
			if ts.DeadEpochs[i] >= ts.Config.PrunePatience {
				ts.physicallyRemoveLayer(n, i)
				return // Indices shifted, abort this pass
			}
			continue
		}

		// If active but bad budget, increment dead count
		if ts.LinkBudgets[i] < ts.Config.PruneThreshold {
			ts.DeadEpochs[i]++
		} else {
			ts.DeadEpochs[i] = 0 // Reset if it recovers
		}
	}

	// 2. Find candidate to toggle off - FORWARD SCAN (Input -> Output)
	// User requested to focus on "Left side" (Low Index) and disable lots of layers
	pruneStartIdx := -1
	batchSize := 3 // How many layers to disable at once

	for i := 1; i < ts.TotalLayers-1; i++ {
		if n.Layers[i].IsDisabled {
			continue
		}
		if ts.DeadEpochs[i] >= ts.Config.PrunePatience {
			pruneStartIdx = i
			break // Found the first candidate
		}
	}

	if pruneStartIdx == -1 {
		return // No toggle needed
	}

	// 3. ALERT & BATCH DISABLE
	if ts.Verbose {
		fmt.Printf("\n✂️  PRUNING ALERT: Batch Disabling Layers %d-%d (Budget %.3f) ✂️\n",
			pruneStartIdx, pruneStartIdx+batchSize-1, ts.LinkBudgets[pruneStartIdx])
	}

	// Batch disable!
	count := 0
	for i := pruneStartIdx; i < ts.TotalLayers-1 && count < batchSize; i++ {
		if !n.Layers[i].IsDisabled {
			n.Layers[i].IsDisabled = true
			ts.DeadEpochs[i] = 0 // Reset timer (now tracks disable duration)
			count++
		}
	}
}

// physicallyRemoveLayer permanently deletes a layer from the Network and TweenState
func (ts *TweenState) physicallyRemoveLayer(n *Network, pruneIdx int) {
	if ts.Verbose {
		fmt.Printf("\n🗑️  DELETE ALERT: Physically Removing Layer %d (Disabled for %d epochs) 🗑️\n",
			pruneIdx, ts.DeadEpochs[pruneIdx])
	}

	// Remove from Network struct
	// This assumes simple sequential structure (1 cell, many layers)
	n.Layers = append(n.Layers[:pruneIdx], n.Layers[pruneIdx+1:]...)
	n.LayersPerCell-- // Shrink the cell depth

	// Resize Network internal storage to avoid index out of bounds
	n.activations = append(n.activations[:pruneIdx+1], n.activations[pruneIdx+2:]...) // +1 offset for input
	n.preActivations = append(n.preActivations[:pruneIdx], n.preActivations[pruneIdx+1:]...)
	// Gradients might not have been allocated yet if training hasn't started, but usually they are
	if len(n.kernelGradients) > pruneIdx {
		n.kernelGradients = append(n.kernelGradients[:pruneIdx], n.kernelGradients[pruneIdx+1:]...)
	}
	if len(n.biasGradients) > pruneIdx {
		n.biasGradients = append(n.biasGradients[:pruneIdx], n.biasGradients[pruneIdx+1:]...)
	}

	// Resize TweenState tracking arrays
	ts.TotalLayers--

	ts.LinkBudgets = append(ts.LinkBudgets[:pruneIdx], ts.LinkBudgets[pruneIdx+1:]...)
	ts.ForwardActs = append(ts.ForwardActs[:pruneIdx], ts.ForwardActs[pruneIdx+1:]...)
	ts.BackwardTargets = append(ts.BackwardTargets[:pruneIdx], ts.BackwardTargets[pruneIdx+1:]...)
	ts.WeightVel = append(ts.WeightVel[:pruneIdx], ts.WeightVel[pruneIdx+1:]...)
	ts.DeadEpochs = append(ts.DeadEpochs[:pruneIdx], ts.DeadEpochs[pruneIdx+1:]...)

	// No need to reset adjacent epochs, as they might be next in line for deletion
}
