package nn

import (
	"fmt"
	"math"
	"math/rand"
)

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

	TotalLayers int
	TweenSteps  int
	LossHistory []float32

	// === TUNABLE LEARNING	// === FRONTIER OSCILLATION ===
	// Instead of cycling through ALL layers, only oscillate within the "frontier" region
	// where layers are healthy (above threshold). Expands frontier as more layers improve.
	FrontierEnabled   bool    // Default: true, use adaptive frontier
	FrontierMin       int     // Minimum layer index to oscillate to (updated dynamically)
	FrontierThreshold float32 // Default: 0.55, layers above this are "healthy"
	FrontierNoise     float32 // Default: 0.0, amount of random noise to inject at frontier edge

	IgnoreThreshold float32 // Default: 0.2, if LinkBudget < Threshold, SKIP updating layer (it's dead)
	DenseRate       float32 // Default: 1.0, multiplier for Dense layers
	RNNRate         float32 // Default: 0.5, multiplier for RNN layers
	LSTMRate        float32 // Default: 0.5, multiplier for LSTM layers
	AttentionRate   float32 // Default: 0.2, multiplier for Attention layers
	NormRate        float32 // Default: 0.1, multiplier for LayerNorm/RMSNorm
	SwiGLURate      float32 // Default: 0.2, multiplier for SwiGLU layers
	Conv2DRate      float32 // Default: 0.1, multiplier for Conv2D layers

	// === MOMENTUM & UPDATE SCALING ===
	Momentum             float32 // Default: 0.9, momentum for weight updates
	BiasRateMultiplier   float32 // Default: 0.1, scale factor for bias updates
	WeightRateMultiplier float32 // Default: 0.01, scale factor for weight updates

	// === BACKWARD PASS CLAMPING ===
	// Clamp ranges for backward target estimation per activation type
	TanhClampMin    float32 // Default: -0.95
	TanhClampMax    float32 // Default: 0.95
	SigmoidClampMin float32 // Default: 0.05
	SigmoidClampMax float32 // Default: 0.95
	ReLUClampMax    float32 // Default: 10.0, prevents exploding targets

	// === TRAINING BEHAVIOR ===
	EarlyStopThreshold float64 // Default: 95.0, stop training when score exceeds this
	EvalFrequency      int     // Default: 5, evaluate every N epochs
	LinkBudgetScale    float32 // Default: 0.5, how much link budget affects learning rate

	// === DYNAMIC LAYER PRUNING (NETWORK SURGERY) ===
	// Automatically removes layers that block signal (Budget < Threshold) for too long
	PruneEnabled   bool    // Default: false, enable dynamic layer removal
	PruneThreshold float32 // Default: 0.1, treat layer as dead if budget below this
	PrunePatience  int     // Default: 10, how many epochs to wait before amputating
	DeadEpochs     []int   // Tracks consecutive epochs a layer has been dead

	// === BATCH TRAINING ===
	// Accumulate gaps across multiple samples before applying weight updates
	BatchSize  int         // Default: 1 (online), set >1 for batch mode
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

// NewTweenState creates tween state with tunable defaults
func NewTweenState(n *Network) *TweenState {
	total := n.TotalLayers()
	ts := &TweenState{
		ForwardActs:     make([][]float32, total+1),
		BackwardTargets: make([][]float32, total+1),
		LinkBudgets:     make([]float32, total),
		Gaps:            make([]float32, total),
		WeightVel:       make([][]float32, total),
		BiasVel:         make([][]float32, total),
		BestWeights:     make([][][]float32, total),
		BestBiases:      make([][][]float32, total),
		TotalLayers:     total,
		// Layer-specific learning rate multipliers
		DenseRate:         1.0,
		RNNRate:           0.5,
		LSTMRate:          0.5,
		AttentionRate:     0.2,
		NormRate:          0.1,
		SwiGLURate:        0.2,
		Conv2DRate:        0.1,
		FrontierEnabled:   true,
		FrontierMin:       total - 1, // Start at output, will expand toward input
		FrontierThreshold: 0.55,      // Layers above this are considered healthy
		FrontierNoise:     0.0,       // No noise by default
		IgnoreThreshold:   0.2,       // Ignore layers with < 20% link budget
		// Momentum & update scaling
		Momentum:             0.9,
		BiasRateMultiplier:   0.1,
		WeightRateMultiplier: 0.01,
		// Backward pass clamping
		TanhClampMin:    -0.95,
		TanhClampMax:    0.95,
		SigmoidClampMin: 0.05,
		SigmoidClampMax: 0.95,
		ReLUClampMax:    10.0,
		// Training behavior
		EarlyStopThreshold: 95.0,
		EvalFrequency:      5,
		LinkBudgetScale:    0.5,
		// Pruning (Surgery)
		PruneEnabled:   false,
		PruneThreshold: 0.1,
		PrunePatience:  10,
		DeadEpochs:     make([]int, total),
		// Batch training
		BatchSize:  1, // Online by default
		BatchGaps:  make([][]float32, total),
		BatchCount: 0,

		// Visualization defaults
		Verbose:             false,
		LinkBudgetHistory:   make([][]float32, 0),
		GapHistory:          make([][]float32, 0),
		DepthBarrier:        make([]float32, total),
		DepthBarrierHistory: make([]float32, 0),
		EpochMetrics:        make([]TweenEpochMetrics, 0),
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
			if totalWeight > 0.01 {
				estimated[in] = importance / float32(totalWeight)
			}

			// Clamp to valid activation range using configurable bounds
			if cfg.Activation == ActivationTanh {
				estimated[in] = clamp(estimated[in], ts.TanhClampMin, ts.TanhClampMax)
			} else if cfg.Activation == ActivationSigmoid {
				estimated[in] = clamp(estimated[in], ts.SigmoidClampMin, ts.SigmoidClampMax)
			} else if cfg.Activation == ActivationScaledReLU || cfg.Activation == ActivationLeakyReLU {
				estimated[in] = clamp(estimated[in], 0, ts.ReLUClampMax)
			}
		}

		ts.BackwardTargets[i] = estimated
		currentTarget = estimated
	}
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
	mom := ts.Momentum

	// Tweening logic
	for i := 0; i < ts.TotalLayers; i++ {
		budget := ts.LinkBudgets[i]

		// DEAD LAYER IGNORING: If budget is too low, don't update!
		// A low budget means the signal is garbage, so updating just adds noise.
		if budget < ts.IgnoreThreshold {
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
		layerRate := rate * (ts.LinkBudgetScale + budget*ts.LinkBudgetScale)

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
			ts.tweenDense(cfg, input, outputGaps, layerRate*ts.DenseRate, mom, i)
		case LayerConv2D:
			ts.tweenConv2D(cfg, input, outputGaps, layerRate*ts.Conv2DRate, mom)
		case LayerMultiHeadAttention:
			ts.tweenAttention(cfg, input, outputGaps, layerRate*ts.AttentionRate, mom)
		case LayerRNN:
			ts.tweenRNN(cfg, input, outputGaps, layerRate*ts.RNNRate, mom)
		case LayerLSTM:
			ts.tweenLSTM(cfg, input, outputGaps, layerRate*ts.LSTMRate, mom)
		case LayerNorm, LayerRMSNorm:
			ts.tweenNorm(cfg, outputGaps, layerRate*ts.NormRate)
		case LayerSwiGLU:
			ts.tweenSwiGLU(cfg, input, outputGaps, layerRate*ts.SwiGLURate, mom)
			// LayerSoftmax, LayerResidual, LayerParallel - no trainable weights
		}

		row := i / (n.GridCols * n.LayersPerCell)
		col := (i / n.LayersPerCell) % n.GridCols
		layer := i % n.LayersPerCell
		n.SetLayer(row, col, layer, *cfg)
	}
}

// tweenDense handles Dense/Fully-connected layers
func (ts *TweenState) tweenDense(cfg *LayerConfig, input, gaps []float32, rate, mom float32, layerIdx int) {
	for out := 0; out < len(gaps) && out < cfg.OutputHeight; out++ {
		gap := gaps[out]

		// Update bias using configurable multiplier
		if out < len(cfg.Bias) {
			cfg.Bias[out] += rate * gap * ts.BiasRateMultiplier
		}

		// Update weights: Hebbian with gap using configurable multiplier
		for in := 0; in < len(input) && in < cfg.InputHeight; in++ {
			wIdx := in*cfg.OutputHeight + out
			if wIdx >= len(cfg.Kernel) {
				continue
			}
			delta := rate * input[in] * gap * ts.WeightRateMultiplier
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
		gap := perFilterGap[f] * rate * ts.WeightRateMultiplier
		for k := 0; k < kernelPerFilter; k++ {
			idx := f*kernelPerFilter + k
			if idx < len(cfg.Kernel) {
				cfg.Kernel[idx] += gap
			}
		}
		if f < len(cfg.Bias) {
			cfg.Bias[f] += perFilterGap[f] * rate * ts.BiasRateMultiplier
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

// TweenStep: One complete bidirectional iteration
func (ts *TweenState) TweenStep(n *Network, input []float32, targetClass int, outputSize int, rate float32) float32 {
	// 1. Forward: push through untrained network
	output := ts.ForwardPass(n, input)

	// 2. Backward: propagate expected output upward
	ts.BackwardPass(n, targetClass, outputSize)

	// 3. Calculate link budgets (information preservation)
	ts.CalculateLinkBudgets()

	// 4. Tween weights to close gaps
	ts.TweenWeights(n, rate)

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

	// 2. Backward: propagate expected output upward
	ts.BackwardPass(n, targetClass, outputSize)

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

	mom := ts.Momentum
	batchScale := 1.0 / float32(ts.BatchCount)

	for i := 0; i < ts.TotalLayers; i++ {
		budget := ts.LinkBudgets[i]
		if budget < ts.IgnoreThreshold {
			continue
		}

		cfg := ts.getLayerCfg(n, i)
		if cfg == nil {
			continue
		}

		input := ts.ForwardActs[i]
		if len(input) == 0 || len(ts.BatchGaps[i]) == 0 {
			continue
		}

		// Scale rate by link budget
		layerRate := rate * (ts.LinkBudgetScale + budget*ts.LinkBudgetScale)

		// Average the accumulated gaps
		avgGaps := make([]float32, len(ts.BatchGaps[i]))
		for j := range ts.BatchGaps[i] {
			avgGaps[j] = ts.BatchGaps[i][j] * batchScale
		}

		// Apply tweening based on layer type
		switch cfg.Type {
		case LayerDense:
			ts.tweenDense(cfg, input, avgGaps, layerRate*ts.DenseRate, mom, i)
		case LayerConv2D:
			ts.tweenConv2D(cfg, input, avgGaps, layerRate*ts.Conv2DRate, mom)
		case LayerMultiHeadAttention:
			ts.tweenAttention(cfg, input, avgGaps, layerRate*ts.AttentionRate, mom)
		case LayerRNN:
			ts.tweenRNN(cfg, input, avgGaps, layerRate*ts.RNNRate, mom)
		case LayerLSTM:
			ts.tweenLSTM(cfg, input, avgGaps, layerRate*ts.LSTMRate, mom)
		case LayerNorm, LayerRMSNorm:
			ts.tweenNorm(cfg, avgGaps, layerRate*ts.NormRate)
		case LayerSwiGLU:
			ts.tweenSwiGLU(cfg, input, avgGaps, layerRate*ts.SwiGLURate, mom)
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

		if epoch%ts.EvalFrequency == 0 || epoch == epochs-1 {
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
			if metrics.Score >= ts.EarlyStopThreshold {
				if ts.Verbose {
					fmt.Printf("\n🎯 Early stop! Score %.2f%% >= threshold %.2f%%\n", metrics.Score, ts.EarlyStopThreshold)
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
		if b < ts.IgnoreThreshold {
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
	if !ts.PruneEnabled {
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
			if ts.DeadEpochs[i] >= ts.PrunePatience {
				ts.physicallyRemoveLayer(n, i)
				return // Indices shifted, abort this pass
			}
			continue
		}

		// If active but bad budget, increment dead count
		if ts.LinkBudgets[i] < ts.PruneThreshold {
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
		if ts.DeadEpochs[i] >= ts.PrunePatience {
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
