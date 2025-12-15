package nn

import (
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
}

// NewTweenState creates tween state
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

			// Clamp to valid activation range
			if cfg.Activation == ActivationTanh {
				estimated[in] = clamp(estimated[in], -0.95, 0.95)
			} else if cfg.Activation == ActivationSigmoid {
				estimated[in] = clamp(estimated[in], 0.05, 0.95)
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
}

// TweenWeights: Adjust weights to close the gap at each layer
// Supports ALL layer types: Dense, Conv2D, Attention, LSTM, LayerNorm, SwiGLU
func (ts *TweenState) TweenWeights(n *Network, rate float32) {
	mom := float32(0.9)

	for i := 0; i < ts.TotalLayers; i++ {
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

		// Scale rate by link budget
		budget := ts.LinkBudgets[i]
		layerRate := rate * (0.5 + budget*0.5)

		// Calculate output gap
		minOut := len(actual)
		if len(target) < minOut {
			minOut = len(target)
		}
		outputGaps := make([]float32, minOut)
		for j := 0; j < minOut; j++ {
			outputGaps[j] = target[j] - actual[j]
		}

		// Tween based on layer type
		switch cfg.Type {
		case LayerDense:
			ts.tweenDense(cfg, input, outputGaps, layerRate, mom, i)
		case LayerConv2D:
			ts.tweenConv2D(cfg, input, outputGaps, layerRate, mom)
		case LayerMultiHeadAttention:
			ts.tweenAttention(cfg, input, outputGaps, layerRate, mom)
		case LayerRNN:
			ts.tweenRNN(cfg, input, outputGaps, layerRate, mom)
		case LayerLSTM:
			ts.tweenLSTM(cfg, input, outputGaps, layerRate, mom)
		case LayerNorm, LayerRMSNorm:
			ts.tweenNorm(cfg, outputGaps, layerRate)
		case LayerSwiGLU:
			ts.tweenSwiGLU(cfg, input, outputGaps, layerRate, mom)
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

		// Update bias
		if out < len(cfg.Bias) {
			cfg.Bias[out] += rate * gap * 0.1
		}

		// Update weights: Hebbian with gap
		for in := 0; in < len(input) && in < cfg.InputHeight; in++ {
			wIdx := in*cfg.OutputHeight + out
			if wIdx >= len(cfg.Kernel) {
				continue
			}
			delta := rate * input[in] * gap * 0.01
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
		gap := perFilterGap[f] * rate * 0.01
		for k := 0; k < kernelPerFilter; k++ {
			idx := f*kernelPerFilter + k
			if idx < len(cfg.Kernel) {
				cfg.Kernel[idx] += gap
			}
		}
		if f < len(cfg.Bias) {
			cfg.Bias[f] += perFilterGap[f] * rate * 0.1
		}
	}
}

// tweenAttention handles Multi-Head Attention layers
func (ts *TweenState) tweenAttention(cfg *LayerConfig, input, gaps []float32, rate, mom float32) {
	// Average gap to get overall direction
	avgGap := float32(0)
	for _, g := range gaps {
		avgGap += g
	}
	if len(gaps) > 0 {
		avgGap /= float32(len(gaps))
	}

	scaledRate := rate * avgGap * 0.001

	// Tween Q, K, V, Output weights
	tweenWeightSlice(cfg.QWeights, input, scaledRate)
	tweenWeightSlice(cfg.KWeights, input, scaledRate)
	tweenWeightSlice(cfg.VWeights, input, scaledRate)
	tweenWeightSlice(cfg.OutputWeight, gaps, scaledRate)

	// Tween biases
	tweenBiasSlice(cfg.QBias, avgGap, rate*0.1)
	tweenBiasSlice(cfg.KBias, avgGap, rate*0.1)
	tweenBiasSlice(cfg.VBias, avgGap, rate*0.1)
	tweenBiasSlice(cfg.OutputBias, avgGap, rate*0.1)
}

// tweenRNN handles Recurrent Neural Network layers
func (ts *TweenState) tweenRNN(cfg *LayerConfig, input, gaps []float32, rate, mom float32) {
	avgGap := avgSlice(gaps)
	scaledRate := rate * avgGap * 0.001

	tweenWeightSlice(cfg.WeightIH, input, scaledRate)
	tweenWeightSlice(cfg.WeightHH, input, scaledRate)
	tweenBiasSlice(cfg.BiasH, avgGap, rate*0.1)
}

// tweenLSTM handles Long Short-Term Memory layers
func (ts *TweenState) tweenLSTM(cfg *LayerConfig, input, gaps []float32, rate, mom float32) {
	avgGap := avgSlice(gaps)
	scaledRate := rate * avgGap * 0.001

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

		if epoch%5 == 0 || epoch == epochs-1 {
			metrics, _ := n.EvaluateNetwork(inputs, expected)
			if metrics.Score > ts.BestScore {
				ts.BestScore = metrics.Score
				ts.SaveBest(n)
			}
			if callback != nil {
				callback(epoch+1, avgLoss, metrics)
			}
			if metrics.Score >= 95 {
				ts.RestoreBest(n)
				return
			}
		}
	}
	if ts.BestScore > 0 {
		ts.RestoreBest(n)
	}
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
