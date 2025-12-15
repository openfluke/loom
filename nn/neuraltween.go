package nn

import (
	"fmt"
	"math"
)

// TweenState tracks the interpolation progress for neural tweening
// Inspired by Flash ActionScript tweening, network link budgeting, and optimal transport
type TweenState struct {
	// Weight snapshots per layer
	SourceWeights [][]float32 // Starting weights (current state)
	TargetWeights [][]float32 // Estimated optimal weights (target state)

	// Bias snapshots
	SourceBiases [][]float32
	TargetBiases [][]float32

	// Progress and budgets
	Progress     float32   // 0.0-1.0 interpolation progress
	LayerBudgets []float32 // Information "budget" per layer (signal strength estimate)

	// Stats for analysis
	TotalLayers   int
	TweenSteps    int
	LastLoss      float32
	LossHistory   []float32
	BudgetHistory [][]float32 // Track budget evolution

	// Accumulated gradients for momentum
	MomentumWeights [][]float32
	MomentumBiases  [][]float32
}

// NewTweenState creates a new tween state from a network
func NewTweenState(n *Network) *TweenState {
	totalLayers := n.TotalLayers()

	ts := &TweenState{
		SourceWeights:   make([][]float32, totalLayers),
		TargetWeights:   make([][]float32, totalLayers),
		SourceBiases:    make([][]float32, totalLayers),
		TargetBiases:    make([][]float32, totalLayers),
		MomentumWeights: make([][]float32, totalLayers),
		MomentumBiases:  make([][]float32, totalLayers),
		LayerBudgets:    make([]float32, totalLayers),
		Progress:        0.0,
		TotalLayers:     totalLayers,
		TweenSteps:      0,
		LossHistory:     []float32{},
		BudgetHistory:   [][]float32{},
	}

	// Snapshot current weights as source and initialize momentum
	for i := 0; i < totalLayers; i++ {
		row := i / (n.GridCols * n.LayersPerCell)
		col := (i / n.LayersPerCell) % n.GridCols
		layer := i % n.LayersPerCell

		cfg := n.GetLayer(row, col, layer)
		if cfg == nil {
			continue
		}

		// Copy current weights as source
		if len(cfg.Kernel) > 0 {
			ts.SourceWeights[i] = make([]float32, len(cfg.Kernel))
			ts.MomentumWeights[i] = make([]float32, len(cfg.Kernel))
			copy(ts.SourceWeights[i], cfg.Kernel)
		}
		if len(cfg.Bias) > 0 {
			ts.SourceBiases[i] = make([]float32, len(cfg.Bias))
			ts.MomentumBiases[i] = make([]float32, len(cfg.Bias))
			copy(ts.SourceBiases[i], cfg.Bias)
		}

		// Initialize budget to 1.0 (full signal strength)
		ts.LayerBudgets[i] = 1.0
	}

	return ts
}

// TweenTrain performs training using the tween approach
// This is a hybrid: gradient-based updates with momentum + interpolation smoothing
func (ts *TweenState) TweenTrain(n *Network, inputs [][]float32, expected []float64,
	epochs int, learningRate float32, momentum float32, callback func(epoch int, loss float32, metrics *DeviationMetrics)) {

	batchSize := 32
	if len(inputs) < batchSize {
		batchSize = len(inputs)
	}

	for epoch := 0; epoch < epochs; epoch++ {
		epochLoss := float32(0)
		numBatches := 0

		// Mini-batch training
		for batchStart := 0; batchStart < len(inputs); batchStart += batchSize {
			batchEnd := batchStart + batchSize
			if batchEnd > len(inputs) {
				batchEnd = len(inputs)
			}

			batchLoss := ts.trainBatch(n, inputs[batchStart:batchEnd],
				expected[batchStart:batchEnd], learningRate, momentum)
			epochLoss += batchLoss
			numBatches++
		}

		epochLoss /= float32(numBatches)
		ts.LossHistory = append(ts.LossHistory, epochLoss)
		ts.LastLoss = epochLoss
		ts.TweenSteps++

		// Evaluate periodically
		if callback != nil && (epoch%5 == 0 || epoch == epochs-1) {
			metrics, _ := n.EvaluateNetwork(inputs, expected)
			callback(epoch+1, epochLoss, metrics)
		}
	}
}

// trainBatch performs one batch of training with momentum
func (ts *TweenState) trainBatch(n *Network, inputs [][]float32, expected []float64,
	lr float32, momentum float32) float32 {

	totalLayers := n.TotalLayers()
	totalLoss := float32(0)

	// Accumulate gradients over batch
	batchKernelGrads := make([][]float32, totalLayers)
	batchBiasGrads := make([][]float32, totalLayers)

	for i := 0; i < totalLayers; i++ {
		row := i / (n.GridCols * n.LayersPerCell)
		col := (i / n.LayersPerCell) % n.GridCols
		layer := i % n.LayersPerCell
		cfg := n.GetLayer(row, col, layer)
		if cfg != nil && len(cfg.Kernel) > 0 {
			batchKernelGrads[i] = make([]float32, len(cfg.Kernel))
			batchBiasGrads[i] = make([]float32, len(cfg.Bias))
		}
	}

	// Process each sample
	for s := range inputs {
		output, _ := n.ForwardCPU(inputs[s])

		// Compute loss and gradient
		errorGrad := make([]float32, len(output))
		for j := range output {
			target := float32(0)
			if j == int(expected[s]) {
				target = 1.0
			}
			// Softmax cross-entropy gradient
			prob := output[j]
			if prob > 0.999 {
				prob = 0.999
			}
			if prob < 0.001 {
				prob = 0.001
			}
			errorGrad[j] = prob - target

			// Cross-entropy loss
			if target > 0.5 {
				totalLoss -= float32(math.Log(float64(prob)))
			}
		}

		// Backward pass
		n.BackwardCPU(errorGrad)

		// Accumulate gradients
		kernelGrads := n.KernelGradients()
		biasGrads := n.BiasGradients()

		for i := 0; i < totalLayers; i++ {
			if i < len(kernelGrads) && len(kernelGrads[i]) > 0 && len(batchKernelGrads[i]) > 0 {
				for j := range batchKernelGrads[i] {
					if j < len(kernelGrads[i]) {
						batchKernelGrads[i][j] += kernelGrads[i][j]
					}
				}
			}
			if i < len(biasGrads) && len(biasGrads[i]) > 0 && len(batchBiasGrads[i]) > 0 {
				for j := range batchBiasGrads[i] {
					if j < len(biasGrads[i]) {
						batchBiasGrads[i][j] += biasGrads[i][j]
					}
				}
			}
		}
	}

	// Average gradients and apply with momentum (SGD with momentum)
	batchLen := float32(len(inputs))
	for i := 0; i < totalLayers; i++ {
		row := i / (n.GridCols * n.LayersPerCell)
		col := (i / n.LayersPerCell) % n.GridCols
		layer := i % n.LayersPerCell
		cfg := n.GetLayer(row, col, layer)
		if cfg == nil {
			continue
		}

		// Apply link budget scaling - layers with higher information flow get larger updates
		budget := ts.LayerBudgets[i]
		if budget < 0.1 {
			budget = 0.1
		}

		// Update kernel weights with momentum
		if len(cfg.Kernel) > 0 && len(batchKernelGrads[i]) > 0 && len(ts.MomentumWeights[i]) > 0 {
			for j := range cfg.Kernel {
				if j < len(batchKernelGrads[i]) && j < len(ts.MomentumWeights[i]) {
					grad := batchKernelGrads[i][j] / batchLen

					// Clip gradients
					if grad > 1.0 {
						grad = 1.0
					} else if grad < -1.0 {
						grad = -1.0
					}

					// Momentum update
					ts.MomentumWeights[i][j] = momentum*ts.MomentumWeights[i][j] + lr*grad*budget
					cfg.Kernel[j] -= ts.MomentumWeights[i][j]
				}
			}
		}

		// Update biases with momentum
		if len(cfg.Bias) > 0 && len(batchBiasGrads[i]) > 0 && len(ts.MomentumBiases[i]) > 0 {
			for j := range cfg.Bias {
				if j < len(batchBiasGrads[i]) && j < len(ts.MomentumBiases[i]) {
					grad := batchBiasGrads[i][j] / batchLen

					// Clip gradients
					if grad > 1.0 {
						grad = 1.0
					} else if grad < -1.0 {
						grad = -1.0
					}

					ts.MomentumBiases[i][j] = momentum*ts.MomentumBiases[i][j] + lr*grad
					cfg.Bias[j] -= ts.MomentumBiases[i][j]
				}
			}
		}

		n.SetLayer(row, col, layer, *cfg)
	}

	// Update link budgets periodically
	if ts.TweenSteps%10 == 0 && len(inputs) > 0 {
		ts.CalculateLinkBudgets(n, inputs[0])
	}

	return totalLoss / batchLen
}

// CalculateLinkBudgets computes information "budget" for each layer
// Similar to WiFi link budget: estimates signal strength/loss through each layer
func (ts *TweenState) CalculateLinkBudgets(n *Network, sampleInput []float32) {
	// Run forward pass and analyze activation statistics
	n.ForwardCPU(sampleInput)
	activations := n.Activations()

	totalLayers := n.TotalLayers()
	if len(ts.LayerBudgets) != totalLayers {
		ts.LayerBudgets = make([]float32, totalLayers)
	}

	// Calculate variance decay across layers (like signal attenuation)
	prevVariance := float32(1.0)

	for i := 0; i < totalLayers; i++ {
		if i+1 >= len(activations) || len(activations[i+1]) == 0 {
			ts.LayerBudgets[i] = 1.0 // Default budget
			continue
		}

		act := activations[i+1]

		// Calculate activation statistics
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

		// Variance ratio indicates information flow
		// High variance = high information, low variance = information loss
		if variance > 0 && prevVariance > 0 {
			ratio := variance / prevVariance
			// Clamp to reasonable range
			if ratio > 2.0 {
				ratio = 2.0
			}
			if ratio < 0.1 {
				ratio = 0.1
			}
			ts.LayerBudgets[i] = ratio
		} else {
			ts.LayerBudgets[i] = 1.0
		}

		prevVariance = variance
		if prevVariance < 0.01 {
			prevVariance = 0.01 // Prevent division by zero
		}
	}

	// Store budget history
	budgetCopy := make([]float32, len(ts.LayerBudgets))
	copy(budgetCopy, ts.LayerBudgets)
	ts.BudgetHistory = append(ts.BudgetHistory, budgetCopy)
}

// GetBudgetSummary returns a summary of link budgets
func (ts *TweenState) GetBudgetSummary() (avgBudget, minBudget, maxBudget float32) {
	if len(ts.LayerBudgets) == 0 {
		return 1.0, 1.0, 1.0
	}

	minBudget = ts.LayerBudgets[0]
	maxBudget = ts.LayerBudgets[0]
	sum := float32(0)

	for _, b := range ts.LayerBudgets {
		sum += b
		if b < minBudget {
			minBudget = b
		}
		if b > maxBudget {
			maxBudget = b
		}
	}

	avgBudget = sum / float32(len(ts.LayerBudgets))
	return
}

// Tween performs neural tween training and returns final metrics
func Tween(n *Network, inputs [][]float32, expected []float64, epochs int) (*DeviationMetrics, *TweenState) {
	ts := NewTweenState(n)

	// Initialize link budgets
	if len(inputs) > 0 {
		ts.CalculateLinkBudgets(n, inputs[0])
	}

	var finalMetrics *DeviationMetrics

	// Training parameters
	learningRate := float32(0.5)
	momentum := float32(0.9)

	ts.TweenTrain(n, inputs, expected, epochs, learningRate, momentum,
		func(epoch int, loss float32, metrics *DeviationMetrics) {
			finalMetrics = metrics
		})

	return finalMetrics, ts
}

// TweenWithVerbose performs neural tweening with detailed progress output
func TweenWithVerbose(n *Network, inputs [][]float32, expected []float64, epochs int,
	printFn func(string)) (*DeviationMetrics, *TweenState) {

	ts := NewTweenState(n)

	// Initialize link budgets
	if len(inputs) > 0 {
		ts.CalculateLinkBudgets(n, inputs[0])
	}

	var finalMetrics *DeviationMetrics

	// Training parameters - higher learning rate for faster convergence
	learningRate := float32(0.5)
	momentum := float32(0.9)

	ts.TweenTrain(n, inputs, expected, epochs, learningRate, momentum,
		func(epoch int, loss float32, metrics *DeviationMetrics) {
			finalMetrics = metrics
			avgBudget, minBudget, maxBudget := ts.GetBudgetSummary()

			progress := float32(epoch) / float32(epochs)
			bar := progressBarStr(progress, 20)

			msg := fmt.Sprintf(
				"Epoch %3d/%d [%s] | Score: %5.1f/100 | Loss: %6.3f | Budget: %.2f (%.2f-%.2f)",
				epoch, epochs, bar, metrics.Score, loss, avgBudget, minBudget, maxBudget)

			if printFn != nil {
				printFn(msg)
			}
		})

	return finalMetrics, ts
}

func progressBarStr(progress float32, width int) string {
	filled := int(progress * float32(width))
	bar := ""
	for i := 0; i < width; i++ {
		if i < filled {
			bar += "█"
		} else {
			bar += "░"
		}
	}
	return bar
}

// min helper for older Go versions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func formatTweenOutput(epoch, totalEpochs int, bar string, score float64, loss float32, avgBudget, minBudget, maxBudget float32) string {
	return "Epoch " + intToString(epoch) + "/" + intToString(totalEpochs) +
		" [" + bar + "] | Score: " + floatToString(score, 1) + "/100" +
		" | Loss: " + floatToString(float64(loss), 3) +
		" | Budget: " + floatToString(float64(avgBudget), 2)
}

func padLeft(s string, width int) string {
	for len(s) < width {
		s = " " + s
	}
	return s
}

func replaceFirst(s, old, new string) string {
	for i := 0; i <= len(s)-len(old); i++ {
		if s[i:i+len(old)] == old {
			return s[:i] + new + s[i+len(old):]
		}
	}
	return s
}

func intToString(n int) string {
	if n == 0 {
		return "0"
	}
	negative := n < 0
	if negative {
		n = -n
	}
	result := ""
	for n > 0 {
		result = string(rune('0'+n%10)) + result
		n /= 10
	}
	if negative {
		result = "-" + result
	}
	return result
}

func floatToString(f float64, precision int) string {
	if math.IsNaN(f) {
		return "NaN"
	}
	if math.IsInf(f, 1) {
		return "Inf"
	}
	if math.IsInf(f, -1) {
		return "-Inf"
	}

	negative := f < 0
	if negative {
		f = -f
	}

	// Round to precision
	mult := math.Pow(10, float64(precision))
	f = math.Round(f*mult) / mult

	intPart := int(f)
	fracPart := f - float64(intPart)

	result := intToString(intPart)

	if precision > 0 {
		result += "."
		fracPart *= mult
		fracStr := intToString(int(math.Round(fracPart)))
		// Pad with zeros if needed
		for len(fracStr) < precision {
			fracStr = "0" + fracStr
		}
		result += fracStr
	}

	if negative {
		result = "-" + result
	}
	return result
}
