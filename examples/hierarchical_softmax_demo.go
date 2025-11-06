package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== Hierarchical Softmax Demo ===")
	fmt.Println("Network outputs multi-level decisions:")
	fmt.Println("  Level 1: Strategy (attack/defend/scout)")
	fmt.Println("  Level 2: Which unit executes it")
	fmt.Println("  Level 3: What action the unit performs")
	fmt.Println()

	// Hierarchical structure
	numStrategies := 3    // attack, defend, scout
	unitsPerStrategy := 3 // 3 units can execute each strategy
	actionsPerUnit := 4   // move, shoot, ability, idle

	totalOutputs := numStrategies * unitsPerStrategy * actionsPerUnit // 36

	fmt.Printf("Output structure: %d strategies × %d units × %d actions = %d outputs\n",
		numStrategies, unitsPerStrategy, actionsPerUnit, totalOutputs)
	fmt.Println()

	// Build network
	batchSize := 1
	network := nn.NewNetwork(64, 1, 1, 4)
	network.BatchSize = batchSize

	fmt.Println("Building network...")

	// Simple 4-layer network
	dense0 := nn.InitDenseLayer(64, 64, nn.ActivationLeakyReLU)
	network.SetLayer(0, 0, 0, dense0)

	dense1 := nn.InitDenseLayer(64, 48, nn.ActivationLeakyReLU)
	network.SetLayer(0, 0, 1, dense1)

	dense2 := nn.InitDenseLayer(48, 36, nn.ActivationLeakyReLU)
	network.SetLayer(0, 0, 2, dense2)

	// Output layer: 36 values arranged hierarchically
	outputLayer := nn.InitDenseLayer(36, totalOutputs, nn.ActivationScaledReLU)
	network.SetLayer(0, 0, 3, outputLayer)

	fmt.Printf("  Output layer: %d values (3-level hierarchy)\n", totalOutputs)
	fmt.Println()

	// Generate training data
	numSamples := 100
	batches := make([]nn.TrainingBatch, numSamples)

	strategies := []string{"attack", "defend", "scout"}
	actions := []string{"move", "shoot", "ability", "idle"}

	fmt.Println("Generating training scenarios...")

	for i := 0; i < numSamples; i++ {
		input := make([]float32, 64)

		// Random game state
		for j := range input {
			input[j] = rand.Float32()
		}

		target := make([]float32, totalOutputs)

		// Determine strategy based on input features
		var chosenStrategy int
		if input[0] > 0.66 {
			chosenStrategy = 0 // attack
		} else if input[0] > 0.33 {
			chosenStrategy = 1 // defend
		} else {
			chosenStrategy = 2 // scout
		}

		// For the chosen strategy, set up unit-action preferences
		strategyStart := chosenStrategy * unitsPerStrategy * actionsPerUnit

		for unit := 0; unit < unitsPerStrategy; unit++ {
			unitStart := strategyStart + unit*actionsPerUnit

			// Each unit has a preferred action based on its position
			preferredAction := unit % actionsPerUnit

			for action := 0; action < actionsPerUnit; action++ {
				if action == preferredAction {
					target[unitStart+action] = 0.8
				} else {
					target[unitStart+action] = 0.2 / float32(actionsPerUnit-1)
				}
			}
		}

		batches[i] = nn.TrainingBatch{
			Input:  input,
			Target: target,
		}
	}

	// Train
	config := &nn.TrainingConfig{
		Epochs:          80,
		LearningRate:    0.01,
		UseGPU:          false,
		PrintEveryBatch: 0,
		GradientClip:    1.0,
		LossType:        "mse",
		Verbose:         false,
	}

	fmt.Println("Training hierarchical network...")
	result, err := network.Train(batches, config)
	if err != nil {
		fmt.Printf("Training failed: %v\n", err)
		return
	}

	fmt.Println()
	fmt.Printf("✓ Training complete!\n")
	fmt.Printf("  Loss: %.6f -> %.6f (%.1f%% reduction)\n",
		result.LossHistory[0], result.FinalLoss,
		100*(result.LossHistory[0]-result.FinalLoss)/result.LossHistory[0])
	fmt.Println()

	// Test hierarchical decision making
	fmt.Println("=== Testing Hierarchical Decisions ===")
	fmt.Println()

	testScenarios := []struct {
		name  string
		value float32
	}{
		{"Enemy spotted - HIGH THREAT", 0.8},
		{"Base under attack - MEDIUM THREAT", 0.5},
		{"All clear - LOW THREAT", 0.1},
	}

	for _, scenario := range testScenarios {
		fmt.Println("Scenario:", scenario.name)
		fmt.Println(strings.Repeat("-", 60))

		testInput := make([]float32, 64)
		testInput[0] = scenario.value
		for i := 1; i < 64; i++ {
			testInput[i] = rand.Float32() * 0.3
		}

		// Get network output
		logits, _ := network.ForwardCPU(testInput)

		// Apply hierarchical softmax
		fmt.Println()
		fmt.Println("LEVEL 1: Strategy Selection")
		strategyLogits := make([]float32, numStrategies)
		for s := 0; s < numStrategies; s++ {
			// Average the logits for this strategy's section
			start := s * unitsPerStrategy * actionsPerUnit
			end := start + unitsPerStrategy*actionsPerUnit
			sum := float32(0)
			for i := start; i < end; i++ {
				sum += logits[i]
			}
			strategyLogits[s] = sum / float32(unitsPerStrategy*actionsPerUnit)
		}
		strategyProbs := softmax(strategyLogits)

		for s, strategy := range strategies {
			bar := strings.Repeat("█", int(strategyProbs[s]*50))
			fmt.Printf("  %s: %.1f%% %s\n", strategy, strategyProbs[s]*100, bar)
		}

		chosenStrategy := argmax(strategyProbs)
		fmt.Printf("  → Chosen: %s\n", strategies[chosenStrategy])

		// Level 2: Unit selection within chosen strategy
		fmt.Println()
		fmt.Println("LEVEL 2: Unit Assignment (for", strategies[chosenStrategy], "strategy)")
		strategyStart := chosenStrategy * unitsPerStrategy * actionsPerUnit

		unitLogits := make([]float32, unitsPerStrategy)
		for u := 0; u < unitsPerStrategy; u++ {
			unitStart := strategyStart + u*actionsPerUnit
			sum := float32(0)
			for i := 0; i < actionsPerUnit; i++ {
				sum += logits[unitStart+i]
			}
			unitLogits[u] = sum / float32(actionsPerUnit)
		}
		unitProbs := softmax(unitLogits)

		for u := 0; u < unitsPerStrategy; u++ {
			bar := strings.Repeat("█", int(unitProbs[u]*50))
			fmt.Printf("  Unit %d: %.1f%% %s\n", u, unitProbs[u]*100, bar)
		}

		chosenUnit := argmax(unitProbs)
		fmt.Printf("  → Chosen: Unit %d\n", chosenUnit)

		// Level 3: Action selection for chosen unit
		fmt.Println()
		fmt.Println("LEVEL 3: Action Selection (for Unit", chosenUnit, ")")
		unitStart := strategyStart + chosenUnit*actionsPerUnit
		actionLogits := logits[unitStart : unitStart+actionsPerUnit]
		actionProbs := softmax(actionLogits)

		for a, action := range actions {
			bar := strings.Repeat("█", int(actionProbs[a]*50))
			fmt.Printf("  %s: %.1f%% %s\n", action, actionProbs[a]*100, bar)
		}

		chosenAction := argmax(actionProbs)
		fmt.Printf("  → Chosen: %s\n", actions[chosenAction])

		fmt.Println()
		fmt.Printf("📋 Final Decision: %s → Unit %d → %s\n",
			strategies[chosenStrategy], chosenUnit, actions[chosenAction])
		fmt.Println()
		fmt.Println()
	}

	fmt.Println("=== Alternative: Flat Grid Softmax ===")
	fmt.Println()
	fmt.Println("Instead of 3 levels, treat as 9 units × 4 actions:")

	testInput := make([]float32, 64)
	testInput[0] = 0.75
	for i := 1; i < 64; i++ {
		testInput[i] = rand.Float32() * 0.3
	}

	logits, _ := network.ForwardCPU(testInput)

	// Treat as 9 strategy-unit combos, each with 4 actions
	numCombos := numStrategies * unitsPerStrategy
	probs := softmaxGrid(logits, numCombos, actionsPerUnit)

	fmt.Println("All 9 strategy-unit combinations:")
	for s := 0; s < numStrategies; s++ {
		for u := 0; u < unitsPerStrategy; u++ {
			comboIdx := s*unitsPerStrategy + u
			start := comboIdx * actionsPerUnit
			end := start + actionsPerUnit
			comboProbs := probs[start:end]

			chosenAction := argmax(comboProbs)
			fmt.Printf("  %s-Unit%d: %s (%.1f%%)\n",
				strategies[s], u, actions[chosenAction], comboProbs[chosenAction]*100)
		}
	}

	fmt.Println()
	fmt.Println("=== Key Insights ===")
	fmt.Println("✓ One 36-output layer can represent 3-level hierarchy")
	fmt.Println("✓ Hierarchical softmax: separate probability distributions per level")
	fmt.Println("✓ Grid softmax: treat all combos independently")
	fmt.Println("✓ Choose approach based on whether levels are independent or sequential")
}

func argmax(v []float32) int {
	maxIdx := 0
	maxVal := v[0]
	for i, val := range v {
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}
	return maxIdx
}

func softmax(logits []float32) []float32 {
	maxLogit := logits[0]
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}

	exps := make([]float32, len(logits))
	sum := float32(0.0)
	for i, v := range logits {
		exps[i] = float32(math.Exp(float64(v - maxLogit)))
		sum += exps[i]
	}

	probs := make([]float32, len(logits))
	for i := range exps {
		probs[i] = exps[i] / sum
	}

	return probs
}

func softmaxGrid(logits []float32, rows, cols int) []float32 {
	result := make([]float32, len(logits))

	for r := 0; r < rows; r++ {
		rowStart := r * cols
		rowEnd := rowStart + cols
		rowSlice := logits[rowStart:rowEnd]
		rowProbs := softmax(rowSlice)
		copy(result[rowStart:rowEnd], rowProbs)
	}

	return result
}
