package main

import (
	"fmt"
	"math/rand"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== Multi-Softmax Network Demo ===")
	fmt.Println("One network with MULTIPLE different softmax layers!")
	fmt.Println()

	// Build a game AI network with different softmax for different decisions
	// Input: 64 features (game state)
	// Output: Multiple decision heads

	batchSize := 1
	network := nn.NewNetwork(64, 1, 1, 10)
	network.BatchSize = batchSize

	fmt.Println("Building multi-head decision network...")
	fmt.Println()

	// ===== SHARED BACKBONE =====
	fmt.Println("SHARED LAYERS:")

	// Layer 0: Input processing
	dense1 := nn.InitDenseLayer(64, 64, nn.ActivationLeakyReLU)
	network.SetLayer(0, 0, 0, dense1)
	fmt.Println("  Layer 0: Dense (64 → 64) - Feature extraction")

	// Layer 1: Deep processing
	dense2 := nn.InitDenseLayer(64, 48, nn.ActivationLeakyReLU)
	network.SetLayer(0, 0, 1, dense2)
	fmt.Println("  Layer 1: Dense (64 → 48) - Deep features")

	// Layer 2: Attention-like gating with SPARSEMAX (only some features active)
	sparseGate := nn.InitSparsemaxLayer()
	network.SetLayer(0, 0, 2, sparseGate)
	fmt.Println("  Layer 2: Sparsemax - Sparse feature gating (can output zeros)")

	fmt.Println()
	fmt.Println("DECISION HEAD 1: Unit Movement (Grid Softmax)")

	// Layer 3: Movement feature extraction
	moveDense := nn.InitDenseLayer(48, 12, nn.ActivationScaledReLU)
	network.SetLayer(0, 0, 3, moveDense)
	fmt.Println("  Layer 3: Dense (48 → 12)")

	// Layer 4: Multi-agent movement decisions (GRID SOFTMAX)
	moveSoftmax := nn.InitGridSoftmaxLayer(3, 4) // 3 units, 4 directions each
	network.SetLayer(0, 0, 4, moveSoftmax)
	fmt.Println("  Layer 4: Grid Softmax (3 units × 4 moves) - Independent decisions")

	fmt.Println()
	fmt.Println("DECISION HEAD 2: Ability Selection (Masked Softmax)")

	// Layer 5: Ability features
	abilityDense := nn.InitDenseLayer(12, 6, nn.ActivationScaledReLU)
	network.SetLayer(0, 0, 5, abilityDense)
	fmt.Println("  Layer 5: Dense (12 → 6)")

	// Layer 6: Ability selection with cooldown masking (MASKED SOFTMAX)
	abilityMasked := nn.InitMaskedSoftmaxLayer(6)
	// Simulate: abilities 1 and 4 are on cooldown
	abilityMasked.Mask = []bool{true, false, true, true, false, true}
	network.SetLayer(0, 0, 6, abilityMasked)
	fmt.Println("  Layer 6: Masked Softmax (6 abilities) - Filters cooldowns")

	fmt.Println()
	fmt.Println("DECISION HEAD 3: Strategy Selection (Hierarchical Softmax)")

	// Layer 7: Strategy features
	stratDense := nn.InitDenseLayer(6, 12, nn.ActivationScaledReLU)
	network.SetLayer(0, 0, 7, stratDense)
	fmt.Println("  Layer 7: Dense (6 → 12)")

	// Layer 8: Hierarchical strategy (HIERARCHICAL SOFTMAX)
	// 3 macro strategies × 4 micro tactics = 12 outputs
	stratSoftmax := nn.InitHierarchicalSoftmaxLayer([]int{3, 4})
	network.SetLayer(0, 0, 8, stratSoftmax)
	fmt.Println("  Layer 8: Hierarchical Softmax (3 strategies × 4 tactics)")

	fmt.Println()
	fmt.Println("DECISION HEAD 4: Exploration (Temperature Softmax)")

	// Layer 9: Exploration decisions with adjustable temperature (TEMPERATURE SOFTMAX)
	// Low temp = exploit (confident), high temp = explore (uncertain)
	exploreSoftmax := nn.InitTemperatureSoftmaxLayer(2.0) // High temp = more exploration
	network.SetLayer(0, 0, 9, exploreSoftmax)
	fmt.Println("  Layer 9: Temperature Softmax (temp=2.0) - Exploratory output")

	fmt.Println()
	fmt.Println("─────────────────────────────────────────────────────")
	fmt.Println("Network has 4 DIFFERENT softmax types in 10 layers!")
	fmt.Println("─────────────────────────────────────────────────────")
	fmt.Println()

	// Generate training data
	numSamples := 80
	batches := make([]nn.TrainingBatch, numSamples)

	fmt.Println("Generating training scenarios...")

	for i := 0; i < numSamples; i++ {
		input := make([]float32, 64)

		// Random game state
		for j := range input {
			input[j] = rand.Float32()
		}

		// Target: final output is 12 values (from layer 9 which reshapes from hierarchical)
		target := make([]float32, 12)

		// Create reasonable targets based on input patterns
		if input[0] > 0.5 {
			// Aggressive scenario
			for j := 0; j < 12; j++ {
				if j < 4 {
					target[j] = 0.8 // First strategy highly weighted
				} else {
					target[j] = 0.2 / 8
				}
			}
		} else {
			// Defensive scenario
			for j := 0; j < 12; j++ {
				if j >= 4 && j < 8 {
					target[j] = 0.8 // Second strategy highly weighted
				} else {
					target[j] = 0.2 / 8
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
		Epochs:          60,
		LearningRate:    0.01,
		UseGPU:          false,
		PrintEveryBatch: 0,
		GradientClip:    1.0,
		LossType:        "mse",
		Verbose:         false,
	}

	fmt.Println("Training multi-softmax network...")
	result, err := network.Train(batches, config)
	if err != nil {
		fmt.Printf("Training failed: %v\n", err)
		return
	}

	fmt.Println()
	fmt.Printf("✓ Training complete!\n")
	fmt.Printf("  Loss: %.6f → %.6f (%.1f%% reduction)\n",
		result.LossHistory[0], result.FinalLoss,
		100*(result.LossHistory[0]-result.FinalLoss)/result.LossHistory[0])
	fmt.Println()

	// Test the multi-softmax network
	fmt.Println("=== Testing Multi-Softmax Decisions ===")
	fmt.Println()

	testScenarios := []struct {
		name  string
		setup func([]float32)
	}{
		{
			name: "AGGRESSIVE SITUATION (high threat)",
			setup: func(input []float32) {
				input[0] = 0.9 // High aggression trigger
				for i := 1; i < 64; i++ {
					input[i] = rand.Float32() * 0.5
				}
			},
		},
		{
			name: "DEFENSIVE SITUATION (low threat)",
			setup: func(input []float32) {
				input[0] = 0.1 // Low aggression trigger
				for i := 1; i < 64; i++ {
					input[i] = rand.Float32() * 0.5
				}
			},
		},
	}

	for _, scenario := range testScenarios {
		fmt.Println(scenario.name)
		fmt.Println("─────────────────────────────────────────────────────")

		testInput := make([]float32, 64)
		scenario.setup(testInput)

		// Build separate test networks for each decision head
		// (since activations are private)

		fmt.Println()
		fmt.Println("📊 Decision Head 1: Unit Movement (Grid Softmax)")
		moveNet := nn.NewNetwork(64, 1, 1, 5)
		moveNet.SetLayer(0, 0, 0, dense1)
		moveNet.SetLayer(0, 0, 1, dense2)
		moveNet.SetLayer(0, 0, 2, sparseGate)
		moveNet.SetLayer(0, 0, 3, moveDense)
		moveNet.SetLayer(0, 0, 4, moveSoftmax)

		moveOutput, _ := moveNet.ForwardCPU(testInput)
		moves := []string{"north", "east", "south", "west"}
		for unit := 0; unit < 3; unit++ {
			fmt.Printf("  Unit %d: ", unit)
			maxProb := float32(0)
			maxMove := 0
			for move := 0; move < 4; move++ {
				idx := unit*4 + move
				if idx < len(moveOutput) {
					fmt.Printf("%s=%.2f ", moves[move], moveOutput[idx])
					if moveOutput[idx] > maxProb {
						maxProb = moveOutput[idx]
						maxMove = move
					}
				}
			}
			fmt.Printf("→ %s\n", moves[maxMove])
		}

		fmt.Println()
		fmt.Println("📊 Decision Head 2: Ability Selection (Masked Softmax)")
		abilityNet := nn.NewNetwork(64, 1, 1, 7)
		abilityNet.SetLayer(0, 0, 0, dense1)
		abilityNet.SetLayer(0, 0, 1, dense2)
		abilityNet.SetLayer(0, 0, 2, sparseGate)
		abilityNet.SetLayer(0, 0, 3, moveDense)
		abilityNet.SetLayer(0, 0, 4, moveSoftmax)
		abilityNet.SetLayer(0, 0, 5, abilityDense)
		abilityNet.SetLayer(0, 0, 6, abilityMasked)

		abilityOutput, _ := abilityNet.ForwardCPU(testInput)
		abilities := []string{"heal", "shield", "damage", "speed", "stun", "teleport"}
		fmt.Print("  Abilities: ")
		maxProb := float32(0)
		maxAbility := 0
		for i := 0; i < 6 && i < len(abilityOutput); i++ {
			fmt.Printf("%s=%.2f ", abilities[i], abilityOutput[i])
			if abilityOutput[i] > maxProb {
				maxProb = abilityOutput[i]
				maxAbility = i
			}
		}
		fmt.Printf("\n  → %s\n", abilities[maxAbility])
		fmt.Println("  Note: 'shield' and 'stun' should be ~0 (masked as on cooldown)")

		fmt.Println()
		fmt.Println("📊 Decision Head 3: Strategy (Hierarchical Softmax)")
		stratNet := nn.NewNetwork(64, 1, 1, 9)
		stratNet.SetLayer(0, 0, 0, dense1)
		stratNet.SetLayer(0, 0, 1, dense2)
		stratNet.SetLayer(0, 0, 2, sparseGate)
		stratNet.SetLayer(0, 0, 3, moveDense)
		stratNet.SetLayer(0, 0, 4, moveSoftmax)
		stratNet.SetLayer(0, 0, 5, abilityDense)
		stratNet.SetLayer(0, 0, 6, abilityMasked)
		stratNet.SetLayer(0, 0, 7, stratDense)
		stratNet.SetLayer(0, 0, 8, stratSoftmax)

		stratOutput, _ := stratNet.ForwardCPU(testInput)
		strategies := []string{"attack", "defend", "scout"}
		tactics := []string{"rush", "flank", "ambush", "retreat"}

		// Show each strategy's tactics
		for s := 0; s < 3; s++ {
			fmt.Printf("  %s: ", strategies[s])
			for t := 0; t < 4; t++ {
				idx := s*4 + t
				if idx < len(stratOutput) {
					fmt.Printf("%s=%.2f ", tactics[t], stratOutput[idx])
				}
			}
			fmt.Println()
		}

		// Find best overall
		maxProb = float32(0)
		maxStrat := 0
		maxTactic := 0
		for s := 0; s < 3; s++ {
			for t := 0; t < 4; t++ {
				idx := s*4 + t
				if idx < len(stratOutput) && stratOutput[idx] > maxProb {
					maxProb = stratOutput[idx]
					maxStrat = s
					maxTactic = t
				}
			}
		}
		fmt.Printf("  → %s + %s (%.2f)\n", strategies[maxStrat], tactics[maxTactic], maxProb)

		fmt.Println()
		fmt.Println("📊 Final Output (Temperature Softmax, temp=2.0)")
		output, _ := network.ForwardCPU(testInput)
		fmt.Print("  [")
		for i := 0; i < len(output) && i < 12; i++ {
			if i > 0 {
				fmt.Print(", ")
			}
			fmt.Printf("%.3f", output[i])
		}
		fmt.Println("]")
		fmt.Printf("  (Smoother distribution from high temperature)\n")

		fmt.Println()
		fmt.Println()
	}

	fmt.Println("═════════════════════════════════════════════════════")
	fmt.Println("✓ One network, 4 DIFFERENT softmax types!")
	fmt.Println("✓ Grid Softmax: Multi-agent movement")
	fmt.Println("✓ Masked Softmax: Ability cooldowns")
	fmt.Println("✓ Hierarchical Softmax: Strategy + tactics")
	fmt.Println("✓ Temperature Softmax: Exploration control")
	fmt.Println()
	fmt.Println("Each softmax layer serves a different purpose!")
	fmt.Println("═════════════════════════════════════════════════════")
}
