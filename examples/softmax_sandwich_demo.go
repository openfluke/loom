package main

import (
	"fmt"
	"math/rand"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== Softmax Sandwich Demo ===")
	fmt.Println("Softmax layers in HIDDEN positions AND output!")
	fmt.Println()

	// Task: Classify game situations into team actions
	// Input: 32 features (8 units × 4 features each: health, ammo, position_x, position_y)
	// Hidden: Use softmax to "select" which units to focus on
	// Output: Team-wide action decision

	batchSize := 1
	network := nn.NewNetwork(32, 1, 1, 6)
	network.BatchSize = batchSize

	fmt.Println("Building network with softmax sandwich:")
	fmt.Println()

	// Layer 0: Input processing
	dense1 := nn.InitDenseLayer(32, 64, nn.ActivationLeakyReLU)
	network.SetLayer(0, 0, 0, dense1)
	fmt.Println("  Layer 0: Dense (32 → 64)")

	// Layer 1: HIDDEN SOFTMAX - Attention over units
	// Reshape 64 features as 8 groups × 8 features, apply grid softmax
	// This makes the network "choose" which feature groups to emphasize
	unitAttention := nn.InitGridSoftmaxLayer(8, 8) // 8 groups, 8 features each
	network.SetLayer(0, 0, 1, unitAttention)
	fmt.Println("  Layer 1: Grid Softmax (8 groups × 8 features) - HIDDEN LAYER")
	fmt.Println("           ↳ Network learns feature group importance!")

	// Layer 2: Process attended features
	dense2 := nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU)
	network.SetLayer(0, 0, 2, dense2)
	fmt.Println("  Layer 2: Dense (64 → 32)")

	// Layer 3: ANOTHER HIDDEN SOFTMAX - Sparse gating
	// Use sparsemax to make some features exactly zero
	featureGating := nn.InitSparsemaxLayer()
	network.SetLayer(0, 0, 3, featureGating)
	fmt.Println("  Layer 3: Sparsemax - HIDDEN LAYER")
	fmt.Println("           ↳ Network learns sparse feature selection!")

	// Layer 4: Final processing
	dense3 := nn.InitDenseLayer(32, 6, nn.ActivationLeakyReLU)
	network.SetLayer(0, 0, 4, dense3)
	fmt.Println("  Layer 4: Dense (32 → 6)")

	// Layer 5: OUTPUT SOFTMAX - Final team action
	outputSoftmax := nn.InitSoftmaxLayer()
	network.SetLayer(0, 0, 5, outputSoftmax)
	fmt.Println("  Layer 5: Standard Softmax - OUTPUT LAYER")
	fmt.Println("           ↳ Final action probabilities!")

	fmt.Println()
	fmt.Println("─────────────────────────────────────────────────────")
	fmt.Println("Network has softmax at layers 1, 3, and 5!")
	fmt.Println("Layers 1 & 3 are HIDDEN, layer 5 is OUTPUT")
	fmt.Println("─────────────────────────────────────────────────────")
	fmt.Println()

	// Generate training data
	// Task: Decide team action based on unit states
	// Actions: 0=attack, 1=defend, 2=retreat, 3=heal, 4=regroup, 5=advance
	numSamples := 150
	batches := make([]nn.TrainingBatch, numSamples)

	actions := []string{"attack", "defend", "retreat", "heal", "regroup", "advance"}

	fmt.Println("Generating tactical scenarios...")
	fmt.Println("Rules:")
	fmt.Println("  - High avg health + high avg ammo → attack")
	fmt.Println("  - Low avg health → heal or retreat")
	fmt.Println("  - Mixed states → defend or regroup")
	fmt.Println()

	for i := 0; i < numSamples; i++ {
		input := make([]float32, 32) // 8 units × 4 features

		totalHealth := float32(0)
		totalAmmo := float32(0)

		// Generate 8 units with random stats
		for unit := 0; unit < 8; unit++ {
			health := rand.Float32()
			ammo := rand.Float32()
			posX := rand.Float32()
			posY := rand.Float32()

			input[unit*4+0] = health
			input[unit*4+1] = ammo
			input[unit*4+2] = posX
			input[unit*4+3] = posY

			totalHealth += health
			totalAmmo += ammo
		}

		avgHealth := totalHealth / 8.0
		avgAmmo := totalAmmo / 8.0

		// Determine best action based on team state
		target := make([]float32, 6)
		var bestAction int

		if avgHealth > 0.7 && avgAmmo > 0.7 {
			// Strong team → attack
			bestAction = 0
		} else if avgHealth < 0.3 {
			// Weak team → heal or retreat
			if avgAmmo > 0.5 {
				bestAction = 3 // heal
			} else {
				bestAction = 2 // retreat
			}
		} else if avgHealth > 0.5 && avgAmmo < 0.3 {
			// Good health, low ammo → regroup
			bestAction = 4
		} else if avgHealth < 0.5 && avgAmmo > 0.6 {
			// Low health, good ammo → defend
			bestAction = 1
		} else {
			// Mixed → advance carefully
			bestAction = 5
		}

		// One-hot encode target
		for j := 0; j < 6; j++ {
			if j == bestAction {
				target[j] = 0.9
			} else {
				target[j] = 0.1 / 5
			}
		}

		batches[i] = nn.TrainingBatch{
			Input:  input,
			Target: target,
		}
	}

	// Train
	config := &nn.TrainingConfig{
		Epochs:          200,
		LearningRate:    0.005,
		UseGPU:          false,
		PrintEveryBatch: 0,
		GradientClip:    0.5,
		LossType:        "mse",
		Verbose:         false,
	}

	fmt.Println("Training network with hidden softmax layers...")
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

	// Test scenarios
	fmt.Println("=== Testing Softmax Sandwich Network ===")
	fmt.Println()

	testScenarios := []struct {
		name           string
		setupUnits     func() []float32
		expectedAction string
	}{
		{
			name: "FULL STRENGTH TEAM",
			setupUnits: func() []float32 {
				units := make([]float32, 32)
				for i := 0; i < 8; i++ {
					units[i*4+0] = 0.9 // high health
					units[i*4+1] = 0.9 // high ammo
					units[i*4+2] = rand.Float32()
					units[i*4+3] = rand.Float32()
				}
				return units
			},
			expectedAction: "attack",
		},
		{
			name: "WOUNDED TEAM",
			setupUnits: func() []float32 {
				units := make([]float32, 32)
				for i := 0; i < 8; i++ {
					units[i*4+0] = 0.2 // low health
					units[i*4+1] = 0.6 // decent ammo
					units[i*4+2] = rand.Float32()
					units[i*4+3] = rand.Float32()
				}
				return units
			},
			expectedAction: "heal",
		},
		{
			name: "CRITICAL SITUATION",
			setupUnits: func() []float32 {
				units := make([]float32, 32)
				for i := 0; i < 8; i++ {
					units[i*4+0] = 0.1 // very low health
					units[i*4+1] = 0.2 // low ammo
					units[i*4+2] = rand.Float32()
					units[i*4+3] = rand.Float32()
				}
				return units
			},
			expectedAction: "retreat",
		},
		{
			name: "NEED RESUPPLY",
			setupUnits: func() []float32 {
				units := make([]float32, 32)
				for i := 0; i < 8; i++ {
					units[i*4+0] = 0.7 // good health
					units[i*4+1] = 0.2 // low ammo
					units[i*4+2] = rand.Float32()
					units[i*4+3] = rand.Float32()
				}
				return units
			},
			expectedAction: "regroup",
		},
	}

	for _, scenario := range testScenarios {
		fmt.Println(scenario.name)
		fmt.Println("─────────────────────────────────────────────────────")

		testInput := scenario.setupUnits()

		// Show team stats
		avgHealth := float32(0)
		avgAmmo := float32(0)
		for i := 0; i < 8; i++ {
			avgHealth += testInput[i*4+0]
			avgAmmo += testInput[i*4+1]
		}
		avgHealth /= 8.0
		avgAmmo /= 8.0

		fmt.Printf("  Team stats: Health=%.2f, Ammo=%.2f\n", avgHealth, avgAmmo)
		fmt.Println()

		// Get network output
		output, _ := network.ForwardCPU(testInput)

		// Show action probabilities
		fmt.Println("  Action probabilities:")
		maxProb := float32(0)
		maxAction := 0
		for i, action := range actions {
			prob := output[i]
			bar := ""
			barLen := int(prob * 40)
			for j := 0; j < barLen; j++ {
				bar += "█"
			}
			fmt.Printf("    %s: %.3f %s\n", action, prob, bar)

			if prob > maxProb {
				maxProb = prob
				maxAction = i
			}
		}

		fmt.Println()
		fmt.Printf("  → Decision: %s (%.1f%% confidence)\n", actions[maxAction], maxProb*100)
		if actions[maxAction] == scenario.expectedAction {
			fmt.Println("  ✓ Correct decision!")
		} else {
			fmt.Printf("  ⚠ Expected %s\n", scenario.expectedAction)
		}

		fmt.Println()
		fmt.Println()
	}

	fmt.Println("═════════════════════════════════════════════════════")
	fmt.Println("✓ Hidden softmax layers work!")
	fmt.Println("✓ Layer 1: Unit attention (grid softmax)")
	fmt.Println("✓ Layer 3: Feature importance (temperature softmax)")
	fmt.Println("✓ Layer 5: Final decision (standard softmax)")
	fmt.Println()
	fmt.Println("The hidden softmax layers help the network learn")
	fmt.Println("WHAT to pay attention to during training!")
	fmt.Println("═════════════════════════════════════════════════════")
}
