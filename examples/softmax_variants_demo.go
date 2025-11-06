package main

import (
	"fmt"
	"math/rand"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== LOOM Softmax Layer - All 10 Variants ===")
	fmt.Println()

	// Create test logits
	testLogits := []float32{2.0, 1.0, 0.1, 3.5, 0.5, 1.5}

	fmt.Println("Input logits:", testLogits)
	fmt.Println()

	// Test each variant
	fmt.Println("─────────────────────────────────────────────────────")
	fmt.Println("1. STANDARD SOFTMAX")
	fmt.Println("   One probability distribution over all values")
	fmt.Println("─────────────────────────────────────────────────────")

	network1 := nn.NewNetwork(6, 1, 1, 1)
	network1.SetLayer(0, 0, 0, nn.InitSoftmaxLayer())

	output1, _ := network1.ForwardCPU(testLogits)
	fmt.Printf("   Output: ")
	printArray(output1)
	fmt.Printf("   Sum: %.4f\n", sum(output1))
	fmt.Println()

	// Grid softmax
	fmt.Println("─────────────────────────────────────────────────────")
	fmt.Println("2. GRID SOFTMAX (Multi-Agent)")
	fmt.Println("   3 agents × 2 actions = independent distributions")
	fmt.Println("─────────────────────────────────────────────────────")

	network2 := nn.NewNetwork(6, 1, 1, 1)
	network2.SetLayer(0, 0, 0, nn.InitGridSoftmaxLayer(3, 2)) // 3 agents, 2 actions each

	output2, _ := network2.ForwardCPU(testLogits)
	fmt.Println("   Agent 0:", output2[0:2], "sum:", sum(output2[0:2]))
	fmt.Println("   Agent 1:", output2[2:4], "sum:", sum(output2[2:4]))
	fmt.Println("   Agent 2:", output2[4:6], "sum:", sum(output2[4:6]))
	fmt.Println()

	// Hierarchical softmax
	fmt.Println("─────────────────────────────────────────────────────")
	fmt.Println("3. HIERARCHICAL SOFTMAX")
	fmt.Println("   2 strategies × 3 actions = nested decisions")
	fmt.Println("─────────────────────────────────────────────────────")

	network3 := nn.NewNetwork(6, 1, 1, 1)
	network3.SetLayer(0, 0, 0, nn.InitHierarchicalSoftmaxLayer([]int{2, 3}))

	output3, _ := network3.ForwardCPU(testLogits)
	fmt.Println("   Strategy 0 actions:", output3[0:3], "sum:", sum(output3[0:3]))
	fmt.Println("   Strategy 1 actions:", output3[3:6], "sum:", sum(output3[3:6]))
	fmt.Println()

	// Temperature softmax - low temperature (sharp)
	fmt.Println("─────────────────────────────────────────────────────")
	fmt.Println("4. TEMPERATURE SOFTMAX (temp=0.1) - Sharp/Confident")
	fmt.Println("   Low temperature makes distribution peaked")
	fmt.Println("─────────────────────────────────────────────────────")

	network4 := nn.NewNetwork(6, 1, 1, 1)
	network4.SetLayer(0, 0, 0, nn.InitTemperatureSoftmaxLayer(0.1))

	output4, _ := network4.ForwardCPU(testLogits)
	fmt.Printf("   Output: ")
	printArray(output4)
	fmt.Printf("   Max prob: %.4f (very confident!)\n", max(output4))
	fmt.Println()

	// Temperature softmax - high temperature (smooth)
	fmt.Println("─────────────────────────────────────────────────────")
	fmt.Println("5. TEMPERATURE SOFTMAX (temp=5.0) - Smooth/Exploratory")
	fmt.Println("   High temperature makes distribution uniform")
	fmt.Println("─────────────────────────────────────────────────────")

	network5 := nn.NewNetwork(6, 1, 1, 1)
	network5.SetLayer(0, 0, 0, nn.InitTemperatureSoftmaxLayer(5.0))

	output5, _ := network5.ForwardCPU(testLogits)
	fmt.Printf("   Output: ")
	printArray(output5)
	fmt.Printf("   Max prob: %.4f (more uniform)\n", max(output5))
	fmt.Println()

	// Gumbel softmax
	fmt.Println("─────────────────────────────────────────────────────")
	fmt.Println("6. GUMBEL SOFTMAX (adds exploration noise)")
	fmt.Println("   Each forward pass adds random Gumbel noise")
	fmt.Println("─────────────────────────────────────────────────────")

	network6 := nn.NewNetwork(6, 1, 1, 1)
	network6.SetLayer(0, 0, 0, nn.InitGumbelSoftmaxLayer(1.0))

	fmt.Println("   Run 1:")
	output6a, _ := network6.ForwardCPU(testLogits)
	printArray(output6a)

	fmt.Println("   Run 2:")
	output6b, _ := network6.ForwardCPU(testLogits)
	printArray(output6b)

	fmt.Println("   Run 3:")
	output6c, _ := network6.ForwardCPU(testLogits)
	printArray(output6c)
	fmt.Println("   (Notice different outputs each time)")
	fmt.Println()

	// Masked softmax
	fmt.Println("─────────────────────────────────────────────────────")
	fmt.Println("7. MASKED SOFTMAX (legal moves only)")
	fmt.Println("   Mask positions [true, false, true, false, true, true]")
	fmt.Println("─────────────────────────────────────────────────────")

	network7 := nn.NewNetwork(6, 1, 1, 1)
	softmaxLayer := nn.InitMaskedSoftmaxLayer(6)
	softmaxLayer.Mask = []bool{true, false, true, false, true, true} // Only positions 0,2,4,5 allowed
	network7.SetLayer(0, 0, 0, softmaxLayer)

	output7, _ := network7.ForwardCPU(testLogits)
	fmt.Printf("   Output: ")
	printArray(output7)
	fmt.Println("   Masked positions (1, 3) are effectively zero!")
	fmt.Println()

	// Sparsemax
	fmt.Println("─────────────────────────────────────────────────────")
	fmt.Println("8. SPARSEMAX (can output exact zeros)")
	fmt.Println("   Unlike softmax, small values become exactly 0")
	fmt.Println("─────────────────────────────────────────────────────")

	network8 := nn.NewNetwork(6, 1, 1, 1)
	network8.SetLayer(0, 0, 0, nn.InitSparsemaxLayer())

	output8, _ := network8.ForwardCPU(testLogits)
	fmt.Printf("   Output: ")
	printArray(output8)
	fmt.Printf("   Sum: %.4f (exact zeros present!)\n", sum(output8))
	fmt.Println()

	// Entmax (alpha=1.5)
	fmt.Println("─────────────────────────────────────────────────────")
	fmt.Println("9. ENTMAX (alpha=1.5) - Between softmax and sparsemax")
	fmt.Println("   alpha=1.0→softmax, alpha=1.5→moderate, alpha=2.0→sparsemax")
	fmt.Println("─────────────────────────────────────────────────────")

	network9 := nn.NewNetwork(6, 1, 1, 1)
	network9.SetLayer(0, 0, 0, nn.InitEntmaxLayer(1.5))

	output9, _ := network9.ForwardCPU(testLogits)
	fmt.Printf("   Output: ")
	printArray(output9)
	fmt.Printf("   Sum: %.4f\n", sum(output9))
	fmt.Println()

	// Game AI Example: Multi-agent with temperature and masking
	fmt.Println("═════════════════════════════════════════════════════")
	fmt.Println("10. PRACTICAL GAME AI EXAMPLE")
	fmt.Println("    3 units, 4 actions each, with illegal moves masked")
	fmt.Println("═════════════════════════════════════════════════════")

	// Build a small network that outputs 12 values (3 units × 4 actions)
	gameNetwork := nn.NewNetwork(64, 1, 1, 3)
	gameNetwork.BatchSize = 1

	// Layer 0: Dense preprocessing
	dense := nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU)
	gameNetwork.SetLayer(0, 0, 0, dense)

	// Layer 1: Dense to 12 outputs
	output := nn.InitDenseLayer(32, 12, nn.ActivationScaledReLU)
	gameNetwork.SetLayer(0, 0, 1, output)

	// Layer 2: Grid softmax for multi-agent
	gridSoftmax := nn.InitGridSoftmaxLayer(3, 4)
	gridSoftmax.Temperature = 0.5 // Slightly confident
	gameNetwork.SetLayer(0, 0, 2, gridSoftmax)

	// Create game state input
	gameState := make([]float32, 64)
	for i := range gameState {
		gameState[i] = rand.Float32()
	}

	fmt.Println("\nGame state processed through network...")
	probs, _ := gameNetwork.ForwardCPU(gameState)

	actions := []string{"move", "shoot", "ability", "idle"}
	fmt.Println("\nAction probabilities:")
	for unit := 0; unit < 3; unit++ {
		fmt.Printf("  Unit %d: ", unit)
		for action := 0; action < 4; action++ {
			idx := unit*4 + action
			fmt.Printf("%s=%.2f ", actions[action], probs[idx])
		}

		// Find chosen action
		bestAction := 0
		bestProb := probs[unit*4]
		for action := 1; action < 4; action++ {
			if probs[unit*4+action] > bestProb {
				bestProb = probs[unit*4+action]
				bestAction = action
			}
		}
		fmt.Printf("→ %s\n", actions[bestAction])
	}

	fmt.Println()
	fmt.Println("═════════════════════════════════════════════════════")
	fmt.Println("✓ All 10 softmax variants implemented!")
	fmt.Println("✓ Can be used as regular network layers")
	fmt.Println("✓ Mix and match with Dense, Conv2D, LSTM, etc.")
	fmt.Println("═════════════════════════════════════════════════════")
}

func printArray(arr []float32) {
	fmt.Print("[")
	for i, v := range arr {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("%.4f", v)
	}
	fmt.Println("]")
}

func sum(arr []float32) float32 {
	s := float32(0)
	for _, v := range arr {
		s += v
	}
	return s
}

func max(arr []float32) float32 {
	m := arr[0]
	for _, v := range arr {
		if v > m {
			m = v
		}
	}
	return m
}
