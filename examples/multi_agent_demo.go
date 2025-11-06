package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== Multi-Agent Network Demo ===")
	fmt.Println("One network controls 3 agents simultaneously")
	fmt.Println()

	// Network configuration
	numAgents := 3
	actionsPerAgent := 4                        // [move_forward, turn_left, turn_right, special]
	totalOutputs := numAgents * actionsPerAgent // 12 outputs

	batchSize := 1
	network := nn.NewNetwork(64, 1, 1, 6)
	network.BatchSize = batchSize

	fmt.Println("Building multi-agent network...")
	fmt.Println()

	// Layer 0: Input processing
	dense0 := nn.InitDenseLayer(64, 64, nn.ActivationLeakyReLU)
	network.SetLayer(0, 0, 0, dense0)
	fmt.Println("  Layer 0: Dense (64 -> 64) - Feature preprocessing")

	// Layer 1: CNN for spatial awareness
	conv := nn.InitConv2DLayer(8, 8, 1, 3, 2, 1, 4, nn.ActivationLeakyReLU)
	network.SetLayer(0, 0, 1, conv)
	fmt.Println("  Layer 1: Conv2D (8x8x1 -> 4x4x4=64) - Spatial perception")

	// Layer 2: LSTM for temporal patterns
	lstm := nn.InitLSTMLayer(8, 8, batchSize, 8)
	network.SetLayer(0, 0, 2, lstm)
	fmt.Println("  Layer 2: LSTM (8 features x 8 steps -> 64) - Temporal memory")

	// Layer 3: Attention for relational reasoning
	attention := nn.InitMultiHeadAttentionLayer(8, 4, 8, nn.ActivationTanh)
	network.SetLayer(0, 0, 3, attention)
	fmt.Println("  Layer 3: Attention (8 entities x 8 dim, 4 heads) - Relational focus")

	// Layer 4: Fusion layer
	dense1 := nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU)
	network.SetLayer(0, 0, 4, dense1)
	fmt.Println("  Layer 4: Dense (64 -> 32) - Feature fusion")

	// Layer 5: MULTI-AGENT OUTPUT LAYER
	// This is the key: output size = num_agents × actions_per_agent
	multiAgentHead := nn.InitDenseLayer(32, totalOutputs, nn.ActivationScaledReLU)
	network.SetLayer(0, 0, 5, multiAgentHead)
	fmt.Printf("  Layer 5: Dense (32 -> %d) - Multi-agent action head\n", totalOutputs)

	fmt.Println()
	fmt.Println("Network Summary:")
	fmt.Printf("  Agents: %d\n", numAgents)
	fmt.Printf("  Actions per agent: %d\n", actionsPerAgent)
	fmt.Printf("  Total outputs: %d (arranged as %dx%d grid)\n", totalOutputs, numAgents, actionsPerAgent)
	fmt.Println()

	// Generate training data
	numSamples := 100
	batches := make([]nn.TrainingBatch, numSamples)

	actions := []string{"move_forward", "turn_left", "turn_right", "special"}

	fmt.Println("Generating multi-agent scenarios...")
	for i := 0; i < numSamples; i++ {
		input := make([]float32, 64)

		// Fill with random game state
		for j := range input {
			input[j] = rand.Float32()
		}

		// Target: each agent has a preferred action based on position
		target := make([]float32, totalOutputs)
		for agent := 0; agent < numAgents; agent++ {
			// Agent's position in input determines preferred action
			agentPos := agent * 20
			if input[agentPos] > 0.5 {
				// High value -> prefer attack (action 3)
				target[agent*actionsPerAgent+3] = 0.9
				target[agent*actionsPerAgent+0] = 0.1
				target[agent*actionsPerAgent+1] = 0.1
				target[agent*actionsPerAgent+2] = 0.1
			} else {
				// Low value -> prefer move forward (action 0)
				target[agent*actionsPerAgent+0] = 0.9
				target[agent*actionsPerAgent+1] = 0.1
				target[agent*actionsPerAgent+2] = 0.1
				target[agent*actionsPerAgent+3] = 0.1
			}
		}

		batches[i] = nn.TrainingBatch{
			Input:  input,
			Target: target,
		}
	}

	// Train
	config := &nn.TrainingConfig{
		Epochs:          50,
		LearningRate:    0.01,
		UseGPU:          false,
		PrintEveryBatch: 0,
		GradientClip:    1.0,
		LossType:        "mse",
		Verbose:         false,
	}

	fmt.Println("Training multi-agent network...")
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

	// Test multi-agent decision making
	fmt.Println("=== Testing Multi-Agent Decisions ===")
	fmt.Println()

	// Create test scenario
	testInput := make([]float32, 64)
	// Agent 0 position: high value -> should attack
	testInput[0] = 0.9
	// Agent 1 position: low value -> should move forward
	testInput[20] = 0.2
	// Agent 2 position: high value -> should attack
	testInput[40] = 0.8

	// Get network output
	logits, _ := network.ForwardCPU(testInput)

	fmt.Println("Raw network output (logits):")
	printGrid("Logits", logits, numAgents, actionsPerAgent)
	fmt.Println()

	// Apply grid softmax
	probs := softmaxGrid(logits, numAgents, actionsPerAgent)

	fmt.Println("After grid softmax (probabilities):")
	printGrid("Probabilities", probs, numAgents, actionsPerAgent)
	fmt.Println()

	// Extract actions per agent
	fmt.Println("Chosen actions:")
	for agent := 0; agent < numAgents; agent++ {
		rowStart := agent * actionsPerAgent
		rowEnd := rowStart + actionsPerAgent
		agentProbs := probs[rowStart:rowEnd]

		actionIdx := argmax(agentProbs)
		fmt.Printf("  Agent %d: %s (prob=%.3f)\n",
			agent, actions[actionIdx], agentProbs[actionIdx])
	}
	fmt.Println()

	fmt.Println("=== Key Insights ===")
	fmt.Println("✓ ONE network outputs actions for ALL agents")
	fmt.Printf("✓ Output layer size: %d = %d agents × %d actions\n", totalOutputs, numAgents, actionsPerAgent)
	fmt.Println("✓ Grid softmax ensures each agent's actions are independent")
	fmt.Println("✓ Network sees full team state, enabling coordination")
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

func printGrid(name string, values []float32, rows, cols int) {
	actions := []string{"fwd", "left", "right", "spec"}
	fmt.Printf("%s (%dx%d):\n", name, rows, cols)
	for r := 0; r < rows; r++ {
		fmt.Printf("  Agent %d: ", r)
		for c := 0; c < cols; c++ {
			idx := r*cols + c
			fmt.Printf("%s=%.3f ", actions[c], values[idx])
		}
		fmt.Println()
	}
}
