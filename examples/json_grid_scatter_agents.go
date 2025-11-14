package main

import (
	"fmt"
	"log"
	"math"

	nn "github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== LOOM Grid Scatter: Multi-Agent Coordination ===")
	fmt.Println("Doing impossible things: Heterogeneous agents with spatial topology")
	fmt.Println()

	// Example 1: Agent Swarm with Specialized Roles
	// 4 agents in a 2x2 grid, each with different architecture
	// Agent [0,0] = Scout (LSTM for temporal patterns)
	// Agent [0,1] = Analyzer (MHA for attention)
	// Agent [1,0] = Executor (Dense ensemble)
	// Agent [1,1] = Coordinator (RNN for sequential decisions)
	fmt.Println("Example 1: Heterogeneous Agent Swarm")
	fmt.Println("4 agents with completely different architectures coordinating")
	fmt.Println()

	example1JSON := `{
		"batch_size": 2,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 20,
				"output_size": 32,
				"activation": "relu",
				"comment": "Shared perception layer"
			},
			{
				"type": "parallel",
				"combine_mode": "grid_scatter",
				"grid_output_rows": 2,
				"grid_output_cols": 2,
				"grid_output_layers": 1,
				"grid_positions": [
					{"branch_index": 0, "target_row": 0, "target_col": 0, "target_layer": 0},
					{"branch_index": 1, "target_row": 0, "target_col": 1, "target_layer": 0},
					{"branch_index": 2, "target_row": 1, "target_col": 0, "target_layer": 0},
					{"branch_index": 3, "target_row": 1, "target_col": 1, "target_layer": 0}
				],
				"branches": [
					{
						"type": "lstm",
						"input_size": 32,
						"hidden_size": 10,
						"seq_length": 1,
						"comment": "Agent 0: Scout (temporal memory)"
					},
					{
						"type": "mha",
						"d_model": 32,
						"num_heads": 4,
						"seq_length": 1,
						"comment": "Agent 1: Analyzer (attention-based)"
					},
					{
						"type": "parallel",
						"combine_mode": "add",
						"branches": [
							{
								"type": "dense",
								"input_size": 32,
								"output_size": 10,
								"activation": "relu"
							},
							{
								"type": "dense",
								"input_size": 32,
								"output_size": 10,
								"activation": "gelu"
							},
							{
								"type": "dense",
								"input_size": 32,
								"output_size": 10,
								"activation": "tanh"
							}
						],
						"comment": "Agent 2: Executor (ensemble decision)"
					},
					{
						"type": "rnn",
						"input_size": 32,
						"hidden_size": 10,
						"seq_length": 1,
						"comment": "Agent 3: Coordinator (sequential processing)"
					}
				]
			}
		]
	}`

	net1, err := nn.BuildNetworkFromJSON(example1JSON)
	if err != nil {
		log.Fatalf("Failed to build network 1: %v", err)
	}
	net1.InitializeWeights()

	input1 := make([]float32, 2*20)
	for i := range input1 {
		input1[i] = float32(i%20) * 0.05
	}

	output1, _ := net1.ForwardCPU(input1)
	fmt.Printf("Input: [batch=2, sensor_data=20]\n")
	fmt.Printf("\nAgent Grid (2x2):\n")
	fmt.Printf("  ┌──────────────────┬──────────────────┐\n")
	fmt.Printf("  │ Scout (LSTM)     │ Analyzer (MHA)   │\n")
	fmt.Printf("  │ 10 decisions     │ 32 decisions     │\n")
	fmt.Printf("  ├──────────────────┼──────────────────┤\n")
	fmt.Printf("  │ Executor (3xDense)│ Coordinator (RNN)│\n")
	fmt.Printf("  │ 10 decisions     │ 10 decisions     │\n")
	fmt.Printf("  └──────────────────┴──────────────────┘\n")
	fmt.Printf("\nTotal outputs: 10 + 32 + 10 + 10 = 62 per sample\n")
	fmt.Printf("Output shape: [batch=2, total=62] = %d values\n", len(output1))
	fmt.Printf("Scout decisions (batch 1): %v\n", output1[0:10])
	fmt.Printf("Analyzer decisions (batch 1): %v\n", output1[10:42])
	fmt.Printf("Executor decisions (batch 1): %v\n", output1[42:52])
	fmt.Printf("Coordinator decisions (batch 1): %v\n", output1[52:62])
	fmt.Println()

	// Example 2: Multi-Scale Feature Processing
	// Different agents process same input at different scales/resolutions
	// Using Conv2D, LayerNorm, RMSNorm, SwiGLU - all different layer types!
	fmt.Println("Example 2: Multi-Scale Processing with Different Normalization")
	fmt.Println("Each agent uses different processing paradigm")
	fmt.Println()

	example2JSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 24,
				"output_size": 24,
				"activation": "relu"
			},
			{
				"type": "parallel",
				"combine_mode": "grid_scatter",
				"grid_output_rows": 3,
				"grid_output_cols": 1,
				"grid_output_layers": 1,
				"grid_positions": [
					{"branch_index": 0, "target_row": 0, "target_col": 0, "target_layer": 0},
					{"branch_index": 1, "target_row": 1, "target_col": 0, "target_layer": 0},
					{"branch_index": 2, "target_row": 2, "target_col": 0, "target_layer": 0}
				],
				"branches": [
					{
						"type": "layer_norm",
						"norm_size": 24,
						"epsilon": 1e-5,
						"comment": "Agent 0: LayerNorm processor"
					},
					{
						"type": "rms_norm",
						"norm_size": 24,
						"epsilon": 1e-5,
						"comment": "Agent 1: RMSNorm processor (Llama-style)"
					},
					{
						"type": "swiglu",
						"input_size": 24,
						"output_size": 24,
						"comment": "Agent 2: SwiGLU gated processor"
					}
				]
			}
		]
	}`

	net2, err := nn.BuildNetworkFromJSON(example2JSON)
	if err != nil {
		log.Fatalf("Failed to build network 2: %v", err)
	}
	net2.InitializeWeights()

	input2 := make([]float32, 24)
	for i := range input2 {
		input2[i] = float32(i)*0.1 - 1.0 // Range from -1.0 to ~1.4
	}

	output2, _ := net2.ForwardCPU(input2)
	fmt.Printf("Input: [24 features] with range -1.0 to 1.4\n")
	fmt.Printf("\nAgent Grid (3x1 - vertical stack):\n")
	fmt.Printf("  ┌──────────────────────┐\n")
	fmt.Printf("  │ LayerNorm (stable)   │ Row 0\n")
	fmt.Printf("  │ 24 normalized        │\n")
	fmt.Printf("  ├──────────────────────┤\n")
	fmt.Printf("  │ RMSNorm (efficient)  │ Row 1\n")
	fmt.Printf("  │ 24 normalized        │\n")
	fmt.Printf("  ├──────────────────────┤\n")
	fmt.Printf("  │ SwiGLU (gated)       │ Row 2\n")
	fmt.Printf("  │ 24 gated features    │\n")
	fmt.Printf("  └──────────────────────┘\n")
	fmt.Printf("\nOutput shape: [72 features] = %d values\n", len(output2))
	fmt.Printf("LayerNorm output: %v\n", output2[0:24])
	fmt.Printf("RMSNorm output: %v\n", output2[24:48])
	fmt.Printf("SwiGLU output: %v\n", output2[48:72])
	fmt.Println()

	// Example 3: Hierarchical Policy Network for RL
	// Top layer = high-level strategy (softmax over 3 strategies)
	// Middle layer = tactic selection per strategy (3 agents, one per strategy)
	// Bottom layer = action execution
	fmt.Println("Example 3: Hierarchical Reinforcement Learning")
	fmt.Println("Strategy → Tactics → Actions with spatial decomposition")
	fmt.Println()

	example3JSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 16,
				"output_size": 16,
				"activation": "relu",
				"comment": "State encoding"
			},
			{
				"type": "parallel",
				"combine_mode": "grid_scatter",
				"grid_output_rows": 1,
				"grid_output_cols": 1,
				"grid_output_layers": 3,
				"grid_positions": [
					{"branch_index": 0, "target_row": 0, "target_col": 0, "target_layer": 0},
					{"branch_index": 1, "target_row": 0, "target_col": 0, "target_layer": 1},
					{"branch_index": 2, "target_row": 0, "target_col": 0, "target_layer": 2}
				],
				"branches": [
					{
						"type": "softmax",
						"softmax_variant": "grid",
						"softmax_rows": 1,
						"softmax_cols": 3,
						"comment": "Layer 0: Strategy selection (3 strategies)"
					},
					{
						"type": "parallel",
						"combine_mode": "concat",
						"branches": [
							{
								"type": "dense",
								"input_size": 16,
								"output_size": 4,
								"activation": "relu",
								"comment": "Tactics for strategy 0"
							},
							{
								"type": "lstm",
								"input_size": 16,
								"hidden_size": 4,
								"seq_length": 1,
								"comment": "Tactics for strategy 1"
							},
							{
								"type": "rnn",
								"input_size": 16,
								"hidden_size": 4,
								"seq_length": 1,
								"comment": "Tactics for strategy 2"
							}
						],
						"comment": "Layer 1: Tactic selection (3 parallel tactic networks)"
					},
					{
						"type": "parallel",
						"combine_mode": "avg",
						"branches": [
							{
								"type": "dense",
								"input_size": 16,
								"output_size": 8,
								"activation": "tanh"
							},
							{
								"type": "dense",
								"input_size": 16,
								"output_size": 8,
								"activation": "sigmoid"
							}
						],
						"comment": "Layer 2: Action execution (averaged policies)"
					}
				]
			}
		]
	}`

	net3, err := nn.BuildNetworkFromJSON(example3JSON)
	if err != nil {
		log.Fatalf("Failed to build network 3: %v", err)
	}
	net3.InitializeWeights()

	input3 := make([]float32, 16)
	for i := range input3 {
		input3[i] = float32(math.Cos(float64(i) * 0.5))
	}

	output3, _ := net3.ForwardCPU(input3)
	fmt.Printf("Input: [state=16]\n")
	fmt.Printf("\nHierarchical RL Grid (1x1x3 - depth layers):\n")
	fmt.Printf("  Layer 0 (Strategy):  Softmax → 3 strategy probabilities\n")
	fmt.Printf("  Layer 1 (Tactics):   Dense+LSTM+RNN → 12 tactic outputs\n")
	fmt.Printf("  Layer 2 (Actions):   2×Dense(avg) → 8 action outputs\n")
	fmt.Printf("\nOutput shape: [23 values] = %d\n", len(output3))
	fmt.Printf("Strategies (softmax): %v\n", output3[0:3])
	fmt.Printf("Tactics (concat): %v\n", output3[3:15])
	fmt.Printf("Actions (avg): %v\n", output3[15:23])

	// Show which strategy was selected
	maxStratIdx := 0
	maxStratVal := output3[0]
	for i := 1; i < 3; i++ {
		if output3[i] > maxStratVal {
			maxStratVal = output3[i]
			maxStratIdx = i
		}
	}
	fmt.Printf("\nSelected Strategy: %d (prob=%.3f)\n", maxStratIdx, maxStratVal)
	fmt.Printf("Corresponding Tactics: %v\n", output3[3+maxStratIdx*4:3+maxStratIdx*4+4])
	fmt.Println()

	// Example 4: TRAINING - Multi-Agent Collaborative Task
	// 3 agents must learn complementary roles to solve a task
	// Agent 0 extracts features, Agent 1 transforms, Agent 2 decides
	fmt.Println("Example 4: Training Multi-Agent Collaboration")
	fmt.Println("Task: 3 agents learn to collaborate for binary classification")
	fmt.Println()

	trainingJSON := `{
		"batch_size": 4,
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 8,
				"output_size": 16,
				"activation": "relu",
				"comment": "Shared sensory processing"
			},
			{
				"type": "parallel",
				"combine_mode": "grid_scatter",
				"grid_output_rows": 3,
				"grid_output_cols": 1,
				"grid_output_layers": 1,
				"grid_positions": [
					{"branch_index": 0, "target_row": 0, "target_col": 0, "target_layer": 0},
					{"branch_index": 1, "target_row": 1, "target_col": 0, "target_layer": 0},
					{"branch_index": 2, "target_row": 2, "target_col": 0, "target_layer": 0}
				],
				"branches": [
					{
						"type": "parallel",
						"combine_mode": "add",
						"branches": [
							{
								"type": "dense",
								"input_size": 16,
								"output_size": 8,
								"activation": "relu"
							},
							{
								"type": "dense",
								"input_size": 16,
								"output_size": 8,
								"activation": "gelu"
							}
						],
						"comment": "Agent 0: Feature Extractor (ensemble)"
					},
					{
						"type": "lstm",
						"input_size": 16,
						"hidden_size": 8,
						"seq_length": 1,
						"comment": "Agent 1: Transformer (temporal)"
					},
					{
						"type": "rnn",
						"input_size": 16,
						"hidden_size": 8,
						"seq_length": 1,
						"comment": "Agent 2: Integrator (sequential)"
					}
				]
			},
			{
				"type": "dense",
				"input_size": 24,
				"output_size": 2,
				"activation": "sigmoid",
				"comment": "Final decision layer"
			}
		]
	}`

	netTrain, err := nn.BuildNetworkFromJSON(trainingJSON)
	if err != nil {
		log.Fatalf("Failed to build training network: %v", err)
	}
	netTrain.InitializeWeights()

	// Create training data: XOR-like problem
	trainData := make([]float32, 4*8)
	trainLabels := []float32{
		1, 0, // Sample 0: Class 0
		0, 1, // Sample 1: Class 1
		0, 1, // Sample 2: Class 1
		1, 0, // Sample 3: Class 0
	}

	// Pattern: if first 4 features sum > second 4 features sum → class 1
	// Sample 0: [0.2, 0.2, 0.2, 0.2,  0.8, 0.8, 0.8, 0.8] → class 0
	for i := 0; i < 4; i++ {
		trainData[0*8+i] = 0.2
		trainData[0*8+4+i] = 0.8
	}
	// Sample 1: [0.9, 0.9, 0.9, 0.9,  0.1, 0.1, 0.1, 0.1] → class 1
	for i := 0; i < 4; i++ {
		trainData[1*8+i] = 0.9
		trainData[1*8+4+i] = 0.1
	}
	// Sample 2: [0.7, 0.7, 0.7, 0.7,  0.3, 0.3, 0.3, 0.3] → class 1
	for i := 0; i < 4; i++ {
		trainData[2*8+i] = 0.7
		trainData[2*8+4+i] = 0.3
	}
	// Sample 3: [0.3, 0.3, 0.3, 0.3,  0.7, 0.7, 0.7, 0.7] → class 0
	for i := 0; i < 4; i++ {
		trainData[3*8+i] = 0.3
		trainData[3*8+4+i] = 0.7
	}

	epochs := 800
	learningRate := float32(0.15)

	fmt.Printf("Training for %d epochs with learning rate %.3f\n", epochs, learningRate)
	fmt.Printf("Architecture:\n")
	fmt.Printf("  Shared Layer → Grid Scatter (3 agents) → Decision\n")
	fmt.Printf("  Agent 0: Feature Extractor (ensemble of 2 dense)\n")
	fmt.Printf("  Agent 1: Transformer (LSTM)\n")
	fmt.Printf("  Agent 2: Integrator (RNN)\n")
	fmt.Printf("Task: Binary classification (sum comparison)\n")
	fmt.Println()

	initialLoss := float64(0)
	finalLoss := float64(0)

	for epoch := 0; epoch < epochs; epoch++ {
		// Forward pass
		output, _ := netTrain.ForwardCPU(trainData)

		// Compute loss (MSE)
		loss := float32(0.0)
		for i := range trainLabels {
			diff := output[i] - trainLabels[i]
			loss += diff * diff
		}
		loss /= float32(len(trainLabels))

		if epoch == 0 {
			initialLoss = float64(loss)
		}
		if epoch == epochs-1 {
			finalLoss = float64(loss)
		}

		// Compute gradients
		gradOutput := make([]float32, len(trainLabels))
		for i := range trainLabels {
			gradOutput[i] = 2 * (output[i] - trainLabels[i]) / float32(len(trainLabels))
		}

		// Backward pass
		netTrain.BackwardCPU(gradOutput)

		// Update weights
		netTrain.UpdateWeights(learningRate)

		if (epoch+1)%200 == 0 || epoch == 0 {
			fmt.Printf("Epoch %4d: Loss = %.6f\n", epoch+1, loss)
		}
	}

	fmt.Println()
	fmt.Printf("Initial Loss: %.6f\n", initialLoss)
	fmt.Printf("Final Loss:   %.6f\n", finalLoss)
	fmt.Printf("Improvement:  %.2f%%\n", (1.0-finalLoss/initialLoss)*100)
	fmt.Println()

	// Test final predictions
	fmt.Println("Final predictions:")
	finalOutput, _ := netTrain.ForwardCPU(trainData)
	for i := 0; i < 4; i++ {
		predicted := finalOutput[i*2 : i*2+2]
		expected := trainLabels[i*2 : i*2+2]

		predClass := 0
		if predicted[1] > predicted[0] {
			predClass = 1
		}

		expClass := 0
		if expected[1] > expected[0] {
			expClass = 1
		}

		correct := "✓"
		if predClass != expClass {
			correct = "✗"
		}

		fmt.Printf("Sample %d: [%.3f, %.3f] → Class %d (expected %d) %s\n",
			i, predicted[0], predicted[1], predClass, expClass, correct)
	}
	fmt.Println()

	fmt.Println("=== Demo Complete ===")
	fmt.Println()
	fmt.Println("What makes this impossible in traditional neural networks:")
	fmt.Println()
	fmt.Println("1. HETEROGENEOUS AGENTS:")
	fmt.Println("   • Each grid position has COMPLETELY DIFFERENT architecture")
	fmt.Println("   • LSTM, MHA, RNN, Dense ensemble - all in same layer!")
	fmt.Println("   • Traditional NNs: all neurons same type per layer")
	fmt.Println()
	fmt.Println("2. SPATIAL TOPOLOGY:")
	fmt.Println("   • Agents arranged in explicit 2D/3D grid structure")
	fmt.Println("   • Each agent occupies specific spatial position")
	fmt.Println("   • Traditional NNs: flat vectors, no spatial meaning")
	fmt.Println()
	fmt.Println("3. HIERARCHICAL DECOMPOSITION:")
	fmt.Println("   • Grid LAYERS create vertical hierarchy")
	fmt.Println("   • Strategy → Tactics → Actions at different depths")
	fmt.Println("   • Traditional NNs: fixed sequential depth")
	fmt.Println()
	fmt.Println("4. SPECIALIZED ROLES:")
	fmt.Println("   • Each agent learns complementary function")
	fmt.Println("   • Extractor → Transformer → Integrator pipeline")
	fmt.Println("   • Emergent division of labor through grid structure")
	fmt.Println()
	fmt.Println("Real-world applications:")
	fmt.Println("→ Multi-robot coordination with heterogeneous robots")
	fmt.Println("→ Hierarchical RL: strategies decompose spatially")
	fmt.Println("→ Multi-agent game playing (StarCraft, Dota)")
	fmt.Println("→ Distributed sensor networks with specialized nodes")
	fmt.Println("→ Ensemble methods with architectural diversity")
	fmt.Println("→ Neural architecture search with spatial constraints")
	fmt.Println("→ Modular neural networks with explicit communication")
}
