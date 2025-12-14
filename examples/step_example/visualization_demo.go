package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"strings"
	"time"

	nn "github.com/openfluke/loom/nn"
)

const (
	vizServerURL    = "http://localhost:8080"
	networkEndpoint = "/network"
	eventsEndpoint  = "/events"
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           LOOM Step Visualization Demo - 3D Network Visualization           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Printf("Visualization server expected at: %s\n", vizServerURL)
	fmt.Println("This demo will:")
	fmt.Println("  1. Send network JSON to server for 3D model generation")
	fmt.Println("  2. Attach observers to emit layer events during training")
	fmt.Println("  3. Demonstrate 3 different network architectures")
	fmt.Println()

	// ═══════════════════════════════════════════════════════════════════════════
	// NETWORK 1: All Layer Types (Kitchen Sink)
	// ═══════════════════════════════════════════════════════════════════════════
	runNetwork1AllTypes()

	fmt.Println("\n" + strings.Repeat("─", 80) + "\n")

	// ═══════════════════════════════════════════════════════════════════════════
	// NETWORK 2: Multi-Agent Grid Scatter
	// ═══════════════════════════════════════════════════════════════════════════
	runNetwork2GridScatter()

	fmt.Println("\n" + strings.Repeat("─", 80) + "\n")

	// ═══════════════════════════════════════════════════════════════════════════
	// NETWORK 3: Simple Dense - Normal vs Stepping Training
	// ═══════════════════════════════════════════════════════════════════════════
	runNetwork3NormalVsStepping()

	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════════════════════════")
	fmt.Println("                              Demo Complete!")
	fmt.Println("═══════════════════════════════════════════════════════════════════════════════")
}

// ═══════════════════════════════════════════════════════════════════════════════
// NETWORK 1: All Layer Types
// ═══════════════════════════════════════════════════════════════════════════════

func runNetwork1AllTypes() {
	fmt.Println("┌──────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ NETWORK 1: All Layer Types (Dense, RNN, LSTM, MHA, LayerNorm, RMSNorm, SwiGLU)│")
	fmt.Println("└──────────────────────────────────────────────────────────────────────────────┘")
	fmt.Println()

	networkJSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 32,
				"output_size": 32,
				"activation": "relu"
			},
			{
				"type": "parallel",
				"combine_mode": "concat",
				"branches": [
					{
						"type": "dense",
						"input_size": 32,
						"output_size": 8,
						"activation": "relu"
					},
					{
						"type": "rnn",
						"input_size": 32,
						"hidden_size": 8,
						"seq_length": 1
					},
					{
						"type": "lstm",
						"input_size": 32,
						"hidden_size": 8,
						"seq_length": 1
					},
					{
						"type": "layer_norm",
						"norm_size": 32
					},
					{
						"type": "rmsnorm",
						"norm_size": 32
					},
					{
						"type": "swiglu",
						"input_size": 32,
						"output_size": 8
					}
				]
			},
			{
				"type": "dense",
				"input_size": 96,
				"output_size": 4,
				"activation": "sigmoid"
			}
		]
	}`

	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Printf("Failed to build network 1: %v", err)
		return
	}
	net.InitializeWeights()

	fmt.Println("Architecture:")
	fmt.Println("  Dense(32→32) → Parallel[Dense(8), RNN(8), LSTM(8), LayerNorm(32), RMSNorm(32), SwiGLU(8)]")
	fmt.Println("               → Dense(96→4)")
	fmt.Println()

	// Send network structure to visualization server
	sendNetworkToServer("network1_all_types", networkJSON)

	// Attach observer to all layers
	httpObserver := nn.NewHTTPObserver(vizServerURL + eventsEndpoint)
	attachObserverToAllLayers(net, httpObserver)

	// Training
	fmt.Println("Training with observers attached (events sent to visualization server)...")
	input := make([]float32, 32)
	for i := range input {
		input[i] = float32(math.Sin(float64(i) * 0.2))
	}

	for epoch := 0; epoch < 50; epoch++ {
		output, _ := net.ForwardCPU(input)

		if epoch%10 == 0 {
			fmt.Printf("  Epoch %3d: output sample = [%.4f, %.4f, %.4f, %.4f]\n",
				epoch, output[0], output[1], output[2], output[3])
		}
	}

	fmt.Println("✓ Network 1 complete")
}

// ═══════════════════════════════════════════════════════════════════════════════
// NETWORK 2: Multi-Agent Grid Scatter
// ═══════════════════════════════════════════════════════════════════════════════

func runNetwork2GridScatter() {
	fmt.Println("┌──────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ NETWORK 2: Multi-Agent Grid Scatter (2x2 Heterogeneous Agents)              │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────────────┘")
	fmt.Println()

	networkJSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 16,
				"output_size": 24,
				"activation": "relu"
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
						"input_size": 24,
						"hidden_size": 6,
						"seq_length": 1
					},
					{
						"type": "rnn",
						"input_size": 24,
						"hidden_size": 6,
						"seq_length": 1
					},
					{
						"type": "dense",
						"input_size": 24,
						"output_size": 6,
						"activation": "gelu"
					},
					{
						"type": "dense",
						"input_size": 24,
						"output_size": 6,
						"activation": "tanh"
					}
				]
			}
		]
	}`

	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Printf("Failed to build network 2: %v", err)
		return
	}
	net.InitializeWeights()

	fmt.Println("Architecture:")
	fmt.Println("  Dense(16→24) → Grid Scatter 2x2:")
	fmt.Println("    ┌─────────────────┬─────────────────┐")
	fmt.Println("    │ [0,0] LSTM(6)   │ [0,1] RNN(6)    │")
	fmt.Println("    ├─────────────────┼─────────────────┤")
	fmt.Println("    │ [1,0] Dense(6)  │ [1,1] Dense(6)  │")
	fmt.Println("    └─────────────────┴─────────────────┘")
	fmt.Println()

	// Send network structure to visualization server
	sendNetworkToServer("network2_grid_scatter", networkJSON)

	// Attach observer
	httpObserver := nn.NewHTTPObserver(vizServerURL + eventsEndpoint)
	attachObserverToAllLayers(net, httpObserver)

	// Training
	fmt.Println("Training with observers attached...")
	input := make([]float32, 16)
	for i := range input {
		input[i] = float32(i) * 0.1
	}

	for epoch := 0; epoch < 50; epoch++ {
		output, _ := net.ForwardCPU(input)

		if epoch%10 == 0 {
			fmt.Printf("  Epoch %3d: output length=%d, first 6 = %v\n",
				epoch, len(output), output[:min(6, len(output))])
		}
	}

	fmt.Println("✓ Network 2 complete")
}

// ═══════════════════════════════════════════════════════════════════════════════
// NETWORK 3: Normal Mode vs Stepping Mode Comparison
// ═══════════════════════════════════════════════════════════════════════════════

func runNetwork3NormalVsStepping() {
	fmt.Println("┌──────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ NETWORK 3: Training Mode Comparison (Normal vs Stepping Mechanism)          │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────────────┘")
	fmt.Println()

	networkJSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 8,
				"output_size": 16,
				"activation": "relu"
			},
			{
				"type": "dense",
				"input_size": 16,
				"output_size": 8,
				"activation": "gelu"
			},
			{
				"type": "dense",
				"input_size": 8,
				"output_size": 2,
				"activation": "sigmoid"
			}
		]
	}`

	fmt.Println("Architecture: Dense(8→16→8→2) - Simple feedforward network")
	fmt.Println()

	// Training data
	batches := []nn.TrainingBatch{
		{Input: []float32{0.8, 0.9, 0.7, 0.8, 0.1, 0.2, 0.3, 0.2}, Target: []float32{1.0, 0.0}},
		{Input: []float32{0.1, 0.2, 0.3, 0.2, 0.8, 0.9, 0.7, 0.8}, Target: []float32{0.0, 1.0}},
		{Input: []float32{0.6, 0.7, 0.8, 0.7, 0.2, 0.3, 0.4, 0.3}, Target: []float32{1.0, 0.0}},
		{Input: []float32{0.3, 0.2, 0.4, 0.3, 0.6, 0.7, 0.8, 0.7}, Target: []float32{0.0, 1.0}},
	}

	// ────────────────────────────────────────────────────────────────────────────
	// Part A: Normal Training Mode
	// ────────────────────────────────────────────────────────────────────────────
	fmt.Println("━━━ PART A: Normal Training Mode ━━━")
	fmt.Println()

	netNormal, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Printf("Failed to build network: %v", err)
		return
	}
	netNormal.InitializeWeights()

	// Send network to viz server
	sendNetworkToServer("network3_normal_mode", networkJSON)

	// Attach observer
	httpObserver := nn.NewHTTPObserver(vizServerURL + eventsEndpoint)
	attachObserverToAllLayers(netNormal, httpObserver)

	config := &nn.TrainingConfig{
		Epochs:       100,
		LearningRate: 0.1,
		UseGPU:       false,
		GradientClip: 1.0,
		LossType:     "mse",
		Verbose:      false,
	}

	fmt.Println("Training with standard Train() method...")
	startNormal := time.Now()
	result, _ := netNormal.Train(batches, config)
	normalDuration := time.Since(startNormal)

	fmt.Printf("  Initial Loss: %.6f\n", result.LossHistory[0])
	fmt.Printf("  Final Loss:   %.6f\n", result.FinalLoss)
	fmt.Printf("  Duration:     %v\n", normalDuration)
	fmt.Println()

	// ────────────────────────────────────────────────────────────────────────────
	// Part B: Stepping Mechanism Training
	// ────────────────────────────────────────────────────────────────────────────
	fmt.Println("━━━ PART B: Stepping Mechanism Training ━━━")
	fmt.Println()

	netStep, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Printf("Failed to build network: %v", err)
		return
	}
	netStep.InitializeWeights()

	// Send network to viz server (different model ID)
	sendNetworkToServer("network3_step_mode", networkJSON)

	// Attach observer
	attachObserverToAllLayers(netStep, httpObserver)

	// Initialize stepping state
	state := netStep.InitStepState(8)

	microConfig := &nn.TrainingConfig{
		Epochs:       1,
		LearningRate: 0.05,
		UseGPU:       false,
		GradientClip: 1.0,
		LossType:     "mse",
		Verbose:      false,
	}

	fmt.Println("Training with StepForward() + micro-training...")
	startStep := time.Now()

	stepLossHistory := make([]float32, 0)
	totalSteps := 400 // 100 epochs * 4 samples, but stepping
	sampleIdx := 0

	for step := 0; step < totalSteps; step++ {
		sample := batches[sampleIdx]

		// Step forward
		state.SetInput(sample.Input)
		netStep.StepForward(state)

		// Calculate loss
		output := state.GetOutput()
		loss := float32(0)
		for i := range output {
			diff := output[i] - sample.Target[i]
			loss += diff * diff
		}
		loss /= float32(len(output))
		stepLossHistory = append(stepLossHistory, loss)

		// Micro-train
		microBatch := []nn.TrainingBatch{sample}
		netStep.Train(microBatch, microConfig)

		sampleIdx = (sampleIdx + 1) % len(batches)
	}

	stepDuration := time.Since(startStep)

	fmt.Printf("  Initial Loss: %.6f\n", stepLossHistory[0])
	fmt.Printf("  Final Loss:   %.6f\n", stepLossHistory[len(stepLossHistory)-1])
	fmt.Printf("  Duration:     %v\n", stepDuration)
	fmt.Printf("  Total Steps:  %d\n", totalSteps)
	fmt.Println()

	// ────────────────────────────────────────────────────────────────────────────
	// Comparison
	// ────────────────────────────────────────────────────────────────────────────
	fmt.Println("━━━ COMPARISON ━━━")
	fmt.Println()
	fmt.Printf("  Mode         │ Initial Loss │ Final Loss │ Duration\n")
	fmt.Printf("  ─────────────┼──────────────┼────────────┼──────────\n")
	fmt.Printf("  Normal       │ %.6f     │ %.6f   │ %v\n",
		result.LossHistory[0], result.FinalLoss, normalDuration)
	fmt.Printf("  Stepping     │ %.6f     │ %.6f   │ %v\n",
		stepLossHistory[0], stepLossHistory[len(stepLossHistory)-1], stepDuration)
	fmt.Println()

	// Test predictions on both networks
	fmt.Println("  Final Predictions:")
	for i, sample := range batches[:2] {
		outNormal, _ := netNormal.ForwardCPU(sample.Input)
		state.SetInput(sample.Input)
		netStep.StepForward(state)
		outStep := state.GetOutput()

		fmt.Printf("  Sample %d: Normal=[%.3f,%.3f]  Stepping=[%.3f,%.3f]  Target=%v\n",
			i, outNormal[0], outNormal[1], outStep[0], outStep[1], sample.Target)
	}

	fmt.Println()
	fmt.Println("✓ Network 3 complete")
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

// sendNetworkToServer sends the network JSON to the visualization server
func sendNetworkToServer(modelID, networkJSON string) {
	payload := map[string]interface{}{
		"model_id": modelID,
		"network":  json.RawMessage(networkJSON),
	}

	data, err := json.Marshal(payload)
	if err != nil {
		fmt.Printf("  ⚠ Failed to marshal network JSON: %v\n", err)
		return
	}

	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Post(vizServerURL+networkEndpoint, "application/json", bytes.NewReader(data))
	if err != nil {
		fmt.Printf("  ⚠ Visualization server not available (will continue without): %v\n", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		fmt.Printf("  ✓ Network structure sent to visualization server: %s\n", modelID)
	} else {
		fmt.Printf("  ⚠ Server returned status %d\n", resp.StatusCode)
	}
}

// attachObserverToAllLayers attaches the same observer to all layers in the network
func attachObserverToAllLayers(net *nn.Network, observer nn.LayerObserver) {
	totalLayers := net.TotalLayers()
	for i := 0; i < totalLayers; i++ {
		row := i / (net.GridCols * net.LayersPerCell)
		remainder := i % (net.GridCols * net.LayersPerCell)
		col := remainder / net.LayersPerCell
		layer := remainder % net.LayersPerCell

		cfg := net.GetLayer(row, col, layer)
		if cfg != nil {
			cfg.Observer = observer
		}
	}
	fmt.Printf("  → Observer attached to %d layers\n", totalLayers)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
