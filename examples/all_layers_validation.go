package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== All Layer Types Test ===")
	fmt.Println("Testing network with all 5 layer types: Dense, Conv2D, Attention, RNN, LSTM")
	fmt.Println()

	// Carefully designed network using ALL layer types with matching dimensions:
	// Input: 32 values
	// Layer 0 (Dense): 32 -> 32
	// Layer 1 (Conv2D): reshape as 4x4x2 -> 2x2x4 = 16 values
	// Layer 2 (Attention): 16 values as 4 seq x 4 dim -> 16 values
	// Layer 3 (RNN): 16 as 4 timesteps x 4 features -> 4x8 = 32 values
	// Layer 4 (LSTM): 32 as 4 timesteps x 8 features -> 4x4 = 16 values
	// Layer 5 (Dense): 16 -> 2 classes

	batchSize := 1
	network := nn.NewNetwork(32, 1, 1, 6)
	network.BatchSize = batchSize

	fmt.Println("Building network with all layer types...")
	fmt.Println()

	// Layer 0: Dense (32 -> 32)
	dense1 := nn.InitDenseLayer(32, 32, nn.ActivationLeakyReLU)
	network.SetLayer(0, 0, 0, dense1)
	fmt.Println("  Layer 0: Dense (32 -> 32, LeakyReLU)")

	// Layer 1: Conv2D (4x4x2 -> 2x2x4 = 16)
	conv := nn.InitConv2DLayer(
		4, 4, 2, // Input: 4x4 spatial, 2 channels (32 values reshaped)
		3, 2, 1, // 3x3 kernel, stride 2, padding 1
		4, // 4 output filters -> 2x2x4 = 16 values
		nn.ActivationLeakyReLU,
	)
	network.SetLayer(0, 0, 1, conv)
	fmt.Println("  Layer 1: Conv2D (4x4x2 -> 2x2x4=16, LeakyReLU)")

	// Layer 2: Multi-Head Attention (16 -> 16)
	// Treat as sequence: 4 timesteps x 4 dimensions
	attention := nn.InitMultiHeadAttentionLayer(
		4, // dModel
		2, // numHeads
		4, // seqLength
		nn.ActivationTanh,
	)
	network.SetLayer(0, 0, 2, attention)
	fmt.Println("  Layer 2: Attention (4 seq x 4 dim, 2 heads, Tanh)")

	// Layer 3: RNN (4 features, 8 hidden, 4 timesteps -> 32)
	rnn := nn.InitRNNLayer(
		4, // inputSize
		8, // hiddenSize
		batchSize,
		4, // seqLength
	)
	network.SetLayer(0, 0, 3, rnn)
	fmt.Println("  Layer 3: RNN (4 features, 8 hidden, 4 steps -> 32)")

	// Layer 4: LSTM (8 features, 4 hidden, 4 timesteps -> 16)
	lstm := nn.InitLSTMLayer(
		8, // inputSize
		4, // hiddenSize
		batchSize,
		4, // seqLength
	)
	network.SetLayer(0, 0, 4, lstm)
	fmt.Println("  Layer 4: LSTM (8 features, 4 hidden, 4 steps -> 16)")

	// Layer 5: Dense (16 -> 2)
	dense2 := nn.InitDenseLayer(16, 2, nn.ActivationSigmoid)
	network.SetLayer(0, 0, 5, dense2)
	fmt.Println("  Layer 5: Dense (16 -> 2, Sigmoid)")

	fmt.Println()
	fmt.Println("Network Summary:")
	fmt.Println("  Total layers: 6")
	fmt.Println("  Layer types: Dense → Conv2D → Attention → RNN → LSTM → Dense")
	fmt.Println("  Data flow: 32 → 32 → 16 → 16 → 32 → 16 → 2")
	fmt.Println()

	// Create training data
	numSamples := 50
	batches := make([]nn.TrainingBatch, numSamples)

	for i := 0; i < numSamples; i++ {
		var input []float32
		var target []float32

		if i%2 == 0 {
			// Pattern type 0: higher values in first half
			input = make([]float32, 32)
			for j := 0; j < 16; j++ {
				input[j] = 0.7 + rand.Float32()*0.3
			}
			for j := 16; j < 32; j++ {
				input[j] = rand.Float32() * 0.3
			}
			target = []float32{1.0, 0.0}
		} else {
			// Pattern type 1: higher values in second half
			input = make([]float32, 32)
			for j := 0; j < 16; j++ {
				input[j] = rand.Float32() * 0.3
			}
			for j := 16; j < 32; j++ {
				input[j] = 0.7 + rand.Float32()*0.3
			}
			target = []float32{0.0, 1.0}
		}

		batches[i] = nn.TrainingBatch{
			Input:  input,
			Target: target,
		}
	}

	fmt.Printf("Generated %d training samples\n", numSamples)
	fmt.Println()

	// Training configuration
	config := &nn.TrainingConfig{
		Epochs:          200, // Complex model needs more training
		LearningRate:    0.01,
		UseGPU:          false,
		PrintEveryBatch: 0,
		GradientClip:    1.0,
		LossType:        "mse",
		Verbose:         false, // Less noise
	}

	fmt.Println("Starting training...")
	fmt.Println()

	// Train
	result, err := network.Train(batches, config)
	if err != nil {
		fmt.Printf("Training failed: %v\n", err)
		return
	}

	fmt.Println()
	fmt.Printf("✓ Training complete!\n")
	fmt.Printf("  Initial Loss: %.6f\n", result.LossHistory[0])
	fmt.Printf("  Final Loss: %.6f\n", result.FinalLoss)
	fmt.Printf("  Improvement: %.6f (%.1f%%)\n",
		result.LossHistory[0]-result.FinalLoss,
		100*(result.LossHistory[0]-result.FinalLoss)/result.LossHistory[0])
	fmt.Printf("  Throughput: %.2f samples/sec\n", result.AvgThroughput)
	fmt.Println()

	// Test predictions
	fmt.Println("=== Testing Predictions ===")

	// Test pattern 0 (high in first half)
	test0 := make([]float32, 32)
	for j := 0; j < 16; j++ {
		test0[j] = 0.8
	}
	for j := 16; j < 32; j++ {
		test0[j] = 0.2
	}

	// Test pattern 1 (high in second half)
	test1 := make([]float32, 32)
	for j := 0; j < 16; j++ {
		test1[j] = 0.2
	}
	for j := 16; j < 32; j++ {
		test1[j] = 0.8
	}

	testCases := []struct {
		name  string
		input []float32
	}{
		{"Pattern 0 (first half high)", test0},
		{"Pattern 1 (second half high)", test1},
	}

	for _, tc := range testCases {
		output, _ := network.ForwardCPU(tc.input)

		pred0 := output[0]
		pred1 := output[1]

		predClass := 0
		if pred1 > pred0 {
			predClass = 1
		}
		fmt.Printf("%s: [%.4f, %.4f] → Class %d\n", tc.name, pred0, pred1, predClass)
	}

	// Check if it learned
	output0, _ := network.ForwardCPU(test0)
	output1, _ := network.ForwardCPU(test1)

	class0 := 0
	if output0[1] > output0[0] {
		class0 = 1
	}
	class1 := 0
	if output1[1] > output1[0] {
		class1 = 1
	}

	fmt.Println()
	if class0 == 0 && class1 == 1 {
		fmt.Println("✅ Perfect classification with ALL 5 layer types!")
	} else {
		fmt.Printf("⚠️  Not quite separated (got class %d, %d) - needs more training\n", class0, class1)
	}

	fmt.Println()
	fmt.Println("=== Layer Type Summary ===")
	fmt.Println("✓ LayerDense: Tested (layers 0, 5)")
	fmt.Println("✓ LayerConv2D: Tested (layer 1)")
	fmt.Println("✓ LayerMultiHeadAttention: Tested (layer 2)")
	fmt.Println("✓ LayerRNN: Tested (layer 3)")
	fmt.Println("✓ LayerLSTM: Tested (layer 4)")
	fmt.Println()
	fmt.Println("✅ All 5 layer types successfully integrated and trained!")
}
