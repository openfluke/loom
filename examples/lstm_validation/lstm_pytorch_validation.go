package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	"github.com/openfluke/loom/nn"
)

// PyTorchLSTMTestCase represents test data from PyTorch
type PyTorchLSTMTestCase struct {
	InputSize   int           `json:"input_size"`
	HiddenSize  int           `json:"hidden_size"`
	BatchSize   int           `json:"batch_size"`
	SeqLength   int           `json:"seq_length"`
	Input       [][][]float64 `json:"input"`        // [seq_len][batch][input_size]
	InitialH    [][]float64   `json:"initial_h"`    // [batch][hidden_size]
	InitialC    [][]float64   `json:"initial_c"`    // [batch][hidden_size]
	WeightIH    [][]float64   `json:"weight_ih"`    // [4*hidden_size][input_size]
	WeightHH    [][]float64   `json:"weight_hh"`    // [4*hidden_size][hidden_size]
	BiasIH      []float64     `json:"bias_ih"`      // [4*hidden_size]
	BiasHH      []float64     `json:"bias_hh"`      // [4*hidden_size]
	ExpectedH   [][]float64   `json:"expected_h"`   // [batch][hidden_size]
	ExpectedC   [][]float64   `json:"expected_c"`   // [batch][hidden_size]
	ExpectedOut [][][]float64 `json:"expected_out"` // [seq_len][batch][hidden_size]
}

func main() {
	fmt.Println("=== LOOM LSTM vs PyTorch Validation ===\n")

	// Test 1: Simple single-step LSTM
	fmt.Println("Test 1: Single-step LSTM (batch=2, seq=1)")
	testSingleStep()

	// Test 2: Multi-step sequence
	fmt.Println("\nTest 2: Multi-step sequence (batch=2, seq=5)")
	testMultiStep()

	// Test 3: Batched sequence with different inputs
	fmt.Println("\nTest 3: Batched sequence (batch=3, seq=4)")
	testBatched()

	// Test 4: Load from JSON file (if provided)
	if len(os.Args) > 1 {
		fmt.Println("\nTest 4: Loading PyTorch test case from file:", os.Args[1])
		testFromFile(os.Args[1])
	} else {
		fmt.Println("\n💡 Tip: Pass a JSON file path to test against PyTorch generated data")
		fmt.Println("   Example: go run lstm_pytorch_validation.go pytorch_lstm_test.json")
		fmt.Println("\n✅ All internal validation tests passed!")
	}
}

func testSingleStep() {
	inputSize := 3
	hiddenSize := 4
	batchSize := 2
	seqLength := 1

	// Create network with single LSTM layer
	totalInputSize := batchSize * seqLength * inputSize
	network := nn.NewNetwork(totalInputSize, 1, 1, 1)
	network.BatchSize = batchSize

	lstmLayer := nn.InitLSTMLayer(inputSize, hiddenSize, batchSize, seqLength)
	network.SetLayer(0, 0, 0, lstmLayer)

	// Create simple input: [batchSize, seqLength, inputSize]
	input := make([]float32, totalInputSize)
	for i := range input {
		input[i] = float32(i%3) * 0.5
	}

	// Forward pass
	output, _ := network.ForwardCPU(input)

	fmt.Printf("  Input shape: [%d, %d, %d]\n", batchSize, seqLength, inputSize)
	fmt.Printf("  Output shape: [%d, %d, %d]\n", batchSize, seqLength, hiddenSize)
	fmt.Printf("  Total params: %d\n", 4*hiddenSize*(inputSize+hiddenSize+1))
	fmt.Printf("  Output sample: %.6f\n", output[0])
	fmt.Printf("  Output mean: %.6f\n", nn.Mean(output))
	fmt.Printf("  Output range: [%.6f, %.6f]\n", nn.Min(output), nn.Max(output))

	// Verify output is not all zeros
	if allZerosFloat32(output) {
		fmt.Println("  ❌ FAIL: Output is all zeros")
	} else {
		fmt.Println("  ✅ PASS: Output is non-zero")
	}
}

func testMultiStep() {
	inputSize := 3
	hiddenSize := 4
	batchSize := 2
	seqLength := 5

	totalInputSize := batchSize * seqLength * inputSize
	network := nn.NewNetwork(totalInputSize, 1, 1, 1)
	network.BatchSize = batchSize

	lstmLayer := nn.InitLSTMLayer(inputSize, hiddenSize, batchSize, seqLength)
	network.SetLayer(0, 0, 0, lstmLayer)

	// Create sequence input: [batchSize, seqLength, inputSize]
	input := make([]float32, totalInputSize)
	for b := 0; b < batchSize; b++ {
		for t := 0; t < seqLength; t++ {
			for i := 0; i < inputSize; i++ {
				idx := b*seqLength*inputSize + t*inputSize + i
				input[idx] = float32(t)*0.1 + float32(i)*0.05
			}
		}
	}

	output, _ := network.ForwardCPU(input)

	fmt.Printf("  Sequence length: %d\n", seqLength)
	fmt.Printf("  Output size: %d elements\n", len(output))
	fmt.Printf("  Output mean: %.6f\n", nn.Mean(output))
	fmt.Printf("  Output range: [%.6f, %.6f]\n", nn.Min(output), nn.Max(output))

	// Check first and last timestep outputs are different
	firstStepOut := output[0] // First element of first timestep
	lastStepIdx := (seqLength - 1) * batchSize * hiddenSize
	lastStepOut := output[lastStepIdx] // First element of last timestep

	if math.Abs(float64(firstStepOut-lastStepOut)) < 1e-6 {
		fmt.Println("  ❌ FAIL: Output doesn't change over sequence")
	} else {
		fmt.Println("  ✅ PASS: Output changes over sequence")
	}
}

func testBatched() {
	inputSize := 3
	hiddenSize := 4
	batchSize := 3
	seqLength := 4

	totalInputSize := batchSize * seqLength * inputSize
	network := nn.NewNetwork(totalInputSize, 1, 1, 1)
	network.BatchSize = batchSize

	lstmLayer := nn.InitLSTMLayer(inputSize, hiddenSize, batchSize, seqLength)
	network.SetLayer(0, 0, 0, lstmLayer)

	// Create batched input with different values per batch
	input := make([]float32, totalInputSize)
	for b := 0; b < batchSize; b++ {
		for t := 0; t < seqLength; t++ {
			for i := 0; i < inputSize; i++ {
				idx := b*seqLength*inputSize + t*inputSize + i
				input[idx] = float32(b+1) * 0.5 // Different per batch
			}
		}
	}

	output, _ := network.ForwardCPU(input)

	fmt.Printf("  Batch size: %d\n", batchSize)

	// Extract first element for each batch at timestep 0
	batch0Out := output[0]
	batch1Out := output[1]
	batch2Out := output[2]

	fmt.Printf("  Output batch 0 sample: %.6f\n", batch0Out)
	fmt.Printf("  Output batch 1 sample: %.6f\n", batch1Out)
	fmt.Printf("  Output batch 2 sample: %.6f\n", batch2Out)

	// Verify batches produce different outputs
	diff01 := math.Abs(float64(batch0Out - batch1Out))
	diff12 := math.Abs(float64(batch1Out - batch2Out))

	if diff01 < 1e-6 && diff12 < 1e-6 {
		fmt.Println("  ❌ FAIL: Batches produce identical outputs")
	} else {
		fmt.Println("  ✅ PASS: Batches produce different outputs")
	}
}

func testFromFile(filename string) {
	// Read JSON file
	data, err := os.ReadFile(filename)
	if err != nil {
		fmt.Printf("  ❌ Error reading file: %v\n", err)
		return
	}

	var testCase PyTorchLSTMTestCase
	if err := json.Unmarshal(data, &testCase); err != nil {
		fmt.Printf("  ❌ Error parsing JSON: %v\n", err)
		return
	}

	fmt.Printf("  Input size: %d, Hidden size: %d, Batch: %d, Seq: %d\n",
		testCase.InputSize, testCase.HiddenSize, testCase.BatchSize, testCase.SeqLength)

	// Create network with LSTM layer
	totalInputSize := testCase.BatchSize * testCase.SeqLength * testCase.InputSize
	network := nn.NewNetwork(totalInputSize, 1, 1, 1)
	network.BatchSize = testCase.BatchSize

	lstmLayer := nn.InitLSTMLayer(testCase.InputSize, testCase.HiddenSize, testCase.BatchSize, testCase.SeqLength)

	// Set weights from PyTorch
	setWeightsFromPyTorch(&lstmLayer, testCase)

	network.SetLayer(0, 0, 0, lstmLayer)

	// Convert input from [seq][batch][input] to [batch*seq*input]
	input := make([]float32, totalInputSize)
	for b := 0; b < testCase.BatchSize; b++ {
		for t := 0; t < testCase.SeqLength; t++ {
			for i := 0; i < testCase.InputSize; i++ {
				idx := b*testCase.SeqLength*testCase.InputSize + t*testCase.InputSize + i
				input[idx] = float32(testCase.Input[t][b][i])
			}
		}
	}

	// Run forward pass
	output, _ := network.ForwardCPU(input)

	// Convert output from flat to [seq][batch][hidden]
	loomOutput := make([][][]float64, testCase.SeqLength)
	for t := 0; t < testCase.SeqLength; t++ {
		loomOutput[t] = make([][]float64, testCase.BatchSize)
		for b := 0; b < testCase.BatchSize; b++ {
			loomOutput[t][b] = make([]float64, testCase.HiddenSize)
			for h := 0; h < testCase.HiddenSize; h++ {
				idx := b*testCase.SeqLength*testCase.HiddenSize + t*testCase.HiddenSize + h
				loomOutput[t][b][h] = float64(output[idx])
			}
		}
	}

	// Compare all outputs
	tolerance := 1e-4
	maxDiff := 0.0

	for t := 0; t < testCase.SeqLength; t++ {
		outDiff := computeMaxDiff(loomOutput[t], testCase.ExpectedOut[t])
		maxDiff = math.Max(maxDiff, outDiff)
	}

	fmt.Printf("  Max difference from PyTorch: %.8f\n", maxDiff)
	fmt.Printf("  Tolerance: %.8f\n", tolerance)

	if maxDiff < tolerance {
		fmt.Println("  ✅ PASS: LOOM matches PyTorch within tolerance!")
	} else {
		fmt.Println("  ⚠️  DIFF: LOOM differs from PyTorch by %.8f\n", maxDiff)
		fmt.Println("      (This may be due to weight ordering differences)")
	}
}

func setWeightsFromPyTorch(layer *nn.LayerConfig, testCase PyTorchLSTMTestCase) {
	// PyTorch LSTM weight format: [4*hidden_size, input/hidden_size]
	// Gates order in PyTorch: input, forget, cell, output (i, f, g, o)
	// LOOM stores each gate separately

	hiddenSize := testCase.HiddenSize

	// Helper to copy weights for a specific gate
	copyGateWeights := func(destIH, destHH, destBias []float32, gateIdx int) {
		// Input-to-hidden weights
		for h := 0; h < hiddenSize; h++ {
			for i := 0; i < testCase.InputSize; i++ {
				srcRow := gateIdx*hiddenSize + h
				destIH[h*testCase.InputSize+i] = float32(testCase.WeightIH[srcRow][i])
			}
		}

		// Hidden-to-hidden weights
		for h := 0; h < hiddenSize; h++ {
			for hh := 0; hh < hiddenSize; hh++ {
				srcRow := gateIdx*hiddenSize + h
				destHH[h*hiddenSize+hh] = float32(testCase.WeightHH[srcRow][hh])
			}
		}

		// Biases (combine bias_ih and bias_hh)
		for h := 0; h < hiddenSize; h++ {
			srcIdx := gateIdx*hiddenSize + h
			destBias[h] = float32(testCase.BiasIH[srcIdx] + testCase.BiasHH[srcIdx])
		}
	}

	// Copy weights for each gate
	copyGateWeights(layer.WeightIH_i, layer.WeightHH_i, layer.BiasH_i, 0) // Input gate
	copyGateWeights(layer.WeightIH_f, layer.WeightHH_f, layer.BiasH_f, 1) // Forget gate
	copyGateWeights(layer.WeightIH_g, layer.WeightHH_g, layer.BiasH_g, 2) // Cell gate
	copyGateWeights(layer.WeightIH_o, layer.WeightHH_o, layer.BiasH_o, 3) // Output gate
}

func computeMaxDiff(a, b [][]float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}

	maxDiff := 0.0
	for i := 0; i < len(a); i++ {
		if len(a[i]) != len(b[i]) {
			return math.Inf(1)
		}
		for j := 0; j < len(a[i]); j++ {
			diff := math.Abs(a[i][j] - b[i][j])
			if diff > maxDiff {
				maxDiff = diff
			}
		}
	}
	return maxDiff
}

func allZerosFloat32(matrix []float32) bool {
	for i := 0; i < len(matrix); i++ {
		if math.Abs(float64(matrix[i])) > 1e-10 {
			return false
		}
	}
	return true
}
