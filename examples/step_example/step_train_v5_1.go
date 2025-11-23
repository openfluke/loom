package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	nn "github.com/openfluke/loom/nn"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== LOOM Stepping Network: 5x5 Robust Classifier ===")
	fmt.Println("Task: Classify Shapes (Box, Cross, Dot)")
	fmt.Println("Fix: Using Tanh activation to prevent dead neurons")
	fmt.Println()

	// ARCHITECTURE:
	// 1. Conv2D: 5x5 Input -> 3x3 Output (No Padding).
	//    We use 8 filters to catch edges/corners.
	//    Activation: TANH (Critical for stability in this engine).
	// 2. Dense: 72 Inputs (3x3x8) -> 3 Outputs.

	networkJSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "conv2d",
				"input_height": 5,
				"input_width": 5,
				"input_channels": 1,
				"filters": 8,
				"kernel_size": 3,
				"stride": 1,
				"padding": 0,
				"output_height": 3,
				"output_width": 3,
				"activation": "tanh"
			},
			{
				"type": "dense",
				"input_height": 72,
				"output_height": 16,
				"activation": "tanh"
			},
			{
				"type": "dense",
				"input_height": 16,
				"output_height": 3,
				"activation": "linear"
			}
		]
	}`
	// Note: Layer 1 Output = 3*3*8 = 72

	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Fatalf("Failed to build network: %v", err)
	}
	net.InitializeWeights()

	state := net.InitStepState(25) // 5x5 flattened

	// --- Robust Shape Generators ---
	// We create slightly thicker/clearer shapes to help the CNN
	setPix := func(img []float32, r, c int, val float32) {
		if r >= 0 && r < 5 && c >= 0 && c < 5 {
			img[r*5+c] = val
		}
	}

	createBox := func() []float32 {
		img := make([]float32, 25)
		// Full Box
		for i := 1; i <= 3; i++ {
			setPix(img, 1, i, 1.0)
			setPix(img, 3, i, 1.0)
			setPix(img, i, 1, 1.0)
			setPix(img, i, 3, 1.0)
		}
		return img
	}

	createCross := func() []float32 {
		img := make([]float32, 25)
		// X shape
		setPix(img, 1, 1, 1.0)
		setPix(img, 1, 3, 1.0)
		setPix(img, 2, 2, 1.0)
		setPix(img, 3, 1, 1.0)
		setPix(img, 3, 3, 1.0)
		return img
	}

	createDot := func() []float32 {
		img := make([]float32, 25)
		// Dot (Plus neighbors to make it distinct from noise)
		setPix(img, 2, 2, 1.0)
		setPix(img, 2, 1, 0.5) // Fade out
		setPix(img, 2, 3, 0.5)
		setPix(img, 1, 2, 0.5)
		setPix(img, 3, 2, 0.5)
		return img
	}

	addNoise := func(img []float32, amount float32) []float32 {
		n := make([]float32, len(img))
		copy(n, img)
		for i := range n {
			n[i] += (rand.Float32() - 0.5) * amount
			if n[i] < 0 {
				n[i] = 0
			}
			if n[i] > 1 {
				n[i] = 1
			}
		}
		return n
	}

	trainingData := []struct {
		input  []float32
		target []float32
		label  string
	}{
		{createBox(), []float32{1, 0, 0}, "Box"},
		{addNoise(createBox(), 0.3), []float32{1, 0, 0}, "Box"},
		{createCross(), []float32{0, 1, 0}, "Cross"},
		{addNoise(createCross(), 0.3), []float32{0, 1, 0}, "Cross"},
		{createDot(), []float32{0, 0, 1}, "Dot"},
		{addNoise(createDot(), 0.3), []float32{0, 0, 1}, "Dot"},
	}

	// --- Training Loop ---
	totalSteps := 10000
	pipelineDepth := 3

	// Conservative Learning Rate for Tanh
	learningRate := float32(0.005)
	decay := float32(0.9998)
	minLR := float32(0.0005)

	fmt.Printf("Training for %d steps (Depth: %d)\n", totalSteps, pipelineDepth)
	startTime := time.Now()

	for step := 0; step < totalSteps; step++ {
		// Shuffle selection
		sample := trainingData[rand.Intn(len(trainingData))]

		state.SetInput(sample.input)

		// Settle pipeline
		for i := 0; i < pipelineDepth; i++ {
			net.StepForward(state)
		}

		output := state.GetOutput()

		// Softmax & Loss
		maxVal := output[0]
		for _, v := range output {
			if v > maxVal {
				maxVal = v
			}
		}
		sumExp := float32(0.0)
		exps := make([]float32, len(output))
		for i, v := range output {
			exps[i] = float32(math.Exp(float64(v - maxVal)))
			sumExp += exps[i]
		}

		gradOutput := make([]float32, len(output))
		loss := float32(0.0)
		for i := range output {
			probs := exps[i] / sumExp
			if sample.target[i] > 0.5 {
				loss -= float32(math.Log(float64(probs + 1e-7)))
			}
			gradOutput[i] = probs - sample.target[i]
		}

		net.StepBackward(state, gradOutput)
		net.ApplyGradients(learningRate)

		learningRate *= decay
		if learningRate < minLR {
			learningRate = minLR
		}

		if step%1000 == 0 {
			pred := 0
			for i := 1; i < len(output); i++ {
				if output[i] > output[pred] {
					pred = i
				}
			}
			exp := 0
			for i := 1; i < len(sample.target); i++ {
				if sample.target[i] > 0.5 {
					exp = i
				}
			}

			mark := "✗"
			if pred == exp {
				mark = "✓"
			}
			fmt.Printf("Step %-5d [%s] Loss: %.4f | Pred: %d Exp: %d | LR: %.5f\n",
				step, mark, loss, pred, exp, learningRate)
		}
	}

	elapsed := time.Since(startTime)
	fmt.Println("────────────────────────────────────────────────────────────")
	fmt.Printf("Training Complete in %v\n", elapsed)

	// --- Final Check ---
	fmt.Println("\n=== Final Evaluation ===")
	correct := 0
	for _, sample := range trainingData {
		state.SetInput(sample.input)
		for i := 0; i < pipelineDepth; i++ {
			net.StepForward(state)
		}

		output := state.GetOutput()
		pred := 0
		for i := 1; i < len(output); i++ {
			if output[i] > output[pred] {
				pred = i
			}
		}
		exp := 0
		for i := 1; i < len(sample.target); i++ {
			if sample.target[i] > 0.5 {
				exp = i
			}
		}

		mark := "✗"
		if pred == exp {
			correct++
			mark = "✓"
		}
		fmt.Printf("%s %-8s: Pred %d Exp %d\n", mark, sample.label, pred, exp)
	}
	fmt.Printf("Accuracy: %d/%d\n", correct, len(trainingData))
}
