package main

import (
	"fmt"
	"math"
)

func main() {
	fmt.Println("=== Softmax: Global vs Grid ===")
	fmt.Println()

	// Example: 4 units, each can take 3 actions
	// Raw network output: 12 values (4 units × 3 actions)
	logits := []float32{
		// Unit 0: [attack=0.8, defend=0.3, move=0.1]
		0.8, 0.3, 0.1,
		// Unit 1: [attack=0.2, defend=0.9, move=0.4]
		0.2, 0.9, 0.4,
		// Unit 2: [attack=0.5, defend=0.5, move=0.6]
		0.5, 0.5, 0.6,
		// Unit 3: [attack=0.1, defend=0.2, move=0.7]
		0.1, 0.2, 0.7,
	}

	actions := []string{"attack", "defend", "move"}
	numUnits := 4
	numActions := 3

	fmt.Println("Raw logits from network:")
	printGrid("Logits", logits, numUnits, numActions)
	fmt.Println()

	// Method 1: Global softmax (WRONG for multi-agent)
	fmt.Println("Method 1: GLOBAL SOFTMAX (treats all 12 values as one distribution)")
	globalProbs := softmax(logits)
	printGrid("Global Probs", globalProbs, numUnits, numActions)
	fmt.Printf("Sum of all probs: %.6f\n", sum(globalProbs))
	fmt.Println("❌ Problem: All units compete! Unit 1's high 'defend' suppresses Unit 0's 'attack'")
	fmt.Println()

	// Method 2: Grid softmax (CORRECT for multi-agent)
	fmt.Println("Method 2: GRID SOFTMAX (per-unit distributions)")
	gridProbs := softmaxGrid(logits, numUnits, numActions)
	printGrid("Grid Probs", gridProbs, numUnits, numActions)
	fmt.Println("✅ Each row sums to 1.0 independently:")
	for unit := 0; unit < numUnits; unit++ {
		rowStart := unit * numActions
		rowEnd := rowStart + numActions
		rowSum := sum(gridProbs[rowStart:rowEnd])
		fmt.Printf("   Unit %d: %.6f\n", unit, rowSum)
	}
	fmt.Println()

	// Show chosen actions
	fmt.Println("Chosen actions per unit:")
	for unit := 0; unit < numUnits; unit++ {
		rowStart := unit * numActions
		rowEnd := rowStart + numActions
		unitProbs := gridProbs[rowStart:rowEnd]
		actionIdx := argmax(unitProbs)
		fmt.Printf("  Unit %d: %s (prob=%.3f)\n", unit, actions[actionIdx], unitProbs[actionIdx])
	}
	fmt.Println()

	// Use case examples
	fmt.Println("=== When to Use Each ===")
	fmt.Println()
	fmt.Println("GLOBAL Softmax (regular):")
	fmt.Println("  - Single agent picking ONE action from many")
	fmt.Println("  - Example: Chess move selection (pick 1 move)")
	fmt.Println("  - Example: Image classification (pick 1 class)")
	fmt.Println()
	fmt.Println("GRID Softmax (spatial):")
	fmt.Println("  - Multiple agents each picking their own action")
	fmt.Println("  - Example: StarCraft (each unit picks independently)")
	fmt.Println("  - Example: Multi-finger robot gripper (each finger independent)")
	fmt.Println("  - Example: Echo VR team (each player picks action)")
	fmt.Println()
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

func sum(v []float32) float32 {
	s := float32(0.0)
	for _, val := range v {
		s += val
	}
	return s
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
	if len(logits) != rows*cols {
		panic(fmt.Sprintf("softmaxGrid: size mismatch %d != %d*%d", len(logits), rows, cols))
	}

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
	fmt.Printf("%s (%dx%d):\n", name, rows, cols)
	for r := 0; r < rows; r++ {
		fmt.Printf("  Unit %d: ", r)
		for c := 0; c < cols; c++ {
			idx := r*cols + c
			fmt.Printf("%.3f ", values[idx])
		}
		fmt.Println()
	}
}
