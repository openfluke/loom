package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

// Neural Tween Experiment Demo
// Demonstrates the neural tweening system inspired by:
// - Flash ActionScript tweening (gradual morphing)
// - Network engineering link budgeting (signal loss estimation)
// - Optimal transport / Neural ODEs (continuous transformation)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║       Neural Tween Experiment - Weight Morphing Demo          ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════╣")
	fmt.Println("║ Concept: Like Flash tweening, gradually morph network weights ║")
	fmt.Println("║ from random initialization towards optimal configuration.     ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Create neural network
	fmt.Println("1. Creating neural network...")
	network := createNetwork()
	fmt.Printf("   ✓ Network: %d layers, input→hidden→output\n", network.TotalLayers())

	// Generate training data
	fmt.Println("\n2. Generating sample data...")
	inputs, expected := generateData(1000)
	fmt.Printf("   ✓ Generated %d samples (binary classification)\n", len(inputs))

	// Evaluate untrained network
	fmt.Println("\n3. Evaluating untrained network...")
	initialMetrics, err := network.EvaluateNetwork(inputs, expected)
	if err != nil {
		fmt.Printf("   ✗ Error: %v\n", err)
		return
	}
	fmt.Printf("   Initial Score: %.1f/100\n", initialMetrics.Score)
	fmt.Printf("   Average Deviation: %.1f%%\n", initialMetrics.AverageDeviation)
	printBucketBar(initialMetrics)

	// Initialize tween state
	fmt.Println("\n4. Initializing Neural Tween...")
	tweenState := nn.NewTweenState(network)
	fmt.Printf("   ✓ TweenState created, tracking %d layers\n", tweenState.TotalLayers)

	// Calculate initial link budgets
	tweenState.CalculateLinkBudgets(network, inputs[0])
	avgBudget, minBudget, maxBudget := tweenState.GetBudgetSummary()
	fmt.Printf("   Initial Link Budgets: avg=%.2f, min=%.2f, max=%.2f\n", avgBudget, minBudget, maxBudget)

	// Run neural tween training
	fmt.Println("\n5. Running Neural Tween Training...")
	fmt.Println("   ─────────────────────────────────────────────────────────────────────")

	epochs := 100
	learningRate := float32(0.5)
	momentum := float32(0.9)
	var finalMetrics *nn.DeviationMetrics

	tweenState.TweenTrain(network, inputs, expected, epochs, learningRate, momentum,
		func(epoch int, loss float32, metrics *nn.DeviationMetrics) {
			finalMetrics = metrics
			avgB, minB, maxB := tweenState.GetBudgetSummary()

			progress := float32(epoch) / float32(epochs)
			bar := progressBar(progress, 25)

			fmt.Printf("   Epoch %3d/%d [%s] | Score: %5.1f/100 | Loss: %6.3f | Budget: %.2f\n",
				epoch, epochs, bar, metrics.Score, loss, avgB)

			// Show bucket summary every 20 epochs
			if epoch%20 == 0 {
				fmt.Print("              Buckets: ")
				printBucketBarInline(metrics)
			}

			_ = minB
			_ = maxB
		})

	fmt.Println("   ─────────────────────────────────────────────────────────────────────")

	// Final evaluation
	fmt.Println("\n6. Final Results:")
	fmt.Printf("   Score Improvement: %.1f → %.1f (+%.1f%%)\n",
		initialMetrics.Score, finalMetrics.Score,
		finalMetrics.Score-initialMetrics.Score)
	fmt.Printf("   Deviation Change: %.1f%% → %.1f%%\n",
		initialMetrics.AverageDeviation, finalMetrics.AverageDeviation)

	// Print final bucket distribution
	fmt.Println("\n7. Final Deviation Bucket Distribution:")
	printBucketDetails(finalMetrics)

	// Check if we hit target
	if finalMetrics.Score >= 95 {
		fmt.Println("\n   🎉 SUCCESS! Achieved 95%+ accuracy!")
	} else if finalMetrics.Score >= 80 {
		fmt.Println("\n   ✓ Good progress! Consider more epochs for 95%+")
	} else {
		fmt.Println("\n   Still training... may need architecture changes")
	}

	// Print loss history
	fmt.Println("\n8. Loss History (lower is better):")
	printLossHistory(tweenState.LossHistory)

	fmt.Println("\n═══════════════════════════════════════════════════════════════")
	fmt.Println(" Neural Tweening Complete!")
	fmt.Println("═══════════════════════════════════════════════════════════════")
}

func createNetwork() *nn.Network {
	jsonConfig := `{
		"id": "tween_experiment",
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 3,
		"layers": [
			{
				"type": "dense",
				"activation": "tanh",
				"input_height": 8,
				"output_height": 32
			},
			{
				"type": "dense",
				"activation": "tanh",
				"input_height": 32,
				"output_height": 16
			},
			{
				"type": "dense",
				"activation": "sigmoid",
				"input_height": 16,
				"output_height": 2
			}
		]
	}`

	network, err := nn.BuildNetworkFromJSON(jsonConfig)
	if err != nil {
		panic(fmt.Sprintf("Failed to create network: %v", err))
	}

	network.InitializeWeights()
	return network
}

func generateData(numSamples int) ([][]float32, []float64) {
	inputs := make([][]float32, numSamples)
	expected := make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		input := make([]float32, 8)
		sum := float32(0)
		for j := 0; j < 8; j++ {
			input[j] = rand.Float32()
			sum += input[j]
		}
		inputs[i] = input

		// Binary classification: class based on sum threshold
		if sum < 4.0 {
			expected[i] = 0
		} else {
			expected[i] = 1
		}
	}

	return inputs, expected
}

func progressBar(progress float32, width int) string {
	filled := int(progress * float32(width))
	bar := ""
	for i := 0; i < width; i++ {
		if i < filled {
			bar += "█"
		} else {
			bar += "░"
		}
	}
	return bar
}

func printBucketBar(metrics *nn.DeviationMetrics) {
	fmt.Print("   Buckets: ")
	printBucketBarInline(metrics)
}

func printBucketBarInline(metrics *nn.DeviationMetrics) {
	bucketOrder := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	for _, name := range bucketOrder {
		bucket := metrics.Buckets[name]
		pct := float64(bucket.Count) / float64(metrics.TotalSamples) * 100
		if pct > 0 {
			fmt.Printf("[%s:%.0f%%] ", name, pct)
		}
	}
	fmt.Println()
}

func printBucketDetails(metrics *nn.DeviationMetrics) {
	bucketOrder := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	colors := []string{"🟢", "🟢", "🟡", "🟡", "🟠", "🔴", "⚫"}

	for i, name := range bucketOrder {
		bucket := metrics.Buckets[name]
		pct := float64(bucket.Count) / float64(metrics.TotalSamples) * 100
		bar := ""
		for j := 0; j < int(pct/5); j++ {
			bar += "█"
		}
		if bucket.Count > 0 {
			fmt.Printf("   %s %8s: %4d samples (%5.1f%%) %s\n",
				colors[i], name, bucket.Count, pct, bar)
		}
	}
}

func printLossHistory(history []float32) {
	if len(history) == 0 {
		fmt.Println("   (no history)")
		return
	}

	// Find min/max for scaling
	minLoss, maxLoss := history[0], history[0]
	for _, l := range history {
		if l < minLoss {
			minLoss = l
		}
		if l > maxLoss {
			maxLoss = l
		}
	}

	// Print ASCII chart
	height := 5
	width := len(history)
	if width > 40 {
		width = 40
	}

	for h := height - 1; h >= 0; h-- {
		threshold := minLoss + (maxLoss-minLoss)*float32(h)/float32(height-1)
		fmt.Print("   ")
		for i := 0; i < width; i++ {
			idx := i * len(history) / width
			if history[idx] >= threshold {
				fmt.Print("█")
			} else {
				fmt.Print(" ")
			}
		}
		fmt.Println()
	}
	fmt.Printf("   └%s┘\n", repeatStr("─", width-2))
	fmt.Printf("    Start → End (%.2f → %.2f)\n", history[0], history[len(history)-1])
}

func repeatStr(s string, n int) string {
	result := ""
	for i := 0; i < n; i++ {
		result += s
	}
	return result
}
