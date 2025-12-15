package main

import (
	"fmt"
	"math/rand"

	"github.com/openfluke/loom/nn"
)

// Tween Evaluation Demo
// Demonstrates neural network evaluation with deviation buckets
// to visualize how well predictions match expected values.

func main() {
	fmt.Println("=== Tween - Neural Network Evaluation Demo ===\n")

	// Create a simple neural network for binary classification
	fmt.Println("1. Creating neural network...")
	network := createNetwork()
	fmt.Printf("   ✓ Network created: %d rows × %d cols = %d total layers\n",
		network.GridRows, network.GridCols, network.TotalLayers())

	// Generate sample data for evaluation
	fmt.Println("\n2. Generating sample data...")
	inputs, expectedOutputs := generateSampleData(1000)
	fmt.Printf("   ✓ Generated %d samples\n", len(inputs))

	// Run evaluation with deviation metrics
	fmt.Println("\n3. Running evaluation...")
	metrics, err := network.EvaluateNetwork(inputs, expectedOutputs)
	if err != nil {
		fmt.Printf("   ✗ Evaluation failed: %v\n", err)
		return
	}

	// Display the deviation bucket distribution
	fmt.Println("\n4. Deviation Bucket Analysis:")
	displayBucketDetails(metrics)

	// Print the standard summary
	metrics.PrintSummary()

	// Show worst performing samples
	fmt.Println("\n5. Worst 5 Predictions:")
	worst := metrics.GetWorstSamples(5)
	for _, pred := range worst {
		fmt.Printf("   Sample #%d: Expected %.2f, Got %.2f, Deviation: %.1f%% [%s]\n",
			pred.SampleIndex, pred.ExpectedOutput, pred.ActualOutput,
			pred.Deviation, pred.Bucket)
	}

	// Show sample distribution per bucket
	fmt.Println("\n6. Sample Indices by Bucket:")
	bucketOrder := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	for _, bucketName := range bucketOrder {
		samples := metrics.GetSamplesInBucket(bucketName)
		if len(samples) > 0 {
			fmt.Printf("   %8s: %v\n", bucketName, truncateSamples(samples, 10))
		}
	}

	// Save metrics to file
	fmt.Println("\n7. Saving evaluation metrics...")
	err = metrics.SaveMetrics("evaluation_results.json")
	if err != nil {
		fmt.Printf("   ✗ Failed to save metrics: %v\n", err)
	} else {
		fmt.Println("   ✓ Saved to evaluation_results.json")
	}

	fmt.Println("\n=== Tween Evaluation Complete ===")
}

// createNetwork builds a simple neural network for demonstration
func createNetwork() *nn.Network {
	jsonConfig := `{
		"id": "tween_demo",
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 2,
		"layers": [
			{
				"type": "dense",
				"activation": "relu",
				"input_height": 8,
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

	// Initialize weights before use
	network.InitializeWeights()

	return network
}

// generateSampleData creates random input samples and expected outputs
// for binary classification (0 or 1)
func generateSampleData(numSamples int) ([][]float32, []float64) {
	inputs := make([][]float32, numSamples)
	expectedOutputs := make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		// Generate 8 input features
		input := make([]float32, 8)
		sum := float32(0)
		for j := 0; j < 8; j++ {
			input[j] = rand.Float32()
			sum += input[j]
		}
		inputs[i] = input

		// Binary classification: class 0 if sum < 4, class 1 otherwise
		if sum < 4 {
			expectedOutputs[i] = 0
		} else {
			expectedOutputs[i] = 1
		}
	}

	return inputs, expectedOutputs
}

// displayBucketDetails shows detailed information about each deviation bucket
func displayBucketDetails(metrics *nn.DeviationMetrics) {
	bucketOrder := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}

	fmt.Println("   ┌─────────────┬───────┬────────────┬─────────────────┐")
	fmt.Println("   │   Bucket    │ Count │ Percentage │   Quality       │")
	fmt.Println("   ├─────────────┼───────┼────────────┼─────────────────┤")

	for _, bucketName := range bucketOrder {
		bucket := metrics.Buckets[bucketName]
		percentage := float64(bucket.Count) / float64(metrics.TotalSamples) * 100

		// Color coding indicator
		var quality string
		switch bucketName {
		case "0-10%":
			quality = "🟢 Excellent"
		case "10-20%":
			quality = "🟢 Very Good"
		case "20-30%":
			quality = "🟡 Good"
		case "30-40%":
			quality = "🟡 Moderate"
		case "40-50%":
			quality = "🟠 Acceptable"
		case "50-100%":
			quality = "🔴 Poor"
		case "100%+":
			quality = "⚫ Failed"
		}

		fmt.Printf("   │ %9s   │ %5d │ %8.1f%%  │ %s\t│\n",
			bucketName, bucket.Count, percentage, quality)
	}

	fmt.Println("   └─────────────┴───────┴────────────┴─────────────────┘")
}

// truncateSamples limits the display of sample indices
func truncateSamples(samples []int, maxShow int) string {
	if len(samples) <= maxShow {
		return fmt.Sprintf("%v", samples)
	}
	return fmt.Sprintf("%v... (%d more)", samples[:maxShow], len(samples)-maxShow)
}
