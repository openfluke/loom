package main

import (
	"fmt"
	"math"
	"strings"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("🧠 LOOM - BERT Tiny Model Test")
	fmt.Println(strings.Repeat("=", 60))

	// Load the converted BERT-Tiny model
	fmt.Println("\n📂 Loading bert-tiny.json...")
	network, err := nn.LoadImportedModel("bert-tiny.json", "bert-tiny")
	if err != nil || network == nil {
		fmt.Printf("❌ Failed to load model: %v\n", err)
		return
	}

	fmt.Printf("✅ Model loaded!\n")
	fmt.Printf("   Total layers: %d\n", network.TotalLayers())
	fmt.Printf("   Input size: %d\n", network.InputSize)

	// Create dummy input (in real use, this comes from text embeddings)
	// For BERT-Tiny: seq_length=128, hidden_size=128
	seqLength := 128
	hiddenSize := 128
	inputSize := seqLength * hiddenSize

	fmt.Printf("\n🎲 Creating input (%d x %d = %d values)...\n", seqLength, hiddenSize, inputSize)
	input := make([]float32, inputSize)

	// Fill with sample values (normally from embedding layer)
	for i := range input {
		input[i] = 0.1 * float32(i%10-5) // Simple pattern
	}

	// Run forward pass
	fmt.Println("\n▶️  Running forward pass...")
	output, _, err := network.ForwardGPU(input)
	if err != nil {
		output, _ = network.ForwardCPU(input)
	}

	if output == nil || len(output) == 0 {
		fmt.Println("❌ Forward pass failed")
		return
	}

	fmt.Println("✅ Forward pass complete!")
	fmt.Printf("   Output size: %d\n", len(output))

	// Show output statistics
	fmt.Println("\n📊 Output statistics:")

	var sum, min, max float32
	min = output[0]
	max = output[0]

	for _, v := range output {
		sum += v
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	mean := sum / float32(len(output))

	// Calculate std dev
	var variance float32
	for _, v := range output {
		diff := v - mean
		variance += diff * diff
	}
	std := float32(math.Sqrt(float64(variance / float32(len(output)))))

	fmt.Printf("   Mean: %.6f\n", mean)
	fmt.Printf("   Std:  %.6f\n", std)
	fmt.Printf("   Min:  %.6f\n", min)
	fmt.Printf("   Max:  %.6f\n", max)

	// Show first 10 output values
	fmt.Println("\n📈 First 10 output values:")
	for i := 0; i < 10 && i < len(output); i++ {
		fmt.Printf("   [%d] %.6f\n", i, output[i])
	}

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("🎉 Success! Pre-trained BERT-Tiny working in LOOM!")
	fmt.Println("\n💡 Next steps:")
	fmt.Println("   1. Add proper text embedding layer")
	fmt.Println("   2. Fine-tune on your task")
	fmt.Println("   3. Deploy to mobile with MAUI/C#")
}
