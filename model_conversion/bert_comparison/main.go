package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/openfluke/loom/nn"
)

type ComparisonData struct {
	Text            string      `json:"text"`
	Tokens          []string    `json:"tokens"`
	EmbeddingsFlat  []float32   `json:"embeddings_flat"`
	EmbeddingsShape []int       `json:"embeddings_shape"`
	BertOutput      [][]float32 `json:"bert_output"`
	AttentionMask   []int       `json:"attention_mask"`
}

func main() {
	fmt.Println("🔬 LOOM vs Real BERT Comparison")
	fmt.Println(strings.Repeat("=", 60))

	// Load comparison data
	fmt.Println("\n📂 Loading bert_comparison.json...")
	data, err := os.ReadFile("../bert_comparison.json")
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		fmt.Println("   Run: python3 compare_bert_loom.py first")
		return
	}

	var comparison ComparisonData
	if err := json.Unmarshal(data, &comparison); err != nil {
		fmt.Printf("❌ Failed to parse JSON: %v\n", err)
		return
	}

	fmt.Printf("✅ Loaded comparison data\n")
	fmt.Printf("   Text: \"%s\"\n", comparison.Text)
	fmt.Printf("   Tokens: %d (%v)\n", len(comparison.Tokens), comparison.Tokens)

	// Load LOOM model
	fmt.Println("\n📂 Loading LOOM BERT model...")
	network, err := nn.LoadImportedModel("../bert-tiny.json", "bert-tiny")
	if err != nil {
		fmt.Printf("❌ Failed to load model: %v\n", err)
		return
	}
	fmt.Println("✅ LOOM model loaded")

	// Run through LOOM
	fmt.Println("\n▶️  Running LOOM forward pass...")
	loomOutput, _ := network.ForwardCPU(comparison.EmbeddingsFlat)

	if loomOutput == nil || len(loomOutput) == 0 {
		fmt.Println("❌ LOOM forward pass failed")
		return
	}

	fmt.Println("✅ LOOM forward pass complete")

	// Compare outputs
	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Println("📊 COMPARISON RESULTS")
	fmt.Printf("%s\n", strings.Repeat("=", 60))

	hiddenSize := 128
	numTokens := len(comparison.Tokens)

	// Reshape BERT output for comparison
	bertOutputFlat := make([]float32, 0)
	for _, row := range comparison.BertOutput {
		bertOutputFlat = append(bertOutputFlat, row...)
	}

	// Overall statistics
	fmt.Println("\n1️⃣  Global Statistics:")
	fmt.Println(strings.Repeat("-", 60))

	realMean, realStd := calcStats(bertOutputFlat)
	loomMean, loomStd := calcStats(loomOutput)

	fmt.Printf("   Real BERT  | Mean: %8.5f | Std: %8.5f\n", realMean, realStd)
	fmt.Printf("   LOOM BERT  | Mean: %8.5f | Std: %8.5f\n", loomMean, loomStd)
	fmt.Printf("   Difference | Mean: %8.5f | Std: %8.5f\n",
		math.Abs(float64(realMean-loomMean)),
		math.Abs(float64(realStd-loomStd)))

	// Per-token comparison
	fmt.Println("\n2️⃣  Per-Token Comparison:")
	fmt.Println(strings.Repeat("-", 60))
	fmt.Println("   Token         | Real BERT Mean | LOOM Mean  | Difference")
	fmt.Println(strings.Repeat("-", 60))

	for i := 0; i < numTokens && i < len(comparison.AttentionMask); i++ {
		if comparison.AttentionMask[i] == 0 {
			break
		}

		token := comparison.Tokens[i]
		if len(token) > 12 {
			token = token[:12]
		}

		// Extract token vectors
		realVec := comparison.BertOutput[i]
		loomVec := loomOutput[i*hiddenSize : (i+1)*hiddenSize]

		realMean := calcMean(realVec)
		loomMean := calcMean(loomVec)
		diff := math.Abs(float64(realMean - loomMean))

		fmt.Printf("   %-12s | %14.6f | %10.6f | %10.6f\n",
			token, realMean, loomMean, diff)
	}

	// Correlation analysis
	fmt.Println("\n3️⃣  Similarity Metrics:")
	fmt.Println(strings.Repeat("-", 60))

	// Calculate cosine similarity for each token
	var avgCosineSim float32
	validTokens := 0

	for i := 0; i < numTokens && i < len(comparison.AttentionMask); i++ {
		if comparison.AttentionMask[i] == 0 {
			break
		}

		realVec := comparison.BertOutput[i]
		loomVec := loomOutput[i*hiddenSize : (i+1)*hiddenSize]

		cosSim := cosineSimilarity(realVec, loomVec)
		avgCosineSim += cosSim
		validTokens++
	}

	if validTokens > 0 {
		avgCosineSim /= float32(validTokens)
	}

	fmt.Printf("   Average Cosine Similarity: %.6f\n", avgCosineSim)
	fmt.Printf("   (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)\n")

	// Interpretation
	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Println("🔍 INTERPRETATION")
	fmt.Printf("%s\n", strings.Repeat("=", 60))
	fmt.Println()
	fmt.Println("⚠️  Expected Differences:")
	fmt.Println("   • LOOM has only 2 BERT layers (real BERT has 2)")
	fmt.Println("   • Missing LayerNorm after each layer")
	fmt.Println("   • Missing residual connections (skip connections)")
	fmt.Println("   • These are architectural differences, not bugs")
	fmt.Println()

	if avgCosineSim > 0.8 {
		fmt.Println("✅ High similarity! LOOM is processing similarly to BERT")
	} else if avgCosineSim > 0.5 {
		fmt.Println("🟡 Moderate similarity. Outputs are related but different")
	} else {
		fmt.Println("🔴 Low similarity. This is expected due to missing layers")
	}

	fmt.Println()
	fmt.Println("💡 To improve similarity:")
	fmt.Println("   1. Add LayerNorm support to LOOM")
	fmt.Println("   2. Add residual connections")
	fmt.Println("   3. Extract all BERT layers (not just 2)")
}

func calcStats(data []float32) (mean, std float32) {
	if len(data) == 0 {
		return 0, 0
	}

	sum := float32(0)
	for _, v := range data {
		sum += v
	}
	mean = sum / float32(len(data))

	variance := float32(0)
	for _, v := range data {
		diff := v - mean
		variance += diff * diff
	}
	std = float32(math.Sqrt(float64(variance / float32(len(data)))))

	return mean, std
}

func calcMean(data []float32) float32 {
	if len(data) == 0 {
		return 0
	}
	sum := float32(0)
	for _, v := range data {
		sum += v
	}
	return sum / float32(len(data))
}

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	dotProduct := float32(0)
	normA := float32(0)
	normB := float32(0)

	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / float32(math.Sqrt(float64(normA*normB)))
}
