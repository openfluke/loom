package main

import (
	"fmt"
	"strings"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("🧠 LOOM - BERT Text Processing Demo")
	fmt.Println(strings.Repeat("=", 60))

	// Load the converted BERT-Tiny model
	fmt.Println("\n📂 Loading bert-tiny.json...")
	network, err := nn.LoadImportedModel("../bert-tiny.json", "bert-tiny")
	if err != nil || network == nil {
		fmt.Printf("❌ Failed to load model: %v\n", err)
		return
	}

	fmt.Printf("✅ Model loaded!\n")
	fmt.Printf("   Total layers: %d\n", network.TotalLayers())
	fmt.Printf("   Input size: %d\n", network.InputSize)

	// Test sentences
	sentences := []string{
		"Hello world",
		"The cat sat on the mat",
		"Machine learning is amazing",
	}

	for i, sentence := range sentences {
		fmt.Printf("\n%s\n", strings.Repeat("─", 60))
		fmt.Printf("📝 Sentence %d: \"%s\"\n", i+1, sentence)
		fmt.Println(strings.Repeat("─", 60))

		// Simple tokenization (word-based for demo)
		tokens := simpleTokenize(sentence)
		fmt.Printf("\n🔤 Tokens (%d): %v\n", len(tokens), tokens)

		// Create embeddings (simplified - using random embeddings for demo)
		// In real BERT, these come from learned embedding tables
		seqLength := 128
		hiddenSize := 128
		embeddings := createSimpleEmbeddings(tokens, seqLength, hiddenSize)

		fmt.Printf("📊 Input shape: [%d tokens × %d hidden] = %d values\n",
			seqLength, hiddenSize, len(embeddings))

		// Run forward pass
		fmt.Println("\n▶️  Running BERT forward pass...")
		output, _ := network.ForwardCPU(embeddings)

		if output == nil || len(output) == 0 {
			fmt.Println("❌ Forward pass failed")
			continue
		}

		fmt.Println("✅ Forward pass complete!")

		// Analyze output per token position
		fmt.Printf("\n📈 Output Analysis:\n")
		analyzeOutput(output, tokens, hiddenSize)

		// Show summary statistics
		var sum, min, max float32
		min, max = output[0], output[0]
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

		fmt.Printf("\n📊 Global Statistics:\n")
		fmt.Printf("   Mean: %.6f\n", mean)
		fmt.Printf("   Min:  %.6f\n", min)
		fmt.Printf("   Max:  %.6f\n", max)
	}

	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Println("💡 Note: This demo uses simplified embeddings.")
	fmt.Println("   For real text processing, you need:")
	fmt.Println("   1. WordPiece tokenizer (BERT vocab)")
	fmt.Println("   2. Learned token embeddings")
	fmt.Println("   3. Position embeddings")
	fmt.Println("   4. Task-specific output layer")
}

// simpleTokenize does basic word tokenization
func simpleTokenize(text string) []string {
	// Convert to lowercase and split on spaces
	text = strings.ToLower(text)
	words := strings.Fields(text)

	// Add [CLS] at start and [SEP] at end (BERT convention)
	tokens := make([]string, 0, len(words)+2)
	tokens = append(tokens, "[CLS]")
	tokens = append(tokens, words...)
	tokens = append(tokens, "[SEP]")

	return tokens
}

// createSimpleEmbeddings creates dummy embeddings for tokens
// In real BERT, these come from learned embedding tables + position embeddings
func createSimpleEmbeddings(tokens []string, seqLength, hiddenSize int) []float32 {
	embeddings := make([]float32, seqLength*hiddenSize)

	// Fill with simple pattern based on token content
	for i := 0; i < len(tokens) && i < seqLength; i++ {
		token := tokens[i]

		// Generate simple embedding based on token characters
		// (This is NOT how real BERT works - just for visualization)
		for j := 0; j < hiddenSize; j++ {
			idx := i*hiddenSize + j

			// Mix of position-based and token-based patterns
			positionFactor := float32(i) / float32(seqLength)
			tokenFactor := float32(len(token)) / 10.0

			if j < hiddenSize/4 {
				// First quarter: position encoding
				embeddings[idx] = positionFactor
			} else if j < hiddenSize/2 {
				// Second quarter: token length
				embeddings[idx] = tokenFactor
			} else if j < 3*hiddenSize/4 {
				// Third quarter: character sum
				charSum := float32(0)
				for _, c := range token {
					charSum += float32(c) / 1000.0
				}
				embeddings[idx] = charSum
			} else {
				// Last quarter: mixed
				embeddings[idx] = (positionFactor + tokenFactor) / 2.0
			}
		}
	}

	// Padding positions get zero embeddings
	for i := len(tokens); i < seqLength; i++ {
		for j := 0; j < hiddenSize; j++ {
			embeddings[i*hiddenSize+j] = 0.0
		}
	}

	return embeddings
}

// analyzeOutput shows per-token output statistics
func analyzeOutput(output []float32, tokens []string, hiddenSize int) {
	numTokens := len(output) / hiddenSize

	// Show stats for each token (up to the actual tokens + a few padding)
	displayLimit := len(tokens) + 2
	if displayLimit > numTokens {
		displayLimit = numTokens
	}

	for i := 0; i < displayLimit; i++ {
		start := i * hiddenSize
		end := start + hiddenSize

		if start >= len(output) {
			break
		}
		if end > len(output) {
			end = len(output)
		}

		tokenVec := output[start:end]

		// Calculate statistics for this token
		var sum, min, max float32
		min, max = tokenVec[0], tokenVec[0]
		nonZero := 0

		for _, v := range tokenVec {
			sum += v
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
			if v != 0 {
				nonZero++
			}
		}

		mean := sum / float32(len(tokenVec))

		// Show token info
		tokenName := "[PAD]"
		if i < len(tokens) {
			tokenName = tokens[i]
		}

		fmt.Printf("   Token %2d %-12s | mean: %7.4f | min: %7.4f | max: %7.4f | active: %d/%d\n",
			i, fmt.Sprintf("\"%s\"", tokenName), mean, min, max, nonZero, hiddenSize)
	}
}
