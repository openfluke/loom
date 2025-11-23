package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	nn "github.com/openfluke/loom/nn"
)

// TargetQueue handles the delay between input and output in the stepping network
type TargetQueue struct {
	targets [][]float32
	maxSize int
}

func NewTargetQueue(size int) *TargetQueue {
	return &TargetQueue{
		targets: make([][]float32, 0, size),
		maxSize: size,
	}
}

func (q *TargetQueue) Push(target []float32) {
	q.targets = append(q.targets, target)
}

func (q *TargetQueue) Pop() []float32 {
	if len(q.targets) == 0 {
		return nil
	}
	t := q.targets[0]
	q.targets = q.targets[1:]
	return t
}

func (q *TargetQueue) IsFull() bool {
	return len(q.targets) >= q.maxSize
}

// WordEmbedding creates simple word embeddings from vocabulary
type WordEmbedding struct {
	vocab      map[string]int
	embeddings [][]float32
	embedSize  int
}

func NewWordEmbedding(words []string, embedSize int) *WordEmbedding {
	vocab := make(map[string]int)
	vocab["<UNK>"] = 0
	vocab["<PAD>"] = 1

	for _, word := range words {
		if _, exists := vocab[word]; !exists {
			vocab[word] = len(vocab)
		}
	}

	// Initialize random embeddings
	embeddings := make([][]float32, len(vocab))
	for i := range embeddings {
		embeddings[i] = make([]float32, embedSize)
		for j := range embeddings[i] {
			embeddings[i][j] = (rand.Float32() - 0.5) * 0.1
		}
	}

	return &WordEmbedding{
		vocab:      vocab,
		embeddings: embeddings,
		embedSize:  embedSize,
	}
}

func (we *WordEmbedding) Encode(word string) []float32 {
	idx, exists := we.vocab[word]
	if !exists {
		idx = 0 // <UNK>
	}
	return we.embeddings[idx]
}

func (we *WordEmbedding) VocabSize() int {
	return len(we.vocab)
}

// SentimentSample represents a text sample with sentiment label
type SentimentSample struct {
	text      string
	words     []string
	sentiment int // 0=negative, 1=neutral, 2=positive
	label     string
}

// loadAliceTextSamples creates sentiment-labeled samples from Alice in Wonderland
func loadAliceTextSamples() []SentimentSample {
	file, err := os.Open("alice.txt")
	if err != nil {
		log.Fatalf("Failed to open alice.txt: %v", err)
	}
	defer file.Close()

	var samples []SentimentSample
	scanner := bufio.NewScanner(file)

	// Read lines and create samples with synthetic sentiment labels
	// We'll use simple heuristics based on keywords
	positiveWords := map[string]bool{
		"wonderful": true, "beautiful": true, "lovely": true, "happy": true,
		"delighted": true, "pleased": true, "curious": true, "pretty": true,
		"nice": true, "good": true, "great": true, "excellent": true,
	}

	negativeWords := map[string]bool{
		"afraid": true, "frightened": true, "angry": true, "sad": true,
		"terrible": true, "horrible": true, "dreadful": true, "anxious": true,
		"worried": true, "upset": true, "bad": true, "poor": true,
	}

	for scanner.Scan() {
		line := scanner.Text()
		line = strings.TrimSpace(line)

		// Skip empty lines and chapter headers
		if len(line) < 20 || strings.HasPrefix(line, "CHAPTER") {
			continue
		}

		words := strings.Fields(strings.ToLower(line))
		if len(words) < 5 || len(words) > 30 {
			continue
		}

		// Determine sentiment based on keywords
		posCount := 0
		negCount := 0
		for _, word := range words {
			word = strings.Trim(word, ".,!?;:\"'")
			if positiveWords[word] {
				posCount++
			}
			if negativeWords[word] {
				negCount++
			}
		}

		sentiment := 1 // neutral
		label := "Neutral"

		if posCount > negCount {
			sentiment = 2
			label = "Positive"
		} else if negCount > posCount {
			sentiment = 0
			label = "Negative"
		}

		samples = append(samples, SentimentSample{
			text:      line,
			words:     words,
			sentiment: sentiment,
			label:     label,
		})

		if len(samples) >= 200 {
			break
		}
	}

	return samples
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== LOOM Stepping Neural Network v7: Sentiment Analysis ===")
	fmt.Println("3-Layer Network: Dense(Embedding) -> LSTM -> Dense(Classifier)")
	fmt.Println("Task: Real-world sentiment analysis on Alice in Wonderland text")
	fmt.Println()

	// 1. Load and prepare data
	fmt.Println("Loading data from alice.txt...")
	samples := loadAliceTextSamples()
	fmt.Printf("Loaded %d samples\n", len(samples))

	if len(samples) == 0 {
		log.Fatal("No samples loaded. Make sure alice.txt exists in the same directory.")
	}

	// Build vocabulary from all words
	allWords := []string{}
	for _, sample := range samples {
		allWords = append(allWords, sample.words...)
	}

	embedSize := 16
	embedding := NewWordEmbedding(allWords, embedSize)
	fmt.Printf("Vocabulary size: %d words\n", embedding.VocabSize())
	fmt.Printf("Embedding dimension: %d\n", embedSize)
	fmt.Println()

	// 2. Define Network Architecture
	// Input: word embedding (16) -> Dense(32) -> LSTM(48) -> Dense(3 classes)
	networkJSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_height": 16,
				"output_height": 32,
				"activation": "relu"
			},
			{
				"type": "lstm",
				"input_size": 32,
				"hidden_size": 48,
				"seq_length": 1,
				"activation": "tanh"
			},
			{
				"type": "dense",
				"input_height": 48,
				"output_height": 3,
				"activation": "softmax"
			}
		]
	}`

	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Fatalf("Failed to build network: %v", err)
	}
	net.InitializeWeights()

	// Initialize stepping state
	state := net.InitStepState(embedSize)

	// 3. Setup Training
	totalSteps := 50000
	targetDelay := 3 // 3-layer network depth
	targetQueue := NewTargetQueue(targetDelay)

	learningRate := float32(0.02)
	minLearningRate := float32(0.001)
	decayRate := float32(0.99998)
	gradientClipValue := float32(2.0)

	fmt.Printf("Training Configuration:\n")
	fmt.Printf("  Total steps: %d\n", totalSteps)
	fmt.Printf("  Target delay: %d steps\n", targetDelay)
	fmt.Printf("  Initial learning rate: %.4f\n", learningRate)
	fmt.Printf("  LR decay: %.6f (min %.4f)\n", decayRate, minLearningRate)
	fmt.Printf("  Gradient clipping: %.2f\n", gradientClipValue)
	fmt.Println()

	// 4. Training Loop
	fmt.Printf("%-8s %-12s %-15s %-30s %-10s\n", "Step", "Sample", "Prediction", "Text Preview", "Loss")
	fmt.Println("─────────────────────────────────────────────────────────────────────────────────────")

	startTime := time.Now()
	stepCount := 0
	currentSampleIdx := 0
	currentWordIdx := 0

	// Statistics
	correctPredictions := 0
	totalPredictions := 0

	for stepCount < totalSteps {
		// Select sample (change every 50 steps to process multiple words from same sentence)
		if stepCount%50 == 0 {
			currentSampleIdx = rand.Intn(len(samples))
			currentWordIdx = 0
		}

		sample := samples[currentSampleIdx]

		// Get current word embedding as input
		if currentWordIdx >= len(sample.words) {
			currentWordIdx = 0
		}
		word := sample.words[currentWordIdx]
		inputEmbed := embedding.Encode(word)

		// Create target (one-hot for sentiment)
		target := make([]float32, 3)
		target[sample.sentiment] = 1.0

		// Set input and step forward
		state.SetInput(inputEmbed)
		net.StepForward(state)

		// Manage target queue
		targetQueue.Push(target)

		if targetQueue.IsFull() {
			delayedTarget := targetQueue.Pop()
			output := state.GetOutput()

			// Calculate loss & gradient (cross-entropy)
			loss := float32(0.0)
			gradOutput := make([]float32, len(output))

			for i := 0; i < len(output); i++ {
				p := output[i]
				if p < 1e-7 {
					p = 1e-7
				}
				if p > 1.0-1e-7 {
					p = 1.0 - 1e-7
				}

				if delayedTarget[i] > 0.5 {
					loss -= float32(math.Log(float64(p)))
				}

				gradOutput[i] = output[i] - delayedTarget[i]
			}

			// Gradient clipping
			gradNorm := float32(0.0)
			for _, g := range gradOutput {
				gradNorm += g * g
			}
			gradNorm = float32(math.Sqrt(float64(gradNorm)))

			if gradNorm > gradientClipValue {
				scale := gradientClipValue / gradNorm
				for i := range gradOutput {
					gradOutput[i] *= scale
				}
			}

			// Backward pass and update
			net.StepBackward(state, gradOutput)
			net.ApplyGradients(learningRate)

			// Decay learning rate
			learningRate *= decayRate
			if learningRate < minLearningRate {
				learningRate = minLearningRate
			}

			// Track accuracy
			predIdx := 0
			for i := 1; i < len(output); i++ {
				if output[i] > output[predIdx] {
					predIdx = i
				}
			}

			targetIdx := sample.sentiment
			if predIdx == targetIdx {
				correctPredictions++
			}
			totalPredictions++

			// Logging
			if stepCount%1000 == 0 {
				predLabels := []string{"Negative", "Neutral", "Positive"}
				predLabel := predLabels[predIdx]

				textPreview := sample.text
				if len(textPreview) > 25 {
					textPreview = textPreview[:25] + "..."
				}

				mark := "✗"
				if predIdx == targetIdx {
					mark = "✓"
				}

				accuracy := float32(0.0)
				if totalPredictions > 0 {
					accuracy = float32(correctPredictions) / float32(totalPredictions) * 100
				}

				fmt.Printf("%-8d %-12s %-15s %-30s Loss: %.4f Acc: %.1f%% %s\n",
					stepCount, sample.label, predLabel, textPreview, loss, accuracy, mark)
			}
		}

		currentWordIdx++
		stepCount++
	}

	totalTime := time.Since(startTime)
	fmt.Println()
	fmt.Println("=== Training Complete ===")
	fmt.Printf("Total Time: %v\n", totalTime)
	fmt.Printf("Speed: %.2f steps/sec\n", float64(totalSteps)/totalTime.Seconds())
	fmt.Printf("Final Accuracy: %.2f%% (%d/%d)\n",
		float32(correctPredictions)/float32(totalPredictions)*100,
		correctPredictions, totalPredictions)
	fmt.Println()

	// 5. Evaluation on test samples
	fmt.Println("=== Evaluation on Sample Sentences ===")

	// Test on a few samples with settling time
	settlingSteps := 15
	testSamples := samples[:min(10, len(samples))]

	correctTest := 0
	for _, sample := range testSamples {
		// Process all words in the sentence with settling
		for _, word := range sample.words {
			inputEmbed := embedding.Encode(word)
			state.SetInput(inputEmbed)
			for i := 0; i < settlingSteps; i++ {
				net.StepForward(state)
			}
		}

		output := state.GetOutput()

		predIdx := 0
		for i := 1; i < len(output); i++ {
			if output[i] > output[predIdx] {
				predIdx = i
			}
		}

		predLabels := []string{"Negative", "Neutral", "Positive"}
		mark := "✗"
		if predIdx == sample.sentiment {
			correctTest++
			mark = "✓"
		}

		textPreview := sample.text
		if len(textPreview) > 60 {
			textPreview = textPreview[:60] + "..."
		}

		fmt.Printf("%s [%s] Predicted: %-8s Expected: %-8s\n   \"%s\"\n",
			mark, sample.label, predLabels[predIdx], sample.label, textPreview)
	}

	fmt.Printf("\nTest Accuracy: %d/%d (%.1f%%)\n",
		correctTest, len(testSamples),
		float32(correctTest)/float32(len(testSamples))*100)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
