package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	nn "github.com/openfluke/loom/nn"
)

// Custom BPE Tokenizer
type BPETokenizer struct {
	vocab     map[string]int
	idToToken map[int]string
	merges    [][]string
	vocabSize int
}

func NewBPETokenizer(text string, numMerges int) *BPETokenizer {
	vocab := make(map[string]int)
	idToToken := make(map[int]string)

	// Special tokens
	vocab["<UNK>"] = 0
	idToToken[0] = "<UNK>"
	vocab["<PAD>"] = 1
	idToToken[1] = "<PAD>"

	idx := 2

	// Add all unique characters
	charSet := make(map[rune]bool)
	for _, r := range text {
		charSet[r] = true
	}

	for char := range charSet {
		s := string(char)
		vocab[s] = idx
		idToToken[idx] = s
		idx++
	}

	// Learn BPE merges
	merges := make([][]string, 0)
	words := strings.Fields(strings.ToLower(text))

	for merge := 0; merge < numMerges; merge++ {
		// Count pairs
		pairCounts := make(map[string]int)

		for _, word := range words {
			tokens := tokenizeWord(word, merges)
			for i := 0; i < len(tokens)-1; i++ {
				pair := tokens[i] + " " + tokens[i+1]
				pairCounts[pair]++
			}
		}

		if len(pairCounts) == 0 {
			break
		}

		// Find most frequent pair
		maxPair := ""
		maxCount := 0
		for pair, count := range pairCounts {
			if count > maxCount {
				maxCount = count
				maxPair = pair
			}
		}

		if maxPair == "" {
			break
		}

		// Add merge
		parts := strings.Split(maxPair, " ")
		merges = append(merges, parts)

		// Add merged token to vocab
		merged := parts[0] + parts[1]
		if _, exists := vocab[merged]; !exists {
			vocab[merged] = idx
			idToToken[idx] = merged
			idx++
		}
	}

	return &BPETokenizer{
		vocab:     vocab,
		idToToken: idToToken,
		merges:    merges,
		vocabSize: idx,
	}
}

func tokenizeWord(word string, merges [][]string) []string {
	// Start with individual characters
	tokens := make([]string, 0)
	for _, r := range word {
		tokens = append(tokens, string(r))
	}

	// Apply merges
	for _, merge := range merges {
		newTokens := make([]string, 0)
		i := 0
		for i < len(tokens) {
			if i < len(tokens)-1 && tokens[i] == merge[0] && tokens[i+1] == merge[1] {
				newTokens = append(newTokens, merge[0]+merge[1])
				i += 2
			} else {
				newTokens = append(newTokens, tokens[i])
				i++
			}
		}
		tokens = newTokens
	}

	return tokens
}

func (t *BPETokenizer) Encode(text string) []int {
	words := strings.Fields(strings.ToLower(text))
	ids := make([]int, 0)

	for _, word := range words {
		tokens := tokenizeWord(word, t.merges)
		for _, token := range tokens {
			if id, exists := t.vocab[token]; exists {
				ids = append(ids, id)
			} else {
				ids = append(ids, 0) // <UNK>
			}
		}
	}

	return ids
}

func (t *BPETokenizer) Decode(id int) string {
	if token, exists := t.idToToken[id]; exists {
		return token
	}
	return "<UNK>"
}

func (t *BPETokenizer) OneHot(id int) []float32 {
	vec := make([]float32, t.vocabSize)
	if id >= 0 && id < t.vocabSize {
		vec[id] = 1.0
	}
	return vec
}

// TargetQueue for delayed targets
type TargetQueue struct {
	targets []int
	maxSize int
}

func NewTargetQueue(size int) *TargetQueue {
	return &TargetQueue{
		targets: make([]int, 0, size),
		maxSize: size,
	}
}

func (q *TargetQueue) Push(target int) {
	q.targets = append(q.targets, target)
}

func (q *TargetQueue) Pop() int {
	if len(q.targets) == 0 {
		return -1
	}
	t := q.targets[0]
	q.targets = q.targets[1:]
	return t
}

func (q *TargetQueue) IsFull() bool {
	return len(q.targets) >= q.maxSize
}

func (q *TargetQueue) Clear() {
	q.targets = q.targets[:0]
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  LOOM Stepping Text Generator v6 - TRUE BPE Token Prediction      ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// 1. Load data
	text := ""
	data, err := os.ReadFile("alice.txt")
	if err == nil {
		text = string(data)
		text = strings.ReplaceAll(text, "\n", " ")
		text = strings.ReplaceAll(text, "\r", "")
		text = strings.ReplaceAll(text, "  ", " ")
		if len(text) > 10000 {
			text = text[:10000]
		}
		fmt.Println("✓ Loaded alice.txt!")
	} else {
		text = `the quick brown fox jumps over the lazy dog ` +
			`the cat sat on the mat and looked around ` +
			`a bird flew across the blue sky today `
		fmt.Println("Using default text")
	}

	fmt.Printf("Text length: %d characters\n", len(text))
	previewLen := 60
	if len(text) < previewLen {
		previewLen = len(text)
	}
	fmt.Printf("Preview: \"%s...\"\n\n", text[:previewLen])

	// 2. Build BPE tokenizer
	fmt.Println("Building BPE tokenizer (learning subword units)...")
	tok := NewBPETokenizer(text, 150)

	fmt.Printf("✓ Vocabulary: %d tokens\n", tok.vocabSize)
	fmt.Printf("  - Learned %d merge operations\n", len(tok.merges))
	fmt.Printf("  - Example merges: ")
	for i := 0; i < 5 && i < len(tok.merges); i++ {
		fmt.Printf("'%s'+'%s' ", tok.merges[i][0], tok.merges[i][1])
	}
	fmt.Println("\n")

	// 3. Tokenize text
	tokenIDs := tok.Encode(text)
	fmt.Printf("Total tokens: %d (from %d characters)\n", len(tokenIDs), len(text))
	fmt.Printf("Compression ratio: %.2fx\n\n", float64(len(text))/float64(len(tokenIDs)))

	// 4. Build network
	networkJSON := fmt.Sprintf(`{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{ "type": "dense", "input_height": %d, "output_height": 64, "activation": "relu" },
			{ "type": "lstm", "input_size": 64, "hidden_size": 64, "seq_length": 1, "activation": "tanh" },
			{ "type": "dense", "input_height": 64, "output_height": %d, "activation": "softmax" }
		]
	}`, tok.vocabSize, tok.vocabSize)

	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Fatalf("Failed to build network: %v", err)
	}
	net.InitializeWeights()
	state := net.InitStepState(tok.vocabSize)

	fmt.Println("Network: Input → Dense(64) → LSTM(64) → Softmax")
	fmt.Println()

	// 5. Training
	totalSteps := 15000
	targetDelay := 3
	targetQueue := NewTargetQueue(targetDelay)

	learningRate := float32(0.1)
	minLR := float32(0.001)
	decayRate := float32(0.9998)
	gradClip := float32(5.0)

	fmt.Printf("Training: %d steps\n", totalSteps)
	fmt.Printf("Learning rate: %.3f → %.3f\n", learningRate, minLR)
	fmt.Println()
	fmt.Println("Progress:")
	fmt.Println("─────────────────────────────────────────────────────────────────────")

	startTime := time.Now()
	dataPtr := 0
	inputVec := make([]float32, tok.vocabSize)

	for step := 0; step < totalSteps; step++ {
		currID := tokenIDs[dataPtr]
		nextPtr := (dataPtr + 1) % len(tokenIDs)
		targetID := tokenIDs[nextPtr]

		// One-hot input
		for i := range inputVec {
			inputVec[i] = 0
		}
		inputVec[currID] = 1.0
		state.SetInput(inputVec)

		net.StepForward(state)
		targetQueue.Push(targetID)

		if targetQueue.IsFull() {
			delayedTarget := targetQueue.Pop()
			output := state.GetOutput()

			// Cross-entropy loss
			prob := output[delayedTarget]
			if prob < 1e-7 {
				prob = 1e-7
			}
			loss := -float32(math.Log(float64(prob)))

			// Gradient
			gradOutput := make([]float32, tok.vocabSize)
			for i := range output {
				if i == delayedTarget {
					gradOutput[i] = output[i] - 1.0
				} else {
					gradOutput[i] = output[i]
				}
			}

			// Clip gradient
			gradNorm := float32(0.0)
			for _, g := range gradOutput {
				gradNorm += g * g
			}
			gradNorm = float32(math.Sqrt(float64(gradNorm)))
			if gradNorm > gradClip {
				scale := gradClip / gradNorm
				for i := range gradOutput {
					gradOutput[i] *= scale
				}
			}

			net.StepBackward(state, gradOutput)
			net.ApplyGradients(learningRate)

			// Logging
			if step%500 == 0 {
				predID := 0
				for i := 1; i < len(output); i++ {
					if output[i] > output[predID] {
						predID = i
					}
				}

				currToken := tok.Decode(currID)
				predToken := tok.Decode(predID)
				targetToken := tok.Decode(delayedTarget)

				mark := "✗"
				if predID == delayedTarget {
					mark = "✓"
				}

				fmt.Printf("Step %5d | Loss: %.4f | LR: %.4f | '%s' → pred:'%s' target:'%s' %s\n",
					step, loss, learningRate, currToken, predToken, targetToken, mark)
			}
		}

		learningRate *= decayRate
		if learningRate < minLR {
			learningRate = minLR
		}

		dataPtr = nextPtr
	}

	elapsed := time.Since(startTime)
	fmt.Println()
	fmt.Printf("Training complete in %v (%.1f steps/sec)\n\n", elapsed, float64(totalSteps)/elapsed.Seconds())

	// 6. Generation test
	fmt.Println("╔════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Generation Test - Combining Subword Tokens                       ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	seeds := []string{"alice", "the", "down"}

	for _, seed := range seeds {
		// Encode seed
		seedIDs := tok.Encode(seed)
		if len(seedIDs) == 0 {
			continue
		}

		// Prime with seed
		for _, id := range seedIDs {
			for i := range inputVec {
				inputVec[i] = 0
			}
			inputVec[id] = 1.0
			state.SetInput(inputVec)
			net.StepForward(state)
		}

		// Generate
		fmt.Printf("Seed: '%s' → ", seed)
		generated := make([]string, 0)

		for i := 0; i < 30; i++ {
			output := state.GetOutput()

			// Sample with temperature
			temp := float32(0.6)
			maxVal := output[0]
			for _, v := range output {
				if v > maxVal {
					maxVal = v
				}
			}

			exps := make([]float32, tok.vocabSize)
			sumExp := float32(0.0)
			for j, v := range output {
				exps[j] = float32(math.Exp(float64((v - maxVal) / temp)))
				sumExp += exps[j]
			}

			r := rand.Float32() * sumExp
			cumSum := float32(0.0)
			nextID := 0
			for j, exp := range exps {
				cumSum += exp
				if cumSum >= r {
					nextID = j
					break
				}
			}

			token := tok.Decode(nextID)
			generated = append(generated, token)

			// Feed back
			for j := range inputVec {
				inputVec[j] = 0
			}
			inputVec[nextID] = 1.0
			state.SetInput(inputVec)
			net.StepForward(state)
		}

		// Combine subword tokens
		result := strings.Join(generated, "")
		fmt.Printf("%s\n", result)
	}

	fmt.Println()
	fmt.Println("✓ Notice how subword tokens combine to form words!")
	fmt.Println("Done!")
}
