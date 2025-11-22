package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	nn "github.com/openfluke/loom/nn"
	tokenizer "github.com/openfluke/loom/tokenizer"
)

// TargetQueue handles the delay between input and output in the stepping network
type TargetQueue struct {
	targets []int // Changed to int for Token IDs
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

// --- Helper: Build Tokenizer ---
func buildSimpleTokenizer(text string) *tokenizer.Tokenizer {
	uniqueChars := []string{"<UNK>", "<PAD>"}
	seen := make(map[rune]bool)
	for _, r := range text {
		if !seen[r] {
			seen[r] = true
			uniqueChars = append(uniqueChars, string(r))
		}
	}

	vocab := make(map[string]int)
	for i, c := range uniqueChars {
		vocab[c] = i
	}

	// Construct minimal JSON config for loader
	// We just need a dummy config since we aren't using the full tokenizer loader capabilities for this simple test
	// But to keep it compatible with the existing struct if needed, we can just manually build it or ignore.
	// Actually, the original code used LoadFromBytes with a JSON. Let's just manually construct the Tokenizer struct if possible,
	// or simpler: just use the map we built.
	// The original code used: tok, _ := tokenizer.LoadFromBytes([]byte(config))
	// Let's stick to that but we need "encoding/json" back if we use it.
	// Wait, I removed encoding/json. Let's put it back or find a simpler way.
	// Actually, I can just manually instantiate the tokenizer if the package allows, or just keep encoding/json.
	// Re-adding encoding/json to imports is cleaner than rewriting this whole helper logic to avoid it.
	// BUT, I already removed it in the first chunk.
	// Let's just construct the JSON string manually without json.Marshal to avoid the import if it's simple map[string]int.
	// Or better, I will just re-add encoding/json in the first chunk? No, I can't edit previous chunks.
	// I will just use fmt.Sprintf to build the json for the vocab since it is simple.

	vocabStr := "{"
	first := true
	for k, v := range vocab {
		if !first {
			vocabStr += ","
		}
		// Escape keys if needed, but for simple text it's fine.
		// For safety let's quote them.
		vocabStr += fmt.Sprintf("%q:%d", k, v)
		first = false
	}
	vocabStr += "}"

	config := fmt.Sprintf(`{"model":{"type":"BPE","vocab":%s,"merges":[]}}`, vocabStr)
	tok, _ := tokenizer.LoadFromBytes([]byte(config))
	return tok
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// 1. Prepare Data
	// Hardcoded sentence for POC
	text := "the quick brown fox jumps over the lazy dog "
	// Repeat it a few times to give it a bit more substance, but keep it simple
	text = text + text + text + text

	tok := buildSimpleTokenizer(text)
	vocabSize := len(tok.Vocab)
	fmt.Printf("=== LOOM Stepping Language Model (Vocab: %d) ===\n", vocabSize)
	fmt.Println("Architecture: Dense(Embed) -> LSTM -> Dense(Logits)")
	fmt.Println("Dataset: '" + text[:40] + "...'")
	fmt.Println()

	// Convert text to integer IDs
	var dataIDs []int
	unkID, _ := tok.TokenToID("<UNK>")
	for _, r := range text {
		id, ok := tok.TokenToID(string(r))
		if !ok {
			id = unkID
		}
		dataIDs = append(dataIDs, id)
	}

	// 2. Define Network Architecture
	// INCREASED SIZE: 64 -> 128 to give it enough memory to escape the "Space Trap"
	networkJSON := fmt.Sprintf(`{
        "batch_size": 1,
        "grid_rows": 1,
        "grid_cols": 3,
        "layers_per_cell": 1,
        "layers": [
            {
                "type": "dense",
                "input_height": %d,
                "output_height": 32,
                "activation": "tanh"
            },
            {
                "type": "lstm",
                "input_size": 32,
                "hidden_size": 64,
                "seq_length": 1,
                "activation": "tanh"
            },
            {
                "type": "dense",
                "input_height": 64,
                "output_height": %d,
                "activation": "linear"
            }
        ]
    }`, vocabSize, vocabSize)

	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Fatalf("Failed to build network: %v", err)
	}
	net.InitializeWeights()

	state := net.InitStepState(vocabSize)

	// 3. Setup Continuous Training Loop
	// Pipeline Depth = 3 layers (Dense->LSTM->Dense)
	targetDelay := 3
	targetQueue := NewTargetQueue(targetDelay)

	// TUNED HYPERPARAMETERS
	learningRate := float32(0.05) // Increased LR significantly
	gradientClip := float32(5.0)  // Looser clipping

	totalSteps := 10000 // Increased steps
	fmt.Printf("Training for %d steps (Pipeline Delay: %d)\n", totalSteps, targetDelay)

	startTime := time.Now()
	dataPtr := 0

	// Pre-allocate input vector to reuse memory
	inputVec := make([]float32, vocabSize)

	fmt.Println("Total Steps:", totalSteps)

	for stepCount := 0; stepCount < totalSteps; stepCount++ {
		// A. Get Current & Target Token
		currID := dataIDs[dataPtr]
		nextPtr := (dataPtr + 1) % len(dataIDs)
		targetID := dataIDs[nextPtr]

		// B. Set Input (One-Hot)
		for i := range inputVec {
			inputVec[i] = 0
		}
		inputVec[currID] = 1.0
		state.SetInput(inputVec)

		// C. Step Forward
		net.StepForward(state)

		// D. Manage Delay
		targetQueue.Push(targetID)

		if targetQueue.IsFull() {
			delayedTargetID := targetQueue.Pop()
			output := state.GetOutput()

			// E. Loss & Gradient
			// Softmax logic
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
				tVal := float32(0.0)
				if i == delayedTargetID {
					tVal = 1.0
				}

				if tVal > 0.5 {
					loss -= float32(math.Log(float64(probs + 1e-7)))
				}
				gradOutput[i] = probs - tVal
			}

			// Gradient Clipping (Crucial for LSTM stability)
			gradNorm := float32(0.0)
			for _, g := range gradOutput {
				gradNorm += g * g
			}
			gradNorm = float32(math.Sqrt(float64(gradNorm)))
			if gradNorm > gradientClip {
				scale := gradientClip / gradNorm
				for i := range gradOutput {
					gradOutput[i] *= scale
				}
			}

			// F. Backward & Update
			net.StepBackward(state, gradOutput)
			net.ApplyGradients(learningRate)

			// Logging
			if stepCount%500 == 0 {
				predID := 0
				for i := 1; i < len(output); i++ {
					if output[i] > output[predID] {
						predID = i
					}
				}

				pChar, _ := tok.IDToToken(predID)
				tChar, _ := tok.IDToToken(delayedTargetID)
				if pChar == "\n" {
					pChar = "\\n"
				}
				if tChar == "\n" {
					tChar = "\\n"
				}

				fmt.Printf("Step %-6d Loss: %.4f | Pred: '%s' Exp: '%s'\n", stepCount, loss, pChar, tChar)
			}
		}

		dataPtr = nextPtr
	}

	fmt.Printf("Training Complete in %v\n\n", time.Since(startTime))

	// 4. Generation Mode
	fmt.Println("=== Generation Mode ===")
	seed := "the"
	fmt.Printf("Seed: \"%s\"", seed)

	// Prime with seed (Sequential Mode to fix inference delay)
	for _, r := range seed {
		id, _ := tok.TokenToID(string(r))
		for i := range inputVec {
			inputVec[i] = 0
		}
		inputVec[id] = 1.0
		state.SetInput(inputVec)
		// Run sequentially to ensure state is updated before next input
		for l := 0; l < 3; l++ {
			net.StepForwardSingle(state, l)
		}
	}

	// Generate
	lastID := 0
	for i := 0; i < 100; i++ {
		output := state.GetOutput()

		// Sample
		maxVal := output[0]
		for _, v := range output {
			if v > maxVal {
				maxVal = v
			}
		}

		exps := make([]float32, len(output))
		sumExps := float32(0.0)
		temp := 0.6

		for k, v := range output {
			exps[k] = float32(math.Exp(float64(v-maxVal) / temp))
			sumExps += exps[k]
		}

		r := rand.Float32() * sumExps
		cum := float32(0.0)
		for k, v := range exps {
			cum += v
			if cum >= r {
				lastID = k
				break
			}
		}

		char, _ := tok.IDToToken(lastID)
		fmt.Print(char)

		// Feed back
		for i := range inputVec {
			inputVec[i] = 0
		}
		inputVec[lastID] = 1.0
		state.SetInput(inputVec)
		for l := 0; l < 3; l++ {
			net.StepForwardSingle(state, l)
		}
	}
	fmt.Println("\n\n=== Done ===")
}
