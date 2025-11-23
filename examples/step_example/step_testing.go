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

// --- Helper: TargetQueue (Copied from step_train_v6.go) ---
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

	vocabStr := "{"
	first := true
	for k, v := range vocab {
		if !first {
			vocabStr += ","
		}
		vocabStr += fmt.Sprintf("%q:%d", k, v)
		first = false
	}
	vocabStr += "}"

	config := fmt.Sprintf(`{"model":{"type":"BPE","vocab":%s,"merges":[]}}`, vocabStr)
	tok, _ := tokenizer.LoadFromBytes([]byte(config))
	return tok
}

// --- Test Runner ---
func runTest(name string, networkJSON string, vocabSize int, text string, tok *tokenizer.Tokenizer) {
	fmt.Printf("=== Testing %s ===\n", name)

	// Build Network
	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Fatalf("Failed to build network for %s: %v", name, err)
	}
	net.InitializeWeights()
	state := net.InitStepState(vocabSize)

	// Setup Data
	var dataIDs []int
	unkID, _ := tok.TokenToID("<UNK>")
	for _, r := range text {
		id, ok := tok.TokenToID(string(r))
		if !ok {
			id = unkID
		}
		dataIDs = append(dataIDs, id)
	}

	// Setup Training Loop
	targetDelay := 3
	targetQueue := NewTargetQueue(targetDelay)
	learningRate := float32(0.01)
	totalSteps := 50 // Short run just to verify it works
	dataPtr := 0
	inputVec := make([]float32, vocabSize)

	start := time.Now()

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

			// Check for NaN
			if math.IsNaN(float64(output[0])) {
				log.Fatalf("NaN detected in output at step %d", stepCount)
			}

			// E. Loss & Gradient (Simplified)
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

			// F. Backward & Update
			net.StepBackward(state, gradOutput)
			net.ApplyGradients(learningRate)
		}
		dataPtr = nextPtr
	}

	fmt.Printf("PASS: %s completed in %v\n\n", name, time.Since(start))
}

func main() {
	rand.Seed(time.Now().UnixNano())
	text := "the quick brown fox jumps over the lazy dog "
	text = text + text // Double it
	tok := buildSimpleTokenizer(text)
	vocabSize := len(tok.Vocab)

	fmt.Printf("Vocab Size: %d\n\n", vocabSize)

	// 1. Dense Network
	// Input -> Dense(32) -> Dense(32) -> Dense(Vocab)
	denseJSON := fmt.Sprintf(`{
        "batch_size": 1,
        "grid_rows": 1,
        "grid_cols": 3,
        "layers_per_cell": 1,
        "layers": [
            { "type": "dense", "input_height": %d, "output_height": 32, "activation": "tanh" },
            { "type": "dense", "input_height": 32, "output_height": 32, "activation": "tanh" },
            { "type": "dense", "input_height": 32, "output_height": %d, "activation": "linear" }
        ]
    }`, vocabSize, vocabSize)
	runTest("Dense Network", denseJSON, vocabSize, text, tok)

	// 2. CNN Network
	// Input -> Dense(64) -> [Reshape 8x8] -> Conv2D -> [Flatten] -> Dense(Vocab)
	// Note: We use a Dense layer to project input to 64 (8x8) so it fits the CNN input
	cnnJSON := fmt.Sprintf(`{
        "batch_size": 1,
        "grid_rows": 1,
        "grid_cols": 3,
        "layers_per_cell": 1,
        "layers": [
            { "type": "dense", "input_height": %d, "output_height": 64, "activation": "tanh" },
            { 
                "type": "conv2d", 
                "input_height": 8, "input_width": 8, "input_channels": 1,
                "output_height": 8, "output_width": 8,
                "kernel_size": 3, "stride": 1, "padding": 1, "filters": 4,
                "activation": "relu"
            },
            { "type": "dense", "input_height": 256, "output_height": %d, "activation": "linear" }
        ]
    }`, vocabSize, vocabSize)
	// CNN Output calculation:
	// Input: 8x8
	// Padding: 1 -> 10x10
	// Kernel: 3x3
	// Output dim: (8 + 2*1 - 3)/1 + 1 = 8
	// Output: 8x8 x 4 filters = 256 flattened size
	runTest("CNN Network", cnnJSON, vocabSize, text, tok)

	// 3. MHA Network
	// Input -> Dense(32) -> MHA -> Dense(Vocab)
	mhaJSON := fmt.Sprintf(`{
        "batch_size": 1,
        "grid_rows": 1,
        "grid_cols": 3,
        "layers_per_cell": 1,
        "layers": [
            { "type": "dense", "input_height": %d, "output_height": 32, "activation": "linear" },
            { 
                "type": "multi_head_attention", 
                "d_model": 32, "num_heads": 4, "head_dim": 8, "seq_length": 1,
                "activation": "linear"
            },
            { "type": "dense", "input_height": 32, "output_height": %d, "activation": "linear" }
        ]
    }`, vocabSize, vocabSize)
	runTest("MHA Network", mhaJSON, vocabSize, text, tok)

	// 4. RNN Network
	// Input -> Dense(32) -> RNN -> Dense(Vocab)
	rnnJSON := fmt.Sprintf(`{
        "batch_size": 1,
        "grid_rows": 1,
        "grid_cols": 3,
        "layers_per_cell": 1,
        "layers": [
            { "type": "dense", "input_height": %d, "output_height": 32, "activation": "tanh" },
            { 
                "type": "rnn", 
                "input_size": 32, "hidden_size": 64, "seq_length": 1,
                "activation": "tanh"
            },
            { "type": "dense", "input_height": 64, "output_height": %d, "activation": "linear" }
        ]
    }`, vocabSize, vocabSize)
	runTest("RNN Network", rnnJSON, vocabSize, text, tok)

	// 5. LSTM Network
	// Input -> Dense(32) -> LSTM -> Dense(Vocab)
	lstmJSON := fmt.Sprintf(`{
        "batch_size": 1,
        "grid_rows": 1,
        "grid_cols": 3,
        "layers_per_cell": 1,
        "layers": [
            { "type": "dense", "input_height": %d, "output_height": 32, "activation": "tanh" },
            { 
                "type": "lstm", 
                "input_size": 32, "hidden_size": 64, "seq_length": 1,
                "activation": "tanh"
            },
            { "type": "dense", "input_height": 64, "output_height": %d, "activation": "linear" }
        ]
    }`, vocabSize, vocabSize)
	runTest("LSTM Network", lstmJSON, vocabSize, text, tok)
}
