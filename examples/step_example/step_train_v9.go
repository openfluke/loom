package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	nn "github.com/openfluke/loom/nn"
	tokenizer "github.com/openfluke/loom/tokenizer"
)

// --- 1. BPE LEARNER (To generate config for loom/tokenizer) ---

func learnBPEAndGetConfig(text string, numMerges int) string {
	fmt.Println("Learning BPE Merges to create Tokenizer Config...")

	// Limit sample size for learning
	sampleText := text
	if len(sampleText) > 500000 {
		sampleText = sampleText[:500000]
	}

	words := strings.Fields(sampleText)
	splits := make([][]string, len(words))
	for i, w := range words {
		chars := strings.Split(w, "")
		splits[i] = chars
	}

	merges := make([]string, 0)
	vocab := make(map[string]int)

	// Base Vocab
	idx := 0
	vocab["<UNK>"] = idx
	idx++
	vocab["<PAD>"] = idx
	idx++

	// Add standard chars
	chars := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-;:"
	for _, c := range chars {
		s := string(c)
		if _, ok := vocab[s]; !ok {
			vocab[s] = idx
			idx++
		}
	}
	// Add any chars found in text
	for _, r := range sampleText {
		s := string(r)
		if _, ok := vocab[s]; !ok {
			vocab[s] = idx
			idx++
		}
	}

	// Learning Loop
	for i := 0; i < numMerges; i++ {
		pairs := make(map[string]int)
		for _, word := range splits {
			for j := 0; j < len(word)-1; j++ {
				pair := word[j] + " " + word[j+1]
				pairs[pair]++
			}
		}

		bestPair := ""
		bestCount := 0
		for p, count := range pairs {
			if count > bestCount {
				bestCount = count
				bestPair = p
			}
		}

		if bestCount == 0 {
			break
		}

		parts := strings.Split(bestPair, " ")
		mergeL, mergeR := parts[0], parts[1]
		newToken := mergeL + mergeR

		// Record merge in format "a b"
		merges = append(merges, bestPair)
		vocab[newToken] = idx
		idx++

		// Apply
		for wIdx, word := range splits {
			var newWord []string
			skip := false
			for j := 0; j < len(word); j++ {
				if skip {
					skip = false
					continue
				}
				if j < len(word)-1 && word[j] == mergeL && word[j+1] == mergeR {
					newWord = append(newWord, newToken)
					skip = true
				} else {
					newWord = append(newWord, word[j])
				}
			}
			splits[wIdx] = newWord
		}
	}

	// Create JSON for loom/tokenizer
	type TokenizerModel struct {
		Type   string         `json:"type"`
		Vocab  map[string]int `json:"vocab"`
		Merges []string       `json:"merges"`
	}
	type TokenizerJSON struct {
		Model       TokenizerModel `json:"model"`
		AddedTokens []struct{}     `json:"added_tokens"`
	}

	config := TokenizerJSON{
		Model: TokenizerModel{
			Type:   "BPE",
			Vocab:  vocab,
			Merges: merges,
		},
		AddedTokens: []struct{}{},
	}

	bytes, _ := json.Marshal(config)
	return string(bytes)
}

// --- Helper: TargetQueue ---
type TargetQueue struct {
	targets []int
	maxSize int
}

func NewTargetQueue(size int) *TargetQueue {
	return &TargetQueue{targets: make([]int, 0, size), maxSize: size}
}
func (q *TargetQueue) Push(t int) { q.targets = append(q.targets, t) }
func (q *TargetQueue) Pop() int {
	if len(q.targets) == 0 {
		return -1
	}
	t := q.targets[0]
	q.targets = q.targets[1:]
	return t
}
func (q *TargetQueue) IsFull() bool { return len(q.targets) >= q.maxSize }
func (q *TargetQueue) Clear()       { q.targets = q.targets[:0] }

// Resize updates the queue size for curriculum learning
func (q *TargetQueue) Resize(newSize int) {
	q.maxSize = newSize
	// If shrinking, discard oldest
	if len(q.targets) > newSize {
		q.targets = q.targets[len(q.targets)-newSize:]
	}
}

// --- Helper: Generate ---
func generateSample(net *nn.Network, tok *tokenizer.Tokenizer, vocabSize int, seed string, length int) string {
	genState := net.InitStepState(vocabSize)
	inputVec := make([]float32, vocabSize)

	// Prime
	// Note: loom/tokenizer Encode returns []uint32 or similar depending on version, casting to int here
	// Assuming Encode returns []uint32 based on previous context
	rawIDs := tok.Encode(seed, false) // assuming signature Encode(text, addSpecial)
	seedIDs := make([]int, len(rawIDs))
	for i, v := range rawIDs {
		seedIDs[i] = int(v)
	}

	for _, id := range seedIDs {
		for i := range inputVec {
			inputVec[i] = 0
		}
		if id < vocabSize {
			inputVec[id] = 1.0
		}
		genState.SetInput(inputVec)
		net.StepForward(genState)
	}

	var sb strings.Builder
	sb.WriteString(seed)
	sb.WriteString(" ")

	for i := 0; i < length; i++ {
		out := genState.GetOutput()

		// Sample
		maxVal := out[0]
		for _, v := range out {
			if v > maxVal {
				maxVal = v
			}
		}
		sumExp := float32(0.0)
		exps := make([]float32, len(out))
		for k, v := range out {
			exps[k] = float32(math.Exp(float64((v - maxVal) / 0.7)))
			sumExp += exps[k]
		}

		r := rand.Float32() * sumExp
		cum := float32(0.0)
		best := 0
		for k, e := range exps {
			cum += e
			if cum >= r {
				best = k
				break
			}
		}

		tokStr, _ := tok.IDToToken(best)
		// Clean up token string if needed (loom might return Ġ for space)
		tokStr = strings.ReplaceAll(tokStr, "Ġ", " ")
		sb.WriteString(tokStr)
		sb.WriteString(" ")

		for i := range inputVec {
			inputVec[i] = 0
		}
		if best < vocabSize {
			inputVec[best] = 1.0
		}
		genState.SetInput(inputVec)
		net.StepForward(genState)
	}
	return sb.String()
}

func main() {
	rand.Seed(time.Now().UnixNano())
	fmt.Println("=== LOOM v13: True BPE Training (Integrated Tokenizer) ===")

	// 1. Load Data
	corpusDir := "./corpus"
	var sb strings.Builder
	files, _ := os.ReadDir(corpusDir)
	if len(files) > 0 {
		fmt.Println("Loading corpus files...")
		for _, f := range files {
			if filepath.Ext(f.Name()) == ".txt" {
				b, err := os.ReadFile(filepath.Join(corpusDir, f.Name()))
				if err == nil {
					sb.Write(b)
					sb.WriteString(" ")
					fmt.Printf("Loaded %s (%d bytes)\n", f.Name(), len(b))
				}
			}
		}
	}
	textData := sb.String()
	if len(textData) == 0 {
		fmt.Println("Using fallback text.")
		base := "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversation?' "
		for i := 0; i < 500; i++ {
			textData += base
		}
	}
	fmt.Printf("Total Text Size: %d chars\n", len(textData))

	// 2. Train BPE & Load into Loom Tokenizer
	// Target 500 merges
	jsonConfig := learnBPEAndGetConfig(textData, 500)
	tok, err := tokenizer.LoadFromBytes([]byte(jsonConfig))
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}
	fmt.Printf("Vocab Size: %d\n", len(tok.Vocab))

	// 3. Encode Data
	fmt.Println("Encoding text...")
	// Assuming Encode returns []uint32
	rawIDs := tok.Encode(textData, false)
	dataIDs := make([]int, len(rawIDs))
	for i, v := range rawIDs {
		dataIDs[i] = int(v)
	}
	fmt.Printf("Total Tokens: %d\n", len(dataIDs))

	// 4. Network
	vocabSize := len(tok.Vocab)
	hiddenSize := 128
	networkJSON := fmt.Sprintf(`{
        "batch_size": 1,
        "grid_rows": 1,
        "grid_cols": 3,
        "layers_per_cell": 1,
        "layers": [
            { "type": "dense", "input_height": %d, "output_height": %d, "activation": "tanh" },
            { "type": "lstm", "input_size": %d, "hidden_size": %d, "seq_length": 1, "activation": "tanh" },
            { "type": "dense", "input_height": %d, "output_height": %d, "activation": "linear" }
        ]
    }`, vocabSize, hiddenSize, hiddenSize, hiddenSize, hiddenSize, vocabSize)

	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Fatal(err)
	}
	net.InitializeWeights()
	state := net.InitStepState(vocabSize)

	// 5. Training Loop
	// Start with a smaller delay and increase it
	currentDelay := 1
	maxDelay := 5
	delayIncreaseInterval := 10000 // Increase delay every 10k steps

	queue := NewTargetQueue(currentDelay)
	inputVec := make([]float32, vocabSize)

	steps := 50000
	lr := float32(0.01)
	ptr := 0
	resetInterval := 1000 // Reset state every 1000 steps

	fmt.Println("Training...")
	start := time.Now()

	for i := 0; i < steps; i++ {
		// Curriculum: Increase delay over time
		if i > 0 && i%delayIncreaseInterval == 0 && currentDelay < maxDelay {
			currentDelay++
			queue.Resize(currentDelay)
			fmt.Printf("\n>>> Increasing Delay to %d\n", currentDelay)
		}

		// Periodic Reset & Priming
		if i > 0 && i%resetInterval == 0 {
			state = net.InitStepState(vocabSize)
			queue.Clear()
			// Prime with a few previous tokens if available
			primeLen := 5
			if ptr >= primeLen {
				primeSeq := dataIDs[ptr-primeLen : ptr]
				for _, id := range primeSeq {
					for k := range inputVec {
						inputVec[k] = 0
					}
					if id < vocabSize {
						inputVec[id] = 1.0
					}
					state.SetInput(inputVec)
					net.StepForward(state)
				}
			}
		}

		// Periodic Generation Monitoring
		if i > 0 && i%500 == 0 {
			fmt.Println("\n------------------------------------------------------------")
			fmt.Printf(" [MONITORING @ Step %d]\n", i)
			// Using "The" as a seed since common words likely exist in vocab
			sample := generateSample(net, tok, vocabSize, "The", 15)
			fmt.Printf(" Output: \"%s...\"\n", sample)
			fmt.Println("------------------------------------------------------------")
		}

		if ptr >= len(dataIDs)-1 {
			ptr = 0
		}

		curr := dataIDs[ptr]
		next := dataIDs[ptr+1]
		ptr++

		// Input
		for k := range inputVec {
			inputVec[k] = 0
		}
		if curr < vocabSize {
			inputVec[curr] = 1.0
		}
		state.SetInput(inputVec)

		// Step
		net.StepForward(state)
		queue.Push(next)

		if queue.IsFull() {
			target := queue.Pop()
			out := state.GetOutput()

			// Loss
			maxVal := out[0]
			for _, v := range out {
				if v > maxVal {
					maxVal = v
				}
			}
			sumExp := float32(0.0)
			exps := make([]float32, len(out))
			for k, v := range out {
				exps[k] = float32(math.Exp(float64(v - maxVal)))
				sumExp += exps[k]
			}

			grad := make([]float32, vocabSize)
			loss := float32(0.0)
			for k := range out {
				prob := exps[k] / sumExp
				t := float32(0.0)
				if k == target {
					t = 1.0
				}
				if t > 0.5 {
					loss -= float32(math.Log(float64(prob + 1e-7)))
				}
				grad[k] = prob - t
			}

			net.StepBackward(state, grad)
			net.ApplyGradients(lr)

			if i%500 == 0 {
				predID := 0
				for k, v := range out {
					if v > out[predID] {
						predID = k
					}
				}

				pTok, _ := tok.IDToToken(predID)
				tTok, _ := tok.IDToToken(target)

				// Cleanup newlines for display
				pTok = strings.ReplaceAll(pTok, "\n", "\\n")
				tTok = strings.ReplaceAll(tTok, "\n", "\\n")

				fmt.Printf("Step %d | Loss %.4f | Pred: '%s' Exp: '%s'\n", i, loss, pTok, tTok)
			}
		}
	}

	fmt.Printf("Training Complete in %v\n", time.Since(start))

	// 6. Test
	fmt.Println("\n=== FINAL GENERATION TEST ===")
	seeds := []string{"Alice", "The", "It"}

	for _, seed := range seeds {
		fmt.Printf("Seed '%s': ", seed)
		fmt.Println(generateSample(net, tok, vocabSize, seed, 30))
	}
}
