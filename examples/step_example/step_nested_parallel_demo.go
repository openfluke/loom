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

// --- 1. BPE Utilities (Standard) ---
func learnBPEAndGetConfig(text string, numMerges int) string {
	fmt.Println("Learning BPE Merges...")
	sampleText := text
	if len(sampleText) > 500000 {
		sampleText = sampleText[:500000]
	}

	words := strings.Fields(sampleText)
	splits := make([][]string, len(words))
	for i, w := range words {
		splits[i] = strings.Split(w, "")
	}

	merges := make([]string, 0)
	vocab := make(map[string]int)
	idx := 0
	vocab["<UNK>"] = idx
	idx++
	vocab["<PAD>"] = idx
	idx++

	chars := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-;:"
	for _, c := range chars {
		s := string(c)
		if _, ok := vocab[s]; !ok {
			vocab[s] = idx
			idx++
		}
	}
	if _, ok := vocab[" "]; !ok {
		vocab[" "] = idx
		idx++
	}

	for _, r := range sampleText {
		s := string(r)
		if _, ok := vocab[s]; !ok {
			vocab[s] = idx
			idx++
		}
	}

	for i := 0; i < numMerges; i++ {
		pairs := make(map[string]int)
		for _, word := range splits {
			for j := 0; j < len(word)-1; j++ {
				pairs[word[j]+" "+word[j+1]]++
			}
		}
		bestPair, bestCount := "", 0
		for p, c := range pairs {
			if c > bestCount {
				bestCount = c
				bestPair = p
			}
		}
		if bestCount == 0 {
			break
		}

		parts := strings.Split(bestPair, " ")
		newToken := parts[0] + parts[1]
		merges = append(merges, bestPair)
		vocab[newToken] = idx
		idx++

		for wIdx, word := range splits {
			var newWord []string
			skip := false
			for j := 0; j < len(word); j++ {
				if skip {
					skip = false
					continue
				}
				if j < len(word)-1 && word[j] == parts[0] && word[j+1] == parts[1] {
					newWord = append(newWord, newToken)
					skip = true
				} else {
					newWord = append(newWord, word[j])
				}
			}
			splits[wIdx] = newWord
		}
	}

	type TokenizerModel struct {
		Type   string         `json:"type"`
		Vocab  map[string]int `json:"vocab"`
		Merges []string       `json:"merges"`
	}
	config := struct {
		Model       TokenizerModel `json:"model"`
		AddedTokens []struct{}     `json:"added_tokens"`
	}{
		Model:       TokenizerModel{Type: "BPE", Vocab: vocab, Merges: merges},
		AddedTokens: []struct{}{},
	}
	b, _ := json.Marshal(config)
	return string(b)
}

// --- 2. Target Queue ---
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

// --- 3. Generator ---
func generate(net *nn.Network, tok *tokenizer.Tokenizer, vocabSize int, seed string, length int) string {
	state := net.InitStepState(vocabSize)
	input := make([]float32, vocabSize)

	rawIDs := tok.Encode(seed, false)
	for _, id := range rawIDs {
		for i := range input {
			input[i] = 0
		}
		if int(id) < vocabSize {
			input[int(id)] = 1.0
		}
		state.SetInput(input)
		net.StepForward(state)
	}

	var sb strings.Builder
	sb.WriteString(seed)
	lastTokens := make([]int, 0, 5)

	for i := 0; i < length; i++ {
		out := state.GetOutput()

		for _, pastID := range lastTokens {
			if pastID < len(out) {
				out[pastID] -= 1.5
			}
		}

		temperature := 0.6
		maxVal := out[0]
		for _, v := range out {
			if v > maxVal {
				maxVal = v
			}
		}

		sumExp := float32(0.0)
		exps := make([]float32, len(out))
		for j, v := range out {
			exps[j] = float32(math.Exp(float64((v - maxVal) / float32(temperature))))
			sumExp += exps[j]
		}

		r := rand.Float32() * sumExp
		cum := float32(0.0)
		best := 0
		for j, e := range exps {
			cum += e
			if cum >= r {
				best = j
				break
			}
		}

		lastTokens = append(lastTokens, best)
		if len(lastTokens) > 5 {
			lastTokens = lastTokens[1:]
		}

		s, _ := tok.IDToToken(best)
		s = strings.ReplaceAll(s, "Ġ", " ")
		sb.WriteString(s)

		for j := range input {
			input[j] = 0
		}
		if best < vocabSize {
			input[best] = 1.0
		}
		state.SetInput(input)
		net.StepForward(state)
	}
	return sb.String()
}

func main() {
	rand.Seed(time.Now().UnixNano())
	fmt.Println("=== LOOM: Momentum-Driven Nested Parallel Demo (FIXED) ===")
	fmt.Println("Architecture: Embedding -> Pre-Norm -> Parallel[LSTM | SwiGLU] -> Mix -> Head")

	// 1. Load Corpus
	corpusDir := "./corpus"
	var sb strings.Builder
	files, _ := os.ReadDir(corpusDir)
	if len(files) > 0 {
		for _, f := range files {
			if filepath.Ext(f.Name()) == ".txt" {
				b, _ := os.ReadFile(filepath.Join(corpusDir, f.Name()))
				sb.Write(b)
				sb.WriteString(" ")
			}
		}
	}
	textData := sb.String()
	if len(textData) == 0 {
		fmt.Println("Using fallback text.")
		base := "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it. "
		for i := 0; i < 400; i++ {
			textData += base
		}
	}

	// 2. Train BPE
	jsonConfig := learnBPEAndGetConfig(textData, 800)
	tok, _ := tokenizer.LoadFromBytes([]byte(jsonConfig))
	vocabSize := len(tok.Vocab)
	fmt.Printf("Vocab Size: %d\n", vocabSize)

	// 3. Encode
	rawIDs := tok.Encode(textData, false)
	dataIDs := make([]int, len(rawIDs))
	for i, v := range rawIDs {
		dataIDs[i] = int(v)
	}
	fmt.Printf("Total Tokens: %d\n", len(dataIDs))

	// 4. Network Construction
	embeddingSize := 48
	lstmHidden := 64
	swigluIntermediate := 64
	mixSize := 64

	// FIX: SwiGLU projects back to input size (embeddingSize)
	// So parallel output is LSTM (64) + SwiGLU (48) = 112
	parallelOutputSize := lstmHidden + embeddingSize

	networkJSON := fmt.Sprintf(`{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 5, 
		"layers_per_cell": 1,
		"layers": [
			{ 
				"type": "dense", 
				"id": "embedding",
				"input_height": %d, 
				"output_height": %d, 
				"activation": "tanh" 
			},
			{
				"type": "layer_norm",
				"id": "pre_norm",
				"norm_size": %d,
				"epsilon": 1e-5
			},
			{
				"type": "parallel",
				"id": "dual_core",
				"combine_mode": "concat",
				"branches": [
					{ 
						"type": "lstm", 
						"id": "context_core",
						"input_size": %d, 
						"hidden_size": %d, 
						"seq_length": 1, 
						"activation": "tanh"
					},
					{
						"type": "swiglu",
						"id": "fact_core",
						"input_size": %d,
						"output_size": %d
					}
				]
			},
			{ 
				"type": "dense", 
				"id": "mixer",
				"input_height": %d, 
				"output_height": %d, 
				"activation": "relu" 
			},
			{ 
				"type": "dense", 
				"id": "head",
				"input_height": %d, 
				"output_height": %d, 
				"activation": "linear" 
			}
		]
	}`,
		vocabSize, embeddingSize, // Embedding
		embeddingSize,             // Pre-Norm
		embeddingSize, lstmHidden, // LSTM Branch
		embeddingSize, swigluIntermediate, // SwiGLU Branch (Input=48, Intermediate=64, Output=48)
		parallelOutputSize, mixSize, // Mixer (Input=112, Output=64)
		mixSize, vocabSize) // Head

	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Fatal(err)
	}
	net.InitializeWeights()
	state := net.InitStepState(vocabSize)

	// 5. Training
	targetDelay := 1
	queue := NewTargetQueue(targetDelay)
	inputVec := make([]float32, vocabSize)

	steps := 100000
	baseLR := float32(0.05)
	ptr := 0

	fmt.Println("Training...")
	start := time.Now()

	for i := 0; i < steps; i++ {
		currentLR := baseLR
		if i < 2000 {
			currentLR = baseLR * (float32(i) / 2000.0)
		} else if i > 50000 {
			currentLR = 0.01
		} else if i > 80000 {
			currentLR = 0.005
		}

		if i > 0 && i%5000 == 0 {
			fmt.Println("\n------------------------------------------------------------")
			fmt.Printf(" [MONITORING @ Step %d | LR %.4f]\n", i, currentLR)
			fmt.Printf(" Output: \"%s...\"\n", generate(net, tok, vocabSize, "Alice", 30))
			fmt.Println("------------------------------------------------------------")
		}

		if ptr >= len(dataIDs)-1 {
			ptr = 0
		}

		curr := dataIDs[ptr]
		next := dataIDs[ptr+1]
		ptr++

		for k := range inputVec {
			inputVec[k] = 0
		}
		if curr < vocabSize {
			inputVec[curr] = 1.0
		}
		state.SetInput(inputVec)

		net.StepForward(state)
		queue.Push(next)

		if queue.IsFull() {
			target := queue.Pop()
			out := state.GetOutput()

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
				grad[k] = prob - t
				if t > 0.5 {
					loss -= float32(math.Log(float64(prob + 1e-9)))
				}
			}

			gradNorm := float32(0.0)
			for _, g := range grad {
				gradNorm += g * g
			}
			gradNorm = float32(math.Sqrt(float64(gradNorm)))
			if gradNorm > 2.0 {
				scale := 2.0 / gradNorm
				for k := range grad {
					grad[k] *= scale
				}
			}

			net.StepBackward(state, grad)
			net.ApplyGradients(currentLR)

			if i%2000 == 0 {
				predID := 0
				for k, v := range out {
					if v > out[predID] {
						predID = k
					}
				}
				pTok, _ := tok.IDToToken(predID)
				tTok, _ := tok.IDToToken(target)
				pTok = strings.ReplaceAll(pTok, "\n", "\\n")
				tTok = strings.ReplaceAll(tTok, "\n", "\\n")

				fmt.Printf("Step %d | Loss %.4f | Pred: '%s' Exp: '%s'\n", i, loss, pTok, tTok)
			}
		}
	}

	fmt.Printf("Training Complete in %v\n", time.Since(start))

	fmt.Println("\n=== FINAL GENERATION SAMPLES ===")
	fmt.Println(generate(net, tok, vocabSize, "Alice", 60))
	fmt.Println(generate(net, tok, vocabSize, "The", 60))
	fmt.Println(generate(net, tok, vocabSize, "She", 60))
}
