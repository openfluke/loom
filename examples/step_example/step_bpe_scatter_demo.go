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

// --- Micro Configuration ---
const (
	ContextSize  = 32
	EmbeddingDim = 64
)

// --- 1. Manual Embedding Table ---
type EmbeddingTable struct {
	Weights [][]float32
	Dim     int
}

func NewEmbeddingTable(vocabSize, dim int) *EmbeddingTable {
	weights := make([][]float32, vocabSize)
	scale := float32(math.Sqrt(2.0 / float64(dim)))
	for i := range weights {
		weights[i] = make([]float32, dim)
		for j := range weights[i] {
			weights[i][j] = (rand.Float32()*2 - 1) * scale
		}
	}
	return &EmbeddingTable{Weights: weights, Dim: dim}
}

func (e *EmbeddingTable) Forward(window []int) []float32 {
	out := make([]float32, len(window)*e.Dim)
	for i, id := range window {
		if id < len(e.Weights) {
			copy(out[i*e.Dim:], e.Weights[id])
		}
	}
	return out
}

func (e *EmbeddingTable) Backward(window []int, gradInput []float32, lr float32) {
	for i, id := range window {
		if id < len(e.Weights) {
			start := i * e.Dim
			end := start + e.Dim
			chunk := gradInput[start:end]
			for j := 0; j < e.Dim; j++ {
				g := chunk[j]
				if g > 0.5 {
					g = 0.5
				}
				if g < -0.5 {
					g = -0.5
				}
				e.Weights[id][j] -= lr * g
			}
		}
	}
}

// --- 2. BPE Utilities ---
func learnBPEAndGetConfig(text string, numMerges int) string {
	fmt.Println("Learning BPE Merges...")
	sampleText := text
	if len(sampleText) > 100000 {
		sampleText = sampleText[:100000]
	}

	safeText := strings.ReplaceAll(sampleText, " ", "Ġ")
	words := strings.Fields(safeText)
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
	vocab["Ġ"] = idx
	idx++

	chars := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-;:"
	for _, c := range chars {
		s := string(c)
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

// --- 3. Generator ---
func generate(net *nn.Network, embs *EmbeddingTable, tok *tokenizer.Tokenizer, vocabSize int, seed string, length int) string {
	window := make([]int, ContextSize)
	for i := range window {
		window[i] = 1
	}

	cleanSeed := strings.ReplaceAll(seed, " ", "Ġ")
	rawIDs := tok.Encode(cleanSeed, false)
	for _, id := range rawIDs {
		copy(window, window[1:])
		window[ContextSize-1] = int(id)
	}

	inputSize := ContextSize * EmbeddingDim
	state := net.InitStepState(inputSize)

	var sb strings.Builder
	sb.WriteString(seed)

	for i := 0; i < length; i++ {
		inputVec := embs.Forward(window)
		state.SetInput(inputVec)
		net.StepForward(state)
		out := state.GetOutput()

		maxVal := out[0]
		for _, v := range out {
			if v > maxVal {
				maxVal = v
			}
		}
		sumExp := float32(0.0)
		exps := make([]float32, vocabSize)
		temperature := 0.5
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

		s, _ := tok.IDToToken(best)
		s = strings.ReplaceAll(s, "Ġ", " ")
		sb.WriteString(s)

		copy(window, window[1:])
		window[ContextSize-1] = best
	}
	return sb.String()
}

func main() {
	rand.Seed(time.Now().UnixNano())
	fmt.Println("=== LOOM: Micro-Scatter Demo (Fixed) ===")
	fmt.Println("Architecture: [4-Tok Window] -> Compress -> Scatter[Guide|Worker] -> Mixer -> Head")

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
		base := "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. "
		for i := 0; i < 600; i++ {
			textData += base
		}
	}

	// 2. Train BPE
	jsonConfig := learnBPEAndGetConfig(textData, 500)
	tok, _ := tokenizer.LoadFromBytes([]byte(jsonConfig))
	vocabSize := len(tok.Vocab)
	fmt.Printf("Vocab Size: %d\n", vocabSize)

	// 3. Encode
	cleanText := strings.ReplaceAll(textData, " ", "Ġ")
	rawIDs := tok.Encode(cleanText, false)
	dataIDs := make([]int, len(rawIDs))
	for i, v := range rawIDs {
		dataIDs[i] = int(v)
	}
	fmt.Printf("Total Tokens: %d\n", len(dataIDs))

	// 4. Network Construction
	// Input: 4 * 32 = 128 floats
	flatInputSize := ContextSize * EmbeddingDim
	embTable := NewEmbeddingTable(vocabSize, EmbeddingDim)

	compressDim := 128
	guideDim := 64
	workerDim := 128 // SwiGLU projects input (64) back to (64)

	// FIX: Correct Scatter Output Size
	// Guide (32) + Worker (64) = 96
	scatterOutput := guideDim + workerDim

	networkJSON := fmt.Sprintf(`{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 5, 
		"layers_per_cell": 1,
		"layers": [
			{ 
				"type": "dense", 
				"id": "compressor",
				"input_height": %d, 
				"output_height": %d, 
				"activation": "tanh" 
			},
			{
				"type": "parallel",
				"id": "scatter_core",
				"combine_mode": "grid_scatter",
				"grid_output_rows": 1,
				"grid_output_cols": 2,
				"grid_output_layers": 1,
				"grid_positions": [
					{"branch_index": 0, "target_row": 0, "target_col": 0, "target_layer": 0},
					{"branch_index": 1, "target_row": 0, "target_col": 1, "target_layer": 0}
				],
				"branches": [
					{ 
						"type": "dense", 
						"id": "guide",
						"input_height": %d, 
						"output_height": %d, 
						"activation": "linear" 
					},
					{ 
						"type": "swiglu", 
						"id": "worker",
						"input_size": %d, 
						"output_size": %d
					}
				]
			},
			{
				"type": "layer_norm",
				"id": "norm",
				"norm_size": %d,
				"epsilon": 1e-5
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
		flatInputSize, compressDim, // 1. Compress (128 -> 64)

		compressDim, guideDim, // Branch 1: Guide (64 -> 32)
		compressDim, compressDim, // Branch 2: Worker (64 -> 64)

		scatterOutput,              // 3. Norm (96) - FIXED
		scatterOutput, compressDim, // 4. Mixer (96 -> 64)
		compressDim, vocabSize) // 5. Head (64 -> Vocab)

	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Fatal(err)
	}
	net.InitializeWeights()
	state := net.InitStepState(flatInputSize)

	// 5. Training
	steps := 150000
	baseLR := float32(0.003)
	ptr := 0

	window := make([]int, ContextSize)
	for i := range window {
		window[i] = 1
	}

	fmt.Println("Training...")
	start := time.Now()

	for i := 0; i < steps; i++ {
		currentLR := baseLR
		if i > 80000 {
			currentLR = 0.005
		}

		if i > 0 && i%5000 == 0 {
			fmt.Println("\n------------------------------------------------------------")
			fmt.Printf(" [MONITORING @ Step %d | LR %.4f]\n", i, currentLR)
			fmt.Printf(" Output: \"%s...\"\n", generate(net, embTable, tok, vocabSize, "Alice", 20))
			fmt.Println("------------------------------------------------------------")
		}

		if ptr+1 >= len(dataIDs) {
			ptr = 0
			for k := range window {
				window[k] = 1
			}
		}

		target := dataIDs[ptr]

		// 1. Embed
		inputVec := embTable.Forward(window)
		state.SetInput(inputVec)

		// 2. Forward
		net.StepForward(state)
		out := state.GetOutput()

		// 3. Loss
		grad := make([]float32, vocabSize)
		maxVal := out[0]
		for _, v := range out {
			if v > maxVal {
				maxVal = v
			}
		}
		sumExp := float32(0.0)
		exps := make([]float32, vocabSize)
		for j, v := range out {
			exps[j] = float32(math.Exp(float64(v - maxVal)))
			sumExp += exps[j]
		}

		loss := float32(0.0)
		for j := range out {
			prob := exps[j] / sumExp
			t := float32(0.0)
			if j == target {
				t = 1.0
			}
			grad[j] = prob - t
			if t > 0.5 {
				loss -= float32(math.Log(float64(prob + 1e-9)))
			}
		}

		// 4. Backward
		gradInput, _ := net.StepBackward(state, grad)
		net.ApplyGradients(currentLR)
		embTable.Backward(window, gradInput, currentLR)

		// 5. Update
		copy(window, window[1:])
		window[ContextSize-1] = target
		ptr++

		if i%5000 == 0 {
			pTok, _ := tok.IDToToken(0)
			bestProb := float32(-1.0)
			for j, p := range exps {
				if p > bestProb {
					bestProb = p
					pTok, _ = tok.IDToToken(j)
				}
			}
			tTok, _ := tok.IDToToken(target)
			pTok = strings.ReplaceAll(strings.ReplaceAll(pTok, "\n", "\\n"), "Ġ", "_")
			tTok = strings.ReplaceAll(strings.ReplaceAll(tTok, "\n", "\\n"), "Ġ", "_")

			fmt.Printf("Step %d | Loss %.4f | Pred: '%s' Exp: '%s'\n", i, loss, pTok, tTok)
		}
	}

	fmt.Printf("Training Complete in %v\n", time.Since(start))
	fmt.Println("\n=== FINAL GENERATION ===")
	fmt.Println(generate(net, embTable, tok, vocabSize, "Alice", 50))
	fmt.Println(generate(net, embTable, tok, vocabSize, "The", 50))
}
