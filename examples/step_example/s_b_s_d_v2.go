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

// --- Configuration ---
const (
	ContextSize  = 32
	EmbeddingDim = 64
	BatchSize    = 32    // Accumulate gradients for 32 steps before updating
	LearningRate = 0.001 // Lower LR because Adam is more efficient than SGD
)

// --- 1. AdamW Optimizer ---
// This struct handles the "Momentum" that fixes the glitchy loss
type AdamW struct {
	lr, beta1, beta2, epsilon, weightDecay float32
	m, v                                   map[string][]float32 // State storage
	t                                      float32              // Time step
}

func NewAdamW(lr float32) *AdamW {
	return &AdamW{
		lr:          lr,
		beta1:       0.9,
		beta2:       0.999,
		epsilon:     1e-8,
		weightDecay: 1e-4,
		m:           make(map[string][]float32),
		v:           make(map[string][]float32),
		t:           0,
	}
}

// Update applies the AdamW logic to a slice of weights using accumulated gradients
// key: unique identifier for this weight set (to persist state)
func (opt *AdamW) Update(key string, weights []float32, grads []float32) {
	if len(weights) == 0 || len(grads) == 0 {
		return
	}

	// Initialize state if first time seeing these weights
	if _, ok := opt.m[key]; !ok {
		opt.m[key] = make([]float32, len(weights))
		opt.v[key] = make([]float32, len(weights))
	}

	m := opt.m[key]
	v := opt.v[key]

	// Adam works better when gradients are averaged, not just summed.
	// Since we summed them in the accumulator, we divide by BatchSize here implicitly
	// or we can just let the LR handle it. Let's do a rough scaling for stability.
	scale := float32(1.0) / float32(BatchSize)

	// Bias corrections
	biasCorrection1 := 1.0 - float32(math.Pow(float64(opt.beta1), float64(opt.t)))
	biasCorrection2 := 1.0 - float32(math.Pow(float64(opt.beta2), float64(opt.t)))

	for i := range weights {
		g := grads[i] * scale // Average the gradient

		// Update moments
		m[i] = opt.beta1*m[i] + (1-opt.beta1)*g
		v[i] = opt.beta2*v[i] + (1-opt.beta2)*g*g

		// Compute update
		mHat := m[i] / biasCorrection1
		vHat := v[i] / biasCorrection2

		update := (opt.lr * mHat) / (float32(math.Sqrt(float64(vHat))) + opt.epsilon)

		// Apply Weight Decay + Update
		weights[i] = weights[i] - update - (opt.lr * opt.weightDecay * weights[i])
	}
}

// --- 2. Embedding Table (With Accumulation) ---
type EmbeddingTable struct {
	Weights [][]float32
	Grads   [][]float32 // Accumulator for gradients
	Dim     int
}

func NewEmbeddingTable(vocabSize, dim int) *EmbeddingTable {
	weights := make([][]float32, vocabSize)
	grads := make([][]float32, vocabSize) // Dense gradient buffer
	scale := float32(math.Sqrt(2.0 / float64(dim)))

	for i := range weights {
		weights[i] = make([]float32, dim)
		grads[i] = make([]float32, dim)
		for j := range weights[i] {
			weights[i][j] = (rand.Float32()*2 - 1) * scale
		}
	}
	return &EmbeddingTable{Weights: weights, Grads: grads, Dim: dim}
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

// Accumulate gradients instead of updating immediately
func (e *EmbeddingTable) Accumulate(window []int, gradInput []float32) {
	for i, id := range window {
		if id < len(e.Grads) {
			start := i * e.Dim
			end := start + e.Dim
			chunk := gradInput[start:end]
			for j := 0; j < e.Dim; j++ {
				e.Grads[id][j] += chunk[j] // Just Add!
			}
		}
	}
}

// Apply updates using AdamW
func (e *EmbeddingTable) Update(opt *AdamW) {
	// Flatten weights and grads for the optimizer to handle them as one block
	// (Or iterate rows. Iterating rows is safer for memory)
	for i := range e.Weights {
		key := fmt.Sprintf("emb_%d", i)
		opt.Update(key, e.Weights[i], e.Grads[i])

		// Reset gradients to zero after update
		for j := range e.Grads[i] {
			e.Grads[i][j] = 0
		}
	}
}

// --- 3. BPE Utilities (Unchanged) ---
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

// --- 4. Generator ---
func generate(net *nn.Network, embs *EmbeddingTable, tok *tokenizer.Tokenizer, vocabSize int, seed string, length int) string {
	window := make([]int, ContextSize)
	for i := range window {
		window[i] = 1 // PAD
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
		temperature := 0.7 // Slightly higher temp for creative generation
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
	fmt.Println("=== LOOM: Smart-Batch Demo (AdamW + Gradient Accumulation) ===")
	fmt.Println("Architecture: [32-Tok Window] -> Compress -> Scatter -> Mixer -> Head")

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
	flatInputSize := ContextSize * EmbeddingDim
	embTable := NewEmbeddingTable(vocabSize, EmbeddingDim)

	compressDim := 128
	guideDim := 64
	workerDim := 128
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
		flatInputSize, compressDim,
		compressDim, guideDim,
		compressDim, compressDim,
		scatterOutput,
		scatterOutput, compressDim,
		compressDim, vocabSize)

	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Fatal(err)
	}
	net.InitializeWeights()
	state := net.InitStepState(flatInputSize)

	// --- SETUP GRADIENT ACCUMULATORS ---
	// We need buffers to store gradients during the batch accumulation
	// because net.KernelGradients() gets overwritten every step.
	accKernels := make([][]float32, len(net.Layers))
	accBiases := make([][]float32, len(net.Layers))
	for i, layer := range net.Layers {
		if len(layer.Kernel) > 0 {
			accKernels[i] = make([]float32, len(layer.Kernel))
		}
		if len(layer.Bias) > 0 {
			accBiases[i] = make([]float32, len(layer.Bias))
		}
	}

	optimizer := NewAdamW(LearningRate)

	// 5. Training Loop
	steps := 150000
	ptr := 0
	batchCounter := 0
	epochLoss := 0.0

	window := make([]int, ContextSize)
	for i := range window {
		window[i] = 1 // PAD
	}

	fmt.Println("Training with AdamW + Batch Accumulation...")
	start := time.Now()

	for i := 0; i < steps; i++ {
		// Reset window if we hit end of data
		if ptr+1 >= len(dataIDs) {
			ptr = 0
			for k := range window {
				window[k] = 1
			}
		}

		target := dataIDs[ptr]

		// 1. Forward Pass
		inputVec := embTable.Forward(window)
		state.SetInput(inputVec)
		net.StepForward(state)
		out := state.GetOutput()

		// 2. Compute Loss & Gradients
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

		stepLoss := float32(0.0)
		for j := range out {
			prob := exps[j] / sumExp
			t := float32(0.0)
			if j == target {
				t = 1.0
			}
			grad[j] = prob - t
			if t > 0.5 {
				stepLoss -= float32(math.Log(float64(prob + 1e-9)))
			}
		}
		epochLoss += float64(stepLoss)

		// 3. Backward Pass (Calculates Gradients for THIS step)
		gradInput, _ := net.StepBackward(state, grad)

		// 4. ACCUMULATE GRADIENTS (Do not update yet!)
		// Accumulate Embedding Gradients
		embTable.Accumulate(window, gradInput)

		// Accumulate Network Gradients
		currentKernelGrads := net.KernelGradients()
		currentBiasGrads := net.BiasGradients()

		for lIdx := range net.Layers {
			// Kernel Accumulation
			if len(accKernels[lIdx]) > 0 && len(currentKernelGrads[lIdx]) > 0 {
				for k := range accKernels[lIdx] {
					accKernels[lIdx][k] += currentKernelGrads[lIdx][k]
				}
			}
			// Bias Accumulation
			if len(accBiases[lIdx]) > 0 && len(currentBiasGrads[lIdx]) > 0 {
				for k := range accBiases[lIdx] {
					accBiases[lIdx][k] += currentBiasGrads[lIdx][k]
				}
			}
		}

		// 5. Sliding Window Update
		copy(window, window[1:])
		window[ContextSize-1] = target
		ptr++
		batchCounter++

		// 6. OPTIMIZER STEP (Only run every BatchSize steps)
		if batchCounter >= BatchSize {
			optimizer.t++ // Increment Optimizer time step

			// Apply AdamW to Network Layers
			for lIdx := range net.Layers {
				// Update Kernels
				if len(net.Layers[lIdx].Kernel) > 0 {
					key := fmt.Sprintf("k_%d", lIdx)
					optimizer.Update(key, net.Layers[lIdx].Kernel, accKernels[lIdx])
					// Zero accumulator
					for k := range accKernels[lIdx] {
						accKernels[lIdx][k] = 0
					}
				}
				// Update Biases
				if len(net.Layers[lIdx].Bias) > 0 {
					key := fmt.Sprintf("b_%d", lIdx)
					optimizer.Update(key, net.Layers[lIdx].Bias, accBiases[lIdx])
					// Zero accumulator
					for k := range accBiases[lIdx] {
						accBiases[lIdx][k] = 0
					}
				}
			}

			// Apply AdamW to Embeddings
			embTable.Update(optimizer)

			batchCounter = 0
		}

		// Monitoring
		if i > 0 && i%5000 == 0 {
			avgLoss := epochLoss / 5000.0
			epochLoss = 0.0

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

			fmt.Println("\n------------------------------------------------------------")
			fmt.Printf(" [Step %d] Avg Loss: %.4f | Pred: '%s' Exp: '%s'\n", i, avgLoss, pTok, tTok)
			fmt.Printf(" Gen: \"%s...\"\n", generate(net, embTable, tok, vocabSize, "Alice", 30))
			fmt.Println("------------------------------------------------------------")
		}
	}

	fmt.Printf("Training Complete in %v\n", time.Since(start))
	fmt.Println("\n=== FINAL GENERATION ===")
	fmt.Println(generate(net, embTable, tok, vocabSize, "Alice", 100))
}
