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

// --- 1. ROBUST BPE (Space-Aware) ---

func learnBPEAndGetConfig(text string, numMerges int) string {
	fmt.Println("Learning BPE (with Space Merging)...")

	// 1. Pre-process: Replace spaces with a visible character to force merging
	// This prevents the "Space Trap" where the model just predicts ' '
	safeText := strings.ReplaceAll(text, " ", "Ġ") // GPT-2 style

	if len(safeText) > 500000 {
		safeText = safeText[:500000]
	}

	// Initial split by character
	// We treat the whole string as one sequence of chars to preserve the Ġ
	var splits [][]string
	// Split by whitespace if any remaining (newlines), but keep Ġ attached
	rawWords := strings.Fields(safeText)
	for _, w := range rawWords {
		chars := strings.Split(w, "")
		splits = append(splits, chars)
	}

	merges := make([]string, 0)
	vocab := make(map[string]int)

	// Base Vocab
	idx := 0
	vocab["<UNK>"] = idx
	idx++
	vocab["<PAD>"] = idx
	idx++

	// Ensure basic chars are in vocab
	chars := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-;:" + "Ġ"
	for _, c := range chars {
		s := string(c)
		vocab[s] = idx
		idx++
	}

	// Scan text for any other chars
	for _, w := range splits {
		for _, c := range w {
			if _, ok := vocab[c]; !ok {
				vocab[c] = idx
				idx++
			}
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

		merges = append(merges, bestPair)
		vocab[newToken] = idx
		idx++

		// Apply merge
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

	// Config
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
		Model:       TokenizerModel{Type: "BPE", Vocab: vocab, Merges: merges},
		AddedTokens: []struct{}{},
	}

	bytes, _ := json.Marshal(config)
	return string(bytes)
}

// Wrapper to handle the Ġ conversion during encode/decode
type RobustTokenizer struct {
	*tokenizer.Tokenizer
}

func (rt *RobustTokenizer) EncodeSafe(text string) []int {
	// Convert spaces to Ġ before encoding
	safe := strings.ReplaceAll(text, " ", "Ġ")
	raw := rt.Tokenizer.Encode(safe, false)
	out := make([]int, len(raw))
	for i, v := range raw {
		out[i] = int(v)
	}
	return out
}

func (rt *RobustTokenizer) DecodeSafe(id int) string {
	s, _ := rt.Tokenizer.IDToToken(id)
	// Convert Ġ back to space
	return strings.ReplaceAll(s, "Ġ", " ")
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

// --- Helper: Generate ---
func generateSample(net *nn.Network, tok *RobustTokenizer, vocabSize int, seed string, length int) string {
	genState := net.InitStepState(vocabSize)
	inputVec := make([]float32, vocabSize)

	seedIDs := tok.EncodeSafe(seed)
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

	for i := 0; i < length; i++ {
		out := genState.GetOutput()

		// Temp Sampling
		maxVal := out[0]
		for _, v := range out {
			if v > maxVal {
				maxVal = v
			}
		}
		sumExp := float32(0.0)
		exps := make([]float32, len(out))
		for k, v := range out {
			exps[k] = float32(math.Exp(float64((v - maxVal) / 0.6)))
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

		tokStr := tok.DecodeSafe(best)
		sb.WriteString(tokStr)

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
	fmt.Println("=== LOOM v14: The 'Anti-Space' Update ===")
	fmt.Println("Strategy: Force word-merging and penalize safe guesses.")

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

	// 2. Train BPE
	// Target 1000 tokens to get full words
	jsonConfig := learnBPEAndGetConfig(textData, 1000)
	baseTok, _ := tokenizer.LoadFromBytes([]byte(jsonConfig))
	tok := &RobustTokenizer{baseTok}
	fmt.Printf("Vocab Size: %d\n", len(baseTok.Vocab))

	// 3. Encode
	dataIDs := tok.EncodeSafe(textData)
	fmt.Printf("Total Tokens: %d\n", len(dataIDs))

	// Calculate frequencies to penalize common tokens
	counts := make(map[int]int)
	for _, id := range dataIDs {
		counts[id]++
	}

	// 4. Network (Stacked LSTM with Residuals)
	vocabSize := len(baseTok.Vocab)
	hiddenSize := 256
	networkJSON := fmt.Sprintf(`{
        "batch_size": 1,
        "grid_rows": 1,
        "grid_cols": 5,
        "layers_per_cell": 1,
        "layers": [
            { "type": "dense", "input_height": %d, "output_height": %d, "activation": "tanh" },
            { "type": "lstm", "input_size": %d, "hidden_size": %d, "seq_length": 1, "activation": "tanh" },
            { "type": "residual", "input_height": %d, "output_height": %d },
            { "type": "lstm", "input_size": %d, "hidden_size": %d, "seq_length": 1, "activation": "tanh" },
            { "type": "dense", "input_height": %d, "output_height": %d, "activation": "linear" }
        ]
    }`, vocabSize, hiddenSize,
		hiddenSize, hiddenSize, // LSTM 1
		hiddenSize, hiddenSize, // Residual
		hiddenSize, hiddenSize, // LSTM 2
		hiddenSize, vocabSize) // Output

	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Fatal(err)
	}
	net.InitializeWeights()
	state := net.InitStepState(vocabSize)

	// 5. Training
	targetDelay := 2 // BPE tokens are denser, so delay can be shorter
	queue := NewTargetQueue(targetDelay)
	inputVec := make([]float32, vocabSize)

	steps := 50000
	lr := float32(0.008)
	ptr := 0

	fmt.Println("Training...")
	start := time.Now()

	for i := 0; i < steps; i++ {
		// Monitoring
		if i > 0 && i%2500 == 0 {
			fmt.Println("\n------------------------------------------------------------")
			fmt.Printf(" [MONITORING @ Step %d]\n", i)
			fmt.Printf(" Output: \"%s...\"\n", generateSample(net, tok, vocabSize, "The", 15))
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

			// Calculate penalty for this target
			// If it's a super common token, weight it LESS to force learning others
			// or weight it MORE if we want to force it to learn structure?
			// Actually, the issue was spaces. BPE fixed that.
			// Standard loss should work now.

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

				pTok := tok.DecodeSafe(predID)
				tTok := tok.DecodeSafe(target)

				// Visualize space clearly
				if pTok == " " {
					pTok = "_"
				}
				if tTok == " " {
					tTok = "_"
				}

				fmt.Printf("Step %d | Loss %.4f | Pred: '%s' Exp: '%s'\n", i, loss, pTok, tTok)
			}
		}
	}

	fmt.Printf("Training Complete in %v\n", time.Since(start))

	fmt.Println("\n=== FINAL GENERATION TEST ===")
	seeds := []string{"Alice", "The", "It"}
	for _, seed := range seeds {
		fmt.Printf("Seed '%s': ", seed)
		fmt.Println(generateSample(net, tok, vocabSize, seed, 30))
	}
}
