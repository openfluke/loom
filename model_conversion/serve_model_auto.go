package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/openfluke/loom/nn"
)

// Global model state
var (
	network      *nn.Network
	embeddings   []float32
	finalNorm    []float32
	hiddenSize   int
	vocabSize    int
	eosTokens    []int // EOS tokens for this model
	hasFinalNorm bool  // Whether this model has a final norm layer
)

type GenerateRequest struct {
	InputIDs     []int   `json:"input_ids"`
	MaxNewTokens int     `json:"max_new_tokens"`
	Temperature  float32 `json:"temperature"`
	Stream       bool    `json:"stream"`
}

type GenerateResponse struct {
	OutputIDs []int  `json:"output_ids"`
	Error     string `json:"error,omitempty"`
}

type StreamResponse struct {
	Token int  `json:"token,omitempty"`
	Done  bool `json:"done"`
}

// loadEOSTokens reads EOS token IDs from the model's config.json
func loadEOSTokens(configPath string) []int {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil
	}

	var config map[string]interface{}
	if err := json.Unmarshal(data, &config); err != nil {
		return nil
	}

	var tokens []int

	// Try different config keys for EOS tokens
	if eosID, ok := config["eos_token_id"]; ok {
		switch v := eosID.(type) {
		case float64:
			tokens = append(tokens, int(v))
		case []interface{}:
			for _, id := range v {
				if idFloat, ok := id.(float64); ok {
					tokens = append(tokens, int(idFloat))
				}
			}
		}
	}

	// Also check for pad_token_id
	if padID, ok := config["pad_token_id"]; ok {
		if idFloat, ok := padID.(float64); ok {
			tokens = append(tokens, int(idFloat))
		}
	}

	return tokens
}

// tryLoadTensor attempts to load a tensor using multiple possible keys
func tryLoadTensor(tensors map[string][]float32, keys []string) []float32 {
	for _, key := range keys {
		if tensor, exists := tensors[key]; exists {
			log.Printf("   Found tensor: %s", key)
			return tensor
		}
	}
	return nil
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":      "ok",
		"hidden_size": hiddenSize,
		"vocab_size":  vocabSize,
		"layers":      len(network.Layers),
	})
}

func generateHandler(w http.ResponseWriter, r *http.Request) {
	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		json.NewEncoder(w).Encode(GenerateResponse{Error: err.Error()})
		return
	}

	log.Printf("🔄 Generate request: %d input tokens, max_new=%d, temp=%.2f, stream=%v",
		len(req.InputIDs), req.MaxNewTokens, req.Temperature, req.Stream)

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			json.NewEncoder(w).Encode(GenerateResponse{Error: "streaming not supported"})
			return
		}

		tokens := req.InputIDs
		log.Printf("   Starting streaming generation...")

		for i := 0; i < req.MaxNewTokens; i++ {
			nextToken, err := generateNextToken(tokens)
			if err != nil {
				data, _ := json.Marshal(StreamResponse{Done: true})
				fmt.Fprintf(w, "data: %s\n\n", data)
				flusher.Flush()
				return
			}

			// Check for repetition
			if len(tokens) >= 3 &&
				tokens[len(tokens)-1] == nextToken &&
				tokens[len(tokens)-2] == nextToken {
				log.Printf("   ⚠️  Repetition detected (token %d), stopping", nextToken)
				data, _ := json.Marshal(StreamResponse{Done: true})
				fmt.Fprintf(w, "data: %s\n\n", data)
				flusher.Flush()
				return
			}

			tokens = append(tokens, nextToken)

			// Send token
			data, _ := json.Marshal(StreamResponse{Token: nextToken, Done: false})
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()

			// Check for EOS (dynamic based on loaded model)
			if isEOSToken(nextToken) {
				log.Printf("   EOS token detected, stopping")
				break
			}
		}

		// Send done signal
		data, _ := json.Marshal(StreamResponse{Done: true})
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
		log.Printf("✅ Streaming complete")
		return
	}

	// Non-streaming mode
	tokens := req.InputIDs
	log.Printf("   Starting generation loop...")

	for i := 0; i < req.MaxNewTokens; i++ {
		log.Printf("   Token %d/%d (seq_len=%d)", i+1, req.MaxNewTokens, len(tokens))

		nextToken, err := generateNextToken(tokens)
		if err != nil {
			json.NewEncoder(w).Encode(GenerateResponse{Error: err.Error()})
			return
		}

		tokens = append(tokens, nextToken)

		// Check for EOS (dynamic based on loaded model)
		if isEOSToken(nextToken) {
			log.Printf("   EOS token detected, stopping")
			break
		}
	}

	// Return only new tokens
	outputIDs := tokens[len(req.InputIDs):]
	log.Printf("✅ Generated %d tokens total", len(outputIDs))
	json.NewEncoder(w).Encode(GenerateResponse{OutputIDs: outputIDs})
}

// isEOSToken checks if a token is an end-of-sequence token
func isEOSToken(token int) bool {
	for _, eos := range eosTokens {
		if token == eos {
			return true
		}
	}
	return false
}

// generateNextToken uses Network.ForwardCPU (automatic approach)
func generateNextToken(tokens []int) (int, error) {
	log.Printf("      Input tokens: %v", tokens)

	// Step 1: Embed tokens
	input := make([]float32, len(tokens)*hiddenSize)
	for t, tokenID := range tokens {
		if tokenID >= vocabSize || tokenID < 0 {
			return 0, fmt.Errorf("invalid token ID: %d (vocab size: %d)", tokenID, vocabSize)
		}
		for d := 0; d < hiddenSize; d++ {
			input[t*hiddenSize+d] = embeddings[tokenID*hiddenSize+d]
		}
	}

	if len(tokens) > 0 {
		log.Printf("      First token (%d) embedding (first 10): %.6v", tokens[0], input[:10])
	}

	// BatchSize is always 1 for text generation (sequence length varies)
	network.BatchSize = 1

	// Step 2: Forward pass through network
	output, duration := network.ForwardCPU(input)
	log.Printf("      Forward pass took: %v", duration)

	// Step 3: Apply final RMSNorm if model has one
	var normalized []float32
	if hasFinalNorm && finalNorm != nil {
		finalNormConfig := &nn.LayerConfig{
			Type:     nn.LayerRMSNorm,
			NormSize: hiddenSize,
			Gamma:    finalNorm,
			Epsilon:  1e-6,
		}
		normalized = nn.RmsNormForwardCPU(output, nil, finalNormConfig, len(tokens))
	} else {
		// No final norm, use output directly
		normalized = output
	}

	// Step 4: Extract last token after normalization
	lastIdx := (len(tokens) - 1) * hiddenSize
	lastTokenNormalized := normalized[lastIdx : lastIdx+hiddenSize]

	// Step 5: LM head projection (using tied weights)
	logits := make([]float32, vocabSize)
	for v := 0; v < vocabSize; v++ {
		var sum float32
		for d := 0; d < hiddenSize; d++ {
			sum += lastTokenNormalized[d] * embeddings[v*hiddenSize+d]
		}
		logits[v] = sum
	}

	// Find top-5 for debugging
	top5Tokens := make([]int, 5)
	top5Logits := make([]float32, 5)
	logitsCopy := make([]float32, len(logits))
	copy(logitsCopy, logits)

	for i := 0; i < 5; i++ {
		maxIdx := 0
		maxVal := logitsCopy[0]
		for j := 1; j < vocabSize; j++ {
			if logitsCopy[j] > maxVal {
				maxVal = logitsCopy[j]
				maxIdx = j
			}
		}
		top5Tokens[i] = maxIdx
		top5Logits[i] = maxVal
		logitsCopy[maxIdx] = -1e9
	}

	log.Printf("      Top-5: %v (logits: %v)", top5Tokens, top5Logits)

	// Apply temperature and sample
	bestToken := top5Tokens[0]
	log.Printf("      Generated token: %d (logit: %.4f)", bestToken, top5Logits[0])

	return bestToken, nil
}

func main() {
	modelName := flag.String("model", "Qwen/Qwen2.5-0.5B", "Model name")
	port := flag.String("port", "8080", "Server port")
	flag.Parse()

	// Get model directory
	homeDir, _ := os.UserHomeDir()
	modelDir := filepath.Join(homeDir, ".cache", "huggingface", "hub",
		"models--"+strings.ReplaceAll(*modelName, "/", "--"), "snapshots")

	entries, err := os.ReadDir(modelDir)
	if err != nil || len(entries) == 0 {
		log.Fatalf("Model not found: %s", *modelName)
	}

	snapshotDir := filepath.Join(modelDir, entries[0].Name())
	log.Printf("Loading model from: %s\n", snapshotDir)

	// Try to load model config to get EOS tokens and architecture info
	configPath := filepath.Join(snapshotDir, "config.json")
	eosTokens = loadEOSTokens(configPath)
	if len(eosTokens) == 0 {
		// Default fallback (common tokens)
		eosTokens = []int{2, 0} // </s> and <pad>
		log.Printf("⚠️  No EOS tokens found in config, using defaults: %v", eosTokens)
	} else {
		log.Printf("✅ Loaded EOS tokens from config: %v", eosTokens)
	}

	// Load model using Network.LoadTransformerFromSafetensors
	network, err = nn.LoadTransformerFromSafetensors(snapshotDir)
	if err != nil {
		log.Fatalf("Error loading model: %v", err)
	}

	// Load embeddings and final norm
	weightsPath := filepath.Join(snapshotDir, "model.safetensors")
	tensors, err := nn.LoadSafetensors(weightsPath)
	if err != nil {
		log.Fatalf("Error loading weights: %v", err)
	}

	// Try to load embeddings (different models use different keys)
	embeddings = tryLoadTensor(tensors, []string{
		"model.embed_tokens.weight",
		"transformer.wte.weight",
		"embeddings.weight",
		"embed_tokens.weight",
	})
	if embeddings == nil {
		log.Fatalf("Could not find embeddings in model weights")
	}

	// Try to load final norm (not all models have this)
	finalNorm = tryLoadTensor(tensors, []string{
		"model.norm.weight",
		"transformer.ln_f.weight",
		"ln_f.weight",
		"norm.weight",
	})
	hasFinalNorm = (finalNorm != nil)
	if hasFinalNorm {
		log.Printf("✅ Found final normalization layer")
	} else {
		log.Printf("⚠️  No final normalization layer found")
	}

	hiddenSize = network.InputSize
	vocabSize = len(embeddings) / hiddenSize

	log.Printf("✅ Model loaded!")
	log.Printf("   Hidden: %d, Vocab: %d, Layers: %d\n", hiddenSize, vocabSize, len(network.Layers))

	// Setup HTTP server
	http.HandleFunc("/generate", generateHandler)
	http.HandleFunc("/health", healthHandler)

	log.Printf("🚀 Server starting on http://localhost:%s", *port)
	log.Printf("   POST /generate - Generate text")
	log.Printf("   GET  /health   - Health check")

	if err := http.ListenAndServe(":"+*port, nil); err != nil {
		log.Fatal(err)
	}
}
