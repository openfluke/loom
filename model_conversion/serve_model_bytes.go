package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/openfluke/loom/nn"
)

// Global model state
var (
	network    *nn.Network
	embeddings []float32
	finalNorm  []float32
	hiddenSize int
	vocabSize  int
	eosTokens  []int
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
	Done  bool `json:"done,omitempty"`
}

type TransformerConfig struct {
	ModelType        string      `json:"model_type"`
	HiddenSize       int         `json:"hidden_size"`
	IntermediateSize int         `json:"intermediate_size"`
	NumLayers        int         `json:"num_hidden_layers"`
	NumHeads         int         `json:"num_attention_heads"`
	NumKVHeads       int         `json:"num_key_value_heads"`
	VocabSize        int         `json:"vocab_size"`
	EOSTokenID       interface{} `json:"eos_token_id"` // Can be int or []int
	PadTokenID       int         `json:"pad_token_id"`
}

func main() {
	modelPath := flag.String("model", "", "Model name (e.g., Qwen/Qwen2.5-0.5B) or full path")
	port := flag.Int("port", 8080, "Port to serve on")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("Please provide -model path or model name (e.g., Qwen/Qwen2.5-0.5B)")
	}

	var finalModelPath string

	// Check if it's a short model name (contains /) or full path
	if strings.Contains(*modelPath, "/") && !strings.HasPrefix(*modelPath, "/") && !strings.HasPrefix(*modelPath, "~") && !strings.HasPrefix(*modelPath, ".") {
		// It's a model name like "Qwen/Qwen2.5-0.5B"
		homeDir, _ := os.UserHomeDir()
		modelDir := filepath.Join(homeDir, ".cache", "huggingface", "hub",
			"models--"+strings.ReplaceAll(*modelPath, "/", "--"), "snapshots")

		entries, err := os.ReadDir(modelDir)
		if err != nil || len(entries) == 0 {
			log.Fatalf("Model not found in HuggingFace cache: %s\nTry: huggingface-cli download %s", *modelPath, *modelPath)
		}

		finalModelPath = filepath.Join(modelDir, entries[0].Name())
		fmt.Printf("Resolved model name '%s' to: %s\n", *modelPath, finalModelPath)
	} else {
		// It's a full path
		finalModelPath = *modelPath
		// Expand tilde in path
		if strings.HasPrefix(finalModelPath, "~/") {
			home, _ := os.UserHomeDir()
			finalModelPath = filepath.Join(home, finalModelPath[2:])
		}
	}

	fmt.Printf("Loading model from: %s\n", finalModelPath)

	// Read config file into memory
	configPath := filepath.Join(finalModelPath, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		log.Fatalf("Failed to read config: %v", err)
	}
	fmt.Printf("✓ Read config.json (%d bytes)\n", len(configData))

	// Parse config to get vocab size and other metadata
	var config TransformerConfig
	if err := json.Unmarshal(configData, &config); err != nil {
		log.Fatalf("Failed to parse config: %v", err)
	}

	hiddenSize = config.HiddenSize
	vocabSize = config.VocabSize

	// Parse EOS tokens
	eosTokens = parseEOSTokens(config.EOSTokenID, config.PadTokenID)
	fmt.Printf("✓ EOS tokens: %v\n", eosTokens)

	// Read safetensors file into memory
	weightsPath := filepath.Join(finalModelPath, "model.safetensors")
	weightsData, err := os.ReadFile(weightsPath)
	if err != nil {
		log.Fatalf("Failed to read weights: %v", err)
	}
	fmt.Printf("✓ Read model.safetensors (%.2f MB)\n", float64(len(weightsData))/(1024*1024))

	// Load model using LoadTransformerFromBytes
	fmt.Println("\nLoading transformer network from bytes...")
	network, err = nn.LoadTransformerFromBytes(configData, weightsData)
	if err != nil {
		log.Fatalf("Failed to load transformer: %v", err)
	}

	// Load embeddings and final norm from safetensors
	fmt.Println("\nLoading embeddings and final norm...")
	tensors, err := nn.LoadSafetensorsFromBytes(weightsData)
	if err != nil {
		log.Fatalf("Failed to parse tensors: %v", err)
	}

	// Try different embedding key names
	embeddings = tryLoadTensor(tensors, []string{
		"model.embed_tokens.weight",
		"transformer.wte.weight",
		"embeddings.weight",
		"tok_embeddings.weight",
	})
	if len(embeddings) == 0 {
		log.Fatal("Failed to load embeddings")
	}
	fmt.Printf("✓ Loaded embeddings: %d values\n", len(embeddings))

	// Try different final norm key names
	finalNorm = tryLoadTensor(tensors, []string{
		"model.norm.weight",
		"transformer.ln_f.weight",
		"norm.weight",
		"ln_f.weight",
	})
	if len(finalNorm) > 0 {
		fmt.Printf("✓ Loaded final norm: %d values\n", len(finalNorm))
	} else {
		fmt.Println("⚠ No final norm found (some models don't have it)")
	}

	fmt.Printf("\n✅ Model loaded successfully!\n")
	fmt.Printf("   Hidden size: %d\n", hiddenSize)
	fmt.Printf("   Vocab size: %d\n", vocabSize)
	fmt.Printf("   Network layers: %d\n", len(network.Layers))

	// Set up HTTP server
	http.HandleFunc("/generate", handleGenerate)
	http.HandleFunc("/health", handleHealth)

	fmt.Printf("\n🚀 Server starting on http://localhost:%d\n", *port)
	fmt.Println("Endpoints:")
	fmt.Println("  POST /generate - Generate text")
	fmt.Println("  GET  /health   - Health check")
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", *port), nil))
}

func parseEOSTokens(eosTokenID interface{}, padTokenID int) []int {
	tokens := []int{}

	switch v := eosTokenID.(type) {
	case float64:
		tokens = append(tokens, int(v))
	case []interface{}:
		for _, item := range v {
			if num, ok := item.(float64); ok {
				tokens = append(tokens, int(num))
			}
		}
	}

	if padTokenID > 0 {
		tokens = append(tokens, padTokenID)
	}

	// Add common defaults if none found
	if len(tokens) == 0 {
		tokens = []int{2, 0} // Common EOS tokens
	}

	return tokens
}

func tryLoadTensor(tensors map[string][]float32, keys []string) []float32 {
	for _, key := range keys {
		if tensor, ok := tensors[key]; ok {
			fmt.Printf("  Found tensor: %s\n", key)
			return tensor
		}
	}
	return []float32{}
}

func isEOSToken(token int) bool {
	for _, eos := range eosTokens {
		if token == eos {
			return true
		}
	}
	return false
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "healthy",
		"model":  "loaded",
	})
}

func handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Set defaults
	if req.MaxNewTokens == 0 {
		req.MaxNewTokens = 100
	}
	if req.Temperature == 0 {
		req.Temperature = 0.7
	}

	if req.Stream {
		handleStreamGenerate(w, r, &req)
	} else {
		handleBatchGenerate(w, r, &req)
	}
}

func handleStreamGenerate(w http.ResponseWriter, r *http.Request, req *GenerateRequest) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	inputIDs := req.InputIDs
	for i := 0; i < req.MaxNewTokens; i++ {
		nextToken := generateNextToken(inputIDs, req.Temperature)

		// Send token
		response := StreamResponse{Token: nextToken}
		data, _ := json.Marshal(response)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()

		// Check for EOS
		if isEOSToken(nextToken) {
			response := StreamResponse{Done: true}
			data, _ := json.Marshal(response)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
			break
		}

		inputIDs = append(inputIDs, nextToken)
	}
}

func handleBatchGenerate(w http.ResponseWriter, r *http.Request, req *GenerateRequest) {
	outputIDs := make([]int, len(req.InputIDs))
	copy(outputIDs, req.InputIDs)

	for i := 0; i < req.MaxNewTokens; i++ {
		nextToken := generateNextToken(outputIDs, req.Temperature)
		outputIDs = append(outputIDs, nextToken)

		if isEOSToken(nextToken) {
			break
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(GenerateResponse{
		OutputIDs: outputIDs,
	})
}

func generateNextToken(inputIDs []int, temperature float32) int {
	seqLen := len(inputIDs)

	// Get embeddings for input tokens
	input := make([]float32, seqLen*hiddenSize)
	for i, tokenID := range inputIDs {
		if tokenID >= 0 && tokenID < vocabSize {
			startIdx := tokenID * hiddenSize
			copy(input[i*hiddenSize:(i+1)*hiddenSize], embeddings[startIdx:startIdx+hiddenSize])
		}
	}

	// Forward pass through network
	output, _ := network.ForwardCPU(input)

	// Apply final norm if available
	var normalized []float32
	if len(finalNorm) > 0 {
		normConfig := &nn.LayerConfig{
			Type:     nn.LayerRMSNorm,
			NormSize: hiddenSize,
			Gamma:    finalNorm,
			Epsilon:  1e-6,
		}
		normalized = nn.RmsNormForwardCPU(output, nil, normConfig, 1)
	} else {
		normalized = output
	}

	// Get last token's hidden state
	lastHidden := normalized[len(normalized)-hiddenSize:]

	// Compute logits (simplified - just dot product with embeddings)
	logits := make([]float32, vocabSize)
	for i := 0; i < vocabSize; i++ {
		embStart := i * hiddenSize
		embEnd := embStart + hiddenSize
		logits[i] = dotProduct(lastHidden, embeddings[embStart:embEnd])
	}

	// Apply temperature and sample
	return sampleToken(logits, temperature)
}

func dotProduct(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func sampleToken(logits []float32, temperature float32) int {
	// Apply temperature
	for i := range logits {
		logits[i] /= temperature
	}

	// Softmax
	maxLogit := logits[0]
	for _, l := range logits {
		if l > maxLogit {
			maxLogit = l
		}
	}

	expSum := float32(0)
	probs := make([]float32, len(logits))
	for i, l := range logits {
		probs[i] = float32(math.Exp(float64(l - maxLogit)))
		expSum += probs[i]
	}
	for i := range probs {
		probs[i] /= expSum
	}

	// Greedy sampling (take argmax)
	maxProb := probs[0]
	maxIdx := 0
	for i, p := range probs {
		if p > maxProb {
			maxProb = p
			maxIdx = i
		}
	}

	return maxIdx
}
