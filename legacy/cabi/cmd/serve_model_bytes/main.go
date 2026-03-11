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

// Global state initialized via C ABI
var (
	configData  []byte
	weightsData []byte
	modelPath   string
	modelName   string
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

type HealthResponse struct {
	Status    string `json:"status"`
	Model     string `json:"model"`
	Tokenizer string `json:"tokenizer"`
	Backend   string `json:"backend"`
}

type TransformerConfig struct {
	ModelType string `json:"model_type"`
	VocabSize int    `json:"vocab_size"`
	NumLayers int    `json:"num_hidden_layers"`
}

func main() {
	model := flag.String("model", "", "Model path (e.g., models/SmolLM2-135M-Instruct)")
	port := flag.Int("port", 8080, "Port to serve on")
	flag.Parse()

	if *model == "" {
		log.Fatal("Please provide -model path")
	}

	modelPath = *model
	modelName = filepath.Base(modelPath)

	// Expand tilde
	if strings.HasPrefix(modelPath, "~/") {
		home, _ := os.UserHomeDir()
		modelPath = filepath.Join(home, modelPath[2:])
	}

	fmt.Printf("🚀 LOOM Transformer Inference Server (C ABI)\n")
	fmt.Printf("============================================\n\n")
	fmt.Printf("Loading model from: %s\n", modelPath)

	// Read files into memory
	var err error
	configData, err = os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		log.Fatalf("Failed to read config.json: %v", err)
	}
	fmt.Printf("✓ Loaded config.json (%d bytes)\n", len(configData))

	weightsData, err = os.ReadFile(filepath.Join(modelPath, "model.safetensors"))
	if err != nil {
		log.Fatalf("Failed to read model.safetensors: %v", err)
	}
	fmt.Printf("✓ Loaded model.safetensors (%.2f MB)\n", float64(len(weightsData))/(1024*1024))

	// Parse config for metadata
	var config TransformerConfig
	if err := json.Unmarshal(configData, &config); err != nil {
		log.Fatalf("Failed to parse config: %v", err)
	}

	fmt.Printf("\n📊 Model Info:\n")
	fmt.Printf("   Type: %s\n", config.ModelType)
	fmt.Printf("   Vocab: %d\n", config.VocabSize)
	fmt.Printf("   Layers: %d\n", config.NumLayers)

	// Set up routes
	http.HandleFunc("/health", handleHealth)
	http.HandleFunc("/config", handleConfig)
	http.HandleFunc("/weights", handleWeights)
	http.HandleFunc("/generate", handleGenerate)

	fmt.Printf("\n🌐 Server running on http://localhost:%d\n", *port)
	fmt.Printf("   GET  /health  - Health check\n")
	fmt.Printf("   GET  /config  - Model config (JSON)\n")
	fmt.Printf("   GET  /weights - Model weights (safetensors)\n")
	fmt.Printf("   POST /generate - Generate tokens\n")
	fmt.Println()

	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", *port), nil))
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	resp := HealthResponse{
		Status:    "ok",
		Model:     modelName,
		Tokenizer: "pure-go-bpe",
		Backend:   "loom-cabi",
	}
	json.NewEncoder(w).Encode(resp)
}

func handleConfig(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Write(configData)
}

func handleWeights(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(weightsData)))
	w.Write(weightsData)
}

func handleGenerate(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		json.NewEncoder(w).Encode(GenerateResponse{Error: err.Error()})
		return
	}

	// This endpoint just returns the input_ids for now
	// The actual generation happens client-side via C ABI or WASM
	json.NewEncoder(w).Encode(GenerateResponse{
		OutputIDs: req.InputIDs,
	})
}

// tryLoadTensor attempts to load a tensor by trying multiple key names
func tryLoadTensor(tensors map[string][]float32, keys []string) []float32 {
	for _, key := range keys {
		if data, ok := tensors[key]; ok {
			return data
		}
	}
	return nil
}

// parseEOSTokens extracts EOS tokens from config
func parseEOSTokens(eosTokenID interface{}, padTokenID int) []int {
	eosTokens := []int{}

	switch v := eosTokenID.(type) {
	case float64:
		eosTokens = append(eosTokens, int(v))
	case []interface{}:
		for _, id := range v {
			if fid, ok := id.(float64); ok {
				eosTokens = append(eosTokens, int(fid))
			}
		}
	}

	// Add pad token if not already included
	found := false
	for _, eos := range eosTokens {
		if eos == padTokenID {
			found = true
			break
		}
	}
	if !found && padTokenID > 0 {
		eosTokens = append(eosTokens, padTokenID)
	}

	return eosTokens
}

// Ensure nn package is imported (for LoadSafetensorsFromBytes if needed)
var _ = nn.Network{}
