package main

import (
	"bufio"
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/openfluke/loom/tokenizer"
)

//go:embed templates/*
var templateFS embed.FS

var (
	tk         *tokenizer.Tokenizer
	backendURL string
	modelName  string
)

type GenerateRequest struct {
	Prompt    string `json:"prompt"`
	MaxTokens int    `json:"max_tokens"`
}

type BackendRequest struct {
	InputIDs     []uint32 `json:"input_ids"`
	MaxNewTokens int      `json:"max_new_tokens"`
	Temperature  float32  `json:"temperature"`
	Stream       bool     `json:"stream"`
}

type BackendResponse struct {
	OutputIDs []int  `json:"output_ids"`
	Error     string `json:"error,omitempty"`
}

type StreamResponse struct {
	Token   string `json:"token,omitempty"`
	TokenID int    `json:"token_id,omitempty"`
	Done    bool   `json:"done,omitempty"`
	Error   string `json:"error,omitempty"`
}

type HealthResponse struct {
	WebInterface string `json:"web_interface"`
	Backend      string `json:"backend"`
	Model        string `json:"model"`
	Tokenizer    string `json:"tokenizer"`
}

func main() {
	modelPath := flag.String("model", "Qwen/Qwen2.5-0.5B", "Model name (same as backend)")
	backend := flag.String("backend", "http://localhost:8080", "Backend server URL")
	port := flag.Int("port", 5000, "Web interface port")
	flag.Parse()

	backendURL = *backend
	modelName = *modelPath

	// Load tokenizer
	fmt.Println("Loading pure Go BPE tokenizer...")
	if err := loadTokenizer(*modelPath); err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}
	fmt.Printf("✓ Tokenizer loaded (vocab size: %d)\n", tk.VocabSize())

	// Check backend health
	if err := checkBackend(); err != nil {
		fmt.Printf("⚠️  Warning: Backend server not detected at %s\n", backendURL)
		fmt.Println("   Make sure serve_model_bytes is running!")
	} else {
		fmt.Println("✅ Backend server detected")
	}

	// Set up routes
	http.HandleFunc("/", handleIndex)
	http.HandleFunc("/generate_stream", handleGenerateStream)
	http.HandleFunc("/generate", handleGenerate)
	http.HandleFunc("/health", handleHealth)

	fmt.Printf("\n🚀 Starting web interface on http://localhost:%d\n", *port)
	fmt.Printf("   Model: %s\n", modelName)
	fmt.Printf("   Backend: %s\n", backendURL)
	fmt.Printf("   Tokenizer: Pure Go BPE\n")

	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", *port), nil))
}

func loadTokenizer(modelPath string) error {
	var tokenizerPath string

	// Check if it's a short model name or full path
	if strings.Contains(modelPath, "/") && !strings.HasPrefix(modelPath, "/") && !strings.HasPrefix(modelPath, "~") && !strings.HasPrefix(modelPath, ".") {
		// It's a model name like "Qwen/Qwen2.5-0.5B"
		homeDir, _ := os.UserHomeDir()
		modelDir := filepath.Join(homeDir, ".cache", "huggingface", "hub",
			"models--"+strings.ReplaceAll(modelPath, "/", "--"), "snapshots")

		entries, err := os.ReadDir(modelDir)
		if err != nil || len(entries) == 0 {
			return fmt.Errorf("model not found in HuggingFace cache: %s", modelPath)
		}

		tokenizerPath = filepath.Join(modelDir, entries[0].Name(), "tokenizer.json")
	} else {
		// It's a full path
		if strings.HasPrefix(modelPath, "~/") {
			homeDir, _ := os.UserHomeDir()
			modelPath = filepath.Join(homeDir, modelPath[2:])
		}
		tokenizerPath = filepath.Join(modelPath, "tokenizer.json")
	}

	// Load tokenizer
	var err error
	tk, err = tokenizer.LoadFromFile(tokenizerPath)
	if err != nil {
		return fmt.Errorf("failed to load tokenizer from %s: %w", tokenizerPath, err)
	}

	return nil
}

func checkBackend() error {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(backendURL + "/health")
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("backend returned status %d", resp.StatusCode)
	}

	return nil
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	data, err := templateFS.ReadFile("templates/index.html")
	if err != nil {
		http.Error(w, "Template not found", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/html")
	w.Write(data)
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	health := HealthResponse{
		WebInterface: "ok",
		Backend:      "unavailable",
		Model:        modelName,
		Tokenizer:    fmt.Sprintf("Pure Go BPE (vocab: %d)", tk.VocabSize()),
	}

	if err := checkBackend(); err == nil {
		health.Backend = "ok"
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
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

	if req.Prompt == "" {
		http.Error(w, "No prompt provided", http.StatusBadRequest)
		return
	}

	if req.MaxTokens == 0 {
		req.MaxTokens = 50
	}

	// Tokenize prompt using pure Go tokenizer
	inputIDs := tk.Encode(req.Prompt, false)

	// Call backend
	backendReq := BackendRequest{
		InputIDs:     inputIDs,
		MaxNewTokens: req.MaxTokens,
		Temperature:  0.7,
		Stream:       false,
	}

	reqBody, _ := json.Marshal(backendReq)
	resp, err := http.Post(backendURL+"/generate", "application/json", strings.NewReader(string(reqBody)))
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]string{
			"error": "Backend server not available. Is serve_model_bytes running?",
		})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{
			"error": fmt.Sprintf("Backend error: %d", resp.StatusCode),
		})
		return
	}

	// Decode response
	var backendResp BackendResponse
	if err := json.NewDecoder(resp.Body).Decode(&backendResp); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Convert output IDs to uint32 and decode
	outputIDs := make([]uint32, len(backendResp.OutputIDs))
	for i, id := range backendResp.OutputIDs {
		outputIDs[i] = uint32(id)
	}

	generatedText := tk.Decode(outputIDs, true)

	// Return response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"generated_text": generatedText,
		"num_tokens":     len(backendResp.OutputIDs) - len(inputIDs),
	})
}

func handleGenerateStream(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.Prompt == "" {
		http.Error(w, "No prompt provided", http.StatusBadRequest)
		return
	}

	if req.MaxTokens == 0 {
		req.MaxTokens = 50
	}

	// Tokenize prompt using pure Go tokenizer
	inputIDs := tk.Encode(req.Prompt, false)

	// Call backend with streaming
	backendReq := BackendRequest{
		InputIDs:     inputIDs,
		MaxNewTokens: req.MaxTokens,
		Temperature:  0.7,
		Stream:       true,
	}

	reqBody, _ := json.Marshal(backendReq)
	resp, err := http.Post(backendURL+"/generate", "application/json", strings.NewReader(string(reqBody)))
	if err != nil {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\n", mustMarshal(StreamResponse{
			Error: "Backend server not available",
		}))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\n", mustMarshal(StreamResponse{
			Error: fmt.Sprintf("Backend error: %d", resp.StatusCode),
		}))
		return
	}

	// Set up SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Stream tokens
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		// Parse backend response
		dataJSON := line[6:] // Remove "data: " prefix
		var backendData map[string]interface{}
		if err := json.Unmarshal([]byte(dataJSON), &backendData); err != nil {
			continue
		}

		// Check if done
		if done, ok := backendData["done"].(bool); ok && done {
			fmt.Fprintf(w, "data: %s\n\n", mustMarshal(StreamResponse{Done: true}))
			flusher.Flush()
			break
		}

		// Decode token using pure Go tokenizer
		if tokenIDFloat, ok := backendData["token"].(float64); ok {
			tokenID := uint32(tokenIDFloat)
			tokenText := tk.Decode([]uint32{tokenID}, false)

			fmt.Fprintf(w, "data: %s\n\n", mustMarshal(StreamResponse{
				Token:   tokenText,
				TokenID: int(tokenID),
			}))
			flusher.Flush()
		}
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		fmt.Fprintf(w, "data: %s\n\n", mustMarshal(StreamResponse{
			Error: err.Error(),
		}))
		flusher.Flush()
	}
}

func mustMarshal(v interface{}) string {
	data, _ := json.Marshal(v)
	return string(data)
}
