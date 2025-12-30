package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/openfluke/loom/nn"
	"github.com/openfluke/loom/tokenizer"
)

var (
	network      *nn.Network
	embeddings   []float32
	finalNorm    []float32
	hiddenSize   int
	vocabSize    int
	eosTokens    []int
	hasFinalNorm bool
	templateType string
	tk           *tokenizer.Tokenizer
)

const MAX_TOKENS = 50

func main() {
	// Discover all models in HuggingFace cache
	homeDir, _ := os.UserHomeDir()
	hubDir := filepath.Join(homeDir, ".cache", "huggingface", "hub")

	entries, err := os.ReadDir(hubDir)
	if err != nil {
		log.Fatalf("Could not read HuggingFace cache: %v", err)
	}

	// Find all model directories
	var models []string
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), "models--") {
			// Convert "models--Qwen--Qwen2.5-0.5B" to "Qwen/Qwen2.5-0.5B"
			modelName := strings.TrimPrefix(entry.Name(), "models--")
			modelName = strings.Replace(modelName, "--", "/", 1)
			models = append(models, modelName)
		}
	}

	if len(models) == 0 {
		log.Fatalf("No models found in HuggingFace cache at: %s", hubDir)
	}

	// Display models and let user select
	fmt.Println("🤖 Available models in cache:")
	for i, model := range models {
		fmt.Printf("  [%d] %s\n", i+1, model)
	}
	fmt.Print("\nSelect model number: ")

	reader := bufio.NewReader(os.Stdin)
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)

	var selectedIdx int
	if _, err := fmt.Sscanf(input, "%d", &selectedIdx); err != nil || selectedIdx < 1 || selectedIdx > len(models) {
		log.Fatalf("Invalid selection")
	}

	modelName := models[selectedIdx-1]
	fmt.Printf("\n📦 Loading model: %s\n", modelName)

	// Get model directory
	modelDir := filepath.Join(hubDir, "models--"+strings.ReplaceAll(modelName, "/", "--"), "snapshots")
	entries, err = os.ReadDir(modelDir)
	if err != nil || len(entries) == 0 {
		log.Fatalf("Model snapshots not found for: %s", modelName)
	}

	snapshotDir := filepath.Join(modelDir, entries[0].Name())
	fmt.Printf("   Path: %s\n", snapshotDir)

	// Load tokenizer
	tokenizerPath := filepath.Join(snapshotDir, "tokenizer.json")
	if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
		fmt.Printf("⚠️  tokenizer.json not found in model directory.\n")

		// Fallback paths to check (using SmolLM2 which has a compatible BPE format)
		fallbackPaths := []string{
			"models/SmolLM2-135M-Instruct/tokenizer.json",                       // If running from loom root
			"../models/SmolLM2-135M-Instruct/tokenizer.json",                    // If running from model_conversion
			"/home/samuel/git/loom/models/SmolLM2-135M-Instruct/tokenizer.json", // Absolute path backup
		}

		foundFallback := false
		for _, fp := range fallbackPaths {
			if _, err := os.Stat(fp); err == nil {
				tokenizerPath = fp
				fmt.Printf("⚠️  Using fallback tokenizer from: %s\n", tokenizerPath)
				foundFallback = true
				break
			}
		}

		if !foundFallback {
			log.Printf("⚠️  Could not find fallback tokenizer in common locations.")
		}
	}

	tk, err = tokenizer.LoadFromFile(tokenizerPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}
	fmt.Printf("✓ Tokenizer loaded (vocab: %d)\n", tk.VocabSize())

	// Load EOS tokens from config
	configPath := filepath.Join(snapshotDir, "config.json")
	eosTokens = loadEOSTokens(configPath)
	if len(eosTokens) == 0 {
		eosTokens = []int{2, 0} // Default
	}
	fmt.Printf("✓ EOS tokens: %v\n", eosTokens)

	// Force raw mode - no chat template wrapping
	templateType = "raw"
	fmt.Println("ℹ️  Using raw mode: prompts passed directly to model")

	// Load model
	network, err = nn.LoadTransformerFromSafetensors(snapshotDir)
	if err != nil {
		log.Fatalf("Error loading model: %v", err)
	}

	// Load weights
	weightsPath := filepath.Join(snapshotDir, "model.safetensors")
	tensors, err := nn.LoadSafetensors(weightsPath)
	if err != nil {
		log.Fatalf("Error loading weights: %v", err)
	}

	// Load embeddings
	embeddings = tryLoadTensor(tensors, []string{
		"model.embed_tokens.weight",
		"transformer.wte.weight",
		"embeddings.weight",
		"embed_tokens.weight",
	})
	if embeddings == nil {
		log.Fatalf("Could not find embeddings")
	}

	// Load final norm
	finalNorm = tryLoadTensor(tensors, []string{
		"model.norm.weight",
		"transformer.ln_f.weight",
		"ln_f.weight",
		"norm.weight",
	})
	hasFinalNorm = (finalNorm != nil)

	hiddenSize = network.InputSize
	vocabSize = len(embeddings) / hiddenSize

	fmt.Printf("✅ Model loaded!\n")
	fmt.Printf("   Hidden: %d, Vocab: %d, Layers: %d\n\n", hiddenSize, vocabSize, len(network.Layers))

	// Interactive chat loop
	fmt.Println("🤖 Chat with the model (max 50 tokens per response)")
	fmt.Println("   Type 'quit' or 'exit' to stop\n")

	for {
		fmt.Print("You: ")
		prompt, _ := reader.ReadString('\n')
		prompt = strings.TrimSpace(prompt)

		if prompt == "quit" || prompt == "exit" {
			fmt.Println("Goodbye!")
			break
		}

		if prompt == "" {
			continue
		}

		// Generate response with streaming
		fmt.Print("Bot: ")
		generate(prompt)
		fmt.Println()
	}
}

func generate(prompt string) string {
	// Tokenize
	var fullPrompt string
	switch templateType {
	case "chatml":
		fullPrompt = fmt.Sprintf("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt)
	case "llama":
		fullPrompt = fmt.Sprintf("[INST] %s [/INST]", prompt)
	case "gemma":
		fullPrompt = fmt.Sprintf("<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", prompt)
	default: // raw
		fullPrompt = prompt
	}

	inputIDs := tk.Encode(fullPrompt, false)
	tokens := make([]int, len(inputIDs))
	for i, id := range inputIDs {
		tokens[i] = int(id)
	}

	// Generate up to MAX_TOKENS with streaming
	for i := 0; i < MAX_TOKENS; i++ {
		nextToken, err := generateNextToken(tokens)
		if err != nil {
			return fmt.Sprintf("\n[Error: %v]", err)
		}

		tokens = append(tokens, nextToken)

		// Decode and print this token immediately (streaming)
		tokenText := tk.Decode([]uint32{uint32(nextToken)}, false)
		fmt.Print(tokenText)

		// Check for EOS
		if isEOSToken(nextToken) {
			break
		}
	}

	fmt.Println() // New line after generation
	return ""     // Already printed, so return empty
}

func generateNextToken(tokens []int) (int, error) {
	// Embed tokens
	input := make([]float32, len(tokens)*hiddenSize)
	for t, tokenID := range tokens {
		if tokenID >= vocabSize || tokenID < 0 {
			return 0, fmt.Errorf("invalid token ID: %d", tokenID)
		}
		for d := 0; d < hiddenSize; d++ {
			input[t*hiddenSize+d] = embeddings[tokenID*hiddenSize+d]
		}
	}

	network.BatchSize = 1

	// Forward pass
	output, _ := network.ForwardCPU(input)

	// Apply final norm if available
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
		normalized = output
	}

	// Extract last token
	lastIdx := (len(tokens) - 1) * hiddenSize
	lastTokenNormalized := normalized[lastIdx : lastIdx+hiddenSize]

	// LM head projection (tied weights)
	logits := make([]float32, vocabSize)
	for v := 0; v < vocabSize; v++ {
		var sum float32
		for d := 0; d < hiddenSize; d++ {
			sum += lastTokenNormalized[d] * embeddings[v*hiddenSize+d]
		}
		logits[v] = sum
	}

	// Greedy sampling (argmax)
	maxIdx := 0
	maxVal := logits[0]
	for j := 1; j < vocabSize; j++ {
		if logits[j] > maxVal {
			maxVal = logits[j]
			maxIdx = j
		}
	}

	return maxIdx, nil
}

func isEOSToken(token int) bool {
	for _, eos := range eosTokens {
		if token == eos {
			return true
		}
	}
	return false
}

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

	if padID, ok := config["pad_token_id"]; ok {
		if idFloat, ok := padID.(float64); ok {
			tokens = append(tokens, int(idFloat))
		}
	}

	return tokens
}

func tryLoadTensor(tensors map[string][]float32, keys []string) []float32 {
	for _, key := range keys {
		if tensor, exists := tensors[key]; exists {
			return tensor
		}
	}
	return nil
}

func hasToken(tk *tokenizer.Tokenizer, token string) bool {
	if _, ok := tk.SpecialTokens[token]; ok {
		return true
	}
	if _, ok := tk.Vocab[token]; ok {
		return true
	}
	return false
}
