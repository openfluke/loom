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

// Global state
var (
	network      *nn.Network
	embeddings   []float32
	finalNorm    []float32
	hiddenSize   int
	vocabSize    int
	eosTokens    []int
	hasFinalNorm bool
	tk           *tokenizer.Tokenizer
	ts           *nn.TweenState
)

// Single fact to teach
const TEST_PROMPT = "Sam's favorite neural network is called"
const TARGET_RESPONSE = " LOOM"

const (
	MAX_TOKENS     = 20
	STEPS_PER_ROUND = 5
	LEARNING_RATE  = 0.01
)

func main() {
	fmt.Println("🧠 StepTweenChain Observation Experiment")
	fmt.Println("=========================================")
	fmt.Println()
	fmt.Printf("📝 Goal: Teach model to complete:\n")
	fmt.Printf("   \"%s\" → \"%s\"\n\n", TEST_PROMPT, TARGET_RESPONSE)

	// Discover models in HuggingFace cache
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
			modelName := strings.TrimPrefix(entry.Name(), "models--")
			modelName = strings.Replace(modelName, "--", "/", 1)
			models = append(models, modelName)
		}
	}

	if len(models) == 0 {
		log.Fatalf("No models found in HuggingFace cache at: %s", hubDir)
	}

	// Display models and let user select
	fmt.Println("📦 Available models in cache:")
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

	// Load tokenizer
	tokenizerPath := filepath.Join(snapshotDir, "tokenizer.json")
	tk, err = tokenizer.LoadFromFile(tokenizerPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}
	fmt.Printf("✓ Tokenizer loaded (vocab: %d)\n", tk.VocabSize())

	// Load EOS tokens
	configPath := filepath.Join(snapshotDir, "config.json")
	eosTokens = loadEOSTokens(configPath)
	if len(eosTokens) == 0 {
		eosTokens = []int{2, 0}
	}

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

	fmt.Printf("✅ Model loaded! (Hidden: %d, Vocab: %d, Layers: %d)\n\n", hiddenSize, vocabSize, len(network.Layers))

	// Initialize TweenState
	ts = nn.NewTweenState(network, nil)
	ts.Config.UseChainRule = true
	ts.Config.DenseRate = 0.1
	ts.Config.AttentionRate = 0.05
	ts.Config.NormRate = 0.01
	ts.Config.Momentum = 0.9
	ts.Config.LinkBudgetScale = 0.5

	totalSteps := 0

	// Test BEFORE any training
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Printf("🔍 BEFORE TRAINING (Step %d):\n", totalSteps)
	fmt.Printf("   Prompt: \"%s\"\n", TEST_PROMPT)
	fmt.Print("   Output: \"")
	generate(TEST_PROMPT)
	fmt.Println("\"")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	// Automatic training loop - no ENTER needed
	maxSteps := 200
	showEvery := 10

	for totalSteps < maxSteps {
		// Train for STEPS_PER_ROUND steps
		fmt.Printf("\n🔧 Training steps %d-%d...\n", totalSteps+1, totalSteps+STEPS_PER_ROUND)
		for i := 0; i < STEPS_PER_ROUND; i++ {
			loss := trainOneStep()
			totalSteps++
			fmt.Printf("   Step %d: Loss = %.4f\n", totalSteps, loss)
		}

		// Show output every showEvery steps
		if totalSteps%showEvery == 0 {
			fmt.Println()
			fmt.Println("═══════════════════════════════════════════════════════════════")
			fmt.Printf("🔍 AFTER TRAINING (Step %d):\n", totalSteps)
			fmt.Printf("   Prompt: \"%s\"\n", TEST_PROMPT)
			fmt.Print("   Output: \"")
			generate(TEST_PROMPT)
			fmt.Println("\"")
			fmt.Printf("   Expected: \"%s\"\n", TARGET_RESPONSE)
			fmt.Println("═══════════════════════════════════════════════════════════════")
		}
	}

	fmt.Println("\n✅ Training complete!")
}

func trainOneStep() float32 {
	// Tokenize prompt and target
	promptTokens := tk.Encode(TEST_PROMPT, false)
	targetTokens := tk.Encode(TARGET_RESPONSE, false)

	if len(targetTokens) == 0 {
		return 0
	}

	// Create input embedding
	input := embedTokens(promptTokens)

	// Forward pass through tween state
	output := ts.ForwardPass(network, input)

	if len(output) == 0 {
		return 0
	}

	// Create target: one-hot encoding for first expected token
	target := make([]float32, len(output))
	targetToken := int(targetTokens[0])
	if targetToken < len(target) {
		target[targetToken] = 1.0
	}

	// Compute output gradient (target - output)
	outputGrad := make([]float32, len(output))
	for i := range outputGrad {
		if i < len(target) {
			outputGrad[i] = target[i] - output[i]
		}
	}

	// Set gradients for chain rule
	totalLayers := network.TotalLayers()
	ts.ChainGradients[totalLayers] = outputGrad
	ts.BackwardTargets[totalLayers] = target

	// Update weights using TweenChain
	ts.TweenWeightsChainRule(network, LEARNING_RATE)

	// Calculate loss
	var loss float32
	for i := range output {
		if i < len(target) {
			diff := output[i] - target[i]
			loss += diff * diff
		}
	}

	return loss
}

func embedTokens(tokenIDs []uint32) []float32 {
	input := make([]float32, len(tokenIDs)*hiddenSize)
	for t, tokenID := range tokenIDs {
		if int(tokenID) >= vocabSize || int(tokenID) < 0 {
			continue
		}
		for d := 0; d < hiddenSize; d++ {
			input[t*hiddenSize+d] = embeddings[int(tokenID)*hiddenSize+d]
		}
	}
	return input
}

func generate(prompt string) string {
	inputIDs := tk.Encode(prompt, false)
	tokens := make([]int, len(inputIDs))
	for i, id := range inputIDs {
		tokens[i] = int(id)
	}

	var generated strings.Builder

	for i := 0; i < MAX_TOKENS; i++ {
		nextToken, err := generateNextToken(tokens)
		if err != nil {
			return generated.String()
		}

		tokens = append(tokens, nextToken)

		tokenText := tk.Decode([]uint32{uint32(nextToken)}, false)
		fmt.Print(tokenText)
		generated.WriteString(tokenText)

		if isEOSToken(nextToken) {
			break
		}
	}

	return generated.String()
}

func generateNextToken(tokens []int) (int, error) {
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
	output, _ := network.ForwardCPU(input)

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

	lastIdx := (len(tokens) - 1) * hiddenSize
	lastTokenNormalized := normalized[lastIdx : lastIdx+hiddenSize]

	logits := make([]float32, vocabSize)
	for v := 0; v < vocabSize; v++ {
		var sum float32
		for d := 0; d < hiddenSize; d++ {
			sum += lastTokenNormalized[d] * embeddings[v*hiddenSize+d]
		}
		logits[v] = sum
	}

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
