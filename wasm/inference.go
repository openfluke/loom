//go:build js && wasm
// +build js,wasm

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"syscall/js"
	"unsafe"

	"github.com/openfluke/loom/nn"
	"github.com/openfluke/loom/tokenizer"
)

// Global state for loaded models
var (
	loadedTokenizer *tokenizer.Tokenizer
	loadedNetwork   *nn.Network
	embeddings      []float32
	finalNorm       *nn.LayerConfig
	vocabSize       int
	hiddenSize      int
	eosTokens       []int
)

// loadTokenizerFromBytes loads tokenizer from JSON bytes
func loadTokenizerFromBytes() js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return jsError("Missing tokenizer data argument")
		}

		// Get byte array from JavaScript
		tokenizerData := make([]byte, args[0].Get("length").Int())
		js.CopyBytesToGo(tokenizerData, args[0])

		// Load tokenizer
		tk, err := tokenizer.LoadFromBytes(tokenizerData)
		if err != nil {
			return jsError(fmt.Sprintf("Failed to load tokenizer: %v", err))
		}

		loadedTokenizer = tk
		return jsSuccess(map[string]interface{}{
			"vocab_size": tk.VocabSize(),
			"message":    "Tokenizer loaded successfully",
		})
	})
}

// loadTransformerFromBytes loads transformer model from config and weights bytes
func loadTransformerFromBytes() js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return jsError("Missing config or weights data")
		}

		// Get config data
		configData := make([]byte, args[0].Get("length").Int())
		js.CopyBytesToGo(configData, args[0])

		// Get weights data
		weightsData := make([]byte, args[1].Get("length").Int())
		js.CopyBytesToGo(weightsData, args[1])

		// Load transformer
		network, err := nn.LoadTransformerFromBytes(configData, weightsData)
		if err != nil {
			return jsError(fmt.Sprintf("Failed to load transformer: %v", err))
		}

		loadedNetwork = network

		// Parse config for metadata
		var config struct {
			VocabSize  int   `json:"vocab_size"`
			HiddenSize int   `json:"hidden_size"`
			NumLayers  int   `json:"num_hidden_layers"`
			EosTokenID int   `json:"eos_token_id"`
			EosTokens  []int `json:"eos_token_ids"`
		}
		if err := json.Unmarshal(configData, &config); err != nil {
			return jsError(fmt.Sprintf("Failed to parse config: %v", err))
		}

		vocabSize = config.VocabSize
		hiddenSize = config.HiddenSize

		// Store EOS tokens
		if len(config.EosTokens) > 0 {
			eosTokens = config.EosTokens
		} else if config.EosTokenID > 0 {
			eosTokens = []int{config.EosTokenID}
		}

		// Load embeddings and final norm from safetensors
		tensors, err := nn.LoadSafetensorsFromBytes(weightsData)
		if err != nil {
			return jsError(fmt.Sprintf("Failed to load safetensors: %v", err))
		}

		// Extract embeddings
		if embTensor, ok := tensors["model.embed_tokens.weight"]; ok {
			embeddings = embTensor
		} else if embTensor, ok := tensors["transformer.wte.weight"]; ok {
			embeddings = embTensor
		} else {
			return jsError("Embedding tensor not found in model")
		}

		// Extract final norm
		if normWeights, ok := tensors["model.norm.weight"]; ok {
			finalNorm = &nn.LayerConfig{
				Gamma: normWeights,
			}
		}

		return jsSuccess(map[string]interface{}{
			"vocab_size":      vocabSize,
			"hidden_size":     hiddenSize,
			"num_layers":      config.NumLayers,
			"embeddings_size": len(embeddings),
			"message":         "Transformer loaded successfully",
		})
	})
}

// loadEmbeddings loads embedding matrix from bytes
func loadEmbeddings() js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return jsError("Missing embeddings data")
		}

		// Get embeddings as Float32Array
		embeddingsJS := args[0]
		length := embeddingsJS.Get("length").Int()
		embeddings = make([]float32, length)

		// Copy from JS to Go
		js.CopyBytesToGo(
			(*(*[1 << 30]byte)(unsafe.Pointer(&embeddings[0])))[:length*4],
			embeddingsJS,
		)

		return jsSuccess(map[string]interface{}{
			"size":    length,
			"message": "Embeddings loaded successfully",
		})
	})
}

// encodeText tokenizes input text
func encodeText() js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if loadedTokenizer == nil {
			return jsError("Tokenizer not loaded")
		}

		if len(args) < 1 {
			return jsError("Missing text argument")
		}

		text := args[0].String()
		addSpecialTokens := true
		if len(args) > 1 {
			addSpecialTokens = args[1].Bool()
		}

		// Encode text
		tokenIDs := loadedTokenizer.Encode(text, addSpecialTokens)

		// Convert to JS array
		jsArray := js.Global().Get("Uint32Array").New(len(tokenIDs))
		for i, id := range tokenIDs {
			jsArray.SetIndex(i, id)
		}

		return jsArray
	})
}

// decodeTokens converts token IDs back to text
func decodeTokens() js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if loadedTokenizer == nil {
			return jsError("Tokenizer not loaded")
		}

		if len(args) < 1 {
			return jsError("Missing token IDs argument")
		}

		// Get token IDs from JS array
		idsJS := args[0]
		length := idsJS.Get("length").Int()
		ids := make([]uint32, length)
		for i := 0; i < length; i++ {
			ids[i] = uint32(idsJS.Index(i).Int())
		}

		skipSpecialTokens := true
		if len(args) > 1 {
			skipSpecialTokens = args[1].Bool()
		}

		// Decode tokens
		text := loadedTokenizer.Decode(ids, skipSpecialTokens)

		return text
	})
}

// generateNextToken runs inference and returns next token
func generateNextToken() js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if loadedNetwork == nil {
			return jsError("Model not loaded")
		}
		if loadedTokenizer == nil {
			return jsError("Tokenizer not loaded")
		}
		if len(embeddings) == 0 {
			return jsError("Embeddings not loaded")
		}

		if len(args) < 1 {
			return jsError("Missing token IDs argument")
		}

		// Get input token IDs
		idsJS := args[0]
		length := idsJS.Get("length").Int()
		inputIDs := make([]int, length)
		for i := 0; i < length; i++ {
			inputIDs[i] = idsJS.Index(i).Int()
		}

		// Get temperature (default 0.7)
		temperature := float32(0.7)
		if len(args) > 1 {
			temperature = float32(args[1].Float())
		}

		// Get last token embedding
		lastTokenID := inputIDs[len(inputIDs)-1]
		if lastTokenID >= vocabSize {
			return jsError(fmt.Sprintf("Token ID %d out of range (vocab size: %d)", lastTokenID, vocabSize))
		}

		// Extract embedding for last token
		input := make([]float32, hiddenSize)
		copy(input, embeddings[lastTokenID*hiddenSize:(lastTokenID+1)*hiddenSize])

		// Run forward pass
		output, _ := loadedNetwork.ForwardCPU(input)

		// Apply final norm if available
		if finalNorm != nil {
			for i := range output {
				output[i] = output[i] * finalNorm.Gamma[i]
			}
		}

		// Compute logits (dot product with embeddings)
		logits := make([]float32, vocabSize)
		for i := 0; i < vocabSize; i++ {
			logits[i] = 0
			for j := 0; j < hiddenSize; j++ {
				logits[i] += output[j] * embeddings[i*hiddenSize+j]
			}
		}

		// Apply temperature and sample
		nextToken := sampleToken(logits, temperature)

		return nextToken
	})
}

// generateText generates multiple tokens
func generateText() js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return jsError("Missing prompt or max_tokens argument")
		}

		prompt := args[0].String()
		maxTokens := args[1].Int()

		// Optional temperature
		temperature := float32(0.7)
		if len(args) > 2 {
			temperature = float32(args[2].Float())
		}

		// Encode prompt
		inputIDs := loadedTokenizer.Encode(prompt, false)
		if len(inputIDs) == 0 {
			return jsError("Empty input after tokenization")
		}

		// Convert to int slice
		tokens := make([]int, len(inputIDs))
		for i, id := range inputIDs {
			tokens[i] = int(id)
		}

		// Generate tokens
		for i := 0; i < maxTokens; i++ {
			// IMPORTANT: Process the FULL sequence, not just the last token
			// This is needed because the model has no KV cache and needs full context

			// Flatten all token embeddings into one sequence
			seqLen := len(tokens)
			input := make([]float32, seqLen*hiddenSize)
			for j, tokenID := range tokens {
				if tokenID >= vocabSize {
					break
				}
				copy(input[j*hiddenSize:(j+1)*hiddenSize], embeddings[tokenID*hiddenSize:(tokenID+1)*hiddenSize])
			}

			// Forward pass with full sequence
			output, _ := loadedNetwork.ForwardCPU(input)

			// Extract output for last position
			lastPosOutput := output[len(output)-hiddenSize:]

			// Apply final norm
			if finalNorm != nil {
				for j := range lastPosOutput {
					lastPosOutput[j] = lastPosOutput[j] * finalNorm.Gamma[j]
				}
			}

			// Compute logits
			logits := make([]float32, vocabSize)
			for j := 0; j < vocabSize; j++ {
				logits[j] = 0
				for k := 0; k < hiddenSize; k++ {
					logits[j] += lastPosOutput[k] * embeddings[j*hiddenSize+k]
				}
			} // Sample
			nextToken := sampleToken(logits, temperature)
			tokens = append(tokens, nextToken)

			// Check for EOS
			isEOS := false
			for _, eos := range eosTokens {
				if nextToken == eos {
					isEOS = true
					break
				}
			}
			if isEOS {
				break
			}
		}

		// Decode tokens
		outputIDs := make([]uint32, len(tokens))
		for i, id := range tokens {
			outputIDs[i] = uint32(id)
		}
		text := loadedTokenizer.Decode(outputIDs, true)

		return jsSuccess(map[string]interface{}{
			"generated_text": text,
			"num_tokens":     len(tokens) - len(inputIDs),
		})
	})
}

// sampleToken samples from logits with temperature
func sampleToken(logits []float32, temperature float32) int {
	// Find max for numerical stability
	maxLogit := logits[0]
	for _, l := range logits {
		if l > maxLogit {
			maxLogit = l
		}
	}

	// Apply temperature and softmax
	var sum float32
	probs := make([]float32, len(logits))
	for i, l := range logits {
		probs[i] = float32(math.Exp(float64((l - maxLogit) / temperature)))
		sum += probs[i]
	}

	// Normalize
	for i := range probs {
		probs[i] /= sum
	}

	// Sample (simple greedy for now)
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

// Helper functions
func jsSuccess(data map[string]interface{}) js.Value {
	data["success"] = true
	jsonData, _ := json.Marshal(data)
	return js.ValueOf(string(jsonData))
}

func jsError(message string) js.Value {
	data := map[string]interface{}{
		"success": false,
		"error":   message,
	}
	jsonData, _ := json.Marshal(data)
	return js.ValueOf(string(jsonData))
}
