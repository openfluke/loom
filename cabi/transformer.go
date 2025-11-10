package main

/*
#include <stdlib.h>
#include <stdbool.h>

// Ensure bool/true/false are available
#ifndef __cplusplus
#ifndef bool
#define bool _Bool
#define true 1
#define false 0
#endif
#endif
*/
import "C"

import (
	"encoding/json"
	"unsafe"

	"github.com/openfluke/loom/nn"
	"github.com/openfluke/loom/tokenizer"
)

// LayerNormWeights holds final normalization weights
type LayerNormWeights struct {
	Gamma []float32
	Beta  []float32
}

// Global transformer state
var (
	loadedTokenizer   *tokenizer.Tokenizer
	loadedTransformer *nn.Network
	transformerEmbeds []float32
	transformerNorm   *LayerNormWeights
	transformerHidden int
	transformerVocab  int
	transformerEOS    []int
)

//export LoadTokenizerFromBytes
func LoadTokenizerFromBytes(dataPtr *C.char, dataLen C.int) *C.char {
	// Convert C data to Go bytes
	data := C.GoBytes(unsafe.Pointer(dataPtr), dataLen)

	// Load tokenizer
	tok, err := tokenizer.LoadFromBytes(data)
	if err != nil {
		return errJSON("failed to load tokenizer: " + err.Error())
	}

	loadedTokenizer = tok

	return asJSON(map[string]interface{}{
		"success":    true,
		"vocab_size": tok.VocabSize(),
		"message":    "Tokenizer loaded successfully",
	})
}

//export LoadTransformerFromBytes
func LoadTransformerFromBytes(configPtr *C.char, configLen C.int, weightsPtr *C.char, weightsLen C.int) *C.char {
	// Convert C data to Go bytes
	configData := C.GoBytes(unsafe.Pointer(configPtr), configLen)
	weightsData := C.GoBytes(unsafe.Pointer(weightsPtr), weightsLen)

	// Parse config for metadata
	var config struct {
		HiddenSize int         `json:"hidden_size"`
		VocabSize  int         `json:"vocab_size"`
		NumLayers  int         `json:"num_hidden_layers"`
		EOSTokenID interface{} `json:"eos_token_id"`
		PadTokenID int         `json:"pad_token_id"`
	}
	if err := json.Unmarshal(configData, &config); err != nil {
		return errJSON("failed to parse config: " + err.Error())
	}

	transformerHidden = config.HiddenSize
	transformerVocab = config.VocabSize

	// Parse EOS tokens
	transformerEOS = parseEOSTokens(config.EOSTokenID, config.PadTokenID)

	// Load transformer network
	network, err := nn.LoadTransformerFromBytes(configData, weightsData)
	if err != nil {
		return errJSON("failed to load transformer: " + err.Error())
	}
	loadedTransformer = network

	// Load embeddings and final norm from safetensors
	tensors, err := nn.LoadSafetensorsFromBytes(weightsData)
	if err != nil {
		return errJSON("failed to parse tensors: " + err.Error())
	}

	// Try different embedding key names
	transformerEmbeds = tryLoadTensor(tensors, []string{
		"model.embed_tokens.weight",
		"transformer.wte.weight",
		"embeddings.weight",
		"tok_embeddings.weight",
		"lm_head.weight",
	})
	if len(transformerEmbeds) == 0 {
		return errJSON("failed to load embeddings tensor")
	}

	// Try to load final norm
	normWeight := tryLoadTensor(tensors, []string{
		"model.norm.weight",
		"transformer.ln_f.weight",
		"norm.weight",
		"ln_f.weight",
	})
	if len(normWeight) > 0 {
		transformerNorm = &LayerNormWeights{
			Gamma: normWeight,
			Beta:  make([]float32, len(normWeight)), // Usually zero
		}
	}

	return asJSON(map[string]interface{}{
		"success":     true,
		"vocab_size":  transformerVocab,
		"hidden_size": transformerHidden,
		"num_layers":  config.NumLayers,
		"message":     "Transformer loaded successfully",
	})
}

//export EncodeText
func EncodeText(textPtr *C.char, addSpecialTokens C.bool) *C.char {
	if loadedTokenizer == nil {
		return errJSON("tokenizer not loaded")
	}

	text := C.GoString(textPtr)
	addSpecial := bool(addSpecialTokens)

	tokenIDs := loadedTokenizer.Encode(text, addSpecial)

	return asJSON(map[string]interface{}{
		"success": true,
		"ids":     tokenIDs,
	})
}

//export DecodeTokens
func DecodeTokens(idsJSON *C.char, skipSpecialTokens C.bool) *C.char {
	if loadedTokenizer == nil {
		return errJSON("tokenizer not loaded")
	}

	// Parse JSON array of token IDs
	var ids []uint32
	if err := json.Unmarshal([]byte(C.GoString(idsJSON)), &ids); err != nil {
		return errJSON("failed to parse token IDs: " + err.Error())
	}

	skipSpecial := bool(skipSpecialTokens)
	text := loadedTokenizer.Decode(ids, skipSpecial)

	return asJSON(map[string]interface{}{
		"success": true,
		"text":    text,
	})
}

//export GenerateText
func GenerateText(promptPtr *C.char, maxTokens C.int, temperature C.float) *C.char {
	if loadedTokenizer == nil {
		return errJSON("tokenizer not loaded")
	}
	if loadedTransformer == nil {
		return errJSON("transformer not loaded")
	}

	prompt := C.GoString(promptPtr)
	maxTok := int(maxTokens)
	temp := float32(temperature)

	// Encode prompt
	inputIDs := loadedTokenizer.Encode(prompt, false)
	if len(inputIDs) == 0 {
		return errJSON("empty input after tokenization")
	}

	// Convert to int slice
	tokens := make([]int, len(inputIDs))
	for i, id := range inputIDs {
		tokens[i] = int(id)
	}

	// Generate tokens
	for i := 0; i < maxTok; i++ {
		// Process full sequence
		seqLen := len(tokens)
		input := make([]float32, seqLen*transformerHidden)
		for j, tokenID := range tokens {
			if tokenID >= transformerVocab {
				break
			}
			copy(input[j*transformerHidden:(j+1)*transformerHidden],
				transformerEmbeds[tokenID*transformerHidden:(tokenID+1)*transformerHidden])
		}

		// Forward pass
		output, _ := loadedTransformer.ForwardCPU(input)

		// Extract last position output
		lastPosOutput := output[len(output)-transformerHidden:]

		// Apply final norm
		if transformerNorm != nil {
			for j := range lastPosOutput {
				lastPosOutput[j] = lastPosOutput[j] * transformerNorm.Gamma[j]
			}
		}

		// Compute logits
		logits := make([]float32, transformerVocab)
		for j := 0; j < transformerVocab; j++ {
			logits[j] = 0
			for k := 0; k < transformerHidden; k++ {
				logits[j] += lastPosOutput[k] * transformerEmbeds[j*transformerHidden+k]
			}
		}

		// Sample
		nextToken := sampleToken(logits, temp)
		tokens = append(tokens, nextToken)

		// Check for EOS
		isEOS := false
		for _, eos := range transformerEOS {
			if nextToken == eos {
				isEOS = true
				break
			}
		}
		if isEOS {
			break
		}
	}

	// Convert back to uint32
	outputIDs := make([]uint32, len(tokens))
	for i, id := range tokens {
		outputIDs[i] = uint32(id)
	}

	// Decode
	generatedText := loadedTokenizer.Decode(outputIDs, true)

	return asJSON(map[string]interface{}{
		"success":        true,
		"generated_text": generatedText,
		"num_tokens":     len(tokens) - len(inputIDs),
		"total_tokens":   len(tokens),
	})
}

//export GenerateNextToken
func GenerateNextToken(idsJSON *C.char, temperature C.float) *C.char {
	if loadedTransformer == nil {
		return errJSON("transformer not loaded")
	}

	// Parse token IDs
	var tokenIDs []int
	if err := json.Unmarshal([]byte(C.GoString(idsJSON)), &tokenIDs); err != nil {
		return errJSON("failed to parse token IDs: " + err.Error())
	}

	if len(tokenIDs) == 0 {
		return errJSON("empty token sequence")
	}

	temp := float32(temperature)

	// Process full sequence
	seqLen := len(tokenIDs)
	input := make([]float32, seqLen*transformerHidden)
	for j, tokenID := range tokenIDs {
		if tokenID >= transformerVocab {
			return errJSON("token ID out of range")
		}
		copy(input[j*transformerHidden:(j+1)*transformerHidden],
			transformerEmbeds[tokenID*transformerHidden:(tokenID+1)*transformerHidden])
	}

	// Forward pass
	output, _ := loadedTransformer.ForwardCPU(input)

	// Extract last position output
	lastPosOutput := output[len(output)-transformerHidden:]

	// Apply final norm
	if transformerNorm != nil {
		for j := range lastPosOutput {
			lastPosOutput[j] = lastPosOutput[j] * transformerNorm.Gamma[j]
		}
	}

	// Compute logits
	logits := make([]float32, transformerVocab)
	for j := 0; j < transformerVocab; j++ {
		logits[j] = 0
		for k := 0; k < transformerHidden; k++ {
			logits[j] += lastPosOutput[k] * transformerEmbeds[j*transformerHidden+k]
		}
	}

	// Sample
	nextToken := sampleToken(logits, temp)

	// Check if this is an EOS token
	isEOS := false
	for _, eos := range transformerEOS {
		if nextToken == eos {
			isEOS = true
			break
		}
	}

	return asJSON(map[string]interface{}{
		"success": true,
		"token":   nextToken,
		"is_eos":  isEOS,
	})
}

// Helper functions

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

func tryLoadTensor(tensors map[string][]float32, keys []string) []float32 {
	for _, key := range keys {
		if data, ok := tensors[key]; ok {
			return data
		}
	}
	return nil
}

func sampleToken(logits []float32, temperature float32) int {
	if temperature <= 0 {
		temperature = 1e-10
	}

	// Find max for numerical stability
	maxLogit := logits[0]
	for _, l := range logits[1:] {
		if l > maxLogit {
			maxLogit = l
		}
	}

	// Apply temperature and compute probabilities (greedy sampling)
	maxProb := float32(0)
	maxIdx := 0
	for i, l := range logits {
		prob := float32(1.0) / (1.0 + float32(1.0)/float32(1.0+l-maxLogit))
		if prob > maxProb {
			maxProb = prob
			maxIdx = i
		}
	}

	return maxIdx
}
