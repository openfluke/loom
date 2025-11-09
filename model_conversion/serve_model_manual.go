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
	embeddings []float32
	finalNorm  []float32
	layers     []*TransformerBlock
	hiddenSize int
	vocabSize  int
	numLayers  int
)

type TransformerBlock struct {
	InputNorm        *nn.LayerConfig
	Attention        *nn.LayerConfig
	PostAttnNorm     *nn.LayerConfig
	GateWeights      []float32
	UpWeights        []float32
	DownWeights      []float32
	IntermediateSize int
}

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

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":      "ok",
		"hidden_size": hiddenSize,
		"vocab_size":  vocabSize,
		"layers":      numLayers * 4,
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
			nextToken, err := generateNextTokenManual(tokens)
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

			// Check for EOS
			if nextToken == 151643 || nextToken == 2 || nextToken == 0 {
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

		nextToken, err := generateNextTokenManual(tokens)
		if err != nil {
			json.NewEncoder(w).Encode(GenerateResponse{Error: err.Error()})
			return
		}

		tokens = append(tokens, nextToken)

		// Check for EOS
		if nextToken == 151643 || nextToken == 2 || nextToken == 0 {
			log.Printf("   EOS token detected, stopping")
			break
		}
	}

	// Return only new tokens
	outputIDs := tokens[len(req.InputIDs):]
	log.Printf("✅ Generated %d tokens total", len(outputIDs))
	json.NewEncoder(w).Encode(GenerateResponse{OutputIDs: outputIDs})
}

// generateNextTokenManual uses the same layer-by-layer approach as the trace
func generateNextTokenManual(tokens []int) (int, error) {
	log.Printf("      Input tokens: %v", tokens)

	// Step 1: Embed tokens
	seqLen := len(tokens)
	hidden := make([]float32, seqLen*hiddenSize)

	for t, tokenID := range tokens {
		if tokenID >= vocabSize || tokenID < 0 {
			return 0, fmt.Errorf("invalid token ID: %d (vocab size: %d)", tokenID, vocabSize)
		}
		for d := 0; d < hiddenSize; d++ {
			hidden[t*hiddenSize+d] = embeddings[tokenID*hiddenSize+d]
		}
	}

	if len(tokens) > 0 {
		log.Printf("      First token (%d) embedding (first 10): %.6v", tokens[0], hidden[:10])
	}

	// Step 2: Process through all transformer blocks
	for blockIdx := 0; blockIdx < numLayers; blockIdx++ {
		block := layers[blockIdx]

		// Sublayer 1: Input LayerNorm
		normed := applyRMSNorm(hidden, block.InputNorm.Gamma, block.InputNorm.Epsilon, hiddenSize)

		// Sublayer 2: Self Attention
		attnOutput, _ := nn.MultiHeadAttentionForwardCPU(normed, block.Attention, 1)

		// Add residual 1
		for i := range hidden {
			hidden[i] += attnOutput[i]
		}

		// Sublayer 3: Post-attention LayerNorm
		normed = applyRMSNorm(hidden, block.PostAttnNorm.Gamma, block.PostAttnNorm.Epsilon, hiddenSize)

		// Sublayer 4: SwiGLU MLP
		mlpOutput := applySwiGLU(normed, block.GateWeights, block.UpWeights, block.DownWeights,
			hiddenSize, block.IntermediateSize)

		// Add residual 2
		for i := range hidden {
			hidden[i] += mlpOutput[i]
		}
	}

	// Step 3: Final RMSNorm
	finalNormed := applyRMSNorm(hidden, finalNorm, 1e-6, hiddenSize)

	// Step 4: LM head projection (using tied weights)
	lastIdx := (seqLen - 1) * hiddenSize
	lastToken := finalNormed[lastIdx : lastIdx+hiddenSize]

	logits := make([]float32, vocabSize)
	for v := 0; v < vocabSize; v++ {
		var sum float32
		for d := 0; d < hiddenSize; d++ {
			sum += lastToken[d] * embeddings[v*hiddenSize+d]
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

// applyRMSNorm applies RMS normalization (same as trace)
func applyRMSNorm(input []float32, gamma []float32, epsilon float32, dModel int) []float32 {
	seqLen := len(input) / dModel
	output := make([]float32, len(input))

	for t := 0; t < seqLen; t++ {
		offset := t * dModel

		// Compute RMS (root mean square)
		sumSquares := float32(0)
		for i := 0; i < dModel; i++ {
			val := input[offset+i]
			sumSquares += val * val
		}
		rms := float32(math.Sqrt(float64(sumSquares/float32(dModel) + epsilon)))

		// Normalize and apply scale (gamma)
		for i := 0; i < dModel; i++ {
			normalized := input[offset+i] / rms
			gammaIdx := i
			if gammaIdx < len(gamma) {
				output[offset+i] = normalized * gamma[gammaIdx]
			} else {
				output[offset+i] = normalized
			}
		}
	}

	return output
}

// applySwiGLU applies the SwiGLU activation (same as trace)
func applySwiGLU(input []float32, gateWeights, upWeights, downWeights []float32, dModel, intermediateSize int) []float32 {
	seqLen := len(input) / dModel
	output := make([]float32, len(input))

	for t := 0; t < seqLen; t++ {
		inOffset := t * dModel

		// Gate projection with SiLU activation
		gate := make([]float32, intermediateSize)
		for i := 0; i < intermediateSize; i++ {
			sum := float32(0)
			for j := 0; j < dModel; j++ {
				sum += input[inOffset+j] * gateWeights[j*intermediateSize+i]
			}
			// SiLU: x * sigmoid(x)
			sigmoid := 1.0 / (1.0 + float32(math.Exp(float64(-sum))))
			gate[i] = sum * sigmoid
		}

		// Up projection (linear)
		up := make([]float32, intermediateSize)
		for i := 0; i < intermediateSize; i++ {
			sum := float32(0)
			for j := 0; j < dModel; j++ {
				sum += input[inOffset+j] * upWeights[j*intermediateSize+i]
			}
			up[i] = sum
		}

		// Element-wise multiply
		gated := make([]float32, intermediateSize)
		for i := 0; i < intermediateSize; i++ {
			gated[i] = gate[i] * up[i]
		}

		// Down projection
		outOffset := t * dModel
		for i := 0; i < dModel; i++ {
			sum := float32(0)
			for j := 0; j < intermediateSize; j++ {
				sum += gated[j] * downWeights[j*dModel+i]
			}
			output[outOffset+i] = sum
		}
	}

	return output
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

	// Load config
	configPath := filepath.Join(snapshotDir, "config.json")
	configFile, err := os.ReadFile(configPath)
	if err != nil {
		log.Fatalf("Error loading config: %v", err)
	}

	var config struct {
		HiddenSize       int     `json:"hidden_size"`
		NumLayers        int     `json:"num_hidden_layers"`
		NumHeads         int     `json:"num_attention_heads"`
		NumKVHeads       int     `json:"num_key_value_heads"`
		IntermediateSize int     `json:"intermediate_size"`
		RMSNormEps       float64 `json:"rms_norm_eps"`
		VocabSize        int     `json:"vocab_size"`
	}

	if err := json.Unmarshal(configFile, &config); err != nil {
		log.Fatalf("Error parsing config: %v", err)
	}

	log.Printf("Loading transformer model:")
	log.Printf("  Hidden size: %d", config.HiddenSize)
	log.Printf("  Layers: %d", config.NumLayers)
	log.Printf("  Attention heads: %d (Q), %d (KV)", config.NumHeads, config.NumKVHeads)
	log.Printf("  Intermediate size: %d", config.IntermediateSize)

	hiddenSize = config.HiddenSize
	numLayers = config.NumLayers
	vocabSize = config.VocabSize

	// Load tensors
	weightsPath := filepath.Join(snapshotDir, "model.safetensors")
	tensors, err := nn.LoadSafetensors(weightsPath)
	if err != nil {
		log.Fatalf("Error loading weights: %v", err)
	}

	log.Printf("Loaded %d tensors", len(tensors))

	// Load embeddings and final norm
	embeddings = tensors["model.embed_tokens.weight"]
	finalNorm = tensors["model.norm.weight"]

	// Load transformer blocks
	log.Printf("Loading %d transformer blocks...", config.NumLayers)
	layers = make([]*TransformerBlock, config.NumLayers)

	for i := 0; i < config.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d", i)

		// Extract Q, K, V weights and biases
		qWeights := tensors[prefix+".self_attn.q_proj.weight"]
		kWeights := tensors[prefix+".self_attn.k_proj.weight"]
		vWeights := tensors[prefix+".self_attn.v_proj.weight"]
		qBias := tensors[prefix+".self_attn.q_proj.bias"]
		kBias := tensors[prefix+".self_attn.k_proj.bias"]
		vBias := tensors[prefix+".self_attn.v_proj.bias"]
		outWeight := tensors[prefix+".self_attn.o_proj.weight"]

		// Transpose weights from PyTorch [out, in] to LOOM [in, out]
		qWeightsT := transposeWeights(qWeights, config.HiddenSize, config.HiddenSize)
		kWeightsT := transposeWeights(kWeights, config.NumKVHeads*config.HiddenSize/config.NumHeads, config.HiddenSize)
		vWeightsT := transposeWeights(vWeights, config.NumKVHeads*config.HiddenSize/config.NumHeads, config.HiddenSize)
		outWeightT := transposeWeights(outWeight, config.HiddenSize, config.HiddenSize)

		layers[i] = &TransformerBlock{
			InputNorm: &nn.LayerConfig{
				Type:     nn.LayerRMSNorm,
				NormSize: config.HiddenSize,
				Gamma:    tensors[prefix+".input_layernorm.weight"],
				Epsilon:  float32(config.RMSNormEps),
			},
			Attention: &nn.LayerConfig{
				Type:         nn.LayerMultiHeadAttention,
				DModel:       config.HiddenSize,
				NumHeads:     config.NumHeads,
				NumKVHeads:   config.NumKVHeads,
				HeadDim:      config.HiddenSize / config.NumHeads,
				QWeights:     qWeightsT,
				KWeights:     kWeightsT,
				VWeights:     vWeightsT,
				QBias:        qBias,
				KBias:        kBias,
				VBias:        vBias,
				OutputWeight: outWeightT,
				OutputBias:   make([]float32, config.HiddenSize),
			},
			PostAttnNorm: &nn.LayerConfig{
				Type:     nn.LayerRMSNorm,
				NormSize: config.HiddenSize,
				Gamma:    tensors[prefix+".post_attention_layernorm.weight"],
				Epsilon:  float32(config.RMSNormEps),
			},
			GateWeights:      transposeWeights(tensors[prefix+".mlp.gate_proj.weight"], config.IntermediateSize, config.HiddenSize),
			UpWeights:        transposeWeights(tensors[prefix+".mlp.up_proj.weight"], config.IntermediateSize, config.HiddenSize),
			DownWeights:      transposeWeights(tensors[prefix+".mlp.down_proj.weight"], config.HiddenSize, config.IntermediateSize),
			IntermediateSize: config.IntermediateSize,
		}

		if (i+1)%6 == 0 || i == config.NumLayers-1 {
			log.Printf("  Loaded %d/%d blocks", i+1, config.NumLayers)
		}
	}

	log.Printf("✅ Model loaded!")
	log.Printf("   Hidden: %d, Vocab: %d, Layers: %d\n", hiddenSize, vocabSize, numLayers)

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

// transposeWeights transposes a weight matrix from [out, in] to [in, out]
func transposeWeights(weights []float32, rows, cols int) []float32 {
	transposed := make([]float32, len(weights))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			transposed[j*rows+i] = weights[i*cols+j]
		}
	}
	return transposed
}
