package nn

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// TransformerConfig represents configuration for Llama-based transformer models
// Supports: Llama, TinyLlama, Qwen2, Mistral, etc.
type TransformerConfig struct {
	ModelType        string   `json:"model_type"`    // "llama", "qwen2", "mistral", etc.
	Architectures    []string `json:"architectures"` // Model architecture names
	HiddenSize       int      `json:"hidden_size"`
	IntermediateSize int      `json:"intermediate_size"`
	NumLayers        int      `json:"num_hidden_layers"`
	NumHeads         int      `json:"num_attention_heads"`
	NumKVHeads       int      `json:"num_key_value_heads"`
	RMSNormEps       float64  `json:"rms_norm_eps"`
	VocabSize        int      `json:"vocab_size"`
	RoPETheta        float64  `json:"rope_theta"` // RoPE base frequency (default 10000.0)
}

// LoadTransformerFromSafetensors loads a Llama-based transformer model directly from safetensors
// Supports: Llama, TinyLlama, Qwen2.5, Mistral, and other models using the Llama architecture
func LoadTransformerFromSafetensors(modelDir string) (*Network, error) {
	// Read config
	configPath := filepath.Join(modelDir, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config: %w", err)
	}

	var config TransformerConfig
	if err := json.Unmarshal(configData, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	// Validate model architecture
	if err := validateArchitecture(config); err != nil {
		return nil, err
	}

	// Validate required fields to prevent panics
	if config.NumHeads == 0 {
		return nil, fmt.Errorf("unsupported model: num_attention_heads is 0 (model may use encoder-decoder architecture)")
	}
	if config.HiddenSize == 0 {
		return nil, fmt.Errorf("unsupported model: hidden_size is 0")
	}
	if config.NumLayers == 0 {
		return nil, fmt.Errorf("unsupported model: num_hidden_layers is 0")
	}

	fmt.Printf("Loading transformer model:\n")
	fmt.Printf("  Model type: %s\n", config.ModelType)
	if len(config.Architectures) > 0 {
		fmt.Printf("  Architecture: %s\n", config.Architectures[0])
	}
	fmt.Printf("  Hidden size: %d\n", config.HiddenSize)
	fmt.Printf("  Layers: %d\n", config.NumLayers)
	fmt.Printf("  Attention heads: %d (Q), %d (KV)\n", config.NumHeads, config.NumKVHeads)
	fmt.Printf("  Intermediate size: %d\n", config.IntermediateSize)

	// Load weights
	weightsPath := filepath.Join(modelDir, "model.safetensors")
	tensors, err := LoadSafetensors(weightsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load weights: %w", err)
	}

	fmt.Printf("Loaded %d tensors\n", len(tensors))

	// Build network (limit layers for numerical stability)
	// Full model may suffer from vanishing activations after many layers
	// For production, use 4-8 layers; for testing, 2 layers is fast
	maxLayers := config.NumLayers // Load ALL layers for proper model behavior
	// Previous versions used fewer layers (8, 16) but this causes quality issues

	network := &Network{
		GridRows:      1,
		GridCols:      1,
		LayersPerCell: maxLayers * 4, // 4 layers per transformer block: Attn, RMSNorm, SwiGLU, RMSNorm
		InputSize:     config.HiddenSize,
		BatchSize:     1,
		Layers:        make([]LayerConfig, 0),
	}

	// For each transformer layer
	fmt.Printf("Loading %d transformer blocks (%d total layers) out of %d available...\n",
		maxLayers, maxLayers*4, config.NumLayers)

	for i := 0; i < maxLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d", i)

		// 1. Pre-attention RMSNorm (input_layernorm)
		inputNorm := LayerConfig{
			Type:     LayerRMSNorm,
			NormSize: config.HiddenSize,
			Gamma:    getTensor(tensors, prefix+".input_layernorm.weight"),
			Epsilon:  float32(config.RMSNormEps),
		}
		network.Layers = append(network.Layers, inputNorm)

		// 2. Multi-head attention (GQA)
		qWeights, kWeights, vWeights, qBias, kBias, vBias := extractQKVWeights(tensors, prefix, config)
		outWeight := getTensor(tensors, prefix+".self_attn.o_proj.weight")
		// Transpose output weight from PyTorch [out, in] to LOOM [in, out]
		outWeightTransposed := transposeWeights(outWeight, config.HiddenSize, config.HiddenSize)
		outBias := make([]float32, config.HiddenSize) // No bias in Llama

		attnLayer := LayerConfig{
			Type:         LayerMultiHeadAttention,
			DModel:       config.HiddenSize,
			NumHeads:     config.NumHeads,
			NumKVHeads:   config.NumKVHeads,
			HeadDim:      config.HiddenSize / config.NumHeads,
			SeqLength:    128, // Default sequence length
			QWeights:     qWeights,
			KWeights:     kWeights,
			VWeights:     vWeights,
			QBias:        qBias,
			KBias:        kBias,
			VBias:        vBias,
			OutputWeight: outWeightTransposed,
			OutputBias:   outBias,
			RoPEFreqBase: float32(config.RoPETheta),
		}
		if attnLayer.RoPEFreqBase == 0 {
			attnLayer.RoPEFreqBase = 10000.0 // Default
		}
		network.Layers = append(network.Layers, attnLayer)

		// 3. Pre-MLP RMSNorm (post_attention_layernorm)
		postAttnNorm := LayerConfig{
			Type:     LayerRMSNorm,
			NormSize: config.HiddenSize,
			Gamma:    getTensor(tensors, prefix+".post_attention_layernorm.weight"),
			Epsilon:  float32(config.RMSNormEps),
		}
		network.Layers = append(network.Layers, postAttnNorm)

		// 4. SwiGLU layer (replaces simple MLP)
		// SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
		gateWeights := getTensor(tensors, prefix+".mlp.gate_proj.weight")
		upWeights := getTensor(tensors, prefix+".mlp.up_proj.weight")
		downWeights := getTensor(tensors, prefix+".mlp.down_proj.weight")

		// Transpose weights from PyTorch [out, in] to LOOM [in, out]
		gateTransposed := transposeWeights(gateWeights, config.IntermediateSize, config.HiddenSize)
		upTransposed := transposeWeights(upWeights, config.IntermediateSize, config.HiddenSize)
		downTransposed := transposeWeights(downWeights, config.HiddenSize, config.IntermediateSize)

		swiGLULayer := LayerConfig{
			Type:         LayerSwiGLU,
			InputHeight:  config.HiddenSize,
			OutputHeight: config.IntermediateSize,
			GateWeights:  gateTransposed,
			UpWeights:    upTransposed,
			DownWeights:  downTransposed,
			GateBias:     make([]float32, config.IntermediateSize),
			UpBias:       make([]float32, config.IntermediateSize),
			DownBias:     make([]float32, config.HiddenSize),
		}
		network.Layers = append(network.Layers, swiGLULayer)

		fmt.Printf("Added layer %d/%d\n", i+1, config.NumLayers)
	}

	fmt.Printf("✅ Loaded transformer model with %d layers\n", len(network.Layers))

	// Initialize activation storage
	network.activations = make([][]float32, len(network.Layers)+1)
	network.preActivations = make([][]float32, len(network.Layers))

	return network, nil
}

// LoadTransformerFromBytes loads a Llama-based transformer model from byte slices
// Supports: Llama, TinyLlama, Qwen2.5, Mistral, and other models using the Llama architecture
// configData: JSON config file contents
// weightsData: safetensors file contents
func LoadTransformerFromBytes(configData []byte, weightsData []byte) (*Network, error) {
	// Parse config
	var config TransformerConfig
	if err := json.Unmarshal(configData, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	// Validate model architecture
	if err := validateArchitecture(config); err != nil {
		return nil, err
	}

	// Validate required fields to prevent panics
	if config.NumHeads == 0 {
		return nil, fmt.Errorf("unsupported model: num_attention_heads is 0 (model may use encoder-decoder architecture)")
	}
	if config.HiddenSize == 0 {
		return nil, fmt.Errorf("unsupported model: hidden_size is 0")
	}
	if config.NumLayers == 0 {
		return nil, fmt.Errorf("unsupported model: num_hidden_layers is 0")
	}

	fmt.Printf("Loading transformer model from bytes:\n")
	fmt.Printf("  Model type: %s\n", config.ModelType)
	if len(config.Architectures) > 0 {
		fmt.Printf("  Architecture: %s\n", config.Architectures[0])
	}
	fmt.Printf("  Hidden size: %d\n", config.HiddenSize)
	fmt.Printf("  Layers: %d\n", config.NumLayers)
	fmt.Printf("  Attention heads: %d (Q), %d (KV)\n", config.NumHeads, config.NumKVHeads)
	fmt.Printf("  Intermediate size: %d\n", config.IntermediateSize)

	// Load weights from bytes
	tensors, err := LoadSafetensorsFromBytes(weightsData)
	if err != nil {
		return nil, fmt.Errorf("failed to load weights: %w", err)
	}

	fmt.Printf("Loaded %d tensors\n", len(tensors))

	// Build network
	maxLayers := config.NumLayers // Load ALL layers for proper model behavior

	network := &Network{
		GridRows:      1,
		GridCols:      1,
		LayersPerCell: maxLayers * 4, // 4 layers per transformer block: Attn, RMSNorm, SwiGLU, RMSNorm
		InputSize:     config.HiddenSize,
		BatchSize:     1,
		Layers:        make([]LayerConfig, 0),
	}

	// For each transformer layer
	fmt.Printf("Loading %d transformer blocks (%d total layers) out of %d available...\n",
		maxLayers, maxLayers*4, config.NumLayers)

	for i := 0; i < maxLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d", i)

		// 1. Pre-attention RMSNorm (input_layernorm)
		inputNorm := LayerConfig{
			Type:     LayerRMSNorm,
			NormSize: config.HiddenSize,
			Gamma:    getTensor(tensors, prefix+".input_layernorm.weight"),
			Epsilon:  float32(config.RMSNormEps),
		}
		network.Layers = append(network.Layers, inputNorm)

		// 2. Multi-head attention (GQA)
		qWeights, kWeights, vWeights, qBias, kBias, vBias := extractQKVWeights(tensors, prefix, config)
		outWeight := getTensor(tensors, prefix+".self_attn.o_proj.weight")
		// Transpose output weight from PyTorch [out, in] to LOOM [in, out]
		outWeightTransposed := transposeWeights(outWeight, config.HiddenSize, config.HiddenSize)
		outBias := make([]float32, config.HiddenSize) // No bias in Llama

		attnLayer := LayerConfig{
			Type:         LayerMultiHeadAttention,
			DModel:       config.HiddenSize,
			NumHeads:     config.NumHeads,
			NumKVHeads:   config.NumKVHeads,
			HeadDim:      config.HiddenSize / config.NumHeads,
			SeqLength:    128, // Default sequence length
			QWeights:     qWeights,
			KWeights:     kWeights,
			VWeights:     vWeights,
			QBias:        qBias,
			KBias:        kBias,
			VBias:        vBias,
			OutputWeight: outWeightTransposed,
			OutputBias:   outBias,
			RoPEFreqBase: float32(config.RoPETheta),
		}
		if attnLayer.RoPEFreqBase == 0 {
			attnLayer.RoPEFreqBase = 10000.0 // Default
		}
		network.Layers = append(network.Layers, attnLayer)

		// 3. Pre-MLP RMSNorm (post_attention_layernorm)
		postAttnNorm := LayerConfig{
			Type:     LayerRMSNorm,
			NormSize: config.HiddenSize,
			Gamma:    getTensor(tensors, prefix+".post_attention_layernorm.weight"),
			Epsilon:  float32(config.RMSNormEps),
		}
		network.Layers = append(network.Layers, postAttnNorm)

		// 4. SwiGLU layer (replaces simple MLP)
		gateWeights := getTensor(tensors, prefix+".mlp.gate_proj.weight")
		upWeights := getTensor(tensors, prefix+".mlp.up_proj.weight")
		downWeights := getTensor(tensors, prefix+".mlp.down_proj.weight")

		// Transpose weights from PyTorch [out, in] to LOOM [in, out]
		gateTransposed := transposeWeights(gateWeights, config.IntermediateSize, config.HiddenSize)
		upTransposed := transposeWeights(upWeights, config.IntermediateSize, config.HiddenSize)
		downTransposed := transposeWeights(downWeights, config.HiddenSize, config.IntermediateSize)

		swiGLULayer := LayerConfig{
			Type:         LayerSwiGLU,
			InputHeight:  config.HiddenSize,
			OutputHeight: config.IntermediateSize,
			GateWeights:  gateTransposed,
			UpWeights:    upTransposed,
			DownWeights:  downTransposed,
			GateBias:     make([]float32, config.IntermediateSize),
			UpBias:       make([]float32, config.IntermediateSize),
			DownBias:     make([]float32, config.HiddenSize),
		}
		network.Layers = append(network.Layers, swiGLULayer)

		fmt.Printf("Added layer %d/%d\n", i+1, config.NumLayers)
	}

	fmt.Printf("✅ Loaded transformer model with %d layers\n", len(network.Layers))

	// Initialize activation storage
	network.activations = make([][]float32, len(network.Layers)+1)
	network.preActivations = make([][]float32, len(network.Layers))

	return network, nil
}

// extractQKVWeights extracts Q, K, V weights for GQA attention
func extractQKVWeights(tensors map[string][]float32, prefix string, config TransformerConfig) ([]float32, []float32, []float32, []float32, []float32, []float32) {
	qWeight := getTensor(tensors, prefix+".self_attn.q_proj.weight")
	kWeight := getTensor(tensors, prefix+".self_attn.k_proj.weight")
	vWeight := getTensor(tensors, prefix+".self_attn.v_proj.weight")

	hiddenSize := config.HiddenSize
	headDim := 0
	if config.NumHeads > 0 {
		headDim = config.HiddenSize / config.NumHeads
	}
	kvDim := config.NumKVHeads * headDim

	// Handle empty weights gracefully (model may not have expected structure)
	var qWeightTransposed, kWeightTransposed, vWeightTransposed []float32

	if len(qWeight) > 0 {
		// Q: [hiddenSize, hiddenSize] -> transpose
		qWeightTransposed = transposeWeights(qWeight, hiddenSize, hiddenSize)
	} else {
		qWeightTransposed = make([]float32, hiddenSize*hiddenSize)
	}

	if len(kWeight) > 0 && kvDim > 0 {
		// K: [kvDim, hiddenSize] -> transpose
		kWeightTransposed = transposeWeights(kWeight, kvDim, hiddenSize)
	} else {
		kWeightTransposed = make([]float32, kvDim*hiddenSize)
	}

	if len(vWeight) > 0 && kvDim > 0 {
		// V: [kvDim, hiddenSize] -> transpose
		vWeightTransposed = transposeWeights(vWeight, kvDim, hiddenSize)
	} else {
		vWeightTransposed = make([]float32, kvDim*hiddenSize)
	}

	// Load biases (Qwen2.5 has them, unlike Llama)
	qBias := getTensor(tensors, prefix+".self_attn.q_proj.bias")
	kBias := getTensor(tensors, prefix+".self_attn.k_proj.bias")
	vBias := getTensor(tensors, prefix+".self_attn.v_proj.bias")

	// If biases don't exist, initialize to zero
	if len(qBias) == 0 {
		qBias = make([]float32, hiddenSize)
	}
	if len(kBias) == 0 {
		kBias = make([]float32, kvDim)
	}
	if len(vBias) == 0 {
		vBias = make([]float32, kvDim)
	}

	return qWeightTransposed, kWeightTransposed, vWeightTransposed, qBias, kBias, vBias
}

// transposeWeights transposes a weight matrix from [rows, cols] to [cols, rows]
func transposeWeights(weights []float32, rows, cols int) []float32 {
	transposed := make([]float32, len(weights))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			transposed[c*rows+r] = weights[r*cols+c]
		}
	}
	return transposed
}

// getTensor retrieves a tensor by name, with error handling
func getTensor(tensors map[string][]float32, name string) []float32 {
	// Try exact match
	if t, ok := tensors[name]; ok {
		fmt.Printf("  ✓ Loaded %s: %d values\n", name, len(t))
		return t
	}

	// Try case-insensitive match
	lowerName := strings.ToLower(name)
	for k, v := range tensors {
		if strings.ToLower(k) == lowerName {
			fmt.Printf("  ✓ Loaded %s (as %s): %d values\n", name, k, len(v))
			return v
		}
	}

	// Return empty tensor if not found (with warning)
	fmt.Printf("⚠️  Warning: tensor '%s' not found\n", name)
	return []float32{}
}

// validateArchitecture checks if the model architecture is supported
func validateArchitecture(config TransformerConfig) error {
	// Models that use encoder-decoder, BERT-style, or incompatible attention patterns
	unsupportedTypes := []string{
		"t5", "mt5", "bart", "bert", "roberta", "encoder-decoder", "marian",
		"detr", "yolos", "rt_detr", "yolo", // Detection models with encoder-decoder
		"vit", "deit", "swin", "beit", // Vision transformers (encoder-only)
	}
	modelType := strings.ToLower(config.ModelType)

	for _, unsup := range unsupportedTypes {
		if strings.Contains(modelType, unsup) {
			return fmt.Errorf("unsupported model type '%s': framework only supports Decoder-only models (Llama, Qwen, Mistral, GPT-NeoX, etc.)", config.ModelType)
		}
	}

	for _, arch := range config.Architectures {
		archLower := strings.ToLower(arch)
		if strings.Contains(archLower, "conditionalgeneration") || strings.Contains(archLower, "encoderdecoder") {
			return fmt.Errorf("unsupported architecture '%s': framework only supports CausalLM models", arch)
		}
	}

	return nil
}
