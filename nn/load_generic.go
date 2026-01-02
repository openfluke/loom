package nn

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// GenericModelConfig holds model configuration parsed from config.json
type GenericModelConfig struct {
	ModelType        string   `json:"model_type"`
	Architectures    []string `json:"architectures"`
	HiddenSize       int      `json:"hidden_size"`
	IntermediateSize int      `json:"intermediate_size"`
	NumLayers        int      `json:"num_hidden_layers"`
	NumHeads         int      `json:"num_attention_heads"`
	NumKVHeads       int      `json:"num_key_value_heads"`
	NumChannels      int      `json:"num_channels"`
	ImageSize        int      `json:"image_size"`
	PatchSize        int      `json:"patch_size"`
	NumClasses       int      `json:"num_labels"`
	VocabSize        int      `json:"vocab_size"`
	RMSNormEps       float64  `json:"rms_norm_eps"`
	LayerNormEps     float64  `json:"layer_norm_eps"`
}

// DetectedTensor represents a tensor with inferred layer type
type DetectedTensor struct {
	Name         string    // Original tensor name
	Shape        []int     // Tensor dimensions
	ShapeType    string    // Shape-based type: "1d", "2d", "3d", "4d"
	LoomType     LayerType // Inferred loom LayerType
	LoomTypeName string    // Human readable type name
	SpecialType  string    // "embedding", "bias", or empty for normal layers
	InSize       int       // Input dimension
	OutSize      int       // Output dimension
	KernelH      int       // For conv layers
	KernelW      int       // For conv layers
	CanLoad      bool      // Whether loom can load this layer
}

// AllSupportedTypes returns all LayerTypes that loom supports
func AllSupportedTypes() []LayerType {
	return []LayerType{
		LayerDense,              // 0
		LayerConv2D,             // 1
		LayerMultiHeadAttention, // 2
		LayerRNN,                // 3
		LayerLSTM,               // 4
		LayerSoftmax,            // 5
		LayerNorm,               // 6
		LayerResidual,           // 7
		LayerRMSNorm,            // 8
		LayerSwiGLU,             // 9
		LayerParallel,           // 10
		LayerEmbedding,          // 11
		LayerConv1D,             // 12
	}
}

// LayerTypeToName converts LayerType to human readable name
func LayerTypeToName(lt LayerType) string {
	switch lt {
	case LayerDense:
		return "Dense"
	case LayerConv2D:
		return "Conv2D"
	case LayerMultiHeadAttention:
		return "MultiHeadAttention"
	case LayerRNN:
		return "RNN"
	case LayerLSTM:
		return "LSTM"
	case LayerSoftmax:
		return "Softmax"
	case LayerNorm:
		return "LayerNorm"
	case LayerResidual:
		return "Residual"
	case LayerRMSNorm:
		return "RMSNorm"
	case LayerSwiGLU:
		return "SwiGLU"
	case LayerParallel:
		return "Parallel"
	case LayerEmbedding:
		return "Embedding"
	case LayerConv1D:
		return "Conv1D"
	default:
		return fmt.Sprintf("Unknown(%d)", lt)
	}
}

// InferLoomLayerType tries to determine the loom LayerType from shape, name hint, and config
// Primary: shape + config-based detection
// Secondary: name hints for disambiguation
// Returns: LayerType, canLoad, specialType ("embedding", "bias", or "")
func InferLoomLayerType(shape []int, nameHint string, config *GenericModelConfig) (LayerType, bool, string) {
	nameLower := strings.ToLower(nameHint)
	
	// Get config values for smarter detection
	hiddenSize := 0
	intermediateSize := 0
	numHeads := 0
	if config != nil {
		hiddenSize = config.HiddenSize
		intermediateSize = config.IntermediateSize
		numHeads = config.NumHeads
	}
	
	switch len(shape) {
	case 1:
		// 1D tensors: Could be LayerNorm, RMSNorm, or bias
		if strings.Contains(nameLower, "rms") {
			return LayerRMSNorm, true, ""
		}
		if strings.Contains(nameLower, "layernorm") || strings.Contains(nameLower, "layer_norm") || strings.Contains(nameLower, "ln") {
			return LayerNorm, true, ""
		}
		if strings.Contains(nameLower, "bias") {
			return LayerDense, false, "bias" // Bias attached to dense, not standalone
		}
		// Default to RMSNorm for gamma-only (more common in modern models)
		return LayerRMSNorm, true, ""
		
	case 2:
		outDim, inDim := shape[0], shape[1]
		
		// Embedding: vocab >> hidden (typically vocab > 4x hidden)
		if outDim > inDim*4 {
			return LayerEmbedding, true, "" // Embeddings are lookup tables
		}
		
		// Use config to detect SwiGLU: intermediate_size x hidden_size pattern
		if intermediateSize > 0 && hiddenSize > 0 {
			// SwiGLU gate/up: [intermediate, hidden]
			if outDim == intermediateSize && inDim == hiddenSize {
				return LayerSwiGLU, true, ""
			}
			// SwiGLU down: [hidden, intermediate]
			if outDim == hiddenSize && inDim == intermediateSize {
				return LayerSwiGLU, true, ""
			}
		}
		
		// Use config to detect MHA: [hidden, hidden] with proper head count
		if numHeads > 0 && hiddenSize > 0 {
			if outDim == hiddenSize && inDim == hiddenSize {
				// Q, K, V, O projections are [hidden, hidden]
				if strings.Contains(nameLower, "q_proj") || strings.Contains(nameLower, "k_proj") || 
				   strings.Contains(nameLower, "v_proj") || strings.Contains(nameLower, "o_proj") ||
				   strings.Contains(nameLower, "query") || strings.Contains(nameLower, "key") ||
				   strings.Contains(nameLower, "value") || strings.Contains(nameLower, "attn") {
					return LayerMultiHeadAttention, true, ""
				}
			}
			// GQA: K,V might be smaller [kv_dim, hidden]
			kvDim := config.NumKVHeads * (hiddenSize / numHeads)
			if config.NumKVHeads > 0 && (outDim == kvDim && inDim == hiddenSize) {
				return LayerMultiHeadAttention, true, ""
			}
		}
		
		// Name-based attention detection fallback
		if strings.Contains(nameLower, "q_proj") || strings.Contains(nameLower, "k_proj") || 
		   strings.Contains(nameLower, "v_proj") || strings.Contains(nameLower, "o_proj") ||
		   strings.Contains(nameLower, "attention") {
			return LayerMultiHeadAttention, true, ""
		}
		
		// SwiGLU name-based fallback
		if strings.Contains(nameLower, "gate") || strings.Contains(nameLower, "up_proj") || 
		   strings.Contains(nameLower, "down_proj") || strings.Contains(nameLower, "mlp") {
			return LayerSwiGLU, true, ""
		}
		
		// LSTM/RNN gates
		if strings.Contains(nameLower, "_ih") || strings.Contains(nameLower, "_hh") {
			if strings.Contains(nameLower, "lstm") {
				return LayerLSTM, true, ""
			}
			return LayerRNN, true, ""
		}
		
		// Default: Dense layer
		return LayerDense, true, ""
		
	case 3:
		// 3D: Conv1D
		return LayerConv1D, true, ""
		
	case 4:
		// 4D: Conv2D
		return LayerConv2D, true, ""
		
	default:
		return LayerDense, false, "unknown"
	}
}

// findMatchingBias looks up the bias tensor for a weight tensor
// Standard pattern: "foo.weight" -> "foo.bias" or "foo.kernel" -> "foo.bias"
func findMatchingBias(tensors map[string]TensorWithShape, weightName string) []float32 {
	// Try replacing .weight with .bias
	if strings.HasSuffix(weightName, ".weight") {
		biasName := strings.TrimSuffix(weightName, ".weight") + ".bias"
		if t, ok := tensors[biasName]; ok {
			return t.Values
		}
	}
	// Try replacing .kernel with .bias (TensorFlow naming)
	if strings.HasSuffix(weightName, ".kernel") {
		biasName := strings.TrimSuffix(weightName, ".kernel") + ".bias"
		if t, ok := tensors[biasName]; ok {
			return t.Values
		}
	}
	// Try appending .bias if no suffix
	if !strings.HasSuffix(weightName, ".weight") && !strings.HasSuffix(weightName, ".kernel") {
		biasName := weightName + ".bias"
		if t, ok := tensors[biasName]; ok {
			return t.Values
		}
	}
	return nil
}

// LoadGenericFromBytes loads any safetensors model, auto-detecting layer types
// weightsData: safetensors bytes (required)
// configData: config.json bytes (optional, can be nil)
// Returns: Network, detected tensors info, error
func LoadGenericFromBytes(weightsData []byte, configData []byte) (*Network, []DetectedTensor, error) {
	tensors, err := LoadSafetensorsWithShapes(weightsData)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load weights: %w", err)
	}

	var config GenericModelConfig
	if configData != nil && len(configData) > 0 {
		json.Unmarshal(configData, &config)
	}

	return buildNetworkAuto(tensors, config)
}

// InspectModel inspects a safetensors file and returns detailed tensor info
// Returns all detected tensors with their inferred loom types
// configData is optional (can be nil) but improves detection accuracy
func InspectModel(weightsData []byte, configData []byte) ([]DetectedTensor, int64, error) {
	tensors, err := LoadSafetensorsWithShapes(weightsData)
	if err != nil {
		return nil, 0, err
	}

	var config *GenericModelConfig
	if configData != nil && len(configData) > 0 {
		var c GenericModelConfig
		if json.Unmarshal(configData, &c) == nil {
			config = &c
		}
	}

	var detected []DetectedTensor
	var totalParams int64

	for name, t := range tensors {
		totalParams += int64(len(t.Values))

		loomType, canLoad, specialType := InferLoomLayerType(t.Shape, name, config)
		
		// Build display name - use specialType if set
		displayName := LayerTypeToName(loomType)
		if specialType != "" {
			displayName = specialType // "embedding", "bias", etc.
		}
		
		d := DetectedTensor{
			Name:         name,
			Shape:        t.Shape,
			ShapeType:    fmt.Sprintf("%dd", len(t.Shape)),
			LoomType:     loomType,
			LoomTypeName: displayName,
			SpecialType:  specialType,
			CanLoad:      canLoad,
		}

		// Extract dimensions
		switch len(t.Shape) {
		case 1:
			d.InSize = t.Shape[0]
			d.OutSize = t.Shape[0]
		case 2:
			d.OutSize = t.Shape[0]
			d.InSize = t.Shape[1]
		case 3:
			d.OutSize = t.Shape[0]
			d.InSize = t.Shape[1]
			d.KernelH = t.Shape[2]
		case 4:
			d.OutSize = t.Shape[0]
			d.InSize = t.Shape[1]
			d.KernelH = t.Shape[2]
			d.KernelW = t.Shape[3]
		}

		detected = append(detected, d)
	}

	return detected, totalParams, nil
}

// buildNetworkAuto constructs a network from tensors, auto-detecting types
func buildNetworkAuto(tensors map[string]TensorWithShape, config GenericModelConfig) (*Network, []DetectedTensor, error) {
	var detected []DetectedTensor
	hiddenSize := config.HiddenSize

	// Sort for consistent ordering
	var names []string
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	// First pass: detect and infer hidden size
	for _, name := range names {
		t := tensors[name]
		loomType, canLoad, specialType := InferLoomLayerType(t.Shape, name, &config)

		// Build display name
		displayName := LayerTypeToName(loomType)
		if specialType != "" {
			displayName = specialType
		}

		d := DetectedTensor{
			Name:         name,
			Shape:        t.Shape,
			ShapeType:    fmt.Sprintf("%dd", len(t.Shape)),
			LoomType:     loomType,
			LoomTypeName: displayName,
			SpecialType:  specialType,
			CanLoad:      canLoad,
		}

		switch len(t.Shape) {
		case 1:
			d.InSize = t.Shape[0]
			d.OutSize = t.Shape[0]
			if hiddenSize == 0 && t.Shape[0] > 64 {
				hiddenSize = t.Shape[0]
			}
		case 2:
			d.OutSize = t.Shape[0]
			d.InSize = t.Shape[1]
			if hiddenSize == 0 && t.Shape[0] == t.Shape[1] {
				hiddenSize = t.Shape[0]
			}
		case 4:
			d.OutSize = t.Shape[0]
			d.InSize = t.Shape[1]
			d.KernelH = t.Shape[2]
			d.KernelW = t.Shape[3]
		}



		detected = append(detected, d)
	}

	// Calculate and set hiddenSize if not already found
	if hiddenSize == 0 {
		hiddenSize = 768
	}

	// Build network
	network := &Network{
		GridRows:      1,
		GridCols:      1,
		LayersPerCell: 1,
		InputSize:     hiddenSize,
		BatchSize:     1,
		Layers:        make([]LayerConfig, 0),
	}

	// Second pass: create simple loadable layers
	for i, name := range names {
		t := tensors[name]
		d := &detected[i]

		if !d.CanLoad {
			continue
		}

		switch d.LoomType {
		case LayerRMSNorm:
			if len(t.Shape) == 1 {
				network.Layers = append(network.Layers, LayerConfig{
					Type:     LayerRMSNorm,
					NormSize: d.InSize,
					Gamma:    t.Values,
					Epsilon:  float32(config.RMSNormEps),
				})
			}

		case LayerNorm:
			if len(t.Shape) == 1 {
				eps := config.LayerNormEps
				if eps == 0 {
					eps = 1e-5
				}
				network.Layers = append(network.Layers, LayerConfig{
					Type:     LayerNorm,
					NormSize: d.InSize,
					Gamma:    t.Values,
					Epsilon:  float32(eps),
				})
			}

		case LayerDense:
			if len(t.Shape) == 2 {
				transposed := transposeWeights(t.Values, d.OutSize, d.InSize)
				// Find matching bias tensor
				bias := findMatchingBias(tensors, name)
				if bias == nil {
					bias = make([]float32, d.OutSize)
				}
				network.Layers = append(network.Layers, LayerConfig{
					Type:         LayerDense,
					InputHeight:  d.InSize,
					OutputHeight: d.OutSize,
					Kernel:       transposed,
					Bias:         bias,
				})
			}

		case LayerConv2D:
			if len(t.Shape) == 4 {
				// Find matching bias tensor
				bias := findMatchingBias(tensors, name)
				if bias == nil {
					bias = make([]float32, d.OutSize)
				}
				network.Layers = append(network.Layers, LayerConfig{
					Type:          LayerConv2D,
					InputChannels: d.InSize,
					Filters:       d.OutSize,
					KernelSize:    d.KernelH, // Assuming square kernel
					Kernel:        t.Values,
					Bias:          bias,
					Stride:        1,
					Padding:       d.KernelH / 2,
				})
			}

		case LayerEmbedding:
			if len(t.Shape) == 2 {
				network.Layers = append(network.Layers, LayerConfig{
					Type:             LayerEmbedding,
					VocabSize:        d.OutSize,
					EmbeddingDim:     d.InSize,
					EmbeddingWeights: t.Values,
				})
			}

		case LayerConv1D:
			if len(t.Shape) == 3 {
				// Find matching bias tensor
				bias := findMatchingBias(tensors, name)
				if bias == nil {
					bias = make([]float32, d.OutSize)
				}
				network.Layers = append(network.Layers, LayerConfig{
					Type:             LayerConv1D,
					Conv1DFilters:    d.OutSize,
					Conv1DInChannels: d.InSize,
					Conv1DKernelSize: d.KernelH,
					Conv1DKernel:     t.Values,
					Conv1DBias:       bias,
					Conv1DStride:     1,
					Conv1DPadding:    d.KernelH / 2,
				})
			}

		// Note: MultiHeadAttention, SwiGLU, LSTM, RNN require multiple tensors
		// They need architecture-specific loading logic
		}
	}

	// Third pass: Identify and construct complex layers (MHA)
	groups := groupRelatedTensors(detected)
	network.processMultiHeadAttentionGroups(groups, tensors, config)

	network.LayersPerCell = len(network.Layers)
	network.activations = make([][]float32, len(network.Layers)+1)
	network.preActivations = make([][]float32, len(network.Layers))

	return network, detected, nil
}

// GetLoomCompatibility returns a summary of what can/can't be loaded
func GetLoomCompatibility(detected []DetectedTensor) map[string]int {
	compatible := make(map[string]int)
	incompatible := make(map[string]int)

	for _, d := range detected {
		if d.CanLoad {
			compatible[d.LoomTypeName]++
		} else {
			incompatible[d.LoomTypeName+"(unsupported)"]++
		}
	}

	result := make(map[string]int)
	for k, v := range compatible {
		result["✓ "+k] = v
	}
	for k, v := range incompatible {
		result["✗ "+k] = v
	}
	return result
}
