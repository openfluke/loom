package nn

import (
	"encoding/json"
	"fmt"
	"os"
)

// ImportedModelFormat represents models converted from PyTorch/TensorFlow
type ImportedModelFormat struct {
	InputSize     int                    `json:"input_size"`
	GridRows      int                    `json:"grid_rows"`
	GridCols      int                    `json:"grid_cols"`
	LayersPerCell int                    `json:"layers_per_cell"`
	BatchSize     int                    `json:"batch_size"`
	UseGPU        bool                   `json:"use_gpu"`
	Layers        []ImportedLayerConfig  `json:"layers"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// ImportedLayerConfig represents a layer from an imported model
type ImportedLayerConfig struct {
	Type       string `json:"type"`
	Activation string `json:"activation,omitempty"`

	// Dense layer
	InputSize  int         `json:"input_size,omitempty"`
	OutputSize int         `json:"output_size,omitempty"`
	Kernel     [][]float32 `json:"kernel,omitempty"`
	Bias       []float32   `json:"bias,omitempty"`

	// Multi-head attention
	DModel       int         `json:"d_model,omitempty"`
	NumHeads     int         `json:"num_heads,omitempty"`
	HeadDim      int         `json:"head_dim,omitempty"`
	SeqLength    int         `json:"seq_length,omitempty"`
	QWeights     [][]float32 `json:"q_weights,omitempty"`
	QBias        []float32   `json:"q_bias,omitempty"`
	KWeights     [][]float32 `json:"k_weights,omitempty"`
	KBias        []float32   `json:"k_bias,omitempty"`
	VWeights     [][]float32 `json:"v_weights,omitempty"`
	VBias        []float32   `json:"v_bias,omitempty"`
	OutputWeight [][]float32 `json:"output_weight,omitempty"`
	OutputBias   []float32   `json:"output_bias,omitempty"`

	// LayerNorm
	NormSize int       `json:"norm_size,omitempty"`
	Gamma    []float32 `json:"gamma,omitempty"`
	Beta     []float32 `json:"beta,omitempty"`
	Epsilon  float32   `json:"epsilon,omitempty"`
}

// LoadImportedModel loads a model converted from PyTorch/TensorFlow/HuggingFace
func LoadImportedModel(filepath string, modelID string) (*Network, error) {
	// Read file
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	// Parse JSON
	var imported ImportedModelFormat
	if err := json.Unmarshal(data, &imported); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	// Validate
	if len(imported.Layers) == 0 {
		return nil, fmt.Errorf("no layers found in model")
	}

	// Create network
	network := NewNetwork(
		imported.InputSize,
		imported.GridRows,
		imported.GridCols,
		imported.LayersPerCell,
	)

	if network == nil {
		return nil, fmt.Errorf("failed to create network")
	}

	// Set batch size if specified
	if imported.BatchSize > 0 {
		network.BatchSize = imported.BatchSize
	} else if len(imported.Layers) > 0 {
		// If batch size not set, try to infer from first MHA layer's seqLength
		for _, layer := range imported.Layers {
			if layer.Type == "multi_head_attention" && layer.SeqLength > 0 {
				network.BatchSize = layer.SeqLength
				fmt.Printf("Inferred batch size from MHA layer: %d\n", network.BatchSize)
				break
			}
		}
	} // Convert and set each layer
	for i, layer := range imported.Layers {
		row := i / (imported.GridCols * imported.LayersPerCell)
		remainder := i % (imported.GridCols * imported.LayersPerCell)
		col := remainder / imported.LayersPerCell
		layerIdx := remainder % imported.LayersPerCell

		config, err := convertImportedLayer(&layer)
		if err != nil {
			return nil, fmt.Errorf("failed to convert layer %d: %w", i, err)
		}

		// Set the layer (dereference pointer)
		network.SetLayer(row, col, layerIdx, *config)
	}

	fmt.Printf("✅ Loaded imported model '%s'\n", modelID)
	fmt.Printf("   Source: %v\n", imported.Metadata["source"])
	if modelName, ok := imported.Metadata["model_name"]; ok {
		fmt.Printf("   Original model: %s\n", modelName)
	}
	fmt.Printf("   Layers: %d\n", len(imported.Layers))
	fmt.Printf("   Input size: %d\n", imported.InputSize)

	return network, nil
}

// convertImportedLayer converts an imported layer config to LOOM LayerConfig
func convertImportedLayer(imported *ImportedLayerConfig) (*LayerConfig, error) {
	config := &LayerConfig{}

	// Parse activation
	config.Activation = parseActivation(imported.Activation)

	switch imported.Type {
	case "dense":
		config.Type = LayerDense
		// Dense layers reuse Conv2D fields for size information
		config.InputHeight = imported.InputSize   // Reused as inputSize
		config.OutputHeight = imported.OutputSize // Reused as outputSize

		// Flatten kernel (already transposed correctly from Python)
		if len(imported.Kernel) > 0 {
			config.Kernel = make([]float32, 0, imported.InputSize*imported.OutputSize)
			for _, row := range imported.Kernel {
				config.Kernel = append(config.Kernel, row...)
			}
		}

		// Copy bias
		if len(imported.Bias) > 0 {
			config.Bias = make([]float32, len(imported.Bias))
			copy(config.Bias, imported.Bias)
		}

	case "multi_head_attention":
		config.Type = LayerMultiHeadAttention
		config.DModel = imported.DModel
		config.NumHeads = imported.NumHeads
		config.HeadDim = imported.HeadDim
		config.SeqLength = imported.SeqLength

		// Flatten Q, K, V, Output weights
		if len(imported.QWeights) > 0 {
			config.QWeights = make([]float32, 0, len(imported.QWeights)*len(imported.QWeights[0]))
			for _, row := range imported.QWeights {
				config.QWeights = append(config.QWeights, row...)
			}
		}

		if len(imported.KWeights) > 0 {
			config.KWeights = make([]float32, 0, len(imported.KWeights)*len(imported.KWeights[0]))
			for _, row := range imported.KWeights {
				config.KWeights = append(config.KWeights, row...)
			}
		}

		if len(imported.VWeights) > 0 {
			config.VWeights = make([]float32, 0, len(imported.VWeights)*len(imported.VWeights[0]))
			for _, row := range imported.VWeights {
				config.VWeights = append(config.VWeights, row...)
			}
		}

		if len(imported.OutputWeight) > 0 {
			config.OutputWeight = make([]float32, 0, len(imported.OutputWeight)*len(imported.OutputWeight[0]))
			for _, row := range imported.OutputWeight {
				config.OutputWeight = append(config.OutputWeight, row...)
			}
		} // Copy biases
		if len(imported.QBias) > 0 {
			config.QBias = make([]float32, len(imported.QBias))
			copy(config.QBias, imported.QBias)
		}

		if len(imported.KBias) > 0 {
			config.KBias = make([]float32, len(imported.KBias))
			copy(config.KBias, imported.KBias)
		}

		if len(imported.VBias) > 0 {
			config.VBias = make([]float32, len(imported.VBias))
			copy(config.VBias, imported.VBias)
		}

		if len(imported.OutputBias) > 0 {
			config.OutputBias = make([]float32, len(imported.OutputBias))
			copy(config.OutputBias, imported.OutputBias)
		}

	case "layer_norm":
		config.Type = LayerNorm
		config.NormSize = imported.NormSize
		config.Epsilon = imported.Epsilon

		if len(imported.Gamma) > 0 {
			config.Gamma = make([]float32, len(imported.Gamma))
			copy(config.Gamma, imported.Gamma)
		}

		if len(imported.Beta) > 0 {
			config.Beta = make([]float32, len(imported.Beta))
			copy(config.Beta, imported.Beta)
		}

	case "rms_norm":
		config.Type = LayerRMSNorm
		config.NormSize = imported.NormSize
		config.Epsilon = imported.Epsilon

		if len(imported.Gamma) > 0 {
			config.Gamma = make([]float32, len(imported.Gamma))
			copy(config.Gamma, imported.Gamma)
		}
		// Note: RMSNorm doesn't use Beta

	default:
		return nil, fmt.Errorf("unsupported layer type: %s", imported.Type)
	}

	return config, nil
}

// parseActivation converts activation string to enum
func parseActivation(name string) ActivationType {
	switch name {
	case "relu":
		return ActivationScaledReLU // Closest available
	case "sigmoid":
		return ActivationSigmoid
	case "tanh":
		return ActivationTanh
	case "gelu":
		return ActivationSoftplus // Closest available (both smooth non-linearities)
	case "softmax":
		return ActivationSigmoid // Softmax is handled at layer level, not activation
	case "linear", "":
		return ActivationScaledReLU // Default
	default:
		fmt.Printf("Warning: unknown activation '%s', using ScaledReLU\n", name)
		return ActivationScaledReLU
	}
}
