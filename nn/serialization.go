package nn

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
)

// ModelBundle represents a collection of saved models
type ModelBundle struct {
	Type    string       `json:"type"`
	Version int          `json:"version"`
	Models  []SavedModel `json:"models"`
}

// SavedModel represents a single saved model with config and weights
type SavedModel struct {
	ID      string         `json:"id"`
	Config  NetworkConfig  `json:"cfg"`
	Weights EncodedWeights `json:"weights"`
}

// NetworkConfig represents the network architecture
type NetworkConfig struct {
	ID            string            `json:"id"`
	BatchSize     int               `json:"batch_size"`
	GridRows      int               `json:"grid_rows"`
	GridCols      int               `json:"grid_cols"`
	LayersPerCell int               `json:"layers_per_cell"`
	Layers        []LayerDefinition `json:"layers"`
	Seed          int64             `json:"seed,omitempty"`
}

// LayerDefinition defines a single layer's configuration
type LayerDefinition struct {
	Type       string `json:"type"`
	Activation string `json:"activation"`

	// Dense layer fields
	Width  int `json:"width,omitempty"`
	Height int `json:"height,omitempty"`

	// Conv2D fields
	InputChannels int `json:"input_channels,omitempty"`
	Filters       int `json:"filters,omitempty"`
	KernelSize    int `json:"kernel_size,omitempty"`
	Stride        int `json:"stride,omitempty"`
	Padding       int `json:"padding,omitempty"`
	InputHeight   int `json:"input_height,omitempty"`
	InputWidth    int `json:"input_width,omitempty"`
	OutputHeight  int `json:"output_height,omitempty"`
	OutputWidth   int `json:"output_width,omitempty"`

	// MHA fields
	DModel    int `json:"d_model,omitempty"`
	NumHeads  int `json:"num_heads,omitempty"`
	SeqLength int `json:"seq_length,omitempty"`

	// RNN/LSTM fields
	InputSize  int `json:"input_size,omitempty"`
	HiddenSize int `json:"hidden_size,omitempty"`

	// Softmax fields
	SoftmaxVariant   string    `json:"softmax_variant,omitempty"`
	SoftmaxRows      int       `json:"softmax_rows,omitempty"`
	SoftmaxCols      int       `json:"softmax_cols,omitempty"`
	Temperature      float32   `json:"temperature,omitempty"`
	GumbelNoise      bool      `json:"gumbel_noise,omitempty"`
	Mask             []bool    `json:"mask,omitempty"`
	HierarchyLevels  []int     `json:"hierarchy_levels,omitempty"`
	AdaptiveClusters [][]int   `json:"adaptive_clusters,omitempty"`
	MixtureWeights   []float32 `json:"mixture_weights,omitempty"`
	EntmaxAlpha      float32   `json:"entmax_alpha,omitempty"`
}

// EncodedWeights stores weights in base64-encoded JSON format
type EncodedWeights struct {
	Format string `json:"fmt"`
	Data   string `json:"data"`
}

// WeightsData represents the actual weight values
type WeightsData struct {
	Type   string         `json:"type"`
	Layers []LayerWeights `json:"layers"`
}

// LayerWeights stores weights for a single layer
type LayerWeights struct {
	// Dense weights (stored as bias per neuron)
	Biases []float32 `json:"biases,omitempty"`

	// Conv2D weights
	Kernel   []float32 `json:"kernel,omitempty"`
	ConvBias []float32 `json:"conv_bias,omitempty"`

	// MHA weights
	QWeights     []float32 `json:"q_weights,omitempty"`
	KWeights     []float32 `json:"k_weights,omitempty"`
	VWeights     []float32 `json:"v_weights,omitempty"`
	OutputWeight []float32 `json:"output_weight,omitempty"`
	QBias        []float32 `json:"q_bias,omitempty"`
	KBias        []float32 `json:"k_bias,omitempty"`
	VBias        []float32 `json:"v_bias,omitempty"`
	OutputBias   []float32 `json:"output_bias,omitempty"`

	// RNN weights
	WeightIH []float32 `json:"weight_ih,omitempty"`
	WeightHH []float32 `json:"weight_hh,omitempty"`
	BiasH    []float32 `json:"bias_h,omitempty"`

	// LSTM weights
	WeightII []float32 `json:"weight_ii,omitempty"`
	WeightIF []float32 `json:"weight_if,omitempty"`
	WeightIG []float32 `json:"weight_ig,omitempty"`
	WeightIO []float32 `json:"weight_io,omitempty"`
	WeightHI []float32 `json:"weight_hi,omitempty"`
	WeightHF []float32 `json:"weight_hf,omitempty"`
	WeightHG []float32 `json:"weight_hg,omitempty"`
	WeightHO []float32 `json:"weight_ho,omitempty"`
	BiasI    []float32 `json:"bias_i,omitempty"`
	BiasF    []float32 `json:"bias_f,omitempty"`
	BiasG    []float32 `json:"bias_g,omitempty"`
	BiasO    []float32 `json:"bias_o,omitempty"`
}

// SaveModel saves a single model to a file
func (n *Network) SaveModel(filename string, modelID string) error {
	bundle := ModelBundle{
		Type:    "modelhost/bundle",
		Version: 1,
		Models:  []SavedModel{},
	}

	savedModel, err := n.SerializeModel(modelID)
	if err != nil {
		return fmt.Errorf("failed to serialize model: %w", err)
	}

	bundle.Models = append(bundle.Models, savedModel)

	return bundle.SaveToFile(filename)
}

// SaveBundle saves multiple models to a bundle file
func SaveBundle(filename string, models map[string]*Network) error {
	bundle := ModelBundle{
		Type:    "modelhost/bundle",
		Version: 1,
		Models:  []SavedModel{},
	}

	for id, network := range models {
		savedModel, err := network.SerializeModel(id)
		if err != nil {
			return fmt.Errorf("failed to serialize model %s: %w", id, err)
		}
		bundle.Models = append(bundle.Models, savedModel)
	}

	return bundle.SaveToFile(filename)
}

// SerializeModel converts the network to a SavedModel structure
func (n *Network) SerializeModel(modelID string) (SavedModel, error) {
	config := NetworkConfig{
		ID:            modelID,
		BatchSize:     n.BatchSize,
		GridRows:      n.GridRows,
		GridCols:      n.GridCols,
		LayersPerCell: n.LayersPerCell,
		Layers:        []LayerDefinition{},
	}

	weightsData := WeightsData{
		Type:   "float32",
		Layers: []LayerWeights{},
	}

	// Serialize each layer
	totalLayers := n.GridRows * n.GridCols * n.LayersPerCell
	for i := 0; i < totalLayers; i++ {
		row := i / (n.GridCols * n.LayersPerCell)
		col := (i / n.LayersPerCell) % n.GridCols
		layer := i % n.LayersPerCell

		layerConfig := n.GetLayer(row, col, layer)

		// Create layer definition
		layerDef := LayerDefinition{
			Type:       layerTypeToString(layerConfig.Type),
			Activation: activationToString(layerConfig.Activation),
		}

		// Create layer weights
		layerWeights := LayerWeights{}

		switch layerConfig.Type {
		case LayerDense:
			layerDef.Width = 1
			layerDef.Height = 1
			layerDef.InputHeight = layerConfig.InputHeight   // inputSize
			layerDef.OutputHeight = layerConfig.OutputHeight // outputSize
			layerWeights.Kernel = layerConfig.Kernel         // Weight matrix
			layerWeights.Biases = layerConfig.Bias           // Bias vector

		case LayerConv2D:
			layerDef.InputChannels = layerConfig.InputChannels
			layerDef.Filters = layerConfig.Filters
			layerDef.KernelSize = layerConfig.KernelSize
			layerDef.Stride = layerConfig.Stride
			layerDef.Padding = layerConfig.Padding
			layerDef.InputHeight = layerConfig.InputHeight
			layerDef.InputWidth = layerConfig.InputWidth
			layerDef.OutputHeight = layerConfig.OutputHeight
			layerDef.OutputWidth = layerConfig.OutputWidth
			layerWeights.Kernel = layerConfig.Kernel
			layerWeights.ConvBias = layerConfig.Bias

		case LayerMultiHeadAttention:
			layerDef.DModel = layerConfig.DModel
			layerDef.NumHeads = layerConfig.NumHeads
			layerDef.SeqLength = layerConfig.SeqLength
			layerWeights.QWeights = layerConfig.QWeights
			layerWeights.KWeights = layerConfig.KWeights
			layerWeights.VWeights = layerConfig.VWeights
			layerWeights.OutputWeight = layerConfig.OutputWeight
			layerWeights.QBias = layerConfig.QBias
			layerWeights.KBias = layerConfig.KBias
			layerWeights.VBias = layerConfig.VBias
			layerWeights.OutputBias = layerConfig.OutputBias

		case LayerRNN:
			layerDef.InputSize = layerConfig.RNNInputSize
			layerDef.HiddenSize = layerConfig.HiddenSize
			layerDef.SeqLength = layerConfig.SeqLength
			layerWeights.WeightIH = layerConfig.WeightIH
			layerWeights.WeightHH = layerConfig.WeightHH
			layerWeights.BiasH = layerConfig.BiasH

		case LayerLSTM:
			layerDef.InputSize = layerConfig.RNNInputSize
			layerDef.HiddenSize = layerConfig.HiddenSize
			layerDef.SeqLength = layerConfig.SeqLength
			layerWeights.WeightII = layerConfig.WeightIH_i
			layerWeights.WeightIF = layerConfig.WeightIH_f
			layerWeights.WeightIG = layerConfig.WeightIH_g
			layerWeights.WeightIO = layerConfig.WeightIH_o
			layerWeights.WeightHI = layerConfig.WeightHH_i
			layerWeights.WeightHF = layerConfig.WeightHH_f
			layerWeights.WeightHG = layerConfig.WeightHH_g
			layerWeights.WeightHO = layerConfig.WeightHH_o
			layerWeights.BiasI = layerConfig.BiasH_i
			layerWeights.BiasF = layerConfig.BiasH_f
			layerWeights.BiasG = layerConfig.BiasH_g
			layerWeights.BiasO = layerConfig.BiasH_o

		case LayerSoftmax:
			layerDef.SoftmaxVariant = softmaxTypeToString(layerConfig.SoftmaxVariant)
			layerDef.SoftmaxRows = layerConfig.SoftmaxRows
			layerDef.SoftmaxCols = layerConfig.SoftmaxCols
			layerDef.Temperature = layerConfig.Temperature
			layerDef.GumbelNoise = layerConfig.GumbelNoise
			layerDef.Mask = layerConfig.Mask
			layerDef.HierarchyLevels = layerConfig.HierarchyLevels
			layerDef.AdaptiveClusters = layerConfig.AdaptiveClusters
			layerDef.MixtureWeights = layerConfig.MixtureWeights
			layerDef.EntmaxAlpha = layerConfig.EntmaxAlpha
			// Softmax has no trainable weights
		}

		config.Layers = append(config.Layers, layerDef)
		weightsData.Layers = append(weightsData.Layers, layerWeights)
	}

	// Encode weights to base64
	weightsJSON, err := json.Marshal(weightsData)
	if err != nil {
		return SavedModel{}, fmt.Errorf("failed to marshal weights: %w", err)
	}

	encodedWeights := EncodedWeights{
		Format: "jsonModelB64",
		Data:   base64.StdEncoding.EncodeToString(weightsJSON),
	}

	return SavedModel{
		ID:      modelID,
		Config:  config,
		Weights: encodedWeights,
	}, nil
}

// LoadModel loads a single model from a file
func LoadModel(filename string, modelID string) (*Network, error) {
	bundle, err := LoadBundle(filename)
	if err != nil {
		return nil, err
	}

	for _, savedModel := range bundle.Models {
		if savedModel.ID == modelID {
			return DeserializeModel(savedModel)
		}
	}

	return nil, fmt.Errorf("model %s not found in bundle", modelID)
}

// LoadBundle loads a model bundle from a file
func LoadBundle(filename string) (*ModelBundle, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	return LoadBundleFromString(string(data))
}

// LoadBundleFromString loads a model bundle from a JSON string
// This is useful for WASM, CABI pipelines, or embedding models directly in code
func LoadBundleFromString(jsonString string) (*ModelBundle, error) {
	var bundle ModelBundle
	if err := json.Unmarshal([]byte(jsonString), &bundle); err != nil {
		return nil, fmt.Errorf("failed to unmarshal bundle: %w", err)
	}

	if bundle.Type != "modelhost/bundle" {
		return nil, fmt.Errorf("invalid bundle type: %s", bundle.Type)
	}

	return &bundle, nil
}

// LoadModelFromString loads a single model from a JSON string
// This is useful for WASM, CABI pipelines, or embedding models directly in code
func LoadModelFromString(jsonString string, modelID string) (*Network, error) {
	bundle, err := LoadBundleFromString(jsonString)
	if err != nil {
		return nil, err
	}

	for _, savedModel := range bundle.Models {
		if savedModel.ID == modelID {
			return DeserializeModel(savedModel)
		}
	}

	return nil, fmt.Errorf("model %s not found in bundle", modelID)
}

// SaveToString converts the bundle to a JSON string
// This is useful for WASM, CABI pipelines, or returning models over network
func (b *ModelBundle) SaveToString() (string, error) {
	data, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal bundle: %w", err)
	}
	return string(data), nil
}

// SaveModelToString saves a single model to a JSON string
// This is useful for WASM, CABI pipelines, or returning models over network
func (n *Network) SaveModelToString(modelID string) (string, error) {
	bundle := ModelBundle{
		Type:    "modelhost/bundle",
		Version: 1,
		Models:  []SavedModel{},
	}

	savedModel, err := n.SerializeModel(modelID)
	if err != nil {
		return "", fmt.Errorf("failed to serialize model: %w", err)
	}

	bundle.Models = append(bundle.Models, savedModel)

	return bundle.SaveToString()
}

// SaveToFile saves the bundle to a file
func (b *ModelBundle) SaveToFile(filename string) error {
	data, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal bundle: %w", err)
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

// DeserializeModel creates a Network from a SavedModel
func DeserializeModel(saved SavedModel) (*Network, error) {
	config := saved.Config

	// Create network
	network := NewNetwork(
		config.BatchSize,
		config.GridRows,
		config.GridCols,
		config.LayersPerCell,
	)
	network.BatchSize = config.BatchSize

	// Decode weights
	weightsJSON, err := base64.StdEncoding.DecodeString(saved.Weights.Data)
	if err != nil {
		return nil, fmt.Errorf("failed to decode weights: %w", err)
	}

	var weightsData WeightsData
	if err := json.Unmarshal(weightsJSON, &weightsData); err != nil {
		return nil, fmt.Errorf("failed to unmarshal weights: %w", err)
	}

	if len(config.Layers) != len(weightsData.Layers) {
		return nil, fmt.Errorf("layer count mismatch: config=%d, weights=%d",
			len(config.Layers), len(weightsData.Layers))
	}

	// Restore each layer
	for i, layerDef := range config.Layers {
		row := i / (config.GridCols * config.LayersPerCell)
		col := (i / config.LayersPerCell) % config.GridCols
		layer := i % config.LayersPerCell

		layerWeights := weightsData.Layers[i]

		var layerConfig LayerConfig

		switch layerDef.Type {
		case "dense":
			layerConfig = LayerConfig{
				Type:         LayerDense,
				Activation:   stringToActivation(layerDef.Activation),
				InputHeight:  layerDef.InputHeight,  // inputSize
				OutputHeight: layerDef.OutputHeight, // outputSize
				Kernel:       layerWeights.Kernel,   // Weight matrix
				Bias:         layerWeights.Biases,   // Bias vector
			}

		case "conv2d":
			layerConfig = LayerConfig{
				Type:          LayerConv2D,
				Activation:    stringToActivation(layerDef.Activation),
				InputChannels: layerDef.InputChannels,
				Filters:       layerDef.Filters,
				KernelSize:    layerDef.KernelSize,
				Stride:        layerDef.Stride,
				Padding:       layerDef.Padding,
				InputHeight:   layerDef.InputHeight,
				InputWidth:    layerDef.InputWidth,
				OutputHeight:  layerDef.OutputHeight,
				OutputWidth:   layerDef.OutputWidth,
				Kernel:        layerWeights.Kernel,
				Bias:          layerWeights.ConvBias,
			}

		case "mha", "multi_head_attention":
			layerConfig = LayerConfig{
				Type:         LayerMultiHeadAttention,
				Activation:   stringToActivation(layerDef.Activation),
				DModel:       layerDef.DModel,
				NumHeads:     layerDef.NumHeads,
				SeqLength:    layerDef.SeqLength,
				HeadDim:      layerDef.DModel / layerDef.NumHeads,
				QWeights:     layerWeights.QWeights,
				KWeights:     layerWeights.KWeights,
				VWeights:     layerWeights.VWeights,
				OutputWeight: layerWeights.OutputWeight,
				QBias:        layerWeights.QBias,
				KBias:        layerWeights.KBias,
				VBias:        layerWeights.VBias,
				OutputBias:   layerWeights.OutputBias,
			}

		case "rnn":
			layerConfig = LayerConfig{
				Type:         LayerRNN,
				Activation:   stringToActivation(layerDef.Activation),
				RNNInputSize: layerDef.InputSize,
				HiddenSize:   layerDef.HiddenSize,
				SeqLength:    layerDef.SeqLength,
				WeightIH:     layerWeights.WeightIH,
				WeightHH:     layerWeights.WeightHH,
				BiasH:        layerWeights.BiasH,
			}

		case "lstm":
			layerConfig = LayerConfig{
				Type:         LayerLSTM,
				Activation:   stringToActivation(layerDef.Activation),
				RNNInputSize: layerDef.InputSize,
				HiddenSize:   layerDef.HiddenSize,
				SeqLength:    layerDef.SeqLength,
				WeightIH_i:   layerWeights.WeightII,
				WeightIH_f:   layerWeights.WeightIF,
				WeightIH_g:   layerWeights.WeightIG,
				WeightIH_o:   layerWeights.WeightIO,
				WeightHH_i:   layerWeights.WeightHI,
				WeightHH_f:   layerWeights.WeightHF,
				WeightHH_g:   layerWeights.WeightHG,
				WeightHH_o:   layerWeights.WeightHO,
				BiasH_i:      layerWeights.BiasI,
				BiasH_f:      layerWeights.BiasF,
				BiasH_g:      layerWeights.BiasG,
				BiasH_o:      layerWeights.BiasO,
			}

		case "softmax":
			// Initialize mask if needed
			mask := layerDef.Mask
			if mask == nil && layerDef.SoftmaxVariant == "masked" {
				// If mask wasn't saved, create a default all-true mask
				// The size should be determined from context
				mask = []bool{}
			}

			layerConfig = LayerConfig{
				Type:             LayerSoftmax,
				SoftmaxVariant:   stringToSoftmaxType(layerDef.SoftmaxVariant),
				SoftmaxRows:      layerDef.SoftmaxRows,
				SoftmaxCols:      layerDef.SoftmaxCols,
				Temperature:      layerDef.Temperature,
				GumbelNoise:      layerDef.GumbelNoise,
				Mask:             mask,
				HierarchyLevels:  layerDef.HierarchyLevels,
				AdaptiveClusters: layerDef.AdaptiveClusters,
				MixtureWeights:   layerDef.MixtureWeights,
				EntmaxAlpha:      layerDef.EntmaxAlpha,
			}

		default:
			return nil, fmt.Errorf("unknown layer type: %s", layerDef.Type)
		}

		network.SetLayer(row, col, layer, layerConfig)
	}

	return network, nil
}

// Helper functions for type conversions
func layerTypeToString(lt LayerType) string {
	switch lt {
	case LayerDense:
		return "dense"
	case LayerConv2D:
		return "conv2d"
	case LayerMultiHeadAttention:
		return "multi_head_attention"
	case LayerRNN:
		return "rnn"
	case LayerLSTM:
		return "lstm"
	case LayerSoftmax:
		return "softmax"
	default:
		return "unknown"
	}
}

func softmaxTypeToString(st SoftmaxType) string {
	switch st {
	case SoftmaxStandard:
		return "standard"
	case SoftmaxGrid:
		return "grid"
	case SoftmaxHierarchical:
		return "hierarchical"
	case SoftmaxTemperature:
		return "temperature"
	case SoftmaxGumbel:
		return "gumbel"
	case SoftmaxMasked:
		return "masked"
	case SoftmaxSparse:
		return "sparse"
	case SoftmaxAdaptive:
		return "adaptive"
	case SoftmaxMixture:
		return "mixture"
	case SoftmaxEntmax:
		return "entmax"
	default:
		return "standard"
	}
}

func stringToSoftmaxType(s string) SoftmaxType {
	switch s {
	case "standard":
		return SoftmaxStandard
	case "grid":
		return SoftmaxGrid
	case "hierarchical":
		return SoftmaxHierarchical
	case "temperature":
		return SoftmaxTemperature
	case "gumbel":
		return SoftmaxGumbel
	case "masked":
		return SoftmaxMasked
	case "sparse":
		return SoftmaxSparse
	case "adaptive":
		return SoftmaxAdaptive
	case "mixture":
		return SoftmaxMixture
	case "entmax":
		return SoftmaxEntmax
	default:
		return SoftmaxStandard
	}
}

func activationToString(a ActivationType) string {
	switch a {
	case ActivationScaledReLU:
		return "relu"
	case ActivationSigmoid:
		return "sigmoid"
	case ActivationTanh:
		return "tanh"
	case ActivationSoftplus:
		return "softplus"
	case ActivationLeakyReLU:
		return "leaky_relu"
	default:
		return "linear"
	}
}

func stringToActivation(s string) ActivationType {
	switch s {
	case "relu":
		return ActivationScaledReLU
	case "sigmoid":
		return ActivationSigmoid
	case "tanh":
		return ActivationTanh
	case "softplus":
		return ActivationSoftplus
	case "leaky_relu":
		return ActivationLeakyReLU
	case "linear":
		return ActivationScaledReLU // Default to ReLU
	default:
		return ActivationScaledReLU
	}
}
