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

	// Dense layer fields (also shared with RNN/LSTM)
	Width      int `json:"width,omitempty"`
	Height     int `json:"height,omitempty"`
	InputSize  int `json:"input_size,omitempty"`  // Used for Dense, RNN, LSTM
	OutputSize int `json:"output_size,omitempty"` // Used for Dense

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

	// Normalization fields
	NormSize int     `json:"norm_size,omitempty"`
	Epsilon  float32 `json:"epsilon,omitempty"`

	// Parallel layer fields
	Branches         []LayerDefinition `json:"branches,omitempty"`
	CombineMode      string            `json:"combine_mode,omitempty"` // "concat", "add", "avg", "grid_scatter"
	GridPositions    []GridPositionDef `json:"grid_positions,omitempty"`
	GridOutputRows   int               `json:"grid_output_rows,omitempty"`
	GridOutputCols   int               `json:"grid_output_cols,omitempty"`
	GridOutputLayers int               `json:"grid_output_layers,omitempty"`
}

// GridPositionDef is the JSON representation of a grid position
type GridPositionDef struct {
	BranchIndex int `json:"branch_index"`
	TargetRow   int `json:"target_row"`
	TargetCol   int `json:"target_col"`
	TargetLayer int `json:"target_layer"`
}

// EncodedWeights stores weights in base64-encoded JSON format
type EncodedWeights struct {
	Format string `json:"fmt"`
	Data   string `json:"data"`
}

// WeightsData represents the actual weight values
// Type field indicates the numeric type used for weights
type WeightsData struct {
	Type   string         `json:"type"`   // "float32", "float64", etc.
	DType  string         `json:"dtype"`  // DType enum value as string (for multi-type support)
	Layers []LayerWeights `json:"layers"`
}

// LayerWeights stores weights for a single layer
// Note: Currently serialized as float32, but generic tensors can be converted on load
type LayerWeights struct {
	// Data type for this layer (optional, defaults to float32)
	DType string `json:"dtype,omitempty"`

	// Dense weights
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

	// LayerNorm / RMSNorm weights
	Gamma []float32 `json:"gamma,omitempty"`
	Beta  []float32 `json:"beta,omitempty"`

	// SwiGLU weights
	GateWeights []float32 `json:"gate_weights,omitempty"`
	UpWeights   []float32 `json:"up_weights,omitempty"`
	DownWeights []float32 `json:"down_weights,omitempty"`
	GateBias    []float32 `json:"gate_bias,omitempty"`
	UpBias      []float32 `json:"up_bias,omitempty"`
	DownBias    []float32 `json:"down_bias,omitempty"`

	// Parallel layer branch weights (recursive)
	BranchWeights []LayerWeights `json:"branch_weights,omitempty"`
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

// serializeBranches recursively serializes parallel layer branches
func serializeBranches(branches []LayerConfig) []LayerDefinition {
	defs := make([]LayerDefinition, len(branches))
	for i, branch := range branches {
		def := LayerDefinition{
			Type:       layerTypeToString(branch.Type),
			Activation: activationToString(branch.Activation),
		}

		switch branch.Type {
		case LayerDense:
			def.InputHeight = branch.InputHeight
			def.OutputHeight = branch.OutputHeight
		case LayerConv2D:
			def.InputChannels = branch.InputChannels
			def.Filters = branch.Filters
			def.KernelSize = branch.KernelSize
			def.Stride = branch.Stride
			def.Padding = branch.Padding
			def.InputHeight = branch.InputHeight
			def.InputWidth = branch.InputWidth
			def.OutputHeight = branch.OutputHeight
			def.OutputWidth = branch.OutputWidth
		case LayerMultiHeadAttention:
			def.DModel = branch.DModel
			def.NumHeads = branch.NumHeads
			def.SeqLength = branch.SeqLength
		case LayerRNN:
			def.InputSize = branch.RNNInputSize
			def.HiddenSize = branch.HiddenSize
			def.SeqLength = branch.SeqLength
		case LayerLSTM:
			def.InputSize = branch.RNNInputSize
			def.HiddenSize = branch.HiddenSize
			def.SeqLength = branch.SeqLength
		case LayerSoftmax:
			def.SoftmaxVariant = softmaxTypeToString(branch.SoftmaxVariant)
			def.SoftmaxRows = branch.SoftmaxRows
			def.SoftmaxCols = branch.SoftmaxCols
			def.Temperature = branch.Temperature
			def.GumbelNoise = branch.GumbelNoise
			def.Mask = branch.Mask
			def.HierarchyLevels = branch.HierarchyLevels
			def.AdaptiveClusters = branch.AdaptiveClusters
			def.MixtureWeights = branch.MixtureWeights
			def.EntmaxAlpha = branch.EntmaxAlpha
		case LayerNorm:
			def.NormSize = branch.NormSize
			def.Epsilon = branch.Epsilon
		case LayerRMSNorm:
			def.NormSize = branch.NormSize
			def.Epsilon = branch.Epsilon
		case LayerSwiGLU:
			def.InputHeight = branch.InputHeight
			def.OutputHeight = branch.OutputHeight
		case LayerParallel:
			def.CombineMode = branch.CombineMode
			def.GridOutputRows = branch.GridOutputRows
			def.GridOutputCols = branch.GridOutputCols
			def.GridOutputLayers = branch.GridOutputLayers

			// Convert GridPosition to GridPositionDef
			for _, gp := range branch.GridPositions {
				def.GridPositions = append(def.GridPositions, GridPositionDef{
					BranchIndex: gp.BranchIndex,
					TargetRow:   gp.TargetRow,
					TargetCol:   gp.TargetCol,
					TargetLayer: gp.TargetLayer,
				})
			}

			def.Branches = serializeBranches(branch.ParallelBranches) // Recursive call
		}

		defs[i] = def
	}
	return defs
}

// serializeBranchWeights recursively serializes weights for parallel layer branches
func serializeBranchWeights(branches []LayerConfig) []LayerWeights {
	weights := make([]LayerWeights, len(branches))
	for i, branch := range branches {
		var w LayerWeights

		switch branch.Type {
		case LayerDense:
			w.Kernel = branch.Kernel
			w.Biases = branch.Bias
		case LayerConv2D:
			w.Kernel = branch.Kernel
			w.ConvBias = branch.Bias
		case LayerMultiHeadAttention:
			w.QWeights = branch.QWeights
			w.KWeights = branch.KWeights
			w.VWeights = branch.VWeights
			w.OutputWeight = branch.OutputWeight
			w.QBias = branch.QBias
			w.KBias = branch.KBias
			w.VBias = branch.VBias
			w.OutputBias = branch.OutputBias
		case LayerRNN:
			w.WeightIH = branch.WeightIH
			w.WeightHH = branch.WeightHH
			w.BiasH = branch.BiasH
		case LayerLSTM:
			w.WeightII = branch.WeightIH_i
			w.WeightIF = branch.WeightIH_f
			w.WeightIG = branch.WeightIH_g
			w.WeightIO = branch.WeightIH_o
			w.WeightHI = branch.WeightHH_i
			w.WeightHF = branch.WeightHH_f
			w.WeightHG = branch.WeightHH_g
			w.WeightHO = branch.WeightHH_o
			w.BiasI = branch.BiasH_i
			w.BiasF = branch.BiasH_f
			w.BiasG = branch.BiasH_g
			w.BiasO = branch.BiasH_o
		case LayerNorm:
			w.Gamma = branch.Gamma
			w.Beta = branch.Beta
		case LayerRMSNorm:
			w.Gamma = branch.Gamma
		case LayerSwiGLU:
			w.GateWeights = branch.GateWeights
			w.UpWeights = branch.UpWeights
			w.DownWeights = branch.DownWeights
			w.GateBias = branch.GateBias
			w.UpBias = branch.UpBias
			w.DownBias = branch.DownBias
		case LayerParallel:
			// Recursively serialize nested branch weights
			w.BranchWeights = serializeBranchWeights(branch.ParallelBranches)
		}

		weights[i] = w
	}
	return weights
}

// deserializeBranches recursively deserializes parallel layer branches with weights
func deserializeBranches(defs []LayerDefinition, weights []LayerWeights) ([]LayerConfig, error) {
	branches := make([]LayerConfig, len(defs))
	for i, def := range defs {
		var config LayerConfig
		var w LayerWeights
		if i < len(weights) {
			w = weights[i]
		}

		switch def.Type {
		case "dense":
			config = LayerConfig{
				Type:         LayerDense,
				Activation:   stringToActivation(def.Activation),
				InputHeight:  def.InputHeight,
				OutputHeight: def.OutputHeight,
				Kernel:       w.Kernel,
				Bias:         w.Biases,
			}
		case "conv2d":
			config = LayerConfig{
				Type:          LayerConv2D,
				Activation:    stringToActivation(def.Activation),
				InputChannels: def.InputChannels,
				Filters:       def.Filters,
				KernelSize:    def.KernelSize,
				Stride:        def.Stride,
				Padding:       def.Padding,
				InputHeight:   def.InputHeight,
				InputWidth:    def.InputWidth,
				OutputHeight:  def.OutputHeight,
				OutputWidth:   def.OutputWidth,
				Kernel:        w.Kernel,
				Bias:          w.ConvBias,
			}
		case "multi_head_attention", "mha":
			config = LayerConfig{
				Type:         LayerMultiHeadAttention,
				DModel:       def.DModel,
				NumHeads:     def.NumHeads,
				SeqLength:    def.SeqLength,
				QWeights:     w.QWeights,
				KWeights:     w.KWeights,
				VWeights:     w.VWeights,
				OutputWeight: w.OutputWeight,
				QBias:        w.QBias,
				KBias:        w.KBias,
				VBias:        w.VBias,
				OutputBias:   w.OutputBias,
			}
		case "rnn":
			config = LayerConfig{
				Type:         LayerRNN,
				Activation:   stringToActivation(def.Activation),
				RNNInputSize: def.InputSize,
				HiddenSize:   def.HiddenSize,
				SeqLength:    def.SeqLength,
				WeightIH:     w.WeightIH,
				WeightHH:     w.WeightHH,
				BiasH:        w.BiasH,
			}
		case "lstm":
			config = LayerConfig{
				Type:         LayerLSTM,
				RNNInputSize: def.InputSize,
				HiddenSize:   def.HiddenSize,
				SeqLength:    def.SeqLength,
				WeightIH_i:   w.WeightII,
				WeightIH_f:   w.WeightIF,
				WeightIH_g:   w.WeightIG,
				WeightIH_o:   w.WeightIO,
				WeightHH_i:   w.WeightHI,
				WeightHH_f:   w.WeightHF,
				WeightHH_g:   w.WeightHG,
				WeightHH_o:   w.WeightHO,
				BiasH_i:      w.BiasI,
				BiasH_f:      w.BiasF,
				BiasH_g:      w.BiasG,
				BiasH_o:      w.BiasO,
			}
		case "softmax":
			config = LayerConfig{
				Type:             LayerSoftmax,
				SoftmaxVariant:   stringToSoftmaxType(def.SoftmaxVariant),
				SoftmaxRows:      def.SoftmaxRows,
				SoftmaxCols:      def.SoftmaxCols,
				Temperature:      def.Temperature,
				GumbelNoise:      def.GumbelNoise,
				Mask:             def.Mask,
				HierarchyLevels:  def.HierarchyLevels,
				AdaptiveClusters: def.AdaptiveClusters,
				MixtureWeights:   def.MixtureWeights,
				EntmaxAlpha:      def.EntmaxAlpha,
			}
		case "layer_norm", "layernorm":
			config = LayerConfig{
				Type:     LayerNorm,
				NormSize: def.NormSize,
				Epsilon:  def.Epsilon,
				Gamma:    w.Gamma,
				Beta:     w.Beta,
			}
		case "rms_norm", "rmsnorm":
			config = LayerConfig{
				Type:     LayerRMSNorm,
				NormSize: def.NormSize,
				Epsilon:  def.Epsilon,
				Gamma:    w.Gamma,
			}
		case "swiglu":
			config = LayerConfig{
				Type:         LayerSwiGLU,
				InputHeight:  def.InputHeight,
				OutputHeight: def.OutputHeight,
				GateWeights:  w.GateWeights,
				UpWeights:    w.UpWeights,
				DownWeights:  w.DownWeights,
				GateBias:     w.GateBias,
				UpBias:       w.UpBias,
				DownBias:     w.DownBias,
			}
		case "parallel":
			// Recursively deserialize nested branches with their weights
			nestedBranches, err := deserializeBranches(def.Branches, w.BranchWeights)
			if err != nil {
				return nil, fmt.Errorf("failed to deserialize nested branches: %w", err)
			}

			// Convert GridPositionDef to GridPosition
			var gridPositions []GridPosition
			for _, gp := range def.GridPositions {
				gridPositions = append(gridPositions, GridPosition{
					BranchIndex: gp.BranchIndex,
					TargetRow:   gp.TargetRow,
					TargetCol:   gp.TargetCol,
					TargetLayer: gp.TargetLayer,
				})
			}

			config = LayerConfig{
				Type:             LayerParallel,
				CombineMode:      def.CombineMode,
				ParallelBranches: nestedBranches,
				GridPositions:    gridPositions,
				GridOutputRows:   def.GridOutputRows,
				GridOutputCols:   def.GridOutputCols,
				GridOutputLayers: def.GridOutputLayers,
			}
		default:
			return nil, fmt.Errorf("unknown branch type: %s", def.Type)
		}

		branches[i] = config
	}
	return branches, nil
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

		case LayerNorm:
			layerDef.NormSize = layerConfig.NormSize
			layerDef.Epsilon = layerConfig.Epsilon
			layerWeights.Gamma = layerConfig.Gamma
			layerWeights.Beta = layerConfig.Beta

		case LayerRMSNorm:
			layerDef.NormSize = layerConfig.NormSize
			layerDef.Epsilon = layerConfig.Epsilon
			layerWeights.Gamma = layerConfig.Gamma
			// RMSNorm has no Beta

		case LayerSwiGLU:
			layerDef.InputHeight = layerConfig.InputHeight   // hidden size
			layerDef.OutputHeight = layerConfig.OutputHeight // intermediate size
			layerWeights.GateWeights = layerConfig.GateWeights
			layerWeights.UpWeights = layerConfig.UpWeights
			layerWeights.DownWeights = layerConfig.DownWeights
			layerWeights.GateBias = layerConfig.GateBias
			layerWeights.UpBias = layerConfig.UpBias
			layerWeights.DownBias = layerConfig.DownBias

		case LayerParallel:
			layerDef.CombineMode = layerConfig.CombineMode
			layerDef.GridOutputRows = layerConfig.GridOutputRows
			layerDef.GridOutputCols = layerConfig.GridOutputCols
			layerDef.GridOutputLayers = layerConfig.GridOutputLayers

			// Convert GridPosition to GridPositionDef
			for _, gp := range layerConfig.GridPositions {
				layerDef.GridPositions = append(layerDef.GridPositions, GridPositionDef{
					BranchIndex: gp.BranchIndex,
					TargetRow:   gp.TargetRow,
					TargetCol:   gp.TargetCol,
					TargetLayer: gp.TargetLayer,
				})
			}

			// Serialize each branch recursively
			layerDef.Branches = serializeBranches(layerConfig.ParallelBranches)
			layerWeights.BranchWeights = serializeBranchWeights(layerConfig.ParallelBranches)
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

		case "layer_norm", "layernorm":
			layerConfig = LayerConfig{
				Type:     LayerNorm,
				NormSize: layerDef.NormSize,
				Epsilon:  layerDef.Epsilon,
				Gamma:    layerWeights.Gamma,
				Beta:     layerWeights.Beta,
			}

		case "rms_norm", "rmsnorm":
			layerConfig = LayerConfig{
				Type:     LayerRMSNorm,
				NormSize: layerDef.NormSize,
				Epsilon:  layerDef.Epsilon,
				Gamma:    layerWeights.Gamma,
				// RMSNorm has no Beta
			}

		case "swiglu":
			layerConfig = LayerConfig{
				Type:         LayerSwiGLU,
				InputHeight:  layerDef.InputHeight,  // hidden size
				OutputHeight: layerDef.OutputHeight, // intermediate size
				GateWeights:  layerWeights.GateWeights,
				UpWeights:    layerWeights.UpWeights,
				DownWeights:  layerWeights.DownWeights,
				GateBias:     layerWeights.GateBias,
				UpBias:       layerWeights.UpBias,
				DownBias:     layerWeights.DownBias,
			}

		case "parallel":
			// Deserialize branches recursively with their weights
			branches, err := deserializeBranches(layerDef.Branches, layerWeights.BranchWeights)
			if err != nil {
				return nil, fmt.Errorf("failed to deserialize parallel branches: %w", err)
			}

			// Convert GridPositionDef to GridPosition
			var gridPositions []GridPosition
			for _, gp := range layerDef.GridPositions {
				gridPositions = append(gridPositions, GridPosition{
					BranchIndex: gp.BranchIndex,
					TargetRow:   gp.TargetRow,
					TargetCol:   gp.TargetCol,
					TargetLayer: gp.TargetLayer,
				})
			}

			layerConfig = LayerConfig{
				Type:             LayerParallel,
				CombineMode:      layerDef.CombineMode,
				ParallelBranches: branches,
				GridPositions:    gridPositions,
				GridOutputRows:   layerDef.GridOutputRows,
				GridOutputCols:   layerDef.GridOutputCols,
				GridOutputLayers: layerDef.GridOutputLayers,
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
	case LayerNorm:
		return "layer_norm"
	case LayerRMSNorm:
		return "rms_norm"
	case LayerSwiGLU:
		return "swiglu"
	case LayerResidual:
		return "residual"
	case LayerParallel:
		return "parallel"
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

// BuildNetworkFromJSON creates a neural network from a JSON configuration string
// This allows building complete neural networks from JSON without manually assigning layers
// The JSON structure matches the NetworkConfig format used in serialization
func BuildNetworkFromJSON(jsonConfig string) (*Network, error) {
	var config NetworkConfig
	if err := json.Unmarshal([]byte(jsonConfig), &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	// Default batch size to 1 if not specified
	batchSize := config.BatchSize
	if batchSize == 0 {
		batchSize = 1
	}

	// Create network with grid structure
	network := NewNetwork(
		batchSize,
		config.GridRows,
		config.GridCols,
		config.LayersPerCell,
	)
	network.BatchSize = batchSize

	// Validate layer count
	expectedLayers := config.GridRows * config.GridCols * config.LayersPerCell
	if len(config.Layers) != expectedLayers {
		return nil, fmt.Errorf("layer count mismatch: expected %d (rows=%d × cols=%d × layers_per_cell=%d), got %d",
			expectedLayers, config.GridRows, config.GridCols, config.LayersPerCell, len(config.Layers))
	}

	// Build each layer from configuration
	for i, layerDef := range config.Layers {
		row := i / (config.GridCols * config.LayersPerCell)
		col := (i / config.LayersPerCell) % config.GridCols
		layer := i % config.LayersPerCell

		layerConfig, err := buildLayerConfig(layerDef)
		if err != nil {
			return nil, fmt.Errorf("failed to build layer %d (row=%d, col=%d, layer=%d): %w",
				i, row, col, layer, err)
		}

		network.SetLayer(row, col, layer, layerConfig)
	}

	return network, nil
}

// BuildNetworkFromFile creates a neural network from a JSON configuration file
func BuildNetworkFromFile(filename string) (*Network, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	return BuildNetworkFromJSON(string(data))
}

// buildLayerConfig constructs a LayerConfig from a LayerDefinition
func buildLayerConfig(def LayerDefinition) (LayerConfig, error) {
	var config LayerConfig

	// Parse activation
	config.Activation = stringToActivation(def.Activation)

	// Build layer based on type
	switch def.Type {
	case "dense":
		config.Type = LayerDense
		// Use InputSize/OutputSize for dense layers if Width/Height aren't specified
		if def.Width > 0 {
			config.InputHeight = def.Width
		} else if def.InputSize > 0 {
			config.InputHeight = def.InputSize
		} else if def.InputHeight > 0 {
			config.InputHeight = def.InputHeight
		}

		if def.Height > 0 {
			config.OutputHeight = def.Height
		} else if def.OutputSize > 0 {
			config.OutputHeight = def.OutputSize
		} else if def.OutputHeight > 0 {
			config.OutputHeight = def.OutputHeight
		}
		// Initialize weights if not provided (random initialization will happen elsewhere)

	case "conv2d":
		config.Type = LayerConv2D
		config.InputChannels = def.InputChannels
		config.Filters = def.Filters
		config.KernelSize = def.KernelSize
		config.Stride = def.Stride
		config.Padding = def.Padding
		config.InputHeight = def.InputHeight
		config.InputWidth = def.InputWidth
		config.OutputHeight = def.OutputHeight
		config.OutputWidth = def.OutputWidth

	case "mha", "multi_head_attention":
		config.Type = LayerMultiHeadAttention
		config.DModel = def.DModel
		config.NumHeads = def.NumHeads
		config.SeqLength = def.SeqLength
		config.HeadDim = def.DModel / def.NumHeads

	case "rnn":
		config.Type = LayerRNN
		config.RNNInputSize = def.InputSize
		config.HiddenSize = def.HiddenSize
		config.SeqLength = def.SeqLength

	case "lstm":
		config.Type = LayerLSTM
		config.RNNInputSize = def.InputSize
		config.HiddenSize = def.HiddenSize
		config.SeqLength = def.SeqLength

	case "softmax":
		config.Type = LayerSoftmax
		config.SoftmaxVariant = stringToSoftmaxType(def.SoftmaxVariant)
		config.SoftmaxRows = def.SoftmaxRows
		config.SoftmaxCols = def.SoftmaxCols
		config.Temperature = def.Temperature
		config.GumbelNoise = def.GumbelNoise
		config.Mask = def.Mask
		config.HierarchyLevels = def.HierarchyLevels
		config.AdaptiveClusters = def.AdaptiveClusters
		config.MixtureWeights = def.MixtureWeights
		config.EntmaxAlpha = def.EntmaxAlpha

	case "layer_norm", "layernorm":
		config.Type = LayerNorm
		config.NormSize = def.NormSize
		if def.Epsilon == 0 {
			config.Epsilon = 1e-5 // Default epsilon
		} else {
			config.Epsilon = def.Epsilon
		}

	case "rms_norm", "rmsnorm":
		config.Type = LayerRMSNorm
		config.NormSize = def.NormSize
		if def.Epsilon == 0 {
			config.Epsilon = 1e-5 // Default epsilon
		} else {
			config.Epsilon = def.Epsilon
		}

	case "swiglu":
		config.Type = LayerSwiGLU
		// Use InputSize/OutputSize for SwiGLU if Width/Height aren't specified
		if def.InputSize > 0 {
			config.InputHeight = def.InputSize
		} else if def.InputHeight > 0 {
			config.InputHeight = def.InputHeight
		}

		if def.OutputSize > 0 {
			config.OutputHeight = def.OutputSize
		} else if def.OutputHeight > 0 {
			config.OutputHeight = def.OutputHeight
		}

	case "residual":
		config.Type = LayerResidual
		// Residual layers typically configured via ResidualSkip field

	case "parallel":
		config.Type = LayerParallel
		config.CombineMode = def.CombineMode
		if config.CombineMode == "" {
			config.CombineMode = "concat" // Default to concatenation
		}

		// Grid scatter specific fields
		config.GridOutputRows = def.GridOutputRows
		config.GridOutputCols = def.GridOutputCols
		config.GridOutputLayers = def.GridOutputLayers

		// Convert GridPositionDef to GridPosition
		if len(def.GridPositions) > 0 {
			config.GridPositions = make([]GridPosition, len(def.GridPositions))
			for i, posDef := range def.GridPositions {
				config.GridPositions[i] = GridPosition{
					BranchIndex: posDef.BranchIndex,
					TargetRow:   posDef.TargetRow,
					TargetCol:   posDef.TargetCol,
					TargetLayer: posDef.TargetLayer,
				}
			}
		}

		// Build branch configurations
		config.ParallelBranches = make([]LayerConfig, len(def.Branches))
		for i, branchDef := range def.Branches {
			branchConfig, err := buildLayerConfig(branchDef)
			if err != nil {
				return config, fmt.Errorf("parallel branch %d: %w", i, err)
			}
			config.ParallelBranches[i] = branchConfig
		}

	default:
		return config, fmt.Errorf("unknown layer type: %s", def.Type)
	}

	return config, nil
}
