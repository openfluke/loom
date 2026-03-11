package nn

// NetworkBlueprint contains the structural information of a network
// extracted after loading.
type NetworkBlueprint struct {
	Models []ModelTelemetry `json:"models"`
}

// ModelTelemetry represents a single network's structure
type ModelTelemetry struct {
	ID          string           `json:"id"`
	TotalLayers int              `json:"total_layers"`
	TotalParams int              `json:"total_parameters"`
	Layers      []LayerTelemetry `json:"layers"`
}

// LayerTelemetry contains metadata about a specific layer
type LayerTelemetry struct {
	// Grid position
	GridRow   int `json:"grid_row"`
	GridCol   int `json:"grid_col"`
	CellLayer int `json:"cell_layer"`

	// Layer info
	Type       string `json:"type"`
	Activation string `json:"activation,omitempty"`
	Parameters int    `json:"parameters"`

	// Dimensions (generic)
	InputShape  []int `json:"input_shape,omitempty"`
	OutputShape []int `json:"output_shape,omitempty"`

	// For nested/parallel layers
	Branches    []LayerTelemetry `json:"branches,omitempty"`
	CombineMode string           `json:"combine_mode,omitempty"` // "concat", "add", "avg", "grid_scatter"
}

// ExtractNetworkBlueprint extracts telemetry data from a loaded network.
func ExtractNetworkBlueprint(n *Network, modelID string) ModelTelemetry {
	telemetry := ModelTelemetry{
		ID:          modelID,
		TotalLayers: len(n.Layers),
		Layers:      make([]LayerTelemetry, 0, len(n.Layers)),
	}

	totalParams := 0

	for i, layerConfig := range n.Layers {
		// Calculate grid position
		rows := n.GridRows
		cols := n.GridCols
		layersPerCell := n.LayersPerCell

		// Handle case where grid dimensions might be 0 (e.g. manually constructed networks)
		if rows == 0 {
			rows = 1
		}
		if cols == 0 {
			cols = 1
		}
		if layersPerCell == 0 {
			layersPerCell = 1
		}

		gridRow := i / (cols * layersPerCell)
		gridCol := (i / layersPerCell) % cols
		cellLayer := i % layersPerCell

		layerTel := extractLayerTelemetry(layerConfig)
		layerTel.GridRow = gridRow
		layerTel.GridCol = gridCol
		layerTel.CellLayer = cellLayer

		telemetry.Layers = append(telemetry.Layers, layerTel)
		totalParams += layerTel.Parameters
	}

	telemetry.TotalParams = totalParams
	return telemetry
}

func extractLayerTelemetry(config LayerConfig) LayerTelemetry {
	tel := LayerTelemetry{
		Type:       layerTypeToString(config.Type),
		Activation: activationToString(config.Activation),
	}

	// Calculate parameters and shapes based on layer type
	params := 0
	switch config.Type {
	case LayerDense:
		// Weights + Biases
		params = (config.InputHeight * config.OutputHeight) + config.OutputHeight
		tel.InputShape = []int{config.InputHeight}
		tel.OutputShape = []int{config.OutputHeight}

	case LayerConv2D:
		// Kernels + Biases
		params = (config.Filters * config.InputChannels * config.KernelSize * config.KernelSize) + config.Filters
		tel.InputShape = []int{config.InputChannels, config.InputHeight, config.InputWidth}
		tel.OutputShape = []int{config.Filters, config.OutputHeight, config.OutputWidth}

	case LayerMultiHeadAttention:
		// Q, K, V, Output weights
		kvDim := config.NumKVHeads * (config.DModel / config.NumHeads)
		if kvDim == 0 {
			kvDim = config.DModel
		} // Fallback

		qParams := config.DModel * config.DModel
		kParams := kvDim * config.DModel
		vParams := kvDim * config.DModel
		outParams := config.DModel * config.DModel

		// Biases
		qBias := config.DModel
		kBias := kvDim
		vBias := kvDim
		outBias := config.DModel

		params = qParams + kParams + vParams + outParams + qBias + kBias + vBias + outBias
		tel.InputShape = []int{config.SeqLength, config.DModel}
		tel.OutputShape = []int{config.SeqLength, config.DModel}

	case LayerRNN:
		// Input + Hidden weights + Bias
		inputWeights := config.RNNInputSize * config.HiddenSize
		hiddenWeights := config.HiddenSize * config.HiddenSize
		bias := config.HiddenSize
		params = inputWeights + hiddenWeights + bias
		tel.InputShape = []int{config.SeqLength, config.RNNInputSize}
		tel.OutputShape = []int{config.SeqLength, config.HiddenSize}

	case LayerLSTM:
		// 4 gates (Input, Forget, Cell, Output)
		// Each has Input weights, Hidden weights, and Biases
		gateParams := (config.RNNInputSize * config.HiddenSize) + (config.HiddenSize * config.HiddenSize) + config.HiddenSize
		params = 4 * gateParams
		tel.InputShape = []int{config.SeqLength, config.RNNInputSize}
		tel.OutputShape = []int{config.SeqLength, config.HiddenSize}

	case LayerNorm:
		params = config.NormSize * 2 // Gamma + Beta
		tel.InputShape = []int{config.NormSize}
		tel.OutputShape = []int{config.NormSize}

	case LayerRMSNorm:
		params = config.NormSize // Gamma only
		tel.InputShape = []int{config.NormSize}
		tel.OutputShape = []int{config.NormSize}

	case LayerSwiGLU:
		// Gate, Up, Down projections
		// Gate: Input -> Intermediate
		// Up: Input -> Intermediate
		// Down: Intermediate -> Input
		gateParams := (config.InputHeight * config.OutputHeight) + config.OutputHeight // + bias
		upParams := (config.InputHeight * config.OutputHeight) + config.OutputHeight
		downParams := (config.OutputHeight * config.InputHeight) + config.InputHeight
		params = gateParams + upParams + downParams
		tel.InputShape = []int{config.InputHeight}
		tel.OutputShape = []int{config.InputHeight} // Output size is essentially input size for the block usually, but strictly it is input size

	case LayerParallel:
		// Sum of branches
		for _, branch := range config.ParallelBranches {
			branchTel := extractLayerTelemetry(branch)
			tel.Branches = append(tel.Branches, branchTel)
			params += branchTel.Parameters
		}
		// Set combine mode (default to concat if empty)
		if config.CombineMode != "" {
			tel.CombineMode = config.CombineMode
		} else {
			tel.CombineMode = "concat"
		}
	}

	tel.Parameters = params
	return tel
}
