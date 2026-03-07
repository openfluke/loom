package poly

import (
	"fmt"
)

// NetworkBlueprint contains the structural information of a network
// extracted after loading or building.
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
	Z int `json:"z"`
	Y int `json:"y"`
	X int `json:"x"`
	L int `json:"l"`

	// Layer info
	Type       string `json:"type"`
	Activation string `json:"activation,omitempty"`
	Parameters int    `json:"parameters"`

	// Dimensions
	InputShape  []int `json:"input_shape,omitempty"`
	OutputShape []int `json:"output_shape,omitempty"`

	// For nested/parallel layers
	Branches    []LayerTelemetry `json:"branches,omitempty"`
	CombineMode string           `json:"combine_mode,omitempty"`
}

// ExtractNetworkBlueprint extracts structural telemetry from a VolumetricNetwork.
func ExtractNetworkBlueprint(n *VolumetricNetwork, modelID string) ModelTelemetry {
	telemetry := ModelTelemetry{
		ID:          modelID,
		TotalLayers: len(n.Layers),
		Layers:      make([]LayerTelemetry, 0, len(n.Layers)),
	}

	totalParams := 0
	for _, layer := range n.Layers {
		tel := ExtractLayerTelemetry(layer)
		telemetry.Layers = append(telemetry.Layers, tel)
		totalParams += tel.Parameters
	}

	telemetry.TotalParams = totalParams
	return telemetry
}

// ExtractLayerTelemetry converts a VolumetricLayer to its telemetry representation.
func ExtractLayerTelemetry(l VolumetricLayer) LayerTelemetry {
	tel := LayerTelemetry{
		Z:          l.Z,
		Y:          l.Y,
		X:          l.X,
		L:          l.L,
		Type:       fmt.Sprintf("%v", l.Type),
		Activation: fmt.Sprintf("%v", l.Activation),
	}

	params := 0
	switch l.Type {
	case LayerDense:
		params = (l.InputHeight * l.OutputHeight)
		tel.InputShape = []int{l.InputHeight}
		tel.OutputShape = []int{l.OutputHeight}

	case LayerCNN1, LayerCNN2, LayerCNN3:
		kSize := l.KernelSize
		if kSize == 0 { kSize = 1 }
		params = l.Filters * l.InputChannels * kSize * kSize
		if l.Type == LayerCNN3 { params *= kSize }
		tel.InputShape = []int{l.InputChannels, l.InputHeight, l.InputWidth}
		tel.OutputShape = []int{l.Filters, l.OutputHeight, l.OutputWidth}

	case LayerConvTransposed1D, LayerConvTransposed2D, LayerConvTransposed3D:
		kSize := l.KernelSize
		if kSize == 0 { kSize = 1 }
		params = l.InputChannels * l.Filters * kSize * kSize
		if l.Type == LayerConvTransposed3D { params *= kSize }
		tel.InputShape = []int{l.InputChannels, l.InputHeight, l.InputWidth}
		tel.OutputShape = []int{l.Filters, l.OutputHeight, l.OutputWidth}

	case LayerMultiHeadAttention:
		hDim := l.HeadDim
		if hDim == 0 && l.NumHeads > 0 { hDim = l.DModel / l.NumHeads }
		kv := l.NumKVHeads * hDim
		params = 2*l.DModel*l.DModel + 2*l.DModel*kv + 2*l.DModel + 2*kv
		tel.InputShape = []int{l.SeqLength, l.DModel}
		tel.OutputShape = []int{l.SeqLength, l.DModel}

	case LayerRNN:
		params = l.InputHeight*l.InputHeight + l.InputHeight*l.InputHeight + l.InputHeight
		tel.InputShape = []int{l.SeqLength, l.InputHeight}
		tel.OutputShape = []int{l.SeqLength, l.InputHeight}

	case LayerLSTM:
		gate := l.InputHeight*l.InputHeight + l.InputHeight*l.InputHeight + l.InputHeight
		params = 4 * gate
		tel.InputShape = []int{l.SeqLength, l.InputHeight}
		tel.OutputShape = []int{l.SeqLength, l.InputHeight}

	case LayerLayerNorm:
		params = l.InputHeight * 2 // Gamma + Beta
		tel.InputShape = []int{l.InputHeight}
		tel.OutputShape = []int{l.InputHeight}

	case LayerRMSNorm:
		params = l.InputHeight
		tel.InputShape = []int{l.InputHeight}
		tel.OutputShape = []int{l.InputHeight}

	case LayerSwiGLU:
		params = l.InputHeight * l.OutputHeight * 3
		tel.InputShape = []int{l.InputHeight}
		tel.OutputShape = []int{l.OutputHeight}

	case LayerEmbedding:
		params = l.InputHeight * l.OutputHeight // vocab * dModel
		tel.InputShape = []int{l.SeqLength}
		tel.OutputShape = []int{l.SeqLength, l.OutputHeight}

	case LayerKMeans:
		params = l.InputHeight * l.OutputHeight // dModel * k
		tel.InputShape = []int{l.InputHeight}
		tel.OutputShape = []int{l.OutputHeight}

	case LayerParallel:
		for _, b := range l.ParallelBranches {
			bt := ExtractLayerTelemetry(b)
			tel.Branches = append(tel.Branches, bt)
			params += bt.Parameters
		}
		tel.CombineMode = l.CombineMode

	case LayerSequential:
		// Usually nested logic or treated as transparent
		tel.InputShape = []int{l.InputHeight}
		tel.OutputShape = []int{l.OutputHeight}

	case LayerSoftmax:
		tel.InputShape = []int{l.InputHeight}
		tel.OutputShape = []int{l.OutputHeight}
	}

	tel.Parameters = params
	return tel
}
