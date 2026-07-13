package poly

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// NetworkSpec represents the top-level JSON structure for a network.
type NetworkSpec struct {
	ID            string      `json:"id"`
	InitSeed      uint64      `json:"init_seed,omitempty"`
	Depth         int         `json:"depth"`
	Rows          int         `json:"rows"`
	Cols          int         `json:"cols"`
	LayersPerCell int         `json:"layers_per_cell"`
	Layers        []LayerSpec `json:"layers"`
}

// LayerSpec represents the JSON structure for a single layer.
type LayerSpec struct {
	// Position
	Z int `json:"z"`
	Y int `json:"y"`
	X int `json:"x"`
	L int `json:"l"`

	// Core Type
	Type       string `json:"type"`
	Activation string `json:"activation"`
	DType      string `json:"dtype"`

	// Dimensions & Config
	InputHeight   int `json:"input_height"`
	InputWidth    int `json:"input_width"`
	InputDepth    int `json:"input_depth"`
	OutputHeight  int `json:"output_height"`
	OutputWidth   int `json:"output_width"`
	OutputDepth   int `json:"output_depth"`
	InputChannels int `json:"input_channels"`
	Filters       int `json:"filters"`
	KernelSize    int `json:"kernel_size"`
	Stride        int `json:"stride"`
	Padding       int `json:"padding"`

	NumHeads   int `json:"num_heads"`
	NumKVHeads int `json:"num_kv_heads"`
	DModel     int `json:"d_model"`
	SeqLength  int `json:"seq_length"`

	VocabSize    int `json:"vocab_size"`
	EmbeddingDim int `json:"embedding_dim"`

	NumClusters int    `json:"num_clusters"`
	OutputMode  string `json:"output_mode"`

	// Recursive structures
	ParallelBranches []LayerSpec `json:"parallel_branches,omitempty"`
	CombineMode      string      `json:"combine_mode,omitempty"`
	SequentialLayers []LayerSpec `json:"sequential_layers,omitempty"`
	IsRemoteLink     bool        `json:"is_remote_link,omitempty"`
	TargetZ          int         `json:"target_z,omitempty"`
	TargetY          int         `json:"target_y,omitempty"`
	TargetX          int         `json:"target_x,omitempty"`
	TargetL          int         `json:"target_l,omitempty"`

	UseTiling bool `json:"use_tiling,omitempty"`
	TileSize  int  `json:"tile_size,omitempty"`

	SoftmaxType     SoftmaxType `json:"softmax_type,omitempty"`
	Temperature     float64     `json:"temperature,omitempty"`
	SoftmaxRows     int         `json:"softmax_rows,omitempty"`
	SoftmaxCols     int         `json:"softmax_cols,omitempty"`
	EntmaxAlpha     float64     `json:"entmax_alpha,omitempty"`
	Mask            []bool      `json:"mask,omitempty"`
	HierarchyLevels []int       `json:"hierarchy_levels,omitempty"`
}

// BuildNetworkFromJSON creates a VolumetricNetwork from a JSON string.
func BuildNetworkFromJSON(jsonData []byte) (*VolumetricNetwork, error) {
	var spec NetworkSpec
	if err := json.Unmarshal(jsonData, &spec); err != nil {
		return nil, fmt.Errorf("unmarshal spec: %v", err)
	}

	net := NewVolumetricNetwork(spec.Depth, spec.Rows, spec.Cols, spec.LayersPerCell)
	net.InitSeed = spec.InitSeed

	for _, ls := range spec.Layers {
		l := net.GetLayer(ls.Z, ls.Y, ls.X, ls.L)
		if err := applyLayerSpec(l, ls); err != nil {
			return nil, fmt.Errorf("layer (%d,%d,%d,%d): %v", ls.Z, ls.Y, ls.X, ls.L, err)
		}
	}

	if net.InitSeed != 0 {
		InitSeededNetwork(net, net.InitSeed)
	}

	net.RefreshRuntimeTileSizes()
	return net, nil
}

func applyLayerSpec(l *VolumetricLayer, ls LayerSpec) error {
	if l == nil {
		return fmt.Errorf("layer at (%d,%d,%d,%d) is nil; check network dimensions", ls.Z, ls.Y, ls.X, ls.L)
	}
	l.Type = ParseLayerType(ls.Type)
	l.Activation = ParseActivationType(ls.Activation)
	l.DType = ParseDType(ls.DType)

	l.InputHeight = ls.InputHeight
	l.InputWidth = ls.InputWidth
	l.InputDepth = ls.InputDepth
	l.OutputHeight = ls.OutputHeight
	l.OutputWidth = ls.OutputWidth
	l.OutputDepth = ls.OutputDepth
	l.InputChannels = ls.InputChannels
	l.Filters = ls.Filters
	l.KernelSize = ls.KernelSize
	l.Stride = ls.Stride
	l.Padding = ls.Padding

	l.NumHeads = ls.NumHeads
	l.NumKVHeads = ls.NumKVHeads
	l.DModel = ls.DModel
	l.SeqLength = ls.SeqLength

	l.VocabSize = ls.VocabSize
	l.EmbeddingDim = ls.EmbeddingDim

	l.NumClusters = ls.NumClusters
	l.KMeansOutputMode = ls.OutputMode

	l.CombineMode = ls.CombineMode
	l.IsRemoteLink = ls.IsRemoteLink
	l.TargetZ = ls.TargetZ
	l.TargetY = ls.TargetY
	l.TargetX = ls.TargetX
	l.TargetL = ls.TargetL

	// Recursive Parallel
	if len(ls.ParallelBranches) > 0 {
		l.ParallelBranches = make([]VolumetricLayer, len(ls.ParallelBranches))
		for i, bs := range ls.ParallelBranches {
			l.ParallelBranches[i].Network = l.Network
			if err := applyLayerSpec(&l.ParallelBranches[i], bs); err != nil {
				return err
			}
		}
	}

	// Recursive Sequential
	if len(ls.SequentialLayers) > 0 {
		l.SequentialLayers = make([]VolumetricLayer, len(ls.SequentialLayers))
		for i, ss := range ls.SequentialLayers {
			l.SequentialLayers[i].Network = l.Network
			if err := applyLayerSpec(&l.SequentialLayers[i], ss); err != nil {
				return err
			}
		}
	}

	l.UseTiling = ls.UseTiling
	l.TileSize = ls.TileSize

	l.SoftmaxType = ls.SoftmaxType
	l.Temperature = ls.Temperature
	l.SoftmaxRows = ls.SoftmaxRows
	l.SoftmaxCols = ls.SoftmaxCols
	l.EntmaxAlpha = ls.EntmaxAlpha
	l.Mask = ls.Mask
	l.HierarchyLevels = ls.HierarchyLevels

	// CNN2: JSON often sets H only; default width from height.
	if l.Type == LayerCNN2 {
		if l.InputWidth <= 0 && l.InputHeight > 0 {
			l.InputWidth = l.InputHeight
		}
		if l.OutputWidth <= 0 && l.OutputHeight > 0 {
			l.OutputWidth = l.OutputHeight
		}
	}

	// CNN3: JSON often sets H/W only; default depth to height so 3D conv is well-defined.
	if l.Type == LayerCNN3 {
		if l.InputDepth <= 0 && l.InputHeight > 0 {
			l.InputDepth = l.InputHeight
		}
		if l.OutputDepth <= 0 && l.OutputHeight > 0 {
			l.OutputDepth = l.OutputHeight
		}
		if l.InputWidth <= 0 && l.InputHeight > 0 {
			l.InputWidth = l.InputHeight
		}
		if l.OutputWidth <= 0 && l.OutputHeight > 0 {
			l.OutputWidth = l.OutputHeight
		}
	}

	// Dynamic weight initialization if possible
	initializeWeights(l)

	return nil
}

// finalizeLayerDimensions applies MHA/SwiGLU dimension defaults without allocating WeightStore.
// Used for .entity topology load (weights arrive later via blob reader).
func finalizeLayerDimensions(l *VolumetricLayer) {
	switch l.Type {
	case LayerMultiHeadAttention:
		if l.NumHeads > 0 && l.HeadDim == 0 {
			l.HeadDim = l.DModel / l.NumHeads
		}
		if l.NumKVHeads == 0 {
			l.NumKVHeads = l.NumHeads
		}
		if l.QueryDim == 0 {
			l.QueryDim = l.DModel
		}
	}
	if l.SeqLength <= 0 {
		l.SeqLength = 1
	}
}

func initializeWeights(l *VolumetricLayer) {
	finalizeLayerDimensions(l)
	var wCount int
	switch l.Type {
	case LayerDense:
		wCount = l.InputHeight * l.OutputHeight
	case LayerRMSNorm:
		wCount = l.InputHeight
	case LayerLayerNorm:
		wCount = 2 * l.InputHeight
	case LayerMultiHeadAttention:
		q := l.QueryDim
		if q == 0 {
			q = l.DModel
		}
		kv := l.NumKVHeads * l.HeadDim
		if kv == 0 {
			kv = l.DModel
		}
		wCount = q*l.DModel + kv*l.DModel + kv*l.DModel + l.DModel*q + q + kv + kv + l.DModel
	case LayerRNN:
		wCount = l.InputHeight*l.OutputHeight + l.OutputHeight*l.OutputHeight + l.OutputHeight
	case LayerLSTM:
		gate := l.InputHeight*l.OutputHeight + l.OutputHeight*l.OutputHeight + l.OutputHeight
		wCount = 4 * gate
	case LayerSwiGLU:
		// gateW + upW + downW + gateB + upB + downB
		wCount = 3*l.InputHeight*l.OutputHeight + 2*l.OutputHeight + l.InputHeight
	case LayerCNN1, LayerCNN2, LayerCNN3:
		k := l.KernelSize
		if k == 0 {
			k = 1
		}
		wCount = l.Filters * l.InputChannels * k
		if l.Type == LayerCNN2 {
			wCount *= k
		}
		if l.Type == LayerCNN3 {
			wCount *= k * k
		}
	case LayerConvTransposed1D, LayerConvTransposed2D, LayerConvTransposed3D:
		k := l.KernelSize
		if k == 0 {
			k = 1
		}
		wCount = l.InputChannels * l.Filters * k
		if l.Type == LayerConvTransposed2D {
			wCount *= k
		}
		if l.Type == LayerConvTransposed3D {
			wCount *= k * k
		}
	case LayerEmbedding:
		wCount = l.VocabSize * l.EmbeddingDim
	case LayerKMeans:
		wCount = l.NumClusters * l.InputHeight
	}

	if wCount > 0 {
		l.WeightStore = NewWeightStore(wCount)
		l.WeightStore.Scale = 1.0
		l.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
	}
}

// ParseLayerType converts a string to a LayerType.
func ParseLayerType(s string) LayerType {
	s = strings.ToUpper(s)
	switch s {
	case "DENSE":
		return LayerDense
	case "MHA", "ATTENTION", "MULTIHEADATTENTION":
		return LayerMultiHeadAttention
	case "SWIGLU":
		return LayerSwiGLU
	case "RMSNORM":
		return LayerRMSNorm
	case "RNN":
		return LayerRNN
	case "LSTM":
		return LayerLSTM
	case "LAYERNORM":
		return LayerLayerNorm
	case "CNN1":
		return LayerCNN1
	case "CNN2":
		return LayerCNN2
	case "CNN3":
		return LayerCNN3
	case "CONVTRANSPOSED1D":
		return LayerConvTransposed1D
	case "CONVTRANSPOSED2D":
		return LayerConvTransposed2D
	case "CONVTRANSPOSED3D":
		return LayerConvTransposed3D
	case "EMBEDDING":
		return LayerEmbedding
	case "KMEANS":
		return LayerKMeans
	case "SOFTMAX":
		return LayerSoftmax
	case "PARALLEL":
		return LayerParallel
	case "RESIDUAL":
		return LayerResidual
	case "SEQUENTIAL":
		return LayerSequential
	case "META", "METACOGNITION":
		return LayerMetacognition
	default:
		return LayerDense
	}
}

// ParseActivationType converts a string to an ActivationType.
func ParseActivationType(s string) ActivationType {
	s = strings.ToUpper(s)
	switch s {
	case "RELU":
		return ActivationReLU
	case "SILU":
		return ActivationSilu
	case "GELU":
		return ActivationGELU
	case "TANH":
		return ActivationTanh
	case "SIGMOID":
		return ActivationSigmoid
	case "LEAKYRELU", "LEAKY_RELU":
		return ActivationLeakyReLU
	case "RELU2":
		return ActivationReLU2
	case "6":
		return ActivationReLU2
	case "LINEAR":
		return ActivationLinear
	default:
		return ActivationLinear
	}
}

// ParseDType converts a string to a DType.
func ParseDType(s string) DType {
	s = strings.ToUpper(s)
	switch s {
	case "FLOAT64", "FP64", "F64":
		return DTypeFloat64
	case "FLOAT32", "FP32", "F32":
		return DTypeFloat32
	case "FLOAT16", "FP16", "F16":
		return DTypeFloat16
	case "BFLOAT16", "BF16":
		return DTypeBFloat16
	case "FP8E4M3", "FP8":
		return DTypeFP8E4M3
	case "FP8E5M2":
		return DTypeFP8E5M2
	case "INT64", "I64":
		return DTypeInt64
	case "INT32", "I32":
		return DTypeInt32
	case "INT16", "I16":
		return DTypeInt16
	case "INT8", "I8":
		return DTypeInt8
	case "UINT64", "U64":
		return DTypeUint64
	case "UINT32", "U32":
		return DTypeUint32
	case "UINT16", "U16":
		return DTypeUint16
	case "UINT8", "U8":
		return DTypeUint8
	case "INT4":
		return DTypeInt4
	case "UINT4":
		return DTypeUint4
	case "FP4", "F4":
		return DTypeFP4
	case "INT2":
		return DTypeInt2
	case "UINT2":
		return DTypeUint2
	case "TERNARY":
		return DTypeTernary
	case "BINARY":
		return DTypeBinary
	default:
		return DTypeFloat32
	}
}
