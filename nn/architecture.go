package nn

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
)

// =============================================================================
// Dynamic Architecture Generation API
// =============================================================================
// This module provides a public API for dynamically creating diverse neural
// network architectures. Originally developed in test43a_ensemble_fusion_2,
// now promoted to the core nn package.

// BrainType defines the type of layer to use in a brain cell
type BrainType int

const (
	BrainMHA        BrainType = iota // Multi-Head Attention
	BrainLSTM                        // Long Short-Term Memory
	BrainRNN                         // Simple Recurrent Neural Network
	BrainDense                       // Dense/Fully-Connected
	BrainSwiGLU                      // SwiGLU activation variant
	BrainNormDense                   // Normalized Dense
	BrainConv2D                      // 2D Convolutional
	BrainConv1D                      // 1D Convolutional (audio/sequence)
	BrainSoftmax                     // Softmax layer
	BrainNorm                        // Layer Normalization
	BrainRMSNorm                     // RMS Normalization
	BrainResidual                    // Residual/Skip connection
	BrainEmbedding                   // Embedding lookup table
	BrainSequential                  // Sequential (runs sub-layers in sequence)
	BrainParallel                    // Parallel (runs sub-layers and combines)
)

// BrainTypeNames maps brain types to their string names
var BrainTypeNames = []string{
	"MHA", "LSTM", "RNN", "Dense", "SwiGLU", "NormDense",
	"Conv2D", "Conv1D", "Softmax", "Norm", "RMSNorm",
	"Residual", "Embedding", "Sequential", "Parallel",
}

// String returns the name of the brain type
func (bt BrainType) String() string {
	if int(bt) < len(BrainTypeNames) {
		return BrainTypeNames[bt]
	}
	return fmt.Sprintf("BrainType(%d)", bt)
}

// GridShape defines a topology for parallel brain cells
type GridShape struct {
	Rows int
	Cols int
	Name string
}

// NumBrains returns the total number of brain cells in this grid shape
func (gs GridShape) NumBrains() int {
	return gs.Rows * gs.Cols
}

// StandardGridShapes provides common grid configurations
var StandardGridShapes = []GridShape{
	{1, 1, "1x1 Mono"},
	{2, 2, "2x2 Standard"},
	{3, 3, "3x3 Complex"},
	{4, 1, "4x1 Tall"},
	{1, 4, "1x4 Wide"},
	{2, 3, "2x3 Rect"},
	{3, 2, "3x2 Rect"},
	{8, 1, "8x1 Scanner"},
	{6, 4, "6x4 Matrix"},
}

// CombineModeNames lists all supported combine modes for parallel layers
var CombineModeNames = []string{"concat", "add", "avg", "grid_scatter", "filter"}

// ActivationTypeNames maps activation types to names
var ActivationTypeNames = []string{"ScaledReLU", "LeakyReLU", "Tanh", "Softplus", "Sigmoid"}

// ArchConfig defines a complete network architecture configuration
type ArchConfig struct {
	ID           int            `json:"id"`
	Name         string         `json:"name"`
	Species      string         `json:"species"`      // Grid shape name
	MutationStr  string         `json:"mutationStr"`  // Compact description of config
	GridRows     int            `json:"gridRows"`
	GridCols     int            `json:"gridCols"`
	NumBrains    int            `json:"numBrains"`    // GridRows * GridCols
	DModel       int            `json:"dModel"`       // Model dimension
	NumHeads     int            `json:"numHeads"`     // Attention heads (for MHA brains)
	LearningRate float32        `json:"learningRate"`
	BudgetScale  float32        `json:"budgetScale"`
	Activation   ActivationType `json:"activation"`
	CombineMode  string         `json:"combineMode"`  // "concat", "add", "avg", "grid_scatter", "filter"
	Brains       []BrainType    `json:"brains"`       // One per brain cell
	BrainNames   []string       `json:"brainNames"`   // String names for JSON
	InitScale    float32        `json:"initScale"`    // Weight initialization scale
	DType        string         `json:"dtype"`        // Numerical type: "float32", "float64", "int32", "int16", "int8"
}

// ArchGenOptions configures the random architecture generator
type ArchGenOptions struct {
	DModels     []int     // Available model dimensions (default: {64, 64, 64, 32} = 75% D64)
	NumHeads    []int     // Available head counts (default: {4, 8})
	GridShapes  []GridShape // Available grid shapes (default: StandardGridShapes)
	LRMin       float32   // Minimum learning rate (default: 0.0001)
	LRMax       float32   // Maximum learning rate (default: 0.01)
	InitScale   float32   // Weight initialization scale (default: 0.5)
	BudgetMin   float32   // Minimum budget scale (default: 0.5)
	BudgetMax   float32   // Maximum budget scale (default: 1.0)

	// Probability distributions (weights, not percentages - will be normalized)
	// BrainDistribution: probability of each BrainType being selected
	// Default: MHA=30%, LSTM=25%, RNN=15%, Dense=15%, SwiGLU=8%, NormDense=7%
	BrainDistribution []float64

	// CombineDistribution: probability of each combine mode
	// Default: avg=35%, add=30%, concat=20%, grid_scatter=15%
	CombineDistribution []float64

	// ActivationDistribution: probability of each activation type
	// Default: uniform distribution across all 5 types
	ActivationDistribution []float64

	// DTypes: available numerical types for weights (default: float32 only)
	// Options: "float32", "float64", "int32", "int16", "int8"
	DTypes []string

	// DTypeDistribution: probability of each dtype being selected
	// Default: 100% float32
	DTypeDistribution []float64
}

// DefaultArchGenOptions returns options matching test43a's distributions
func DefaultArchGenOptions() *ArchGenOptions {
	return &ArchGenOptions{
		DModels:    []int{64, 64, 64, 32}, // 75% D64, 25% D32
		NumHeads:   []int{4, 8},
		GridShapes: StandardGridShapes,
		LRMin:      0.0001,
		LRMax:      0.01,
		InitScale:  0.5,
		BudgetMin:  0.5,
		BudgetMax:  1.0,
		// Brain distribution: MHA=30%, LSTM=25%, RNN=15%, Dense=15%, SwiGLU=8%, NormDense=7%
		BrainDistribution: []float64{0.30, 0.25, 0.15, 0.15, 0.08, 0.07},
		// Combine mode: avg=35%, add=30%, concat=20%, grid_scatter=15%
		CombineDistribution: []float64{0.35, 0.30, 0.20, 0.15},
		// Activation: uniform
		ActivationDistribution: []float64{0.2, 0.2, 0.2, 0.2, 0.2},
		// DType: default to float32 only
		DTypes:            []string{"float32"},
		DTypeDistribution: []float64{1.0},
	}
}

// =============================================================================
// Brain Creation Functions
// =============================================================================

// InitMHABrain creates a Multi-Head Attention brain layer
func InitMHABrain(dModel, numHeads int, scale float32) LayerConfig {
	headDim := dModel / numHeads
	mha := LayerConfig{
		Type:      LayerMultiHeadAttention,
		DModel:    dModel,
		NumHeads:  numHeads,
		SeqLength: 1,
	}
	mha.QWeights = make([]float32, dModel*dModel)
	mha.KWeights = make([]float32, dModel*dModel)
	mha.VWeights = make([]float32, dModel*dModel)
	mha.OutputWeight = make([]float32, dModel*dModel)
	mha.QBias = make([]float32, dModel)
	mha.KBias = make([]float32, dModel)
	mha.VBias = make([]float32, dModel)
	mha.OutputBias = make([]float32, dModel)

	qkScale := scale / float32(math.Sqrt(float64(headDim)))
	outScale := scale / float32(math.Sqrt(float64(dModel)))
	initRandomWeights(mha.QWeights, qkScale)
	initRandomWeights(mha.KWeights, qkScale)
	initRandomWeights(mha.VWeights, qkScale)
	initRandomWeights(mha.OutputWeight, outScale)
	return mha
}

// InitLSTMBrain creates an LSTM brain layer
func InitLSTMBrain(dModel int, scale float32) LayerConfig {
	lstm := LayerConfig{
		Type:         LayerLSTM,
		RNNInputSize: dModel,
		HiddenSize:   dModel,
		SeqLength:    1,
		OutputHeight: dModel,
	}
	initLSTMBrainWeights(&lstm, scale)
	return lstm
}

// InitRNNBrain creates a simple RNN brain layer
func InitRNNBrain(dModel int, scale float32) LayerConfig {
	rnn := LayerConfig{
		Type:         LayerRNN,
		RNNInputSize: dModel,
		HiddenSize:   dModel,
		SeqLength:    1,
		OutputHeight: dModel,
	}
	initRNNBrainWeights(&rnn, scale)
	return rnn
}

// InitDenseBrain creates a Dense brain layer with specified activation
func InitDenseBrain(dModel int, activation ActivationType, scale float32) LayerConfig {
	dense := InitDenseLayer(dModel, dModel, activation)
	scaleWeights(dense.Kernel, scale)
	return dense
}

// InitSwiGLUBrain creates a SwiGLU-style brain layer
func InitSwiGLUBrain(dModel int, scale float32) LayerConfig {
	dense := InitDenseLayer(dModel, dModel, ActivationLeakyReLU)
	scaleWeights(dense.Kernel, scale*0.7)
	return dense
}

// InitNormDenseBrain creates a normalized Dense brain layer
func InitNormDenseBrain(dModel int, activation ActivationType, scale float32) LayerConfig {
	dense := InitDenseLayer(dModel, dModel, activation)
	scaleWeights(dense.Kernel, scale*0.8)
	return dense
}

// InitConv2DBrain creates a Conv2D brain layer
// Assumes input is square (e.g., 4x4x1 for inputSize=16)
func InitConv2DBrain(inputSize int, scale float32) LayerConfig {
	// Calculate square dimensions from input size
	side := int(math.Sqrt(float64(inputSize)))
	if side*side != inputSize {
		side = 4 // default to 4x4 if not perfect square
	}
	// Conv2D: side x side x 1 with 2 filters, kernel 2x2, stride 1, padding 2
	conv := InitConv2DLayer(side, side, 1, 2, 2, 1, 2, ActivationLeakyReLU)
	return conv
}

// InitConv1DBrain creates a Conv1D brain layer
func InitConv1DBrain(dModel int, scale float32) LayerConfig {
	// Conv1D with kernel size 3, 2 filters
	conv := LayerConfig{
		Type:       LayerConv1D,
		KernelSize: 3,
		Stride:     1,
		Padding:    1,
		Filters:    2,
		Activation: ActivationLeakyReLU,
	}
	// Initialize kernel: filters × kernelSize × inputChannels
	conv.Kernel = make([]float32, 2*3*1)
	for i := range conv.Kernel {
		conv.Kernel[i] = (rand.Float32()*2 - 1) * scale
	}
	conv.Bias = make([]float32, 2)
	return conv
}

// InitSoftmaxBrain creates a Softmax brain layer
func InitSoftmaxBrain(dModel int) LayerConfig {
	return LayerConfig{
		Type:           LayerSoftmax,
		SoftmaxVariant: SoftmaxStandard,
	}
}

// InitNormBrain creates a Layer Normalization brain layer
func InitNormBrain(dModel int, scale float32) LayerConfig {
	norm := LayerConfig{
		Type:     LayerNorm,
		NormSize: dModel,
		Epsilon:  1e-5,
	}
	norm.Gamma = make([]float32, dModel)
	norm.Beta = make([]float32, dModel)
	for i := 0; i < dModel; i++ {
		norm.Gamma[i] = 1.0
		norm.Beta[i] = 0.0
	}
	return norm
}

// InitRMSNormBrain creates an RMS Normalization brain layer
func InitRMSNormBrain(dModel int, scale float32) LayerConfig {
	norm := LayerConfig{
		Type:     LayerRMSNorm,
		NormSize: dModel,
		Epsilon:  1e-5,
	}
	norm.Gamma = make([]float32, dModel)
	for i := 0; i < dModel; i++ {
		norm.Gamma[i] = 1.0
	}
	return norm
}

// InitResidualBrain creates a Residual/Skip connection brain layer
func InitResidualBrain() LayerConfig {
	return LayerConfig{
		Type: LayerResidual,
	}
}

// InitEmbeddingBrain creates an Embedding lookup table brain layer
func InitEmbeddingBrain(vocabSize, embedDim int, scale float32) LayerConfig {
	emb := LayerConfig{
		Type:         LayerEmbedding,
		VocabSize:    vocabSize,
		EmbeddingDim: embedDim,
	}
	emb.EmbeddingWeights = make([]float32, vocabSize*embedDim)
	for i := range emb.EmbeddingWeights {
		emb.EmbeddingWeights[i] = (rand.Float32()*2 - 1) * scale
	}
	return emb
}

// =============================================================================
// Simple Network Builder (for benchmarks and simple architectures)
// =============================================================================

// SimpleNetworkConfig defines configuration for simple single-layer-type networks
type SimpleNetworkConfig struct {
	LayerType    BrainType      // Primary layer type (Dense, Conv2D, RNN, LSTM, MHA)
	InputSize    int            // Input dimension
	HiddenSize   int            // Hidden layer dimension
	OutputSize   int            // Output dimension
	Activation   ActivationType // Activation function
	InitScale    float32        // Weight initialization scale
	NumLayers    int            // Number of hidden layers (default 2)
	DType        DType          // Numerical type (DTypeFloat32, DTypeFloat64, etc.)
}

// StandardDTypes lists all supported numerical types as DType enum values
var StandardDTypeList = []DType{
	DTypeFloat32, DTypeFloat64,
	DTypeInt8, DTypeInt16, DTypeInt32, DTypeInt64,
	DTypeUint8, DTypeUint16, DTypeUint32, DTypeUint64,
}

// StandardDTypes lists all supported numerical types as strings (for compatibility)
var StandardDTypes = []string{
	"float32", "float64",
	"int8", "int16", "int32", "int64",
	"uint8", "uint16", "uint32", "uint64",
}

// DTypeToString converts a DType to its string representation
func DTypeToString(dt DType) string {
	switch dt {
	case DTypeFloat32:
		return "float32"
	case DTypeFloat64:
		return "float64"
	case DTypeFloat16:
		return "float16"
	case DTypeInt8:
		return "int8"
	case DTypeInt16:
		return "int16"
	case DTypeInt32:
		return "int32"
	case DTypeInt64:
		return "int64"
	case DTypeUint8:
		return "uint8"
	case DTypeUint16:
		return "uint16"
	case DTypeUint32:
		return "uint32"
	case DTypeUint64:
		return "uint64"
	default:
		return "float32"
	}
}

// DTypeFromString converts a string to DType
func DTypeFromString(s string) DType {
	switch s {
	case "float32":
		return DTypeFloat32
	case "float64":
		return DTypeFloat64
	case "float16":
		return DTypeFloat16
	case "int8":
		return DTypeInt8
	case "int16":
		return DTypeInt16
	case "int32":
		return DTypeInt32
	case "int64":
		return DTypeInt64
	case "uint8":
		return DTypeUint8
	case "uint16":
		return DTypeUint16
	case "uint32":
		return DTypeUint32
	case "uint64":
		return DTypeUint64
	default:
		return DTypeFloat32
	}
}

// DefaultSimpleConfig returns a default simple network configuration
func DefaultSimpleConfig() SimpleNetworkConfig {
	return SimpleNetworkConfig{
		LayerType:  BrainDense,
		InputSize:  16,
		HiddenSize: 16,
		OutputSize: 1,
		Activation: ActivationLeakyReLU,
		InitScale:  0.5,
		NumLayers:  2,
		DType:      DTypeFloat32,
	}
}

// BuildSimpleNetwork creates a simple network with a single layer type
// If DType is not DTypeFloat32, the network is converted to that type
func BuildSimpleNetwork(config SimpleNetworkConfig) *Network {
	// Build the base float32 network
	var net *Network
	switch config.LayerType {
	case BrainConv2D:
		net = buildSimpleConv2DNetwork(config)
	case BrainRNN:
		net = buildSimpleRNNNetwork(config)
	case BrainLSTM:
		net = buildSimpleLSTMNetwork(config)
	case BrainMHA:
		net = buildSimpleMHANetwork(config)
	case BrainSwiGLU:
		net = buildSimpleSwiGLUNetwork(config)
	case BrainNormDense:
		net = buildSimpleNormDenseNetwork(config)
	case BrainConv1D:
		net = buildSimpleConv1DNetwork(config)
	case BrainResidual:
		net = buildSimpleResidualNetwork(config)
	default:
		net = buildSimpleDenseNetwork(config)
	}

	// Note: DType conversion would happen here if ConvertNetwork was available
	// For now, return the float32 network - type conversion is handled externally

	return net
}

func buildSimpleDenseNetwork(config SimpleNetworkConfig) *Network {
	net := NewNetwork(config.InputSize, 1, 1, 3)
	net.BatchSize = 1

	layer0 := InitDenseLayer(config.InputSize, config.HiddenSize, config.Activation)
	scaleWeights(layer0.Kernel, config.InitScale)
	net.SetLayer(0, 0, 0, layer0)

	layer1 := InitDenseLayer(config.HiddenSize, config.HiddenSize, config.Activation)
	scaleWeights(layer1.Kernel, config.InitScale)
	net.SetLayer(0, 0, 1, layer1)

	layer2 := InitDenseLayer(config.HiddenSize, config.OutputSize, ActivationSigmoid)
	scaleWeights(layer2.Kernel, config.InitScale)
	net.SetLayer(0, 0, 2, layer2)

	return net
}

func buildSimpleConv2DNetwork(config SimpleNetworkConfig) *Network {
	net := NewNetwork(config.InputSize, 1, 1, 2)
	net.BatchSize = 1

	// Calculate square dimensions from input size
	side := int(math.Sqrt(float64(config.InputSize)))
	if side*side != config.InputSize {
		side = 4 // default to 4x4 if not perfect square
	}

	// Conv2D: side x side x 1 with 2 filters, kernel 2x2, stride 1, padding 2
	conv := InitConv2DLayer(side, side, 1, 2, 2, 1, 2, ActivationLeakyReLU)
	net.SetLayer(0, 0, 0, conv)

	// Calculate conv output size: ((side - kernel) / stride + 1)^2 * filters
	// Note: The generic Conv2D implementation currently operates with VALID padding (ignoring the padding arg for shape)
	outSide := (side - 2) / 1 + 1
	convOutputSize := outSide * outSide * 2

	// Dense output layer
	dense := InitDenseLayer(convOutputSize, config.OutputSize, ActivationSigmoid)
	scaleWeights(dense.Kernel, config.InitScale)
	net.SetLayer(0, 0, 1, dense)

	return net
}

func buildSimpleRNNNetwork(config SimpleNetworkConfig) *Network {
	net := NewNetwork(config.InputSize, 1, 1, 3)
	net.BatchSize = 1

	rnn := InitRNNLayer(config.InputSize, config.HiddenSize, 1, 1)
	net.SetLayer(0, 0, 0, rnn)

	layer1 := InitDenseLayer(config.HiddenSize, config.HiddenSize, config.Activation)
	scaleWeights(layer1.Kernel, config.InitScale)
	net.SetLayer(0, 0, 1, layer1)

	layer2 := InitDenseLayer(config.HiddenSize, config.OutputSize, ActivationSigmoid)
	scaleWeights(layer2.Kernel, config.InitScale)
	net.SetLayer(0, 0, 2, layer2)

	return net
}

func buildSimpleLSTMNetwork(config SimpleNetworkConfig) *Network {
	net := NewNetwork(config.InputSize, 1, 1, 3)
	net.BatchSize = 1

	lstm := InitLSTMLayer(config.InputSize, config.HiddenSize, 1, 1)
	net.SetLayer(0, 0, 0, lstm)

	layer1 := InitDenseLayer(config.HiddenSize, config.HiddenSize, config.Activation)
	scaleWeights(layer1.Kernel, config.InitScale)
	net.SetLayer(0, 0, 1, layer1)

	layer2 := InitDenseLayer(config.HiddenSize, config.OutputSize, ActivationSigmoid)
	scaleWeights(layer2.Kernel, config.InitScale)
	net.SetLayer(0, 0, 2, layer2)

	return net
}



func buildSimpleMHANetwork(config SimpleNetworkConfig) *Network {
	net := NewNetwork(config.InputSize, 1, 1, 3)
	net.BatchSize = 1

	// Input projection to dModel
	layer0 := InitDenseLayer(config.InputSize, config.HiddenSize, config.Activation)
	scaleWeights(layer0.Kernel, config.InitScale)
	net.SetLayer(0, 0, 0, layer0)

	// MHA layer (4 heads by default)
	numHeads := 4
	if config.HiddenSize%numHeads != 0 {
		numHeads = 2
	}
	mha := InitMHABrain(config.HiddenSize, numHeads, config.InitScale)
	net.SetLayer(0, 0, 1, mha)

	// Output projection
	layer2 := InitDenseLayer(config.HiddenSize, config.OutputSize, ActivationSigmoid)
	scaleWeights(layer2.Kernel, config.InitScale)
	net.SetLayer(0, 0, 2, layer2)

	return net
}

func buildSimpleSwiGLUNetwork(config SimpleNetworkConfig) *Network {
	net := NewNetwork(config.InputSize, 1, 1, 3)
	net.BatchSize = 1

	// SwiGLU: d_model -> d_model (internally expands to 4*d_model or similar)
	l0 := InitSwiGLUBrain(config.InputSize, config.InitScale)
	net.SetLayer(0, 0, 0, l0)

	l1 := InitSwiGLUBrain(config.HiddenSize, config.InitScale)
	net.SetLayer(0, 0, 1, l1)

	l2 := InitDenseLayer(config.HiddenSize, config.OutputSize, ActivationSigmoid)
	scaleWeights(l2.Kernel, config.InitScale)
	net.SetLayer(0, 0, 2, l2)

	return net
}

func buildSimpleNormDenseNetwork(config SimpleNetworkConfig) *Network {
	net := NewNetwork(config.InputSize, 1, 1, 3)
	net.BatchSize = 1

	l0 := InitNormDenseBrain(config.InputSize, config.Activation, config.InitScale)
	net.SetLayer(0, 0, 0, l0)

	l1 := InitNormDenseBrain(config.HiddenSize, config.Activation, config.InitScale)
	net.SetLayer(0, 0, 1, l1)

	l2 := InitDenseLayer(config.HiddenSize, config.OutputSize, ActivationSigmoid)
	scaleWeights(l2.Kernel, config.InitScale)
	net.SetLayer(0, 0, 2, l2)

	return net
}

func buildSimpleConv1DNetwork(config SimpleNetworkConfig) *Network {
	net := NewNetwork(config.InputSize, 1, 1, 3)
	net.BatchSize = 1

	// Conv1D layer
	l0 := InitConv1DBrain(config.InputSize, config.InitScale)
	net.SetLayer(0, 0, 0, l0)

	// Simple projection to HiddenSize
	l1 := InitDenseLayer(config.HiddenSize, config.HiddenSize, ActivationLeakyReLU)
	scaleWeights(l1.Kernel, config.InitScale)
	net.SetLayer(0, 0, 1, l1)

	l2 := InitDenseLayer(config.HiddenSize, config.OutputSize, ActivationSigmoid)
	scaleWeights(l2.Kernel, config.InitScale)
	net.SetLayer(0, 0, 2, l2)

	return net
}

func buildSimpleResidualNetwork(config SimpleNetworkConfig) *Network {
	net := NewNetwork(config.InputSize, 1, 1, 3)
	net.BatchSize = 1

	// Input projection
	l0 := InitDenseLayer(config.InputSize, config.HiddenSize, ActivationLeakyReLU)
	scaleWeights(l0.Kernel, config.InitScale)
	net.SetLayer(0, 0, 0, l0)

	// Residual connection
	l1 := InitResidualBrain()
	net.SetLayer(0, 0, 1, l1)

	l2 := InitDenseLayer(config.HiddenSize, config.OutputSize, ActivationSigmoid)
	scaleWeights(l2.Kernel, config.InitScale)
	net.SetLayer(0, 0, 2, l2)

	return net
}



// BrainTypeFromString converts a string name to BrainType
func BrainTypeFromString(name string) BrainType {
	for i, n := range BrainTypeNames {
		if strings.EqualFold(n, name) {
			return BrainType(i)
		}
	}
	return BrainDense // default
}

// StandardBrainTypes lists common brain types suitable for simple networks
var StandardBrainTypes = []BrainType{
	BrainDense,
	BrainConv2D,
	BrainRNN,
	BrainLSTM,
	BrainMHA,
}

// AllBrainTypes lists all available brain types
var AllBrainTypes = []BrainType{
	BrainMHA,
	BrainLSTM,
	BrainRNN,
	BrainDense,
	BrainSwiGLU,
	BrainNormDense,
	BrainConv2D,
	BrainConv1D,
	BrainSoftmax,
	BrainNorm,
	BrainRMSNorm,
	BrainResidual,
	BrainEmbedding,
	BrainSequential,
	BrainParallel,
}

// GenerateAllSimpleConfigs generates all permutations of layer types × numerical types
// Returns len(layerTypes) × len(dtypes) configurations
func GenerateAllSimpleConfigs(base SimpleNetworkConfig, layerTypes []BrainType, dtypes []DType) []SimpleNetworkConfig {
	if len(layerTypes) == 0 {
		layerTypes = StandardBrainTypes
	}
	if len(dtypes) == 0 {
		dtypes = StandardDTypeList
	}

	configs := make([]SimpleNetworkConfig, 0, len(layerTypes)*len(dtypes))
	for _, lt := range layerTypes {
		for _, dt := range dtypes {
			cfg := base
			cfg.LayerType = lt
			cfg.DType = dt
			configs = append(configs, cfg)
		}
	}
	return configs
}

// GenerateSimpleConfigs generates configs for specific layer types with all dtypes
func GenerateSimpleConfigs(base SimpleNetworkConfig, layerTypes []BrainType) []SimpleNetworkConfig {
	return GenerateAllSimpleConfigs(base, layerTypes, StandardDTypeList)
}

// =============================================================================
// Hive/Parallel Layer Creation
// =============================================================================

// InitDiverseHive creates a parallel layer with diverse brain types
func InitDiverseHive(config ArchConfig) LayerConfig {
	numBrains := config.GridRows * config.GridCols
	branches := make([]LayerConfig, numBrains)
	positions := make([]GridPosition, numBrains)

	for i := 0; i < numBrains; i++ {
		brainType := config.Brains[i]
		switch brainType {
		case BrainMHA:
			branches[i] = InitMHABrain(config.DModel, config.NumHeads, config.InitScale)
		case BrainLSTM:
			branches[i] = InitLSTMBrain(config.DModel, config.InitScale)
		case BrainRNN:
			branches[i] = InitRNNBrain(config.DModel, config.InitScale)
		case BrainDense:
			branches[i] = InitDenseBrain(config.DModel, config.Activation, config.InitScale)
		case BrainSwiGLU:
			branches[i] = InitSwiGLUBrain(config.DModel, config.InitScale)
		case BrainNormDense:
			branches[i] = InitNormDenseBrain(config.DModel, config.Activation, config.InitScale)
		default:
			branches[i] = InitDenseBrain(config.DModel, config.Activation, config.InitScale)
		}

		row := i / config.GridCols
		col := i % config.GridCols
		positions[i] = GridPosition{
			BranchIndex: i,
			TargetRow:   row,
			TargetCol:   col,
			TargetLayer: 0,
		}
	}

	layer := LayerConfig{
		Type:             LayerParallel,
		CombineMode:      config.CombineMode,
		ParallelBranches: branches,
	}

	// Only set grid positions for grid_scatter mode
	if config.CombineMode == "grid_scatter" {
		layer.GridOutputRows = config.GridRows
		layer.GridOutputCols = config.GridCols
		layer.GridOutputLayers = 1
		layer.GridPositions = positions
	}

	return layer
}

// =============================================================================
// Network Builder
// =============================================================================

// BuildDiverseNetwork creates a complete network from an ArchConfig
func BuildDiverseNetwork(config ArchConfig, inputSize int) *Network {
	totalLayers := 4
	net := NewNetwork(inputSize, 1, 1, totalLayers)
	net.BatchSize = 1

	layerIdx := 0

	// Input layer
	inputLayer := InitDenseLayer(inputSize, config.DModel, config.Activation)
	scaleWeights(inputLayer.Kernel, config.InitScale)
	net.SetLayer(0, 0, layerIdx, inputLayer)
	layerIdx++

	// Parallel hive layer with configurable combine mode
	parallelLayer := InitDiverseHive(config)
	net.SetLayer(0, 0, layerIdx, parallelLayer)
	layerIdx++

	// Merger layer - size depends on combine mode
	var mergerInputSize int
	switch config.CombineMode {
	case "concat", "grid_scatter":
		mergerInputSize = config.DModel * config.GridRows * config.GridCols
	case "add", "avg", "filter":
		mergerInputSize = config.DModel
	default:
		mergerInputSize = config.DModel
	}
	mergerLayer := InitDenseLayer(mergerInputSize, config.DModel, config.Activation)
	scaleWeights(mergerLayer.Kernel, config.InitScale)
	net.SetLayer(0, 0, layerIdx, mergerLayer)
	layerIdx++

	// Output layer
	outputLayer := InitDenseLayer(config.DModel, inputSize, ActivationSigmoid)
	scaleWeights(outputLayer.Kernel, config.InitScale)
	net.SetLayer(0, 0, layerIdx, outputLayer)

	return net
}

// =============================================================================
// Configuration Generator
// =============================================================================

// GenerateDiverseConfigs creates a slice of randomized architecture configurations
// If opts is nil, DefaultArchGenOptions() is used
func GenerateDiverseConfigs(count int, opts *ArchGenOptions) []ArchConfig {
	if opts == nil {
		opts = DefaultArchGenOptions()
	}

	configs := make([]ArchConfig, count)

	for i := 0; i < count; i++ {
		// Random grid shape
		shape := opts.GridShapes[rand.Intn(len(opts.GridShapes))]
		numBrains := shape.NumBrains()

		// Generate brain types
		brains := make([]BrainType, numBrains)
		brainNames := make([]string, numBrains)
		for b := 0; b < numBrains; b++ {
			brains[b] = selectBrainType(opts.BrainDistribution)
			brainNames[b] = brains[b].String()
		}

		// Random DModel and heads
		dModel := opts.DModels[rand.Intn(len(opts.DModels))]
		heads := opts.NumHeads[rand.Intn(len(opts.NumHeads))]
		// Ensure dModel is divisible by heads
		for dModel%heads != 0 {
			heads = opts.NumHeads[rand.Intn(len(opts.NumHeads))]
		}

		// Log-uniform learning rate
		logMin := math.Log(float64(opts.LRMin))
		logMax := math.Log(float64(opts.LRMax))
		lr := float32(math.Exp(logMin + rand.Float64()*(logMax-logMin)))

		// Random activation
		activation := selectActivation(opts.ActivationDistribution)

		// Random combine mode
		combineMode := selectCombineMode(opts.CombineDistribution)

		// Random budget scale
		budgetScale := opts.BudgetMin + float32(rand.Float64())*(opts.BudgetMax-opts.BudgetMin)

		// Random DType
		dtype := "float32" // default
		if len(opts.DTypes) > 0 {
			dtype = selectDType(opts.DTypes, opts.DTypeDistribution)
		}

		// Build mutation string for tracking
		brainStr := strings.Join(brainNames, "-")
		if len(brainStr) > 30 {
			brainStr = brainStr[:27] + "..."
		}
		mutationStr := fmt.Sprintf("%dx%d_%s_%s_D%d_%s_LR%.4f",
			shape.Rows, shape.Cols,
			combineMode,
			ActivationTypeNames[activation],
			dModel, dtype, lr)

		configs[i] = ArchConfig{
			ID:           i,
			Name:         fmt.Sprintf("Net-%d", i),
			Species:      shape.Name,
			MutationStr:  mutationStr,
			GridRows:     shape.Rows,
			GridCols:     shape.Cols,
			NumBrains:    numBrains,
			DModel:       dModel,
			NumHeads:     heads,
			LearningRate: lr,
			BudgetScale:  budgetScale,
			Activation:   activation,
			CombineMode:  combineMode,
			Brains:       brains,
			BrainNames:   brainNames,
			InitScale:    opts.InitScale,
			DType:        dtype,
		}
	}

	return configs
}

// =============================================================================
// Helper Functions
// =============================================================================

// selectBrainType selects a brain type based on probability distribution
func selectBrainType(dist []float64) BrainType {
	// Normalize and select
	r := rand.Float64()
	cumulative := 0.0
	for i, p := range dist {
		cumulative += p
		if r < cumulative {
			return BrainType(i)
		}
	}
	return BrainDense // fallback
}

// selectCombineMode selects a combine mode based on probability distribution
func selectCombineMode(dist []float64) string {
	modes := []string{"avg", "add", "concat", "grid_scatter"}
	r := rand.Float64()
	cumulative := 0.0
	for i, p := range dist {
		cumulative += p
		if r < cumulative && i < len(modes) {
			return modes[i]
		}
	}
	return "avg" // fallback
}

// selectActivation selects an activation type based on probability distribution
func selectActivation(dist []float64) ActivationType {
	r := rand.Float64()
	cumulative := 0.0
	for i, p := range dist {
		cumulative += p
		if r < cumulative {
			return ActivationType(i)
		}
	}
	return ActivationLeakyReLU // fallback
}

// selectDType selects a numerical type based on probability distribution
func selectDType(dtypes []string, dist []float64) string {
	if len(dtypes) == 0 {
		return "float32"
	}
	if len(dist) == 0 {
		return dtypes[rand.Intn(len(dtypes))]
	}
	r := rand.Float64()
	cumulative := 0.0
	for i, p := range dist {
		cumulative += p
		if r < cumulative && i < len(dtypes) {
			return dtypes[i]
		}
	}
	return dtypes[0] // fallback
}

// initRandomWeights fills a slice with random values scaled by scale
func initRandomWeights(weights []float32, scale float32) {
	for i := range weights {
		weights[i] = (rand.Float32()*2 - 1) * scale
	}
}

// scaleWeights multiplies all weights by scale
func scaleWeights(weights []float32, scale float32) {
	for i := range weights {
		weights[i] *= scale
	}
}

// initRNNBrainWeights initializes RNN layer weights
func initRNNBrainWeights(cfg *LayerConfig, scale float32) {
	inputSize := cfg.RNNInputSize
	hiddenSize := cfg.HiddenSize

	cfg.WeightIH = make([]float32, hiddenSize*inputSize)
	cfg.WeightHH = make([]float32, hiddenSize*hiddenSize)
	cfg.BiasH = make([]float32, hiddenSize)

	wScale := scale / float32(math.Sqrt(float64(hiddenSize)))
	initRandomWeights(cfg.WeightIH, wScale)
	initRandomWeights(cfg.WeightHH, wScale)
}

// initLSTMBrainWeights initializes LSTM layer weights
func initLSTMBrainWeights(cfg *LayerConfig, scale float32) {
	inputSize := cfg.RNNInputSize
	hiddenSize := cfg.HiddenSize

	cfg.WeightIH_i = make([]float32, hiddenSize*inputSize)
	cfg.WeightIH_f = make([]float32, hiddenSize*inputSize)
	cfg.WeightIH_g = make([]float32, hiddenSize*inputSize)
	cfg.WeightIH_o = make([]float32, hiddenSize*inputSize)
	cfg.WeightHH_i = make([]float32, hiddenSize*hiddenSize)
	cfg.WeightHH_f = make([]float32, hiddenSize*hiddenSize)
	cfg.WeightHH_g = make([]float32, hiddenSize*hiddenSize)
	cfg.WeightHH_o = make([]float32, hiddenSize*hiddenSize)
	cfg.BiasH_i = make([]float32, hiddenSize)
	cfg.BiasH_f = make([]float32, hiddenSize)
	cfg.BiasH_g = make([]float32, hiddenSize)
	cfg.BiasH_o = make([]float32, hiddenSize)

	wScale := scale / float32(math.Sqrt(float64(hiddenSize)))
	initRandomWeights(cfg.WeightIH_i, wScale)
	initRandomWeights(cfg.WeightIH_f, wScale)
	initRandomWeights(cfg.WeightIH_g, wScale)
	initRandomWeights(cfg.WeightIH_o, wScale)
	initRandomWeights(cfg.WeightHH_i, wScale)
	initRandomWeights(cfg.WeightHH_f, wScale)
	initRandomWeights(cfg.WeightHH_g, wScale)
	initRandomWeights(cfg.WeightHH_o, wScale)
}

// =============================================================================
// Serialization - WASM-Compatible Byte Format
// =============================================================================

// ArchConfigBundle holds multiple ArchConfigs for serialization
type ArchConfigBundle struct {
	Version int          `json:"version"`
	Configs []ArchConfig `json:"configs"`
}

// ToBytes serializes the ArchConfig to bytes (WASM-compatible)
func (ac *ArchConfig) ToBytes() ([]byte, error) {
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(ac); err != nil {
		return nil, fmt.Errorf("failed to encode ArchConfig: %w", err)
	}
	return buf.Bytes(), nil
}

// ArchConfigFromBytes deserializes an ArchConfig from bytes
func ArchConfigFromBytes(data []byte) (*ArchConfig, error) {
	var ac ArchConfig
	if err := json.NewDecoder(bytes.NewReader(data)).Decode(&ac); err != nil {
		return nil, fmt.Errorf("failed to decode ArchConfig: %w", err)
	}
	return &ac, nil
}

// ToBytes serializes the bundle to bytes (WASM-compatible)
func (b *ArchConfigBundle) ToBytes() ([]byte, error) {
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(b); err != nil {
		return nil, fmt.Errorf("failed to encode ArchConfigBundle: %w", err)
	}
	return buf.Bytes(), nil
}

// ArchConfigBundleFromBytes deserializes an ArchConfigBundle from bytes
func ArchConfigBundleFromBytes(data []byte) (*ArchConfigBundle, error) {
	var b ArchConfigBundle
	if err := json.NewDecoder(bytes.NewReader(data)).Decode(&b); err != nil {
		return nil, fmt.Errorf("failed to decode ArchConfigBundle: %w", err)
	}
	return &b, nil
}

// SaveToFile saves the bundle to a file
func (b *ArchConfigBundle) SaveToFile(filename string) error {
	data, err := b.ToBytes()
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

// LoadArchConfigBundle loads an ArchConfigBundle from a file
func LoadArchConfigBundle(filename string) (*ArchConfigBundle, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}
	return ArchConfigBundleFromBytes(data)
}

// ToJSON returns a pretty-printed JSON string (useful for debugging)
func (b *ArchConfigBundle) ToJSON() (string, error) {
	data, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}
