package nn

import (
	"github.com/openfluke/webgpu/wgpu"
)

// DType defines the numerical type stored in a Tensor
type DType int

const (
	DTypeFloat32 DType = 0 // Standard 32-bit float (default)
	DTypeFloat64 DType = 1 // 64-bit float (high precision)
	DTypeFloat16 DType = 2 // 16-bit float storage (computation upcasts to F32)
	DTypeInt8    DType = 3 // 8-bit int (quantized, requires scale factor)
	DTypeInt16   DType = 4 // 16-bit int
	DTypeInt32   DType = 5 // 32-bit int
	DTypeInt64   DType = 6 // 64-bit int
	DTypeUint8   DType = 7 // 8-bit unsigned int
	DTypeUint16  DType = 8 // 16-bit unsigned int
	DTypeUint32  DType = 9 // 32-bit unsigned int
	DTypeUint64  DType = 10 // 64-bit unsigned int
)

// Numeric is a type constraint for all numeric types that Tensors can hold.
// This enables generic tensor operations across int and float types.
type Numeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64
}

// IsIntegerType checks if T is an integer type
func IsIntegerType[T Numeric]() bool {
	var z T
	switch any(z).(type) {
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return true
	}
	return false
}

// IsFloatType checks if T is a floating-point type
func IsFloatType[T Numeric]() bool {
	var z T
	switch any(z).(type) {
	case float32, float64:
		return true
	}
	return false
}

// Tensor wraps numerical data with metadata for type-agnostic operations.
// It replaces raw []float32 slices to enable multi-precision training.
type Tensor[T Numeric] struct {
	Data    []T     // Underlying data storage
	DType   DType   // Type identifier for runtime checks
	Shape   []int   // Dimensions (e.g., [batch, channels, height, width])
	Strides []int   // Step sizes for each dimension
	Scale   float32 // Quantization scale factor (used only for Int8)
	Offset  int     // Offset into Data for views/slices
}

// NewTensor creates a new tensor with the given shape.
// Data is allocated but not initialized.
func NewTensor[T Numeric](shape ...int) *Tensor[T] {
	size := 1
	for _, dim := range shape {
		size *= dim
	}

	// Compute strides (row-major order)
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return &Tensor[T]{
		Data:    make([]T, size),
		Shape:   shape,
		Strides: strides,
		Scale:   1.0,
	}
}

// NewTensorFromSlice creates a tensor from existing data.
// The slice is used directly (not copied) for efficiency.
func NewTensorFromSlice[T Numeric](data []T, shape ...int) *Tensor[T] {
	// Validate size matches
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	if len(data) < size {
		// Extend data if needed
		extended := make([]T, size)
		copy(extended, data)
		data = extended
	}

	// Compute strides
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return &Tensor[T]{
		Data:    data,
		Shape:   shape,
		Strides: strides,
		Scale:   1.0,
	}
}

// Size returns the total number of elements in the tensor.
func (t *Tensor[T]) Size() int {
	size := 1
	for _, dim := range t.Shape {
		size *= dim
	}
	return size
}

// Clone creates a deep copy of the tensor.
func (t *Tensor[T]) Clone() *Tensor[T] {
	dataCopy := make([]T, len(t.Data))
	copy(dataCopy, t.Data)

	shapeCopy := make([]int, len(t.Shape))
	copy(shapeCopy, t.Shape)

	stridesCopy := make([]int, len(t.Strides))
	copy(stridesCopy, t.Strides)

	return &Tensor[T]{
		Data:    dataCopy,
		DType:   t.DType,
		Shape:   shapeCopy,
		Strides: stridesCopy,
		Scale:   t.Scale,
		Offset:  t.Offset,
	}
}

// Reshape returns a new tensor with a different shape but same data.
// The total size must remain the same.
func (t *Tensor[T]) Reshape(shape ...int) *Tensor[T] {
	newSize := 1
	for _, dim := range shape {
		newSize *= dim
	}
	if newSize != t.Size() {
		return nil // Invalid reshape
	}

	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return &Tensor[T]{
		Data:    t.Data,
		DType:   t.DType,
		Shape:   shape,
		Strides: strides,
		Scale:   t.Scale,
		Offset:  t.Offset,
	}
}

// ActivationType defines the activation function used in a layer
type ActivationType int

const (
	ActivationScaledReLU ActivationType = 0 // v * 1.1, then ReLU
	ActivationSigmoid    ActivationType = 1 // 1 / (1 + exp(-v))
	ActivationTanh       ActivationType = 2 // tanh(v)
	ActivationSoftplus   ActivationType = 3 // log(1 + exp(v))
	ActivationLeakyReLU  ActivationType = 4 // v if v >= 0, else v * 0.1
)

// LayerType defines the type of neural network layer
type LayerType int

const (
	LayerDense              LayerType = 0  // Dense/Fully-connected layer (element-wise activation)
	LayerConv2D             LayerType = 1  // 2D Convolutional layer
	LayerMultiHeadAttention LayerType = 2  // Multi-Head Attention layer
	LayerRNN                LayerType = 3  // Recurrent Neural Network layer
	LayerLSTM               LayerType = 4  // Long Short-Term Memory layer
	LayerSoftmax            LayerType = 5  // Softmax layer with multiple variants
	LayerNorm               LayerType = 6  // Layer Normalization
	LayerResidual           LayerType = 7  // Residual/Skip connection (adds stored input)
	LayerRMSNorm            LayerType = 8  // RMS Normalization (Llama-style, no beta)
	LayerSwiGLU             LayerType = 9  // SwiGLU gated activation (gate_proj * silu(up_proj))
	LayerParallel           LayerType = 10 // Parallel layer (runs multiple sub-layers and concatenates outputs)
	LayerEmbedding          LayerType = 11 // Embedding lookup table (token/position -> vector)
	LayerConv1D             LayerType = 12 // 1D Convolutional layer (for audio/sequence data)
	LayerSequential         LayerType = 13 // Sequential layer (runs multiple sub-layers in sequence)
)

// SoftmaxType defines the variant of softmax to use
type SoftmaxType int

const (
	SoftmaxStandard     SoftmaxType = 0 // Standard softmax: one distribution
	SoftmaxGrid         SoftmaxType = 1 // Grid softmax: independent distributions per row
	SoftmaxHierarchical SoftmaxType = 2 // Hierarchical: nested softmax levels
	SoftmaxTemperature  SoftmaxType = 3 // Temperature-scaled softmax
	SoftmaxGumbel       SoftmaxType = 4 // Gumbel softmax (adds noise)
	SoftmaxMasked       SoftmaxType = 5 // Masked softmax (ignores certain positions)
	SoftmaxSparse       SoftmaxType = 6 // Sparsemax (can output exact zeros)
	SoftmaxAdaptive     SoftmaxType = 7 // Adaptive softmax (for large vocabularies)
	SoftmaxMixture      SoftmaxType = 8 // Mixture of softmaxes
	SoftmaxEntmax       SoftmaxType = 9 // Entmax (generalization of softmax/sparsemax)
)

// LayerConfig holds configuration for a specific layer in the grid
type LayerConfig struct {
	Type       LayerType
	Activation ActivationType

	// Conv2D specific parameters
	KernelSize int       // Size of convolution kernel (e.g., 3 for 3x3)
	Stride     int       // Stride for convolution
	Padding    int       // Padding for convolution
	Filters    int       // Number of output filters/channels
	Kernel     []float32 // Convolution kernel weights [filters][inChannels][kernelH][kernelW]
	Bias       []float32 // Bias terms [filters]

	// Shape information (for Conv2D)
	InputHeight   int
	InputWidth    int
	InputChannels int
	OutputHeight  int
	OutputWidth   int

	// Multi-Head Attention specific parameters
	NumHeads     int       // Number of attention heads (query heads)
	NumKVHeads   int       // Number of key/value heads (for GQA, 0 = same as NumHeads)
	HeadDim      int       // Dimension per head (dModel / numHeads)
	DModel       int       // Model dimension (embedding size)
	SeqLength    int       // Sequence length
	QWeights     []float32 // Query projection weights [dModel][dModel]
	KWeights     []float32 // Key projection weights [dModel][dModel or smaller for GQA]
	VWeights     []float32 // Value projection weights [dModel][dModel or smaller for GQA]
	OutputWeight []float32 // Output projection weights [dModel][dModel]
	QBias        []float32 // Query bias [dModel]
	KBias        []float32 // Key bias [dModel or smaller for GQA]
	VBias        []float32 // Value bias [dModel or smaller for GQA]
	OutputBias   []float32 // Output bias [dModel]

	// RNN/LSTM specific parameters
	HiddenSize   int       // Hidden state size
	RNNInputSize int       // Input feature size (different from network InputSize)
	WeightIH     []float32 // Input-to-hidden weights [hiddenSize][inputSize]
	WeightHH     []float32 // Hidden-to-hidden weights [hiddenSize][hiddenSize]
	BiasH        []float32 // Hidden bias [hiddenSize]

	// LSTM specific parameters (gates: i=input, f=forget, g=cell, o=output)
	WeightIH_i []float32 // Input gate: input-to-hidden [hiddenSize][inputSize]
	WeightHH_i []float32 // Input gate: hidden-to-hidden [hiddenSize][hiddenSize]
	BiasH_i    []float32 // Input gate bias [hiddenSize]

	WeightIH_f []float32 // Forget gate: input-to-hidden [hiddenSize][inputSize]
	WeightHH_f []float32 // Forget gate: hidden-to-hidden [hiddenSize][hiddenSize]
	BiasH_f    []float32 // Forget gate bias [hiddenSize]

	WeightIH_g []float32 // Cell gate: input-to-hidden [hiddenSize][inputSize]
	WeightHH_g []float32 // Cell gate: hidden-to-hidden [hiddenSize][hiddenSize]
	BiasH_g    []float32 // Cell gate bias [hiddenSize]

	WeightIH_o []float32 // Output gate: input-to-hidden [hiddenSize][inputSize]
	WeightHH_o []float32 // Output gate: hidden-to-hidden [hiddenSize][hiddenSize]
	BiasH_o    []float32 // Output gate bias [hiddenSize]

	// Softmax specific parameters
	SoftmaxVariant   SoftmaxType // Which softmax variant to use
	SoftmaxRows      int         // For grid softmax: number of rows (agents/groups)
	SoftmaxCols      int         // For grid softmax: number of columns (actions per row)
	Temperature      float32     // For temperature softmax (default 1.0)
	GumbelNoise      bool        // For Gumbel softmax: whether to add noise
	Mask             []bool      // For masked softmax: which positions to include
	HierarchyLevels  []int       // For hierarchical softmax: sizes at each level [strategies, units, actions]
	AdaptiveClusters [][]int     // For adaptive softmax: item indices per cluster
	MixtureWeights   []float32   // For mixture softmax: weights for each component
	EntmaxAlpha      float32     // For entmax: alpha parameter (1.0=softmax, 2.0=sparsemax)

	// LayerNorm specific parameters
	NormSize int       // Size of the normalization dimension
	Gamma    []float32 // Scale parameters [normSize]
	Beta     []float32 // Shift parameters [normSize]
	Epsilon  float32   // Small constant for numerical stability (default 1e-5)

	// SwiGLU specific parameters (gated FFN: down_proj(silu(gate_proj(x)) * up_proj(x)))
	GateWeights []float32 // Gate projection weights [intermediate][hidden]
	UpWeights   []float32 // Up projection weights [intermediate][hidden]
	DownWeights []float32 // Down projection weights [hidden][intermediate]
	GateBias    []float32 // Gate bias [intermediate]
	UpBias      []float32 // Up bias [intermediate]
	DownBias    []float32 // Down bias [hidden]

	// Residual connection
	ResidualSkip int // How many layers back to skip for residual (0 = no residual)

	// Embedding layer specific parameters
	VocabSize        int       // Size of vocabulary (number of unique tokens)
	EmbeddingDim     int       // Dimension of embedding vectors
	EmbeddingWeights []float32 // Embedding lookup table [VocabSize * EmbeddingDim]

	// Conv1D specific parameters (for audio/sequence data)
	Conv1DFilters     int       // Number of output filters
	Conv1DKernelSize  int       // Size of 1D kernel
	Conv1DStride      int       // Stride for convolution
	Conv1DPadding     int       // Padding for convolution
	Conv1DKernel      []float32 // Kernel weights [filters][inChannels][kernelSize]
	Conv1DBias        []float32 // Bias terms [filters]
	Conv1DInChannels  int       // Input channels

	// Parallel layer specific parameters
	ParallelBranches []LayerConfig  // Sub-layers to run in parallel
	CombineMode      string         // How to combine outputs: "concat", "add", "avg", "grid_scatter"
	GridPositions    []GridPosition // For grid_scatter: where to place each branch output
	GridOutputRows   int            // For grid_scatter: output grid dimensions
	GridOutputCols   int
	GridOutputLayers int

	// Filter combine mode (gated parallel / mixture of experts)
	FilterGateConfig  *LayerConfig // Gate layer to compute routing weights (Dense, MHA, etc.)
	FilterSoftmax     SoftmaxType  // Softmax variant for gating (default: SoftmaxStandard)
	FilterTemperature float32      // Temperature for softmax (lower = sharper selection)

	// Observer for debugging/recording (nil = no observation)
	Observer LayerObserver

	// Grid position (set by Network when layer is accessed)
	GridRow   int    // Row in the grid
	GridCol   int    // Column in the grid
	CellLayer int    // Layer index within the cell
	ModelID   string // Identifier for the model (for multi-model visualization)

	// Pruning support
	IsDisabled bool // If true, this layer acts as an identity function (pass-through)

	// Training control
	Frozen bool // If true, weights in this layer will not be updated during training
}

// GridPosition specifies where a parallel branch output should be placed in the grid
type GridPosition struct {
	BranchIndex int // Which branch this position is for
	TargetRow   int // Grid row to place output
	TargetCol   int // Grid column to place output
	TargetLayer int // Layer index within that cell
}

// LayerStats contains summary statistics for layer activity
type LayerStats struct {
	AvgActivation float32 // Mean activation value
	MaxActivation float32 // Maximum activation value
	MinActivation float32 // Minimum activation value
	ActiveNeurons int     // Count of neurons with activation > threshold
	TotalNeurons  int     // Total neuron count
	LayerType     string  // "dense", "conv2d", "attention", etc.
}

// LayerEvent represents an event during forward/backward pass
type LayerEvent struct {
	Mode      string     `json:"mode"` // "normal" or "step"
	Type      string     // "forward", "backward"
	LayerIdx  int        // Which layer in the network (flattened index)
	LayerType LayerType  // Type of layer
	Stats     LayerStats // Summary statistics
	Input     []float32  // Input data (optional, can be nil to save memory)
	Output    []float32  // Output data (optional, can be nil to save memory)
	StepCount uint64     `json:"step_count"` // For step-based execution

	// Grid position info for visualization
	GridRow   int    `json:"grid_row"`   // Row in the grid
	GridCol   int    `json:"grid_col"`   // Column in the grid
	CellLayer int    `json:"cell_layer"` // Layer index within the cell
	ModelID   string `json:"model_id"`   // Identifier for the model

	// Branch tracking for parallel layers
	BranchIdx        int  `json:"branch_idx"`         // Which branch within parallel layer (-1 if not a branch)
	IsParallelBranch bool `json:"is_parallel_branch"` // True if this is a branch inside a parallel layer
}

// LayerObserver receives events during network execution
// Implement this interface for console logging, HTTP streaming, visualization, etc.
type LayerObserver interface {
	// OnForward is called after a layer's forward pass completes
	OnForward(event LayerEvent)
	// OnBackward is called after a layer's backward pass completes
	OnBackward(event LayerEvent)
}

// Network represents a grid neural network
// Data flows through a 2D grid of cells, where each cell contains multiple layers
type Network struct {
	GridRows      int // Number of rows in the grid
	GridCols      int // Number of columns in the grid
	LayersPerCell int // Number of layers per grid cell
	InputSize     int // Total input size
	BatchSize     int // Batch size for Conv2D layers
	deviceInfo    *GPUDeviceInfo

	// Layer configuration for each position in the grid
	// Indexed by flattened position: row*GridCols*LayersPerCell + col*LayersPerCell + layer
	Layers []LayerConfig

	// Storage for intermediate activations (needed for backward pass)
	// activations[0] = input, activations[i] = output of layer i-1
	activations [][]float32

	// Storage for pre-activation values (needed for derivatives)
	preActivations [][]float32

	// Gradient storage for kernel weights (Conv2D layers)
	kernelGradients [][]float32
	biasGradients   [][]float32

	// Learning rate for parallel layer branch updates (set during UpdateWeights)
	learningRate float32

	// Optimizer (optional - if nil, uses manual gradient application)
	optimizer     Optimizer
	optimizerType string
}

// GPUDeviceInfo holds WebGPU resources for GPU execution
type GPUDeviceInfo struct {
	Device     *wgpu.Device
	Queue      *wgpu.Queue
	WorkgroupX uint32
	release    func()

	// Cached GPU pipelines to avoid recreating them every forward/backward pass
	forwardPipelines  []*wgpu.ComputePipeline
	forwardBGLs       []*wgpu.BindGroupLayout
	backwardPipelines []*wgpu.ComputePipeline
	backwardBGLs      []*wgpu.BindGroupLayout
}

// NewNetwork creates a new grid neural network with dense layers
// gridRows: number of rows in the grid
// gridCols: number of columns in the grid
// layersPerCell: number of layers in each grid cell
// inputSize: batch size of input data
func NewNetwork(inputSize, gridRows, gridCols, layersPerCell int) *Network {
	totalLayers := gridRows * gridCols * layersPerCell

	// Create default dense layers with cycling activations
	layers := make([]LayerConfig, totalLayers)
	for i := 0; i < totalLayers; i++ {
		row := i / (gridCols * layersPerCell)
		remainder := i % (gridCols * layersPerCell)
		col := remainder / layersPerCell
		layer := remainder % layersPerCell

		cellIdx := row*gridCols + col
		activation := (cellIdx*layersPerCell + layer) % 5

		layers[i] = LayerConfig{
			Type:       LayerDense,
			Activation: ActivationType(activation),
		}
	}

	return &Network{
		GridRows:        gridRows,
		GridCols:        gridCols,
		LayersPerCell:   layersPerCell,
		InputSize:       inputSize,
		BatchSize:       1, // Default batch size
		Layers:          layers,
		activations:     make([][]float32, totalLayers+1), // +1 for input
		preActivations:  make([][]float32, totalLayers),
		kernelGradients: make([][]float32, totalLayers),
		biasGradients:   make([][]float32, totalLayers),
	}
}

// TotalLayers returns the total number of layers in the grid
func (n *Network) TotalLayers() int {
	return n.GridRows * n.GridCols * n.LayersPerCell
}

// GetLayer returns the layer configuration for a specific position in the grid
func (n *Network) GetLayer(row, col, layer int) *LayerConfig {
	idx := row*n.GridCols*n.LayersPerCell + col*n.LayersPerCell + layer
	if idx >= 0 && idx < len(n.Layers) {
		return &n.Layers[idx]
	}
	return nil
}

// SetLayer sets the layer configuration for a specific position in the grid
func (n *Network) SetLayer(row, col, layer int, config LayerConfig) {
	idx := row*n.GridCols*n.LayersPerCell + col*n.LayersPerCell + layer
	if idx >= 0 && idx < len(n.Layers) {
		n.Layers[idx] = config
	}
}

// GetActivation returns the activation function for a specific position in the grid
// (For backward compatibility with dense-only code)
func (n *Network) GetActivation(row, col, layer int) ActivationType {
	layerCfg := n.GetLayer(row, col, layer)
	if layerCfg != nil {
		return layerCfg.Activation
	}
	cellIdx := row*n.GridCols + col
	activation := (cellIdx*n.LayersPerCell + layer) % 5
	return ActivationType(activation)
}

// Activations returns the activation values for all layers
func (n *Network) Activations() [][]float32 {
	return n.activations
}

// KernelGradients returns the kernel gradients for all layers
func (n *Network) KernelGradients() [][]float32 {
	return n.kernelGradients
}

// BiasGradients returns the bias gradients for all layers
func (n *Network) BiasGradients() [][]float32 {
	return n.biasGradients
}

// ReleaseGPU releases GPU resources
func (n *Network) ReleaseGPU() {
	if n.deviceInfo != nil {
		// Clean up cached pipelines
		if n.deviceInfo.forwardPipelines != nil {
			for _, p := range n.deviceInfo.forwardPipelines {
				if p != nil {
					p.Release()
				}
			}
			n.deviceInfo.forwardPipelines = nil
		}
		if n.deviceInfo.forwardBGLs != nil {
			for _, bgl := range n.deviceInfo.forwardBGLs {
				if bgl != nil {
					bgl.Release()
				}
			}
			n.deviceInfo.forwardBGLs = nil
		}
		if n.deviceInfo.backwardPipelines != nil {
			for _, p := range n.deviceInfo.backwardPipelines {
				if p != nil {
					p.Release()
				}
			}
			n.deviceInfo.backwardPipelines = nil
		}
		if n.deviceInfo.backwardBGLs != nil {
			for _, bgl := range n.deviceInfo.backwardBGLs {
				if bgl != nil {
					bgl.Release()
				}
			}
			n.deviceInfo.backwardBGLs = nil
		}

		if n.deviceInfo.release != nil {
			n.deviceInfo.release()
		}
		n.deviceInfo = nil
	}
}

// InitializeWeights initializes all trainable weights in the network with random values
func (n *Network) InitializeWeights() {
	for i := 0; i < n.TotalLayers(); i++ {
		row := i / (n.GridCols * n.LayersPerCell)
		col := (i / n.LayersPerCell) % n.GridCols
		layer := i % n.LayersPerCell

		cfg := n.GetLayer(row, col, layer)
		if cfg == nil {
			continue
		}

		switch cfg.Type {
		case LayerDense:
			// Xavier initialization for dense layers
			if cfg.InputHeight > 0 && cfg.OutputHeight > 0 {
				numWeights := cfg.InputHeight * cfg.OutputHeight
				cfg.Kernel = make([]float32, numWeights)
				scale := float32(1.0) / float32(cfg.InputHeight)
				for j := range cfg.Kernel {
					cfg.Kernel[j] = (randomFloat()*2 - 1) * scale
				}
				cfg.Bias = make([]float32, cfg.OutputHeight)
				for j := range cfg.Bias {
					cfg.Bias[j] = randomFloat() * 0.01
				}
			}

		case LayerConv2D:
			// He initialization for conv layers
			if cfg.Filters > 0 && cfg.InputChannels > 0 && cfg.KernelSize > 0 {
				kernelSize := cfg.Filters * cfg.InputChannels * cfg.KernelSize * cfg.KernelSize
				cfg.Kernel = make([]float32, kernelSize)
				scale := float32(1.0) / float32(cfg.InputChannels*cfg.KernelSize*cfg.KernelSize)
				for j := range cfg.Kernel {
					cfg.Kernel[j] = (randomFloat()*2 - 1) * scale
				}
				cfg.Bias = make([]float32, cfg.Filters)
				for j := range cfg.Bias {
					cfg.Bias[j] = randomFloat() * 0.01
				}
			}

		case LayerMultiHeadAttention:
			// Small random initialization for attention
			if cfg.DModel > 0 {
				size := cfg.DModel * cfg.DModel
				cfg.QWeights = make([]float32, size)
				cfg.KWeights = make([]float32, size)
				cfg.VWeights = make([]float32, size)
				cfg.OutputWeight = make([]float32, size)
				for j := range cfg.QWeights {
					cfg.QWeights[j] = randomFloat()*0.2 - 0.1
					cfg.KWeights[j] = randomFloat()*0.2 - 0.1
					cfg.VWeights[j] = randomFloat()*0.2 - 0.1
					cfg.OutputWeight[j] = randomFloat()*0.2 - 0.1
				}
				cfg.QBias = make([]float32, cfg.DModel)
				cfg.KBias = make([]float32, cfg.DModel)
				cfg.VBias = make([]float32, cfg.DModel)
				cfg.OutputBias = make([]float32, cfg.DModel)
			}

		case LayerRNN:
			// Small random initialization for RNN
			if cfg.RNNInputSize > 0 && cfg.HiddenSize > 0 {
				cfg.WeightIH = make([]float32, cfg.HiddenSize*cfg.RNNInputSize)
				cfg.WeightHH = make([]float32, cfg.HiddenSize*cfg.HiddenSize)
				for j := range cfg.WeightIH {
					cfg.WeightIH[j] = randomFloat()*0.2 - 0.1
				}
				for j := range cfg.WeightHH {
					cfg.WeightHH[j] = randomFloat()*0.2 - 0.1
				}
				cfg.BiasH = make([]float32, cfg.HiddenSize)
			}

		case LayerLSTM:
			// LSTM weights for 4 gates
			if cfg.RNNInputSize > 0 && cfg.HiddenSize > 0 {
				ihSize := cfg.HiddenSize * cfg.RNNInputSize
				hhSize := cfg.HiddenSize * cfg.HiddenSize

				cfg.WeightIH_i = make([]float32, ihSize)
				cfg.WeightIH_f = make([]float32, ihSize)
				cfg.WeightIH_g = make([]float32, ihSize)
				cfg.WeightIH_o = make([]float32, ihSize)

				cfg.WeightHH_i = make([]float32, hhSize)
				cfg.WeightHH_f = make([]float32, hhSize)
				cfg.WeightHH_g = make([]float32, hhSize)
				cfg.WeightHH_o = make([]float32, hhSize)

				for j := 0; j < ihSize; j++ {
					cfg.WeightIH_i[j] = randomFloat()*0.2 - 0.1
					cfg.WeightIH_f[j] = randomFloat()*0.2 - 0.1
					cfg.WeightIH_g[j] = randomFloat()*0.2 - 0.1
					cfg.WeightIH_o[j] = randomFloat()*0.2 - 0.1
				}
				for j := 0; j < hhSize; j++ {
					cfg.WeightHH_i[j] = randomFloat()*0.2 - 0.1
					cfg.WeightHH_f[j] = randomFloat()*0.2 - 0.1
					cfg.WeightHH_g[j] = randomFloat()*0.2 - 0.1
					cfg.WeightHH_o[j] = randomFloat()*0.2 - 0.1
				}

				cfg.BiasH_i = make([]float32, cfg.HiddenSize)
				cfg.BiasH_f = make([]float32, cfg.HiddenSize)
				cfg.BiasH_g = make([]float32, cfg.HiddenSize)
				cfg.BiasH_o = make([]float32, cfg.HiddenSize)
			}

		case LayerSwiGLU:
			// SwiGLU weights
			if cfg.InputHeight > 0 && cfg.OutputHeight > 0 {
				gateSize := cfg.InputHeight * cfg.OutputHeight
				downSize := cfg.OutputHeight * cfg.InputHeight

				cfg.GateWeights = make([]float32, gateSize)
				cfg.UpWeights = make([]float32, gateSize)
				cfg.DownWeights = make([]float32, downSize)

				for j := range cfg.GateWeights {
					cfg.GateWeights[j] = randomFloat()*0.2 - 0.1
					cfg.UpWeights[j] = randomFloat()*0.2 - 0.1
				}
				for j := range cfg.DownWeights {
					cfg.DownWeights[j] = randomFloat()*0.2 - 0.1
				}

				cfg.GateBias = make([]float32, cfg.OutputHeight)
				cfg.UpBias = make([]float32, cfg.OutputHeight)
				cfg.DownBias = make([]float32, cfg.InputHeight)
			}

		case LayerNorm:
			// LayerNorm parameters (scale=1, shift=0)
			if cfg.NormSize > 0 {
				cfg.Gamma = make([]float32, cfg.NormSize)
				cfg.Beta = make([]float32, cfg.NormSize)
				for j := range cfg.Gamma {
					cfg.Gamma[j] = 1.0
					cfg.Beta[j] = 0.0
				}
			}

		case LayerRMSNorm:
			// RMSNorm parameters (scale=1)
			if cfg.NormSize > 0 {
				cfg.Gamma = make([]float32, cfg.NormSize)
				for j := range cfg.Gamma {
					cfg.Gamma[j] = 1.0
				}
			}

		case LayerParallel:
			// Initialize each branch recursively
			for b := range cfg.ParallelBranches {
				branchCfg := &cfg.ParallelBranches[b]

				// Create a temporary mini-network for initialization
				tempNet := &Network{
					GridRows:      1,
					GridCols:      1,
					LayersPerCell: 1,
					Layers:        []LayerConfig{*branchCfg},
				}

				// Initialize the branch
				tempNet.InitializeWeights()

				// Copy back the initialized config
				cfg.ParallelBranches[b] = tempNet.Layers[0]
			}
		}

		n.SetLayer(row, col, layer, *cfg)
	}
}

// CloneForParallel creates a copy of the Network for parallel processing.
// The clone has its own activation buffers (to avoid race conditions during forward pass)
// but SHARES the weight pointers (Kernel, Bias, etc.) with the original for memory efficiency.
// This is thread-safe because weights are only READ during forward pass.
func (n *Network) CloneForParallel() *Network {
	totalLayers := n.TotalLayers()

	clone := &Network{
		GridRows:        n.GridRows,
		GridCols:        n.GridCols,
		LayersPerCell:   n.LayersPerCell,
		InputSize:       n.InputSize,
		BatchSize:       n.BatchSize,
		Layers:          make([]LayerConfig, len(n.Layers)),
		activations:     make([][]float32, totalLayers+1),
		preActivations:  make([][]float32, totalLayers),
		kernelGradients: nil, // Workers don't accumulate gradients
		biasGradients:   nil,
		learningRate:    n.learningRate,
		// Don't copy optimizer or GPU resources - worker uses CPU only
	}

	// Copy layer configs - weights are SHARED (pointers), but that's safe for read-only forward pass
	for i, layer := range n.Layers {
		// Shallow copy the struct (shares weight slice pointers)
		clone.Layers[i] = layer
	}

	// Activation buffers will be allocated during ForwardCPU
	// Pre-allocate if sizes are known
	for i := 0; i <= totalLayers; i++ {
		if i < len(n.activations) && n.activations[i] != nil {
			clone.activations[i] = make([]float32, len(n.activations[i]))
		}
	}
	for i := 0; i < totalLayers; i++ {
		if i < len(n.preActivations) && n.preActivations[i] != nil {
			clone.preActivations[i] = make([]float32, len(n.preActivations[i]))
		}
	}

	return clone
}
