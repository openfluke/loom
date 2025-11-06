package nn

import (
	"github.com/openfluke/webgpu/wgpu"
)

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
	LayerDense              LayerType = 0 // Dense/Fully-connected layer (element-wise activation)
	LayerConv2D             LayerType = 1 // 2D Convolutional layer
	LayerMultiHeadAttention LayerType = 2 // Multi-Head Attention layer
	LayerRNN                LayerType = 3 // Recurrent Neural Network layer
	LayerLSTM               LayerType = 4 // Long Short-Term Memory layer
	LayerSoftmax            LayerType = 5 // Softmax layer with multiple variants
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
	NumHeads     int       // Number of attention heads
	HeadDim      int       // Dimension per head (dModel / numHeads)
	DModel       int       // Model dimension (embedding size)
	SeqLength    int       // Sequence length
	QWeights     []float32 // Query projection weights [dModel][dModel]
	KWeights     []float32 // Key projection weights [dModel][dModel]
	VWeights     []float32 // Value projection weights [dModel][dModel]
	OutputWeight []float32 // Output projection weights [dModel][dModel]
	QBias        []float32 // Query bias [dModel]
	KBias        []float32 // Key bias [dModel]
	VBias        []float32 // Value bias [dModel]
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
