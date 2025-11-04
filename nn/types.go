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

// Network represents a grid neural network
// Data flows through a 2D grid of cells, where each cell contains multiple layers
type Network struct {
	GridRows      int // Number of rows in the grid
	GridCols      int // Number of columns in the grid
	LayersPerCell int // Number of layers per grid cell
	InputSize     int // Batch size
	deviceInfo    *GPUDeviceInfo

	// Storage for intermediate activations (needed for backward pass)
	// activations[0] = input, activations[i] = output of layer i-1
	activations [][]float32

	// Storage for pre-activation values (needed for derivatives)
	preActivations [][]float32
}

// GPUDeviceInfo holds WebGPU resources for GPU execution
type GPUDeviceInfo struct {
	Device     *wgpu.Device
	Queue      *wgpu.Queue
	WorkgroupX uint32
	release    func()
}

// NewNetwork creates a new grid neural network
// gridRows: number of rows in the grid
// gridCols: number of columns in the grid
// layersPerCell: number of layers in each grid cell
// inputSize: batch size of input data
func NewNetwork(inputSize, gridRows, gridCols, layersPerCell int) *Network {
	totalLayers := gridRows * gridCols * layersPerCell
	return &Network{
		GridRows:       gridRows,
		GridCols:       gridCols,
		LayersPerCell:  layersPerCell,
		InputSize:      inputSize,
		activations:    make([][]float32, totalLayers+1), // +1 for input
		preActivations: make([][]float32, totalLayers),
	}
}

// TotalLayers returns the total number of layers in the grid
func (n *Network) TotalLayers() int {
	return n.GridRows * n.GridCols * n.LayersPerCell
}

// GetActivation returns the activation function for a specific position in the grid
// The activation cycles through the 5 types based on cell and layer position
func (n *Network) GetActivation(row, col, layer int) ActivationType {
	cellIdx := row*n.GridCols + col
	activation := (cellIdx*n.LayersPerCell + layer) % 5
	return ActivationType(activation)
}

// ReleaseGPU releases GPU resources
func (n *Network) ReleaseGPU() {
	if n.deviceInfo != nil && n.deviceInfo.release != nil {
		n.deviceInfo.release()
		n.deviceInfo = nil
	}
}
