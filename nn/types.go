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

// Layer represents a single neural network layer
type Layer struct {
	Activation ActivationType
}

// Network represents a neural network with sequential layers
type Network struct {
	Layers     []Layer
	InputSize  int
	deviceInfo *GPUDeviceInfo

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

// NewNetwork creates a new neural network with the specified layers
func NewNetwork(inputSize int, layers []Layer) *Network {
	return &Network{
		Layers:         layers,
		InputSize:      inputSize,
		activations:    make([][]float32, len(layers)+1), // +1 for input
		preActivations: make([][]float32, len(layers)),   // one per layer
	}
}

// ReleaseGPU releases GPU resources
func (n *Network) ReleaseGPU() {
	if n.deviceInfo != nil && n.deviceInfo.release != nil {
		n.deviceInfo.release()
		n.deviceInfo = nil
	}
}
