package nn

import (
	"fmt"
	"time"
)

// StepForwardCPU executes one step of the grid network on CPU, processing a single layer at a time.
func (n *Network) StepForwardCPU(input []float32, layerIdx int) ([]float32, error) {
	if layerIdx >= n.TotalLayers() {
		return nil, fmt.Errorf("layerIdx %d exceeds total layers %d", layerIdx, n.TotalLayers())
	}

	// Determine the grid position for the current layer
	row := layerIdx / (n.GridCols * n.LayersPerCell)
	remainder := layerIdx % (n.GridCols * n.LayersPerCell)
	col := remainder / n.LayersPerCell
	layer := remainder % n.LayersPerCell

	// Get layer configuration
	config := n.GetLayer(row, col, layer)

	// Process the layer based on its type
	var data []float32
	var err error

	switch config.Type {
	case LayerConv2D:
		_, data = conv2DForwardCPU(input, config, n.BatchSize)
	case LayerMultiHeadAttention:
		_, data = MultiHeadAttentionForwardCPU(input, config, n.BatchSize)
	case LayerRNN:
		data, _ = rnnForwardCPU(config, input, n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)
	case LayerLSTM:
		data, _ = lstmForwardCPU(config, input, n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)
	case LayerSoftmax:
		data, err = ForwardSoftmaxCPU(input, config)
		if err != nil {
			data = softmaxStandard(input, 1.0)
		}
	case LayerDense:
		_, data = denseForwardCPU(input, config, n.BatchSize)
	case LayerSwiGLU:
		_, data = SwiGLUForwardCPU(input, config, n.BatchSize)
	case LayerNorm:
		data = layerNormForwardCPU(input, nil, config, n.BatchSize)
	case LayerRMSNorm:
		data = rmsNormForwardCPU(input, nil, config, n.BatchSize)
	case LayerParallel:
		data, _, err = parallelForwardCPU(input, config, n.BatchSize)
		if err != nil {
			data = input // Pass through unchanged on error
		}
	default:
		// Default: apply activation function
		data = make([]float32, len(input))
		copy(data, input)
		for i := 0; i < len(data); i++ {
			data[i] = activateCPU(data[i], config.Activation)
		}
	}

	// Store the output of this layer
	n.activations[layerIdx+1] = make([]float32, len(data))
	copy(n.activations[layerIdx+1], data)

	return data, nil
}

// StepThroughNetworkCPU steps through all layers of the network one at a time.
func (n *Network) StepThroughNetworkCPU(input []float32) ([]float32, time.Duration) {
	start := time.Now()
	data := input

	for layerIdx := 0; layerIdx < n.TotalLayers(); layerIdx++ {
		var err error
		data, err = n.StepForwardCPU(data, layerIdx)
		if err != nil {
			return nil, time.Since(start)
		}
	}

	return data, time.Since(start)
}

// StepThroughNetworkWithTimer steps through all layers of the network one at a time, with a timer loop.
func (n *Network) StepThroughNetworkWithTimer(input []float32, interval time.Duration, steps int) ([]float32, error) {
	data := input

	for step := 0; step < steps; step++ {
		for layerIdx := 0; layerIdx < n.TotalLayers(); layerIdx++ {
			var err error
			data, err = n.StepForwardCPU(data, layerIdx)
			if err != nil {
				return nil, err
			}
		}

		// Wait for the specified interval before the next step
		time.Sleep(interval)
	}

	return data, nil
}
