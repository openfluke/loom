// Package nn provides a neural network implementation with both CPU and GPU execution.
//
// The network supports forward and backward propagation with multiple activation functions:
//   - ScaledReLU: v * 1.1, then ReLU
//   - Sigmoid: 1 / (1 + exp(-v))
//   - Tanh: tanh(v)
//   - Softplus: log(1 + exp(v))
//   - LeakyReLU: v if v >= 0, else v * 0.1
//
// Example usage:
//
//	layers := []nn.Layer{
//	    {Activation: nn.ActivationSigmoid},
//	    {Activation: nn.ActivationTanh},
//	}
//	network := nn.NewNetwork(batchSize, layers)
//
//	// Forward pass on CPU
//	output, _ := network.ForwardCPU(input)
//
//	// Forward pass on GPU
//	network.InitGPU()
//	defer network.ReleaseGPU()
//	outputGPU, _, _ := network.ForwardGPU(input)
//
//	// Backward pass
//	gradInput, _ := network.BackwardCPU(gradOutput)
package nn
