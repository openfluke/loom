// Package nn provides a grid neural network implementation with both CPU and GPU execution.
//
// A grid neural network organizes layers into a 2D grid structure where:
//   - Data flows sequentially through grid cells (row by row, column by column)
//   - Each grid cell contains multiple layers
//   - Total layers = gridRows × gridCols × layersPerCell
//
// The network supports forward and backward propagation with multiple activation functions:
//   - ScaledReLU: v * 1.1, then ReLU
//   - Sigmoid: 1 / (1 + exp(-v))
//   - Tanh: tanh(v)
//   - Softplus: log(1 + exp(v))
//   - LeakyReLU: v if v >= 0, else v * 0.1
//
// Activations are assigned cyclically based on grid position: (cellIdx * layersPerCell + layer) % 5
//
// Example usage:
//
//	network := nn.NewNetwork(batchSize, gridRows, gridCols, layersPerCell)
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
