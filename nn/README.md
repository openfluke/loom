# Neural Network Package

A high-performance grid neural network implementation in Go with support for multiple layer types, CPU/GPU execution, and automatic differentiation.

## Features

### Grid Architecture

- **Grid Structure**: Organizes layers in a 2D grid (rows × columns × layers per cell)
- **Flexible Configuration**: Each grid position can have different layer types and activations
- **Layer Types**:
  - **Dense**: Element-wise activation functions
  - **Conv2D**: 2D convolutional layers with configurable kernels

### Activation Functions

1. **ScaledReLU**: `max(0, 1.1 * x)` with derivative `1.1` for x > 0
2. **Sigmoid**: `1 / (1 + e^(-x))`
3. **Tanh**: `(e^(2x) - 1) / (e^(2x) + 1)`
4. **Softplus**: `log(1 + e^x)`
5. **LeakyReLU**: `x` if x ≥ 0, else `0.1 * x`

### Execution Modes

- **CPU**: Pure Go implementation with full backward propagation
- **GPU**: WebGPU/WGSL compute shaders for parallel execution
- **Automatic Gradient Computation**: Stores activations and pre-activations for backprop

## File Structure

```
nn/
├── nn.go                 # Package documentation
├── types.go              # Core types (Network, LayerConfig, LayerType)
├── activations.go        # Activation functions and derivatives
├── forward.go            # Forward propagation (CPU/GPU)
├── backward.go           # Backward propagation (CPU/GPU)
├── gpu.go                # WebGPU initialization and shader generation
├── utils.go              # Utility functions (MaxAbsDiff, Min, Max, Mean)
└── cnn.go                # Conv2D implementation (forward/backward)
```

## Usage

### Creating a Dense Neural Network

```go
// Create a 2x2 grid with 2 layers per cell = 8 total layers
network := nn.NewNetwork(
    1024,  // inputSize (batch size for dense layers)
    2,     // gridRows
    2,     // gridCols
    2,     // layersPerCell
)

// Forward pass on CPU
input := make([]float32, 1024)
output, cpuTime := network.ForwardCPU(input)

// Forward pass on GPU
network.InitGPU()
defer network.ReleaseGPU()
outputGPU, gpuTime, err := network.ForwardGPU(input)

// Backward pass
gradOutput := make([]float32, len(output))
gradInput, bwdTime := network.BackwardCPU(gradOutput)
```

### Creating a Conv2D Network

```go
// Create network
batchSize := 2
network := nn.NewNetwork(inputSize, 1, 1, 1)
network.BatchSize = batchSize

// Configure Conv2D layer
conv2dConfig := nn.InitConv2DLayer(
    4,  // inputHeight
    4,  // inputWidth
    3,  // inputChannels
    3,  // kernelSize
    1,  // stride
    1,  // padding
    2,  // filters
    nn.ActivationScaledReLU,
)
network.SetLayer(0, 0, 0, conv2dConfig)

// Forward and backward passes work the same
output, _ := network.ForwardCPU(input)
gradInput, _ := network.BackwardCPU(gradOutput)
```

### Mixed Layer Types

```go
// Create grid with both Dense and Conv2D layers
network := nn.NewNetwork(inputSize, 2, 1, 2)
network.BatchSize = batchSize

// Cell [0,0], layer 0: Conv2D
conv2d := nn.InitConv2DLayer(4, 4, 3, 3, 1, 1, 2, nn.ActivationScaledReLU)
network.SetLayer(0, 0, 0, conv2d)

// Cell [0,0], layer 1: Dense (default)
// Cell [1,0] layers: Dense (default)

// Network automatically routes to appropriate layer type
output, _ := network.ForwardCPU(input)
```

## Conv2D Details

### Forward Pass

- Input shape: `[batch, inChannels, height, width]`
- Output shape: `[batch, filters, outHeight, outWidth]`
- Output dimensions: `outH = (inH + 2*padding - kernelSize) / stride + 1`
- Kernel shape: `[filters, inChannels, kernelSize, kernelSize]`

### Backward Pass

Computes three gradients:

1. **∂L/∂input**: Gradient with respect to input (for backprop to previous layer)
2. **∂L/∂kernel**: Gradient with respect to kernel weights (for weight updates)
3. **∂L/∂bias**: Gradient with respect to bias (for weight updates)

### Weight Initialization

- **Kernel**: He initialization with `stddev = sqrt(2 / (inChannels * kernelSize²))`
- **Bias**: Initialized to zero

## Performance

- **CPU/GPU Accuracy**: Typically < 1e-7 difference between CPU and GPU forward pass
- **Gradient Accuracy**: Typically < 1e-9 difference between CPU and GPU backward pass
- **GPU Speedup**: Varies by network size and hardware, typically 5-20x for large networks

## Testing

Run all tests:

```bash
cd fabric/examples
go test -v
```

Run specific test:

```bash
go test -v -run TestConv2DLayer
go test -v -run TestMixedLayerTypes
```

## Demos

### Dense Network Demo

```bash
cd fabric
go run main.go
# Select option 9
```

### Conv2D Demo

```bash
cd fabric
go run main.go
# Select option 10
```

## Implementation Notes

### Grid Indexing

Layers are stored in a flattened array with index:

```
idx = row * GridCols * LayersPerCell + col * LayersPerCell + layer
```

### Activation Storage

- `activations[0]`: Input to network
- `activations[i]`: Output of layer i-1 (post-activation)
- `preActivations[i]`: Pre-activation values for layer i (needed for derivatives)

### Layer Routing

Forward and backward passes check `LayerConfig.Type`:

- `LayerDense`: Element-wise activation
- `LayerConv2D`: 2D convolution with kernel

### GPU Shaders

- **Dense Forward**: Element-wise activation on each value
- **Dense Backward**: Multiply gradient by activation derivative
- **Conv2D Forward**: 4D tensor convolution with bounds checking
- **Conv2D Backward**: Gradient computation with proper indexing

## Future Enhancements

Potential additions:

- [ ] MaxPool2D and AvgPool2D layers
- [ ] Batch normalization
- [ ] Dropout layers
- [ ] RNN/LSTM layers
- [ ] Automatic mixed precision (FP16/FP32)
- [ ] Multi-GPU support
- [ ] Optimizers (SGD, Adam, RMSprop)
- [ ] Loss functions (cross-entropy, MSE)
- [ ] Model serialization (save/load weights)
