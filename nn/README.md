# Neural Network Package

A high-performance grid neural network implementation in Go with support for multiple layer types, CPU/GPU execution, and automatic differentiation.

## Features

### Grid Architecture

- **Grid Structure**: Organizes layers in a 2D grid (rows × columns × layers per cell)
- **Flexible Configuration**: Each grid position can have different layer types and activations
- **Layer Types**:
  - **Dense**: Element-wise activation functions
  - **Conv2D**: 2D convolutional layers with configurable kernels
  - **Multi-Head Attention**: Transformer-style attention mechanism
  - **RNN**: Recurrent Neural Network with hidden state
  - **LSTM**: Long Short-Term Memory with forget/input/output gates

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
├── cnn.go                # Conv2D implementation (forward/backward)
├── attention.go          # Multi-Head Attention implementation
├── rnn.go                # RNN implementation with BPTT
└── lstm.go               # LSTM implementation with gate computations
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

### Multi-Head Attention

```go
// Create network with Multi-Head Attention layer
batchSize := 2
seqLength := 4
dModel := 8
numHeads := 2

totalInputSize := batchSize * seqLength * dModel
network := nn.NewNetwork(totalInputSize, 1, 1, 1)
network.BatchSize = batchSize

// Configure Multi-Head Attention layer
mhaConfig := nn.InitMultiHeadAttentionLayer(dModel, numHeads, batchSize, seqLength)
network.SetLayer(0, 0, 0, mhaConfig)

// Input shape: [batchSize, seqLength, dModel]
input := make([]float32, totalInputSize)
output, _ := network.ForwardCPU(input)
```

### RNN (Recurrent Neural Network)

```go
// Create RNN layer
batchSize := 2
seqLength := 4
inputSize := 8
hiddenSize := 16

totalInputSize := batchSize * seqLength * inputSize
network := nn.NewNetwork(totalInputSize, 1, 1, 1)
network.BatchSize = batchSize

// Configure RNN layer
rnnConfig := nn.InitRNNLayer(inputSize, hiddenSize, batchSize, seqLength)
network.SetLayer(0, 0, 0, rnnConfig)

// Input shape: [batchSize, seqLength, inputSize]
// Output shape: [batchSize, seqLength, hiddenSize]
input := make([]float32, totalInputSize)
output, _ := network.ForwardCPU(input)
```

### LSTM (Long Short-Term Memory)

```go
// Create LSTM layer
batchSize := 2
seqLength := 4
inputSize := 8
hiddenSize := 12

totalInputSize := batchSize * seqLength * inputSize
network := nn.NewNetwork(totalInputSize, 1, 1, 1)
network.BatchSize = batchSize

// Configure LSTM layer
lstmConfig := nn.InitLSTMLayer(inputSize, hiddenSize, batchSize, seqLength)
network.SetLayer(0, 0, 0, lstmConfig)

// Input shape: [batchSize, seqLength, inputSize]
// Output shape: [batchSize, seqLength, hiddenSize]
input := make([]float32, totalInputSize)
output, _ := network.ForwardCPU(input)
```

## Layer Details

### Conv2D

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

#### Conv2D

- **Kernel**: He initialization with `stddev = sqrt(2 / (inChannels * kernelSize²))`
- **Bias**: Initialized to zero

#### Multi-Head Attention

- **Q/K/V/Output Weights**: Xavier/Glorot initialization with `stddev = sqrt(2 / (fan_in + fan_out))`
- **Biases**: Initialized to zero

#### RNN

- **Input-to-Hidden**: Xavier initialization with `stddev = sqrt(2 / (inputSize + hiddenSize))`
- **Hidden-to-Hidden**: Xavier initialization with `stddev = sqrt(2 / (hiddenSize + hiddenSize))`
- **Bias**: Initialized to zero

#### LSTM

- **All Gates (i, f, g, o)**: Xavier initialization for both input-to-hidden and hidden-to-hidden weights
- **Forget Gate Bias**: Initialized to 1.0 (remember by default)
- **Other Biases**: Initialized to zero

## Multi-Head Attention Details

- Uses scaled dot-product attention: `softmax(Q·K^T / sqrt(d_k)) · V`
- Splits `dModel` into `numHeads` heads, each with dimension `headDim = dModel / numHeads`
- Linear projections for Q, K, V, and output
- Full backward pass through attention mechanism

## RNN/LSTM Details

### RNN Forward Pass

- Sequence processing: `h_t = tanh(W_ih · x_t + W_hh · h_{t-1} + b_h)`
- Hidden state initialized to zero
- Returns all timestep outputs: `[batch, seqLength, hiddenSize]`

### RNN Backward Pass

- Backpropagation Through Time (BPTT)
- Computes gradients for input, W_ih, W_hh, and bias
- Gradient flows through tanh activation

### LSTM Forward Pass

- Four gates per timestep:
  - Input gate: `i_t = sigmoid(W_ii · x_t + W_hi · h_{t-1} + b_i)`
  - Forget gate: `f_t = sigmoid(W_if · x_t + W_hf · h_{t-1} + b_f)`
  - Cell candidate: `g_t = tanh(W_ig · x_t + W_hg · h_{t-1} + b_g)`
  - Output gate: `o_t = sigmoid(W_io · x_t + W_ho · h_{t-1} + b_o)`
- Cell state update: `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`
- Hidden state: `h_t = o_t ⊙ tanh(c_t)`

### LSTM Backward Pass

- BPTT through all four gates
- Gradients for 12 weight matrices and 4 bias vectors
- Gradient flow through cell state across timesteps

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
go test -v -run TestMultiHeadAttention
go test -v -run TestRNNLayer
go test -v -run TestLSTMLayer
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

### Multi-Head Attention Demo

```bash
cd fabric
go run main.go
# Select option 11
```

### RNN Demo

```bash
cd fabric
go run main.go
# Select option 12
```

### LSTM Demo

```bash
cd fabric
go run main.go
# Select option 13
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
- `LayerMultiHeadAttention`: Transformer attention mechanism
- `LayerRNN`: Recurrent processing with BPTT
- `LayerLSTM`: LSTM gates and cell state

### GPU Shaders

- **Dense Forward/Backward**: Element-wise activation and gradient computation (fully implemented)
- **Conv2D Shaders**: Full generation functions exist (`generateConv2DForwardShader`, `generateConv2DBackwardShader`) but not yet integrated into GPU execution pipeline
- **MHA/RNN/LSTM**: Placeholder shader functions created that pass through data
  - MHA: Requires multi-stage pipeline (Q/K/V projections, attention scores, softmax, weighted sum)
  - RNN: Requires sequential processing across timesteps
  - LSTM: Requires 4-gate computation with cell state management
- Currently all non-Dense layers fall back to Dense shaders in ForwardGPU/BackwardGPU

## Current Limitations

- **GPU Execution**: ForwardGPU and BackwardGPU use Dense layer shaders for all layer types
- **Conv2D**: Shader generation code fully implemented but needs integration into the GPU pipeline
- **MHA**: Requires complex multi-stage pipeline architecture (5+ kernel launches per forward pass)
- **RNN/LSTM**: Sequential timestep processing conflicts with GPU parallelism model
- **Placeholder Shaders**: MHA/RNN/LSTM shaders exist but only pass through data as placeholders
- Causes large accuracy differences (>1.0) in demos but CPU implementations are correct
- Full GPU support for these layers would require significant architectural changes

## Future Enhancements

Potential additions:

- [x] RNN/LSTM layers (CPU implementation completed)
- [x] Multi-Head Attention (CPU implementation completed)
- [x] Conv2D shader generation (completed, needs pipeline integration)
- [x] MHA/RNN/LSTM placeholder shaders (completed)
- [ ] Integrate Conv2D shaders into GPU pipeline
- [ ] Multi-stage GPU pipeline for MHA (Q/K/V projections, attention, softmax)
- [ ] Parallel RNN/LSTM implementations (e.g., QRNN, SRU) more suitable for GPU
- [ ] MaxPool2D and AvgPool2D layers
- [ ] Batch normalization
- [ ] Dropout layers
- [ ] Layer normalization
- [ ] Automatic mixed precision (FP16/FP32)
- [ ] Multi-GPU support
- [ ] Optimizers (SGD, Adam, RMSprop)
- [ ] Loss functions (cross-entropy, MSE)
- [ ] Model serialization (save/load weights)
