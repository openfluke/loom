# welvet - LOOM Python Bindings

**Wrapper for Embedding Loom Via External (C-ABI) Toolchain**

High-performance neural network library with WebGPU acceleration for Python via C-ABI bindings.

## Installation

```bash
pip install welvet
```

## Quick Start

```python
import loom_py

# Create a neural network with GPU
network = loom_py.create_network(
    input_size=4,
    grid_rows=1,
    grid_cols=1,
    layers_per_cell=2,  # 2 layers: hidden + output
    use_gpu=True
)

# Configure network architecture: 4 -> 8 -> 2
loom_py.configure_sequential_network(
    network,
    layer_sizes=[4, 8, 2],
    activations=[loom_py.Activation.RELU, loom_py.Activation.SIGMOID]
)

# Training data
inputs = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
targets = [[1.0, 0.0], [0.0, 1.0]]

# Train for 10 epochs
for epoch in range(10):
    loss = loom_py.train_epoch(network, inputs, targets, learning_rate=0.1)
    print(f"Epoch {epoch+1}: loss = {loss:.4f}")

# Test the network
output = loom_py.forward(network, [0.1, 0.2, 0.3, 0.4])
print(f"Output: {output}")

# Clean up
loom_py.cleanup_gpu(network)
loom_py.free_network(network)
```

## Features

- 🚀 **GPU Acceleration**: WebGPU-powered compute shaders for high performance
- 🎯 **Cross-Platform**: Pre-compiled binaries for Linux, macOS, Windows, Android
- 📦 **Easy Integration**: Simple Python API with high-level helpers
- ⚡ **Grid Architecture**: Flexible grid-based neural network topology
- 🔧 **Low-Level Access**: Direct control over layers and training loop
- 🎓 **Training Helpers**: Built-in functions for common training tasks

## API Reference

### Network Management

#### `create_network(input_size, grid_rows=2, grid_cols=2, layers_per_cell=3, use_gpu=False)`

Creates a new grid-based neural network.

**Parameters:**

- `input_size` (int): Number of input features
- `grid_rows` (int): Grid rows (default: 2)
- `grid_cols` (int): Grid columns (default: 2)
- `layers_per_cell` (int): Layers per grid cell (default: 3)
- `use_gpu` (bool): Enable GPU acceleration (default: False)

**Simplified API:**

- `create_network(input_size, hidden_size, output_size, use_gpu=False)` - Auto-calculates grid

**Returns:** Network handle (int)

#### `free_network(handle)`

Frees network resources.

**Parameters:**

- `handle` (int): Network handle

### Layer Configuration

#### `Activation` (Class)

Activation function constants:

- `Activation.RELU` (0) - ReLU activation
- `Activation.SIGMOID` (1) - Sigmoid activation
- `Activation.TANH` (2) - Tanh activation
- `Activation.LINEAR` (3) - Linear activation

#### `init_dense_layer(input_size, output_size, activation=0)`

Initialize a dense layer configuration.

**Parameters:**

- `input_size` (int): Input neurons
- `output_size` (int): Output neurons
- `activation` (int): Activation function (use `Activation` constants)

**Returns:** Layer configuration dict

#### `set_layer(handle, row, col, layer_index, layer_config)`

Set a layer in the network grid.

**Parameters:**

- `handle` (int): Network handle
- `row` (int): Grid row (0-indexed)
- `col` (int): Grid column (0-indexed)
- `layer_index` (int): Layer index in cell (0-indexed)
- `layer_config` (dict): Layer config from `init_dense_layer()`

#### `configure_sequential_network(handle, layer_sizes, activations=None)`

High-level helper to configure a simple feedforward network.

**Parameters:**

- `handle` (int): Network handle (must have 1x1 grid)
- `layer_sizes` (List[int]): Layer sizes `[input, hidden1, ..., output]`
- `activations` (List[int], optional): Activation for each layer. Defaults to ReLU for hidden, Sigmoid for output.

**Example:**

```python
net = create_network(input_size=784, grid_rows=1, grid_cols=1, layers_per_cell=2)
configure_sequential_network(net, [784, 128, 10])  # MNIST classifier
```

#### `get_network_info(handle)`

Get network information.

**Returns:** Dict with `type`, `gpu_enabled`, `grid_rows`, `grid_cols`, `layers_per_cell`, `total_layers`

### Operations

#### `forward(handle, input_data)`

Performs forward pass through the network.

**Parameters:**

- `handle` (int): Network handle
- `input_data` (List[float]): Input vector

**Returns:** Output vector (List[float])

#### `backward(handle, target_data)`

Performs backward pass for training.

**Parameters:**

- `handle` (int): Network handle
- `target_data` (List[float]): Target/label vector

#### `update_weights(handle, learning_rate)`

Updates network weights using computed gradients.

**Parameters:**

- `handle` (int): Network handle
- `learning_rate` (float): Learning rate for gradient descent

### Training Helpers

#### `train_epoch(handle, inputs, targets, learning_rate=0.01)`

Train the network for one epoch.

**Parameters:**

- `handle` (int): Network handle
- `inputs` (List[List[float]]): List of input vectors
- `targets` (List[List[float]]): List of target vectors
- `learning_rate` (float): Learning rate (default: 0.01)

**Returns:** Average loss for the epoch (float)

**Example:**

```python
loss = train_epoch(net, train_inputs, train_targets, learning_rate=0.1)
print(f"Epoch loss: {loss:.4f}")
```

### GPU Management

#### `initialize_gpu(handle)`

Explicitly initialize GPU resources.

**Returns:** True if successful, False otherwise

#### `cleanup_gpu(handle)`

Release GPU resources.

**Parameters:**

- `handle` (int): Network handle

#### `get_version()`

Get LOOM library version string.

**Returns:** Version string (e.g., "LOOM C ABI v1.0")

## Examples

### Basic Training Example

```python
import loom_py

# Create network with GPU
net = loom_py.create_network(
    input_size=4,
    grid_rows=1,
    grid_cols=1,
    layers_per_cell=2,
    use_gpu=True
)

# Configure architecture: 4 -> 8 -> 2
loom_py.configure_sequential_network(net, [4, 8, 2])

# Training data
inputs = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
targets = [[1.0, 0.0], [0.0, 1.0]]

# Train for 50 epochs
for epoch in range(50):
    loss = loom_py.train_epoch(net, inputs, targets, learning_rate=0.1)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: loss = {loss:.6f}")

# Test
output = loom_py.forward(net, [0.1, 0.2, 0.3, 0.4])
print(f"Output: {output}")

# Cleanup
loom_py.cleanup_gpu(net)
loom_py.free_network(net)
```

### Custom Layer Configuration

```python
import loom_py

# Create network
net = loom_py.create_network(
    input_size=10,
    grid_rows=2,
    grid_cols=2,
    layers_per_cell=3,
    use_gpu=False
)

# Configure individual layers
for row in range(2):
    for col in range(2):
        # Layer 0: 10 -> 20 (ReLU)
        layer0 = loom_py.init_dense_layer(10, 20, loom_py.Activation.RELU)
        loom_py.set_layer(net, row, col, 0, layer0)

        # Layer 1: 20 -> 15 (Tanh)
        layer1 = loom_py.init_dense_layer(20, 15, loom_py.Activation.TANH)
        loom_py.set_layer(net, row, col, 1, layer1)

        # Layer 2: 15 -> 5 (Sigmoid)
        layer2 = loom_py.init_dense_layer(15, 5, loom_py.Activation.SIGMOID)
        loom_py.set_layer(net, row, col, 2, layer2)

# Network is now configured
info = loom_py.get_network_info(net)
print(f"Total layers: {info['total_layers']}")

loom_py.free_network(net)
```

## Testing

Run the included examples to verify installation:

```bash
# Basic GPU training test
python examples/train_gpu.py
```

Or test programmatically:

```python
import loom_py

# Test basic functionality
net = loom_py.create_network(input_size=2, grid_rows=1, grid_cols=1,
                             layers_per_cell=1, use_gpu=False)
loom_py.configure_sequential_network(net, [2, 4, 2])

# Verify forward pass works
output = loom_py.forward(net, [0.5, 0.5])
assert len(output) == 2, "Forward pass failed"

# Verify training works
inputs = [[0.0, 0.0], [1.0, 1.0]]
targets = [[1.0, 0.0], [0.0, 1.0]]
loss = loom_py.train_epoch(net, inputs, targets, learning_rate=0.1)
assert loss > 0, "Training failed"

loom_py.free_network(net)
print("✅ All tests passed!")
```

## Platform Support

Pre-compiled binaries included for:

- **Linux**: x86_64, ARM64
- **macOS**: ARM64 (Apple Silicon)
- **Windows**: x86_64
- **Android**: ARM64

## Building from Source

See the main [LOOM repository](https://github.com/openfluke/loom) for building the C ABI from source.

## License

Apache License 2.0

## Links

- [GitHub Repository](https://github.com/openfluke/loom)
- [C ABI Documentation](https://github.com/openfluke/loom/tree/main/cabi)
- [Issue Tracker](https://github.com/openfluke/loom/issues)
