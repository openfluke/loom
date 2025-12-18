# welvet - LOOM Python Bindings

**Wrapper for Embedding Loom Via External (C-ABI) Toolchain**

High-performance neural network library with **transformer inference** for Python via C-ABI bindings. CPU-first with reliable execution.

## Installation

```bash
pip install welvet
```

## Quick Start

### 🎉 NEW: Simple API (Recommended)

The new simple API provides a streamlined interface with **cross-platform consistency**:

```python
import welvet
import json

# Create network from JSON configuration
config = {
    "batch_size": 1,
    "grid_rows": 1,
    "grid_cols": 3,
    "layers_per_cell": 1,
    "layers": [
        {"type": "dense", "input_size": 8, "output_size": 16, "activation": "relu"},
        {
            "type": "parallel",
            "combine_mode": "grid_scatter",
            "grid_output_rows": 3,
            "grid_output_cols": 1,
            "grid_output_layers": 1,
            "grid_positions": [
                {"branch_index": 0, "target_row": 0, "target_col": 0, "target_layer": 0},
                {"branch_index": 1, "target_row": 1, "target_col": 0, "target_layer": 0},
                {"branch_index": 2, "target_row": 2, "target_col": 0, "target_layer": 0}
            ],
            "branches": [
                {"type": "parallel", "combine_mode": "add", "branches": [
                    {"type": "dense", "input_size": 16, "output_size": 8, "activation": "relu"},
                    {"type": "dense", "input_size": 16, "output_size": 8, "activation": "gelu"}
                ]},
                {"type": "lstm", "input_size": 16, "hidden_size": 8, "seq_length": 1},
                {"type": "rnn", "input_size": 16, "hidden_size": 8, "seq_length": 1}
            ]
        },
        {"type": "dense", "input_size": 24, "output_size": 2, "activation": "sigmoid"}
    ]
}

# Create network (no handle management!)
welvet.create_network_from_json(config)

# Training data
batches = [
    {"Input": [0.2] * 4 + [0.8] * 4, "Target": [1.0, 0.0]},
    {"Input": [0.9] * 4 + [0.1] * 4, "Target": [0.0, 1.0]},
    {"Input": [0.7] * 4 + [0.3] * 4, "Target": [0.0, 1.0]},
    {"Input": [0.3] * 4 + [0.7] * 4, "Target": [1.0, 0.0]},
]

# Train
training_config = {
    "Epochs": 800,
    "LearningRate": 0.15,
    "UseGPU": False,
    "PrintEveryBatch": 0,
    "GradientClip": 1.0,
    "LossType": "mse",
    "Verbose": False
}
result = welvet.train_simple(batches, training_config)

# Forward pass
output = welvet.forward_simple([0.2] * 4 + [0.8] * 4)
print(f"Output: {output}")  # [0.950, 0.050] → Class 0

# Evaluate with deviation metrics
inputs = [batch["Input"] for batch in batches]
expected = [0, 1, 1, 0]
metrics = welvet.evaluate_network_simple(inputs, expected)
print(f"Quality Score: {metrics['score']}/100")  # 100/100
print(f"Average Deviation: {metrics['avg_deviation']:.2f}%")  # 0.00%

# Save/Load model
model_json = welvet.save_model_simple("my_model")
print(f"Model size: {len(model_json)} bytes")  # ~26KB

# Load in new session
welvet.load_model_simple(model_json, "my_model")
output2 = welvet.forward_simple([0.2] * 4 + [0.8] * 4)
# output2 == output (bit-for-bit identical!)
```

**Simple API Functions:**

- `create_network_from_json(config)` - Create from JSON
- `forward_simple(inputs)` - Forward pass
- `backward_simple(gradients)` - Backward pass
- `update_weights_simple(learning_rate)` - Update weights
- `train_simple(batches, config)` - Train network
- `save_model_simple(model_id)` - Save to JSON string
- `load_model_simple(json_string, model_id)` - Load from JSON
- `get_network_info_simple()` - Get network info
- `evaluate_network_simple(inputs, expected)` - Evaluate with metrics

**Cross-Platform Consistency:**
The simple API matches the API in TypeScript, C#, C, and WASM - identical function signatures and behavior!

See `examples/grid_scatter_demo.py` for a complete working example.

### ⚡ Stepping API - Fine-Grained Execution Control

**NEW:** Execute networks one step at a time for online learning and stateful processing:

```python
from welvet import create_network_from_json, StepState, apply_gradients

# Create network
config = {"batch_size": 1, "layers": [
    {"type": "dense", "input_height": 4, "output_height": 8, "activation": "relu"},
    {"type": "lstm", "input_size": 8, "hidden_size": 12, "seq_length": 1},
    {"type": "dense", "input_height": 12, "output_height": 3, "activation": "softmax"}
]}
create_network_from_json(config)

# Initialize stepping state
state = StepState(input_size=4)

# Training loop - update weights after EACH step
for step in range(100000):
    state.set_input([0.1, 0.2, 0.1, 0.3])
    state.step_forward()
    output = state.get_output()
    
    # Calculate gradients
    gradients = [output[i] - target[i] for i in range(len(output))]
    
    # Backward pass
    state.step_backward(gradients)
    
    # Update weights immediately
    apply_gradients(learning_rate=0.01)
```

**Stepping API Functions:**
- `StepState(input_size)` - Initialize stepping state
- `state.set_input(data)` - Set input for current step
- `state.step_forward()` - Execute forward pass
- `state.get_output()` - Get output from last layer
- `state.step_backward(gradients)` - Execute backward pass
- `apply_gradients(learning_rate)` - Update network weights

See `examples/step_train_v3.py` for a complete example achieving 100% accuracy.

### 🧬 Neural Tweening API - Real-Time Adaptation

**NEW:** Neural tweening enables networks to adapt to changing goals in real-time without full backpropagation:

```python
import welvet

# Create network
config = '''{"batch_size": 1, "layers": [
    {"type": "dense", "input_size": 8, "output_size": 32, "activation": "leaky_relu"},
    {"type": "dense", "input_size": 32, "output_size": 16, "activation": "leaky_relu"},
    {"type": "dense", "input_size": 16, "output_size": 4, "activation": "sigmoid"}
]}'''
welvet.create_network_from_json(config)

# Create TweenState with chain rule (StepTweenChain mode)
tween = welvet.TweenState(use_chain_rule=True)

# Continuously adapt to targets
for observation, target_class in training_stream:
    gap = tween.step(observation, target_class=target_class, output_size=4, learning_rate=0.02)
    print(f"Adaptation gap: {gap:.4f}")

tween.close()
```

**TweenState Class:**
- `TweenState(use_chain_rule=False)` - Create tween state (chain rule = TweenChain mode)
- `tween.step(input, target_class, output_size, learning_rate)` - Apply one tween step
- `tween.close()` - Free resources

### 📊 AdaptationTracker - Benchmark Task Switching

Track accuracy across task changes for benchmarking real-time adaptation:

```python
import welvet

# Create tracker with 1s windows over 10s test
tracker = welvet.AdaptationTracker(window_duration_ms=1000, total_duration_ms=10000)
tracker.set_model_info("Dense-5L", "StepTweenChain")

# Schedule task changes at 1/3 and 2/3 of test
tracker.schedule_task_change(3333, task_id=1, task_name="AVOID")
tracker.schedule_task_change(6666, task_id=0, task_name="CHASE")

tracker.start("CHASE", task_id=0)

# Run test loop
while time_elapsed < 10000:
    current_task = tracker.get_current_task()
    # ... run network ...
    tracker.record_output(is_correct=True)

# Get results
results = tracker.finalize()
print(f"Avg accuracy: {results['avg_accuracy']:.1f}%")
print(f"Window accuracies: {results['window_accuracies']}")

tracker.close()
```

**AdaptationTracker Class:**
- `AdaptationTracker(window_duration_ms, total_duration_ms)` - Create tracker
- `tracker.set_model_info(model_name, mode_name)` - Set model info
- `tracker.schedule_task_change(at_offset_ms, task_id, task_name)` - Schedule task change
- `tracker.start(task_name, task_id)` - Start tracking
- `tracker.get_current_task()` - Get current task ID
- `tracker.record_output(is_correct)` - Record output
- `tracker.finalize()` - Get results JSON
- `tracker.close()` - Free resources

See `examples/test18_adaptation.py` for a complete multi-architecture benchmark comparing 5 training modes across Dense, Conv2D, RNN, LSTM, and Attention networks.

### 🚀 Transformer Inference (LLMs)

Run LLaMA, SmolLM, GPT-2, and other transformers with **streaming support**!

```python
import welvet

# Load tokenizer and model
with open('models/SmolLM2-135M-Instruct/tokenizer.json', 'rb') as f:
    welvet.load_tokenizer_from_bytes(f.read())

with open('models/SmolLM2-135M-Instruct/config.json', 'rb') as f:
    config = f.read()
with open('models/SmolLM2-135M-Instruct/model.safetensors', 'rb') as f:
    weights = f.read()

welvet.load_transformer_from_bytes(config, weights)

# Generate text with streaming!
for token in welvet.generate_stream("Once upon a time", max_tokens=50):
    print(token, end='', flush=True)

# Or generate all at once
text = welvet.generate_text("Once upon a time", max_tokens=50, temperature=0.7)
print(text)
```

#### Web Interface Example

```bash
cd examples
./transformer_web_interface.py ../../models/SmolLM2-135M-Instruct 8080
# Open http://localhost:8080/inference.html
```

See `examples/test_transformer.py` for a complete example.

### ✨ Neural Network Training - Load Complete Models

```python
import welvet

# Load a complete model (structure + all weights) in ONE LINE!
network = welvet.load_model_from_string(model_json, "my_model")

# That's it! Network is ready to use
output = welvet.forward(network, input_data)

# Train it
welvet.backward(network, gradient)
welvet.update_weights(network, learning_rate=0.01)

# Save it
model_json = welvet.save_model_to_string(network, "my_model")
```

### Building Networks from Scratch

```python
import welvet

# Create a neural network with all 6 layer types
network = welvet.create_network(
    input_size=32,
    grid_rows=1,
    grid_cols=1,
    layers_per_cell=7,
    use_gpu=True
)

# Initialize layers using registry-based system
dense1 = welvet.call_layer_init("InitDenseLayer", [32, 32, welvet.Activation.LEAKY_RELU])
conv2d = welvet.call_layer_init("InitConv2DLayer", [4, 4, 2, 4, 3, 2, 1, welvet.Activation.LEAKY_RELU])
attention = welvet.call_layer_init("InitMultiHeadAttentionLayer", [4, 4, 2, welvet.Activation.TANH])
rnn = welvet.call_layer_init("InitRNNLayer", [4, 8, 4, 32])
lstm = welvet.call_layer_init("InitLSTMLayer", [8, 4, 4, 16])
parallel = welvet.call_layer_init("InitParallelLayer", [0, 2])  # concat mode, 2 branches
dense2 = welvet.call_layer_init("InitDenseLayer", [16, 2, welvet.Activation.SIGMOID])

# Set layers in network
welvet.set_layer(network, 0, 0, 0, dense1)
welvet.set_layer(network, 0, 0, 1, conv2d)
welvet.set_layer(network, 0, 0, 2, attention)
welvet.set_layer(network, 0, 0, 3, rnn)
welvet.set_layer(network, 0, 0, 4, lstm)
welvet.set_layer(network, 0, 0, 5, parallel)
welvet.set_layer(network, 0, 0, 6, dense2)

# Prepare training data
batches = [
    {"Input": [0.8] * 16 + [0.2] * 16, "Target": [1.0, 0.0]},
    {"Input": [0.2] * 16 + [0.8] * 16, "Target": [0.0, 1.0]},
]

# Train using high-level API
result = welvet.train(
    network,
    batches,
    epochs=10,
    learning_rate=0.003,
    gradient_clip=1.0,
    loss_type="mse"
)

print(f"Final Loss: {result['FinalLoss']:.6f}")
print(f"Throughput: {result['AvgThroughput']:.0f} samples/sec")

# Clean up
welvet.cleanup_gpu(network)
welvet.free_network(network)
```

### Complete Example: All Layers Test

See `examples/all_layers_test.py` for a comprehensive test that:

1. Downloads a complete model from localhost:3123
2. Loads it with `load_model_from_string()` - ONE line!
3. Runs inference and compares outputs
4. Trains to verify weights are mutable

```bash
# Start the file server (serves test.json)
cd ../../examples
./serve_files.sh

# Run the test (in another terminal)
cd ../python/examples
python3 all_layers_test.py
```

Output:

```
✅ test.json loaded (26.4 KB)
✅ ✨ Model loaded completely! (handle: 1)
✅ All 16 layers with weights loaded automatically!
✅ Outputs match with small differences (expected with softmax)
✅ Weights successfully changed!
```

## Features

- 🧠 **8 Layer Types (All CPU)**: Dense, Conv2D, Multi-Head Attention, LayerNorm, RNN, LSTM, Softmax (10 variants), Parallel (4 combine modes)
- ✅ **Full CPU Implementation**: Every layer works on CPU with complete forward/backward passes
- 🚀 **Reliable CPU Execution**: All layers fully tested on CPU with complete forward/backward passes (GPU code exists but untested)
- 🎯 **Registry-based Initialization**: Dynamic layer creation via `call_layer_init()` for any layer type
- ⚡ **High-Level Training API**: Built-in `train()` function with automatic gradients and loss tracking
- 🎯 **Cross-Platform**: Pre-compiled binaries for Linux, macOS, Windows, Android
- 📦 **Easy Integration**: Simple Python API with high-level helpers
- 🔧 **Low-Level Access**: Direct control over layers and training loop via C-ABI
- 🏗️ **Grid Architecture**: Flexible grid-based neural network topology
- 📊 **Comprehensive Activations**: ReLU, Sigmoid, Tanh, Softplus, LeakyReLU, Linear

## API Reference

### Network Management

#### `load_model_from_string(model_json, model_id="loaded_model")` ✨

**The Easy Way!** Load a complete model (structure + all weights) from JSON string.

**Parameters:**

- `model_json` (str): JSON string containing the complete model
- `model_id` (str): Model identifier (default: "loaded_model")

**Returns:** Network handle (int)

**Example:**

```python
# Load from file
with open('model.json', 'r') as f:
    model_json = f.read()

network = welvet.load_model_from_string(model_json, "my_model")
# Done! All layers + weights loaded, ready to use
```

#### `save_model_to_string(handle, model_id="saved_model")`

Save a complete model (structure + all weights) to JSON string.

**Parameters:**

- `handle` (int): Network handle
- `model_id` (str): Model identifier (default: "saved_model")

**Returns:** JSON string containing the complete model

**Example:**

```python
model_json = welvet.save_model_to_string(network, "my_model")

# Save to file
with open('model.json', 'w') as f:
    f.write(model_json)
```

#### `create_network(input_size, grid_rows=2, grid_cols=2, layers_per_cell=3, use_gpu=False)`

Creates a new grid-based neural network.

**Parameters:**

- `input_size` (int): Number of input features
- `grid_rows` (int): Grid rows (default: 2)
- `grid_cols` (int): Grid columns (default: 2)
- `layers_per_cell` (int): Layers per grid cell (default: 3)
- `use_gpu` (bool): Enable GPU acceleration (default: False) - **Note: GPU code exists but is untested; use CPU for reliable execution**

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

- `Activation.RELU` (0) - Scaled ReLU (1.1x) activation
- `Activation.SIGMOID` (1) - Sigmoid activation
- `Activation.TANH` (2) - Tanh activation
- `Activation.SOFTPLUS` (3) - Softplus activation
- `Activation.LEAKY_RELU` (4) - LeakyReLU (0.1x negative slope)
- `Activation.LINEAR` (5) - Linear (no activation)

### Layer Initialization (Registry-based)

#### `call_layer_init(function_name, params)`

Dynamically create any layer type using the registry system.

**Parameters:**

- `function_name` (str): Name of the layer init function
  - `"InitDenseLayer"` - Fully-connected layer
  - `"InitConv2DLayer"` - 2D Convolutional layer
  - `"InitMultiHeadAttentionLayer"` - Multi-head attention layer
  - `"InitRNNLayer"` - Recurrent Neural Network layer
  - `"InitLSTMLayer"` - Long Short-Term Memory layer
- `params` (list): Parameters for the layer (varies by type)

**Returns:** LayerConfig dictionary

**Examples:**

```python
# Dense layer: [inputSize, outputSize, activation]
dense = welvet.call_layer_init("InitDenseLayer", [128, 64, welvet.Activation.RELU])

# Conv2D: [height, width, channels, filters, kernelSize, stride, padding, activation]
conv = welvet.call_layer_init("InitConv2DLayer", [28, 28, 1, 32, 3, 1, 1, welvet.Activation.RELU])

# Attention: [seqLength, dModel, numHeads, activation]
attn = welvet.call_layer_init("InitMultiHeadAttentionLayer", [10, 64, 8, welvet.Activation.TANH])

# RNN: [inputSize, hiddenSize, seqLength, outputSize]
rnn = welvet.call_layer_init("InitRNNLayer", [32, 64, 10, 640])

# LSTM: [inputSize, hiddenSize, seqLength, outputSize]
lstm = welvet.call_layer_init("InitLSTMLayer", [32, 64, 10, 640])
```

#### `list_layer_init_functions()`

Get metadata about all available layer initialization functions.

**Returns:** List of dictionaries with function metadata

```python
functions = welvet.list_layer_init_functions()
for func in functions:
    print(f"{func['Name']}: {func['Parameters']}")
```

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
import welvet

# Create network with GPU
net = welvet.create_network(
    input_size=4,
    grid_rows=1,
    grid_cols=1,
    layers_per_cell=2,
    use_gpu=True
)

# Configure architecture: 4 -> 8 -> 2
welvet.configure_sequential_network(net, [4, 8, 2])

# Training data
inputs = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
targets = [[1.0, 0.0], [0.0, 1.0]]

# Train for 50 epochs
for epoch in range(50):
    loss = welvet.train_epoch(net, inputs, targets, learning_rate=0.1)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: loss = {loss:.6f}")

# Test
output = welvet.forward(net, [0.1, 0.2, 0.3, 0.4])
print(f"Output: {output}")

# Cleanup
welvet.cleanup_gpu(net)
welvet.free_network(net)
```

### Custom Layer Configuration

```python
import welvet

# Create network
net = welvet.create_network(
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
        layer0 = welvet.init_dense_layer(10, 20, welvet.Activation.RELU)
        welvet.set_layer(net, row, col, 0, layer0)

        # Layer 1: 20 -> 15 (Tanh)
        layer1 = welvet.init_dense_layer(20, 15, welvet.Activation.TANH)
        welvet.set_layer(net, row, col, 1, layer1)

        # Layer 2: 15 -> 5 (Sigmoid)
        layer2 = welvet.init_dense_layer(15, 5, welvet.Activation.SIGMOID)
        welvet.set_layer(net, row, col, 2, layer2)

# Network is now configured
info = welvet.get_network_info(net)
print(f"Total layers: {info['total_layers']}")

welvet.free_network(net)
```

## Transformer API Reference

### Loading Models

```python
# Load tokenizer from bytes
result = welvet.load_tokenizer_from_bytes(tokenizer_bytes)
# Returns: {'success': True, 'vocab_size': 49152}

# Load transformer model
result = welvet.load_transformer_from_bytes(config_bytes, weights_bytes)
# Returns: {'success': True, 'num_layers': 30, 'hidden_size': 576, 'vocab_size': 49152}
```

### Text Processing

```python
# Encode text to token IDs
ids = welvet.encode_text("Hello world", add_special_tokens=True)
# Returns: [123, 456, 789]

# Decode token IDs to text
text = welvet.decode_tokens([123, 456, 789], skip_special_tokens=True)
# Returns: "Hello world"
```

### Generation

```python
# Generate text all at once
text = welvet.generate_text("Once upon a time", max_tokens=50, temperature=0.7)

# Generate with streaming (yields tokens one by one)
for token in welvet.generate_stream("Once upon a time", max_tokens=50, temperature=0.7):
    print(token, end='', flush=True)
```

## Testing

Run the included examples to verify installation:

```bash
# Test transformer inference
python examples/test_transformer.py ../../models/SmolLM2-135M-Instruct

# Run web interface
python examples/transformer_web_interface.py ../../models/SmolLM2-135M-Instruct 8080

# Basic GPU training test (neural networks)
python examples/train_gpu.py
```

Or test programmatically:

```python
import welvet

# Test basic functionality
net = welvet.create_network(input_size=2, grid_rows=1, grid_cols=1,
                             layers_per_cell=1, use_gpu=False)
welvet.configure_sequential_network(net, [2, 4, 2])

# Verify forward pass works
output = welvet.forward(net, [0.5, 0.5])
assert len(output) == 2, "Forward pass failed"

# Verify training works
inputs = [[0.0, 0.0], [1.0, 1.0]]
targets = [[1.0, 0.0], [0.0, 1.0]]
loss = welvet.train_epoch(net, inputs, targets, learning_rate=0.1)
assert loss > 0, "Training failed"

welvet.free_network(net)
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
