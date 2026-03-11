#!/usr/bin/env python3
"""
LOOM GPU Training Example
Demonstrates GPU-accelerated neural network training with the LOOM Python package.
"""

import welvet

print("="*70)
print("  LOOM GPU Training Example")
print("="*70)
print()

# Create network with GPU enabled
print("Creating neural network with GPU acceleration...")
network = welvet.create_network(
    input_size=4,
    grid_rows=1,
    grid_cols=1,
    layers_per_cell=2,
    use_gpu=True  # Enable GPU
)

# Check GPU status
info = welvet.get_network_info(network)
print(f"✓ Network created (handle={network})")
print(f"✓ GPU enabled: {info['gpu_enabled']}")
print(f"✓ Architecture: {info['grid_rows']}x{info['grid_cols']} grid, "
      f"{info['layers_per_cell']} layers/cell")
print()

# Configure network layers
print("Configuring network architecture...")
welvet.configure_sequential_network(
    network,
    layer_sizes=[4, 8, 2],  # 4 inputs -> 8 hidden -> 2 outputs
    activations=[welvet.Activation.RELU, welvet.Activation.SIGMOID]
)
print("✓ Network configured: 4 -> 8 (ReLU) -> 2 (Sigmoid)")
print()

# Prepare training data
print("Preparing training data...")
training_inputs = [
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 0.8, 0.7, 0.6],
    [0.5, 0.4, 0.3, 0.2]
]

training_targets = [
    [1.0, 0.0],  # Class 0
    [0.0, 1.0],  # Class 1
    [1.0, 0.0],  # Class 0
    [0.0, 1.0]   # Class 1
]
print(f"✓ {len(training_inputs)} training samples")
print()

# Train the network on GPU
print("Training on GPU...")
epochs = 50
learning_rate = 0.1

for epoch in range(epochs):
    loss = welvet.train_epoch(
        network,
        training_inputs,
        training_targets,
        learning_rate=learning_rate
    )
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{epochs}: loss = {loss:.6f}")

print()
print("Training complete!")
print()

# Test the trained network
print("Testing trained network...")
print("  Input                    -> Output         -> Predicted Class")
print("  " + "-"*62)

for i, (inp, target) in enumerate(zip(training_inputs, training_targets)):
    output = welvet.forward(network, inp)
    predicted_class = 1 if output[1] > output[0] else 0
    expected_class = 1 if target[1] > target[0] else 0
    
    inp_str = f"[{inp[0]:.1f}, {inp[1]:.1f}, {inp[2]:.1f}, {inp[3]:.1f}]"
    out_str = f"[{output[0]:.4f}, {output[1]:.4f}]"
    
    match = "✓" if predicted_class == expected_class else "✗"
    print(f"  {inp_str:24s} -> {out_str:18s} -> Class {predicted_class} {match}")

print()

# Cleanup
print("Cleaning up...")
welvet.cleanup_gpu(network)
welvet.free_network(network)
print("✓ GPU resources released")
print()

print("="*70)
print("  ✅ Example Complete!")
print("="*70)
print()
print("This example demonstrated:")
print("  • Creating a network with GPU acceleration")
print("  • Configuring network layers with the high-level API")
print("  • Training on GPU with automatic batching")
print("  • Testing the trained network")
print("  • Proper resource cleanup")
print()
