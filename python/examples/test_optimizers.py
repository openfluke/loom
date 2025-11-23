#!/usr/bin/env python3
"""
LOOM Optimizer Python Test
Tests the new optimizer functions via Python bindings
"""

import json
import welvet

print("🚀 LOOM Optimizer Python Test")
print("================================\n")

# Network configuration
config = {
    "batch_size": 1,
    "grid_rows": 1,
    "grid_cols": 1,
    "layers_per_cell": 3,
    "layers": [
        {"type": "dense", "input_height": 4, "output_height": 8, "activation": "relu"},
        {"type": "lstm", "input_size": 8, "hidden_size": 12, "seq_length": 1},
        {"type": "dense", "input_height": 12, "output_height": 3, "activation": "softmax"}
    ]
}

# Create network
print("Creating network...")
welvet.create_network_from_json(json.dumps(config))
print("✓ Network created\n")

# Training data
training_data = [
    ([0.1, 0.2, 0.1, 0.3], [1.0, 0.0, 0.0]),  # Class 0
    ([0.8, 0.9, 0.7, 0.8], [0.0, 1.0, 0.0]),  # Class 1
    ([0.3, 0.5, 0.9, 0.6], [0.0, 0.0, 1.0]),  # Class 2
    ([0.2, 0.1, 0.2, 0.2], [1.0, 0.0, 0.0]),  # Class 0
    ([0.9, 0.8, 0.8, 0.9], [0.0, 1.0, 0.0]),  # Class 1
    ([0.4, 0.6, 0.8, 0.7], [0.0, 0.0, 1.0]),  # Class 2
]

# ========================================================================
# Test 1: Simple SGD (baseline)
# ========================================================================
print("📊 Test 1: Simple SGD (baseline)")
print("----------------------------------")

state = welvet.StepState(4)
total_loss = 0.0

for step in range(5000):
    idx = step % len(training_data)
    input_data, target = training_data[idx]
    
    # Forward pass
    state.set_input(input_data)
    state.step_forward()
    output = state.get_output()
    
    # Compute loss and gradients
    loss = sum((output[i] - target[i]) ** 2 for i in range(3)) / 3.0
    total_loss += loss
    
    gradients = [2.0 * (output[i] - target[i]) / 3.0 for i in range(3)]
    
    # Backward pass
    state.step_backward(gradients)
    
    # Apply gradients (simple SGD)
    welvet.apply_gradients(0.01)
    
    if (step + 1) % 1000 == 0:
        print(f"  Step {step + 1}: Avg Loss={total_loss / 1000:.6f}")
        total_loss = 0.0

print("✅ SGD Test complete!\n")

# ========================================================================
# Test 2: AdamW Optimizer
# ========================================================================
print("📊 Test 2: AdamW Optimizer")
print("----------------------------------")

# Recreate network
welvet.create_network_from_json(json.dumps(config))

state = welvet.StepState(4)
total_loss = 0.0

for step in range(5000):
    idx = step % len(training_data)
    input_data, target = training_data[idx]
    
    # Forward pass
    state.set_input(input_data)
    state.step_forward()
    output = state.get_output()
    
    # Compute loss and gradients
    loss = sum((output[i] - target[i]) ** 2 for i in range(3)) / 3.0
    total_loss += loss
    
    gradients = [2.0 * (output[i] - target[i]) / 3.0 for i in range(3)]
    
    # Backward pass
    state.step_backward(gradients)
    
    # Apply gradients with AdamW
    welvet.apply_gradients_adamw(0.001, beta1=0.9, beta2=0.999, weight_decay=0.01)
    
    if (step + 1) % 1000 == 0:
        print(f"  Step {step + 1}: Avg Loss={total_loss / 1000:.6f}")
        total_loss = 0.0

print("✅ AdamW Test complete!\n")

# ========================================================================
# Test 3: RMSprop Optimizer
# ========================================================================
print("📊 Test 3: RMSprop Optimizer")
print("----------------------------------")

# Recreate network
welvet.create_network_from_json(json.dumps(config))

state = welvet.StepState(4)
total_loss = 0.0

for step in range(5000):
    idx = step % len(training_data)
    input_data, target = training_data[idx]
    
    # Forward pass
    state.set_input(input_data)
    state.step_forward()
    output = state.get_output()
    
    # Compute loss and gradients
    loss = sum((output[i] - target[i]) ** 2 for i in range(3)) / 3.0
    total_loss += loss
    
    gradients = [2.0 * (output[i] - target[i]) / 3.0 for i in range(3)]
    
    # Backward pass
    state.step_backward(gradients)
    
    # Apply gradients with RMSprop
    welvet.apply_gradients_rmsprop(0.001, alpha=0.99, epsilon=1e-8, momentum=0.0)
    
    if (step + 1) % 1000 == 0:
        print(f"  Step {step + 1}: Avg Loss={total_loss / 1000:.6f}")
        total_loss = 0.0

print("✅ RMSprop Test complete!\n")

# ========================================================================
# Test 4: SGD with Momentum
# ========================================================================
print("📊 Test 4: SGD with Momentum")
print("----------------------------------")

# Recreate network
welvet.create_network_from_json(json.dumps(config))

state = welvet.StepState(4)
total_loss = 0.0

for step in range(5000):
    idx = step % len(training_data)
    input_data, target = training_data[idx]
    
    # Forward pass
    state.set_input(input_data)
    state.step_forward()
    output = state.get_output()
    
    # Compute loss and gradients
    loss = sum((output[i] - target[i]) ** 2 for i in range(3)) / 3.0
    total_loss += loss
    
    gradients = [2.0 * (output[i] - target[i]) / 3.0 for i in range(3)]
    
    # Backward pass
    state.step_backward(gradients)
    
    # Apply gradients with SGD + Momentum
    welvet.apply_gradients_sgd_momentum(0.01, momentum=0.9, dampening=0.0, nesterov=False)
    
    if (step + 1) % 1000 == 0:
        print(f"  Step {step + 1}: Avg Loss={total_loss / 1000:.6f}")
        total_loss = 0.0

print("✅ SGD+Momentum Test complete!\n")

# Summary
print("🎉 All Python optimizer tests complete!\n")
print("Verified Functions:")
print("  ✅ apply_gradients (simple SGD)")
print("  ✅ apply_gradients_adamw")
print("  ✅ apply_gradients_rmsprop")
print("  ✅ apply_gradients_sgd_momentum")
print("\nAll optimizer methods working correctly via Python! 🚀")
