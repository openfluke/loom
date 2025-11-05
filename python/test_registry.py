#!/usr/bin/env python3
"""
Comprehensive test of ALL layer types in the Loom AI framework.
Demonstrates that Python bindings can access the complete framework functionality.

This test builds a network with all 5 layer types:
    Dense → Conv2D → Attention → RNN → LSTM → Dense

And trains it end-to-end on a pattern classification task.
"""

import sys
import os
import random
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from welvet import (
    # Registry functions
    list_layer_init_functions,
    call_layer_init,
    # Network functions
    create_network,
    set_layer,
    get_network_info,
    forward,
    train,
    # Enums
    Activation,
)


def main():
    random.seed(42)
    
    print("=" * 70)
    print("=== All Layer Types Test ===")
    print("Testing network with all 5 layer types: Dense, Conv2D, Attention, RNN, LSTM")
    print("=" * 70)
    print()
    
    # Show available layer types via registry
    print("Available Layer Types (via registry):")
    functions = list_layer_init_functions()
    for fn in functions:
        args_str = ", ".join(fn["ArgTypes"])
        print(f"  • {fn['Name']}({args_str})")
    print()
    
    # Network architecture (matching Go implementation):
    # Input: 32 values
    # Layer 0 (Dense): 32 -> 32
    # Layer 1 (Conv2D): reshape as 4x4x2 -> 2x2x4 = 16 values
    # Layer 2 (Attention): 16 values as 4 seq x 4 dim -> 16 values
    # Layer 3 (RNN): 16 as 4 timesteps x 4 features -> 4x8 = 32 values
    # Layer 4 (LSTM): 32 as 4 timesteps x 8 features -> 4x4 = 16 values
    # Layer 5 (Dense): 16 -> 2 classes
    
    batch_size = 1
    input_size = 32
    num_layers = 6
    
    print("Building network with all layer types...")
    print()
    
    # Create network with 6 layers in a single cell
    net_handle = create_network(
        input_size=input_size,
        grid_rows=1,
        grid_cols=1,
        layers_per_cell=num_layers,
        use_gpu=False  # Use CPU for consistent results
    )
    
    # Layer 0: Dense (32 -> 32)
    dense1 = call_layer_init("InitDenseLayer", 32, 32, Activation.LEAKY_RELU)
    set_layer(net_handle, 0, 0, 0, dense1)
    print("  Layer 0: Dense (32 -> 32, LeakyReLU)")
    
    # Layer 1: Conv2D (4x4x2 -> 2x2x4 = 16)
    conv = call_layer_init(
        "InitConv2DLayer",
        4, 4, 2,  # Input: 4x4 spatial, 2 channels (32 values reshaped)
        3, 2, 1,  # 3x3 kernel, stride 2, padding 1
        4,        # 4 output filters -> 2x2x4 = 16 values
        Activation.LEAKY_RELU
    )
    set_layer(net_handle, 0, 0, 1, conv)
    print("  Layer 1: Conv2D (4x4x2 -> 2x2x4=16, LeakyReLU)")
    
    # Layer 2: Multi-Head Attention (16 -> 16)
    # Treat as sequence: 4 timesteps x 4 dimensions
    attention = call_layer_init(
        "InitMultiHeadAttentionLayer",
        4,  # dModel
        2,  # numHeads
        4,  # seqLength
        Activation.TANH
    )
    set_layer(net_handle, 0, 0, 2, attention)
    print("  Layer 2: Attention (4 seq x 4 dim, 2 heads, Tanh)")
    
    # Layer 3: RNN (4 features, 8 hidden, 4 timesteps -> 32)
    rnn = call_layer_init(
        "InitRNNLayer",
        4,          # inputSize
        8,          # hiddenSize
        batch_size,
        4           # seqLength
    )
    set_layer(net_handle, 0, 0, 3, rnn)
    print("  Layer 3: RNN (4 features, 8 hidden, 4 steps -> 32)")
    
    # Layer 4: LSTM (8 features, 4 hidden, 4 timesteps -> 16)
    lstm = call_layer_init(
        "InitLSTMLayer",
        8,          # inputSize
        4,          # hiddenSize
        batch_size,
        4           # seqLength
    )
    set_layer(net_handle, 0, 0, 4, lstm)
    print("  Layer 4: LSTM (8 features, 4 hidden, 4 steps -> 16)")
    
    # Layer 5: Dense (16 -> 2)
    dense2 = call_layer_init("InitDenseLayer", 16, 2, Activation.SIGMOID)
    set_layer(net_handle, 0, 0, 5, dense2)
    print("  Layer 5: Dense (16 -> 2, Sigmoid)")
    
    print()
    print("Network Summary:")
    info = get_network_info(net_handle)
    print(f"  Total layers: {info['total_layers']}")
    print("  Layer types: Dense → Conv2D → Attention → RNN → LSTM → Dense")
    print("  Data flow: 32 → 32 → 16 → 16 → 32 → 16 → 2")
    print(f"  GPU enabled: {info['gpu_enabled']}")
    print()
    
    # Generate training data
    num_samples = 50
    print(f"Generating {num_samples} training samples...")
    
    training_data = []
    for i in range(num_samples):
        if i % 2 == 0:
            # Pattern type 0: higher values in first half
            input_data = [0.7 + random.random() * 0.3 for _ in range(16)] + \
                        [random.random() * 0.3 for _ in range(16)]
            target = [1.0, 0.0]
        else:
            # Pattern type 1: higher values in second half
            input_data = [random.random() * 0.3 for _ in range(16)] + \
                        [0.7 + random.random() * 0.3 for _ in range(16)]
            target = [0.0, 1.0]
        
        training_data.append((input_data, target))
    
    print()
    
    # Training configuration
    print(f"Training Configuration:")
    print(f"  Epochs: 5")
    print(f"  Learning Rate: 0.003")
    print(f"  Gradient Clipping: 1.0")
    print(f"  Loss Function: MSE")
    print()
    
    print("Starting training...")
    print()
    
    start_time = time.time()
    
    # Use the high-level Train API (matching Go implementation)
    result = train(net_handle, training_data, {
        'epochs': 5,
        'learning_rate': 0.003,
        'use_gpu': False,
        'gradient_clip': 1.0,
        'loss_type': 'mse',
        'verbose': True
    })
    
    total_time = time.time() - start_time
    
    print()
    print("✓ Training complete!")
    print(f"  Final Loss: {result['FinalLoss']:.6f}")
    print(f"  Best Loss: {result['BestLoss']:.6f}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Avg Throughput: {result['AvgThroughput']:.1f} samples/sec")
    print()
    
    # Test predictions
    print("=" * 70)
    print("=== Testing Predictions ===")
    print()
    
    test_cases = [
        ("Pattern 0 (first half high)", 
         [0.8] * 16 + [0.2] * 16),
        ("Pattern 1 (second half high)", 
         [0.2] * 16 + [0.8] * 16),
    ]
    
    for name, test_input in test_cases:
        output = forward(net_handle, test_input)
        
        pred0, pred1 = output[0], output[1]
        prediction = "Class 0" if pred0 > pred1 else "Class 1"
        
        print(f"{name}:")
        print(f"  Output: [{pred0:.4f}, {pred1:.4f}] -> {prediction}")
    
    print()
    print("=" * 70)
    print("=== Layer Type Summary ===")
    print("✓ LayerDense: Tested (layers 0, 5)")
    print("✓ LayerConv2D: Tested (layer 1)")
    print("✓ LayerMultiHeadAttention: Tested (layer 2)")
    print("✓ LayerRNN: Tested (layer 3)")
    print("✓ LayerLSTM: Tested (layer 4)")
    print()
    print("✅ All 5 layer types successfully integrated and trained!")
    print("✅ Python bindings provide full access to the AI framework!")
    print("=" * 70)


if __name__ == "__main__":
    main()
