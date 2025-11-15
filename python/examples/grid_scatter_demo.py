#!/usr/bin/env python3
"""
Grid Scatter Multi-Agent Training Demo
Demonstrates the new simple LOOM API with save/load verification
"""

import json
from welvet import (
    create_network_from_json,
    forward_simple,
    train_simple,
    save_model_simple,
    load_model_simple,
    evaluate_network_simple,
)


def main():
    print("🤖 LOOM Python - Grid Scatter Multi-Agent Training")
    print("Task: 3 agents learn to collaborate for binary classification\n")

    # Network configuration
    config = {
        "batch_size": 1,
        "grid_rows": 1,
        "grid_cols": 3,
        "layers_per_cell": 1,
        "layers": [
            {
                "type": "dense",
                "input_size": 8,
                "output_size": 16,
                "activation": "relu"
            },
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
                    {
                        "type": "parallel",
                        "combine_mode": "add",
                        "branches": [
                            {
                                "type": "dense",
                                "input_size": 16,
                                "output_size": 8,
                                "activation": "relu"
                            },
                            {
                                "type": "dense",
                                "input_size": 16,
                                "output_size": 8,
                                "activation": "gelu"
                            }
                        ]
                    },
                    {
                        "type": "lstm",
                        "input_size": 16,
                        "hidden_size": 8,
                        "seq_length": 1
                    },
                    {
                        "type": "rnn",
                        "input_size": 16,
                        "hidden_size": 8,
                        "seq_length": 1
                    }
                ]
            },
            {
                "type": "dense",
                "input_size": 24,
                "output_size": 2,
                "activation": "sigmoid"
            }
        ]
    }

    print("Architecture:")
    print("  Shared Layer → Grid Scatter (3 agents) → Decision")
    print("  Agent 0: Feature Extractor (ensemble of 2 dense)")
    print("  Agent 1: Transformer (LSTM)")
    print("  Agent 2: Integrator (RNN)")
    print("Task: Binary classification (sum comparison)\n")

    print("Building network from JSON...")
    create_network_from_json(config)
    print("✅ Agent network created!\n")

    # Training data: binary classification based on sum
    train_batches = [
        {"Input": [0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8], "Target": [1.0, 0.0]},
        {"Input": [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1], "Target": [0.0, 1.0]},
        {"Input": [0.7, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3], "Target": [0.0, 1.0]},
        {"Input": [0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7], "Target": [1.0, 0.0]},
    ]

    training_config = {
        "Epochs": 800,
        "LearningRate": 0.15,
        "UseGPU": False,
        "PrintEveryBatch": 0,
        "GradientClip": 1.0,
        "LossType": "mse",
        "Verbose": False
    }

    print(f"Training for {training_config['Epochs']} epochs with learning rate {training_config['LearningRate']:.3f}\n")

    # Train
    result = train_simple(train_batches, training_config)

    # Test predictions
    print("📊 After Training:")
    test_inputs = [batch["Input"] for batch in train_batches]
    expected = [0, 1, 1, 0]
    
    original_preds = []
    for i, inp in enumerate(test_inputs):
        output = forward_simple(inp)
        original_preds.append(output)
        predicted_class = 0 if output[0] > output[1] else 1
        expected_class = expected[i]
        match = "✓" if predicted_class == expected_class else "✗"
        print(f"Sample {i}: [{output[0]:.3f}, {output[1]:.3f}] → Class {predicted_class} (expected {expected_class}) {match}")

    # Evaluate network
    print("\n📊 Evaluating with EvaluateNetwork...")
    metrics = evaluate_network_simple(test_inputs, expected)

    print("\n=== Evaluation Metrics ===")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Quality Score: {metrics['score']:.2f}/100")
    print(f"Average Deviation: {metrics['avg_deviation']:.2f}%")
    print(f"Failures (>100% deviation): {metrics['failures']}\n")

    print("Deviation Distribution:")
    buckets = metrics['buckets']
    total = metrics['total_samples']
    
    for bucket_name, bucket_data in buckets.items():
        count = bucket_data['count']
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 5)  # Scale to 20 chars max
        print(f"  {bucket_name:8s}: {count} samples ({pct:.1f}%) {bar}")

    print("\n✅ Multi-agent training complete!\n")

    # Save/Load verification
    print("💾 Testing model save/load...")
    model_json = save_model_simple("my_model")
    print(f"✓ Model saved ({len(model_json)} bytes)")

    print("Loading model from saved state...")
    load_model_simple(model_json, "my_model")
    print("✓ Model loaded\n")

    print("Verifying predictions match:")
    max_diff = 0.0
    all_match = True
    
    for i, inp in enumerate(test_inputs):
        output = forward_simple(inp)
        
        # Calculate difference
        diff = max(abs(output[j] - original_preds[i][j]) for j in range(len(output)))
        max_diff = max(max_diff, diff)
        
        match = "✓" if diff < 1e-6 else "✗"
        if diff >= 1e-6:
            all_match = False
        
        print(f"Sample {i}: [{output[0]:.3f}, {output[1]:.3f}] (diff: {diff:.2e}) {match}")

    if all_match:
        print("\n✅ Save/Load verification passed! All predictions match.")
    else:
        print(f"\n⚠️  Save/Load verification failed! Max difference: {max_diff:.2e}")


if __name__ == "__main__":
    main()
