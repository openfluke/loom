#!/usr/bin/env python3
"""
Stepping Network Training Example (Python Version)
Replicates the exact logic and output format of examples/step_example/step_train_v3.go
"""

import json
import time
import random
import math
from welvet import (
    create_network_from_json,
    StepState,
    apply_gradients,
)

class TargetQueue:
    """Handles the delay between input and output in the stepping network."""
    def __init__(self, size):
        self.targets = []
        self.max_size = size

    def push(self, target):
        self.targets.append(target)

    def pop(self):
        if not self.targets:
            return None
        return self.targets.pop(0)

    def is_full(self):
        return len(self.targets) >= self.max_size

def main():
    print("=== LOOM Stepping Neural Network v3: LSTM Middle Layer ===")
    print("3-Layer Network: Dense -> LSTM -> Dense")
    print()

    # 1. Define Network Architecture
    config = {
        "batch_size": 1,
        "grid_rows": 1,
        "grid_cols": 3,
        "layers_per_cell": 1,
        "layers": [
            {
                "type": "dense",
                "input_height": 4,
                "output_height": 8,
                "activation": "relu"
            },
            {
                "type": "lstm",
                "input_size": 8,
                "hidden_size": 12,
                "seq_length": 1,
                "activation": "tanh"
            },
            {
                "type": "dense",
                "input_height": 12,
                "output_height": 3,
                "activation": "softmax"
            }
        ]
    }

    create_network_from_json(config)

    # Initialize stepping state
    input_size = 4
    state = StepState(input_size=input_size)

    # 2. Create Training Data (3 Classes)
    training_data = [
        # Class 0: Low values
        {"input": [0.1, 0.2, 0.1, 0.3], "target": [1.0, 0.0, 0.0], "label": "Low"},
        {"input": [0.2, 0.1, 0.3, 0.2], "target": [1.0, 0.0, 0.0], "label": "Low"},
        # Class 1: High values
        {"input": [0.8, 0.9, 0.8, 0.7], "target": [0.0, 1.0, 0.0], "label": "High"},
        {"input": [0.9, 0.8, 0.7, 0.9], "target": [0.0, 1.0, 0.0], "label": "High"},
        # Class 2: Mixed
        {"input": [0.1, 0.9, 0.1, 0.9], "target": [0.0, 0.0, 1.0], "label": "Mix"},
        {"input": [0.9, 0.1, 0.9, 0.1], "target": [0.0, 0.0, 1.0], "label": "Mix"},
    ]

    # 3. Setup Continuous Training Loop
    total_steps = 100000
    target_delay = 3
    target_queue = TargetQueue(target_delay)

    learning_rate = 0.015
    min_learning_rate = 0.001
    decay_rate = 0.99995
    gradient_clip_value = 1.0

    print(f"Training for {total_steps} steps (Max Speed)")
    print(f"Target Delay: {target_delay} steps (accounts for LSTM internal state)")
    print(f"LR Decay: {decay_rate:.4f} per step (min {min_learning_rate:.4f})")
    print(f"Gradient Clipping: {gradient_clip_value:.2f}")
    print()

    start_time = time.time()
    step_count = 0
    current_sample_idx = 0

    print(f"{'Step':<6} {'Input':<10} {'Output (ArgMax)':<25} {'Loss':<10}")
    print("──────────────────────────────────────────────────────────")

    while step_count < total_steps:
        # Rotate sample every 20 steps
        if step_count % 20 == 0:
            current_sample_idx = random.randint(0, len(training_data) - 1)
        
        sample = training_data[current_sample_idx]

        # B. Set Input
        state.set_input(sample["input"])

        # C. Step Forward
        state.step_forward()

        # D. Manage Target Queue
        target_queue.push(sample["target"])

        if target_queue.is_full():
            delayed_target = target_queue.pop()
            output = state.get_output()

            # F. Calculate Loss & Gradient
            loss = 0.0
            grad_output = [0.0] * len(output)

            for i in range(len(output)):
                p = output[i]
                # Clamp for numerical stability
                p = max(1e-7, min(1.0 - 1e-7, p))

                if delayed_target[i] > 0.5:
                    loss -= math.log(p)

                # Gradient for Softmax + CrossEntropy
                grad_output[i] = output[i] - delayed_target[i]

            # Apply gradient clipping
            grad_norm = sum(g * g for g in grad_output)
            grad_norm = math.sqrt(grad_norm)

            if grad_norm > gradient_clip_value:
                scale = gradient_clip_value / grad_norm
                for i in range(len(grad_output)):
                    grad_output[i] *= scale

            # G. Backward Pass
            state.step_backward(grad_output)

            # H. Update Weights
            apply_gradients(learning_rate)

            # Decay Learning Rate
            learning_rate *= decay_rate
            learning_rate = max(learning_rate, min_learning_rate)

            # I. Logging
            if step_count % 500 == 0:
                max_idx = output.index(max(output))
                max_val = output[max_idx]
                
                t_max_idx = delayed_target.index(max(delayed_target))
                
                mark = "✗"
                if max_idx == t_max_idx:
                    mark = "✓"

                print(f"{step_count:<6} {sample['label']:<10} Class {max_idx} ({max_val:.2f}) [{mark}] Exp: {t_max_idx}  Loss: {loss:.4f}  LR: {learning_rate:.5f}")

        step_count += 1

    total_time = time.time() - start_time
    print()
    print("=== Training Complete ===")
    print(f"Total Time: {total_time:.3f}s")
    print(f"Speed: {total_steps/total_time:.2f} steps/sec")
    print()

    # Final Evaluation
    print("Evaluating on all samples (with settling time)...")
    correct = 0
    settling_steps = 10

    for sample in training_data:
        state.set_input(sample["input"])
        # Settle
        for _ in range(settling_steps):
            state.step_forward()
            
        output = state.get_output()
        max_idx = output.index(max(output))
        max_val = output[max_idx]
        
        t_max_idx = sample["target"].index(max(sample["target"]))
        
        mark = "✗"
        if max_idx == t_max_idx:
            correct += 1
            mark = "✓"
            
        print(f"{mark} {sample['label']}: Pred {max_idx} ({max_val:.2f}) Exp {t_max_idx}")
        
    print(f"Final Accuracy: {correct}/{len(training_data)}")

if __name__ == "__main__":
    main()
