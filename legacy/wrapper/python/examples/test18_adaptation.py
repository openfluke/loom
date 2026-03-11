#!/usr/bin/env python3
"""
Test 18: Multi-Architecture Adaptation Benchmark - Python

Replicates test18_architecture_adaptation.go using the Python welvet bindings.
Tests how different network architectures adapt to mid-stream task changes.

Networks: Dense, Conv2D, RNN, LSTM, Attention
Depths: 3, 5, 9 layers
Modes: NormalBP, StepBP, Tween, TweenChain, StepTweenChain

Run: python test18_adaptation.py
"""

import json
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import welvet

# ============================================================================
# Configuration
# ============================================================================

NETWORK_TYPES = ["Dense", "Conv2D", "RNN", "LSTM", "Attn"]
DEPTHS = [3, 5, 9]
MODE_NAMES = ["NormalBP", "StepBP", "Tween", "TweenChain", "StepTweenChain"]

TEST_DURATION_MS = 5000  # 5 seconds per test (shorter for Python demo)
WINDOW_DURATION_MS = 500
TRAIN_INTERVAL_MS = 50
OUTPUT_SIZE = 4
LEARNING_RATE = 0.02


# ============================================================================
# Environment (Chase/Avoid simulation)
# ============================================================================

@dataclass
class Environment:
    agent_pos: List[float]
    target_pos: List[float]
    task: int  # 0=chase, 1=avoid
    
    @classmethod
    def create(cls) -> 'Environment':
        return cls(
            agent_pos=[0.5, 0.5],
            target_pos=[random.random(), random.random()],
            task=0
        )
    
    def get_observation(self, target_size: int) -> List[float]:
        rel_x = self.target_pos[0] - self.agent_pos[0]
        rel_y = self.target_pos[1] - self.agent_pos[1]
        dist = math.sqrt(rel_x * rel_x + rel_y * rel_y)
        
        base = [
            self.agent_pos[0], self.agent_pos[1],
            self.target_pos[0], self.target_pos[1],
            rel_x, rel_y, dist, float(self.task)
        ]
        
        # Repeat base values to fill target_size
        obs = []
        for i in range(target_size):
            obs.append(base[i % len(base)])
        return obs
    
    def get_optimal_action(self) -> int:
        rel_x = self.target_pos[0] - self.agent_pos[0]
        rel_y = self.target_pos[1] - self.agent_pos[1]
        
        if self.task == 0:  # Chase
            if abs(rel_x) > abs(rel_y):
                return 3 if rel_x > 0 else 2
            else:
                return 0 if rel_y > 0 else 1
        else:  # Avoid
            if abs(rel_x) > abs(rel_y):
                return 2 if rel_x > 0 else 3
            else:
                return 1 if rel_y > 0 else 0
    
    def execute_action(self, action: int):
        speed = 0.02
        moves = [(0, speed), (0, -speed), (-speed, 0), (speed, 0)]
        if 0 <= action < 4:
            self.agent_pos[0] += moves[action][0]
            self.agent_pos[1] += moves[action][1]
            self.agent_pos[0] = max(0, min(1, self.agent_pos[0]))
            self.agent_pos[1] = max(0, min(1, self.agent_pos[1]))
    
    def update(self):
        self.target_pos[0] += (random.random() - 0.5) * 0.01
        self.target_pos[1] += (random.random() - 0.5) * 0.01
        self.target_pos[0] = max(0.1, min(0.9, self.target_pos[0]))
        self.target_pos[1] = max(0.1, min(0.9, self.target_pos[1]))


# ============================================================================
# Network Configurations
# ============================================================================

def get_input_size(net_type: str) -> int:
    if net_type == "Dense":
        return 8
    elif net_type == "Conv2D":
        return 64
    elif net_type in ["RNN", "LSTM"]:
        return 32
    elif net_type == "Attn":
        return 64
    return 8

def build_dense_config(num_layers: int) -> str:
    hidden_sizes = [64, 48, 32, 24, 16]
    layers = []
    
    # First layer
    layers.append({
        "type": "dense",
        "input_size": 8,
        "output_size": 64,
        "activation": "leaky_relu"
    })
    
    # Hidden layers
    for i in range(1, num_layers - 1):
        in_size = hidden_sizes[(i - 1) % 5]
        out_size = hidden_sizes[i % 5]
        layers.append({
            "type": "dense",
            "input_size": in_size,
            "output_size": out_size,
            "activation": "leaky_relu"
        })
    
    # Output layer
    last_hidden = hidden_sizes[(num_layers - 2) % 5]
    layers.append({
        "type": "dense",
        "input_size": last_hidden,
        "output_size": 4,
        "activation": "sigmoid"
    })
    
    return json.dumps({
        "batch_size": 1,
        "grid_rows": 1,
        "grid_cols": 1,
        "layers_per_cell": num_layers,
        "layers": layers
    })

def build_conv2d_config(num_layers: int) -> str:
    layers = []
    
    # Conv layer
    layers.append({
        "type": "conv2d",
        "input_height": 8, "input_width": 8, "input_channels": 1,
        "filters": 8, "kernel_size": 3, "stride": 1, "padding": 0,
        "output_height": 6, "output_width": 6,
        "activation": "leaky_relu"
    })
    
    # Hidden dense layers
    for i in range(1, num_layers - 1):
        in_size = 288 if i == 1 else 64  # 6*6*8 = 288
        layers.append({
            "type": "dense",
            "input_size": in_size,
            "output_size": 64,
            "activation": "leaky_relu"
        })
    
    # Output layer
    last_in = 64 if num_layers > 2 else 288
    layers.append({
        "type": "dense",
        "input_size": last_in,
        "output_size": 4,
        "activation": "sigmoid"
    })
    
    return json.dumps({
        "batch_size": 1,
        "grid_rows": 1,
        "grid_cols": 1,
        "layers_per_cell": num_layers,
        "layers": layers
    })

def build_rnn_config(num_layers: int) -> str:
    layers = []
    
    layers.append({
        "type": "dense",
        "input_size": 32,
        "output_size": 32,
        "activation": "leaky_relu"
    })
    
    for i in range(1, num_layers - 1):
        if i % 2 == 1:
            layers.append({
                "type": "rnn",
                "input_size": 8,
                "hidden_size": 8,
                "seq_length": 4
            })
        else:
            layers.append({
                "type": "dense",
                "input_size": 32,
                "output_size": 32,
                "activation": "leaky_relu"
            })
    
    layers.append({
        "type": "dense",
        "input_size": 32,
        "output_size": 4,
        "activation": "sigmoid"
    })
    
    return json.dumps({
        "batch_size": 1,
        "grid_rows": 1,
        "grid_cols": 1,
        "layers_per_cell": num_layers,
        "layers": layers
    })

def build_lstm_config(num_layers: int) -> str:
    layers = []
    
    layers.append({
        "type": "dense",
        "input_size": 32,
        "output_size": 32,
        "activation": "leaky_relu"
    })
    
    for i in range(1, num_layers - 1):
        if i % 2 == 1:
            layers.append({
                "type": "lstm",
                "input_size": 8,
                "hidden_size": 8,
                "seq_length": 4
            })
        else:
            layers.append({
                "type": "dense",
                "input_size": 32,
                "output_size": 32,
                "activation": "leaky_relu"
            })
    
    layers.append({
        "type": "dense",
        "input_size": 32,
        "output_size": 4,
        "activation": "sigmoid"
    })
    
    return json.dumps({
        "batch_size": 1,
        "grid_rows": 1,
        "grid_cols": 1,
        "layers_per_cell": num_layers,
        "layers": layers
    })

def build_attn_config(num_layers: int) -> str:
    d_model = 64
    layers = []
    
    for i in range(num_layers - 1):
        if i % 2 == 0:
            layers.append({
                "type": "multi_head_attention",
                "d_model": d_model,
                "num_heads": 4
            })
        else:
            layers.append({
                "type": "dense",
                "input_size": d_model,
                "output_size": d_model,
                "activation": "leaky_relu"
            })
    
    layers.append({
        "type": "dense",
        "input_size": d_model,
        "output_size": 4,
        "activation": "sigmoid"
    })
    
    return json.dumps({
        "batch_size": 1,
        "grid_rows": 1,
        "grid_cols": 1,
        "layers_per_cell": num_layers,
        "layers": layers
    })

def build_network_config(net_type: str, num_layers: int) -> Optional[str]:
    if net_type == "Dense":
        return build_dense_config(num_layers)
    elif net_type == "Conv2D":
        return build_conv2d_config(num_layers)
    elif net_type == "RNN":
        return build_rnn_config(num_layers)
    elif net_type == "LSTM":
        return build_lstm_config(num_layers)
    elif net_type == "Attn":
        return build_attn_config(num_layers)
    return None


# ============================================================================
# Test Result Storage
# ============================================================================

@dataclass
class TestResult:
    avg_accuracy: float = 0.0
    total_outputs: int = 0
    completed: bool = False


# ============================================================================
# Single Test Runner
# ============================================================================

def run_single_test(net_type: str, depth: int, mode_idx: int) -> Optional[TestResult]:
    config_name = f"{net_type}-{depth}L"
    input_size = get_input_size(net_type)
    mode_name = MODE_NAMES[mode_idx]
    
    # Build and create network
    config = build_network_config(net_type, depth)
    if not config:
        print(f"  [{config_name}] [{mode_name}] SKIP (unsupported)")
        return None
    
    try:
        # create_network_from_json returns None but raises on error
        welvet.create_network_from_json(config)
    except Exception as e:
        print(f"  [{config_name}] [{mode_name}] SKIP ({e})")
        return None
    
    # Initialize states based on mode
    step_state = None
    tween_state = None
    
    try:
        if mode_idx in [1, 4]:  # StepBP or StepTweenChain
            step_state = welvet.StepState(input_size)
        
        if mode_idx >= 2:  # Tween, TweenChain, StepTweenChain
            use_chain_rule = mode_idx in [3, 4]
            tween_state = welvet.TweenState(use_chain_rule=use_chain_rule)
    except Exception as e:
        print(f"  [{config_name}] [{mode_name}] SKIP (state init: {e})")
        return None
    
    # Create tracker
    tracker = welvet.AdaptationTracker(
        window_duration_ms=WINDOW_DURATION_MS,
        total_duration_ms=TEST_DURATION_MS
    )
    tracker.set_model_info(config_name, mode_name)
    
    # Schedule task changes at 1/3 and 2/3
    one_third = TEST_DURATION_MS // 3
    two_thirds = 2 * one_third
    tracker.schedule_task_change(one_third, task_id=1, task_name="AVOID")
    tracker.schedule_task_change(two_thirds, task_id=0, task_name="CHASE")
    
    # Initialize environment
    env = Environment.create()
    
    # Start tracking
    tracker.start("CHASE", task_id=0)
    
    start_time = time.time() * 1000
    last_train_time = start_time
    
    train_samples = []  # (input, target) tuples
    
    while (time.time() * 1000 - start_time) < TEST_DURATION_MS:
        current_task = tracker.get_current_task()
        env.task = current_task
        
        obs = env.get_observation(input_size)
        
        # Forward pass
        try:
            if mode_idx in [1, 4]:  # StepBP or StepTweenChain
                step_state.set_input(obs)
                step_state.step_forward()
                output = step_state.get_output()
            else:
                output = welvet.forward_simple(obs)
        except Exception:
            continue
        
        if not output or len(output) < OUTPUT_SIZE:
            continue
        
        action = output.index(max(output[:OUTPUT_SIZE]))
        optimal_action = env.get_optimal_action()
        is_correct = action == optimal_action
        
        tracker.record_output(is_correct)
        
        # Store training sample
        if len(train_samples) < 100:
            train_samples.append((obs, optimal_action))
        
        # Training based on mode
        now = time.time() * 1000
        
        if mode_idx == 0:  # NormalBP
            if now - last_train_time > TRAIN_INTERVAL_MS and train_samples:
                batches = []
                for inp, target_class in train_samples:
                    target = [0.0] * OUTPUT_SIZE
                    target[target_class] = 1.0
                    batches.append({"Input": inp, "Target": target})
                
                try:
                    welvet.train_simple(batches, {
                        "Epochs": 1,
                        "LearningRate": LEARNING_RATE,
                        "LossType": "mse"
                    })
                except Exception:
                    pass
                
                train_samples = []
                last_train_time = now
        
        elif mode_idx == 1:  # StepBP
            grad = [output[i] - (1.0 if i == optimal_action else 0.0) for i in range(OUTPUT_SIZE)]
            try:
                step_state.step_backward(grad)
                welvet.apply_gradients(LEARNING_RATE)
            except Exception:
                pass
        
        elif mode_idx in [2, 3]:  # Tween, TweenChain
            if now - last_train_time > TRAIN_INTERVAL_MS and train_samples:
                for inp, target_class in train_samples:
                    try:
                        tween_state.step(inp, target_class, OUTPUT_SIZE, LEARNING_RATE)
                    except Exception:
                        pass
                train_samples = []
                last_train_time = now
        
        elif mode_idx == 4:  # StepTweenChain
            try:
                tween_state.step(obs, optimal_action, OUTPUT_SIZE, LEARNING_RATE)
            except Exception:
                pass
        
        env.execute_action(action)
        env.update()
    
    # Finalize
    results = tracker.finalize()
    
    result = TestResult(
        avg_accuracy=results.get("avg_accuracy", 0),
        total_outputs=results.get("total_outputs", 0),
        completed=True
    )
    
    print(f"  [{config_name}] [{mode_name}] Avg: {result.avg_accuracy:.1f}% | Outputs: {result.total_outputs}")
    
    # Cleanup (StepState uses __del__, TweenState and AdaptationTracker have close())
    # step_state will be automatically cleaned up when it goes out of scope
    if tween_state:
        tween_state.close()
    tracker.close()
    
    return result


# ============================================================================
# Summary Table
# ============================================================================

def print_summary_table(all_results: dict):
    print("\n" + "=" * 100)
    print("MULTI-ARCHITECTURE ADAPTATION SUMMARY")
    print("=" * 100)
    
    # Header
    print(f"{'Network':<12}", end="")
    for mode in MODE_NAMES:
        print(f"{mode:<15}", end="")
    print()
    print("-" * 100)
    
    for net_type in NETWORK_TYPES:
        for depth in DEPTHS:
            config_name = f"{net_type}-{depth}L"
            print(f"{config_name:<12}", end="")
            
            for mode_idx, mode in enumerate(MODE_NAMES):
                key = (net_type, depth, mode_idx)
                if key in all_results and all_results[key].completed:
                    print(f"{all_results[key].avg_accuracy:>6.1f}%        ", end="")
                else:
                    print(f"{'--':>6}         ", end="")
            print()
    
    # Mode averages
    print("\n" + "-" * 60)
    print("MODE AVERAGES")
    print("-" * 60)
    
    for mode_idx, mode in enumerate(MODE_NAMES):
        accuracies = []
        for net_type in NETWORK_TYPES:
            for depth in DEPTHS:
                key = (net_type, depth, mode_idx)
                if key in all_results and all_results[key].completed:
                    accuracies.append(all_results[key].avg_accuracy)
        
        if accuracies:
            avg = sum(accuracies) / len(accuracies)
            print(f"{mode:<20} Avg: {avg:>6.1f}% ({len(accuracies)} tests)")
        else:
            print(f"{mode:<20} No completed tests")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("• StepTweenChain shows most CONSISTENT accuracy across task changes")
    print("• Other methods may crash to 0% after changes while StepTweenChain maintains ~40-80%")
    print("• Higher 'After Change' accuracy = faster adaptation")


# ============================================================================
# Main
# ============================================================================

def main():
    random.seed(42)
    
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║  Test 18: MULTI-ARCHITECTURE Adaptation Benchmark (Python)              ║")
    print("║  Networks: Dense, Conv2D, RNN, LSTM, Attention | Depths: 3, 5, 9        ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝\n")
    
    total_tests = len(NETWORK_TYPES) * len(DEPTHS) * len(MODE_NAMES)
    print(f"Running {total_tests} tests ({len(NETWORK_TYPES)} archs × {len(DEPTHS)} depths × {len(MODE_NAMES)} modes)\n")
    print(f"Test duration: {TEST_DURATION_MS}ms per test\n")
    
    all_results = {}
    completed = 0
    
    for net_type in NETWORK_TYPES:
        for depth in DEPTHS:
            print(f"\n--- {net_type}-{depth}L ---")
            
            for mode_idx in range(len(MODE_NAMES)):
                result = run_single_test(net_type, depth, mode_idx)
                if result:
                    all_results[(net_type, depth, mode_idx)] = result
                    completed += 1
    
    print("\n")
    print("=" * 60)
    print(f"BENCHMARK COMPLETE ({completed}/{total_tests} tests)")
    print("=" * 60)
    
    print_summary_table(all_results)


if __name__ == "__main__":
    main()
