# Neural Tweening Adaptation Benchmarks

This directory contains examples and benchmarks for **Neural Tweening**, focusing on its ability to handle sudden task changes and adapt in real-time.

## Key Concepts

- **Neural Tweening**: A framework for training neural networks during inference (embodied learning).
- **StepTweenChain**: The most advanced mode, combining per-step execution with a chain-rule optimized tweening algorithm.
- **Mid-Stream Adaptation**: Testing how quickly a model can switch from one task (e.g., "Chase") to another (e.g., "Avoid") without catastrophic forgetting or long delay.

## Benchmarks

### [Test 17: Real-Time Adaptation](test17_realtime_decision.go)
A 15-second benchmark showing how a 6-layer Dense network handles task changes.
- **Timeline**: [Chase 5s] → [AVOID 5s] → [Chase 5s]

### [Test 18: Multi-Architecture Adaptation](test18_architecture_adaptation.go)
A comprehensive benchmark across multiple architectures, depths, and training modes.
- **Architectures**: Dense, Conv2D, RNN, LSTM, Attention
- **Depths**: 3, 5, 9 layers
- **Modes**: NormalBP, Step+BP, Tween, TweenChain, StepTweenChain

## Visualization

To visualize the results of Test 18:

1. Run the benchmark:
   ```bash
   go run test18_architecture_adaptation.go
   ```
2. This generates `test18_results.json`.
3. Open the visualization tool:
   ```bash
   go run viz_server.go
   ```
4. Navigate to `http://localhost:8000` to see interactive charts.

## Core Findings

★ **StepTweenChain** maintains high accuracy (~60-80%) even immediately after a task change.
★ **NormalBP** and standard **Tween** often lag, dropping to 0-10% accuracy while waiting for the next training batch or gradient update.
★ **Step+BP** provides good adaptation but suffers from high computational overhead, leading to lower output throughput.
