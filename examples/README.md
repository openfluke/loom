# LOOM Neural Network Examples

This directory contains comprehensive examples demonstrating LOOM's unique neural network capabilities, particularly its flexible softmax layer system.

## Table of Contents

- [Layer Types Overview](#layer-types-overview)
- [Softmax Layer - The Unique Feature](#softmax-layer---the-unique-feature)
- [Examples Guide](#examples-guide)
- [Key Concepts](#key-concepts)
- [Quick Start](#quick-start)

---

## Layer Types Overview

LOOM supports 6 layer types that can be mixed and matched:

| Layer Type             | Purpose                | Example Use Case                            |
| ---------------------- | ---------------------- | ------------------------------------------- |
| **Dense**              | Fully-connected layer  | Standard neural networks                    |
| **Conv2D**             | 2D Convolution         | Image processing, spatial features          |
| **MultiHeadAttention** | Transformer attention  | Sequence modeling, relational reasoning     |
| **RNN**                | Recurrent network      | Temporal sequences                          |
| **LSTM**               | Long Short-Term Memory | Long-term dependencies                      |
| **Softmax**            | 10 different variants  | Action selection, probability distributions |

---

## Softmax Layer - The Unique Feature

### What Makes LOOM Different?

Most frameworks treat softmax as a function you manually apply at the output. LOOM makes **softmax a first-class layer** with **10 built-in variants**, and you can use it **anywhere** in your network (hidden layers OR output).

### The 10 Softmax Variants

| Variant             | Purpose                           | When to Use                                    |
| ------------------- | --------------------------------- | ---------------------------------------------- |
| **1. Standard**     | One probability distribution      | Classification tasks                           |
| **2. Grid**         | Independent distributions per row | Multi-agent action selection                   |
| **3. Hierarchical** | Nested decision trees             | Strategy → Tactic → Action                     |
| **4. Temperature**  | Control exploration/exploitation  | Adjustable confidence (low=sharp, high=smooth) |
| **5. Gumbel**       | Add exploration noise             | Training with randomness                       |
| **6. Masked**       | Filter illegal options            | Legal moves only in games                      |
| **7. Sparsemax**    | Exact zeros in output             | Interpretable attention                        |
| **8. Entmax**       | Blend softmax/sparsemax           | Moderate sparsity (α=1.0→2.0)                  |
| **9. Adaptive**     | Hierarchical vocabulary           | Large output spaces                            |
| **10. Mixture**     | Blend multiple distributions      | Ensemble decisions                             |

### Grid Softmax Explained

Grid softmax is revolutionary for game AI:

```
Standard Softmax (12 outputs):
[0.08, 0.09, 0.11, 0.15, 0.07, 0.06, 0.12, 0.08, 0.09, 0.07, 0.04, 0.04]
↳ All 12 values compete, sum to 1.0

Grid Softmax (3 agents × 4 actions):
Agent 0: [0.25, 0.30, 0.20, 0.25] ← sum = 1.0
Agent 1: [0.40, 0.20, 0.15, 0.25] ← sum = 1.0
Agent 2: [0.10, 0.50, 0.25, 0.15] ← sum = 1.0
↳ Each row is independent!
```

**Used in:** AlphaStar (StarCraft), OpenAI Five (Dota), Multi-agent robotics

---

## Examples Guide

### Basic Examples

#### `all_layers_validation.go` ✅

**Purpose:** Validate all 5 layer types work correctly

**What it proves:**

- Dense → Conv2D → Attention → RNN → LSTM → Dense stack
- 200 epochs training
- 93.6% loss reduction
- Perfect binary classification

**Key takeaway:** Complex layer stacks can learn effectively

```bash
go run all_layers_validation.go
```

---

### Softmax Variants

#### `softmax_variants_demo.go` 🎯

**Purpose:** Demonstrate all 10 softmax variants

**What you'll see:**

- Standard softmax: Basic probabilities
- Grid softmax: 3 agents × 4 actions (independent)
- Temperature: Sharp (0.1) vs smooth (5.0)
- Gumbel: Different output each run (exploration noise)
- Masked: Positions 1 and 3 forced to zero
- Sparsemax: Exact zeros in output
- Entmax: Blend between softmax and sparsemax
- Practical game AI example with 3 units

**Key insight:** Same network, different softmax = different behavior!

```bash
go run softmax_variants_demo.go
```

---

### Multi-Agent AI

#### `multi_agent_demo.go` 🤖

**Purpose:** One network controls multiple agents

**Architecture:**

```
Input (64) → Dense → CNN → LSTM → Attention → Dense → Grid Softmax
                                                         ↓
                                          3 agents × 4 actions = 12 outputs
```

**What it teaches:**

- **Shared network** learns general strategies
- **Grid softmax** applies them independently per agent
- Each agent gets its own action distribution
- Used in AlphaStar for controlling 200+ units

```bash
go run multi_agent_demo.go
```

---

### Hierarchical Decisions

#### `hierarchical_softmax_demo.go` 🌳

**Purpose:** Multi-level decision trees

**Example structure:**

```
Level 1: Strategy (attack/defend/scout)
    ↓
Level 2: Unit assignment (which unit executes)
    ↓
Level 3: Action (move/shoot/ability/idle)
```

**Output:** 3 strategies × 3 units × 4 actions = 36 outputs

**Two modes:**

1. **Hierarchical:** Choose strategy → then unit → then action
2. **Flat grid:** All 9 combos decide independently

```bash
go run hierarchical_softmax_demo.go
```

---

### Advanced Examples

#### `multi_softmax_network.go` 🔥

**Purpose:** Multiple DIFFERENT softmax types in ONE network

**Network layers:**

```
Layer 0: Dense
Layer 1: Dense
Layer 2: SPARSEMAX (hidden layer - sparse gating)
Layer 3: Dense
Layer 4: GRID SOFTMAX (hidden layer - routing)
Layer 5: Dense
Layer 6: MASKED SOFTMAX (hidden layer - filtering)
Layer 7: Dense
Layer 8: HIERARCHICAL SOFTMAX (hidden layer)
Layer 9: TEMPERATURE SOFTMAX (output layer)
```

**Mind-blowing fact:** You can use softmax in HIDDEN layers, not just output!

**Use cases for hidden softmax:**

- Attention mechanisms (which features to focus on)
- Gating (which paths to activate)
- Routing (mixture of experts)
- Feature selection (sparse activation)

```bash
go run multi_softmax_network.go
```

#### `softmax_sandwich_demo.go` 🥪

**Purpose:** Softmax in hidden AND output positions

**Architecture:**

```
Layer 0: Dense
Layer 1: Grid Softmax (HIDDEN) - learns which features to emphasize
Layer 2: Dense
Layer 3: Sparsemax (HIDDEN) - sparse feature selection
Layer 4: Dense
Layer 5: Standard Softmax (OUTPUT) - final decision
```

**What it teaches:**

- Hidden softmax layers learn attention/gating
- Network trains end-to-end (71.7% loss reduction)
- Softmax can be used for internal routing

```bash
go run softmax_sandwich_demo.go
```

---

### Game AI Examples

#### `game_ai_fusion.go` 🎮

**Purpose:** Multi-modal fusion for game AI

**Architecture:**

```
Dense → CNN (spatial) → LSTM (temporal) → MHA (relational) → Dense → Dense
```

**Task:** Learn when to attack vs retreat based on:

- Enemy distance (close/far)
- Health status (high/low)

**Demonstrates:**

- Combining multiple layer types
- Manual softmax application
- Decision-making from game state

```bash
go run game_ai_fusion.go
```

#### `softmax_comparison.go` 📊

**Purpose:** Visual comparison of global vs grid softmax

**Example output:**

```
GLOBAL SOFTMAX (4 units × 3 actions = 12 outputs compete):
All values: sum = 1.0 (highest wins everything)

GRID SOFTMAX (4 independent distributions):
Unit 0: [0.475, 0.288, 0.236] sum=1.0 → attack
Unit 1: [0.236, 0.475, 0.288] sum=1.0 → defend
Unit 2: [0.288, 0.236, 0.475] sum=1.0 → move
Unit 3: [0.333, 0.333, 0.333] sum=1.0 → confused
```

**Key insight:** Grid softmax enables independent multi-agent decisions!

```bash
go run softmax_comparison.go
```

---

### Serialization

#### `softmax_save_load_demo.go` 💾

**Purpose:** Prove softmax layers save/load correctly

**What it tests:**

- Save network with Grid, Masked, and Temperature softmax
- Load from JSON file
- Verify outputs match perfectly
- Check all configuration preserved (rows, cols, temperature, mask)

**Result:** 0.0 difference between saved and loaded networks!

```bash
go run softmax_save_load_demo.go
```

---

## Key Concepts

### 1. Softmax as a Layer Type

**Traditional approach (PyTorch/TensorFlow):**

```python
# Softmax is a function, not a layer
output = model(input)
probs = torch.softmax(output, dim=-1)  # Manual application
```

**LOOM approach:**

```go
// Softmax is a layer, just like Dense or Conv2D
softmax := nn.InitGridSoftmaxLayer(3, 4)
network.SetLayer(0, 0, 5, softmax)
// Automatically applied during forward pass!
```

### 2. Grid Softmax for Multi-Agent

**The Problem:** How do you make ONE network output actions for MULTIPLE agents?

**The Solution:** Grid softmax!

```go
// 3 agents, 4 actions each = 12 outputs
network := nn.InitGridSoftmaxLayer(3, 4)

// Network outputs 12 values
// Grid softmax applies 3 independent softmax operations:
//   Rows 0-3:   Agent 0's actions (sum=1.0)
//   Rows 4-7:   Agent 1's actions (sum=1.0)
//   Rows 8-11:  Agent 2's actions (sum=1.0)
```

**Used by:** AlphaStar (200+ StarCraft units), OpenAI Five (5 Dota heroes)

### 3. Hierarchical Softmax for Strategy Trees

**The Problem:** You want nested decisions (strategy → tactic → action)

**The Solution:** Hierarchical softmax!

```go
// 3 strategies × 3 units × 4 actions = 36 outputs
hierarchical := nn.InitHierarchicalSoftmaxLayer([]int{3, 3, 4})
```

Outputs form a decision tree where each level gets its own probability distribution.

### 4. Masked Softmax for Legal Moves

**The Problem:** In games, not all actions are always legal

**The Solution:** Masked softmax!

```go
masked := nn.InitMaskedSoftmaxLayer(6)
// Abilities on cooldown? Mask them out!
masked.Mask = []bool{true, false, true, true, false, true}
// Positions 1 and 4 will be forced to ~0.0
```

### 5. Temperature Softmax for Exploration

**The Problem:** Want to control exploration vs exploitation?

**The Solution:** Temperature softmax!

```go
// Low temperature (0.1) = sharp/confident/exploit
exploit := nn.InitTemperatureSoftmaxLayer(0.1)
// Output: [0.01, 0.01, 0.98] - very confident!

// High temperature (5.0) = smooth/exploratory
explore := nn.InitTemperatureSoftmaxLayer(5.0)
// Output: [0.32, 0.34, 0.34] - very uncertain
```

### 6. Softmax in Hidden Layers

**Mind-blowing discovery:** Softmax can be used ANYWHERE in the network!

**Use cases:**

- **Attention:** Which features to focus on
- **Gating:** Which neurons to activate
- **Routing:** Which expert to use
- **Sparse selection:** Turn off irrelevant features

```go
network := nn.NewNetwork(64, 1, 1, 5)
network.SetLayer(0, 0, 0, nn.InitDenseLayer(64, 64, nn.ActivationLeakyReLU))
network.SetLayer(0, 0, 1, nn.InitSparsemaxLayer())  // ← HIDDEN softmax!
network.SetLayer(0, 0, 2, nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU))
network.SetLayer(0, 0, 3, nn.InitDenseLayer(32, 10, nn.ActivationLeakyReLU))
network.SetLayer(0, 0, 4, nn.InitSoftmaxLayer())     // ← OUTPUT softmax!
```

---

## Quick Start

### Basic Network (Single Output)

```go
network := nn.NewNetwork(64, 1, 1, 3)

// Layer 0: Input processing
dense1 := nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU)
network.SetLayer(0, 0, 0, dense1)

// Layer 1: Hidden layer
dense2 := nn.InitDenseLayer(32, 6, nn.ActivationLeakyReLU)
network.SetLayer(0, 0, 1, dense2)

// Layer 2: Output with softmax
softmax := nn.InitSoftmaxLayer()
network.SetLayer(0, 0, 2, softmax)

// Forward pass
output, _ := network.ForwardCPU(input)
// output is now a probability distribution!
```

### Multi-Agent Network

```go
network := nn.NewNetwork(64, 1, 1, 4)

// Layers 0-2: Shared processing
dense1 := nn.InitDenseLayer(64, 64, nn.ActivationLeakyReLU)
network.SetLayer(0, 0, 0, dense1)

dense2 := nn.InitDenseLayer(64, 12, nn.ActivationLeakyReLU)
network.SetLayer(0, 0, 1, dense2)

// Layer 2: Grid softmax for 3 agents × 4 actions
gridSoftmax := nn.InitGridSoftmaxLayer(3, 4)
network.SetLayer(0, 0, 2, gridSoftmax)

// Forward pass
output, _ := network.ForwardCPU(input)

// Extract actions per agent
for agent := 0; agent < 3; agent++ {
    agentActions := output[agent*4 : agent*4+4]
    // Each agent has its own probability distribution!
}
```

### Game AI with Legal Moves

```go
network := nn.NewNetwork(64, 1, 1, 3)

// Processing layers...
dense := nn.InitDenseLayer(64, 6, nn.ActivationLeakyReLU)
network.SetLayer(0, 0, 0, dense)

// Masked softmax for legal moves only
masked := nn.InitMaskedSoftmaxLayer(6)
network.SetLayer(0, 0, 1, masked)

// During gameplay, update mask based on legal moves
layer := network.GetLayer(0, 0, 1)
layer.Mask = []bool{true, false, true, true, false, true}
// Moves 1 and 4 are illegal, will be forced to ~0.0

output, _ := network.ForwardCPU(gameState)
// Only legal moves will have non-zero probabilities!
```

---

## Comparison with Other Frameworks

| Feature              | LOOM                  | PyTorch/TensorFlow          |
| -------------------- | --------------------- | --------------------------- |
| Softmax as layer     | ✅ First-class layer  | ❌ Manual function call     |
| Softmax variants     | ✅ 10 built-in types  | ❌ Implement yourself       |
| Grid softmax         | ✅ Built-in           | ❌ Manual reshaping         |
| Masked softmax       | ✅ Built-in           | ❌ Manual masking with -inf |
| Hidden softmax       | ✅ Use anywhere       | ⚠️ Possible but manual      |
| Hierarchical softmax | ✅ Built-in           | ❌ Implement yourself       |
| Serialization        | ✅ All variants saved | ⚠️ Custom implementation    |

---

## What We Learned

1. **Softmax is more than just output activation** - It can be used as a hidden layer for attention, gating, and routing

2. **Grid softmax unlocks multi-agent AI** - One network can control many agents simultaneously with independent decisions

3. **Temperature controls exploration** - Low temp = exploit (confident), high temp = explore (uncertain)

4. **Masking enables legal moves** - Essential for game AI where not all actions are always valid

5. **Hierarchical softmax enables strategy trees** - Natural way to represent nested decisions

6. **LOOM is uniquely flexible** - No other framework treats softmax as a first-class layer with 10 variants

7. **You can mix MULTIPLE softmax types** - Grid + Masked + Temperature in the same network!

8. **Serialization preserves everything** - Save/load works perfectly with all softmax variants

---

## Performance Notes

- **Softmax has no trainable weights** - It's a pure activation/normalization layer
- **Grid softmax** is just multiple standard softmax operations (rows × independent)
- **Temperature scaling** is a simple division before exp (very fast)
- **Masked softmax** sets masked positions to -1e9 before softmax (efficient)
- **Sparsemax** is more expensive than softmax (requires sorting)

---

## Next Steps

Want to build your own game AI? Start with:

1. **`multi_agent_demo.go`** - Learn grid softmax
2. **`game_ai_fusion.go`** - Learn multi-modal architectures
3. **`multi_softmax_network.go`** - Learn advanced patterns

Want to experiment? Try:

- Combining Temperature + Masked softmax for exploration with legal moves
- Using Sparsemax in hidden layers for interpretable attention
- Building hierarchical strategies for complex games

---

## License

Apache 2.0 - Same as LOOM framework

## Questions?

This is experimental territory that most frameworks don't explore. If you discover new patterns or use cases, please share them!

**The key insight:** Softmax is not just for output - it's a powerful tool for routing, attention, and multi-agent coordination anywhere in your network! 🚀
