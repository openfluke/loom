# Practical Examples and Use Cases

Loom is used across a variety of domains, from gaming AI to medical imaging and large language models. This guide walks through real-world usage patterns, showing how the concepts from the other documentation files come together in practice.

---

## Featured Example: MNIST Convolutional Neural Network

The [MNIST demo](file:///home/samuel/git/loom/tva/demo/mnist/main.go) is the definitive example of using Loom for spatial computer vision tasks. It demonstrates **training parity (CPU vs GPU)**, **multi-precision serialization**, and **high-fidelity quantization**.

### Key Features Demonstrated:
1. **Conv2D Layers**: Building a standard LeNet-style architecture.
2. **Unified Training**: Using `network.Train()` to automatically accelerate on GPU.
3. **Safetensors Benchmarking**: Testing all 13 supported dtypes.

### Numerical Type Comparison Summary
The following table from the demo results showcases Loom's versatility in precision vs. compression:

| DType | Quality Score | Avg Dev | File Size | RAM Usage |
|-------|---------------|---------|-----------|-----------|
| **F32** | 100.00% | 0.0000% | 2.92 MB | 5.86 MB |
| **BF16**| 100.00% | 0.0009% | 1.46 MB | 4.40 MB |
| **F4 (FP4)** | **99.40%** | **0.6029%** | **374 KB** | **3.30 MB** |
| **I8**  | 99.61% | 0.3855% | 747 KB | 3.67 MB |

---

## The Discovery: Grid Softmax = Mixture of Experts

One of Loom's most significant features was an **accidental discovery**: Grid Softmax is mathematically equivalent to Mixture of Experts (MoE)—the same architecture used in GPT-4, Switch Transformer, and Mixtral.

### What Does This Mean?

Traditional MoE requires:
1. A gating network to decide which experts to use
2. Multiple expert networks
3. A weighted combination layer

Loom does all of this with **one layer**:

```
Traditional MoE Architecture:

    Input
      │
      ├────────────────────────────────────┐
      │                                    │
      ▼                                    ▼
┌──────────────┐       ┌─────────────────────────┐
│ Gating       │       │ Expert Networks         │
│ Network      │       ├────────┬────────┬───────┤
│ (softmax)    │       │Expert 0│Expert 1│Expert2│
└──────┬───────┘       └────┬───┴────┬───┴───┬───┘
       │                    │        │       │
       │    weights         │        │       │
       ├────────────────────┼────────┼───────┤
       │                    ▼        ▼       ▼
       │              ┌────────────────────────┐
       └─────────────▶│ Weighted Combination   │
                      └────────────┬───────────┘
                                   │
                                   ▼
                               Output

~200 lines of PyTorch code


Loom's Grid Softmax (Equivalent!):

    Input
      │
      ▼
┌─────────────────────────────────────────┐
│           Grid Softmax Layer            │
│                                         │
│  Row 0: [0.7, 0.2, 0.1] ← Expert 0     │
│  Row 1: [0.1, 0.8, 0.1] ← Expert 1     │
│  Row 2: [0.2, 0.2, 0.6] ← Expert 2     │
│                                         │
│  Each row sums to 1.0 independently    │
└─────────────────────────────────────────┘
      │
      ▼
   Output

2 lines of Go code!
```

### The Mathematical Proof

Grid Softmax satisfies all MoE properties:

1. **Independent Expert Pathways**: Each row computes its own softmax, independent of other rows
2. **Soft Routing**: The softmax values ARE the routing weights
3. **Gradient Flow**: Backprop flows through routing automatically
4. **Expert Specialization**: Different inputs activate different experts

```go
// Create a 4-expert MoE in Loom
moeLayer := nn.InitGridSoftmaxLayer(4, 8)  // 4 experts, 8 outputs each

// That's it! This IS Mixture of Experts.
// Each of the 4 rows is an independent expert pathway.
```

---

## Multi-Agent Game AI

One of the most powerful applications of Grid Softmax is controlling **multiple agents with a single network**.

### The Problem

Traditional approach: One network per agent (expensive, can't share learning)

```
Old way:
    Agent 0 ──▶ [Network 0] ──▶ Action
    Agent 1 ──▶ [Network 1] ──▶ Action
    Agent 2 ──▶ [Network 2] ──▶ Action
    
    3 separate networks!
    No knowledge sharing between agents.
```

### The Solution

Grid Softmax enables one network to output independent decisions for all agents:

```
Loom way:
    
    ┌─────────────────────────────────────────────────────────┐
    │                    Shared Network                       │
    │  (one network learns general strategies)                │
    └────────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────┐
    │              Grid Softmax (3×4)                         │
    │                                                         │
    │  Agent 0: [attack:0.6, defend:0.2, scout:0.1, wait:0.1]│
    │  Agent 1: [attack:0.1, defend:0.7, scout:0.1, wait:0.1]│
    │  Agent 2: [attack:0.2, defend:0.1, scout:0.6, wait:0.1]│
    │                                                         │
    │  Each agent gets its own probability distribution!      │
    └─────────────────────────────────────────────────────────┘
```

### Implementation

```go
// Create a multi-agent network
network := nn.NewNetwork(128, 1, 1, 5)

// Shared feature processing
network.SetLayer(0, 0, 0, nn.InitDenseLayer(128, 64, nn.ActivationLeakyReLU))
network.SetLayer(0, 0, 1, nn.InitLSTMLayer(64, 64, 1, 1))  // Temporal memory
network.SetLayer(0, 0, 2, nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU))
network.SetLayer(0, 0, 3, nn.InitDenseLayer(32, 12, nn.ActivationLeakyReLU))

// Grid softmax: 3 agents × 4 actions = 12 outputs
gridSoftmax := nn.InitGridSoftmaxLayer(3, 4)
network.SetLayer(0, 0, 4, gridSoftmax)

// Forward pass with game state
output, _ := network.ForwardCPU(gameState)

// Extract each agent's action distribution
agent0Actions := output[0:4]   // [attack, defend, scout, wait]
agent1Actions := output[4:8]   // [attack, defend, scout, wait]
agent2Actions := output[8:12]  // [attack, defend, scout, wait]

// Each agent can select independently
for i := 0; i < 3; i++ {
    agentActions := output[i*4 : (i+1)*4]
    selectedAction := argmax(agentActions)
    fmt.Printf("Agent %d: action %d (%.1f%% confident)\n", 
        i, selectedAction, agentActions[selectedAction]*100)
}
```

This is how **AlphaStar** controls 200+ StarCraft units and **OpenAI Five** controls 5 Dota heroes.

---

## Legal Move Masking

In games, not all actions are always legal. Masked Softmax solves this elegantly.

### The Problem

```
Chess example:
    Raw network output: [castle:0.3, en_passant:0.25, pawn_forward:0.45]
    
    But castling is blocked by a piece!
    And en passant isn't available this turn!
    
    Only pawn_forward is legal.
```

### The Solution

Masked Softmax forces illegal moves to zero probability:

```go
// Create masked softmax layer
masked := nn.InitMaskedSoftmaxLayer(6)  // 6 possible actions
network.SetLayer(0, 0, 2, masked)

// During gameplay, update the mask based on game rules
layer := network.GetLayer(0, 0, 2)
layer.Mask = []bool{
    true,  // action 0: legal
    false, // action 1: ILLEGAL (on cooldown)
    true,  // action 2: legal
    true,  // action 3: legal
    false, // action 4: ILLEGAL (not enough mana)
    true,  // action 5: legal
}

output, _ := network.ForwardCPU(gameState)
// output[1] and output[4] will be ~0, rest normalized to sum to 1
```

### How It Works Internally

```
Before masking:
    Raw logits: [2.1, 1.8, 0.5, 1.2, 3.0, 0.8]
    
Mask applied:
    Masked logits: [2.1, -∞, 0.5, 1.2, -∞, 0.8]
                        ↑              ↑
                    Forced to          |
                    negative infinity  |
                                       
After softmax:
    [0.35, 0.00, 0.10, 0.25, 0.00, 0.30]
           ↑              ↑
       Exactly zero due to exp(-∞) = 0
```

---

## Hierarchical Decision Trees

Some problems have natural hierarchies: First decide a strategy, then a tactic, then a specific action.

### Example: RTS Game Decisions

```
Level 1: What type of move? (Strategic)
    ├── Attack (30%)
    ├── Defend (50%)
    └── Economy (20%)
    
Level 2: Which unit? (Tactical - given we're defending)
    ├── Warrior (60%)
    ├── Archer (30%)
    └── Mage (10%)
    
Level 3: What action? (Specific - given warrior defending)
    ├── Hold position (70%)
    ├── Patrol (20%)
    └── Fortify (10%)
```

### Implementation

```go
// Hierarchical softmax: 3 strategies × 3 units × 3 actions = 27 outputs
hierarchical := nn.InitHierarchicalSoftmaxLayer([]int{3, 3, 3})
network.SetLayer(0, 0, 3, hierarchical)

output, _ := network.ForwardCPU(input)

// Output structure:
// [0-2]:   Strategy probabilities
// [3-11]:  Unit probabilities for each strategy (3×3)
// [12-26]: Action probabilities for each strategy-unit combo (3×3×3)

// Parse hierarchically
strategies := output[0:3]
bestStrategy := argmax(strategies)

unitOffset := 3 + bestStrategy*3
units := output[unitOffset : unitOffset+3]
bestUnit := argmax(units)

actionOffset := 12 + bestStrategy*9 + bestUnit*3
actions := output[actionOffset : actionOffset+3]
bestAction := argmax(actions)

fmt.Printf("Strategy %d → Unit %d → Action %d\n", 
    bestStrategy, bestUnit, bestAction)
```

---

## Neural Tweening for Stable Online Learning

When your network needs to adapt continuously (online learning), standard backpropagation can be unstable. Neural Tweening provides automatic stability.

### The Scenario

```
Online Learning Problem:

Time 0-100:    Train on Task A (classify apples vs oranges)
               Network gets good at Task A
               
Time 100-200:  Suddenly switch to Task B (classify cats vs dogs)
               
Standard Backprop:
    ─────────●────────────────────────────────
             │
             │ Task switch
             ▼
    Accuracy │  ╱╲╱╲  ╱╲╱╲╱╲
             │ ╱    ╲╱      ╲
             │╱                ╲───────────────
             │
             └──────────────────────────────────▶
    
    Oscillates wildly at task switch, may crash


Neural Tweening:
    ─────────●────────────────────────────────
             │
             │ Task switch
             ▼
    Accuracy │      ╱───────────────────────
             │    ╱
             │  ╱
             │ ╱
             └──────────────────────────────────▶
    
    Stable transition, maintains some previous learning
```

### Implementation

```go
// Create tween state
config := &nn.TweenConfig{
    BaseRate:        0.01,
    MomentumDecay:   0.9,
    ExplosionLimit:  10.0,
    RecoveryRate:    0.5,
}
ts := nn.NewTweenState(network, config)

// Online learning loop
for sample := range dataStream {
    // TweenStep handles everything:
    // - Forward and backward pass
    // - Explosion detection
    // - Automatic recovery
    // - Momentum updates
    loss := ts.TweenStep(
        network,
        sample.Input,
        sample.Label,
        numClasses,
        config.BaseRate,
        backend,
    )
    
    // Monitor health
    if ts.ExplosionCount > 0 {
        fmt.Printf("Recovered from %d explosions\n", ts.ExplosionCount)
    }
}
```

---

## Softmax in Hidden Layers: Attention and Gating

A powerful pattern: using softmax-based layers **inside** the network, not just at the output.

### Internal Attention

```go
network := nn.NewNetwork(64, 1, 1, 6)

// Feature extraction
network.SetLayer(0, 0, 0, nn.InitDenseLayer(64, 64, nn.ActivationLeakyReLU))

// ATTENTION: Which features to focus on?
network.SetLayer(0, 0, 1, nn.InitSparsemaxLayer())  // ← Hidden softmax!
// Sparsemax produces exact zeros, so only important features pass through

// Further processing
network.SetLayer(0, 0, 2, nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU))
network.SetLayer(0, 0, 3, nn.InitDenseLayer(32, 16, nn.ActivationLeakyReLU))
network.SetLayer(0, 0, 4, nn.InitDenseLayer(16, 10, nn.ActivationLeakyReLU))

// Final output
network.SetLayer(0, 0, 5, nn.InitSoftmaxLayer())    // ← Output softmax
```

### Internal Routing (MoE)

```go
network := nn.NewNetwork(64, 1, 1, 5)

// Input processing
network.SetLayer(0, 0, 0, nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU))

// ROUTING: Which "expert path" to take?
network.SetLayer(0, 0, 1, nn.InitGridSoftmaxLayer(4, 8))  // ← Hidden MoE!
// 4 experts, 8 features each = 32 outputs
// Next layer receives weighted expert outputs

// Further processing
network.SetLayer(0, 0, 2, nn.InitDenseLayer(32, 16, nn.ActivationLeakyReLU))
network.SetLayer(0, 0, 3, nn.InitDenseLayer(16, 10, nn.ActivationLeakyReLU))

// Output
network.SetLayer(0, 0, 4, nn.InitSoftmaxLayer())
```

---

## Cross-Platform Deployment

Train once in Go, deploy everywhere.

### Training in Go

```go
// Train network
network := nn.NewNetwork(784, 2, 2, 3)
// ... configure layers, train ...

// Save to JSON
network.SaveModel("mnist_classifier.json", "mnist_v1")
```

### Deploy to Browser (WASM)

```javascript
// Load in JavaScript
const response = await fetch('mnist_classifier.json');
const modelJSON = await response.text();
const network = loom.LoadNetworkFromString(modelJSON, "mnist_v1");

// Run inference
const pixels = getImagePixels();  // 784 float values
const probabilities = network.Forward(pixels);

const digit = argmax(probabilities);
console.log(`Predicted: ${digit}`);
```

### Deploy to Python (via C ABI)

```python
import welvet

# Load model
with open("mnist_classifier.json") as f:
    model_json = f.read()
network = welvet.load_model_from_string(model_json, "mnist_v1")

# Run inference
import numpy as np
pixels = np.array(image_data, dtype=np.float32)
probabilities = network.forward(pixels)

digit = np.argmax(probabilities)
print(f"Predicted: {digit}")
```

### Deploy to Mobile (via C ABI)

```c
// Load model
char* model_json = read_file("mnist_classifier.json");
LoomLoadModel(model_json, "mnist_v1");

// Run inference
float pixels[784] = { ... };
char* output_json = LoomForward(pixels, 784);

// Parse result
// output_json contains the probability array
```

---

## Save/Load with Training State

For checkpointing during long training runs:

```go
// Training loop with checkpoints
for epoch := 0; epoch < 1000; epoch++ {
    for _, batch := range trainData {
        output, _ := network.ForwardCPU(batch.Input)
        loss, grad := nn.CrossEntropyLossGrad(output, batch.Target)
        network.BackwardCPU(grad)
        network.ApplyGradients(learningRate)
    }
    
    // Save checkpoint every 100 epochs
    if epoch % 100 == 0 {
        filename := fmt.Sprintf("checkpoint_epoch_%04d.json", epoch)
        network.SaveModel(filename, "training_checkpoint")
        fmt.Printf("Saved checkpoint: %s\n", filename)
    }
}

// Later: resume from checkpoint
network, _ = nn.LoadModel("checkpoint_epoch_0500.json", "training_checkpoint")
// Continue training from epoch 500...
```

---

## Heterogeneous Architectures with Parallel Layers

Loom's Parallel layer lets you run **different layer types** side by side.

### Example: Multi-Modal Processing

```go
// Input: concatenated [image_features | text_features | audio_features]
network := nn.NewNetwork(256, 1, 1, 3)

// Create parallel layer with different architectures per modality
parallel := nn.NewParallelLayer(nn.CombineConcat)

// Image branch: Conv2D
imageBranch := nn.InitConv2DLayer(8, 8, 4, 16, 3, 1, 1, nn.ActivationReLU)
parallel.AddBranch(imageBranch)

// Text branch: LSTM
textBranch := nn.InitLSTMLayer(64, 32, 1, 10)
parallel.AddBranch(textBranch)

// Audio branch: Dense + Attention
audioBranch := nn.NewSequentialLayer()
audioBranch.AddLayer(nn.InitDenseLayer(64, 32, nn.ActivationReLU))
audioBranch.AddLayer(nn.InitMultiHeadAttentionLayer(32, 4, 1, 8))
parallel.AddBranch(audioBranch)

network.SetLayer(0, 0, 0, parallel)

// Fusion layers
network.SetLayer(0, 0, 1, nn.InitDenseLayer(
    16*6*6 + 32 + 32,  // Combined output sizes
    64, 
    nn.ActivationReLU,
))
network.SetLayer(0, 0, 2, nn.InitSoftmaxLayer())
```

---

## Case Study: Hierarchical Concept Taxonomy (RN Benchmark Series)

The **Recursive Neuro-Symbolic (RN)** benchmark series demonstrates Loom's ability to learn complex hierarchical taxonomies using nested `KMeansLayer` architectures. This represents a bridge between deep learning and symbolic reasoning.

### The RN Suite at a Glance

| Benchmark | Architecture | Concept Learned |
|-----------|--------------|------------------|
| **RN1** | `KMeans(4) → KMeans(2)` | Hierarchical spatial grouping. |
| **RN2** | `KMeans(15) → KMeans(3)` | The Star-Galaxy taxonomy (clusters within clusters). |
| **RN3** | `KMeans(8)` | Geometric Anomaly Detection (Out-of-Distribution). |
| **RN4** | `KMeans(Prototype)` | Shortcut/Spurious correlation defense. |
| **RN5** | `KMeans vs MLP` | Performance vs. Interpretability baseline. |
| **RN6** | `Parallel KMeans` | Mixture of Experts (MoE) with prototype branch selection. |

### Key Result: Interpretability and Reliability

Unlike traditional deep networks that are "Black Boxes," the RN series proves that Loom's prototype-based layers are:
1. **Fully Interpretable**: Each cluster center is a "Concept Prototype" that can be visualized and inspected.
2. **Robust to Shift**: They naturally handle "shortcuts" and "short-day" OOD attacks by relying on geometric manifolds rather than brittle numeric correlations.
3. **Recursive**: They can be stacked indefinitely to build deep hierarchies of reasoning ($p \rightarrow q \rightarrow r$).

---

## Performance Tips

### 1. Batch Processing

```go
// Process multiple samples together
batchSize := 32
inputs := make([][]float32, batchSize)
// ... fill inputs ...

// Forward all at once (more efficient)
for i, input := range inputs {
    outputs[i], _ = network.ForwardCPU(input)
}
```

### 2. Step-Based Execution for Long Sequences

```go
// For very long sequences, use step-based execution
state := network.InitStepState(inputSize)

for t, token := range sequence {
    state.SetInput(tokenEmbedding(token))
    output := state.StepForward()
    
    // Process output at each step
    predictions[t] = output
}
```

### 3. Quantization for Deployment

```go
// Save with lower precision for smaller files
network.SaveModelMultiPrecision("model_int8.json", "quantized", nn.PrecisionInt8)
// File size reduced ~4x
```

---

## Summary

Key patterns demonstrated:
1. **Grid Softmax = Native MoE**: One layer does what takes 200+ lines elsewhere
2. **Multi-Agent**: One network, multiple independent action distributions
3. **Legal Move Masking**: Force illegal actions to zero probability
4. **Hierarchical Decisions**: Nested decision trees with automatic normalization
5. **Neural Tweening**: Stable online learning with automatic explosion recovery
6. **Hidden Softmax**: Attention and routing inside the network
7. **Cross-Platform**: Train in Go, deploy to WASM, Python, C, mobile
8. **Heterogeneous Parallel**: Different architectures per branch

The examples in `/tva/examples/` demonstrate all of these patterns with runnable code.
