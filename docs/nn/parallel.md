# Understanding Parallel Layers

This guide explains Loom's Parallel layer—how to run multiple sub-networks simultaneously, combine their outputs in different ways, and build complex architectures like Mixture of Experts.

---

## What is a Parallel Layer?

A Parallel layer splits input, processes it through multiple "branches", then combines the results:

```
                        Input
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
        ┌───────┐     ┌───────┐     ┌───────┐
        │Branch │     │Branch │     │Branch │
        │   0   │     │   1   │     │   2   │
        │(Dense)│     │(LSTM) │     │ (MHA) │
        └───┬───┘     └───┬───┘     └───┬───┘
            │             │             │
            └─────────────┼─────────────┘
                          │
                      Combine
                    (concat/add/avg/filter/grid_scatter)
                          │
                          ▼
                       Output
```

Each branch can be a **different layer type**—this is what makes Parallel layers so powerful.

---

## Combine Modes

The key decision: How do you combine branch outputs?

### Concat (Default)

Concatenate all outputs into one large vector:

```
Branch 0 output: [a, b, c]      (3 values)
Branch 1 output: [d, e]         (2 values)
Branch 2 output: [f, g, h, i]   (4 values)

Combined (concat): [a, b, c, d, e, f, g, h, i]  (9 values)
```

Use when:
- Branches produce different feature types
- You want next layer to see all information
- Output sizes differ between branches

### Add

Element-wise addition (requires same-sized outputs):

```
Branch 0 output: [1.0, 2.0, 3.0]
Branch 1 output: [0.5, 0.5, 0.5]
Branch 2 output: [0.2, 0.3, 0.2]

Combined (add): [1.7, 2.8, 3.7]
```

Use when:
- Branches are processing the same features differently
- You want to aggregate responses
- Building residual-like connections

### Average

Element-wise average (requires same-sized outputs):

```
Branch 0 output: [1.0, 2.0, 3.0]
Branch 1 output: [0.5, 0.5, 0.5]
Branch 2 output: [0.2, 0.3, 0.2]

Combined (avg): [0.57, 0.93, 1.23]  (mean of each position)
```

Use when:
- Building ensemble predictions
- You want balanced contribution from each branch

### Grid Scatter

Place outputs at specific 2D/3D grid positions:

```
Branch 0 → position (0, 0)
Branch 1 → position (0, 1)
Branch 2 → position (1, 0)
Branch 3 → position (1, 1)

Grid output:
    ┌─────────────┬─────────────┐
    │  Branch 0   │  Branch 1   │  Row 0
    │  output     │  output     │
    ├─────────────┼─────────────┤
    │  Branch 2   │  Branch 3   │  Row 1
    │  output     │  output     │
    └─────────────┴─────────────┘
       Col 0         Col 1
```

Use when:
- Building spatially-aware architectures
- Multi-agent systems with spatial positioning
- Image processing with region-specific branches

### Filter (Softmax-Gated) — Dynamic Logic Gates

The most powerful combine mode: a **learnable gate network** decides how much to use each branch.

```
                              Input
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ Gate Network  │       │   Expert 0    │       │   Expert 1    │
│ (Dense→Softmax)│       │   (Dense)     │       │   (LSTM)      │
└───────┬───────┘       └───────┬───────┘       └───────┬───────┘
        │                       │                       │
        ▼                       ▼                       ▼
   [0.7, 0.3]           [e0_out...]             [e1_out...]
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                          Weighted Sum:
                    0.7 × Expert0 + 0.3 × Expert1
                                │
                                ▼
                            Output
```

**This is a learnable conditional computation system**—the gate learns WHEN to use each expert.

#### How Filter Mode Works

```
Gate network predicts: [0.6, 0.3, 0.1]  (sum to 1.0 via softmax)

Branch 0 output: [1.0, 2.0, 3.0]  × 0.6 = [0.60, 1.20, 1.80]
Branch 1 output: [0.5, 0.5, 0.5]  × 0.3 = [0.15, 0.15, 0.15]
Branch 2 output: [0.2, 0.3, 0.2]  × 0.1 = [0.02, 0.03, 0.02]

Combined (filter): [0.77, 1.38, 1.97]
```

#### The Power: Dynamic Learned Routing

Unlike static combine modes, the gate network **learns from data** which expert to use for which inputs:

```
Training data:
    Pattern A: input[0] > 0.5 → Expert 0 is better
    Pattern B: input[0] ≤ 0.5 → Expert 1 is better

After training:
    Input with high input[0] → gate outputs [0.95, 0.05]
    Input with low input[0]  → gate outputs [0.10, 0.90]
    
The gate has learned to route each input to the right expert!
```

#### Building a Filtered Parallel Layer

```go
inputSize := 16
expertSize := 8

// Create two expert branches (can be any layer type)
expert1 := nn.InitDenseLayer(inputSize, expertSize, nn.ActivationLeakyReLU)
expert2 := nn.InitDenseLayer(inputSize, expertSize, nn.ActivationLeakyReLU)

// Create gate layer: Input → 2 outputs (one per expert)
gateLayer := nn.InitDenseLayer(inputSize, 2, nn.ActivationScaledReLU)

// Build the filtered parallel layer
filterLayer := nn.LayerConfig{
    Type:              nn.LayerParallel,
    ParallelBranches:  []nn.LayerConfig{expert1, expert2},
    CombineMode:       "filter",
    FilterGateConfig:  &gateLayer,
    FilterSoftmax:     nn.SoftmaxStandard,  // How to normalize gate outputs
    FilterTemperature: 1.0,                 // Controls routing sharpness
}

// Add to network
network.SetLayer(0, 0, 1, filterLayer)
```

#### Gate Configuration Options

**FilterSoftmax** — Controls how gate outputs are normalized:

| Type | Effect | Use Case |
|------|--------|----------|
| `SoftmaxStandard` | Smooth routing, all experts get some weight | Ensemble learning |
| `SoftmaxEntmax` | Sparse routing, some experts get exact zero | Efficiency |
| `SoftmaxSparsemax` | Very sparse, picks 1-2 experts | Hard routing |
| `SoftmaxTemperature` | Adjustable sharpness | Curriculum learning |

**FilterTemperature** — Controls routing "sharpness":

```
Temperature = 1.0 (default):
    Gate logits [2.0, 1.0] → [0.73, 0.27]  (soft mix)

Temperature = 0.5 (sharper):
    Gate logits [2.0, 1.0] → [0.88, 0.12]  (mostly expert 0)

Temperature = 0.1 (nearly hard):
    Gate logits [2.0, 1.0] → [0.99, 0.01]  (almost hard selection)
```

#### Advanced: Expert Pre-Training + Gate Training

A powerful pattern: **train experts separately, then train just the gate**.

```go
// ============================================================
// STEP 1: Pre-train Expert 1 on "high signal" patterns
// ============================================================
expert1 := nn.InitDenseLayer(8, 8, nn.ActivationSigmoid)
e1Net := nn.NewNetwork(8, 1, 1, 1)
e1Net.SetLayer(0, 0, 0, expert1)

// Train: high first element → high output
trainData1 := make([]nn.TrainingBatch, 2000)
for i := range trainData1 {
    input := randomInput(8)
    if rand.Float32() > 0.5 {
        input[0] = 0.7 + rand.Float32()*0.3  // High
        target = 1.0
    } else {
        input[0] = rand.Float32()*0.3  // Low
        target = 0.0
    }
    trainData1[i] = nn.TrainingBatch{Input: input, Target: []float32{target}}
}
e1Net.Train(trainData1, &nn.TrainingConfig{Epochs: 10, LearningRate: 0.1})
expert1 = *e1Net.GetLayer(0, 0, 0)  // Get trained weights

// ============================================================
// STEP 2: Pre-train Expert 2 on "low signal" patterns
// ============================================================
expert2 := nn.InitDenseLayer(8, 8, nn.ActivationSigmoid)
// (similar training, but responds to low input[0])

// ============================================================
// STEP 3: Combine with filter layer, train ONLY the gate
// ============================================================
gateLayer := nn.InitDenseLayer(8, 2, nn.ActivationScaledReLU)

filterLayer := nn.LayerConfig{
    Type:              nn.LayerParallel,
    ParallelBranches:  []nn.LayerConfig{expert1, expert2},  // Pre-trained!
    CombineMode:       "filter",
    FilterGateConfig:  &gateLayer,  // This will be trained
    FilterSoftmax:     nn.SoftmaxStandard,
    FilterTemperature: 0.5,  // Sharper routing
}

net := nn.NewNetwork(8, 1, 1, 2)
net.SetLayer(0, 0, 0, filterLayer)
net.SetLayer(0, 0, 1, nn.InitDenseLayer(8, 1, nn.ActivationSigmoid))

// Train with tweening (gate learns to route to correct expert)
ts := nn.NewTweenState(net, nil)
for epoch := 0; epoch < 2000; epoch++ {
    input := randomInput(8)
    if epoch%2 == 0 {
        input[0] = 0.7 + rand.Float32()*0.3  // High → should route to expert1
    } else {
        input[0] = rand.Float32() * 0.3       // Low → should route to expert2
    }
    ts.TweenStep(net, input, 0, 1, 0.01)
}
```

After training, the gate will have learned:
- High `input[0]` → route to Expert 1
- Low `input[0]` → route to Expert 2

**This is essentially a learned IF/ELSE statement!**

#### Freezing Layers: Train Only the Gate

Loom supports **freezing layers** so their weights don't update during training. This is essential for the filter pattern: freeze pre-trained experts, train only the gate.

```go
// LayerConfig has a Frozen field
type LayerConfig struct {
    // ... other fields ...
    Frozen bool  // If true, weights will NOT be updated during training
}
```

##### Using the Frozen Field

```go
// Freeze a single layer
expertLayer.Frozen = true

// Freeze recursively (for Sequential/Parallel layers)
func freezeLayer(cfg *nn.LayerConfig) {
    cfg.Frozen = true
    // Recurse into nested branches
    for i := range cfg.ParallelBranches {
        freezeLayer(&cfg.ParallelBranches[i])
    }
}
```

##### Complete Pattern: Frozen Experts + Trainable Gate

```go
// ============================================================
// STEP 1: Create and pre-train Expert 1
// ============================================================
expert1 := nn.InitSequentialLayer(
    nn.InitDenseLayer(8, 8, nn.ActivationLeakyReLU),
    nn.InitDenseLayer(8, 1, nn.ActivationSigmoid),
)

// Train expert1 to respond to HIGH input[0]
e1Net := nn.NewNetwork(8, 1, 1, 1)
e1Net.SetLayer(0, 0, 0, expert1)
e1Net.Train(highPatternData, &nn.TrainingConfig{
    Epochs: 5, LearningRate: 0.05,
})
expert1 = *e1Net.GetLayer(0, 0, 0)

// ============================================================
// STEP 2: Create and pre-train Expert 2
// ============================================================
expert2 := nn.InitSequentialLayer(
    nn.InitDenseLayer(8, 8, nn.ActivationLeakyReLU),
    nn.InitDenseLayer(8, 1, nn.ActivationSigmoid),
)

// Train expert2 to respond to LOW input[0]
e2Net := nn.NewNetwork(8, 1, 1, 1)
e2Net.SetLayer(0, 0, 0, expert2)
e2Net.Train(lowPatternData, &nn.TrainingConfig{
    Epochs: 5, LearningRate: 0.05,
})
expert2 = *e2Net.GetLayer(0, 0, 0)

// ============================================================
// STEP 3: FREEZE both experts
// ============================================================
freezeLayer(&expert1)  // expert1.Frozen = true (recursive)
freezeLayer(&expert2)  // expert2.Frozen = true (recursive)

// ============================================================
// STEP 4: Create filter layer with frozen experts + trainable gate
// ============================================================
gateLayer := nn.InitDenseLayer(8, 2, nn.ActivationScaledReLU)
// Note: gateLayer.Frozen is FALSE (default) - it WILL be trained

filterLayer := nn.LayerConfig{
    Type:              nn.LayerParallel,
    ParallelBranches:  []nn.LayerConfig{expert1, expert2},  // FROZEN
    CombineMode:       "filter",
    FilterGateConfig:  &gateLayer,  // TRAINABLE
    FilterSoftmax:     nn.SoftmaxStandard,
    FilterTemperature: 0.5,
}

net := nn.NewNetwork(8, 1, 1, 1)
net.SetLayer(0, 0, 0, filterLayer)

// ============================================================
// STEP 5: Train - only gate weights will update!
// ============================================================
ts := nn.NewTweenState(net, nil)
ts.Config.UseChainRule = true

for epoch := 0; epoch < 1000; epoch++ {
    input := randomInput(8)
    if epoch%2 == 0 {
        input[0] = 0.9  // Should route to expert1
    } else {
        input[0] = 0.1  // Should route to expert2
    }
    
    ts.TweenStep(net, input, 0, 1, 0.05)
    // Gate learns to route correctly
    // Expert weights stay FROZEN - no updates
}
```

##### How Freezing Works Internally

When a layer has `Frozen = true`:

```
During backward pass:
    
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Input     │────▶│   Expert    │────▶│   Output    │
    └─────────────┘     │  (FROZEN)   │     └─────────────┘
                        └──────┬──────┘
                               │
                               ▼
                     Gradients still flow THROUGH
                     (for upstream layers)
                     
                     But weights are NOT updated:
                     kernel_grad = 0
                     bias_grad = 0
```

The gradient **passes through** frozen layers (so upstream trainable layers can learn), but the frozen layer's own weights are **never modified**.

##### Visualization of Filter Training with Frozen Experts

```
Forward Pass:
                         Input
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
       ▼                   ▼                   ▼
 ┌───────────┐      ┌───────────┐      ┌───────────┐
 │   Gate    │      │  Expert1  │      │  Expert2  │
 │(trainable)│      │ (FROZEN)  │      │ (FROZEN)  │
 └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
       │                  │                  │
       ▼                  ▼                  ▼
   [0.8, 0.2]           [0.9]              [0.1]
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
                    0.8×0.9 + 0.2×0.1 = 0.74
                          │
                          ▼
                       Output


Backward Pass:
                      ∂Loss/∂Output
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
       ▼                   ▼                   ▼
 ┌───────────┐      ┌───────────┐      ┌───────────┐
 │   Gate    │      │  Expert1  │      │  Expert2  │
 │ UPDATED!  │      │ NOT UPD   │      │ NOT UPD   │
 │ dW = ...  │      │ dW = 0    │      │ dW = 0    │
 └───────────┘      └───────────┘      └───────────┘

Only the gate learns! Experts stay frozen.
```

##### Use Cases for Frozen Layers

| Scenario | What to Freeze | What to Train |
|----------|----------------|---------------|
| Pre-trained experts + gate | Expert branches | Gate layer only |
| Transfer learning | Base model layers | New head layers |
| Feature extraction | Encoder | Decoder |
| Fine-tuning on new task | Lower layers | Output layers |
| Debugging | Suspected broken layer | Rest of network |

#### The Dynamic Logic Gate Concept

Filter mode enables **networks that learn conditional logic**:

```
Traditional programming:
    if (input[0] > 0.5) {
        use_expert_1()
    } else {
        use_expert_2()
    }

Filter mode equivalent:
    Gate learns the decision boundary automatically!
    
    Different experts can specialize in:
    - Different input ranges
    - Different feature patterns
    - Different task types
    - Different modalities
```

#### Use Cases

| Scenario | Experts | What Gate Learns |
|----------|---------|------------------|
| Multi-task learning | Task A expert, Task B expert | Which task this input belongs to |
| Feature specialization | Low-frequency expert, High-frequency expert | Signal characteristics |
| Temporal patterns | Recent-memory expert (Dense), Long-memory expert (LSTM) | Time horizon to focus on |
| Difficulty routing | Simple expert (small), Complex expert (deep) | Input complexity |
| Modality fusion | Image expert (Conv), Text expert (LSTM) | Which modality is more informative |

Use when:
- Building Mixture of Experts architectures
- You want the network to learn conditional computation
- Different inputs need fundamentally different processing
- You want interpretable routing decisions

#### Case Study: Parallel KMeans Experts (The RN6 Pattern)

A powerful application of Parallel layers is using multiple KMeans "experts" in parallel. This is the core of the **RN6 (Recursive Neuro-Symbolic 6)** benchmark.

```
                  Input
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    ┌───────────┐       ┌───────────┐
    │  Expert A │       │  Expert B │
    │ (KMeans)  │       │ (KMeans)  │
    └─────┬─────┘       └─────▼─────┘
          │                   │
          └─────────┬─────────┘
                    │
                 Combine
             (concat/filter)
                    │
                    ▼
               Dense Head
                    │
                    ▼
               Classification
```

**Why do this?**
- **Diverse Perspectives**: Each expert can learn to cluster the data differently (e.g., one focusing on spatial proximity, another on feature-based similarity).
- **Redundancy & Reliability**: If one expert fails to capture a complex boundary, others can compensate.
- **Interpretable MoE**: In `filter` mode, you can see exactly which "cluster" or "concept" expert the model is choosing for any given input.

---

## Stitching Oddly-Shaped Networks

One of Loom's unique capabilities: **combine networks with different output sizes** using stitch layers.

### The Problem

Filter mode (and avg mode) require all branches to output the **same size**. But what if your pre-trained experts have different architectures?

```
Expert A: input → 16 features
Expert B: input → 32 features
Expert C: input → 7 features

Filter mode needs all outputs to be the same size!
```

### The Solution: Stitch Layers

Use `InitStitchLayer()` to project each expert's output to a common size:

```
                    Input
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │Expert A │   │Expert B │   │Expert C │
   │ (→16)   │   │ (→32)   │   │ (→7)    │
   └────┬────┘   └────┬────┘   └────┬────┘
        │             │             │
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │Stitch   │   │Stitch   │   │Stitch   │
   │(16→10)  │   │(32→10)  │   │(7→10)   │
   └────┬────┘   └────┬────┘   └────┬────┘
        │             │             │
        ▼             ▼             ▼
      [10]          [10]          [10]   ← All same size now!
        │             │             │
        └─────────────┼─────────────┘
                      │
                 Filter Combine
                      │
                      ▼
                   [10]
```

### InitStitchLayer

A stitch layer is a linear projection (Dense layer without activation):

```go
// Create a stitch layer: 16 → 10
stitch := nn.InitStitchLayer(16, 10)

// Equivalent to:
stitch := nn.InitDenseLayer(16, 10, nn.ActivationType(-1)) // Linear
```

### Building Stitched Branches

Wrap each expert with its stitch layer using `InitSequentialLayer`:

```go
inputSize := 16
commonOutputSize := 10

// Expert 1: outputs 5 features → stitch to 10
expert1 := nn.InitDenseLayer(inputSize, 5, nn.ActivationLeakyReLU)
stitch1 := nn.InitStitchLayer(5, commonOutputSize)
branch1 := nn.InitSequentialLayer(expert1, stitch1)

// Expert 2: outputs 7 features → stitch to 10
expert2 := nn.InitDenseLayer(inputSize, 7, nn.ActivationSigmoid)
stitch2 := nn.InitStitchLayer(7, commonOutputSize)
branch2 := nn.InitSequentialLayer(expert2, stitch2)

// Now both branches output [10] - ready for filter combine!
gateLayer := nn.InitDenseLayer(inputSize, 2, nn.ActivationScaledReLU)

filterLayer := nn.LayerConfig{
    Type:              nn.LayerParallel,
    ParallelBranches:  []nn.LayerConfig{branch1, branch2},
    CombineMode:       "filter",
    FilterGateConfig:  &gateLayer,
    FilterSoftmax:     nn.SoftmaxStandard,
    FilterTemperature: 1.0,
}
```

### Multi-Expert Stitching Example

Combine 4 experts with wildly different output sizes:

```go
inputSize := 16
commonOutputSize := 8
expertSizes := []int{4, 12, 6, 20}  // Very different!

branches := make([]nn.LayerConfig, len(expertSizes))
for i, size := range expertSizes {
    expert := nn.InitDenseLayer(inputSize, size, nn.ActivationLeakyReLU)
    stitch := nn.InitStitchLayer(size, commonOutputSize)
    branches[i] = nn.InitSequentialLayer(expert, stitch)
}

gateLayer := nn.InitDenseLayer(inputSize, len(branches), nn.ActivationScaledReLU)

filterLayer := nn.LayerConfig{
    Type:              nn.LayerParallel,
    ParallelBranches:  branches,
    CombineMode:       "filter",
    FilterGateConfig:  &gateLayer,
    FilterSoftmax:     nn.SoftmaxEntmax,  // Sparse routing
    FilterTemperature: 0.5,               // Sharp selection
}

// All 4 experts (sizes 4, 12, 6, 20) now work together!
```

### Pre-Training + Stitching + Gate Training

The complete pattern with oddly-shaped pre-trained networks:

```go
// ============================================================
// STEP 1: Pre-train Expert 1 (outputs 3 features)
// ============================================================
expert1Core := nn.InitDenseLayer(8, 3, nn.ActivationSigmoid)
stitch1 := nn.InitStitchLayer(3, commonOutputSize)

net1 := nn.NewNetwork(8, 1, 1, 2)
net1.SetLayer(0, 0, 0, expert1Core)
net1.SetLayer(0, 0, 1, stitch1)

// Train on "high input" detection task
net1.Train(highInputData, &nn.TrainingConfig{
    Epochs: 10, LearningRate: 0.1,
})

// Bundle expert + stitch as one branch
branch1 := nn.InitSequentialLayer(
    *net1.GetLayer(0, 0, 0),  // Pre-trained expert
    *net1.GetLayer(0, 0, 1),  // Pre-trained stitch
)

// ============================================================
// STEP 2: Pre-train Expert 2 (outputs 5 features)
// ============================================================
expert2Core := nn.InitDenseLayer(8, 5, nn.ActivationSigmoid)
stitch2 := nn.InitStitchLayer(5, commonOutputSize)

net2 := nn.NewNetwork(8, 1, 1, 2)
net2.SetLayer(0, 0, 0, expert2Core)
net2.SetLayer(0, 0, 1, stitch2)

// Train on "low input" detection task
net2.Train(lowInputData, &nn.TrainingConfig{
    Epochs: 10, LearningRate: 0.1,
})

branch2 := nn.InitSequentialLayer(
    *net2.GetLayer(0, 0, 0),
    *net2.GetLayer(0, 0, 1),
)

// ============================================================
// STEP 3: Freeze experts, combine with filter, train gate
// ============================================================
freezeLayer(&branch1)
freezeLayer(&branch2)

gateLayer := nn.InitDenseLayer(8, 2, nn.ActivationScaledReLU)

filterLayer := nn.LayerConfig{
    Type:              nn.LayerParallel,
    ParallelBranches:  []nn.LayerConfig{branch1, branch2},
    CombineMode:       "filter",
    FilterGateConfig:  &gateLayer,  // Trainable!
    FilterSoftmax:     nn.SoftmaxStandard,
    FilterTemperature: 0.5,
}

net := nn.NewNetwork(8, 1, 1, 2)
net.SetLayer(0, 0, 0, filterLayer)
net.SetLayer(0, 0, 1, nn.InitDenseLayer(commonOutputSize, 1, nn.ActivationSigmoid))

// Train gate only
ts := nn.NewTweenState(net, nil)
ts.Config.UseChainRule = true

for epoch := 0; epoch < 1000; epoch++ {
    input := randomInput(8)
    input[0] = (epoch%2 == 0) ? 0.9 : 0.1  // Alternate high/low
    ts.TweenStep(net, input, 0, 1, 0.05)
}
```

### When to Use Stitching

| Scenario | Solution |
|----------|----------|
| Experts have different output sizes | Stitch to common size |
| Loading pre-trained models with different architectures | Stitch before combining |
| Ensemble of heterogeneous models | Stitch outputs, then average or filter |
| Transfer learning from models with different feature dimensions | Stitch to target dimension |

---

## Creating Parallel Layers

### Basic Parallel (Concat)

```go
// Create a parallel layer with 3 branches
parallel := nn.InitParallelLayer()
parallel.CombineMode = "concat"

// Add branches of different types
parallel.ParallelBranches = []nn.LayerConfig{
    nn.InitDenseLayer(64, 32, nn.ActivationReLU),      // Dense branch
    nn.InitLSTMLayer(64, 32, 1, 10),                   // LSTM branch  
    nn.InitMultiHeadAttentionLayer(64, 4, 1, 16),      // Attention branch
}

// Add to network
network.SetLayer(0, 0, 1, parallel)

// Output size: 32 + 32 + 64 = 128 (concatenated)
```

### Parallel with Add Mode

```go
parallel := nn.InitParallelLayer()
parallel.CombineMode = "add"

// All branches must output same size!
parallel.ParallelBranches = []nn.LayerConfig{
    nn.InitDenseLayer(64, 32, nn.ActivationReLU),
    nn.InitDenseLayer(64, 32, nn.ActivationTanh),
    nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU),
}

// Output size: 32 (element-wise sum)
```

### Grid Scatter

```go
parallel := nn.InitParallelLayer()
parallel.CombineMode = "grid_scatter"
parallel.GridOutputRows = 2
parallel.GridOutputCols = 2
parallel.GridOutputLayers = 1

// Position each branch in the grid
parallel.GridPositions = []nn.GridPosition{
    {TargetRow: 0, TargetCol: 0, TargetLayer: 0},  // Branch 0 → top-left
    {TargetRow: 0, TargetCol: 1, TargetLayer: 0},  // Branch 1 → top-right
    {TargetRow: 1, TargetCol: 0, TargetLayer: 0},  // Branch 2 → bottom-left
    {TargetRow: 1, TargetCol: 1, TargetLayer: 0},  // Branch 3 → bottom-right
}

parallel.ParallelBranches = []nn.LayerConfig{
    nn.InitDenseLayer(64, 16, nn.ActivationReLU),
    nn.InitLSTMLayer(64, 16, 1, 5),
    nn.InitDenseLayer(64, 16, nn.ActivationTanh),
    nn.InitMultiHeadAttentionLayer(64, 2, 1, 8),
}
```

### Filtered Parallel (Mixture of Experts)

```go
// Create MoE with softmax gating
branches := []nn.LayerConfig{
    nn.InitDenseLayer(64, 32, nn.ActivationReLU),     // Expert 0
    nn.InitDenseLayer(64, 32, nn.ActivationReLU),     // Expert 1
    nn.InitDenseLayer(64, 32, nn.ActivationReLU),     // Expert 2
    nn.InitDenseLayer(64, 32, nn.ActivationReLU),     // Expert 3
}

moe := nn.InitFilteredParallelLayer(
    branches,          // The expert branches
    64,               // Input size for gate network
    nn.SoftmaxStandard,  // Softmax type for gating
    1.0,              // Temperature (1.0 = no scaling)
)

network.SetLayer(0, 0, 1, moe)
```

---

## Heterogeneous Architectures

The real power: Each branch can be a completely different architecture.

### Multi-Modal Fusion

```
Input: [image_features | text_features | audio_features]

┌────────────────────────────────────────────────────────────────┐
│                     Parallel Layer                             │
│                                                                │
│  Branch 0: Conv2D           Branch 1: LSTM        Branch 2:    │
│  (for image)               (for text)            Dense+MHA    │
│                                                  (for audio)   │
│  ┌──────────────┐          ┌──────────────┐     ┌────────────┐│
│  │ Conv2D 3×3   │          │ LSTM 64→32   │     │ Dense 64→32││
│  │ Filters=16   │          │ SeqLen=10    │     │ MHA heads=4││
│  └──────┬───────┘          └──────┬───────┘     └─────┬──────┘│
│         │                         │                   │       │
└─────────┼─────────────────────────┼───────────────────┼───────┘
          │                         │                   │
          └─────────────────────────┼───────────────────┘
                                    │
                              Concat: 256 + 32 + 32 = 320
```

### Expert Specialization

Different experts for different input types:

```go
// Fast expert (small, quick)
fastExpert := nn.InitDenseLayer(64, 32, nn.ActivationReLU)

// Deep expert (more layers, better quality)
deepExpert := nn.InitSequentialLayer()
deepExpert.ParallelBranches = []nn.LayerConfig{
    nn.InitDenseLayer(64, 128, nn.ActivationReLU),
    nn.InitDenseLayer(128, 64, nn.ActivationReLU),
    nn.InitDenseLayer(64, 32, nn.ActivationReLU),
}

// Memory expert (LSTM for temporal patterns)
memoryExpert := nn.InitLSTMLayer(64, 32, 1, 10)

// Combine with gating
moe := nn.InitFilteredParallelLayer(
    []nn.LayerConfig{fastExpert, deepExpert, memoryExpert},
    64, nn.SoftmaxTemperature, 0.5,  // Low temp = sharper routing
)
```

---

## Nested Parallel Layers

Parallel layers can contain other parallel layers:

```
                              Input
                                │
                    ┌───────────┼───────────┐
                    │           │           │
                    ▼           ▼           ▼
              ┌─────────┐  ┌─────────┐  ┌─────────┐
              │Parallel │  │  Dense  │  │  LSTM   │
              │  (MoE)  │  │         │  │         │
              │ ┌─┬─┬─┐ │  │         │  │         │
              │ │E│E│E│ │  │         │  │         │
              │ └─┴─┴─┘ │  │         │  │         │
              └────┬────┘  └────┬────┘  └────┬────┘
                   │            │            │
                   └────────────┼────────────┘
                                │
                            Combine
```

```go
// Inner parallel (level 1 MoE)
innerMoE := nn.InitFilteredParallelLayer(
    []nn.LayerConfig{
        nn.InitDenseLayer(64, 32, nn.ActivationReLU),
        nn.InitDenseLayer(64, 32, nn.ActivationReLU),
        nn.InitDenseLayer(64, 32, nn.ActivationReLU),
    },
    64, nn.SoftmaxStandard, 1.0,
)

// Outer parallel (combines MoE with other branches)
outerParallel := nn.InitParallelLayer()
outerParallel.CombineMode = "concat"
outerParallel.ParallelBranches = []nn.LayerConfig{
    innerMoE,                                          // Nested MoE
    nn.InitDenseLayer(64, 32, nn.ActivationReLU),      // Simple dense
    nn.InitLSTMLayer(64, 32, 1, 10),                   // LSTM
}
```

---

## How Gradients Flow

Gradients flow through parallel layers differently based on combine mode:

### Concat Mode

```
Gradient splits by output regions:

gradOutput = [g0, g1, g2, g3, g4, g5, g6, g7, g8]
                 │        │        │
                 ▼        ▼        ▼
              Branch 0  Branch 1  Branch 2
              gets      gets      gets
              [g0,g1,g2] [g3,g4]  [g5,g6,g7,g8]
```

### Add Mode

```
Each branch gets full gradient:

gradOutput = [g0, g1, g2]
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
 Branch 0      Branch 1      Branch 2
 [g0,g1,g2]    [g0,g1,g2]    [g0,g1,g2]

All branches receive identical gradients.
```

### Filter Mode

```
Each branch gets gradient weighted by its gate value:

gradOutput = [g0, g1, g2]
gateWeights = [0.6, 0.3, 0.1]

Branch 0: [g0*0.6, g1*0.6, g2*0.6]
Branch 1: [g0*0.3, g1*0.3, g2*0.3]
Branch 2: [g0*0.1, g1*0.1, g2*0.1]

Gate also gets gradients to learn better routing.
```

---

## Auto-Padding in Filter Mode

Filter mode automatically pads smaller outputs to match the largest:

```
Branch 0 output: [1.0, 2.0, 3.0, 4.0]     (4 values)
Branch 1 output: [5.0, 6.0]               (2 values)
Branch 2 output: [7.0, 8.0, 9.0]          (3 values)

After auto-padding:
Branch 0: [1.0, 2.0, 3.0, 4.0]   (unchanged)
Branch 1: [5.0, 6.0, 0.0, 0.0]   (padded with zeros)
Branch 2: [7.0, 8.0, 9.0, 0.0]   (padded with zeros)

Now weighted sum works element-wise.
```

This allows mixing branches of different output sizes in filter mode.

---

## Sequential Layers

For completeness: Sequential layers run branches in order, not parallel:

```
Input → Branch0 → Branch1 → Branch2 → Output

Each branch's output becomes the next branch's input.
```

```go
sequential := nn.InitSequentialLayer()
sequential.ParallelBranches = []nn.LayerConfig{
    nn.InitDenseLayer(64, 128, nn.ActivationReLU),
    nn.InitDenseLayer(128, 64, nn.ActivationReLU),
    nn.InitDenseLayer(64, 32, nn.ActivationReLU),
}
```

This is useful inside parallel branches when you want a multi-layer expert.

---

## Observers for Parallel Layers

You can attach observers to monitor each branch:

```go
parallel.Observer = nn.NewConsoleObserver()

// During forward pass, you'll see output from each branch:
// [Branch 0] Dense: 64 → 32, output mean=0.12, std=0.45
// [Branch 1] LSTM: 64 → 32, output mean=0.08, std=0.32
// [Branch 2] MHA: 64 → 64, output mean=0.15, std=0.51
```

---

## Practical Example: Multi-Agent System

```go
// Input: game state (64 features)
// Output: 3 agents × 4 actions = 12 action probabilities

network := nn.NewNetwork(64, 1, 1, 4)

// Shared feature extraction
network.SetLayer(0, 0, 0, nn.InitDenseLayer(64, 32, nn.ActivationReLU))

// Parallel agent heads - each agent has different architecture
agentHeads := nn.InitParallelLayer()
agentHeads.CombineMode = "concat"
agentHeads.ParallelBranches = []nn.LayerConfig{
    // Agent 0: Fast reactive (Dense)
    nn.InitDenseLayer(32, 4, nn.ActivationReLU),
    
    // Agent 1: Memory-based (LSTM)
    nn.InitSequentialLayer(),  // Contains LSTM + Dense
    
    // Agent 2: Attention-based (MHA + Dense)
    nn.InitSequentialLayer(),  // Contains MHA + Dense
}

// Configure Agent 1's sequential branch
agentHeads.ParallelBranches[1].ParallelBranches = []nn.LayerConfig{
    nn.InitLSTMLayer(32, 16, 1, 5),
    nn.InitDenseLayer(16, 4, nn.ActivationReLU),
}

// Configure Agent 2's sequential branch
agentHeads.ParallelBranches[2].ParallelBranches = []nn.LayerConfig{
    nn.InitMultiHeadAttentionLayer(32, 2, 1, 8),
    nn.InitDenseLayer(32, 4, nn.ActivationReLU),
}

network.SetLayer(0, 0, 1, agentHeads)

// Grid softmax: 3 agents × 4 actions
network.SetLayer(0, 0, 2, nn.InitGridSoftmaxLayer(3, 4))

// Each agent now has:
// - Shared feature extraction (layer 0)
// - Specialized decision making (parallel branches)
// - Independent action probabilities (grid softmax)
```

---

## Summary

Parallel layers enable:

**Multiple Branch Types**
- Mix Dense, Conv, LSTM, Attention, etc.
- Each branch can have different architecture

**Combine Modes**
- `concat`: Concatenate all outputs
- `add`: Element-wise sum (same size required)
- `avg`: Element-wise average
- `grid_scatter`: Place at 2D/3D positions
- `filter`: Softmax-gated weighted sum (MoE)

**Nesting**
- Parallel can contain Parallel
- Sequential branches for multi-layer experts
- Hierarchical MoE architectures

**Auto-Features**
- Auto-padding for filter mode
- Gradient routing handled automatically
- Observer support for debugging

Use parallel layers to build ensemble models, mixture of experts, multi-modal fusion, and agent-based systems.
