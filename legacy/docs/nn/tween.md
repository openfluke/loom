# Understanding Neural Tweening

Neural Tweening is Loom's unique training algorithm. Unlike standard backpropagation that pushes information in one direction, Tweening works from **both ends simultaneously**—meeting in the middle.

This document explains *what* Neural Tweening does, *why* it works, and *when* you'd want to use it.

---

## The Problem with Standard Backpropagation

In traditional training, information flows like this:

```
Standard Backpropagation:

Forward Pass:
Input ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▶ Output → Loss

Backward Pass:
Input ◀━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Output ← Loss Gradient
                                              │
                                              │ "How wrong were we?"
                                              │
                                              ▼
                                       Update weights
```

The problem? **Gradients have a long journey**. In a 50-layer network, the gradient must travel through all 50 layers. Two bad things can happen:

1. **Vanishing Gradients**: Gradients get multiplied by small numbers repeatedly and shrink to nearly zero. Early layers barely learn.

2. **Gradient Explosion**: Gradients get multiplied by large numbers and grow exponentially. Training becomes unstable.

```
Vanishing Gradient Problem:

Layer 50 gradient: 1.0
    × 0.9
Layer 49 gradient: 0.9
    × 0.9
Layer 48 gradient: 0.81
    × 0.9
    ...
Layer 1 gradient:  0.9^50 ≈ 0.005  ← Almost nothing!

The first layers learn 200× slower than the last layers.
```

---

## Neural Tweening: The Bidirectional Approach

Tweening attacks the problem from both directions at once:

```
Neural Tweening:

                          "Meet in the middle"
                                  │
                                  ▼
          ┌───────────────────────┬───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
Input ━━━▶  Forward   ━━━▶ Layer 25 ◀━━━  Backward  ◀━━━ Target
           Activations             │      Targets
                                   │
                                   ▼
                              Compare!
                            "What's the gap?"
```

Here's the key insight: **we propagate from BOTH ends and measure where they disagree**.

### Step 1: Forward Pass - Capture Activations

We run input through the network and save every layer's output:

```
Input [x₁, x₂, x₃...]
    │
    ▼
Layer 0 ───▶ Save activation a₀
    │
    ▼
Layer 1 ───▶ Save activation a₁
    │
    ▼
Layer 2 ───▶ Save activation a₂
    │
    ...
    │
    ▼
Layer N ───▶ Save activation aₙ = Output
```

These saved activations tell us: "What did the network actually compute at each step?"

### Step 2: Backward Pass - Compute Target Gradients

Starting from the desired output (target), we propagate backward to compute what each layer *should* have produced:

```
                                           Target [t₁, t₂, t₃...]
                                                      │
                                                      ▼
Save target tₙ ◀─── Compute from target ─── Layer N  │
                                                      ▼
Save target tₙ₋₁ ◀─── Backprop ──────────── Layer N-1│
                                                      ▼
Save target tₙ₋₂ ◀─── Backprop ──────────── Layer N-2│
                                                      │
                                                     ...
```

These backward targets tell us: "What should each layer have produced to get the right answer?"

### Step 3: Compute Gaps

Now we compare what each layer *did* produce (forward activations) vs what it *should* have produced (backward targets):

```
Layer    Forward              Backward            Gap
         Activation           Target              
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  0      [0.2, 0.8, 0.1]      [0.3, 0.7, 0.2]     0.15
  1      [0.5, 0.3, 0.2]      [0.4, 0.4, 0.2]     0.10
  2      [0.7, 0.2, 0.1]      [0.6, 0.3, 0.1]     0.12
  3      [0.9, 0.05, 0.05]    [0.8, 0.1, 0.1]     0.15
  ...
  
Gap = how "wrong" each layer is = Mean Squared Error between activation and target
```

### Step 4: Update Weights Based on Gaps

Now we know exactly where each layer is going wrong. We can update weights to reduce the gap:

```
For each layer:
    
    gap = activation - target
    
    Compute gradient using chain rule:
        ∂Loss/∂weights = ∂Loss/∂activation × ∂activation/∂weights
    
    Apply gradient with momentum:
        velocity = momentum × old_velocity - learning_rate × gradient
        weights += velocity
```

The key advantage: **Every layer gets meaningful gradient information**. We don't rely on gradients surviving a 50-layer journey.

---

## Visual Comparison

```
Standard Backprop:

    Input ━━━━━▶ ━━━━━▶ ━━━━━▶ ━━━━━▶ Output
                                       │
     ◀━━━━━ ◀━━━━━ ◀━━━━━ ◀━━━━━ ◀━━━━┛
     weak    weak   weak   weak  strong
    gradient                    gradient
    
    Problem: Early layers get weak signals


Neural Tweening:

    Input ━━━━━▶ ━━━━━▶    ━━━━━▶ ━━━━━▶ Output
                      │    │
                      ▼    ▼
                    Compare at
                    every layer
                      │    │
    Target ◀━━━━━ ◀━━┘    └━━▶ ◀━━━━━ Target
    
    Every layer gets direct gradient information!
```

---

## Link Budgets: Measuring Information Flow

A unique feature of Tweening is **link budget tracking**. This measures how well information is preserved as it flows through the network.

### What is a Link Budget?

Think of it like a signal passing through a chain of amplifiers:

```
Signal strength through network:

Input: 1.0 ━━━▶ Layer 0 ━━━▶ Layer 1 ━━━▶ Layer 2 ━━━▶ Output
        │         │           │            │             │
     Strength   0.95        0.87         0.75          0.60
        
Link budgets:        0.95        0.92         0.86
                   (0.95/1.0)  (0.87/0.95)  (0.75/0.87)

Link budget < 1.0: Information is being lost (compression)
Link budget = 1.0: Information is preserved
Link budget > 1.0: Information is being amplified
```

### Why Track Link Budgets?

Link budgets reveal **where information bottlenecks exist**:

```
Good network:
    Layer 0: 0.95  ← Healthy
    Layer 1: 0.92  ← Healthy
    Layer 2: 0.98  ← Healthy
    Layer 3: 0.89  ← Healthy

Problematic network:
    Layer 0: 0.95  ← Healthy
    Layer 1: 0.30  ← BOTTLENECK! Information is being crushed
    Layer 2: 1.50  ← Trying to compensate by amplifying
    Layer 3: 0.20  ← Unstable
```

If link budgets are too low, the network is losing information. If they're too high and unstable, the network might be in danger of exploding.

---

## Explosion Detection: Automatic Stability

One of Tweening's most practical features is **automatic explosion detection**. Deep networks sometimes enter unstable states where gradients or activations explode.

### How It Works

Tweening tracks the **gap growth rate**:

```
Step 1: Average gap = 0.10
Step 2: Average gap = 0.12  → Growth rate = 1.2×
Step 3: Average gap = 0.15  → Growth rate = 1.25×
Step 4: Average gap = 0.45  → Growth rate = 3.0× ← EXPLOSION DETECTED!
Step 5: Average gap = 1.82  → Would be 4.0× ← But we intervened!
```

When explosion is detected:
1. **Reduce learning rate** temporarily
2. **Reset momentum** to break the feedback loop
3. **Resume training** with more conservative updates

```
Explosion handling:

Gap timeline:
     │    ╭───────╮
     │   ╱   Boom! ╲
     │  ╱    ↓     ╲
 Gap │ ╱  [Detect]  ╲────── Recovery
     │╱      │           ╲
     └───────────────────────▶ Time
              │
     Learning rate reduced here
     Momentum reset here
```

### The Recovery Process

```go
if gapGrowthRate > explosionLimit {
    explosionCount++
    
    // Reduce learning rate temporarily
    adaptiveRate *= recoveryRate  // e.g., 0.5
    
    // Clear momentum to stop the feedback loop
    for i := range weightVelocity {
        weightVelocity[i] = zeros
    }
    
    // Continue training with conservative updates
}
```

This automatic stability is why Tweening **never crashes to 0% accuracy** during task changes—it detects instability before it becomes catastrophic.

---

## The TweenState: What It Tracks

The `TweenState` structure maintains everything needed for bidirectional training:

```
TweenState {
    // Forward activations (what the network computed)
    ForwardActs: [
        activation_layer_0,
        activation_layer_1,
        activation_layer_2,
        ...
    ]
    
    // Backward targets (what each layer should have computed)
    BackwardTargets: [
        target_layer_0,
        target_layer_1,
        target_layer_2,
        ...
    ]
    
    // Gaps between forwards and backwards
    Gaps: [0.12, 0.08, 0.15, ...]
    
    // Information preservation metrics
    LinkBudgets: [0.95, 0.92, 0.88, ...]
    
    // Momentum for stable updates
    WeightVelocity: [[...], [...], ...]
    BiasVelocity:   [[...], [...], ...]
    
    // Chain rule gradients for each layer
    ChainGradients: [[...], [...], ...]
    
    // Stability tracking
    PrevAvgGap: 0.10
    GapGrowthRate: 1.15
    ExplosionCount: 0
    AdaptiveRate: 0.01
    
    // History for analysis
    LossHistory: [1.5, 1.2, 0.9, 0.6, ...]
}
```

---

## The TweenStep: One Complete Iteration

Here's what happens in a single `TweenStep`:

```
TweenStep(network, input, target, learningRate):

    ┌─────────────────────────────────────────────────────────┐
    │ 1. FORWARD PASS                                         │
    │    • Run input through network                          │
    │    • Save activation at each layer                      │
    │    • Get final output                                   │
    └────────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 2. BACKWARD PASS                                        │
    │    • Start from (output - target) gradient              │
    │    • Propagate backward through each layer              │
    │    • Save gradient/target at each layer                 │
    │    • Apply depth scaling for stability                  │
    └────────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 3. COMPUTE METRICS                                      │
    │    • Calculate gaps: |forward - backward|               │
    │    • Calculate link budgets: output_norm / input_norm   │
    │    • Check for explosion                                │
    └────────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 4. EXPLOSION HANDLING (if needed)                       │
    │    • Reduce adaptiveRate                                │
    │    • Reset momentum                                     │
    │    • Increment explosionCount                           │
    └────────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 5. UPDATE WEIGHTS                                       │
    │    For each layer:                                      │
    │        velocity = momentum × velocity - rate × gradient │
    │        weights += velocity                              │
    │        bias += biasVelocity                             │
    └────────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 6. COMPUTE LOSS                                         │
    │    • loss = CrossEntropy(output, target)                │
    │    • Append to lossHistory                              │
    │    • Return loss                                        │
    └─────────────────────────────────────────────────────────┘
```

---

## Depth Scaling: Why Deep Layers Get Special Treatment

In very deep networks, later layers (closer to output) have more direct influence on the loss. To prevent them from dominating training, Tweening applies **depth scaling**:

```
Depth scaling formula:
    
    scale = 1.0 / sqrt(totalLayers - layerIndex)
    
    Example with 16 layers:
    
    Layer  Distance    Scale
           to output   factor
    ─────────────────────────
    0      16          0.25   ← Far from output, small gradient
    4      12          0.29
    8      8           0.35
    12     4           0.50
    15     1           1.00   ← Close to output, full gradient
```

This ensures gradients are balanced across the network depth.

---

## When to Use Neural Tweening

Tweening excels in specific scenarios:

### ✅ Use Tweening When:

| Scenario | Why Tweening Helps |
|----------|-------------------|
| **Shallow networks (< 10 layers)** | Can achieve 100% accuracy |
| **Task switching** | Maintains stability during adaptation |
| **Online learning** | Continuous updates without crashes |
| **Debugging gradient flow** | Link budgets reveal bottlenecks |
| **Unstable training** | Explosion detection prevents crashes |

### ❌ Consider Standard Backprop When:

| Scenario | Why Standard May Be Better |
|----------|---------------------------|
| **Very deep networks (50+ layers)** | Tweening overhead adds up |
| **Batch training** | Standard backprop better optimized |
| **Well-tuned architectures** | If gradients already flow well |
| **Maximum speed** | Tweening has 2× forward passes |

---

## Practical Example

```go
// Create network
network := nn.NewNetwork(784, 2, 1, 3)  // MNIST-sized

// Set up layers...

// Create TweenState
config := &nn.TweenConfig{
    BaseRate:        0.01,     // Base learning rate
    MomentumDecay:   0.9,      // Momentum coefficient
    LinkBudgetScale: 1.0,      // Scale for budget computation
    GapThreshold:    0.1,      // When to react to gaps
    ExplosionLimit:  10.0,     // Gradient explosion threshold
    RecoveryRate:    0.5,      // LR reduction on explosion
}

ts := nn.NewTweenState(network, config)

// Training loop
for epoch := 0; epoch < 100; epoch++ {
    for _, sample := range trainingData {
        // One complete bidirectional step
        loss := ts.TweenStep(
            network, 
            sample.Input,        // Input tensor
            sample.Label,        // Target class (for classification)
            10,                  // Number of classes
            config.BaseRate,     // Learning rate
            backend,             // CPU or GPU backend
        )
        
        if epoch % 10 == 0 {
            // Monitor training health
            avgGap := nn.Mean(ts.Gaps)
            avgBudget := nn.Mean(ts.LinkBudgets)
            
            fmt.Printf("Loss: %.4f, Gap: %.4f, Budget: %.4f\n",
                loss, avgGap, avgBudget)
            
            if avgBudget < 0.5 {
                fmt.Println("⚠️ Information bottleneck detected!")
            }
        }
    }
}

// After training, check explosion count
fmt.Printf("Training completed with %d explosions detected/recovered\n",
    ts.ExplosionCount)
```

---

## Performance Characteristics

Based on benchmarks from `/docs/step_tween_assessment.md`:

| Metric | Standard Backprop | Neural Tweening |
|--------|------------------|-----------------|
| **Shallow network accuracy** | 90-95% | **100%** |
| **Task switch stability** | 0-20% | **40-80%** |
| **Explosion rate** | Variable | ~0% (auto-recovery) |
| **Convergence speed** | Fast | Moderate |
| **Memory overhead** | 1× | ~2× (stores forward + backward) |
| **Compute overhead** | 1× | ~1.5× (extra metrics) |

The tradeoff: **slower but more stable and interpretable**.

---

## Deep Dive: The Chain Rule Gradients

For those interested in the math, here's how gradients are computed for different layers:

### Dense Layer

```
Input: x (size n)
Weights: W (size m × n)
Bias: b (size m)
Output: y = activation(W × x + b)

Gradient from output: ∂L/∂y

Gradient for weights:
    ∂L/∂W[i,j] = ∂L/∂y[i] × activation'(preact[i]) × x[j]

Gradient for input:
    ∂L/∂x[j] = Σᵢ ∂L/∂y[i] × activation'(preact[i]) × W[i,j]

Gradient for bias:
    ∂L/∂b[i] = ∂L/∂y[i] × activation'(preact[i])
```

### Attention Layer

```
Q, K, V projections
Attention weights: A = softmax(Q × K.T / sqrt(d_k))
Output: O = A × V

Gradient flow:
    • Through V → to values
    • Through A → softmax Jacobian → to QK product
    • Through Q, K → to input projections
```

### LSTM Gates

```
Gates: forget (f), input (i), cell candidate (g), output (o)

Each gate has its own gradient path.
Cell state provides a "highway" for gradients.
Tweening measures gaps at hidden state AND cell state.
```

---

## Summary

Neural Tweening is a bidirectional training algorithm that:

1. **Propagates from both ends** - Forward activations meet backward targets
2. **Measures gaps** - Identifies exactly where each layer is wrong
3. **Tracks link budgets** - Monitors information preservation
4. **Detects explosions** - Automatically recovers from instability
5. **Scales with depth** - Balances gradient importance across layers

It trades some speed for significantly improved stability and interpretability. Use it when you need reliable training, especially for task switching and online learning scenarios.
