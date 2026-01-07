# Understanding Optimizers and Learning Rate Schedules

This guide explains how optimizers actually update weights—not just the formulas, but the intuition behind why they work and when to use each one.

---

## The Basic Problem: How Do We Improve?

After computing gradients, we know *which direction* would increase the loss. We want to move *opposite* to that direction. But how far should we step?

```
Current weights → Loss = 1.5
                     │
                     │ gradient = -0.2 (pointing toward higher loss)
                     ▼
                     
Step 1: weights += learning_rate × (-gradient)
        weights += 0.01 × 0.2
        weights += 0.002
                     │
                     │ We moved opposite to the gradient
                     ▼
                     
New weights → Loss = 1.4  (improved!)
```

Simple gradient descent works, but it has problems:
1. **Same step size everywhere**: Flat regions need big steps, steep regions need small steps
2. **Oscillation**: Can bounce back and forth in "valleys"
3. **Getting stuck**: Can get trapped in local minima or saddle points

Optimizers solve these problems.

---

## Visualizing the Loss Landscape

Think of training as navigating a hilly landscape, trying to find the lowest valley:

```
     ╱╲    The Loss Landscape
    ╱  ╲   (2D slice for visualization)
   ╱    ╲
  ╱      ╲        ╱╲
 ╱        ╲      ╱  ╲
╱          ╲    ╱    ╲        ╱╲
            ╲  ╱      ╲      ╱  ╲
             ╲╱        ╲____╱    ╲____
                         ↑
                    Global minimum
                    (best solution)
```

Challenges:
- **Steep cliffs**: Gradient is huge, might overshoot
- **Flat plateaus**: Gradient is tiny, progress is slow
- **Narrow valleys**: Oscillate between walls

---

## SGD: The Foundation

Stochastic Gradient Descent is the simplest optimizer:

```go
optimizer := nn.NewSGDOptimizer(momentum)
```

### Basic SGD (No Momentum)

```
weight = weight - learning_rate × gradient

That's it! Move opposite to the gradient, scaled by learning rate.
```

### The Problem: Oscillation

When the loss surface is like a valley (steep in one direction, shallow in another):

```
Top view of a valley-shaped loss landscape:

        Steep walls
            ↓
    ────────────────────
    │                  │
    │   ╱╲  ╱╲  ╱╲    │ ← Path without momentum
    │  ╱  ╲╱  ╲╱  ╲   │    (bouncing between walls)
    │ ╱            ╲  │
    │                 │
    ────────●──────────
             ↑
         Goal (minimum)
```

The gradient points toward the walls, so we bounce back and forth instead of moving toward the minimum.

### Solution: Momentum

Momentum remembers which direction we've been moving and continues in that direction:

```
velocity = momentum × old_velocity + gradient
weight = weight - learning_rate × velocity
```

Think of it like a ball rolling downhill:

```
Without momentum:              With momentum:
    
    ╱╲  ╱╲  ╱╲                       ╲
   ╱  ╲╱  ╲╱  ╲                       ╲
  ╱            ╲                       ╲─────────●
                                               Goal!
  
  Bounces between walls       Builds speed in consistent direction,
                              dampens oscillation
```

### Momentum Values

```
momentum = 0.0:  Pure gradient descent (no memory)
momentum = 0.9:  Standard choice (remembers ~10 past gradients)
momentum = 0.99: Strong momentum (smoother, slower to change direction)
```

### Advanced: Nesterov Momentum

Nesterov looks ahead before computing the gradient:

```
Standard momentum:
    Look at current position → Compute gradient → Update velocity

Nesterov momentum:
    Estimate where we're going → Compute gradient THERE → Update velocity

              Current       →        Where we're headed
                ●────────────────────────▶ ●
                                           │
                                           │ Compute gradient here!
                                           ▼
                                        
This often leads to better convergence because we're planning ahead.
```

---

## AdamW: The Modern Default

Adam (Adaptive Moment Estimation) + Weight Decay. This is what you should use unless you have a specific reason not to.

```go
optimizer := nn.NewAdamWOptimizer(
    0.9,    // beta1 - momentum decay (like SGD momentum)
    0.999,  // beta2 - variance decay (for adaptive LR)
    1e-8,   // epsilon - prevents division by zero
    0.01,   // weight decay - L2 regularization
)
```

### What Adam Does Differently

Adam tracks TWO exponential averages:
1. **First moment (m)**: Average of gradients → like momentum
2. **Second moment (v)**: Average of squared gradients → measures variance

```
m = β₁ × m + (1 - β₁) × gradient       ← Direction (which way to go)
v = β₂ × v + (1 - β₂) × gradient²      ← Scale (how noisy is this param?)

update = m / (√v + ε)         ← Divide momentum by standard deviation
weight -= learning_rate × update
```

### Why This Works

Different parameters need different learning rates:

```
Parameter with consistent gradients (low variance):
    gradient over time: [0.1, 0.12, 0.09, 0.11, 0.10]
    m (momentum): 0.1 (pointing clearly in one direction)
    v (variance): 0.01 (very stable)
    update = 0.1 / √0.01 = 1.0  ← Strong update!

Parameter with noisy gradients (high variance):
    gradient over time: [0.5, -0.3, 0.8, -0.6, 0.2]
    m (momentum): ~0 (averaging out)
    v (variance): 0.25 (very noisy)
    update = 0 / √0.25 = 0.0   ← No update (noise cancels out)
```

### Bias Correction

At the start of training, m and v are initialized to zero. This biases them toward zero. Adam corrects:

```
m̂ = m / (1 - β₁ᵗ)    ← Corrects first moment
v̂ = v / (1 - β₂ᵗ)    ← Corrects second moment

At step 1:
    Uncorrected m = 0.1 × gradient (mostly zero)
    Corrected m̂ = 0.1 × gradient / (1 - 0.9) = gradient (proper scale)
```

### Weight Decay (The "W" in AdamW)

Original Adam had a problem: weight decay interacted badly with adaptive learning rates. AdamW fixes this:

```
Original Adam (wrong):
    Add weight decay to gradient → Affects adaptive scaling

AdamW (correct):
    weight = weight - learning_rate × (update + weight_decay × weight)
                                                ↑
                         Decoupled! Doesn't affect Adam's adaptive behavior
```

### When to Use AdamW

- **Transformers**: Almost always AdamW
- **LLMs**: AdamW is standard
- **Most deep learning**: Safe default choice
- **Fast convergence needed**: Adam finds good solutions quickly

---

## RMSprop: For Recurrent Networks

RMSprop was one of the first adaptive learning rate methods, predating Adam.

```go
optimizer := nn.NewRMSpropOptimizer(
    0.99,   // alpha - decay rate for variance estimate
    1e-8,   // epsilon - numerical stability
    0.0,    // momentum - optional momentum term
)
```

### How It Works

```
v = α × v + (1 - α) × gradient²     ← Running average of squared gradients
weight -= learning_rate × gradient / (√v + ε)
```

This is like Adam's second moment, but without the first moment (momentum).

### When to Use RMSprop

- **RNNs/LSTMs**: Gradients can vary wildly between time steps
- **Non-stationary problems**: Loss landscape changes during training
- **When Adam overfits**: Sometimes simpler is better

---

## Comparing Optimizers

```
                  SGD        SGD+Mom     Adam       AdamW      RMSprop
                  ─────────────────────────────────────────────────────
Convergence:      Slow       Medium      Fast       Fast       Medium
Generalization:   Best       Good        Good       Good       Good
Memory:           O(n)       O(2n)       O(3n)      O(3n)      O(2n)
Hyperparams:      LR         LR, Mom     LR, β1,β2  LR, β1,β2, LR, α
                                                    WD
Use case:         CNNs       CNNs        Most       LLMs       RNNs
```

---

## Learning Rate Schedules

The optimal learning rate changes during training. At first, you want to explore quickly. Later, you want to fine-tune carefully.

### Why Schedules Matter

```
Fixed high LR:                    Fixed low LR:
         │                                 │
 Loss    │  ╱╲  ╱╲  ╱╲                    │╲
         │ ╱  ╲╱  ╲╱                       │ ╲
         │╱                                │  ╲
         └────────────────                 │   ╲
                                           └────────────────────────
         
         Oscillates near minimum,          Takes forever to reach
         never converges precisely         the minimum


Scheduled LR:
         │
 Loss    │╲
         │ ╲
         │  ╲
         │   ╲
         │    ╲──────
         └────────────
         
         Fast initial progress,
         then precise convergence
```

### Constant (No Schedule)

```go
scheduler := nn.NewConstantScheduler(0.001)
```

```
LR │━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   │
   └──────────────────────────────────────▶ Step
   
Simple. Use for debugging or very short training runs.
```

### Linear Decay

```go
scheduler := nn.NewLinearDecayScheduler(
    0.001,   // start LR
    0.0001,  // end LR
    10000,   // total steps
)
```

```
LR │╲
   │ ╲
   │  ╲
   │   ╲
   │    ╲
   │     ╲
   │      ╲━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   └──────────────────────────────────────▶ Step
   
Linear decrease. Simple and effective.
```

### Cosine Annealing

```go
scheduler := nn.NewCosineAnnealingScheduler(
    0.001,   // max LR
    0.0001,  // min LR
    10000,   // total steps
)
```

```
LR │━━━╮
   │   │
   │    ╲
   │     ╲
   │      ╲
   │       ╲
   │         ╲
   │           ╲________━━━━━━━━━━━━━━━━━━
   └──────────────────────────────────────▶ Step
   
Follows cosine curve. Slow decay at start and end, 
faster in the middle. Very popular for transformers.
```

### Cosine with Warm Restarts

```go
scheduler := nn.NewCosineAnnealingWarmRestartsScheduler(
    0.001,   // max LR
    0.0001,  // min LR
    1000,    // T_0 - first period
    2,       // T_mult - multiply period after each restart
)
```

```
LR │╮  ╭───╮     ╭───────────╮
   │ ╲╱    ╲   ╱               ╲
   │        ╲╱                   ╲
   │                               ╲
   │                                 ╲    ╱───────────
   │                                  ╲__╱
   └──────────────────────────────────────▶ Step
        │      │           │
     Restart  Restart    Restart
     (1000)   (2000)     (4000)
     
Periodically "reheats" to escape local minima.
Each cycle is longer than the last.
```

### Warmup

```go
scheduler := nn.NewWarmupScheduler(
    0.001,   // target LR
    1000,    // warmup steps
)
```

```
LR │            ━━━━━━━━━━━━━━━━━━━━━━━━━━
   │          ╱
   │        ╱
   │      ╱
   │    ╱
   │  ╱
   │╱
   └──────────────────────────────────────▶ Step
        │
    Warmup ends
    
Essential for large models. Prevents gradient explosion
at the start when weights are random.
```

### Warmup + Cosine (The Standard)

Combine warmup with another schedule:

```go
warmup := nn.NewWarmupScheduler(0.001, 1000)
cosine := nn.NewCosineAnnealingScheduler(0.001, 0.0001, 9000)
scheduler := nn.NewCompositeScheduler(warmup, cosine, 1000)
```

```
LR │          ╭───╮
   │        ╱      ╲
   │      ╱          ╲
   │    ╱              ╲
   │  ╱                  ╲
   │╱                      ╲___________
   └──────────────────────────────────────▶ Step
       │
   Warmup ends, cosine begins

The most common schedule for training LLMs.
```

### Step Decay

```go
scheduler := nn.NewStepDecayScheduler(
    0.01,    // initial LR
    0.1,     // gamma (decay factor)
    3000,    // step size
)
```

```
LR │━━━━━━━━━━━━╮
   │            │
   │            └━━━━━━━━━━━━╮
   │                         │
   │                         └━━━━━━━━━━━━
   │
   └──────────────────────────────────────▶ Step
             │              │
        LR × 0.1        LR × 0.1

Classic approach for CNNs. "Drop LR every N epochs."
```

### Polynomial Decay

```go
scheduler := nn.NewPolynomialDecayScheduler(
    0.001,   // initial LR
    0.0001,  // final LR
    10000,   // total steps
    2.0,     // power
)
```

```
power = 1.0:  Linear (same as linear decay)
power = 2.0:  Quadratic (starts slow, speeds up)
power = 0.5:  Square root (starts fast, slows down)

Power = 2.0:
LR │╲
   │ ╲
   │  ╲
   │   ╲
   │     ╲
   │        ╲
   │            ╲━━━━━━━━━━━━━━━━━━━━━━━━
   └──────────────────────────────────────▶ Step
   
Adjustable curve shape.
```

---

## Putting It Together

A typical training setup:

```go
// Create network
network := nn.NewNetwork(...)

// Set up optimizer
optimizer := nn.NewAdamWOptimizer(0.9, 0.999, 1e-8, 0.01)
network.SetOptimizer(optimizer)

// Set up schedule
warmupSteps := 1000
totalSteps := 100000
warmup := nn.NewWarmupScheduler(0.001, warmupSteps)
cosine := nn.NewCosineAnnealingScheduler(0.001, 1e-5, totalSteps-warmupSteps)
scheduler := nn.NewCompositeScheduler(warmup, cosine, warmupSteps)

// Training loop
for step := 0; step < totalSteps; step++ {
    lr := scheduler.GetLR(step)
    
    output, _ := network.ForwardCPU(input)
    loss, grad := nn.CrossEntropyLossGrad(output, target)
    network.BackwardCPU(grad)
    network.ClipGradients(1.0)
    network.ApplyGradients(lr)  // Optimizer handles the rest
}
```

---

## Quick Reference: Which to Use?

| Situation | Optimizer | Schedule |
|-----------|-----------|----------|
| **Transformers / LLMs** | AdamW | Warmup + Cosine |
| **CNNs (ImageNet)** | SGD + Momentum | Step decay or Cosine |
| **RNNs / LSTMs** | RMSprop or Adam | Exponential decay |
| **Fine-tuning** | AdamW | Constant (low LR) |
| **Quick experiments** | Adam | Constant |
| **Maximum accuracy** | SGD + Momentum | Warmup + Cosine |

Default values that usually work:
```
AdamW:  LR=0.001, β1=0.9, β2=0.999, weight_decay=0.01
SGD:    LR=0.1, momentum=0.9
Warmup: 1% to 10% of total steps
```
