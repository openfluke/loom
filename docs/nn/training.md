# Understanding Training in Loom

This guide explains how training actually works—the flow of data, the math behind gradient updates, and the practical decisions you'll need to make.

---

## The Training Loop: What's Really Happening

Training a neural network is fundamentally about **adjusting weights to reduce error**. Here's the conceptual loop:

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                            │
│                                                                 │
│   ┌─────────┐                                                   │
│   │  Input  │    Your data (images, text, etc.)                │
│   └────┬────┘                                                   │
│        │                                                        │
│        ▼                                                        │
│   ┌──────────────┐                                              │
│   │ Forward Pass │   Data flows through network                 │
│   │              │   "What does the network predict?"           │
│   └──────┬───────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────┐                                              │
│   │  Prediction  │   Network's output                          │
│   └──────┬───────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────┐   ┌────────┐                                │
│   │  Compute     │◀──│ Target │   Compare to ground truth      │
│   │    Loss      │   │        │   "How wrong are we?"          │
│   └──────┬───────┘   └────────┘                                │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────┐                                              │
│   │ Backward Pass│   Gradients flow backward                   │
│   │              │   "Which weights caused the error?"         │
│   └──────┬───────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────┐                                              │
│   │   Update     │   Weights adjusted to reduce error          │
│   │   Weights    │   "Fix the weights a little bit"            │
│   └──────┬───────┘                                              │
│          │                                                      │
│          └──────────────────────────────────────────────────────┘
│                              │
│                              │ Repeat for many iterations
│                              ▼
│                       Network improves!
└─────────────────────────────────────────────────────────────────┘
```

Let's break down each step.

---

## Step 1: Forward Pass

The forward pass computes what the network predicts given some input.

```go
output, duration := network.ForwardCPU(input)
```

What happens internally:

```
Input: [0.5, 0.3, 0.8, 0.1, ...]   (e.g., pixel values of an image)
         │
         ▼
    ┌─────────────┐
    │   Layer 0   │   weights₀ × input + bias₀ → activation
    │   (Dense)   │   
    └─────┬───────┘
          │
          │   [0.2, 0.7, -0.3, ...]  (hidden representation)
          ▼
    ┌─────────────┐
    │   Layer 1   │   weights₁ × hidden + bias₁ → activation
    │   (Dense)   │
    └─────┬───────┘
          │
          │   [0.1, 0.6, 0.2, 0.05, 0.05]  (class scores)
          ▼
    ┌─────────────┐
    │   Softmax   │   Convert scores to probabilities
    └─────┬───────┘
          │
          ▼
Output: [0.08, 0.45, 0.18, 0.15, 0.14]  ← "45% likely class 1"
                  ↑
            Highest probability
```

Crucially, Loom **saves intermediate values** during the forward pass. These are needed for backpropagation:

```
Layer 0:
    PreActivation: [-0.1, 0.8, -0.5, ...]   Before activation function
    Activation:    [0.0, 0.8, 0.0, ...]     After ReLU
    
Layer 1:
    PreActivation: [0.1, 0.6, 0.2, ...]
    Activation:    [0.1, 0.6, 0.2, ...]
```

---

## Step 2: Compute Loss

The loss function measures **how wrong** the prediction is compared to the target.

### Mean Squared Error (MSE)

Used for regression tasks (predicting continuous values):

```
MSE = (1/n) × Σ(predicted - target)²

Example:
    Predicted: [0.8, 0.3, 0.5]
    Target:    [1.0, 0.0, 0.5]
    
    Errors:    [0.2, 0.3, 0.0]
    Squared:   [0.04, 0.09, 0.0]
    
    MSE = (0.04 + 0.09 + 0.0) / 3 = 0.043


Visual intuition:

Target:     [1.0]────────────────────────●
                                         │
Predicted:  [0.8]──────────────────●     │ error = 0.2
                                   │     │
                                   └─────┘
                                   
Squared error emphasizes larger mistakes.
```

### Cross-Entropy Loss

Used for classification (predicting discrete categories):

```
CrossEntropy = -Σ target[i] × log(predicted[i])

Example (classifying into 3 classes):
    Predicted: [0.7, 0.2, 0.1]   ← Network says "probably class 0"
    Target:    [1.0, 0.0, 0.0]   ← Correct answer IS class 0
    
    CrossEntropy = -(1.0 × log(0.7) + 0.0 × log(0.2) + 0.0 × log(0.1))
                 = -log(0.7)
                 = 0.357
                 
If network was more confident (predicted [0.95, 0.025, 0.025]):
    CrossEntropy = -log(0.95) = 0.051  ← Lower loss (better!)
    
If network was wrong (predicted [0.1, 0.7, 0.2]):
    CrossEntropy = -log(0.1) = 2.303   ← Higher loss (worse!)
```

The key property: **Cross-entropy heavily penalizes confident wrong answers**.

---

## Step 3: Backward Pass

The backward pass figures out **how to change each weight to reduce the loss**. This is where calculus comes in—specifically, the chain rule.

### The Chain Rule Intuition

Imagine you're adjusting a dial on a machine:
- The dial affects some internal gear
- That gear affects another gear
- Eventually, it affects the output

To know how much your dial affects the output, you multiply the effects at each stage:

```
dial → gear1 → gear2 → output

∂output/∂dial = (∂output/∂gear2) × (∂gear2/∂gear1) × (∂gear1/∂dial)

"How much does output change" = product of local sensitivities
```

### Computing Gradients

Starting from the loss, we compute how much each weight contributed:

```
Loss: 0.357
  │
  │  ∂Loss/∂output = gradient of loss function
  ▼
┌──────────────────────────────────────────────────────────────┐
│ Softmax Layer  (output)                                      │
│                                                              │
│   ∂Loss/∂logits = softmax_gradient(∂Loss/∂output)           │
│                                                              │
│   For cross-entropy + softmax, this simplifies to:          │
│   ∂Loss/∂logits = predicted - target                        │
│                 = [0.7, 0.2, 0.1] - [1.0, 0.0, 0.0]         │
│                 = [-0.3, 0.2, 0.1]                          │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             │  Pass gradient to previous layer
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ Dense Layer 1                                                │
│                                                              │
│   gradOutput = [-0.3, 0.2, 0.1]                             │
│                                                              │
│   For weights:                                               │
│       ∂Loss/∂W[i,j] = gradOutput[i] × input[j]              │
│                                                              │
│   For input (to pass backward):                              │
│       gradInput[j] = Σᵢ gradOutput[i] × W[i,j]              │
│                                                              │
│   For bias:                                                  │
│       ∂Loss/∂bias[i] = gradOutput[i]                        │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             │  Pass gradient to previous layer
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ Dense Layer 0                                                │
│                                                              │
│   Same computation with gradInput from above                 │
│                                                              │
│   Also consider activation function derivative:              │
│       For ReLU: gradient is 0 if preActivation < 0          │
│                 gradient passes through if preActivation ≥ 0│
└──────────────────────────────────────────────────────────────┘
```

### Activation Function Derivatives

The activation function's derivative determines how gradients flow:

```
ReLU:
    Forward:  y = max(0, x)
    Backward: ∂y/∂x = 1 if x > 0, else 0
    
    Interpretation: ReLU "kills" gradients for negative inputs.
    
           │        ╱
           │       ╱
    ───────┼──────╱────────
           │     ╱
           │    ╱
           
    Gradient can only flow through the positive half.


Sigmoid:
    Forward:  y = 1 / (1 + e^(-x))
    Backward: ∂y/∂x = y × (1 - y)
    
    Maximum gradient at y = 0.5 (uncertainty)
    Gradient vanishes as y → 0 or y → 1 (confidence)
    
           │    ┌───────────
           │   ╱
    ───────┼──╱────────────
           │ ╱
    ───────╱───────────────


Tanh:
    Forward:  y = (e^x - e^(-x)) / (e^x + e^(-x))
    Backward: ∂y/∂x = 1 - y²
    
    Similar to sigmoid but centered at 0.
```

---

## Step 4: Update Weights

Now we have gradients—directions that would **increase** the loss. We move **opposite** to the gradients to decrease loss.

### Basic Gradient Descent

```go
network.ApplyGradients(learningRate)
```

For each weight:
```
new_weight = old_weight - learning_rate × gradient

        Example:
        weight = 0.5
        gradient = 0.2
        learning_rate = 0.01
        
        new_weight = 0.5 - (0.01 × 0.2) = 0.498
        
        Small step in the direction that reduces loss!
```

### Why Learning Rate Matters

```
Learning rate too small:
    ───●───●───●───●───●───●───●───●───●───●───●───●───●───○
    
    Takes forever to reach the goal (minimum loss)


Learning rate too large:
    ──────●
          │
          │ ────── ●
                   │
           ● ──────│
           │
    ●──────│
    
    Overshoots and oscillates, may never converge


Just right:
    ──────●
           ╲
            ╲───●
                 ╲
                  ●──●──●──○
                  
    Steady progress to the minimum
```

Typical learning rates: `0.001`, `0.01`, `0.0001` depending on the optimizer.

---

## Gradient Clipping: Preventing Explosions

Sometimes gradients get very large, especially in recurrent networks. Gradient clipping limits their magnitude:

```go
network.ClipGradients(maxNorm)  // e.g., maxNorm = 1.0
```

How it works:

```
Before clipping:
    gradients = [100.0, -200.0, 50.0]
    norm = sqrt(100² + 200² + 50²) = 229.1
    
With maxNorm = 1.0:
    scale = 1.0 / 229.1 = 0.0044
    clipped = [0.44, -0.87, 0.22]
    
    Direction preserved, magnitude limited!


Visual:
    
    Before:                After (maxNorm=1):
    
         ↑ [200]                 ↑ [0.87]
         │                       │
         │                   ────●────
    ─────●─────             
         │                       │
                                 │
         
    Huge gradient              Unit gradient (same direction)
```

---

## Optimizers: Smarter Weight Updates

Plain gradient descent treats all directions equally. Optimizers use history to make smarter updates.

### SGD with Momentum

Momentum builds up speed in consistent directions:

```
Without momentum:          With momentum:
     
     ╱╲╱╲╱╲╱╲              ────────────────→
       │                          │
       ●  (goal)                  ●  (goal)
       
Oscillates in one          Smoothly accelerates
dimension, slow in          in consistent direction
another
```

The math:
```
velocity = momentum × old_velocity + gradient
weight = weight - learning_rate × velocity

With momentum = 0.9:
    If gradient keeps pointing the same way:
        velocity grows: 1 → 1.9 → 2.71 → 3.44 → ...
        Accelerates toward the minimum!
    
    If gradient oscillates (sign changes):
        velocity dampens: 1 → 0.9-1 → 0.1 → ...
        Reduces oscillation!
```

### AdamW: The Modern Choice

AdamW combines momentum with adaptive learning rates:

```
Adam tracks:
    m = exponential average of gradients (momentum)
    v = exponential average of squared gradients (variance)

Update:
    m = β₁ × m + (1-β₁) × gradient
    v = β₂ × v + (1-β₂) × gradient²
    
    adjusted_lr = learning_rate / (√v + ε)
    
    weight = weight - adjusted_lr × m

What this means:
    • Parameters with consistent gradients get accelerated (momentum)
    • Parameters with noisy gradients get smoothed (variance tracking)
    • Parameters with large gradients get smaller steps (adaptive LR)


AdamW also adds weight decay:
    weight = weight - weight_decay × weight
    
    This prevents weights from growing too large (regularization).
```

### RMSprop: For Recurrent Networks

RMSprop adapts learning rate based on recent gradient history:

```
v = α × v + (1-α) × gradient²
weight = weight - learning_rate × gradient / (√v + ε)

Good for:
    • Recurrent networks (gradients can vary wildly)
    • Non-stationary problems (changing loss landscape)
```

---

## Learning Rate Schedules

The optimal learning rate changes during training. Schedulers adjust it over time.

### Common Schedules

```
Constant:
    ──────────────────────────────────────
    Same rate throughout. Simple but often suboptimal.


Linear Decay:
    ╲
     ╲
      ╲
       ╲
        ╲─────────────────────────────────
    Start high, decrease linearly to final rate.


Cosine Annealing:
    ╭───╮
    │   │
    │    ╲
    │     ╲
    │      ╲
    │       ╲
    ╯        ╲──────────────────────────────
    Smooth decay following cosine curve.
    Popular for transformers.


Warmup + Decay:
           ╱╲
          ╱  ╲
         ╱    ╲
        ╱      ╲
       ╱        ╲
    ──╱          ╲────────────────────────
    
    Start low (warmup), increase, then decay.
    Essential for large models—prevents early instability.


Step Decay:
    ────────┐
            │
            └──────┐
                   │
                   └──────────────────────
    
    Drop by factor every N epochs.
    Classic approach for CNNs.
```

### Why This Matters

```
Early training:
    • Loss landscape is rough
    • Network is far from optimum
    • Large learning rate explores quickly

Late training:
    • Loss landscape is smoother (near optimum)
    • Fine-tuning requires precision
    • Small learning rate prevents overshooting

Hence: warm up → train at high rate → cool down
```

---

## Putting It Together: A Complete Training Loop

```go
func train(network *nn.Network, data []Sample, epochs int) {
    // 1. Set up optimizer
    optimizer := nn.NewAdamWOptimizer(0.9, 0.999, 1e-8, 0.01)
    network.SetOptimizer(optimizer)
    
    // 2. Set up learning rate schedule
    warmupSteps := 100
    totalSteps := len(data) * epochs
    scheduler := nn.NewWarmupScheduler(0.001, warmupSteps)
    
    // 3. Training loop
    step := 0
    for epoch := 0; epoch < epochs; epoch++ {
        epochLoss := float32(0)
        
        for _, sample := range data {
            // Get current learning rate
            lr := scheduler.GetLR(step)
            
            // Forward pass
            output, _ := network.ForwardCPU(sample.Input)
            
            // Compute loss and gradient
            loss, gradOutput := nn.CrossEntropyLossGrad(output, sample.Target)
            epochLoss += loss
            
            // Backward pass
            network.BackwardCPU(gradOutput)
            
            // Gradient clipping
            network.ClipGradients(1.0)
            
            // Update weights
            network.ApplyGradients(lr)
            
            step++
        }
        
        avgLoss := epochLoss / float32(len(data))
        fmt.Printf("Epoch %d: Loss = %.4f\n", epoch, avgLoss)
    }
}
```

### What Each Line Does

1. **Optimizer setup**: AdamW will track momentum and variance for each parameter
2. **Scheduler setup**: Learning rate starts low, warms up over 100 steps
3. **Forward pass**: Data flows through network, activations cached
4. **Loss + gradient**: Measure error and compute output gradient
5. **Backward pass**: Propagate gradients through all layers
6. **Clip gradients**: Prevent explosions by limiting gradient magnitude
7. **Apply gradients**: Optimizer updates weights using gradients

---

## Debugging Training: What to Watch

### Loss Not Decreasing?

```
Possible causes:

1. Learning rate too small
   → Gradients are tiny, weights barely move
   → Try: Increase by 10×

2. Learning rate too large
   → Loss oscillates or explodes
   → Try: Decrease by 10×

3. Vanishing gradients
   → Early layers have near-zero gradients
   → Try: Check activation functions (avoid deep sigmoid)

4. Exploding gradients
   → Loss becomes NaN or Inf
   → Try: Gradient clipping, smaller learning rate

5. Data problem
   → Network can't learn the pattern at all
   → Try: Simpler problem first, check data preprocessing
```

### Debugging Commands

```go
// Print gradient statistics
network.PrintGradientStats()
// Shows: min, max, mean gradient per layer
// Look for: zeros (vanishing) or huge values (exploding)

// Monitor with observer
observer := nn.NewConsoleObserver()
network.SetObserver(observer)
// Prints activation statistics after each layer
```

---

## Summary

Training is a loop of:
1. **Forward**: Compute predictions
2. **Loss**: Measure error
3. **Backward**: Compute gradients (who's responsible?)
4. **Update**: Adjust weights (fix the responsible parts)

Key decisions:
- **Loss function**: MSE for regression, CrossEntropy for classification
- **Optimizer**: AdamW is a safe default, SGD+Momentum for CNNs
- **Learning rate**: Start around 0.001, adjust based on results
- **Schedule**: Warmup + decay for large models
- **Clipping**: Use gradient clipping for stability

The art of training is balancing exploration (high learning rate, quick convergence) vs precision (low learning rate, optimal solution).

## Training on GPU

Loom supports training on GPU using the **exact same API** as CPU training. You don't need to change your loop structure.

### Enabling GPU Training

Just initialize the GPU before your loop:

```go
// 1. Initialize GPU
if err := network.InitGPU(); err != nil {
    panic(err)
}
defer network.ReleaseGPU()

// 2. Run your standard loop
for epoch := 0; epoch < epochs; epoch++ {
    // ...
    // ForwardCPU automatically routes to GPU
    output, _ := network.ForwardCPU(input)
    
    // ...
    // BackwardCPU automatically routes to GPU
    network.BackwardCPU(gradOutput)
    
    // Updates happen on GPU memory
    network.ApplyGradients(lr)
}
```

See [GPU Layers](./gpu_layers.md) for a list of layers that support GPU training.
