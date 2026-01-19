# Step Tween Chain: Real-Time Neural Dynamics

The **Step Tween Chain (STC)** is the operational "heartbeat" of Loom. While standard [Neural Tweening](tween.md) provides the bidirectional framework, the **Step Tween Chain** is the process of applying this logic continuously—step-by-step—during live operation.

This document explains how each layer type in `nn/types.go` behaves when it becomes a link in a Step Tween Chain.

---

## The Core Concept: Equilibrium vs. Learning

In a Step Tween Chain, the network isn't just "learning" from a static dataset; it is constantly trying to reach **equilibrium** between:
1.  **Bottom-Up Data (Forward):** What actually came from the input.
2.  **Top-Down Intent (Backward):** What the output layer or goal layer *needs* right now.

The **"Gap"** at each layer represents the distance from this equilibrium.

---

## Layer-by-Layer Dynamics

### 1. Dense Layers: Spatial Error Resolution
Dense layers act as the "flexible muscle" of the chain. 
*   **Behavior:** During a Step Tween, the Dense layer tries to rotate its entire weight matrix to align with the gradient. 
*   **The Chain Effect:** It acts as a **Global Resolver**. If the error is large and spread out, the Dense layer attempts to find a linear shortcut to bridge the gap.
*   **Example:** If your input is a flat vector of sensor data, the Dense layer tweens to "map" those sensors to the desired output as fast as possible.
*   **Best For:** Simple classification, regression, or mapping raw numbers (like temperature/pressure) to a status. Use when the spatial relationship between inputs doesn't matter.

### 2. Conv1D & Conv2D: Spatio-Temporal Stabilization
Convolutional layers are the "Spatial Anchors" of the chain.
*   **Behavior (The Holographic Effect):** Conv2D doesn't just see a flat image; it sees the **displacement of features** over time. As you tween, the convolution kernels act like **Pseudo-3D filters**.
*   **The Chain Effect:** They stabilize the training by enforcing local consistency. Because a kernel slides across the data, it refuses to let one single pixel "explode" unless the neighboring pixels agree.
*   **3D Projections:** In a Step Tween Chain, a Conv2D layer essentially learns the **momentum of features**, allowing it to project where a pattern *will be* in the next step.
*   **Best For:** Video tracking, real-time image stabilization, or any data with spatial structure. Use when you need the network to identify "objects" or "patterns" and how they move through space.

### 3. RNN & LSTM: Temporal Synchronization
Recurrent layers are the "Memory Buffers" of the chain.
*   **Behavior:** While other layers tween spatially, these layers tween **temporarily**. They compare the "Gap" of the current step against the cached memory of the previous steps.
*   **The Chain Effect:** They act as **Temporal Dampers**. They prevent the network from "Panic Training" on a single bad frame. If the current frame suggests a massive change, but the previous 10 frames were stable, the LSTM gates will "dampen" the tweening signal.
*   **Example:** In a real-time conversation or motion sequence, the LSTM ensures the "Gap" closes smoothly across time, not just in a single burst.
*   **Best For:** Natural Language Processing (NLP), sequence prediction (stock prices, weather), or real-time control systems (robotics). Use when the "reason" for the current event happened 10 steps ago.

### 4. Multi-Head Attention (MHA): Contextual Steering
Attention layers are the "Searchlights" of the chain.
*   **Behavior:** MHA doesn't care about "where" data is; it cares about "what" data is related to. During a tween, it shifts its "Heads" to focus on whichever part of the input is causing the largest Gap.
*   **The Chain Effect:** It acts as a **Dynamic Re-Router**. If the error is coming from a specific tiny detail in a large image, the MHA will tween its Query/Key values to "shine the light" purely on that detail, ignoring the rest of the stable network.
*   **Best For:** Complex decision making, multi-object tracking, or large context windows (like reading a long code file). Use when the network needs to "ignore" 90% of the input to focus on the 10% that matters.

### 5. Norm Layers (LayerNorm, RMSNorm): Voltage Regulation
Normalization layers are the "Safety Fuses" of the chain.
*   **Behavior:** As tweening forces weights to move, values can get very large. Norm layers constantly "squash" these values back to a standard deviation of 1.
*   **The Chain Effect:** They provide **Scale Invariance**. They ensure that a "Gap" found in a high-precision layer isn't 1000x larger than a "Gap" in a low-precision layer. They keep the tweening signal consistent across the whole chain.
*   **Best For:** Multi-precision training (F16/F32) and deep grids. Use when you want to prevent one layer from "screaming" louder than the others during tweening just because its numbers are bigger.

### 6. Structural Layers (Residual, Parallel)
These define the **Topology of the Gap**.
*   **Residual (Skip):** Acts as the **Bypass**. If a layer is too "stiff" and can't tween fast enough to close its gap, the Residual connection carries the signal around it. This prevents the chain from "stalling."
*   **Parallel (Branches):** Acts as the **Ensemble Fusion**. You are tweening multiple "opinions" at once. If Branch A has a high link budget and Branch B has a low one, the chain will automatically favor Branch A's contribution to the final equilibrium.
*   **Best For:** Redundant systems (e.g., combining camera + radar data) or hierarchical learning. Use when you want the network to have "fail-safes"—if one branch fails to close its gap, the others can take over.

---

## Practical Example: A Vision-to-Motion Chain

Imagine a network with: `Conv2D -> LSTM -> Dense`

1.  **Forward:** An image comes in (Conv), it moves (LSTM), a motor command is produced (Dense).
2.  **Tween Step:** The motor command was slightly off (Gap at the end).
3.  **Backprop Target:** The error travels back.
4.  **Dense Logic:** "I need to turn the motor 5 degrees left."
5.  **LSTM Logic:** "I remember we were moving right, so I'll smooth this transition."
6.  **Conv2D Logic:** "The object moved on the 2D plane; I'll adjust my spatial filters to track that motion better in the next step."

**Result:** Within a few steps of the chain, the "Gap" closes and the network is synchronized with the live reality.

---

## Theoretical Insights: Neural Fluid Dynamics

Loom's Step Tween Chain differs from standard AI frameworks (like PyTorch or TensorFlow) by treating the network as a **living biological system** rather than a static computation graph.

### 1. The Information Field (vs. The Sweep)
Standard frameworks use a "sweep" approach: data flows in, is processed, and is forgotten.
*   **The Loom Way:** Using **double buffering** in `StepForward`, information literally "lives" inside the layers. This turns the network into a simulation of a **physical field** where inputs and targets "vibrate" against each other until they reach equilibrium.

### 2. Liquid Gradients
The implementation of `applySoftmaxGradientScaling` in `step_backward.go` introduces what we call **Liquid Gradients**.
*   **Traditional:** Standard clipping "hits gradients with a hammer" if they get too large.
*   **The Loom Way:** We use a **Softmax Score** to redistribute error energy across a layer. This allows the network to "flow" into a new shape during task changes rather than breaking or stalling.

### 3. Continuous Plasticity (Real-Time TTT)
While most AI is "frozen" after training, Loom achieves **Continuous Plasticity**. 
*   **Behavior:** Because forward and backward steps happen every clock cycle, the network is in a state of **Continuous Test-Time Training (TTT)**. 
*   **Impact:** This is why the network can adapt to new patterns in real-time without losing its foundation—it is training in the "imaginary space" between every execution step.

---

## Summary Table

| Layer Type | STC Role | Real-Time Impact |
| :--- | :--- | :--- |
| **Dense** | Global Resolver | Rapid mapping of input to output. |
| **Conv2D** | Spatial Stabilizer | Captures 3D feature motion; smooths convergence. |
| **LSTM** | Temporal Damper | Prevents over-reaction to noise; preserves history. |
| **MHA** | Dynamic Router | Focuses training purely on the source of the error. |
| **Norm** | Energy Regulator | Prevents gradient explosion; standardizes the Gap. |
| **Residual** | Signal Bypass | Keeps the chain flowing even when layers stall. |

---

> [!TIP]
> **O_O Observation:** If you see your "Link Budgets" jumping around during a Step Tween, it usually means your **Norm** layers are working overtime or your **Residual** connections are saving the network from a bottleneck!
