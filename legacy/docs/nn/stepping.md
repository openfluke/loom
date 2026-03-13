# The Stepping Mechanism
**Understanding "Neural Traffic Control" in Loom**

Loom introduces a novel execution model called **Stepping**, which differs fundamentally from the traditional "Forward Pass" used in PyTorch or TensorFlow.

The best way to understand this is by analogy: **Traffic Lights vs. Roundabouts**.

---

## 1. The Analogy

### 🚦 Traditional: The Traffic Light (Blocking)
In a standard neural network `Forward()` pass:
*   Layer 1 processes a batch of data. **Layer 2 waits.**
*   Layer 1 finishes. The light turns green.
*   Layer 2 processes. **Layer 3 waits.**
*   The signal propagates sequentially. The "latency" is the sum of all waiting times.

This is simple, but it creates **"Pipeline Bubbles"**. While Layer 1 is working on Frame 2, Layer 10 is sitting idle waiting for Frame 2 to reach it.

```text
Time ➜
[Input]  ╾— Frame 1 —╼ ╾— Frame 2 —╼ (Driver waits)
            ⬇
[Layer1]    █ Process █ ╾— Wait... —╼ █ Process █
                        ⬇
[Layer2]    ╾— Wait... —╼ █ Process █ ╾— Wait... —╼
                                    ⬇
[Layer3]    ╾— Wait... —╼ ╾— Wait... —╼ █ Process █
```

### 🔄 Loom Stepping: The Roundabout (Non-Blocking)
In Loom's `StepForward()`:
*   **All layers active simultaneously.**
*   Layer 1 processes Frame `T` (Input).
*   Layer 2 processes Frame `T-1` (Layer 1's previous output).
*   Layer 3 processes Frame `T-2`.

It is a **continuous flow**. Just like a roundabout allows cars to enter and exit without stopping, Loom's layers process whatever data is sitting in their input buffer *right now*.

```text
Time ➜
[Input]  ╾— Frame 3 —╼ ╾— Frame 4 —╼ ╾— Frame 5 —╼
            ⬇             ⬇             ⬇
[Layer1] █ Proc F2 █   █ Proc F3 █   █ Proc F4 █
            ⬇             ⬇             ⬇
[Layer2] █ Proc F1 █   █ Proc F2 █   █ Proc F3 █
            ⬇             ⬇             ⬇
[Layer3] █ Proc F0 █   █ Proc F1 █   █ Proc F2 █
```

---

## 2. How it Works: Double Buffering

The magic is in the `StepState` struct (found in `nn/step_forward.go`). It uses a technique from game engine graphics called **Double Buffering**.

```go
// Simplified Logic
func (n *Network) StepForward(state *StepState) {
    // 1. Snapshot: Create a "Next Frame" buffer
    newOutputs := make([][]float32, totalLayers)

    // 2. Parallel Execution: Every layer runs at the same time
    //    They read from 'current' state, write to 'newOutputs'
    for layer := range allLayers {
        input := state.layerData[layer]  // Read T
        output := Process(input)
        newOutputs[layer+1] = output     // Write T+1
    }

    // 3. Atomically Swap: Update the world state instantly
    state.layerData = newOutputs
}
```

### The Result: High-Frequency Inference
Because no layer waits for another, the **System Latency** is determined only by the *slowest single layer*, not the sum of all layers.

*   **Traditional Latency**: `Time(L1) + Time(L2) + Time(L3) = 30ms`
*   **Stepping Latency**: `Max(Time(L1), Time(L2), Time(L3)) = 10ms`

You get **3x higher throughput** for streaming data (like audio or video), at the cost of a slight "phase delay" (the output represents input from 3 steps ago).

---

## 3. StepBackward: The Time Machine

Refining the weights (`StepBackward.go`) works on the same principle but in reverse.

*   To learn from an error at the Output Layer, we need to know what the Input Layer was doing *steps ago*.
*   Loom maintains a **Time Horizon** (history buffer) to align these events.
*   When a "Correction Signal" (gradient) arrives at Layer 5, it meets the "Activation Memory" of what Layer 5 did when it produced that error.

This allows **Real-Time Reinforcement Learning** without pausing the world. The drone can be flying (Forward Stepping) and learning (Backward Stepping) in the same clock cycle.

---

## 4. Summary

| Feature | Traditional Forward() | Loom StepForward() |
| :--- | :--- | :--- |
| **Flow Control** | Blocking (Stop & Go) | Continuous (Flow) |
| **Component** | Traffic Light 🚦 | Roundabout 🔄 |
| **Latency** | Sum of all depths | Max of single depth |
| **Use Case** | Single Image Classification | Real-Time Video/Audio Stream |

This mechanism is why Loom can run **180 Networks** simultaneously on a customized "Mega Grid" without locking up the CPU.
