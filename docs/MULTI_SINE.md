# Multi-Sine Wave Benchmark: Complete Analysis

> **300 Combinations**: 6 Training Modes × 10 Numerical Types × 5 Layer Types  
> **Task**: Real-time sine wave frequency adaptation with continuous inference

## Overview

This benchmark tests Loom's training algorithms on a challenging real-time task: adapting to changing sine wave frequencies while maintaining continuous inference availability. The frequency switches every 2.5 seconds (1x → 2x → 3x → 4x), and models must adapt quickly while still producing predictions.

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║   LAYERS: Dense, Conv2D, RNN, LSTM, Attention (5 types)                                      ║
║   MODES:  NormalBP, StepBP, Tween, TweenChain, StepTween, StepTweenChain (6 modes)          ║
║   TYPES:  int8-int64, uint8-uint64, float32, float64 (10 types)                             ║
║   TOTAL:  6 × 10 × 5 = 300 combinations                                                      ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 🏆 Key Findings

### Absolute Best Configuration

| Metric | Value |
|--------|-------|
| **Mode** | StepTweenChain |
| **Layer** | Conv2D |
| **Type** | float32 |
| **Score** | 1187 |
| **Accuracy** | 98.7% |
| **Throughput** | 120,255/sec |
| **Availability** | 100.0% |

### Why Conv2D Wins

Conv2D with its 2x2 kernel acts as a **pattern detector** across adjacent time steps in the sliding window input. For sine wave prediction, this captures local curve shapes (slopes, inflection points) more effectively than Dense layers treating all inputs equally.

### Why StepTweenChain Wins

StepTweenChain provides:
- **100% availability**: Never blocks for batch training
- **Immediate learning**: Trains on every sample
- **Chain rule accuracy**: Proper gradient propagation

---

## 💡 The Core Thesis

> **Backprop optimizes correctness. StepTweenChain optimizes existence.**  
> **Real-time systems need both — and the ability to move between them.**

### Learning is Not a Single Problem

Most AI discourse assumes:
- Offline training
- Frozen deployment
- No hard latency constraints
- No requirement for perpetual availability

**This benchmark breaks that assumption.**

What these results empirically demonstrate:
- No single method dominates across **availability, latency, throughput, accuracy, and recovery**
- The *best system* is not a better optimizer — it's a **policy over optimizers**

**A learning system without a spectrum is forced to solve every phase of learning with the wrong tool.**

---

## 🔧 The Role of Each Training Mode

These aren't "variants" of backprop. They're **distinct roles** that a real-time learning system needs:

### 🔹 NormalBP (Backpropagation)

| Aspect | Details |
|--------|---------|
| **Role** | Convergence + Correctness |
| **Strength** | Once the signal stabilizes, it locks onto the right basin |
| **Weakness** | Blocks inference, stalls real-time systems |
| **Availability** | ~30-50% (blocked during batch training) |
| **Use When** | You *can afford* to pause the world |

**Best for**: Offline training, batch processing, when accuracy is the only metric.

```
⚠️ NormalBP achieves 98%+ accuracy but only 49% availability
   → Half the time, your system can't respond to inputs
```

---

### 🔹 StepBP (Step-Based Backprop)

| Aspect | Details |
|--------|---------|
| **Role** | Controlled Progression |
| **Strength** | Keeps training moving without catastrophic blocking |
| **Weakness** | Still assumes gradient reliability |
| **Availability** | 100% |
| **Use When** | Signal exists but timing matters |

**Best for**: When you need gradients but can't afford full batch pauses.

---

### 🔹 Tween (Neural Tweening)

| Aspect | Details |
|--------|---------|
| **Role** | *Existence under zero signal* |
| **Strength** | Best-effort motion when gradients are meaningless |
| **Weakness** | Not precise, not stable long-term |
| **Availability** | ~60% |
| **Use When** | **Everything is 0% accuracy but the system must stay alive** |

**This is the key one people underestimate.**

Tween keeps the network moving toward targets even when:
- Gradients are vanishing
- Loss landscape is flat
- Traditional training would stall completely

```
💡 Tween is NOT about being accurate.
   Tween is about NOT DYING when accuracy is impossible.
```

---

### 🔹 TweenChain (Tween with Chain Rule)

| Aspect | Details |
|--------|---------|
| **Role** | Temporal Depth Alignment |
| **Strength** | Handles deeper networks by interpolating across time, not layers |
| **Weakness** | Can still block during batch processing |
| **Availability** | ~60% |
| **Use When** | Depth + time both matter |

**Best for**: Deep networks where layer-by-layer tweening loses coherence.

---

### 🔹 StepTween (Step + Tween)

| Aspect | Details |
|--------|---------|
| **Role** | Availability-First Learning |
| **Strength** | Zero blocking, massive throughput, immediate response |
| **Weakness** | Slight accuracy tradeoff in some cases |
| **Availability** | 100% |
| **Use When** | **The system must always respond** |

**Best for**: Real-time inference systems that need continuous learning.

---

### 🔹 StepTweenChain (Step + Tween + Chain Rule)

| Aspect | Details |
|--------|---------|
| **Role** | **Unified Real-Time Learning Regime** |
| **Strength** | ~100% availability, near-BP accuracy, highest throughput, zero blocking |
| **Weakness** | Requires more complex state management |
| **Availability** | 100% |
| **Use When** | The system *lives in the world* |

**This is why it wins overall.**

```
StepTweenChain = The mode for systems that cannot afford to stop.

Robotics, real-time control, live adaptation, always-on AI.
```

---

## 🎯 When to Use Each Mode

### Decision Matrix

| Scenario | Recommended Mode | Why |
|----------|------------------|-----|
| **Offline batch training** | NormalBP | Maximum accuracy, time doesn't matter |
| **Fine-tuning with deadline** | StepBP | Gradient-based but non-blocking |
| **Catastrophic signal loss** | Tween | Keeps weights moving when gradients fail |
| **Deep network real-time** | TweenChain | Handles depth with temporal coherence |
| **Always-on inference + learning** | StepTween | Never blocks, always responds |
| **Production real-time AI** | **StepTweenChain** | Best overall for living systems |

### The Spectrum in Action

```
Situation: Real-time robot arm tracking a moving target

1. Target acquired, stable signal → StepTweenChain (learn + respond)
2. Target occluded, signal lost → Tween (survive, don't stall)
3. Target reacquired → StepTweenChain (resume learning)
4. Periodic maintenance window → NormalBP (deep refinement)

A single learning rule cannot handle this.
The robot needs a POLICY over learning modes.
```

---

## 🔬 Why Conv2D + StepTweenChain Wins

> *"Conv2D probably wins because it best captures frames of time and can better tween between them"*

Yes — and more precisely:

| Component | Role |
|-----------|------|
| **Conv2D** | Encodes **spatial continuity** — pattern detection across time windows |
| **StepTweenChain** | Encodes **temporal continuity** — smooth interpolation through time |
| **float32** | Hits the **numerical sweet spot** for SIMD, cache efficiency, noise tolerance |

### What You've Actually Built: A Spatiotemporal Interpolator

The combination of Conv2D + StepTweenChain isn't just "good" — it's a **spatiotemporal interpolator with adaptive correction**:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        SPATIOTEMPORAL INTERPOLATOR                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│   INPUT: [sin(t-15), sin(t-14), ..., sin(t)]  ← 16-sample sliding window         │
│                           │                                                        │
│                           ▼                                                        │
│   ┌─────────────────────────────────────────┐                                     │
│   │         CONV2D (4×4 kernel)             │  ← SPATIAL: detects local patterns │
│   │   • Captures curve shapes               │     (slopes, peaks, inflections)   │
│   │   • Encodes position-relative features  │                                     │
│   │   • Translation-invariant detection     │                                     │
│   └─────────────────────────────────────────┘                                     │
│                           │                                                        │
│                           ▼                                                        │
│   ┌─────────────────────────────────────────┐                                     │
│   │       STEPTWEENCHAIN TRAINING           │  ← TEMPORAL: smooth weight updates │
│   │   • Interpolates toward targets         │     (no discontinuities)           │
│   │   • Chain rule preserves gradients      │                                     │
│   │   • Never blocks (100% availability)    │                                     │
│   └─────────────────────────────────────────┘                                     │
│                           │                                                        │
│                           ▼                                                        │
│   OUTPUT: sin(t+1)  ← Prediction with adaptive correction                        │
│                                                                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Breaking Down the Math

| Component | Mathematical Role | Why It Matters |
|-----------|-------------------|----------------|
| **Conv2D spatial** | `f(x) = Σ w[i,j] · x[i,j]` | Detects **where** patterns occur in the window |
| **StepTween temporal** | `w[t+1] = w[t] + α(target - current)` | Smoothly **interpolates** weights toward targets |
| **Chain rule correction** | `∂L/∂w = ∂L/∂y · ∂y/∂w` | **Corrects** interpolation with gradient information |
| **float32 precision** | 32-bit IEEE 754 | Balances accuracy vs speed for SIMD ops |

### Why This Emerges Naturally

The architecture isn't accidental — it's **inevitable** given the constraints:

1. **Real-time demands availability** → Must use step-based (non-blocking)
2. **Continuous signal demands smoothness** → Must interpolate, not jump
3. **Prediction demands pattern detection** → Must use spatial kernels (Conv)
4. **Accuracy demands correction** → Must include chain rule gradients

**Result**: A neural network that acts like a **Kalman filter meets video codec**:
- Predicts the next frame (spatial patterns)
- Smoothly updates its model (temporal interpolation)
- Corrects errors when they occur (adaptive feedback)
- Never stalls while doing so (real-time capable)

---

## 🚫 What This Benchmark Does NOT Claim

This is important to be clear about:

| ❌ NOT Claiming | ✅ Actually Claiming |
|-----------------|---------------------|
| "StepTweenChain beats backprop" | StepTweenChain wins on **real-time metrics** |
| "Backprop is wrong" | Backprop is insufficient **alone** for real-time |
| "One mode is universally best" | Different modes serve different **operating conditions** |
| "This replaces gradient descent" | This **complements** gradient descent with alternatives |

The thesis is:

> **Static training ≠ Real-time learning**  
> **Batch convergence ≠ Continuous adaptation**  
> **Accuracy ≠ Availability**

You didn't "beat backprop". You showed **why backprop alone is insufficient** for systems that must exist in the world.

---

## Summary by Layer Type

### Accuracy Across All Types (%)

| Layer | Mode | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float32 | float64 | Avg |
|-------|------|------|-------|-------|-------|-------|--------|--------|--------|---------|---------|-----|
| **Dense** | NormalBP | 13 | 20 | 20 | 21 | 13 | 23 | 21 | 13 | **99** | **99** | 34 |
| **Conv2D** | StepTweenChain | 13 | 20 | 22 | 22 | 13 | 21 | 21 | 22 | **99** | 21 | 27 |
| **RNN** | StepTween | 13 | 21 | 23 | 22 | 13 | 23 | 21 | 23 | **77** | 21 | 26 |
| **LSTM** | NormalBP | 13 | 20 | 21 | 21 | 13 | 23 | 21 | 13 | 54 | **59** | 26 |
| **Attention** | StepTween | 13 | 20 | 21 | 21 | 13 | 22 | 21 | 21 | **90** | 21 | 26 |

> **Note**: Integer types (int8-uint64) perform poorly (~13-23%) because sine wave values scaled to integers lose precision. Float32 dominates for accuracy.

---

## Best Performers Per Layer

| Layer | Best Mode | Best Type | Score | Accuracy | Throughput | Availability |
|-------|-----------|-----------|-------|----------|------------|--------------|
| Dense | StepTween | float32 | 379 | 42.5% | 89,116/s | 100.0% |
| **Conv2D** | **StepTweenChain** | **float32** | **1187** | **98.7%** | **120,255/s** | **100.0%** |
| RNN | StepTween | float32 | 663 | 76.5% | 86,624/s | 100.0% |
| LSTM | NormalBP | float64 | 49 | 58.5% | 5,098/s | 28.7% |
| Attention | StepTween | float32 | 830 | 90.1% | 92,099/s | 100.0% |

---

## Score Matrix: Mode × Layer (float32)

```
┌──────────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬───────────────────┐
│ Mode             │ Dense      │ Conv2D     │ RNN        │ LSTM       │ Attention  │ BEST LAYER        │
├──────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼───────────────────┤
│ NormalBP         │        113 │        323 │         65 │          8 │        106 │ ★ Conv2D (323)    │
│ StepBP           │        132 │        611 │        139 │          6 │        156 │ ★ Conv2D (611)    │
│ Tween            │        106 │        398 │        186 │          7 │        261 │ ★ Conv2D (398)    │
│ TweenChain       │        302 │        310 │        169 │          6 │        404 │ ★ Attention (404) │
│ StepTween        │        379 │       1012 │        663 │         20 │        830 │ ★ Conv2D (1012)   │
│ StepTweenChain   │        299 │       1187 │        660 │         20 │        754 │ ★ Conv2D (1187)   │
└──────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴───────────────────┘
```

---

## Availability Analysis

### NormalBP vs StepTweenChain Comparison

| Layer | NormalBP Score | StepTweenChain | Winner | Accuracy Δ | Avail Δ | Throughput Δ |
|-------|----------------|----------------|--------|------------|---------|--------------|
| Dense | 113 | 299 | **StepTweenChain** | -64.4% | +63.4% | +55,983 |
| Conv2D | 323 | 1187 | **StepTweenChain** | +1.1% | +51.2% | +52,326 |
| RNN | 65 | 660 | **StepTweenChain** | -19.4% | +71.8% | +61,518 |
| LSTM | 8 | 20 | **StepTweenChain** | -39.3% | +71.3% | +8,852 |
| Attention | 106 | 754 | **StepTweenChain** | -16.3% | +62.2% | +63,386 |

> **Key Insight**: NormalBP achieves high accuracy BUT blocks inference during batch training (~30-50% availability). StepTweenChain maintains ~100% availability while still training every sample!

---

## Detailed Timeline: Conv2D + float32

### Accuracy Per Second

```
┌──────────────────┬─────────────────────────────────────────────────────┬───────┬──────────┐
│ Mode             │ 1s  2s  3s  4s  5s  6s  7s  8s  9s  10s             │ Avg   │ Score    │
├──────────────────┼─────────────────────────────────────────────────────┼───────┼──────────┤
│ NormalBP         │ 91% 100% 92% 100% 100% 97% 100% 96% 100% 100%       │  98%  │      323 │
│ StepBP           │ 88% 100% 95% 100% 100% 98% 100% 98% 100% 100%       │  98%  │      611 │
│ Tween            │ 89% 100% 92% 100% 100% 96% 100% 97% 100% 100%       │  97%  │      398 │
│ TweenChain       │ 50%  95% 77%  76% 100% 37%  78% 69%  74%  78%       │  73%  │      310 │
│ StepTween        │ 93% 100% 96% 100% 100% 99% 100% 99% 100% 100%       │  99%  │     1012 │
│ StepTweenChain   │ 94% 100% 96% 100% 100% 98% 100% 99% 100% 100%       │  99%  │     1187 │
└──────────────────┴─────────────────────────────────────────────────────┴───────┴──────────┘
```

### Outputs Per Second (Throughput)

```
┌──────────────────┬───────────────────────────────────────────────────────────────┬────────┬──────────┐
│ Mode             │  1s     2s     3s     4s     5s     6s     7s     8s     9s    10s   │ Total  │ Avail%   │
├──────────────────┼───────────────────────────────────────────────────────────────┼────────┼──────────┤
│ NormalBP         │ 70257  69044  68556  68313  62441  70429  68377  69656  65024  67193 │ 679290 │   48.8%  │
│ StepBP           │ 64231  61955  62613  62883  63046  60503  61451  62792  63186  61675 │ 624335 │  100.0%  │
│ Tween            │ 81588  73045  78791  73609  76155  72884  60999  58741  53446  62030 │ 691288 │   59.1%  │
│ TweenChain       │ 68728  68956  67985  71506  70778  67773  67374  67675  69772  69502 │ 690049 │   61.3%  │
│ StepTween        │ 99911 103089 101711 101449 105031 104181 101647 101959 106168 100444 │1025590 │  100.0%  │
│ StepTweenChain   │120280 121466 114197 120122 121843 121253 120902 120297 121633 120561 │1202554 │  100.0%  │
└──────────────────┴───────────────────────────────────────────────────────────────┴────────┴──────────┘
```

---

## Best Numerical Type Per Layer+Mode

```
┌────────────┬────────────────┬────────────────┬────────────────┬────────────────┬────────────────┬────────────────┐
│ Layer      │ NormalBP       │ StepBP         │ Tween          │ TweenChain     │ StepTween      │ StepTweenChain │
├────────────┼────────────────┼────────────────┼────────────────┼────────────────┼────────────────┼────────────────┤
│ Dense      │ float64   224  │ uint16    141  │ float32   106  │ float32   302  │ float32   379  │ float32   299  │
│ Conv2D     │ float64   673  │ float32   611  │ float32   398  │ float32   310  │ float32  1012  │ float32  1187  │
│ RNN        │ float64   163  │ uint16    151  │ float32   186  │ float32   169  │ float32   663  │ float32   660  │
│ LSTM       │ float64    49  │ uint8      37  │ int16      27  │ int16      26  │ int16      28  │ int16      20  │
│ Attention  │ float64   208  │ float32   156  │ float32   261  │ float32   404  │ float32   830  │ float32   754  │
└────────────┴────────────────┴────────────────┴────────────────┴────────────────┴────────────────┴────────────────┘
```

### Type Wins Summary

| Type | Wins | Visual |
|------|------|--------|
| float32 | 18 | ██████████████████ |
| float64 | 5 | █████ |
| int16 | 4 | ████ |
| uint16 | 2 | ██ |
| uint8 | 1 | █ |

---

## Understanding the Score Formula

```
Score = (Throughput × Availability% × Accuracy%) / 10000
```

This formula rewards configurations that:
1. **High Throughput**: Produce many predictions per second
2. **High Availability**: Don't block during training
3. **High Accuracy**: Make correct predictions

### Why StepTweenChain Dominates

| Factor | NormalBP | StepTweenChain | Impact |
|--------|----------|----------------|--------|
| Accuracy | ~98% | ~99% | Similar |
| Throughput | ~68k/s | ~120k/s | **+77%** |
| Availability | ~49% | 100% | **+104%** |
| **Score** | 323 | **1187** | **+267%** |

Even with similar accuracy, StepTweenChain's massive throughput and availability advantages result in a **3.7x higher score**.

---

## Conclusions

### For Real-Time Applications
Use **Conv2D + StepTweenChain + float32** for:
- Highest overall score (1187)
- 100% inference availability
- Near-perfect accuracy (98.7%)
- Maximum throughput (120k/s)

### For Maximum Accuracy (no real-time constraint)
Use **Conv2D + NormalBP + float64** for:
- Best accuracy potential
- Acceptable for batch processing
- Higher precision with float64

### LSTM Underperforms
LSTM struggles on this task because:
- The sliding window input already captures temporal context
- LSTM's recurrence adds overhead without benefit
- Simpler Conv2D pattern matching works better

---

## Running the Benchmark

```bash
cd tva/examples
go run all_sine_wave_multi.go
```

Results are saved to `all_sine_wave_multi_results.json`.
