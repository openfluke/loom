# Neural Tween Performance Analysis

## Overview

This document analyzes the performance of Neural Tweening across all tests, particularly focusing on the WDM (Wavelength-Division Multiplexing) multi-signal training approach.

## Key Findings

### 1. The "39.9% Problem" - Models Stuck at Local Minima

In Test 12, most layer types show both Single and WDM scoring ~39.9%. This indicates:
- Models are predicting a single class (majority class baseline)
- The 4-class dataset (sum-based quadrants) has ~40% of samples in one class
- **Neither training method escapes the local minimum in 30 epochs**

This is not a WDM failure - it's insufficient training time for the data complexity.

### 2. WDM Prevents Gradient Explosion in Attention

The **only clear WDM win** (+13.5%) is on MultiHeadAttention:

```
Single-Signal: Score 26.4% (hit NaN during training)
WDM:           Score 39.9% (stable training)
```

**Why?** Attention layers have multiplicative operations (Q·K·V). Single samples can cause extreme activations. WDM's signal averaging acts as implicit regularization, preventing explosion.

### 3. Tween vs Backprop Overall Results

| Test | Tween Wins | BP Wins | Ties |
|------|------------|---------|------|
| Test 6 (1-layer) | 4 (LSTM, LayerNorm, SwiGLU, Attention) | 2 (Dense, RNN) | 2 |
| Test 7 (5-layer) | 4 (Dense, LayerNorm, RMSNorm, LayerNorm) | 1 (Attention-NaN) | 3 |
| Test 12 WDM | 1 (Attention) | 0 | 5 |

**Pattern: Tween excels at normalization layers and recurrent structures, but struggles with basic Dense layers.**

### 4. Why WDM Doesn't Show Dramatic Improvement

1. **Signal Dilution**: Averaging forward activations from different classes creates a "blurred centroid" that doesn't represent any single class well
   
2. **Target Conflict**: Different samples want different outputs - averaging backward targets creates conflicting weight update signals

3. **Link Budget Already High**: WDM shows better depth barriers (0.85+ vs 0.35) but this doesn't translate to better classification because the issue is *what* travels, not *how much*

### 5. When WDM Helps

WDM is beneficial when:
- **Layers are unstable** (Attention, deep networks hitting NaN)
- **Noise reduction needed** (averaging smooths stochastic weight updates)
- **Speed matters** (WDM batching is 20-40% faster due to fewer weight updates per sample)

WDM doesn't help when:
- Models are stuck at local minima (more epochs needed, not more signals)
- Simple layer types (Dense) that don't benefit from averaging

## Depth Charge Analysis (Test 10)

The 20-layer depth test reveals a key issue:

```
Tween: 85.7% (with pruning, 5 layers remaining)
Backprop: 93.4% (all 20 layers, but 4.6x slower)
```

**Insight**: Tween's pruning successfully identifies and removes dead layers, but the aggressive surgery reduces capacity. When Backprop works (doesn't vanish), it outperforms pruned Tween.

## Recommendations

### For WDM to be Effective:

1. **Use for Attention-heavy architectures** - Primary benefit is stability
2. **Increase epochs** - 30 epochs insufficient for complex data
3. **Try class-stratified batching** - Group samples by class in WDM batches to create coherent signals
4. **Combine with higher learning rates** - Stability allows more aggressive updates

### For Neural Tween Generally:

1. **Best for**: LSTM, LayerNorm, normalization layers, moderate depth
2. **Caution with**: Deep Dense stacks (link budget collapses at layer 0)
3. **Enable pruning** for very deep networks to remove dead layers
4. **Consider hybrid approach**: Tween for initial epochs, then switch to Backprop for fine-tuning

## Future Work

1. **Class-aware WDM**: Batch samples by class to prevent target dilution
2. **Adaptive channel count**: Start with more channels, reduce as training progresses
3. **Residual attention**: Add skip connections to prevent attention NaN
4. **Frontier expansion**: Combine WDM with adaptive frontier to focus on healthy layers

## RMT (Resonant Multi-Pass Training) Analysis

After WDM failed, we implemented RMT - a different multi-signal approach where the **same sample** is processed multiple times with perturbations (noise, scaling, masking), all targeting the **same class**.

### The Theory (Why We Expected It To Work)

| Pass | Input Modification | Target |
|------|-------------------|--------|
| 1 | Original | Same class |
| 2 | +Gaussian noise (5%) | Same class |
| 3 | ×Random scale (0.9-1.1) | Same class |
| 4 | Random mask (10% zeros) | Same class |

**Expected benefit**: Data augmentation during training, creating regularization and robustness.

### The Results (Why It Actually Failed)

```
All layer types: 40.7% (same as single-signal)
LinkBudget at L0: Still 0.5 (the bottleneck remains)
```

**RMT didn't fail because of the approach - it failed because it couldn't fix the underlying problem**:

1. **Same L0 bottleneck**: Whether we perturb the input or not, Layer 0 still chokes the signal
2. **Averaging doesn't help dead layers**: If a layer can't pass information, showing it the same data 4 times doesn't help
3. **Output averaging creates noise**: Averaging 4 outputs when 3 are from perturbed inputs dilutes the "true" answer

### Root Cause: The Problem Isn't Input-Side

Both WDM and RMT tried to improve training by **modifying inputs**:
- WDM: Different samples → conflicting targets
- RMT: Same sample with perturbations → same dead layers

**The real issue is target propagation**, not input processing. The backward pass estimation isn't reaching Layer 0 effectively.

---

## HLT (Harmonic Layer Training) Results

We tested layer voting on targets - each layer generates an "opinion" weighted by its LinkBudget.

### The Results: Another Tie 😅

| Layer Type | Single | HLT | Delta |
|------------|--------|-----|-------|
| Dense | 42.0% | 42.0% | +0.0% |
| SwiGLU | 42.0% | 42.0% | +0.0% |
| MHA | 27.8% | 27.8% | +0.0% |
| RNN | 86.4% | 88.0% | +1.6% |
| LSTM | 42.0% | 42.0% | +0.0% |
| Conv2D | 42.0% | 42.0% | +0.0% |

**Average Improvement: +0.27%** - Effectively no change.

### One Bright Spot: RNN

RNN showed actual learning (86.4% → 88.0%)! Both methods succeeded here, suggesting:
- RNN layers work better with Neural Tweening in general
- The 160-epoch breakthrough (from 42% to 74%+) happened for both methods
- HLT didn't help or hurt - RNN just... works

### Why HLT Failed

1. **Voting was too weak**: 40% consensus + 60% hard target meant consensus barely influenced anything
2. **Dead layers can't generate useful votes**: If L0 has 0.5 budget, its "vote" is near-random
3. **Same L0 bottleneck**: Dense still stuck at `min 0.500 @L0` throughout

### Key Observation: Attention Still Explodes

Both Single and HLT hit NaN around epoch 21 for MultiHeadAttention:
```
Epoch 11: Score 68.9% (great!)
Epoch 21: NaN (explosion)
```

**WDM was actually better here** - it prevented the NaN by averaging across samples. Neither RMT nor HLT could stabilize Attention.

---

## Conclusion: The Real Problem

After testing **3 different multi-signal approaches**:

| Approach | Strategy | Result |
|----------|----------|--------|
| WDM | Multiple samples | Ties (but stabilized Attention) |
| RMT | Same sample + perturbations | Ties |
| HLT | Layer voting on targets | Ties |

**The fundamental issue isn't the training signal - it's the architecture.**

The Dense `8→32` adapter at Layer 0 consistently shows:
- LinkBudget: 0.500 (minimum, essentially dead)
- DepthBarrier: ~0.04 (only 4% of signal reaches output)

**What would actually help:**
1. Skip connections (bypass dead L0)
2. Residual blocks (keep signal flowing)
3. Better initialization for the 8→32 expansion
4. More training time (RNN needed 60+ epochs to break through)

The multi-signal experiments were useful for understanding the problem, but the fix needs to be architectural, not algorithmic.
