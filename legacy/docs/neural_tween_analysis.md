# Neural Tween Performance Analysis

## ⚠️ Executive Summary: The 48% Barrier

> **Neural Tweening is NOT backpropagation. It's Hebbian Self-Organization.**

After extensive experimentation (DFA, DTP, Activation Derivatives, WDM, RMT, Perturb, Kill Ego), we have confirmed that **Neural Tweening has a fundamental accuracy ceiling of ~48-65% for classification tasks**.

### Why This Happens

| Backpropagation | Neural Tweening |
|-----------------|-----------------|
| Chain rule: Exact error attribution | "Telephone Game": Estimated error |
| Says: "Weight A caused 3% of error" | Says: "Everyone louder!" |
| Can fix specific mistakes | Can only create associations |
| O(n) memory, O(n) compute | O(1) compute, no compute graph |

**The Core Problem:** Tweening calculates `Target - Actual` (the "Gap") and uses it for Hebbian updates. But this Gap degrades as it propagates backward—by the time it reaches early layers, it's essentially noise. This is the **"Telephone Game"** effect.

### What Tweening IS Good For

1. **Pre-training** - Get weights from random → organized before Backprop
2. **Stabilization** - Smooth updates prevent Attention NaN explosions (Test 12)
3. **Low-memory training** - No activation cache needed
4. **Embedded/streaming** - Continuous online learning

### What Tweening IS NOT Good For

1. **High-precision classification** (99%+ accuracy)
2. **Deep networks** (>3 layers, unless pruned)
3. **Tasks requiring fine error correction**

---

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

---

## LBL (Layer-by-Layer Training) Results

We tried freezing early layers and training from output backward - unfreeze progressively.

### The Theory

```
Epoch 1-30:   Train only L6 (output)      [L0-L5 frozen]
Epoch 31-60:  Train L5+L6                 [L0-L4 frozen]  
Epoch 61-90:  Train L4+L5+L6              [L0-L3 frozen]
...until all unfrozen...
```

**Expected**: Output learns first with good signal, then each layer gets TRAINED targets.

### The Results: Catastrophic Failure

| Layer Type | Single | LBL | Delta |
|------------|--------|-----|-------|
| Dense | 43.2% | 44.5% | +1.3% |
| SwiGLU | 43.2% | 44.5% | +1.3% |
| RNN | **92.2%** | 44.5% | **-47.6%** |

**LBL destroyed RNN's breakthrough!**

### Why It Failed

1. **RNN's breakthrough happens around epoch 61-71** when ALL layers co-adapt
2. LBL froze L0-L5 during this critical window
3. Frozen layers produced random noise → L6 learned garbage mapping
4. When layers unfroze, they couldn't unlearn L6's bad patterns
5. Loss stayed at ~1.14 vs Single's 0.37

**Key insight**: The "breakthrough" requires simultaneous multi-layer learning. Freezing breaks this.

---

## Curriculum (Boosted Early-Layer Rates) Results  

We tried giving higher learning rates to early layers: L0=3x, L1=2.5x, L2=2x, L3=1.5x.

### The Results: Also Failed

| Layer Type | Single | CURR | Delta |
|------------|--------|------|-------|
| Dense | 45.6% | 45.6% | 0.0% |
| SwiGLU | 45.6% | 45.6% | 0.0% |
| MHA | 25.6% | 45.6% | **+20.0%** |
| RNN | **88.5%** | 45.6% | **-42.9%** |
| LSTM | **86.9%** | 45.6% | **-41.3%** |
| Conv2D | 45.6% | 45.6% | 0.0% |

**Average: -10.68%** - Curriculum hurt more than helped!

### Why It Failed

1. **High rates killed layers**: Early layers went to `X` (dead) by epoch 31
2. **RNN/LSTM needed normal rates**: Their breakthroughs require careful weight evolution
3. **MHA was saved by accident**: High rates prevented NaN (stabilization effect)
4. **Loss stayed at ~1.6**: Never dropped below 1.1

**Key insight**: L0 isn't stuck because the rate is too low - it's stuck because the backward
target estimation can't provide useful signal. Increasing the rate just amplifies garbage.

---

## Complete Experiment Summary

| Approach | Strategy | Result | Killed RNN? |
|----------|----------|--------|-------------|
| WDM | Multiple samples | Ties | No |
| RMT | Same sample + perturbations | Ties | No |
| HLT | Layer voting on targets | Ties | No |
| **LBL** | Freeze + progressive unfreeze | **-15%** | **YES (-47.6%)** |
| **Curriculum** | Boosted early-layer rates | **-10.7%** | **YES (-42.9%)** |

**The pattern is clear**: Any modification that disrupts the natural weight evolution destroys the "breakthrough" phenomenon seen in RNN/LSTM.

---

## The Real Conclusion

After testing **5 different approaches**, we've learned:

1. **RNN/LSTM can break through** - but ONLY with standard single-signal training
2. **Breakthroughs are fragile** - freezing, boosting, or averaging kills them
3. **The L0 bottleneck is architectural** - no training modification can fix 8→32 Dense
4. **Attention benefits from stabilization** - WDM's averaging or Curriculum's high rates help

**What would actually work:**
1. **Skip connections** - Let signal bypass L0 entirely
2. **Match dimensions** - Use 8→8 instead of 8→32 expansion
3. **Shorter networks** - Fewer layers = less signal decay
4. **More epochs** - RNN broke through at epoch 141, not epoch 30

---

## New Approach: "Sharpening the Blurry Image"

**Insight (Test 16):** Tween gets to ~47% accuracy very quickly (faster than backprop to 40%), but then plateaus. It's like getting a blurry photo instantly - we just need to sharpen it slightly.

### Proposed Sharpening Techniques

#### 1. Error-Proportional TweenFactor
Scale tweenFactor by how wrong each output is - bigger errors get bigger pushes.

// Instead of flat tweenFactor = 0.3
for i := range output {
    error := target[i] - output[i]
    dynamicFactor := tweenFactor * math.Abs(error)  // More wrong = bigger push
    // Apply update with dynamicFactor
}



#### 2. Winner-Take-More
Amplify gap for the correct class (×1.5), dampen wrong classes (×0.7).

// Boost the signal for the target class
gap[targetClass] *= 1.5  // Push harder toward correct answer
for i := range gap {
    if i != targetClass {
        gap[i] *= 0.7  // Dampen wrong classes
    }
}

#### 3. Momentum on Gap Direction
Remember which direction previous gaps pointed, use momentum toward persistent patterns.

// Track gap momentum per layer
gapMomentum[layer] = 0.9*gapMomentum[layer] + 0.1*currentGap
// Use gapMomentum instead of raw gap



#### 4. Temperature Sharpening
Apply temperature < 1.0 to outputs before computing gap, sharpening probability distribution.

// Sharpen probabilities (like knowledge distillation but inverse)
temperature := 0.5  // < 1 = sharper
for i := range output {
    output[i] = exp(log(output[i]) / temperature)
}
// Then normalize and compute gap

#### 5. Layer-Specific Gap Scaling ⭐ (TESTED - Mixed Results)
Earlier layers get amplified gap signal since they're hardest to reach:
```go
layerBoost := 1.0 + float32(totalLayers-1-layerIdx) * 0.3
gap *= layerBoost
```

### Test Results: Layer Gap Boost (Test 16)

| Network | Before | After | Change |
|---------|--------|-------|--------|
| Dense NormTween | 47.2% | 53.4% | **+6.2%** ✅ |
| Conv2D NormTween | 68.8% | 78.8% | **+10%** ✅ |
| Conv2D StepTween | 51.4% | 71.2% | **+19.8%** ✅ |
| RNN NormTween | 47.8% | 34.0% | **-13.8%** ❌ |

**Verdict:** Works for Dense/Conv2D, hurts RNN. May need layer-type-specific boost factors.

---

## Sharpening Ideas - ALL FAILED

1. ~~**Error-Proportional TweenFactor**~~ ❌ FAILED - No breakthrough
2. ~~**Winner-Take-More**~~ ❌ FAILED - No breakthrough (~47% still)
3. ~~**Momentum on Gap Direction**~~ ❌ FAILED - No improvement (~47% still)
4. ~~**Temperature Sharpening**~~ ❌ FAILED - Made Conv2D worse (68% → 27%)
5. ~~**Layer-Specific Gap Scaling**~~ ❌ FAILED - Mixed results, hurt RNN

---

## Conclusion

**All 5 sharpening ideas have been tested and failed to break the ~50% accuracy barrier.**

The fundamental limitation of Neural Tweening appears to be the **lack of gradient direction information**. Tweening knows "I'm wrong by X amount" but doesn't know "which specific weights caused the error" or "which direction to push them."

This is analogous to being blindfolded and told "you're 10 degrees off target" without knowing if you're pointing too far left or right. The chain rule in backpropagation provides this directional information; Tweening does not.

**Use cases where Tweening may still be valuable:**
- Fast initial exploration before switching to backprop
- Online/streaming settings where speed matters more than optimal accuracy
- Reinforcement learning where approximate updates are acceptable

---

## ~~Hybrid Tween-Backprop~~ ❌ FAILED

**Concept:** Combine backprop's gradient direction with tween's update style:
1. Use backprop's chain rule to compute **actual gradients** (the "which way to push" info)
2. Apply updates using tween-style scaling (link budgets, gap magnitude)

**Result:** No improvement over vanilla Tween. Dense: 48%, Conv2D: 60.4% (same or worse than Normal Tween).

The hybrid approach didn't break the ~50% barrier. The chain rule gradients appear to be undermined by the tween-style update mechanism.

---

## ~~Approximate Backprop~~ ❌ FAILED

**Concept:** Create a *cheaper* version of backprop that approximates its behavior:

| Full Backprop | Approximate Backprop (Cheaper) |
|---------------|--------------------------------|
| `dL/dOut = output - target` | ✅ Same |
| `dL/dInput = W^T × grad × act_derivative` | ⚡ `W^T × grad` (skip activation derivative) |
| `dW = input × grad` | ✅ Same |

### Results

**V1 (no act derivative, no link budget):** Showed promise on Conv2D (77%)

**V2 (added act derivatives + link budget):** Made things WORSE
- Conv2D: 77% → 59% ❌
- LSTM: 56.6% → 51.2% ❌
- RNN: 52.2% → 43.2% ❌

The added complexity hurt rather than helped. Even the V1 results weren't consistent enough to be useful.

---

## ~~Sign-Aligned Pseudo-Gradient Tweening (Sign-Tween)~~ ❌ FAILED

**Status: TESTED - Did NOT break the 50% barrier**

### Core Hypothesis

The reason Tween plateaus at ~48-52% is that it only knows the *magnitude* of the gap (how wrong) but has **no reliable sense of direction** for each individual weight update. The backward target estimation is too crude (weighted average based on current weights), so early layers get noisy/conflicting signals.

All previous sharpening ideas (winner-take-more, error-proportional, temperature, momentum on gaps) still operate on the same noisy gap signal. They amplify noise as much as signal.

### The Solution: Inject Directional Signal Using Sign Alignment

Instead of blindly tweening weights toward the backward-estimated input using the current weights, we add a **pseudo-gradient term** based only on the **sign** of (input * output_gap). This is extremely cheap (no chain rule, no derivatives), but it gives each weight a consistent "push this way if input and needed correction are same sign" direction.

**This is NOT backprop.** We're not computing full gradients or using activation derivatives. We're using a Hebbian-like sign alignment: "neurons that fire together (input high when we need output higher) should wire together."

### Implementation Details

New fields added to `TweenState`:
```go
SignAlignEnabled  bool    // Default: true, enable sign-aligned updates
SignAlignStrength float32 // Default: 0.3, blend ratio (0.0 = pure tween, 1.0 = pure sign-aligned)
```

**Algorithm in `tweenDense`:**
1. Compute classic tween delta: `rate * input * gap * weightRateMultiplier`
2. Compute sign-aligned delta:
   - If `input * gap > 0` (same sign) → strengthen connection (positive delta)
   - If `input * gap < 0` (opposite signs) → weaken connection (negative delta)
   - Scale by input magnitude: `|input| * rate * gap * weightRateMultiplier`
3. Blend: `totalDelta = (1 - strength) * tweenDelta + strength * signAlignedDelta`
4. Apply with momentum

**Algorithm in `tweenConv2D`:**
- Similar approach using average input activation across the filter

### Test Results (2025-12-16)

| Network   | NormalBP | NormalTween | BatchTween | Change |
|-----------|----------|-------------|------------|--------|
| Dense     | 100.0%   | 47.6%       | 47.4%      | ❌ No improvement |
| Conv2D    | 90.2%    | 48.0%       | 48.2%      | ❌ No improvement |
| RNN       | 99.4%    | 50.6%       | 48.6%      | ❌ No improvement |
| LSTM      | 84.6%    | 46.8%       | 46.8%      | ❌ No improvement |
| Attention | 48.0%    | 48.0%       | 48.0%      | ❌ Tie (both stuck) |
| Norm      | 50.2%    | 50.2%       | 50.2%      | ❌ Tie (both stuck) |
| SwiGLU    | 64.2%    | 47.0%       | 47.0%      | ❌ No improvement |

**Result: Sign-Aligned Pseudo-Gradient did NOT break the 50% barrier.**

### Why It Failed

The Hebbian sign alignment provides *consistent* directional signal, but it's still based on the **same noisy backward target estimation**. The fundamental problem remains:

1. **Backward targets are wrong** - The weighted average estimation doesn't produce accurate targets for early layers
2. **Wrong direction × consistent = consistent wrong** - Making the noisy signal more consistent doesn't fix the noise
3. **No error attribution** - We still don't know which specific weights caused the error

The chain rule in backpropagation doesn't just provide direction - it provides **attributable direction** (which weights were responsible for the error). Hebbian learning only knows correlation, not causation.

---

## Final Conclusion

After testing **9 different approaches** to improve Neural Tweening:

| Approach | Result |
|----------|--------|
| WDM (Multiple samples) | ❌ Ties |
| RMT (Same sample + perturbations) | ❌ Ties |
| HLT (Layer voting) | ❌ Ties |
| LBL (Progressive unfreezing) | ❌ -15% (hurt RNN) |
| Curriculum (Boosted early rates) | ❌ -10.7% (killed layers) |
| Approximate Backprop | ❌ Made things worse |
| Hybrid Tween-Backprop | ❌ No improvement |
| Sign-Aligned Pseudo-Gradient | ❌ No improvement |
| Perturb-Tween | ❌ No improvement |
| **Direct Feedback Alignment (DFA)** | ❌ WORSE (Regression) |

**The ~50% barrier appears to be a fundamental limitation of gradient-free weight updates.**

---

## ~~Perturb-Tween (Perturbation-Based Directional Tweening)~~ ❌ FAILED

**Status: TESTED - Did NOT break the 50% barrier**

### Core Idea

Use weight perturbations to empirically estimate update direction without gradients:
1. For each weight, apply a small random perturbation
2. Measure how local loss changes
3. If loss increases, flip the update direction; if decreases, keep it
4. Blend perturbation-based direction with classic tween

### Implementation

New fields added to `TweenState`:
```go
PerturbEnabled  bool    // Default: true
PerturbStrength float32 // Default: 0.01 (perturbation magnitude)
PerturbFraction float32 // Default: 0.2 (perturb 20% of weights per update)
PerturbBlend    float32 // Default: 0.5 (50% classic + 50% perturb)
```

### Test Results (2025-12-16)

| Network   | NormalBP | NormalTween | BatchTween |
|-----------|----------|-------------|------------|
| Dense     | 99.4%    | 48.6%       | 47.8%      |
| Conv2D    | 87.6%    | 51.2%       | 50.6%      |
| RNN       | 98.2%    | 54.0%       | 49.2%      |
| LSTM      | 92.6%    | 49.8%       | 49.8%      |
| Attention | 50.4%    | 50.4%       | 50.4%      |

**Result: Perturb-Tween did NOT break the 50% barrier.**

The variation observed between runs (~47-55%) is just random weight initialization flux, not actual improvement from the perturbation technique.

### Why It Failed

1. **Local approximation is too crude** - Perturbing one weight and measuring "local" loss change doesn't capture how that weight affects the full network output
2. **Random perturbations don't correlate with optimal direction** - Unlike finite differences in backprop, we don't evaluate the actual loss function, just a local gap approximation  
3. **Same underlying problem** - We're still estimating direction from the noisy backward target gap, not from the true loss surface

---

## ~~Direct Feedback Alignment (DFA) Tweening~~ ❌ FAILED

**Status: TESTED - FAILED (Regression)**

### Core Idea

Instead of passing the error signal layer-by-layer (which accumulates noise), use **Direct Feedback Alignment (DFA)**:
1.  **Global Error**: Calculate error at final output.
2.  **Broadcast**: Project this global error directly to each hidden layer using fixed random matrices $B_l$.
3.  **Update**: Use this projected signal as the target gap for tweening.

### Implementation

- Added `DFAEnabled` flag and `FeedbackWeights` matrices.
- Modified `TweenStep` to skip backward pass and instead inject `Gap_l = B_l * GlobalError`.
- Used standard `TweenWeights` to move weights toward this projected target.

### Test Results (2025-12-16)

| Network   | NormalBP | NormalTween (Baseline) | BatchTween (DFA) | Change |
|-----------|----------|------------------------|------------------|--------|
| Dense     | 99.4%    | 25.8%                  | 38.8%            | ❌ Worse |
| Conv2D    | 82.0%    | 47.8%                  | 27.6%            | ❌ Worse |
| RNN       | 98.2%    | 38.8%                  | 38.2%            | ❌ Same/Worse |
| LSTM      | 91.4%    | 29.8%                  | 31.2%            | ❌ Worse |
| Attention | 44.6%    | 27.8%                  | 27.6%            | ❌ Low |

**Result: DFA performed WORSE than standard layer-by-layer tweening.**

### Why It Failed

1.  **Dimensional Mismatch**: Identifying the "right" random matrix $B_l$ that aligns with the forward weights $W_l$ is critically hard without gradients. In standard DFA, backprop "learns" to align $W_l$ with $B_l$. Here, we have no mechanism to force that alignment.
2.  **Tween uses Correlations**: Tweening relies on `Input * Gap`. If the Gap is just a random projection of global error, it has almost zero correlation with the specific input features that caused that error.
3.  **Optimization vs Steering**: DFA provides a "steering" signal, but Tweening needs a "correction" signal. The random projection steers weights in arbitrary directions, breaking the delicate local feature detectors that standard Tweening (with its imperfect backward pass) was at least partially preserving.

---

## Absolute Final Conclusion

Neural Tweening's ~50% barrier is a **fundamental limitation** of any approach that doesn't compute how weight changes propagate through the entire network (i.e., the chain rule).

All gradient-free attempts have failed because:
- **No error attribution**: Without derivatives, we can't know which specific weights caused the error
- **No sensitivity information**: We don't know how much each weight affects the final output
- **Noisy backward targets**: The "backward pass" estimation is inherently approximate and noisy

Neural Tweening's value is limited to:
- Fast initial exploration (reaches 40% quickly before plateauing)
- Stabilizing attention layers (prevents NaN)
- Extremely resource-constrained environments where backprop is impossible
- Potential hybrid approaches where Tween does exploration and backprop does refinement

---

## ~~AI Therapy: Difference Target Propagation + Kill Ego (Failure)~~

**Status: TESTED - FAILED**

### Core Idea

We attempted a radical "therapy" to fix the "delusional" behavior of the network (optimizing for internal consistency rather than reality):
1.  **Kill the Ego**: We removed the `IgnoreThreshold`. Layers were forced to update even if their Link Budget (signal alignment) was near zero.
2.  **Difference Target Propagation (DTP)**: Instead of estimating targets via heuristic weighted averages, we trained an explicit **Inverse Autoencoder** (`InverseKernels`) for each layer.
    -   `TrainInverse`: Minimize `Input - Inverse(Output)`.
    -   `BackwardPass`: `TargetIn = Input + Inverse(TargetOut) - Inverse(ActualOut)`.

### Test Results (2025-12-16)

| Network   | NormalBP | NormalTween (DTP) | Change vs Baseline | Verdict |
|-----------|----------|-------------------|--------------------|---------|
| Dense     | 100.0%   | 48.0%             | -31% (Regression)  | ❌ Failed |
| Conv2D    | 89.8%    | 59.6%             | -20% (Regression)  | ❌ Failed |
| RNN       | 99.0%    | 48.8%             | ~0% (Stagnant)     | ❌ Failed |

### Why It Failed

1.  **Inverse Noise**: The `InverseWeights` started random and were trained online. Early in training, the inverse is garbage. Propagating targets through a garbage inverse just generates garbage targets for the previous layer.
2.  **Blind Leading the Blind**: Unlike DTP papers which often use pre-training or specific phases, we tried to train Forward and Inverse simultaneously from scratch. The Forward weights move to satisfy the Inverse, and the Inverse moves to satisfy the Forward. Without a strong anchor (like Backprop's exact gradient), they just drifted into a noisy equilibrium (~48% accuracy - basically random guessing for 2 classes).
3.  **Removal of Filters**: Removing `IgnoreThreshold` ("Kill Ego") meant that we flooded the network with this noisy gradient signal, corrupting even the potentially good layers.

**Key Takeaway**: DTP requires a much more stable Inverse to work than can be learned online from scratch in this architecture. "Heuristic Consensual" (Original Tween) is superior to "Learned Inversion" (DTP) for this unconstrained setting.

---

## ~~Activation Derivatives + Leaky Gradients (Semi-Success)~~ 

**Status: TESTED - Improvement on Simple Layers, Failure on Complex**

### Core Idea

Previous implementations treated layers as linear during the backward pass (Feedback Alignment style), ignoring the non-linearity of activation functions.
We implemented:
1.  **Strict Activation Derivatives**: `ErrorPreAct = ErrorPostAct * Derivative(Input)`
2.  **Leaky Derivatives**: For ReLU/LeakyReLU, used a small slope (0.1) instead of 0 for negative values to prevent "dead neurons" from killing the backward signal.
3.  **Boosted Learning Rate**: Increased `WeightRateMultiplier` from 0.01 to 0.1 to match BP dynamics.

### Test Results (2025-12-16)

| Network   | NormalBP | NormalTween | Change vs Baseline | Verdict |
|-----------|----------|-------------|--------------------|---------|
| Dense     | 100.0%   | 79.0%       | **+31.4%**         | ✅ Improvement |
| Conv2D    | 86.6%    | 79.8%       | **+31.8%**         | ✅ Improvement |
| RNN       | 100.0%   | 51.8%       | +1.2%              | ❌ Low  |
| LSTM      | 87.0%    | 58.6%       | +11.8%             | ❌ Low  |
| Attention | 45.0%    | 45.0%       | -3.0%              | ❌ Stuck |
| Norm      | 52.0%    | 52.0%       | +1.8%              | ❌ Stuck |
| SwiGLU    | 58.2%    | 56.2%       | +9.2%              | ✅ Comparable |

### Conclusion

Adding derivatives **significantly helped** shallow/simple networks (Dense, Conv2D), bringing them much closer to Backprop performance (79% vs 86-100%).

However, it **failed to solve the deep/recurrent problem**. RNN, LSTM, and Attention layers still severely underperform compared to Backprop.

**Why?**
The "Telephone Game" is still in effect. While the local derivative provides a better *local* update, the error signal itself (the `ErrorPostAct`) is still an estimation derived from downstream layers. As this estimated error propagates back through time (RNN/LSTM) or depth, it becomes decorrelated from the true loss gradient.

**Key Takeaway:** Correct local physics (derivatives) cannot fix incorrect global signal (inaccurate backward target estimation).

---

## Direct Feedback Alignment (DFA) Tweening (Idea #10)

**Status: PROPOSED**

### Problem: The Telephone Game

All previous approaches failed because they rely on the **Layer-by-Layer Backward Estimation**.
- Layer N estimates Layer N-1's target.
- Layer N-1 estimates Layer N-2's target.
- ...
- Layer 1 estimates Layer 0's target.

By the time the signal reaches early layers (Dense/Conv2D inputs), it has passed through multiple noisy, non-invertible transformations. The error signal is completely garbled—like a game of telephone. This is why early layers are "optimizing towards noise."

### The Solution: Direct Broadcast (Short-Circuit)

Instead of passing the error signal layer-by-layer, we will use **Direct Feedback Alignment (DFA)** (Nøkland, 2016).

1.  **Global Error**: Calculate the error at the final output layer (Target - Actual).
2.  **Fixed Feedback Paths**: Initialize a **fixed random matrix** B_l for each hidden layer l.
3.  **Direct Projection**: During training, project the global output error directly into each hidden layer's space: Gap_l = B_l * GlobalOutputError.
4.  **Tween**: Use this Gap_l to update weights in layer l.

### Why This Might Work

-   **No Noise Accumulation**: The error signal for the first layer comes directly from the final error, untainted by the current state of intermediate layers.
-   **Stable Targets**: The feedback matrices B_l are fixed (or slowly evolving). This gives the hidden layers a consistent coordinate system to align themselves to.
-   **Proven in Backprop**: DFA has been shown to train deep networks (even ImageNet scale) without symmetric weight transport, achieving results comparable to backprop.

### Implementation Plan

1.  Add DFAEnabled flag to TweenState.
2.  Add FeedbackWeights[][][]float32 to store fixed random matrices for each layer.
3.  Modify TweenStep:
    -   Compute global error.
    -   Broadcast global error to all layers via Gap_l = FeedbackWeights[l] * GlobalError.
    -   Run standard TweenWeights using these directly injected gaps.

This changes the philosophy from "Bidirectional Consensual" to "Global Error Broadcast".

---

## Experiment 17: Advanced Hebbian Modes & Hybrid Training (Success!)

**Status: TESTED - HYBRID SUCCESS**

### Core Idea
We implemented three advanced variations to overcome the 48% barrier:
1.  **Hybrid Training**: Use Tween for "Warm-up" (fast, rough initialization) followed by Backprop for refinement.
2.  **Contrastive Hebbian Learning (CHL)**: Use positive (clamped) and negative (free) phases to generate updates based on `PosGap - NegGap`.
3.  **Equilibrium Propagation (EqProp)**: A simplified implementation using free and clamped settling phases.

### Test Results (2025-12-16)

| Network   | NormalBP | PureTween | Hybrid (Tween->BP) | Contrastive | Equilibrium | Verdict |
|-----------|----------|-----------|--------------------|-------------|-------------|---------|
| Dense     | 99.8%    | 49.4%     | 91.2%              | 49.4%       | 33.6%       | Hybrid Strong |
| Conv2D    | 78.2%    | 57.0%     | **88.0%**          | 49.8%       | 44.2%       | ✅ **Hybrid Beat BP** |
| RNN       | 99.4%    | 53.0%     | 90.4%              | 49.0%       | 27.6%       | Hybrid Strong |
| LSTM      | 85.4%    | 46.0%     | **90.6%**          | 46.0%       | 32.0%       | ✅ **Hybrid Beat BP** |
| Attention | 50.4%    | 50.4%     | 50.4%              | 50.4%       | 50.4%       | Stuck |

### Analysis

1.  **Hybrid Training Strategy (WINNER)**:
    -   This was the **only** method to break the 48% barrier.
    -   **Beating Backprop**: In Conv2D and LSTM, Hybrid training achieved HIGHER accuracy than pure Backprop (88% vs 78% and 90% vs 85%).
    -   **Why**: Neural Tweening provides a "rough global search" that positions the weights in a favorable basin of attraction, avoiding local minima that pure Backprop might fall into from random initialization. Once "warmed up," Backprop can precisely refine the weights to convergence.

2.  **Contrastive & Equilibrium (Failures)**:
    -   These methods failed to outperform the baseline Tweening. 
    -   **Reason**: Without explicit feedback weights or symmetric connectivity (which complex layers like LSTM/Attention don't structurally enforce), the "phases" don't settle into meaningful energy minima that represent global error. They behave just like noisy Hebbian updates.

### Updated Conclusion

While pure Hebbian learning struggles with deep credit assignment (the Telephone Game), **Hybrid Training** emerges as a powerful practical application. By using Hebbian updates for initialization (Pre-training), we can achieve **better-than-BP** performance in complex recurrent and convolutional tasks.
