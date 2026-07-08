# Dense Adaptation Benchmark (with SIMD Variants)

This folder contains LOOM's dense mid-stream adaptation benchmark: a compact
"test-time training" scenario that compares training/update strategies under
task flips while decisions must keep flowing.

It now benchmarks **12 variants**:

- 6 update modes
- each with `SIMD=off` and `SIMD=on` forward paths

The SIMD column uses LOOM Plan 9 CPU vector kernels (AVX2 on x86, NEON on arm)
for forward matmul-heavy layers. Training/backward semantics are unchanged.

## What This Tests

Timeline:

- `0-5s`: CHASE target
- `5-10s`: AVOID target
- `10-15s`: CHASE target again

Model:

```text
6-layer Dense: 8 -> 32 -> 64 -> 64 -> 64 -> 32 -> 4
Engine: loom/poly VolumetricNetwork
```

Decision deadline:

- `10ms` per action loop (used in activity/latency metrics)

The point is not "final offline convergence." The point is operational behavior:
how well a mode adapts *while live decisions continue*.

## Concepts and Diagrams

![Six LOOM training modes](train_modes.jpg)

The six update families are:

- `NormalBP`: periodic batch BP (`poly.Train`)
- `Step+BP`: per-output BP update
- `Tween`: periodic target-prop style updates
- `TweenChain`: chain-rule tween periodic updates
- `StepTween`: per-output tween updates
- `StepTweenChain`: per-output chain-rule tween updates

![LOOM continuous test-time training overview](ttt.jpg)

The TTT framing in this benchmark is:

1. **Zero-downtime learning**  
   Keep predict-update-predict flowing without pausing the decision loop.
2. **Proof of adaptation under change**  
   Compare how each mode handles sudden objective flips.
3. **Competence measuring and self-healing hooks**  
   Track latency, drift-like behavior, and online recovery metrics.

## Running It

```text
cd lucy
go run .
# choose [2] Tests — dense mid-stream adaptation benchmark
```

Output behavior:

- prints full benchmark live in terminal
- also auto-saves a full log to:
  - `lucy/lucy_testing_output/dense_adaptation_<timestamp>.txt`

## Latest Captured Run (SIMD on/off)

Source log:

- `lucy/lucy_testing_output/dense_adaptation_20260708_194000.txt`

### Total Outputs

| Mode | SIMD off | SIMD on | Delta |
| --- | ---: | ---: | ---: |
| `NormalBP` | 66,136 | 216,109 | +226.8% |
| `Step+BP` | 10,175 | 11,159 | +9.7% |
| `Tween` | 132,128 | 314,194 | +137.8% |
| `TweenChain` | 84,218 | 212,812 | +152.7% |
| `StepTween` | 48,558 | 54,581 | +12.4% |
| `StepTweenChain` | 7,752 | 8,059 | +4.0% |

### Avg Accuracy

| Mode | SIMD off | SIMD on | Delta |
| --- | ---: | ---: | ---: |
| `NormalBP` | 29.1 | 34.1 | +5.0 |
| `Step+BP` | 37.7 | 38.5 | +0.8 |
| `Tween` | 30.4 | 20.3 | -10.1 |
| `TweenChain` | 28.6 | 34.8 | +6.2 |
| `StepTween` | 3.5 | 4.2 | +0.7 |
| `StepTweenChain` | 43.0 | 44.0 | +1.0 |

### Operational Metrics Snapshot

| Mode | SIMD | Throughput/s | Availability | Peak lat | Avg lat | Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `NormalBP` | off | 4,409 | 57.7% | 94ms | 227us | 741 |
| `NormalBP` | on  | 14,407 | 60.5% | 80ms | 69us | 2,971 |
| `Step+BP` | off | 678 | 100.0% | 29ms | 1ms | 256 |
| `Step+BP` | on  | 744 | 100.0% | 30ms | 1ms | 287 |
| `Tween` | off | 8,809 | 88.6% | 44ms | 113us | 2,377 |
| `Tween` | on  | 20,946 | 89.6% | 31ms | 48us | 3,816 |
| `TweenChain` | off | 5,615 | 56.2% | 95ms | 178us | 903 |
| `TweenChain` | on  | 14,187 | 58.4% | 86ms | 70us | 2,879 |
| `StepTween` | off | 3,237 | 100.0% | 20ms | 309us | 114 |
| `StepTween` | on  | 3,639 | 100.0% | 20ms | 275us | 154 |
| `StepTweenChain` | off | 517 | 100.0% | 30ms | 2ms | 222 |
| `StepTweenChain` | on  | 537 | 100.0% | 27ms | 2ms | 236 |

### Practical Winner Summary (this run)

- **Best adaptation quality:** `StepTweenChain` (SIMD on)
- **Best raw speed:** `Tween` (SIMD on)
- **Largest SIMD speedup:** `NormalBP`, `Tween`, `TweenChain`

## Mode-by-Mode Applications

### 1) NormalBP

Use when you can tolerate training windows and want strong throughput.

- batch micro-retrain style operation
- useful for scheduled refresh cycles
- can block more than step modes

### 2) Step+BP

Use for online correctness-first adaptation with full BP signal.

- per-output updates
- strong availability and stable adaptation
- lower throughput than pure forward-heavy modes

### 3) Tween

Use as a lightweight adaptive option where batch windows are acceptable.

- cheaper than full BP in many flows
- can be very fast with SIMD forward
- adaptation quality depends heavily on scenario dynamics

### 4) TweenChain

Use when chain-rule tween improves competence while keeping tween semantics.

- often better adaptation than plain Tween
- still periodic/batch behavior in this benchmark variant

### 5) StepTween

Use when maintaining continuous decision activity is the top priority.

- very high availability
- per-output updates keep activity flowing
- competence can be weak without stronger chain signal

### 6) StepTweenChain

Best fit for "always-on + meaningful adaptation" in this benchmark family.

- online chain-rule tween update each output
- high no-pause behavior
- strongest combined adaptation in latest run

## Mapping to Real Systems

The CHASE/AVOID task switch is a proxy for production change-points:

- attack behavior shifts
- new exploit sequence appears
- benign baseline drifts after deployment/event
- decision boundary from yesterday goes stale today

Operational question:

> Can the system keep making timely decisions while adapting, or does it pause
> and miss the window?

### Example deployment mapping

- `NormalBP`: scheduled retraining windows, throughput-oriented offline refresh
- `Step+BP`: online adaptation where correctness under drift dominates
- `Tween`/`TweenChain`: lightweight adaptation with bounded complexity
- `StepTween`: continuous decision stream priority (reactivity first)
- `StepTweenChain`: continuous decision stream + strongest online adaptation blend

## Metric Cheat Sheet

- **AvgAcc**: mean accuracy across all 15 windows
- **Availability**: non-blocked runtime share
- **Throughput/s**: actions per second
- **Score**: `Throughput * Availability% * AvgAcc / 10,000`
- **ZDT**: `AvgAcc * Availability% / 100`
- **Activity**: `Availability% * DeadlineHit% / 100`
- **Reactive**: `ZDT * DeadlineHit% / 100`
- **Smoothness**: `100 - mean(abs(delta adjacent windows))`
- **CompetenceSmooth**: `AvgAcc - mean(abs(delta adjacent windows))`

## Notes on SIMD Scope

- SIMD toggle here accelerates **forward pass** workloads.
- Backward/training update math remains the same algorithmically.
- This is why some modes show huge speed gains while adaptation quality can vary:
  speed is improved directly, but update signal quality/behavior is mode-dependent.

