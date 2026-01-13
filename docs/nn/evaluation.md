# Understanding Evaluation and Metrics

This guide explains Loom's evaluation system—how to measure model performance beyond simple accuracy, track adaptation over time, and compare different training methods.

---

## Beyond Binary Accuracy

Traditional accuracy tells you "80% correct"—but it hides important information:
- How wrong were the 20% that failed?
- Are errors clustered on specific inputs?
- Is the model getting better or worse over time?

Loom's evaluation system answers these questions with **deviation metrics**.

---

## Deviation Metrics: Quality Heatmap

Instead of "right" or "wrong", DeviationMetrics measures *how far off* each prediction is:

```
Traditional Accuracy:

    Correct: ████████████████████████████████████  80%
    Wrong:   ████████████                          20%
    
    "80% accurate" — but how wrong is the 20%?


Deviation Metrics:

    0-10%:   ██████████████████████████  75%  ← Nearly perfect
    10-20%:  █████████                   15%  ← Small error
    20-30%:  ███                          5%  ← Moderate error
    30-50%:  ██                           3%  ← Significant error
    50-100%: █                            1%  ← Very wrong
    100%+:                                1%  ← Completely wrong
    
    Much more informative!
```

### How Deviation is Calculated

```
deviation = |actual - expected| / |expected| × 100%

Example 1: Expected 100, got 95
    deviation = |95 - 100| / 100 × 100% = 5%
    Bucket: "0-10%" (nearly perfect)

Example 2: Expected 100, got 50
    deviation = |50 - 100| / 100 × 100% = 50%
    Bucket: "50-100%" (very wrong)

Example 3: Expected 100, got 250
    deviation = |250 - 100| / 100 × 100% = 150%
    Bucket: "100%+" (catastrophically wrong)
```

### Using DeviationMetrics

```go
// Create metrics tracker
metrics := nn.NewDeviationMetrics()

// Evaluate predictions
for i := range testData {
    output, _ := network.ForwardCPU(testData[i].Input)
    
    // For classification: compare predicted class to expected
    predicted := argmax(output)
    expected := testData[i].Label
    
    result := nn.EvaluatePrediction(i, float64(expected), float64(predicted))
    metrics.UpdateMetrics(result)
}

// Compute final scores
metrics.ComputeFinalMetrics()

// Print summary
metrics.PrintSummary()
```

Output:
```
=== Model Evaluation Summary ===
Total Samples: 1000
Quality Score: 87.32/100
Average Deviation: 12.68%
Failures (>100% deviation): 5 (0.5%)

Deviation Distribution:
    0-10%:  825 samples (82.5%) ████████████████████████████████████████
   10-20%:  100 samples (10.0%) ████
   20-30%:   45 samples  (4.5%) ██
   30-40%:   15 samples  (1.5%) █
   40-50%:    8 samples  (0.8%) 
  50-100%:    2 samples  (0.2%) 
   100%+:     5 samples  (0.5%) 
```

### Quality Score

The quality score is a weighted metric that rewards better predictions:

```
Score contribution per sample:
    perfect (0% deviation):  contributes 100 points
    10% deviation:           contributes 90 points
    50% deviation:           contributes 50 points
    100%+ deviation:         contributes 0 points

Final score = mean(all contributions)

Score ranges:
    90-100: Excellent model
    70-90:  Good model
    50-70:  Needs improvement
    <50:    Significant problems
```

### Finding Worst Predictions

Identify which samples caused the most trouble:

```go
worst := metrics.GetWorstSamples(10)

for _, sample := range worst {
    fmt.Printf("Sample %d: %.1f%% deviation\n", 
        sample.SampleIndex, sample.Deviation)
    fmt.Printf("  Expected: %.2f\n", sample.ExpectedOutput)
    fmt.Printf("  Got:      %.2f\n", sample.ActualOutput)
}
```

This helps you:
- Find edge cases in your data
- Identify patterns in failures
- Debug specific inputs

---

## Training Metrics: Comparing Methods

When testing different training approaches (backprop vs tweening, different optimizers), you need structured comparison.

### TrainingMetrics Structure

```go
type TrainingMetrics struct {
    Steps        int           // Total training iterations
    Accuracy     float64       // Final accuracy %
    Loss         float32       // Final loss value
    TimeTotal    time.Duration // Total training time
    TimeToTarget time.Duration // Time to reach target accuracy
    MemoryPeakMB float64       // Peak memory usage
    Milestones   map[int]time.Duration // Time to reach 10%, 20%, ... 100%
}
```

### Milestone Tracking

Track how quickly the model reaches accuracy milestones:

```go
metrics := nn.NewTrainingMetrics()
start := time.Now()

for step := 0; step < totalSteps; step++ {
    // Train...
    accuracy := evaluate(network, testSet)
    
    // Record milestones
    metrics.UpdateMilestone(accuracy, time.Since(start))
}

// See how long each milestone took
for threshold := 10; threshold <= 100; threshold += 10 {
    if duration, ok := metrics.Milestones[threshold]; ok && duration > 0 {
        fmt.Printf("%d%% accuracy: reached at %v\n", threshold, duration)
    }
}
```

Output:
```
10% accuracy: reached at 2.3s
20% accuracy: reached at 5.1s
30% accuracy: reached at 8.7s
40% accuracy: reached at 14.2s
50% accuracy: reached at 22.5s
60% accuracy: reached at 35.8s
70% accuracy: reached at 52.3s
80% accuracy: reached at 1m12s
90% accuracy: reached at 1m45s
```

### Comparing Training Methods

```go
comparison := nn.ComparisonResult{
    Name:      "MNIST Classification",
    NumLayers: 5,
}

// Test normal backprop
comparison.NormalBP = trainWithBackprop(network)

// Test tweening
comparison.StepTween = trainWithTweening(network)

// Test step+tween chain mode
comparison.StepTweenChain = trainWithTweenChain(network)

// Find the best
best := comparison.DetermineBest()
fmt.Printf("Best method: %s\n", best)
```

The comparison considers both accuracy and loss to determine the winner:

```
Method Comparison:
                  Accuracy    Loss    Time
────────────────────────────────────────────
NormalBP          92.5%       0.23    45s
NormTween         89.1%       0.31    52s
Step+BP           91.8%       0.25    48s
StepTween         93.2%       0.21    55s   ← Best (highest accuracy)
TChain            94.1%       0.18    62s   ← Also excellent
BatchTween        88.5%       0.35    40s
StepBatch         90.2%       0.28    43s
```

---

## Adaptation Tracking: Online Learning Performance

For systems that must adapt to changing tasks in real-time, traditional metrics don't capture the full picture. Loom's **AdaptationTracker** monitors performance over time windows.

### The Scenario

```
Time-based evaluation:

Second 0-5:    Learn Task A (classify animals)
Second 5-10:   TASK SWITCH → Learn Task B (classify vehicles)
Second 10-15:  Continue Task B

Questions:
  • How quickly does accuracy drop at the switch?
  • How quickly does the model recover?
  • What's the average accuracy over the whole period?
```

### Using AdaptationTracker

```go
// Create tracker: 1-second windows over 15 seconds total
tracker := nn.NewAdaptationTracker(
    1*time.Second,   // Window duration
    15*time.Second,  // Total duration
)

tracker.SetModelInfo("MyNetwork", "StepTweenChain")

// Schedule task changes
tracker.ScheduleTaskChange(5*time.Second, 1, "vehicles")

// Start tracking
tracker.Start("animals", 0)

// Training loop
for {
    sample := getNextSample()
    output, _ := network.ForwardCPU(sample.Input)
    
    // Determine correctness based on current task
    currentTask := tracker.GetCurrentTask()
    correct := checkCorrect(output, sample.Label, currentTask)
    
    // Record this output
    tracker.RecordOutput(correct)
    
    if done {
        break
    }
}

// Get results
result := tracker.Finalize()
result.PrintTimeline()
result.PrintAdaptationSummary()
```

### Understanding the Output

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║  MyNetwork — ACCURACY OVER TIME (per window)                                                     ║
║  [animals]                                [vehicles]                                             ║
╠═══════════════════╦════╦════╦════╦════╦════╦════╦════╦════╦════╦════╦════╦════╦════╦════╦════╗
║ Window            ║  1s║  2s║  3s║  4s║  5s║  6s║  7s║  8s║  9s║ 10s║ 11s║ 12s║ 13s║ 14s║ 15s║
╠═══════════════════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╣
║ StepTweenChain    ║ 25%║ 52%║ 78%║ 91%║ 95%║ 35%║ 48%║ 67%║ 82%║ 89%║ 92%║ 94%║ 95%║ 96%║ 96%║
╚═══════════════════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╝
                                              ↑ CHANGE

Interpretation:
  • Before switch (s1-5): Accuracy climbs from 25% to 95% as model learns Task A
  • At switch (s6): Accuracy drops to 35% — model confused by new task
  • Recovery (s6-10): Accuracy recovers to 89% — model adapts to Task B
  • Stability (s11-15): Accuracy stabilizes at 95-96%
```

### Task Change Metrics

The tracker captures detailed information about each task switch:

```go
for _, tc := range result.TaskChanges {
    fmt.Printf("Switch: %s → %s\n", tc.FromTask, tc.ToTask)
    fmt.Printf("  Pre-accuracy:  %.0f%%\n", tc.PreAccuracy)
    fmt.Printf("  Post-accuracy: %.0f%%\n", tc.PostAccuracy)
    fmt.Printf("  Recovery time: %d windows (%v)\n", 
        tc.RecoveryWindows, tc.RecoveryTime)
}
```

Output:
```
Switch: animals → vehicles
  Pre-accuracy:  95%
  Post-accuracy: 35%
  Recovery time: 4 windows (4s)
```

---

## Comparing Multiple Models

When testing several models or training modes, use `AdaptationComparison`:

```go
comparison := nn.NewAdaptationComparison()

// Test each mode
for _, mode := range []string{"NormalBP", "StepBP", "StepTween", "StepTweenChain"} {
    tracker := nn.NewAdaptationTracker(1*time.Second, 15*time.Second)
    tracker.SetModelInfo("MyNetwork", mode)
    tracker.ScheduleTaskChange(5*time.Second, 1, "task_b")
    tracker.Start("task_a", 0)
    
    // Run training for this mode...
    
    result := tracker.Finalize()
    comparison.AddResult(result)
}

// Compare all modes
comparison.PrintComparisonTimeline("Task Adaptation Test", 15)
comparison.PrintComparisonSummary("Task Adaptation Test")

// Find best performers
best := comparison.GetBestByAvgAccuracy()
fmt.Printf("Highest accuracy: %s (%.1f%%)\n", best.ModeName, best.AvgAccuracy)

mostStable := comparison.GetMostStable()
fmt.Printf("Most stable: %s\n", mostStable.ModeName)
```

Output:
```
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║  Task Adaptation Test — ACCURACY OVER TIME                                                       ║
╠═══════════════════╦════╦════╦════╦════╦════╦════╦════╦════╦════╦════╦════╦════╦════╦════╦════╦
║ Mode              ║  1s║  2s║  3s║  4s║  5s║  6s║  7s║  8s║  9s║ 10s║ 11s║ 12s║ 13s║ 14s║ 15s║
╠═══════════════════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╣
║ NormalBP          ║ 20%║ 45%║ 68%║ 82%║ 88%║  5%║ 12%║ 35%║ 55%║ 70%║ 78%║ 82%║ 85%║ 87%║ 88%║
║ StepBP            ║ 22%║ 48%║ 72%║ 85%║ 90%║ 10%║ 25%║ 45%║ 65%║ 78%║ 84%║ 88%║ 90%║ 91%║ 92%║
║ StepTween         ║ 25%║ 55%║ 78%║ 88%║ 93%║ 20%║ 38%║ 58%║ 75%║ 85%║ 90%║ 93%║ 94%║ 95%║ 95%║
║ StepTweenChain    ║ 28%║ 58%║ 82%║ 92%║ 96%║ 35%║ 52%║ 72%║ 85%║ 92%║ 94%║ 96%║ 96%║ 97%║ 97%║
╚═══════════════════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╝

Observations:
  • StepTweenChain recovers fastest from task switch (35% vs 5-20% for others)
  • StepTweenChain reaches highest final accuracy (97%)
  • NormalBP completely crashes at task switch (drops to 5%)
```

---

## Ensemble Discovery: Finding Complementary Models

If you have multiple trained models, Loom can help identify **pairs that complement each other** (high combined coverage, low overlap). This is useful when you want a lightweight ensemble without stacking everything.

```go
// Build correctness masks from evaluation results
buildMask := func(metrics *nn.DeviationMetrics) []bool {
    mask := make([]bool, len(metrics.Results))
    for i, r := range metrics.Results {
        // Treat <=10% deviation as "correct" for ensemble coverage
        mask[i] = r.Deviation <= 10
    }
    return mask
}

models := []nn.ModelPerformance{
    {ModelID: "model_a", Mask: buildMask(metricsA)},
    {ModelID: "model_b", Mask: buildMask(metricsB)},
    {ModelID: "model_c", Mask: buildMask(metricsC)},
}

matches := nn.FindComplementaryMatches(models, 0.95) // require 95% coverage
nn.PrintEnsembleReport(matches, 5)
```

If you want to cluster model behavior, convert masks to float vectors first:

```go
vec := nn.ConvertMaskToFloat(models[0].Mask)
```

---

## Correlation Analysis: Understanding Feature Relationships

For datasets or intermediate activations, Loom provides correlation helpers to spot redundancy or surprising feature relationships.

```go
labels := []string{"f0", "f1", "f2", "f3"}
result := nn.ComputeCorrelationMatrix(data, labels)

// Top correlations by absolute value
pairs := result.GetStrongCorrelations(0.8)
for _, p := range pairs {
    fmt.Printf("%s vs %s: %.3f\n", p.Feature1, p.Feature2, p.Correlation)
}
```

For monotonic-but-non-linear relationships, use Spearman:

```go
spearman := nn.ComputeSpearmanMatrix(data, labels)
```

Correlation results can be serialized for dashboards or WASM:

```go
jsonStr, _ := result.ToJSONCompact()
restored, _ := nn.CorrelationResultFromJSON(jsonStr)
```

---

## Saving and Loading Metrics

### Save Metrics to JSON

```go
// Save deviation metrics
metrics.SaveMetrics("evaluation_results.json")

// Save comparison results
comparison.SaveToJSON("comparison_results.json")
```

### Load Metrics

```go
// Load deviation metrics
loaded, err := nn.LoadMetrics("evaluation_results.json")
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Quality Score: %.2f\n", loaded.Score)
```

---

## Summary

Loom's evaluation system provides:

**DeviationMetrics** - Quality heatmap
- Measures how far off predictions are, not just right/wrong
- Quality score (0-100) that rewards better predictions
- Identifies worst-performing samples

**TrainingMetrics** - Training comparison
- Tracks accuracy milestones over time
- Compares different training methods
- Measures time to target accuracy

**AdaptationTracker** - Real-time performance
- Monitors accuracy over time windows
- Tracks task changes and recovery time
- Compares multiple models/modes

Use these tools to understand not just *if* your model works, but *how well* and *where it struggles*.
