# Understanding Introspection and Telemetry

This guide explains Loom's observability features—how to look inside a running network, monitor its health, and debug training problems.

---

## Why Observability Matters

Training neural networks is often a black box:
- Forward pass: numbers go in, numbers come out
- Backward pass: gradients flow somewhere
- After 1000 epochs: did it work?

Loom breaks open this black box with three systems:
1. **Introspection**: Discover what methods and capabilities exist
2. **Telemetry**: Extract structural info and live statistics
3. **Observers**: Watch layer-by-layer execution in real time

---

## Introspection: What Can This Network Do?

Introspection answers: "What methods exist on this Network object?"

This is especially useful when you're:
- Building dynamic UIs that need to know available operations
- Working across languages (WASM, C ABI) where you can't just look at Go source
- Auto-generating documentation or bindings

### Discovering Methods

```go
network := nn.NewNetwork(1024, 2, 2, 2)

// Get all methods
methods, err := network.GetMethods()

for _, m := range methods {
    fmt.Printf("%s(", m.MethodName)
    for i, p := range m.Parameters {
        if i > 0 { fmt.Print(", ") }
        fmt.Printf("%s", p.Type)
    }
    fmt.Printf(") → %s\n", strings.Join(m.Returns, ", "))
}
```

Output:
```
ForwardCPU([]float32) → []float32, time.Duration
BackwardCPU([]float32) → []float32, time.Duration
Train([]nn.Batch, *nn.TrainingConfig) → *nn.TrainingResult, error
SaveModel(string, string) → error
GetBlueprint() → nn.NetworkBlueprint
...
```

### The MethodInfo Structure

```
MethodInfo
┌───────────────────────────────────────────────────────────────┐
│ MethodName: "ForwardCPU"                                      │
│                                                               │
│ Parameters:                                                   │
│   ┌─────────────────────────────────────────────────────────┐ │
│   │ [0] Name: "input"    Type: "[]float32"                  │ │
│   └─────────────────────────────────────────────────────────┘ │
│                                                               │
│ Returns:                                                      │
│   ["[]float32", "time.Duration"]                              │
└───────────────────────────────────────────────────────────────┘
```

### Checking for Specific Methods

```go
// Does this network have GPU support?
if network.HasMethod("ForwardGPU") {
    fmt.Println("GPU acceleration available!")
}

// Get signature for a specific method
sig, _ := network.GetMethodSignature("Train")
fmt.Println(sig)
// Output: Train([]nn.Batch, *nn.TrainingConfig) (*nn.TrainingResult, error)
```

### JSON Export (for WASM/API)

```go
jsonStr, _ := network.GetMethodsJSON()
// Returns complete method info as JSON for JavaScript/API use
```

The JSON is useful when you need to:
- Generate TypeScript types automatically
- Build dynamic UIs that show all available operations
- Document API endpoints

---

## Telemetry: Network Structure and Statistics

Telemetry gives you a bird's-eye view of your network.

### Network Blueprint

A blueprint describes the network's architecture:

```go
blueprint := network.GetBlueprint()

fmt.Printf("Grid: %d×%d, Layers/Cell: %d\n",
    blueprint.GridRows, blueprint.GridCols, blueprint.LayersPerCell)
fmt.Printf("Total Layers: %d\n", blueprint.TotalLayers)
fmt.Printf("Total Parameters: %d\n", blueprint.TotalParams)
```

Output:
```
Grid: 2×3, Layers/Cell: 2
Total Layers: 12
Total Parameters: 2,359,296
```

### Layer-by-Layer Breakdown

```go
for _, layer := range blueprint.Layers {
    fmt.Printf("[%d,%d,%d] %s: %d → %d (%d params)\n",
        layer.Row, layer.Col, layer.Layer,
        layer.Type,
        layer.InputSize, layer.OutputSize,
        layer.NumParams)
}
```

Output:
```
[0,0,0] Dense: 1024 → 512 (524,800 params)
[0,0,1] Dense: 512 → 256 (131,328 params)
[0,1,0] Attention: 256 → 256 (262,656 params)
[0,1,1] Dense: 256 → 128 (32,896 params)
...
```

This is like having an X-ray of your network:

```
Blueprint Visualization:

┌─────────────────────────────────────────────────────────────────┐
│                           NETWORK                               │
│                                                                 │
│  Grid: 2 rows × 3 columns                                       │
│  Total: 12 layers, 2.3M parameters                              │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Dense 1024→512│  │ MHA 256→256  │  │ Dense 256→128│          │
│  │ Dense 512→256 │  │ Norm 256→256 │  │ Softmax 128  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ LSTM 128→64  │  │ Dense 64→32  │  │ Output 32→10 │          │
│  │ Norm 64→64   │  │ ReLU         │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Live Telemetry

Beyond structure, you can get live statistics:

```go
telemetry := network.GetTelemetry()

fmt.Printf("Memory Usage: %.2f MB\n", float64(telemetry.MemoryUsage)/1024/1024)

for _, layer := range telemetry.LayerStats {
    fmt.Printf("Layer %d: weights=%.4f, gradients=%.4f, activation=[%.2f, %.2f]\n",
        layer.Index,
        layer.WeightNorm,
        layer.GradientNorm,
        layer.ActivationMin, layer.ActivationMax)
}
```

Output:
```
Memory Usage: 9.12 MB

Layer 0: weights=15.2341, gradients=0.0234, activation=[-2.15, 4.82]
Layer 1: weights=12.8965, gradients=0.0189, activation=[-1.89, 3.21]
Layer 2: weights=18.4521, gradients=0.0412, activation=[-3.45, 5.67]
...
```

---

## Observers: Real-Time Monitoring

Observers watch what happens inside the network during execution.

### The Observer Interface

```go
type Observer interface {
    OnLayerForward(layerIdx int, stats LayerStats)
    OnLayerBackward(layerIdx int, stats LayerStats)
    OnTrainingEvent(event TrainingEvent)
}
```

Every time a layer processes data, the observer gets notified.

### LayerStats: What You See

```
LayerStats
┌─────────────────────────────────────────────────────────────────┐
│ LayerIndex:   3                                                 │
│ LayerType:    "Dense"                                           │
│ InputShape:   [32, 256]                                         │
│ OutputShape:  [32, 128]                                         │
│                                                                 │
│ Weight Statistics:                                              │
│   WeightNorm:  12.45                                            │
│   BiasNorm:    0.89                                             │
│                                                                 │
│ Output Statistics:                                              │
│   OutputMin:   -3.21                                            │
│   OutputMax:    4.56                                            │
│   OutputMean:   0.12                                            │
│   OutputStd:    1.45                                            │
│                                                                 │
│ Performance:                                                    │
│   ComputeTimeNs:  125000  (0.125 ms)                            │
└─────────────────────────────────────────────────────────────────┘
```

### Console Observer: Print Everything

```go
observer := nn.NewConsoleObserver()
network.SetObserver(observer)

output, _ := network.ForwardCPU(input)
```

Output:
```
[Layer 0] Dense: 1024 → 512
  Output: mean=0.0234, std=0.8921, range=[-2.15, 4.82]
  Time: 0.125ms

[Layer 1] Dense: 512 → 256
  Output: mean=0.0156, std=0.7234, range=[-1.89, 3.21]
  Time: 0.089ms

[Layer 2] Attention: 256 → 256
  Output: mean=0.0089, std=0.6543, range=[-1.45, 2.98]
  Time: 0.342ms
...
```

### Recording Observer: Capture History

```go
observer := nn.NewRecordingObserver()
network.SetObserver(observer)

// Run some forward passes
for _, batch := range data {
    network.ForwardCPU(batch.Input)
}

// Get the recorded history
history := observer.GetHistory()

// Analyze: did any layer's output explode?
for _, record := range history {
    if record.OutputMax > 100 {
        fmt.Printf("⚠️ Layer %d had large output: %.2f\n",
            record.LayerIndex, record.OutputMax)
    }
}

// Save for later analysis
observer.SaveToFile("training_recording.json")
```

### Channel Observer: Custom Processing

For advanced use cases, send stats to a Go channel:

```go
statsChan := make(chan nn.LayerStats, 1000)
observer := nn.NewChannelObserver(statsChan)
network.SetObserver(observer)

// Process in background
go func() {
    layerMeans := make(map[int][]float32)
    
    for stats := range statsChan {
        layerMeans[stats.LayerIndex] = append(
            layerMeans[stats.LayerIndex], 
            stats.OutputMean,
        )
        
        // Detect drift
        if len(layerMeans[stats.LayerIndex]) > 100 {
            recent := layerMeans[stats.LayerIndex][len(layerMeans[stats.LayerIndex])-100:]
            if trend(recent) > 0.1 {
                fmt.Printf("⚠️ Layer %d is drifting upward\n", stats.LayerIndex)
            }
        }
    }
}()
```

### HTTP Observer: Remote Monitoring

Send telemetry to a monitoring service:

```go
observer := nn.NewHTTPObserver("http://localhost:8080/telemetry")
network.SetObserver(observer)

// Stats are POSTed to the endpoint during training
```

---

## Training Events

Observers also receive training events:

```go
type TrainingEvent struct {
    Type      string   // "epoch_start", "epoch_end", "batch_end"
    Epoch     int
    Batch     int
    Loss      float32
    Accuracy  float32
}
```

Example handler:
```go
func (o *MyObserver) OnTrainingEvent(event TrainingEvent) {
    switch event.Type {
    case "epoch_end":
        fmt.Printf("Epoch %d completed. Loss: %.4f\n", 
            event.Epoch, event.Loss)
    case "batch_end":
        // Update progress bar
        updateProgress(event.Batch)
    }
}
```

---

## Evaluation Metrics

Loom includes a built-in evaluation system that tracks accuracy across different error buckets.

### DeviationMetrics

```go
metrics := nn.NewDeviationMetrics()

for _, sample := range testSet {
    output, _ := network.ForwardCPU(sample.Input)
    metrics.Update(output, sample.Target)
}

fmt.Printf("Quality Score: %.1f%%\n", metrics.Score)
```

### The Deviation Buckets

Instead of just "right" or "wrong", DeviationMetrics tracks *how* right or wrong:

```
Deviation Buckets:

Perfect    │████████████        │  0% deviation   (exactly right)
Excellent  │██████████          │  <5% deviation  (very close)
Good       │████████            │  <10% deviation (close enough)
Acceptable │██████              │  <20% deviation (okay)
Poor       │████                │  <50% deviation (significant error)
Bad        │███                 │  <100% deviation (very wrong)
Failed     │                    │  >100% deviation (completely wrong)
           └────────────────────┘
              Number of samples

Quality Score = weighted combination favoring better buckets
```

### Why This Matters

Binary accuracy hides information:

```
Model A:                           Model B:
  Correct: 80%                       Correct: 80%
  Wrong: 20%                         Wrong: 20%
  
  Looks the same!

But with DeviationMetrics:

Model A (consistent):              Model B (unstable):
  Perfect:    75%                    Perfect:    30%
  Excellent:  15%                    Excellent:  20%
  Good:        5%                    Good:       30%
  Acceptable:  3%                    Acceptable: 15%
  Poor:        2%                    Poor:        5%
  
  Mostly very accurate                Mixed results

Model A is clearly better for deployment!
```

### Sample-Level Tracking

Find your worst predictions:

```go
worst := metrics.GetWorstSamples(10)

for _, s := range worst {
    fmt.Printf("Sample %d: %.1f%% deviation\n", s.SampleID, s.Deviation*100)
    fmt.Printf("  Expected: %v\n", s.Expected)
    fmt.Printf("  Got:      %v\n", s.Predicted)
}
```

This helps you:
- Find problematic inputs
- Discover edge cases
- Debug specific failures

---

## Debugging Common Problems

### Problem: Vanishing Gradients

Symptom: Early layers have near-zero gradients

```go
observer := nn.NewRecordingObserver()
network.SetObserver(observer)

// After backward pass
for _, record := range observer.GetBackwardHistory() {
    if record.GradientNorm < 1e-6 {
        fmt.Printf("⚠️ Layer %d has vanishing gradients: %.2e\n",
            record.LayerIndex, record.GradientNorm)
    }
}
```

Solutions:
- Use ReLU instead of sigmoid/tanh
- Add residual connections
- Use batch/layer normalization
- Try Neural Tweening

### Problem: Exploding Gradients

Symptom: Gradients or activations become very large or NaN

```go
for _, record := range observer.GetHistory() {
    if math.IsNaN(float64(record.OutputMean)) {
        fmt.Printf("❌ Layer %d produces NaN!\n", record.LayerIndex)
    }
    if record.GradientNorm > 100 {
        fmt.Printf("⚠️ Layer %d has exploding gradients: %.2f\n",
            record.LayerIndex, record.GradientNorm)
    }
}
```

Solutions:
- Gradient clipping
- Lower learning rate
- Better weight initialization
- Neural Tweening (has automatic explosion detection)

### Problem: Dead ReLU

Symptom: Many neurons always output zero

```go
for _, record := range observer.GetHistory() {
    zeros := countZeros(record.OutputValues)
    ratio := float32(zeros) / float32(len(record.OutputValues))
    
    if ratio > 0.5 {
        fmt.Printf("⚠️ Layer %d has %.0f%% dead neurons\n",
            record.LayerIndex, ratio*100)
    }
}
```

Solutions:
- Use LeakyReLU instead of ReLU
- Lower learning rate
- Better initialization

---

## Putting It Together

A complete monitoring setup:

```go
func trainWithMonitoring(network *nn.Network, data []Sample) {
    // Set up observer
    observer := nn.NewRecordingObserver()
    network.SetObserver(observer)
    
    // Set up evaluation
    metrics := nn.NewDeviationMetrics()
    
    // Print blueprint
    blueprint := network.GetBlueprint()
    fmt.Printf("Training %s with %d layers, %d parameters\n",
        "MyModel", blueprint.TotalLayers, blueprint.TotalParams)
    
    for epoch := 0; epoch < 100; epoch++ {
        for _, sample := range data {
            output, _ := network.ForwardCPU(sample.Input)
            loss, grad := nn.CrossEntropyLossGrad(output, sample.Target)
            network.BackwardCPU(grad)
            network.ApplyGradients(0.001)
        }
        
        // Check for problems
        history := observer.GetHistory()
        for _, record := range history {
            if record.GradientNorm > 50 {
                fmt.Printf("Epoch %d: ⚠️ High gradient at layer %d\n",
                    epoch, record.LayerIndex)
            }
        }
        observer.Clear()
        
        // Evaluate
        for _, sample := range testData {
            output, _ := network.ForwardCPU(sample.Input)
            metrics.Update(output, sample.Target)
        }
        
        fmt.Printf("Epoch %d: Score=%.1f%%\n", epoch, metrics.Score)
        metrics.Reset()
    }
    
    // Final analysis
    telemetry := network.GetTelemetry()
    fmt.Printf("Final memory usage: %.2f MB\n",
        float64(telemetry.MemoryUsage)/1024/1024)
    
    // Save recording for post-training analysis
    observer.SaveToFile("training_history.json")
}
```

---

## Summary

Loom's observability features let you:

**Introspection** - Discover network capabilities
- `GetMethods()` - List all available methods
- `HasMethod()` - Check for specific capabilities
- `GetMethodsJSON()` - Export for WASM/API

**Telemetry** - Understand network structure
- `GetBlueprint()` - Architecture overview
- `GetTelemetry()` - Live statistics

**Observers** - Watch execution in real time
- `ConsoleObserver` - Print everything
- `RecordingObserver` - Capture history
- `ChannelObserver` - Custom processing
- `HTTPObserver` - Remote monitoring

**Evaluation** - Measure quality
- `DeviationMetrics` - Accuracy buckets
- Sample tracking - Find worst predictions

Use these tools to debug problems, optimize training, and understand what your network is really doing.
