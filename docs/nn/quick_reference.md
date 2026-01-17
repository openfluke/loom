# Quick Reference

Concise reference for common operations in the `nn` package — Loom's **Deterministic Neural Virtual Machine** core.

---

## Network Creation

```go
network := nn.NewNetwork(
    inputSize,    // Input dimension
    gridRows,     // Grid rows
    gridCols,     // Grid columns
    layersPerCell, // Layers per grid cell
)
```

---

## Layer Initialization

### Dense

```go
config := nn.InitDenseLayer(inputSize, outputSize, activation)
network.SetLayer(row, col, layer, config)
```

### Conv2D

```go
config := nn.InitConv2DLayer(
    height, width, channels,  // Input
    filters, kernelSize,      // Convolution
    stride, padding,          // Spatial
    activation,               // Activation
)
```

### Conv1D

```go
config := nn.InitConv1DLayer(
    seqLen, inChannels,       // Input
    kernelSize, stride, padding,
    filters,
    activation,
)
```

### Multi-Head Attention

```go
config := nn.InitMultiHeadAttentionLayer(dModel, numHeads, batchSize, seqLength)
```

### Embedding

```go
config := nn.InitEmbeddingLayer(vocabSize, embeddingDim)
```

### RNN / LSTM

```go
rnn := nn.InitRNNLayer(inputSize, hiddenSize, batchSize, seqLength)
lstm := nn.InitLSTMLayer(inputSize, hiddenSize, batchSize, seqLength)
```

### SwiGLU

```go
config := nn.InitSwiGLUBrain(dModel, initScale)
```

### Softmax

```go
standard := nn.InitSoftmaxLayer()
grid := nn.InitGridSoftmaxLayer(rows, cols)           // MoE
temp := nn.InitTemperatureSoftmaxLayer(temperature)
masked := nn.InitMaskedSoftmaxLayer(size)
```

### Normalization

```go
layerNorm := nn.InitLayerNormLayer(size, epsilon)
rmsNorm := nn.InitRMSNormLayer(size, epsilon)
```

### Structural

```go
sequential := nn.NewSequentialLayer()
parallel := nn.NewParallelLayer(combineMode)
residual := nn.InitResidualLayer()
```

---

## Forward / Backward

```go
// Forward
output, duration := network.ForwardCPU(input)

// Backward
gradInput, duration := network.BackwardCPU(gradOutput)
```

---

## Training

### Basic Loop

```go
for epoch := 0; epoch < epochs; epoch++ {
    for _, batch := range data {
        // Forward
        output, _ := network.ForwardCPU(batch.Input)
        
        // Loss + gradient
        loss, gradOutput := nn.CrossEntropyLossGrad(output, batch.Target)
        
        // Backward
        network.BackwardCPU(gradOutput)
        
        // Update
        network.ClipGradients(1.0)
        network.ApplyGradients(learningRate)
    }
}
```

### With Optimizer

```go
optimizer := nn.NewAdamWOptimizer(0.9, 0.999, 1e-8, 0.01)
network.SetOptimizer(optimizer)
network.ApplyGradients(learningRate)
```

### With Scheduler

```go
scheduler := nn.NewCosineAnnealingScheduler(maxLR, minLR, totalSteps)
for step := 0; step < totalSteps; step++ {
    lr := scheduler.GetLR(step)
    network.ApplyGradients(lr)
}
```

---

## Neural Tweening

```go
config := &nn.TweenConfig{
    BaseRate:      0.01,
    MomentumDecay: 0.9,
}
ts := nn.NewTweenState(network, config)

loss := ts.TweenStep(network, input, targetClass, outputSize, rate, backend)
```

---

## Serialization

### Save / Load

```go
// Save
network.SaveModel("model.json", "model_id")

// Load
network, _ := nn.LoadModel("model.json", "model_id")
```

### String-Based (WASM/FFI)

```go
// Save to string
jsonStr, _ := network.SaveModelToString("model_id")

// Load from string
network, _ := nn.LoadModelFromString(jsonStr, "model_id")
```

---

## Activation Types

| Type | Value | Formula |
|------|-------|---------|
| `ActivationReLU` | 0 | `max(0, 1.1x)` |
| `ActivationSigmoid` | 1 | `1/(1+e^(-x))` |
| `ActivationTanh` | 2 | `tanh(x)` |
| `ActivationSoftplus` | 3 | `log(1+e^x)` |
| `ActivationLeakyReLU` | 4 | `x if x≥0, else 0.1x` |
| `ActivationLinear` | 5 | `x` |

---

## Optimizers

| Optimizer | Creation |
|-----------|----------|
| SGD | `nn.NewSGDOptimizer(momentum)` |
| AdamW | `nn.NewAdamWOptimizer(β1, β2, ε, weightDecay)` |
| RMSprop | `nn.NewRMSpropOptimizer(α, ε, momentum)` |

---

## Schedulers

| Scheduler | Creation |
|-----------|----------|
| Constant | `nn.NewConstantScheduler(lr)` |
| Linear | `nn.NewLinearDecayScheduler(init, final, steps)` |
| Cosine | `nn.NewCosineAnnealingScheduler(max, min, steps)` |
| Exponential | `nn.NewExponentialDecayScheduler(init, γ)` |
| Warmup | `nn.NewWarmupScheduler(target, warmupSteps)` |
| Step | `nn.NewStepDecayScheduler(init, γ, stepSize)` |
| Polynomial | `nn.NewPolynomialDecayScheduler(init, final, steps, power)` |

---

## Loss Functions

```go
// MSE
loss := nn.MSELoss(output, target)
loss, grad := nn.MSELossGrad(output, target)

// Cross-Entropy
loss := nn.CrossEntropyLoss(output, target)
loss, grad := nn.CrossEntropyLossGrad(output, target)
```

---

## Introspection

```go
methods, _ := network.GetMethods()           // []MethodInfo
jsonStr, _ := network.GetMethodsJSON()       // JSON string
names := network.ListMethods()               // []string
exists := network.HasMethod("ForwardCPU")    // bool
sig, _ := network.GetMethodSignature("...")  // string
```

---

## Telemetry

```go
blueprint := network.GetBlueprint()   // NetworkBlueprint
telemetry := network.GetTelemetry()   // ModelTelemetry
```

---

## Observer

```go
observer := nn.NewConsoleObserver()
network.SetObserver(observer)

observer := nn.NewRecordingObserver()
observer.SaveToFile("recording.json")
```

---

## Utility Functions

```go
diff := nn.MaxAbsDiff(a, b)  // float64
min := nn.Min(slice)         // float32
max := nn.Max(slice)         // float32
mean := nn.Mean(slice)       // float32
```

---

## Generic Types

```go
// Any numeric type
tensor := nn.NewTensor[float32](shape)
tensor := nn.NewTensor[int8](shape)

// Generic layer forward
output := nn.DenseForwardGeneric[T](cfg, backend, input)
```

---

## Combine Modes (Parallel)

| Mode | Description |
|------|-------------|
| `CombineConcat` | Concatenate outputs |
| `CombineAdd` | Element-wise add |
| `CombineAvg` | Element-wise average |
| `CombineGridScatter` | Place at grid positions |
| `CombineFilter` | Softmax-gated selection |

---

## Model Loading

### Generic (Auto-detect)

```go
network, detected, _ := nn.LoadGenericFromBytes(weightsData, configData)
```

### Transformer (Llama-based)

```go
network, _ := nn.LoadTransformerFromSafetensors("./model_dir/")
```
