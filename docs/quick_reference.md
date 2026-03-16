# Quick Reference: Common Code Snippets

Concise, copy-paste-ready patterns for the most common `poly/` tasks. Each snippet assumes `import poly "github.com/openfluke/soul/loom/poly"` (adjust to your module path).

---

## Creating a Network

```go
// NewVolumetricNetwork(id, depth, rows, cols, layersPerCell)
network := poly.NewVolumetricNetwork("my-net", 1, 3, 1, 1)
// 1×3×1 grid = 3 layers stacked in the Y dimension
```

---

## Adding and Configuring Layers

```go
// Retrieve a layer by 4D coordinate (z, y, x, layerIndex)
l := network.GetLayer(0, 0, 0, 0)

// Dense layer
l.Type         = poly.LayerDense
l.InputHeight  = 128
l.OutputHeight = 64
l.Activation   = poly.ActivationReLU
l.DType        = poly.DTypeFloat32

// Initialize weights (random)
poly.InitializeLayerWeights(l)
```

---

## Forward Pass

```go
input := poly.NewTensor[float32](128)  // flat [128] input
copy(input.Data, myInputData)

output, inputs, preActs := poly.ForwardPolymorphic[float32](network, input)
// output  = final layer's output tensor
// inputs  = cached inputs for each layer (needed for backward)
// preActs = cached pre-activations for each layer
```

---

## Backward Pass

```go
// Compute loss gradient (e.g., MSE gradient)
target := poly.NewTensor[float32](64)
copy(target.Data, myTargetData)

gradOutput := poly.ComputeLossGradient[float32](output, target, poly.LossMSE)

// Backpropagate
gradInput, layerGrads := poly.BackwardPolymorphic[float32](network, gradOutput, inputs, preActs)
```

---

## Applying Gradients

```go
lr := float32(0.001)
poly.ApplyRecursiveGradients[float32](network, layerGrads, lr)
```

---

## Full Training Loop (Manual)

```go
for epoch := 0; epoch < 100; epoch++ {
    output, inputs, preActs := poly.ForwardPolymorphic[float32](network, input)
    loss := poly.CalculateLoss[float32](output, target, poly.LossMSE)
    gradOutput := poly.ComputeLossGradient[float32](output, target, poly.LossMSE)
    _, layerGrads := poly.BackwardPolymorphic[float32](network, gradOutput, inputs, preActs)
    poly.ApplyRecursiveGradients[float32](network, layerGrads, 0.001)
    fmt.Printf("epoch %d  loss=%.4f\n", epoch, loss)
}
```

---

## Batch Training (High-Level)

```go
config := poly.TrainingConfig{
    LearningRate: 0.001,
    Epochs:       50,
    BatchSize:    32,
    LossFunction: poly.LossMSE,
    UseGPU:       false,
}

result := poly.Train[float32](network, trainingData, config)
fmt.Printf("final loss: %.4f\n", result.FinalLoss)
```

---

## Type-Switching with Generics

```go
// Run forward pass with any numeric type
func runForward[T poly.Numeric](net *poly.VolumetricNetwork, data []T) *poly.Tensor[T] {
    input := poly.NewTensor[T](len(data))
    copy(input.Data, data)
    out, _, _ := poly.ForwardPolymorphic[T](net, input)
    return out
}

// Call with float32
out32 := runForward[float32](network, myFloat32Data)

// Call with int8
out8 := runForward[int8](network, myInt8Data)
```

---

## Quantizing a Trained Network

```go
// Convert all layers to Int8
poly.MorphLayer(network, poly.DTypeInt8)

// Convert to Int4 (4-bit)
poly.MorphLayer(network, poly.DTypeInt4)

// Revert: clear versions and retrain or re-morph
for i := range network.Layers {
    network.Layers[i].WeightStore.Versions = make(map[poly.DType]any)
}
poly.MorphLayer(network, poly.DTypeBFloat16)
```

---

## Saving and Loading (Full Weights)

```go
// Save
jsonData, err := poly.SerializeNetwork(network)
if err != nil { log.Fatal(err) }
os.WriteFile("model.json", jsonData, 0644)

// Load
jsonData, _ := os.ReadFile("model.json")
network, err := poly.DeserializeNetwork(jsonData)
if err != nil { log.Fatal(err) }
```

---

## Architecture-Only JSON (Random Weights)

```go
spec := `{
  "id": "my-net",
  "depth": 1, "rows": 2, "cols": 1, "layers_per_cell": 1,
  "layers": [
    {"z":0,"y":0,"x":0,"l":0,"type":"Dense","activation":"ReLU",
     "dtype":"float32","input_height":128,"output_height":64},
    {"z":0,"y":1,"x":0,"l":0,"type":"Dense","activation":"Linear",
     "dtype":"float32","input_height":64,"output_height":10}
  ]
}`

network, err := poly.BuildNetworkFromJSON([]byte(spec))
```

---

## Parallel Branches

```go
l.Type        = poly.LayerParallel
l.CombineMode = "concat"
l.ParallelBranches = []poly.VolumetricLayer{
    {Type: poly.LayerDense, InputHeight: 64, OutputHeight: 32,
     Activation: poly.ActivationReLU, DType: poly.DTypeFloat32},
    {Type: poly.LayerRNN,   InputHeight: 64, OutputHeight: 32,
     Activation: poly.ActivationTanh, DType: poly.DTypeFloat32},
}
```

---

## Sequential Sub-Layers

```go
l.Type = poly.LayerSequential
l.SequentialLayers = []poly.VolumetricLayer{
    {Type: poly.LayerRMSNorm, InputHeight: 256, OutputHeight: 256},
    {Type: poly.LayerDense,   InputHeight: 256, OutputHeight: 256,
     Activation: poly.ActivationGELU, DType: poly.DTypeFloat32},
}
```

---

## Soft Mixture of Experts

```go
l.Type        = poly.LayerParallel
l.CombineMode = "filter"
l.FilterGateConfig = &poly.VolumetricLayer{
    Type:         poly.LayerDense,
    InputHeight:  64,
    OutputHeight: 3,  // one weight per expert
    Activation:   poly.ActivationLinear,
}
l.ParallelBranches = []poly.VolumetricLayer{
    {Type: poly.LayerDense, InputHeight: 64, OutputHeight: 32, ...},
    {Type: poly.LayerDense, InputHeight: 64, OutputHeight: 32, ...},
    {Type: poly.LayerDense, InputHeight: 64, OutputHeight: 32, ...},
}
```

---

## Remote Link (Spatial Hop)

```go
// Layer at (0,1,0,0) reads output from (0,0,0,0) instead of its immediate predecessor
l := network.GetLayer(0, 1, 0, 0)
l.IsRemoteLink = true
l.TargetZ, l.TargetY, l.TargetX, l.TargetL = 0, 0, 0, 0
```

---

## Systolic (Continuous) Operation

```go
state := poly.NewSystolicState[float32](network)
state.SetInput(inputTensor)

for tick := 0; tick < 1000; tick++ {
    poly.SystolicForward(network, state, false)  // false = no history
    // read current output from state.LayerData[lastLayerIdx]
}

// Online learning (no history required)
poly.SystolicApplyTargetProp(network, state, targetTensor, 0.001)
```

---

## Systolic with BPTT (Training)

```go
state := poly.NewSystolicState[float32](network)
state.SetInput(inputTensor)

for tick := 0; tick < numSteps; tick++ {
    poly.SystolicForward(network, state, true)  // true = capture history
}

gradIn, layerGrads, err := poly.SystolicBackward(network, state, gradOutput)
poly.ApplyRecursiveGradients[float32](network, layerGrads, lr)
```

---

## DNA Comparison

```go
// Snapshot before training
dna1 := poly.ExtractDNA(network)

// Train ...
poly.Train[float32](network, data, config)

// Snapshot after training
dna2 := poly.ExtractDNA(network)

result := poly.CompareNetworks(dna1, dna2)
fmt.Printf("Similarity: %.4f\n", result.OverallOverlap)
for _, shift := range result.LogicShifts {
    fmt.Printf("Logic migrated: %s → %s (%.3f)\n",
        shift.SourcePos, shift.TargetPos, shift.Overlap)
}
```

---

## GPU Initialization

```go
network.UseGPU = true
ctx, err := poly.InitWGPU()
if err != nil { log.Fatal("GPU init failed:", err) }
network.GPUContext = ctx

// Sync all layer weights to VRAM
for i := range network.Layers {
    network.Layers[i].SyncToGPU()
}

// GPU batch training
config := poly.TrainingConfig{UseGPU: true, LearningRate: 0.001, Epochs: 100}
result := poly.Train[float32](network, data, config)
```

---

## Transformer Inference

```go
transformer := poly.NewTransformer[float32](
    network,
    embeddingWeights,
    lmHeadWeights,
    finalNormWeights,
    chatTemplate,
)
transformer.EnableTiling(0)  // auto tile size

output := transformer.Generate(
    tokenizer.Encode,
    tokenizer.Decode,
    []poly.Turn{},  // no history
    "You are a helpful assistant.",
    "What is 2 + 2?",
    poly.GenOptions{
        MaxTokens:   256,
        Temperature: 0.7,
        TopK:        40,
    },
)
fmt.Println(output)
```

---

## Softmax Variants

```go
// Temperature softmax
l.Type          = poly.LayerSoftmax
l.SoftmaxType   = poly.SoftmaxTemperature
l.Temperature   = 0.5

// Masked softmax (causal)
l.SoftmaxType   = poly.SoftmaxMasked
l.Mask          = []bool{true, true, false, false}  // mask out last 2

// Sparse (exact zeros)
l.SoftmaxType   = poly.SoftmaxSparse

// Entmax (tunable sparsity)
l.SoftmaxType   = poly.SoftmaxEntmax
l.EntmaxAlpha   = 1.5
```

---

## Q4_0 Block Quantization

```go
// Quantize a weight slice into 32-weight blocks
blocks := poly.QuantizeQ4_0(myWeights)
// blocks[i].Scale   = per-block float32 scale
// blocks[i].Weights = [16]byte with 32 packed nibbles

// Dequantize back to float32
recovered := poly.DequantizeQ4_0(blocks, len(myWeights))
```

---

## DType / Activation / LayerType Parsing

```go
// From string (case-insensitive, aliases accepted)
dtype, err      := poly.ParseDType("int8")       // → DTypeInt8
activation, err := poly.ParseActivationType("relu") // → ActivationReLU
layerType, err  := poly.ParseLayerType("Dense")   // → LayerDense
```

---

## TargetProp (Layer-Local Learning)

```go
tpConfig := poly.TargetPropConfig{
    UseChainRule: true,  // false = gap-based (for systolic meshes)
    LearningRate: 0.01,
}
tpState := poly.NewTargetPropState[float32](network)

// Forward + backward + weight update in one call
poly.TargetPropForward[float32](network, tpState, input)
poly.TargetPropBackward[float32](network, tpState, globalTarget)
poly.ApplyTargetPropGaps[float32](network, tpState, 0.01)
```

---

## Tensor Creation

```go
// 1D tensor
t1 := poly.NewTensor[float32](128)

// 2D tensor (e.g., [seqLen, hiddenSize])
t2 := poly.NewTensor[float32](16, 512)

// With initial data
t3 := poly.NewTensor[int8](8)
for i := range t3.Data { t3.Data[i] = int8(i) }

// Check shape
fmt.Println(t2.Shape)  // [16, 512]
fmt.Println(len(t2.Data))  // 8192
```
