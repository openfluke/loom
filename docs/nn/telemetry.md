# Understanding Network Telemetry

This guide explains Loom's telemetry system—how to extract structural information from a network, understand parameter counts, and inspect layer configurations programmatically.

---

## What is Telemetry?

Telemetry gives you a **structural X-ray** of your network:

```
┌─────────────────────────────────────────────────────────────────┐
│                       NETWORK TELEMETRY                         │
│                                                                 │
│  Model ID: "my-classifier"                                      │
│  Total Layers: 8                                                │
│  Total Parameters: 2,359,312                                    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Layer  │ Type    │ Grid Pos │ Input    │ Output  │ Params  ││
│  ├────────┼─────────┼──────────┼──────────┼─────────┼─────────┤│
│  │ 0      │ Dense   │ (0,0,0)  │ [1024]   │ [512]   │ 524,800 ││
│  │ 1      │ Dense   │ (0,0,1)  │ [512]    │ [256]   │ 131,328 ││
│  │ 2      │ Attn    │ (0,1,0)  │ [10,256] │ [10,256]│ 525,312 ││
│  │ 3      │ Dense   │ (0,1,1)  │ [256]    │ [128]   │ 32,896  ││
│  │ ...    │ ...     │ ...      │ ...      │ ...     │ ...     ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

This is essential for:
- Debugging architecture issues
- Validating model sizes before deployment
- Comparing network structures
- Building model inspection UIs

---

## NetworkBlueprint

The main telemetry structure:

```go
type NetworkBlueprint struct {
    Models []ModelTelemetry `json:"models"`
}
```

A blueprint can contain multiple models (useful for bundles with encoder-decoder pairs).

## ModelTelemetry

Describes a single network:

```go
type ModelTelemetry struct {
    ID          string           `json:"id"`               // Model identifier
    TotalLayers int              `json:"total_layers"`     // Number of layers
    TotalParams int              `json:"total_parameters"` // Sum of all parameters
    Layers      []LayerTelemetry `json:"layers"`           // Per-layer info
}
```

## LayerTelemetry

Details about each layer:

```go
type LayerTelemetry struct {
    // Grid position
    GridRow   int `json:"grid_row"`    // Row in Loom grid
    GridCol   int `json:"grid_col"`    // Column in Loom grid
    CellLayer int `json:"cell_layer"`  // Layer within cell

    // Layer info
    Type       string `json:"type"`                 // "Dense", "Conv2D", etc.
    Activation string `json:"activation,omitempty"` // "ReLU", "Tanh", etc.
    Parameters int    `json:"parameters"`           // Number of trainable params

    // Dimensions
    InputShape  []int `json:"input_shape,omitempty"`  // e.g., [1024]
    OutputShape []int `json:"output_shape,omitempty"` // e.g., [512]

    // For Parallel layers
    Branches    []LayerTelemetry `json:"branches,omitempty"`    // Nested branches
    CombineMode string           `json:"combine_mode,omitempty"` // "concat", etc.
}
```

---

## Extracting Telemetry

### From a Network

```go
network := nn.NewNetwork(1024, 2, 2, 3)
// ... configure layers ...

// Extract telemetry
telemetry := nn.ExtractNetworkBlueprint(network, "my-classifier")

fmt.Printf("Model: %s\n", telemetry.ID)
fmt.Printf("Total Layers: %d\n", telemetry.TotalLayers)
fmt.Printf("Total Parameters: %d\n", telemetry.TotalParams)
```

### Inspecting Layers

```go
for _, layer := range telemetry.Layers {
    fmt.Printf("[%d,%d,%d] %s: %v → %v (%d params)\n",
        layer.GridRow, layer.GridCol, layer.CellLayer,
        layer.Type,
        layer.InputShape, layer.OutputShape,
        layer.Parameters)
}
```

Output:
```
[0,0,0] Dense: [1024] → [512] (524800 params)
[0,0,1] Dense: [512] → [256] (131328 params)
[0,0,2] Dense: [256] → [128] (32896 params)
[0,1,0] MultiHeadAttention: [10 256] → [10 256] (525312 params)
[0,1,1] Dense: [256] → [64] (16448 params)
[0,1,2] Softmax: [] → [] (0 params)
[1,0,0] LSTM: [10 64] → [10 32] (24704 params)
[1,0,1] Dense: [32] → [10] (330 params)
[1,0,2] Softmax: [] → [] (0 params)
...
```

---

## Parameter Counting

Telemetry automatically counts parameters for each layer type:

### Dense Layer

```
Parameters = (input_size × output_size) + output_size
                    ↑                        ↑
              Weight matrix              Bias vector

Example: Dense 1024 → 512
    Weights: 1024 × 512 = 524,288
    Biases:  512
    Total:   524,800 parameters
```

### Conv2D Layer

```
Parameters = (filters × channels × kernel² ) + filters
                         ↑                       ↑
                  Weight kernels              Biases

Example: Conv2D 16 filters, 3 channels, 3×3 kernel
    Weights: 16 × 3 × 3 × 3 = 432
    Biases:  16
    Total:   448 parameters
```

### Multi-Head Attention

```
Parameters = Q + K + V + Output projections + Biases

Q projection: dModel × dModel
K projection: kv_dim × dModel
V projection: kv_dim × dModel
Output:       dModel × dModel

Example: dModel=256, numHeads=8, numKVHeads=4
    kv_dim = 4 × (256/8) = 128
    Q: 256 × 256 = 65,536
    K: 128 × 256 = 32,768
    V: 128 × 256 = 32,768
    O: 256 × 256 = 65,536
    Biases: 256 + 128 + 128 + 256 = 768
    Total: 197,376 parameters
```

### LSTM Layer

```
4 gates (Input, Forget, Cell, Output)
Each gate: input_weights + hidden_weights + bias

Parameters = 4 × [(input × hidden) + (hidden × hidden) + hidden]

Example: LSTM input=64, hidden=32
    Per gate: (64 × 32) + (32 × 32) + 32 = 3,104
    Total: 4 × 3,104 = 12,416 parameters
```

### RNN Layer

```
Parameters = (input × hidden) + (hidden × hidden) + hidden

Example: RNN input=64, hidden=32
    Input weights:  64 × 32 = 2,048
    Hidden weights: 32 × 32 = 1,024
    Bias:           32
    Total:          3,104 parameters
```

### Normalization Layers

```
LayerNorm:  Parameters = 2 × size (gamma + beta)
RMSNorm:    Parameters = size (gamma only)

Example: NormSize=256
    LayerNorm: 512 parameters
    RMSNorm:   256 parameters
```

### SwiGLU

```
3 projections: Gate, Up, Down

Gate: input × intermediate + bias
Up:   input × intermediate + bias
Down: intermediate × input + bias

Example: input=512, intermediate=2048
    Gate: 512 × 2048 + 2048 = 1,050,624
    Up:   512 × 2048 + 2048 = 1,050,624
    Down: 2048 × 512 + 512 = 1,049,088
    Total: 3,150,336 parameters
```

### Softmax Layer

```
Parameters = 0 (softmax has no trainable parameters)
```

### Parallel Layer

```
Parameters = sum of all branch parameters

Example: Parallel with 3 Dense branches
    Branch 0: 64 → 32 = 2,080
    Branch 1: 64 → 32 = 2,080
    Branch 2: 64 → 32 = 2,080
    Total: 6,240 parameters
```

---

## Grid Position Mapping

Telemetry includes grid coordinates for each layer:

```
Layer index → Grid position
    
    Grid: 2 rows × 2 columns × 3 layers per cell

    Index 0 → (0, 0, 0)  Cell (0,0), Layer 0
    Index 1 → (0, 0, 1)  Cell (0,0), Layer 1
    Index 2 → (0, 0, 2)  Cell (0,0), Layer 2
    Index 3 → (0, 1, 0)  Cell (0,1), Layer 0
    Index 4 → (0, 1, 1)  Cell (0,1), Layer 1
    Index 5 → (0, 1, 2)  Cell (0,1), Layer 2
    Index 6 → (1, 0, 0)  Cell (1,0), Layer 0
    ...

Calculation:
    gridRow   = index / (cols × layersPerCell)
    gridCol   = (index / layersPerCell) % cols
    cellLayer = index % layersPerCell
```

---

## Inspecting Parallel Layers

For parallel layers, telemetry includes branch information:

```go
for _, layer := range telemetry.Layers {
    if layer.Type == "Parallel" {
        fmt.Printf("Parallel layer at (%d,%d,%d):\n",
            layer.GridRow, layer.GridCol, layer.CellLayer)
        fmt.Printf("  Combine mode: %s\n", layer.CombineMode)
        fmt.Printf("  Branches:\n")
        
        for i, branch := range layer.Branches {
            fmt.Printf("    [%d] %s: %d params\n",
                i, branch.Type, branch.Parameters)
        }
    }
}
```

Output:
```
Parallel layer at (0,1,0):
  Combine mode: filter
  Branches:
    [0] Dense: 2080 params
    [1] LSTM: 12416 params
    [2] MultiHeadAttention: 197376 params
```

---

## JSON Export

Telemetry structures are JSON-serializable:

```go
import "encoding/json"

telemetry := nn.ExtractNetworkBlueprint(network, "my-model")

data, _ := json.MarshalIndent(telemetry, "", "  ")
fmt.Println(string(data))
```

Output:
```json
{
  "id": "my-model",
  "total_layers": 8,
  "total_parameters": 2359312,
  "layers": [
    {
      "grid_row": 0,
      "grid_col": 0,
      "cell_layer": 0,
      "type": "Dense",
      "activation": "ReLU",
      "parameters": 524800,
      "input_shape": [1024],
      "output_shape": [512]
    },
    {
      "grid_row": 0,
      "grid_col": 0,
      "cell_layer": 1,
      "type": "Dense",
      "activation": "ReLU",
      "parameters": 131328,
      "input_shape": [512],
      "output_shape": [256]
    },
    ...
  ]
}
```

---

## Use Cases

### Validation Before Deployment

```go
telemetry := nn.ExtractNetworkBlueprint(network, "production-model")

// Check size constraints
maxParams := 10_000_000  // 10M parameter budget
if telemetry.TotalParams > maxParams {
    log.Fatalf("Model too large: %d > %d parameters",
        telemetry.TotalParams, maxParams)
}

// Check layer types
for _, layer := range telemetry.Layers {
    if layer.Type == "LSTM" {
        log.Printf("Warning: LSTM found at (%d,%d,%d) - may be slow on mobile",
            layer.GridRow, layer.GridCol, layer.CellLayer)
    }
}
```

### Architecture Comparison

```go
tel1 := nn.ExtractNetworkBlueprint(network1, "v1")
tel2 := nn.ExtractNetworkBlueprint(network2, "v2")

fmt.Printf("v1: %d layers, %d params\n", tel1.TotalLayers, tel1.TotalParams)
fmt.Printf("v2: %d layers, %d params\n", tel2.TotalLayers, tel2.TotalParams)

// Find differences
for i := range tel1.Layers {
    if i >= len(tel2.Layers) {
        fmt.Printf("Layer %d: only in v1\n", i)
        continue
    }
    if tel1.Layers[i].Type != tel2.Layers[i].Type {
        fmt.Printf("Layer %d: %s vs %s\n", i,
            tel1.Layers[i].Type, tel2.Layers[i].Type)
    }
}
```

### Building Model Explorer UI

```go
// API endpoint for model inspection
http.HandleFunc("/api/model/info", func(w http.ResponseWriter, r *http.Request) {
    telemetry := nn.ExtractNetworkBlueprint(currentNetwork, "live-model")
    
    json.NewEncoder(w).Encode(telemetry)
})

// Frontend can then visualize:
// - Layer graph
// - Parameter distribution
// - Architecture diagram
```

### Memory Estimation

```go
telemetry := nn.ExtractNetworkBlueprint(network, "my-model")

// Estimate memory usage (float32 = 4 bytes)
bytesForParams := telemetry.TotalParams * 4
mbForParams := float64(bytesForParams) / 1024 / 1024

// Account for gradients (2x during training)
mbDuringTraining := mbForParams * 2

// Account for activations (rough estimate)
activationMB := estimateActivations(telemetry)

fmt.Printf("Memory estimate:\n")
fmt.Printf("  Parameters:  %.2f MB\n", mbForParams)
fmt.Printf("  Training:    %.2f MB (with gradients)\n", mbDuringTraining)
fmt.Printf("  Activations: %.2f MB (estimated)\n", activationMB)
```

---

## Summary

Telemetry provides:

**NetworkBlueprint**
- Complete structural snapshot
- Multi-model support (bundles)

**ModelTelemetry**
- Model ID and totals
- Per-layer breakdown

**LayerTelemetry**
- Grid positions
- Type and activation
- Parameter counts
- Input/output shapes
- Nested branch info

Use telemetry to:
- Validate architectures
- Compare models
- Estimate resources
- Build inspection tools
- Debug layer configurations
