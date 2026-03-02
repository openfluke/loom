# Dynamic Architecture Generation

`architecture.go` provides a programmatic API for generating, configuring, and building diverse neural network architectures. It is the foundation of Loom's Neural Architecture Search (NAS) and the evolutionary ensemble system (`test43a`).

---

## Core Concept: BrainType

Every layer in a "hive" architecture is called a **brain**. `BrainType` is an enum over all supported layer kinds:

| Constant | String | Layer Kind |
|---|---|---|
| `BrainMHA` | `"MHA"` | Multi-Head Attention |
| `BrainLSTM` | `"LSTM"` | Long Short-Term Memory |
| `BrainRNN` | `"RNN"` | Simple RNN |
| `BrainDense` | `"Dense"` | Fully-connected |
| `BrainSwiGLU` | `"SwiGLU"` | Gated MLP block |
| `BrainNormDense` | `"NormDense"` | Scaled Dense |
| `BrainConv2D` | `"Conv2D"` | 2D Convolution |
| `BrainConv1D` | `"Conv1D"` | 1D Convolution |
| `BrainSoftmax` | `"Softmax"` | Softmax |
| `BrainNorm` | `"Norm"` | LayerNorm |
| `BrainRMSNorm` | `"RMSNorm"` | RMSNorm |
| `BrainResidual` | `"Residual"` | Residual / skip |
| `BrainEmbedding` | `"Embedding"` | Token lookup |
| `BrainSequential` | `"Sequential"` | Sequential wrapper |
| `BrainParallel` | `"Parallel"` | Parallel combiner |

```go
bt := nn.BrainMHA
fmt.Println(bt.String())            // "MHA"
bt2 := nn.BrainTypeFromString("LSTM") // BrainLSTM
```

---

## ArchConfig: A Complete Architecture Blueprint

`ArchConfig` fully describes one network configuration — used to build AND track it across NAS runs:

```go
type ArchConfig struct {
    ID          int            // Unique ID in the search space
    Name        string         // Human-readable name (e.g. "Net-42")
    Species     string         // Grid topology name (e.g. "2x2 Standard")
    MutationStr string         // Compact fingerprint string
    GridRows    int            // Rows in parallel hive grid
    GridCols    int            // Cols in parallel hive grid
    NumBrains   int            // GridRows × GridCols
    DModel      int            // Model dimension
    NumHeads    int            // Attention heads (for MHA brains)
    LearningRate float32       // Assigned learning rate
    BudgetScale  float32       // Compute budget multiplier [0.5, 1.0]
    Activation   ActivationType
    CombineMode  string        // "concat", "add", "avg", "filter", "grid_scatter"
    Brains       []BrainType   // One per brain cell
    BrainNames   []string      // String names for JSON
    InitScale    float32       // Weight init scale
    DType        string        // "float32", "int8", etc.
}
```

The `MutationStr` is a compact fingerprint like:
```
2x2_filter_LeakyReLU_D64_float32_LR0.0023
```

Used for deduplication and logging across NAS runs.

---

## Building Networks

### Simple Single-Type Network

For benchmarking a single layer type:

```go
config := nn.SimpleNetworkConfig{
    LayerType:  nn.BrainLSTM,
    InputSize:  16,
    HiddenSize: 16,
    OutputSize: 1,
    Activation: nn.ActivationLeakyReLU,
    InitScale:  0.5,
    NumLayers:  2,
    DType:      nn.DTypeFloat32,
}

net := nn.BuildSimpleNetwork(config)
```

This builds: `Input → [LSTM] → [Dense] → [Dense(Sigmoid)] → Output`

Supported `LayerType` values for simple networks: `BrainDense`, `BrainConv2D`, `BrainRNN`, `BrainLSTM`, `BrainMHA`, `BrainSwiGLU`, `BrainNormDense`, `BrainConv1D`, `BrainResidual`.

### Diverse Hive Network (NAS)

For a full NAS-style network with a parallel hive:

```go
config := nn.ArchConfig{
    GridRows:    2,
    GridCols:    2,
    DModel:      64,
    NumHeads:    4,
    Brains:      []nn.BrainType{nn.BrainMHA, nn.BrainLSTM, nn.BrainDense, nn.BrainRNN},
    CombineMode: "filter",
    Activation:  nn.ActivationLeakyReLU,
    InitScale:   0.5,
}

net := nn.BuildDiverseNetwork(config, inputSize)
```

This builds a 4-layer network: `Input → [Parallel Hive: 4 brains] → [Merger] → [Output]`

The merger input size is automatically calculated from the combine mode:
- `"concat"` / `"grid_scatter"` → `DModel × NumBrains`
- `"add"` / `"avg"` / `"filter"` → `DModel`

---

## Grid Shapes

`GridShape` defines the topology of the parallel hive. Standard shapes are available:

```go
nn.StandardGridShapes = []nn.GridShape{
    {1, 1, "1x1 Mono"},
    {2, 2, "2x2 Standard"},
    {3, 3, "3x3 Complex"},
    {4, 1, "4x1 Tall"},
    {1, 4, "1x4 Wide"},
    {2, 3, "2x3 Rect"},
    {3, 2, "3x2 Rect"},
    {8, 1, "8x1 Scanner"},
    {6, 4, "6x4 Matrix"},
}
```

Each shape's `NumBrains()` = `Rows × Cols`.

---

## Random Architecture Generation (NAS)

Generate a batch of randomised configs for search:

```go
opts := nn.DefaultArchGenOptions()
// Default distributions:
//   Brain:   MHA=30%, LSTM=25%, RNN=15%, Dense=15%, SwiGLU=8%, NormDense=7%
//   Combine: avg=35%, add=30%, concat=20%, grid_scatter=15%
//   DModel:  75% D64, 25% D32
//   LR:      log-uniform in [0.0001, 0.01]

configs := nn.GenerateDiverseConfigs(100, opts)
```

Customise the distribution:

```go
opts := &nn.ArchGenOptions{
    DModels:     []int{128, 256},          // Only large models
    NumHeads:    []int{4, 8, 16},
    GridShapes:  nn.StandardGridShapes,
    LRMin:       0.00001,
    LRMax:       0.001,
    InitScale:   0.3,
    BudgetMin:   0.7,
    BudgetMax:   1.0,
    BrainDistribution: []float64{0.5, 0.3, 0.1, 0.1}, // Heavy MHA
    CombineDistribution: []float64{0.5, 0.3, 0.2, 0.0},
    DTypes:            []string{"float32", "float16"},
    DTypeDistribution: []float64{0.8, 0.2},
}
configs := nn.GenerateDiverseConfigs(500, opts)
```

### Permutation Grid (all layer × dtype combos)

For a comprehensive benchmark matrix:

```go
base := nn.DefaultSimpleConfig()

// All 5 standard layers × 10 dtypes = 50 configs
configs := nn.GenerateAllSimpleConfigs(base,
    nn.StandardBrainTypes,
    nn.StandardDTypeList,
)
```

---

## Individual Brain Init Functions

Create a single brain layer directly:

```go
// MHA: D=64, heads=4, init scale=0.5
mha := nn.InitMHABrain(64, 4, 0.5)           // returns LayerConfig

// LSTM: D=64, scale=0.5
lstm := nn.InitLSTMBrain(64, 0.5)

// RNN
rnn := nn.InitRNNBrain(64, 0.5)

// Dense with activation
dense := nn.InitDenseBrain(64, nn.ActivationLeakyReLU, 0.5)

// SwiGLU (weight scale slightly reduced)
swiglu := nn.InitSwiGLUBrain(64, 0.5)

// Norms
layernorm := nn.InitNormBrain(64, 0.5)
rmsnorm   := nn.InitRMSNormBrain(64, 0.5)

// Embedding
emb := nn.InitEmbeddingBrain(32000, 64, 0.02)

// Residual (no weights)
res := nn.InitResidualBrain()
```

---

## ArchConfig Serialization

`ArchConfig` and `ArchConfigBundle` are JSON-serializable and WASM-compatible:

```go
// Serialize a single config
data, _ := config.ToBytes()
restored, _ := nn.ArchConfigFromBytes(data)

// Bundle multiple configs
bundle := &nn.ArchConfigBundle{
    Version: 1,
    Configs: configs,
}

// Save to file
bundle.SaveToFile("search_space.json")

// Load from file
loaded, _ := nn.LoadArchConfigBundle("search_space.json")

// Pretty-print for debugging
jsonStr, _ := bundle.ToJSON()
fmt.Println(jsonStr)
```

---

## DType Helpers

```go
// String ↔ DType enum
str := nn.DTypeToString(nn.DTypeFloat16)   // "float16"
dt  := nn.DTypeFromString("int8")          // DTypeInt8

// All supported DType values
nn.StandardDTypeList  // []DType{DTypeFloat32, DTypeFloat64, DTypeInt8, ...}
nn.StandardDTypes     // []string{"float32", "float64", "int8", ...}
```

---

## Connection to NAS

This module is the foundation for the NAS pipeline:

```
GenerateDiverseConfigs()    ← Sample architecture configs
       ↓
BuildDiverseNetwork()       ← Construct and train each
       ↓
KMeansCluster() (clustering.go) ← Group similar architectures
       ↓
GraftNetworks() (grafting.go)   ← Combine best experts
       ↓
ComputeSilhouetteScore()    ← Measure cluster quality
```

See [clustering.md](clustering.md) and [grafting.md](grafting.md) for the downstream steps.
