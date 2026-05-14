package poly

import (
	"fmt"
	"math"
	"reflect"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// LayerType defines the type of neural network layer
type LayerType int

const (
	LayerDense              LayerType = 0
	LayerMultiHeadAttention LayerType = 1
	LayerSwiGLU             LayerType = 2
	LayerRMSNorm            LayerType = 3
	LayerCNN1               LayerType = 4
	LayerCNN2               LayerType = 5
	LayerCNN3               LayerType = 6
	LayerRNN                LayerType = 7
	LayerLSTM               LayerType = 8
	LayerLayerNorm          LayerType = 9
	LayerConvTransposed1D   LayerType = 10
	LayerConvTransposed2D   LayerType = 11
	LayerConvTransposed3D   LayerType = 12
	LayerEmbedding          LayerType = 13
	LayerKMeans             LayerType = 14
	LayerSoftmax            LayerType = 15
	LayerParallel           LayerType = 16
	LayerSequential         LayerType = 17
	LayerResidual           LayerType = 18
	LayerMetacognition      LayerType = 19
)

func (t LayerType) String() string {
	switch t {
	case LayerDense:
		return "Dense"
	case LayerMultiHeadAttention:
		return "MultiHeadAttention"
	case LayerSwiGLU:
		return "SwiGLU"
	case LayerRMSNorm:
		return "RMSNorm"
	case LayerCNN1:
		return "CNN1"
	case LayerCNN2:
		return "CNN2"
	case LayerCNN3:
		return "CNN3"
	case LayerRNN:
		return "RNN"
	case LayerLSTM:
		return "LSTM"
	case LayerLayerNorm:
		return "LayerNorm"
	case LayerConvTransposed1D:
		return "ConvTransposed1D"
	case LayerConvTransposed2D:
		return "ConvTransposed2D"
	case LayerConvTransposed3D:
		return "ConvTransposed3D"
	case LayerEmbedding:
		return "Embedding"
	case LayerKMeans:
		return "KMeans"
	case LayerSoftmax:
		return "Softmax"
	case LayerParallel:
		return "Parallel"
	case LayerSequential:
		return "Sequential"
	case LayerResidual:
		return "Residual"
	case LayerMetacognition:
		return "Metacognition"
	default:
		return fmt.Sprintf("LayerType(%d)", t)
	}
}

// ActivationType defines the activation function
type ActivationType int

const (
	ActivationReLU      ActivationType = 0
	ActivationSilu      ActivationType = 1
	ActivationGELU      ActivationType = 2
	ActivationTanh      ActivationType = 3
	ActivationSigmoid   ActivationType = 4
	ActivationLeakyReLU ActivationType = 5
	ActivationReLU2     ActivationType = 6
	ActivationLinear    ActivationType = -1
)

// SoftmaxType defines the variant of softmax to use
type SoftmaxType int

const (
	SoftmaxStandard     SoftmaxType = 0
	SoftmaxGrid         SoftmaxType = 1
	SoftmaxHierarchical SoftmaxType = 2
	SoftmaxTemperature  SoftmaxType = 3
	SoftmaxGumbel       SoftmaxType = 4
	SoftmaxMasked       SoftmaxType = 5
	SoftmaxSparse       SoftmaxType = 6
	SoftmaxAdaptive     SoftmaxType = 7
	SoftmaxMixture      SoftmaxType = 8
	SoftmaxEntmax       SoftmaxType = 9
)

func (a ActivationType) String() string {
	switch a {
	case ActivationReLU:
		return "ReLU"
	case ActivationSilu:
		return "Silu"
	case ActivationGELU:
		return "GELU"
	case ActivationTanh:
		return "Tanh"
	case ActivationSigmoid:
		return "Sigmoid"
	case ActivationLeakyReLU:
		return "LeakyReLU"
	case ActivationReLU2:
		return "ReLU2"
	case ActivationLinear:
		return "Linear"
	default:
		return "Linear"
	}
}

func (s SoftmaxType) String() string {
	switch s {
	case SoftmaxStandard:
		return "Standard"
	case SoftmaxGrid:
		return "Grid"
	case SoftmaxHierarchical:
		return "Hierarchical"
	case SoftmaxTemperature:
		return "Temperature"
	case SoftmaxGumbel:
		return "Gumbel"
	case SoftmaxMasked:
		return "Masked"
	case SoftmaxSparse:
		return "Sparse"
	case SoftmaxAdaptive:
		return "Adaptive"
	case SoftmaxMixture:
		return "Mixture"
	case SoftmaxEntmax:
		return "Entmax"
	default:
		return "Standard"
	}
}

// Activate applies the activation function to a value.
func Activate[T Numeric](v T, act ActivationType) T {
	switch act {
	case ActivationReLU:
		if v < 0 {
			return 0
		}
		return v
	case ActivationSilu:
		// x * sigmoid(x)
		return v * T(1.0/(1.0+math.Exp(-float64(v))))
	case ActivationGELU:
		// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
		v64 := float64(v)
		return T(0.5 * v64 * (1.0 + math.Tanh(math.Sqrt(2.0/math.Pi)*(v64+0.044715*math.Pow(v64, 3)))))
	case ActivationTanh:
		return T(math.Tanh(float64(v)))
	case ActivationSigmoid:
		return T(1.0 / (1.0 + math.Exp(-float64(v))))
	case ActivationLeakyReLU:
		if v < 0 {
			return T(float64(v) * 0.01)
		}
		return v
	case ActivationReLU2:
		if v < 0 {
			return 0
		}
		return v * v
	case ActivationLinear:
		return v
	default:
		return v
	}
}

// ActivateDerivative returns the derivative of the activation function.
func ActivateDerivative[T Numeric](v T, act ActivationType) T {
	switch act {
	case ActivationReLU:
		if v <= 0 {
			return 0
		}
		return 1
	case ActivationSilu:
		// SiLU(x) = x * sig(x)
		// d/dx = sig(x) + x * sig(x) * (1 - sig(x)) = sig(x) * (1 + x*(1-sig(x)))
		x := float64(v)
		sig := 1.0 / (1.0 + math.Exp(-x))
		return T(sig * (1.0 + x*(1.0-sig)))
	case ActivationGELU:
		// Approximate derivative: 0.5 * (1 + erf(x/sqrt(2))) + (x/sqrt(2*pi)) * exp(-x^2/2)
		// For simplicity using the tanh approximation derivative form:
		v64 := float64(v)
		cdf := 0.5 * (1.0 + math.Tanh(math.Sqrt(2.0/math.Pi)*(v64+0.044715*math.Pow(v64, 3))))
		pdf := math.Exp(-0.5*v64*v64) / math.Sqrt(2.0*math.Pi)
		return T(cdf + v64*pdf)
	case ActivationTanh:
		t := math.Tanh(float64(v))
		return T(1.0 - t*t)
	case ActivationSigmoid:
		s := 1.0 / (1.0 + math.Exp(-float64(v)))
		return T(s * (1.0 - s))
	case ActivationLeakyReLU:
		if v <= 0 {
			return T(1) / 100
		}
		return T(1)
	case ActivationReLU2:
		if v <= 0 {
			return 0
		}
		return 2 * v
	case ActivationLinear:
		return 1
	default:
		return 1
	}
}

/*
M-POLY-VTD: Multi-numerical POLYmorphic Volumetric Tiled-tensor Dispatcher
--------------------------------------------------------------------------
An asynchronous 3D coordinate-based inference engine designed to approximate
biological neural firing through spatial layer-hopping and real-time
VRAM-resident numerical metamorphosis.

I. MULTI-NUMERICAL ARCHITECTURE (The "M" in M-POLY)
--------------------------------------------------
This engine supports native forward/backward passes across diverse numerical
types (FP32, FP16, INT8, and FP4 E2M1) directly on the GPU.

1. Bandwidth Optimization (The 192 GB/s Wall):
   - Targets a 75-80% reduction in weight size via low-bit quantization.
   - Specifically optimized for Turing (GTX 1650 Super) memory constraints,
     where global memory reads are the primary bottleneck for SmolLM2-135M.

2. In-VRAM Metamorphosis:
   - Supports mid-stream precision shifts managed entirely in VRAM. A layer can
     synchronize its multi-numerical representations (e.g., FP32 high-precision
     and INT8 inference-optimized) without CPU intervention.
   - Training and inference are unified; the GPU maintains the low-bit
     state for throughput and the high-precision state for accumulation.

3. Hardware-Aware Dispatching:
   - Since Turing lacks native FP4 Tensor Cores, the "Multi-Numerical"
     bus handles vectorized unpacking (Stage 3 optimization). It treats
     low-bit types as "packed payloads" to be expanded in-shader,
     reclaiming bandwidth while maintaining high-fidelity computation.

II. POLYMORPHIC LAYER-MORPHING (The "POLY")
-------------------------------------------
- Compartmentalization: Every layer is a polymorphic unit that can instantly
  switch between active numerical representations (e.g., FP32 -> INT8)
  using the WeightStore's GPU-resident versioning system.
- Direct-on-Quant: Bypasses traditional FP32 master weight dependencies
  for execution, allowing forward passes to run natively on bit-packed
  payloads while the backward pass manages gradient updates.

III. VOLUMETRIC TENSOR DISPATCH (The "VTD")
-------------------------------------------
- 3D Grid Representation: Replaces the 1D sequential stack with a
  (Row, Col, Layer) coordinate system.
- Spatial Hopping: Enables (0,0,0) recursive passing, simulating the
  recursive feedback loops of biological brains through spatial layer-hopping.
- Tiling Strategy (The Stage 2/4 Win): Each 3D coordinate maps to a
  GPU workgroup tile. By keeping the "Volumetric Tile" in Shared Memory,
  we avoid redundant global reads, aiming for the 70 tok/s performance ceiling.
*/

// DType defines the numerical type stored in a Tensor or WeightStore
type DType int

const (
	DTypeFloat64  DType = 0  // 64-bit double
	DTypeFloat32  DType = 1  // Standard 32-bit float
	DTypeFloat16  DType = 2  // 16-bit float
	DTypeBFloat16 DType = 3  // 16-bit Brain Float
	DTypeFP8E4M3  DType = 4  // 8-bit FP8 (E4M3)
	DTypeFP8E5M2  DType = 5  // 8-bit FP8 (E5M2)
	DTypeInt64    DType = 6  // 64-bit integer
	DTypeInt32    DType = 7  // 32-bit integer
	DTypeInt16    DType = 8  // 16-bit integer
	DTypeInt8     DType = 9  // 8-bit integer
	DTypeUint64   DType = 10 // 64-bit unsigned
	DTypeUint32   DType = 11 // 32-bit unsigned
	DTypeUint16   DType = 12 // 16-bit unsigned
	DTypeUint8    DType = 13 // 8-bit unsigned
	DTypeInt4     DType = 14 // 4-bit integer
	DTypeUint4    DType = 15 // 4-bit unsigned
	DTypeFP4      DType = 16 // 4-bit E2M1
	DTypeInt2     DType = 17 // 2-bit integer
	DTypeUint2    DType = 18 // 2-bit unsigned
	DTypeTernary  DType = 19 // 2-bit (Ternary: -1, 0, 1)
	DTypeBinary   DType = 20 // 1-bit (XNOR-Net)

	// Sub-weight identifiers for specific layer types (used as keys in GPUWeights map)
	WeightMHAQuery        DType = 200
	WeightMHAKey          DType = 201
	WeightMHAValue        DType = 202
	WeightMHAProjection   DType = 203
	WeightMHAQNorm        DType = 204
	WeightMHAKNorm        DType = 205
	WeightMHAInnerNorm    DType = 206
	WeightSwiGLUInnerNorm DType = 120
)

func (d DType) String() string {
	switch d {
	case DTypeFloat64:
		return "float64"
	case DTypeFloat32:
		return "float32"
	case DTypeFloat16:
		return "float16"
	case DTypeBFloat16:
		return "bfloat16"
	case DTypeFP8E4M3:
		return "fp8e4m3"
	case DTypeFP8E5M2:
		return "fp8e5m2"
	case DTypeInt64:
		return "int64"
	case DTypeInt32:
		return "int32"
	case DTypeInt16:
		return "int16"
	case DTypeInt8:
		return "int8"
	case DTypeUint64:
		return "uint64"
	case DTypeUint32:
		return "uint32"
	case DTypeUint16:
		return "uint16"
	case DTypeUint8:
		return "uint8"
	case DTypeInt4:
		return "int4"
	case DTypeUint4:
		return "uint4"
	case DTypeFP4:
		return "fp4"
	case DTypeInt2:
		return "int2"
	case DTypeUint2:
		return "uint2"
	case DTypeTernary:
		return "ternary"
	case DTypeBinary:
		return "binary"
	default:
		return fmt.Sprintf("DType(%d)", d)
	}
}

// DTypeBits returns the number of bits used for the given numerical type.
func DTypeBits(d DType) int {
	switch d {
	case DTypeFloat64, DTypeInt64, DTypeUint64:
		return 64
	case DTypeFloat32, DTypeInt32, DTypeUint32:
		return 32
	case DTypeFloat16, DTypeBFloat16, DTypeInt16, DTypeUint16:
		return 16
	case DTypeFP8E4M3, DTypeFP8E5M2, DTypeInt8, DTypeUint8:
		return 8
	case DTypeInt4, DTypeUint4, DTypeFP4:
		return 4
	case DTypeInt2, DTypeUint2, DTypeTernary:
		return 2
	case DTypeBinary:
		return 1
	default:
		return 32
	}
}

// Numeric is a type constraint for all numeric types that Tensors can hold.
type Numeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64
}

// Tensor wraps numerical data with metadata.
type Tensor[T Numeric] struct {
	Data   []T
	DType  DType
	Shape  []int
	Nested []*Tensor[T] // For recursive activation caching in Parallel/Sequential layers
}

// NewTensor creates a new tensor with the given shape.
func NewTensor[T Numeric](shape ...int) *Tensor[T] {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return &Tensor[T]{
		Data:  make([]T, size),
		Shape: shape,
	}
}

// NewTensorFromSlice creates a tensor from existing data.
func NewTensorFromSlice[T Numeric](data []T, shape ...int) *Tensor[T] {
	return &Tensor[T]{
		Data:  data,
		Shape: shape,
	}
}

// Clone creates a deep copy of the tensor.
func (t *Tensor[T]) Clone() *Tensor[T] {
	if t == nil {
		return nil
	}
	newData := make([]T, len(t.Data))
	copy(newData, t.Data)
	var nested []*Tensor[T]
	if len(t.Nested) > 0 {
		nested = make([]*Tensor[T], len(t.Nested))
		for i, n := range t.Nested {
			nested[i] = n.Clone()
		}
	}
	return &Tensor[T]{
		Data:   newData,
		DType:  t.DType,
		Shape:  t.Shape,
		Nested: nested,
	}
}

// HasInvalid returns true if any element in the tensor is NaN or Inf.
func (t *Tensor[T]) HasInvalid() bool {
	if t == nil {
		return false
	}
	for _, v := range t.Data {
		v64 := float64(v)
		if math.IsNaN(v64) || math.IsInf(v64, 0) {
			return true
		}
	}
	for _, n := range t.Nested {
		if n.HasInvalid() {
			return true
		}
	}
	return false
}

// Add adds another tensor's data to this one (in-place).
func (t *Tensor[T]) Add(other *Tensor[T]) {
	if t == nil || other == nil || len(t.Data) != len(other.Data) {
		return
	}
	for i := range t.Data {
		t.Data[i] += other.Data[i]
	}
	// Recursively add nested tensors
	if len(t.Nested) > 0 && len(other.Nested) == len(t.Nested) {
		for i := range t.Nested {
			t.Nested[i].Add(other.Nested[i])
		}
	}
}

// ConvertTensor converts a tensor from one numeric type to another.
func ConvertTensor[In Numeric, Out Numeric](in *Tensor[In]) *Tensor[Out] {
	if in == nil {
		return nil
	}
	outData := make([]Out, len(in.Data))
	for i, v := range in.Data {
		outData[i] = Out(v)
	}

	var nested []*Tensor[Out]
	if len(in.Nested) > 0 {
		nested = make([]*Tensor[Out], len(in.Nested))
		for i, n := range in.Nested {
			nested[i] = ConvertTensor[In, Out](n)
		}
	}

	return &Tensor[Out]{
		Data:   outData,
		Shape:  in.Shape,
		DType:  in.DType,
		Nested: nested,
	}
}

// VolumetricNetwork represents a 3D grid neural network.
type VolumetricNetwork struct {
	Depth         int
	Rows          int
	Cols          int
	LayersPerCell int

	Layers []VolumetricLayer

	// Global Tiling & GPU Switches
	UseTiling             bool
	EnableMultiCoreTiling bool
	UseGPU                bool
	UseExactDType         bool

	// GPU Acceleration context
	GPUContext *WGPUContext

	// Persistent GPU buffers to avoid allocations
	GPUHiddenState []any // map[DType]wgpu.Buffer or similar, use any for now
	GPULogits      any   // wgpu.Buffer

	GPUEmbeddings any // *wgpu.Buffer
	GPULMHead     any // *wgpu.Buffer

	// Tanhi (see docs/tanhi.md) enables sparse UDP layer telemetry; nil = off.
	Tanhi *TanhiUDPConfig
}

// VolumetricLayer represents a processing unit in the 3D volumetric grid.
type VolumetricLayer struct {
	Network     *VolumetricNetwork
	Type        LayerType
	Activation  ActivationType
	DType       DType
	WeightStore *WeightStore
	IsDisabled  bool

	// 3D Coordinates
	Z int // Depth
	Y int // Row
	X int // Col
	L int // Layer index within cell

	// Config (Expanding from LayerConfig)
	InputHeight   int
	InputWidth    int
	InputDepth    int
	OutputHeight  int
	OutputWidth   int
	OutputDepth   int
	InputChannels int
	Filters       int
	KernelSize    int
	Stride        int
	Padding       int
	OutputPadding int

	NumHeads     int
	NumKVHeads   int
	HeadDim      int
	QueryDim     int
	DModel       int
	SeqLength    int
	RoPEFreqBase float64
	RMSNormEps   float64

	// Optional per-head RMSNorm scales used by Qwen-style attention.
	QNormWeight []float32
	KNormWeight []float32

	// Optional BitNet inner RMSNorm scales used inside attention and MLP blocks.
	InnerNormWeight []float32

	VocabSize    int
	EmbeddingDim int

	NumClusters       int
	KMeansTemperature float64
	KMeansOutputMode  string // "probabilities" or "features"

	SoftmaxType     SoftmaxType
	Temperature     float64
	SoftmaxRows     int
	SoftmaxCols     int
	HierarchyLevels []int
	EntmaxAlpha     float64
	Mask            []bool
	GumbelNoise     bool

	ParallelBranches []VolumetricLayer
	CombineMode      string // "concat", "add", "avg", "filter", "grid_scatter"
	FilterGateConfig *VolumetricLayer

	// Spatial Routing (Remote Links)
	IsRemoteLink bool
	TargetZ      int
	TargetY      int
	TargetX      int
	TargetL      int

	SequentialLayers []VolumetricLayer

	// Tiling & GPU Config
	UseTiling             bool
	EnableMultiCoreTiling bool
	TileSize              int           // legacy fallback; prefer per-dtype maps below
	CPUTileSizes          map[DType]int // CPU tile size per numerical type
	GPUSCTileSizes        map[DType]int // GPU single-core tile size per numerical type
	GPUMCTileSizes        map[DType]int // GPU multi-core tile size per numerical type
	UseGPU                bool

	IsGPUResident        bool
	IsKVCacheGPUResident bool

	Observer PolyObserver

	// KV Cache (for MHA)
	KVCacheK  any
	KVCacheV  any
	KVOffset  int
	MaxSeqLen int

	// Persistent GPU KV buffers
	GPUKVCacheK any // *wgpu.Buffer
	GPUKVCacheV any // *wgpu.Buffer

	// Metacognition (Sub-network)
	MetaNetwork       *VolumetricNetwork
	MetaSource        string // "input", "stats", "activations", "weights", "combined"
	MetaSourceLayer   int    // Optional: Index of layer to observe (-1 for self/input)
	MetaEffect        string // "gate", "residual", "weight_modulation", "select_branch"
	MetaRules         []MetaRule
	MetaObservedLayer *VolumetricLayer // Optional: Direct reference to observed layer

	// gpuParScratch holds nested forward buffers for one GPU batch when Type==LayerParallel (training).
	gpuParScratch *gpuParallelScratch
}

// ResetState clears persistent internal state of the layer (e.g. KV caches).
func (l *VolumetricLayer) ResetState() {
	l.KVOffset = 0
	l.KVCacheK = nil
	l.KVCacheV = nil
	// Note: We don't null out GPU base buffers here to avoid expensive re-allocation
	// unless specifically requested, but the offset reset will force overwriting.
}

// AlignedFloat32 allocates a slice of float32 aligned to 64-byte boundaries.
func AlignedFloat32(n int) []float32 {
	const align = 64
	b := make([]byte, n*4+align)
	ptr := uintptr(unsafe.Pointer(&b[0]))
	offset := uintptr(0)
	if ptr%uintptr(align) != 0 {
		offset = uintptr(align) - (ptr % uintptr(align))
	}
	var res []float32
	header := (*reflect.SliceHeader)(unsafe.Pointer(&res))
	header.Data = ptr + offset
	header.Len = n
	header.Cap = n
	return res
}

// NewVolumetricNetwork initializes a 3D grid of layers.
func NewVolumetricNetwork(depth, rows, cols, layersPerCell int) *VolumetricNetwork {
	total := depth * rows * cols * layersPerCell
	layers := make([]VolumetricLayer, total)

	n := &VolumetricNetwork{
		Depth:         depth,
		Rows:          rows,
		Cols:          cols,
		LayersPerCell: layersPerCell,
		Layers:        layers,
	}

	// Initialize default positions
	for z := 0; z < depth; z++ {
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				for l := 0; l < layersPerCell; l++ {
					idx := n.GetIndex(z, y, x, l)
					layers[idx] = VolumetricLayer{
						Network:    n,
						Type:       LayerDense,
						Activation: ActivationReLU,
						DType:      DTypeFloat32,
						Z:          z,
						Y:          y,
						X:          x,
						L:          l,
					}
				}
			}
		}
	}

	return n
}

// GetIndex calculates the flattened index for a 3D coordinate.
func (n *VolumetricNetwork) GetIndex(z, y, x, l int) int {
	return (z * n.Rows * n.Cols * n.LayersPerCell) + (y * n.Cols * n.LayersPerCell) + (x * n.LayersPerCell) + l
}

// SyncToGPU mirrors all layers to the GPU.
func (n *VolumetricNetwork) SyncToGPU() error {
	for i := range n.Layers {
		if err := n.Layers[i].SyncToGPU(); err != nil {
			return err
		}
	}
	return nil
}

// Release releases all GPU resources (weights) for the network.
func (n *VolumetricNetwork) Release() {
	for i := range n.Layers {
		n.Layers[i].Release()
	}

	if buf, ok := n.GPUEmbeddings.(*wgpu.Buffer); ok && buf != nil {
		buf.Release()
	}
	if buf, ok := n.GPULMHead.(*wgpu.Buffer); ok && buf != nil {
		if n.GPULMHead != n.GPUEmbeddings {
			buf.Release()
		}
	}
	n.GPUEmbeddings = nil
	n.GPULMHead = nil

	if n.GPUContext != nil {
		n.GPUContext.Release()
	}
}

// Release releases GPU weight buffers for this layer.
func (l *VolumetricLayer) Release() {
	for i := range l.ParallelBranches {
		l.ParallelBranches[i].Release()
	}
	for i := range l.SequentialLayers {
		l.SequentialLayers[i].Release()
	}

	if l.WeightStore != nil {
		l.WeightStore.Release()
	}

	if buf, ok := l.GPUKVCacheK.(*wgpu.Buffer); ok && buf != nil {
		buf.Release()
	}
	if buf, ok := l.GPUKVCacheV.(*wgpu.Buffer); ok && buf != nil {
		buf.Release()
	}
	l.GPUKVCacheK = nil
	l.GPUKVCacheV = nil
	l.IsGPUResident = false
	l.IsKVCacheGPUResident = false
}

// SyncToCPU prepares the network for multi-core CPU execution by calculating optimal tiling parameters.
func (n *VolumetricNetwork) SyncToCPU() {
	for i := range n.Layers {
		n.Layers[i].SyncToCPU()
	}
}

// GetLayer returns the layer at specific 3D coordinates.
func (n *VolumetricNetwork) GetLayer(z, y, x, l int) *VolumetricLayer {
	idx := n.GetIndex(z, y, x, l)
	if idx >= 0 && idx < len(n.Layers) {
		return &n.Layers[idx]
	}
	return nil
}

func volumetricLayerCPUSizeRecursive(l *VolumetricLayer) int {
	if l == nil {
		return 0
	}
	n := 0
	if l.WeightStore != nil {
		n += l.WeightStore.SizeInBytes(l.DType)
	}
	for i := range l.ParallelBranches {
		n += volumetricLayerCPUSizeRecursive(&l.ParallelBranches[i])
	}
	for i := range l.SequentialLayers {
		n += volumetricLayerCPUSizeRecursive(&l.SequentialLayers[i])
	}
	return n
}

func volumetricLayerVRAMWeightsRecursive(l *VolumetricLayer) int64 {
	if l == nil {
		return 0
	}
	type sizer interface{ GetSize() uint64 }
	var total int64
	if l.WeightStore != nil {
		for _, wAny := range l.WeightStore.GPUWeights {
			if buf, ok := wAny.(sizer); ok && buf != nil {
				total += int64(buf.GetSize())
			}
		}
		for _, sBuf := range l.WeightStore.GPUScales {
			if sBuf != nil {
				total += int64(sBuf.GetSize())
			}
		}
	}
	for i := range l.ParallelBranches {
		total += volumetricLayerVRAMWeightsRecursive(&l.ParallelBranches[i])
	}
	for i := range l.SequentialLayers {
		total += volumetricLayerVRAMWeightsRecursive(&l.SequentialLayers[i])
	}
	return total
}

func volumetricLayerVRAMKVRecursive(l *VolumetricLayer) int64 {
	if l == nil {
		return 0
	}
	type sizer interface{ GetSize() uint64 }
	var total int64
	if buf, ok := l.GPUKVCacheK.(sizer); ok && buf != nil {
		total += int64(buf.GetSize())
	}
	if buf, ok := l.GPUKVCacheV.(sizer); ok && buf != nil {
		total += int64(buf.GetSize())
	}
	for i := range l.ParallelBranches {
		total += volumetricLayerVRAMKVRecursive(&l.ParallelBranches[i])
	}
	for i := range l.SequentialLayers {
		total += volumetricLayerVRAMKVRecursive(&l.SequentialLayers[i])
	}
	return total
}

// CalculateTotalMemory returns the total size of all layers in bytes.
func (n *VolumetricNetwork) CalculateTotalMemory() int {
	total := 0
	for i := range n.Layers {
		total += volumetricLayerCPUSizeRecursive(&n.Layers[i])
	}
	return total
}

// GetVRAMWeightsBytes returns GPU bytes for embeddings, LM head (if distinct), layer weights/scales,
// and nested stacks. It excludes KV cache buffers and internal activation/uniform pools.
func (n *VolumetricNetwork) GetVRAMWeightsBytes() int64 {
	if n.GPUContext == nil {
		return 0
	}
	type sizer interface{ GetSize() uint64 }
	var total int64
	if buf, ok := n.GPUEmbeddings.(sizer); ok && buf != nil {
		total += int64(buf.GetSize())
	}
	if buf, ok := n.GPULMHead.(sizer); ok && buf != nil {
		if n.GPULMHead != n.GPUEmbeddings {
			total += int64(buf.GetSize())
		}
	}
	for i := range n.Layers {
		total += volumetricLayerVRAMWeightsRecursive(&n.Layers[i])
	}
	return total
}

// GetVRAMKVCacheBytes returns GPU bytes for per-layer K/V cache buffers only.
func (n *VolumetricNetwork) GetVRAMKVCacheBytes() int64 {
	if n.GPUContext == nil {
		return 0
	}
	var total int64
	for i := range n.Layers {
		total += volumetricLayerVRAMKVRecursive(&n.Layers[i])
	}
	return total
}

// GetVRAMUsage calculates the total GPU memory allocated by the network in bytes.
func (n *VolumetricNetwork) GetVRAMUsage() int64 {
	if n.GPUContext == nil {
		return 0
	}
	total := n.GetVRAMWeightsBytes() + n.GetVRAMKVCacheBytes()
	for _, buf := range n.GPUContext.ActivationPool {
		total += int64(buf.GetSize())
	}
	for _, buf := range n.GPUContext.UniformPool {
		total += int64(buf.GetSize())
	}
	return total
}

// MorphLayer performs an on-the-fly conversion of a layer's weights to a new DType.
func MorphLayer(layer *VolumetricLayer, target DType) error {
	if layer.WeightStore == nil {
		return fmt.Errorf("layer has no WeightStore to morph")
	}
	layer.WeightStore.Morph(target)
	layer.DType = target
	return nil
}

// stripLayerGPUWeights releases GPU weight/scale/KV buffers for one layer tree.
func stripLayerGPUWeights(l *VolumetricLayer) {
	for i := range l.ParallelBranches {
		stripLayerGPUWeights(&l.ParallelBranches[i])
	}
	for i := range l.SequentialLayers {
		stripLayerGPUWeights(&l.SequentialLayers[i])
	}
	if l.WeightStore != nil {
		for dtype, wg := range l.WeightStore.GPUWeights {
			if buf, ok := wg.(*wgpu.Buffer); ok && buf != nil {
				buf.Release()
			}
			delete(l.WeightStore.GPUWeights, dtype)
		}
		for dtype, buf := range l.WeightStore.GPUScales {
			if buf != nil {
				buf.Release()
			}
			delete(l.WeightStore.GPUScales, dtype)
		}
	}
	if buf, ok := l.GPUKVCacheK.(*wgpu.Buffer); ok && buf != nil {
		buf.Release()
	}
	if buf, ok := l.GPUKVCacheV.(*wgpu.Buffer); ok && buf != nil {
		buf.Release()
	}
	l.GPUKVCacheK = nil
	l.GPUKVCacheV = nil
	l.IsGPUResident = false
	l.IsKVCacheGPUResident = false
}

// DestroyWGPU releases all GPU resources associated with the network.
func (n *VolumetricNetwork) DestroyWGPU() {
	if n.GPUContext == nil {
		return
	}
	ctx := n.GPUContext

	// Explicitly release all layer weights and caches (including nested parallel/sequential).
	for i := range n.Layers {
		stripLayerGPUWeights(&n.Layers[i])
	}

	// Release network-level persistent buffers
	if buf, ok := n.GPUEmbeddings.(*wgpu.Buffer); ok && buf != nil {
		buf.Release()
	}
	if buf, ok := n.GPULMHead.(*wgpu.Buffer); ok && buf != nil {
		buf.Release()
	}

	ctx.Release() // Releases device and all pools/caches
	n.GPUContext = nil
	n.GPUEmbeddings = nil
	n.GPULMHead = nil
}

// SyncAllToGPU mirrors the entire network state to VRAM.
func (n *VolumetricNetwork) SyncAllToGPU() error {
	if n.GPUContext == nil {
		return fmt.Errorf("GPU context not initialized")
	}

	for i := range n.Layers {
		l := &n.Layers[i]
		if err := l.SyncToGPU(); err != nil {
			return err
		}

		// Initialize GPU KV cache buffers for MHA layers
		if l.Type == LayerMultiHeadAttention && l.GPUKVCacheK == nil {
			// Create empty KV cache buffers in VRAM
			kvSize := l.MaxSeqLen * l.NumKVHeads * l.HeadDim
			kBuf, _ := n.GPUContext.CreatePersistentBuffer(make([]float32, kvSize), "GPU K Cache")
			vBuf, _ := n.GPUContext.CreatePersistentBuffer(make([]float32, kvSize), "GPU V Cache")
			l.GPUKVCacheK = kBuf
			l.GPUKVCacheV = vBuf
			l.IsKVCacheGPUResident = true
		}
	}
	// Pre-allocate common activation buffers for zero-latency inference
	hSize := 0
	if len(n.Layers) > 0 {
		hSize = n.Layers[0].DModel
		if hSize == 0 {
			hSize = n.Layers[0].InputHeight
		}
	}
	if hSize > 0 {
		// Allocate for single-token decode immediately
		n.GPUContext.GetActivationBuffer("hidden_A", uint64(hSize*4), wgpu.BufferUsageStorage)
		n.GPUContext.GetActivationBuffer("hidden_B", uint64(hSize*4), wgpu.BufferUsageStorage)
		n.GPUContext.GetActivationBuffer("norm_out", uint64(hSize*4), wgpu.BufferUsageStorage)
		n.GPUContext.GetActivationBuffer("q_proj", uint64(hSize*4), wgpu.BufferUsageStorage)
		n.GPUContext.GetActivationBuffer("attn_out", uint64(hSize*4), wgpu.BufferUsageStorage)
		n.GPUContext.GetActivationBuffer("staging", uint64(hSize*4), wgpu.BufferUsageMapRead)

		// Pre-allocate for MHA projections (K/V might be smaller but hSize is safe upper bound)
		n.GPUContext.GetActivationBuffer("k_proj", uint64(hSize*4), wgpu.BufferUsageStorage)
		n.GPUContext.GetActivationBuffer("v_proj", uint64(hSize*4), wgpu.BufferUsageStorage)

		// MLP inter is usually larger (e.g. 4x hidden)
		// We'll peek at a SwiGLU layer if possible
		interSize := hSize * 4
		for _, l := range n.Layers {
			if l.Type == LayerSwiGLU && l.OutputHeight > hSize {
				interSize = l.OutputHeight
				break
			}
		}
		n.GPUContext.GetActivationBuffer("mlp_inter", uint64(interSize*4), wgpu.BufferUsageStorage)
	}

	n.UseGPU = true
	return nil
}

// SyncToGPU mirrors active weights and KV caches to the GPU.
func (l *VolumetricLayer) SyncToGPU() error {
	if l.Network.GPUContext == nil {
		return fmt.Errorf("GPU context not initialized")
	}
	ctx := l.Network.GPUContext

	// 1. Sync WeightStore
	if l.WeightStore != nil {
		// Specific sync logic for different layer types/dtypes.
		// IMPORTANT: syncQuantizedDenseI8 / syncQuantizedDense pack weights into a
		// u32 byte-packed buffer that only the Dense INT8/INT4 shader can decode.
		// All other layer types (CNN1/CNN2/CNN3, RNN, LSTM, Residual, etc.) use
		// array<f32> shaders and must receive float32 buffers.  Sending them the
		// packed u32 format causes all-zero outputs because the shader misinterprets
		// the bits.  Restrict the packed paths strictly to layers with matching shaders.
		hasQuantizedShader := l.Type == LayerDense || l.Type == LayerSwiGLU || l.Type == LayerMultiHeadAttention
		hasCNN1NativePackedPath := l.Type == LayerCNN1 && isCNN1NativeGPUQuantDType(l.DType)
		if l.Type == LayerRMSNorm {
			// RMSNorm MUST stay in FP32
		} else if hasQuantizedShader && l.DType == DTypeTernary {
			if l.Type == LayerSwiGLU {
				if err := l.syncBitNetPackedSwiGLU(ctx); err != nil {
					return err
				}
			} else if l.Type == LayerMultiHeadAttention {
				if err := l.syncBitNetPackedMHA(ctx); err != nil {
					return err
				}
			} else {
				if err := l.syncBitNetPackedDense(ctx); err != nil {
					return err
				}
			}
		} else if hasQuantizedShader && DTypeBits(l.DType) == 8 {
			if l.Type == LayerSwiGLU {
				h, inter := l.InputHeight, l.OutputHeight
				l.syncQuantizedSwiGLU_I8(ctx, h, inter)
			} else if l.Type == LayerMultiHeadAttention {
				l.syncQuantizedMHA_I8(ctx)
			} else if l.Type == LayerDense && l.DType == DTypeInt8 {
				// Packed INT8 dense path only — Uint8/FP8 use PTQ→F32 buffers like F16.
				l.syncQuantizedDenseI8(ctx, "Layer Weights")
			}
		} else if hasQuantizedShader && DTypeBits(l.DType) <= 4 {
			if l.Type == LayerSwiGLU {
				h, inter := l.InputHeight, l.OutputHeight
				l.syncQuantizedSwiGLU(ctx, h, inter)
			} else if l.Type == LayerMultiHeadAttention {
				l.syncQuantizedMHA(ctx)
			} else {
				l.syncQuantizedDense(ctx, "Layer Weights")
			}
		} else if l.Type == LayerSwiGLU {
			l.syncFP32SwiGLU(ctx)
		} else if l.Type == LayerMultiHeadAttention {
			l.syncFP32MHA(ctx)
		} else if hasCNN1NativePackedPath {
			l.syncNativeCNN1Packed(ctx)
		}

		// 2. Upload weights in the layer's native DType, but only for layers
		// that do NOT already have a dedicated quantized GPU path (syncQuantizedSwiGLU,
		// syncQuantizedMHA, syncQuantizedDenseI8 etc.).  Those paths already populated
		// GPUWeights with the correctly-packed buffers; adding a redundant Float32 copy
		// here would waste significant VRAM during LLM inference.
		//
		// For layers without a special quantized path (CNN1-3, RNN, LSTM, Embedding with
		// non-Float32 dtype), we upload a PTQ-simulated buffer: master weights are
		// quantized to the target dtype and dequantized back to float32
		// (MorphToFloat32ForGPU). The shader still reads array<f32>, so no new shaders
		// are needed — inference sees the precision-limited weights, training updates
		// the float32 master.
		//
		// Float64 → Float32 since GPU shaders work in f32.
		// The buffer is always refreshed so weight restorations are reflected on the GPU.
		if len(l.WeightStore.Master) > 0 {
			fwdDType := l.DType
			if fwdDType == DTypeFloat64 {
				fwdDType = DTypeFloat32
			}

			// Dense: skip redundant PTQ→native-key upload only when forward uses a packed
			// native buffer (BitNet, signed INT8, or ≤4-bit / Q4-style). Uint8/FP8 are not
			// the INT8 packed dense path; they use dequantized F32 in the generic upload.
			denseUsedNativeQuantGPU := l.Type == LayerDense &&
				(l.DType == DTypeTernary || l.DType == DTypeInt8 || DTypeBits(l.DType) <= 4)
			hasSpecialPath := fwdDType != DTypeFloat32 &&
				(hasCNN1NativePackedPath ||
					l.Type == LayerSwiGLU ||
					l.Type == LayerMultiHeadAttention ||
					denseUsedNativeQuantGPU)

			if !hasSpecialPath {
				gpuData := l.WeightStore.MorphToFloat32ForGPU(fwdDType)
				if existingAny, ok := l.WeightStore.GPUWeights[fwdDType]; ok {
					if wBuf, ok2 := existingAny.(*wgpu.Buffer); ok2 && wBuf != nil {
						ctx.Queue.WriteBuffer(wBuf, 0, wgpu.ToBytes(gpuData))
					}
				} else {
					buf, err := ctx.CreatePersistentBuffer(gpuData, "Layer Weights ["+fwdDType.String()+"]")
					if err != nil {
						return err
					}
					l.WeightStore.GPUWeights[fwdDType] = buf
				}
			}
		}

		// Dense backward tiled kernels always read dequantized float32 coefficients (same
		// PTQ simulation as MorphToFloat32ForGPU), not packed INT8/Q4 payloads.
		if l.Type == LayerDense {
			if err := l.syncDenseFloat32CoefForBackward(ctx); err != nil {
				return err
			}
		}
	}

	// 2. Sync KV Cache if present
	if l.Type == LayerMultiHeadAttention {
		if l.GPUKVCacheK == nil {
			size := uint64(l.MaxSeqLen * l.NumKVHeads * l.HeadDim * 4)
			bufK, _ := ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
				Label: "KV Cache K",
				Size:  size,
				Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
			})
			bufV, _ := ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
				Label: "KV Cache V",
				Size:  size,
				Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
			})
			l.GPUKVCacheK = bufK
			l.GPUKVCacheV = bufV
		}
		l.IsKVCacheGPUResident = true
	}

	// 3. Populate per-dtype GPU tile size maps from the GPU context.
	if l.Network.GPUContext != nil {
		l.refreshRuntimeGPUTileSizes()
	}

	// 4. Nested stacks (parallel branches, sequential substeps) carry their own WeightStores.
	for i := range l.ParallelBranches {
		if err := (&l.ParallelBranches[i]).SyncToGPU(); err != nil {
			return err
		}
	}
	for i := range l.SequentialLayers {
		if err := (&l.SequentialLayers[i]).SyncToGPU(); err != nil {
			return err
		}
	}

	l.IsGPUResident = true
	return nil
}

// ReleaseInferenceHostWeights frees CPU copies of weights after GPU sync (Master, optional Q/K norm
// slices). Safe when inference runs entirely on GPU; CPU fallback will no longer work for this layer.
func (l *VolumetricLayer) ReleaseInferenceHostWeights() {
	if l == nil {
		return
	}
	if l.IsGPUResident && l.WeightStore != nil && len(l.WeightStore.GPUWeights) > 0 {
		l.WeightStore.ReleaseInferenceHostWeights()
	}
	l.QNormWeight = nil
	l.KNormWeight = nil
	l.InnerNormWeight = nil
	for i := range l.ParallelBranches {
		l.ParallelBranches[i].ReleaseInferenceHostWeights()
	}
	for i := range l.SequentialLayers {
		l.SequentialLayers[i].ReleaseInferenceHostWeights()
	}
}

// syncDenseFloat32CoefForBackward uploads dequantized float32 weights for Dense layers.
// Forward may use packed INT8/Q4/BitNet buffers; tiled backward (DispatchDenseBackward*)
// always consumes the same PTQ-simulated f32 matrix as MorphToFloat32ForGPU.
func (l *VolumetricLayer) syncDenseFloat32CoefForBackward(ctx *WGPUContext) error {
	ws := l.WeightStore
	if ws == nil || len(ws.Master) == 0 {
		return nil
	}
	gpuData := ws.MorphToFloat32ForGPU(l.DType)
	if len(gpuData) == 0 {
		return nil
	}
	wantBytes := uint64(len(gpuData) * 4)
	type sizer interface{ GetSize() uint64 }
	if existingAny, ok := ws.GPUWeights[DTypeFloat32]; ok {
		if wBuf, ok2 := existingAny.(*wgpu.Buffer); ok2 && wBuf != nil {
			if sz, ok3 := interface{}(wBuf).(sizer); ok3 && sz.GetSize() >= wantBytes {
				ctx.Queue.WriteBuffer(wBuf, 0, wgpu.ToBytes(gpuData))
				return nil
			}
			wBuf.Release()
			delete(ws.GPUWeights, DTypeFloat32)
		}
	}
	buf, err := ctx.CreatePersistentBuffer(gpuData, "Dense coef F32 (backward)")
	if err != nil {
		return err
	}
	ws.GPUWeights[DTypeFloat32] = buf
	return nil
}

// syncQuantizedDense handles the quantization and upload of a single weight buffer.
func (l *VolumetricLayer) syncQuantizedDense(ctx *WGPUContext, label string) {
	blocks := QuantizeQ4_0(l.WeightStore.Master)

	// Ensure packed size is a multiple of tile/workgroup alignment and meets
	// minimum driver-enforced binding ranges (2048 bytes / 512 uint32s).
	numBlocks := len(blocks)
	packedSize := numBlocks * 4
	alignedSize := (packedSize + 63) &^ 63
	if alignedSize < 512 {
		alignedSize = 512
	}

	scales := make([]float32, len(blocks))
	packed := make([]uint32, alignedSize)

	for i, b := range blocks {
		scales[i] = b.Scale
		for j := 0; j < 4; j++ {
			// Pack 4 bytes into one u32 (lower nibbles first)
			packed[i*4+j] = uint32(b.Weights[j*4]) |
				(uint32(b.Weights[j*4+1]) << 8) |
				(uint32(b.Weights[j*4+2]) << 16) |
				(uint32(b.Weights[j*4+3]) << 24)
		}
	}

	sBuf, _ := ctx.CreatePersistentBuffer(scales, label+" Scales")

	pBuf, _ := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    label + " Packed",
		Contents: wgpu.ToBytes(packed),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})

	l.WeightStore.GPUScales[DTypeInt4] = sBuf
	l.WeightStore.GPUWeights[DTypeInt4] = pBuf
}

func isCNN1NativeGPUQuantDType(dtype DType) bool {
	switch dtype {
	case DTypeInt8, DTypeInt4, DTypeInt2, DTypeFP4, DTypeTernary, DTypeBinary,
		DTypeFP8E4M3, DTypeFP8E5M2, DTypeUint8, DTypeUint4, DTypeUint2,
		DTypeFloat16, DTypeBFloat16, DTypeInt16:
		return true
	default:
		return false
	}
}

func alignedPackedWordCount(wordCount int) int {
	aligned := (wordCount + 63) &^ 63
	if aligned < 512 {
		aligned = 512
	}
	return aligned
}

func packSignedNibblesToU32(data []uint8) []uint32 {
	wordCount := (len(data) + 7) / 8
	packed := make([]uint32, alignedPackedWordCount(wordCount))
	for i, v := range data {
		shift := uint((i % 8) * 4)
		packed[i/8] |= uint32(v&0x0F) << shift
	}
	return packed
}

func packSignedBytesToU32(data []int8) []uint32 {
	wordCount := (len(data) + 3) / 4
	packed := make([]uint32, alignedPackedWordCount(wordCount))
	for i, v := range data {
		shift := uint((i % 4) * 8)
		packed[i/4] |= uint32(uint8(v)) << shift
	}
	return packed
}

func packSigned2ToU32(data []uint8) []uint32 {
	wordCount := (len(data) + 15) / 16
	packed := make([]uint32, alignedPackedWordCount(wordCount))
	for i, v := range data {
		shift := uint((i % 16) * 2)
		packed[i/16] |= uint32(v&0x03) << shift
	}
	return packed
}
func pack16BitToU32[T ~uint16 | ~int16](data []T) []uint32 {
	wordCount := (len(data) + 1) / 2
	packed := make([]uint32, alignedPackedWordCount(wordCount))
	for i, v := range data {
		shift := uint((i % 2) * 16)
		packed[i/2] |= uint32(uint16(v)) << shift
	}
	return packed
}

func pack8BitToU32(data []uint8) []uint32 {
	wordCount := (len(data) + 3) / 4
	packed := make([]uint32, alignedPackedWordCount(wordCount))
	for i, v := range data {
		shift := uint((i % 4) * 8)
		packed[i/4] |= uint32(v) << shift
	}
	return packed
}

func packUnsignedNibblesToU32(data []uint8) []uint32 {
	wordCount := (len(data) + 7) / 8
	packed := make([]uint32, alignedPackedWordCount(wordCount))
	for i, v := range data {
		shift := uint((i % 8) * 4)
		packed[i/8] |= uint32(v&0x0F) << shift
	}
	return packed
}

func packUnsigned2ToU32(data []uint8) []uint32 {
	wordCount := (len(data) + 15) / 16
	packed := make([]uint32, alignedPackedWordCount(wordCount))
	for i, v := range data {
		shift := uint((i % 16) * 2)
		packed[i/16] |= uint32(v&0x03) << shift
	}
	return packed
}

func packFP4CodesToU32(data []uint8) []uint32 {
	wordCount := (len(data) + 7) / 8
	packed := make([]uint32, alignedPackedWordCount(wordCount))
	for i, v := range data {
		shift := uint((i % 8) * 4)
		packed[i/8] |= uint32(v&0x0F) << shift
	}
	return packed
}

func packTernaryToU32(data []uint8) []uint32 {
	wordCount := (len(data) + 15) / 16
	packed := make([]uint32, alignedPackedWordCount(wordCount))
	for i, v := range data {
		val := int8(v)
		code := uint32(1)
		switch val {
		case -1:
			code = 0
		case 0:
			code = 1
		case 1:
			code = 2
		}
		shift := uint((i % 16) * 2)
		packed[i/16] |= code << shift
	}
	return packed
}

func pack32BitToU32[T ~int32 | ~uint32](data []T) []uint32 {
	packed := make([]uint32, alignedPackedWordCount(len(data)))
	for i, v := range data {
		packed[i] = uint32(v)
	}
	return packed
}

func pack64BitToU32[T ~int64 | ~uint64](data []T) []uint32 {
	packed := make([]uint32, alignedPackedWordCount(len(data)*2))
	for i, v := range data {
		u := uint64(v)
		packed[i*2] = uint32(u & 0xFFFFFFFF)
		packed[i*2+1] = uint32(u >> 32)
	}
	return packed
}

func packBinaryToU32(data []uint8) []uint32 {
	wordCount := (len(data) + 31) / 32
	packed := make([]uint32, alignedPackedWordCount(wordCount))
	for i, v := range data {
		// Treated as signed int8: 1 is positive, 255 is -1
		if int8(v) > 0 {
			shift := uint(i % 32)
			packed[i/32] |= 1 << shift
		}
	}
	return packed
}

func (l *VolumetricLayer) syncNativeCNN1Packed(ctx *WGPUContext) {
	ws := l.WeightStore
	if ws == nil {
		return
	}
	dtype := l.DType

	// Ensure weights exist in requested bit-depth before packing
	if _, ok := ws.Versions[dtype]; !ok {
		ws.Morph(dtype)
	}

	// Check for existing GPU buffer
	if existingAny, ok := ws.GPUWeights[dtype]; ok {
		if buf, ok2 := existingAny.(*wgpu.Buffer); ok2 && buf != nil {
			// Already uploaded
			return
		}
	}

	var packed []uint32
	switch dtype {
	case DTypeFloat16, DTypeBFloat16, DTypeInt16, DTypeUint16:
		switch v := ws.Versions[dtype].(type) {
		case []uint16:
			packed = pack16BitToU32(v)
		case []int16:
			packed = pack16BitToU32(v)
		}
	case DTypeInt32, DTypeUint32:
		switch v := ws.Versions[dtype].(type) {
		case []int32:
			packed = pack32BitToU32(v)
		case []uint32:
			packed = pack32BitToU32(v)
		}
	case DTypeInt64, DTypeUint64:
		switch v := ws.Versions[dtype].(type) {
		case []int64:
			packed = pack64BitToU32(v)
		case []uint64:
			packed = pack64BitToU32(v)
		}
	case DTypeInt8, DTypeUint8, DTypeFP8E4M3, DTypeFP8E5M2:
		switch v := ws.Versions[dtype].(type) {
		case []uint8:
			packed = pack8BitToU32(v)
		case []int8:
			packed = packSignedBytesToU32(v)
		}
	case DTypeInt4, DTypeUint4, DTypeFP4:
		if v, ok := ws.Versions[dtype].([]uint8); ok {
			packed = packUnsignedNibblesToU32(v)
		}
	case DTypeInt2, DTypeUint2, DTypeTernary:
		if v, ok := ws.Versions[dtype].([]uint8); ok {
			if dtype == DTypeTernary {
				packed = packTernaryToU32(v)
			} else {
				packed = packUnsigned2ToU32(v)
			}
		}
	case DTypeBinary:
		if v, ok := ws.Versions[dtype].([]uint8); ok {
			packed = packBinaryToU32(v)
		}
	default:
		return
	}

	if packed == nil {
		return
	}

	pBuf, _ := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "CNN1 Packed [" + l.DType.String() + "]",
		Contents: wgpu.ToBytes(packed),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	ws.GPUWeights[dtype] = pBuf
}

// syncQuantizedDenseI8 packs 4 int8 weights into a u32 for GPU storage.
func (l *VolumetricLayer) syncQuantizedDenseI8(ctx *WGPUContext, label string) {
	ws := l.WeightStore

	// Morph to INT8 if not already done
	ws.Morph(DTypeInt8)
	i8Weights := ws.Int8Slice(DTypeInt8)

	numElements := len(i8Weights)
	packedSize := (numElements + 3) / 4
	alignedSize := (packedSize + 63) &^ 63 // Align to 64 uint32s (256 bytes)
	if alignedSize < 512 {
		alignedSize = 512
	} // Enforce 2048 byte minimum for validation parity

	packed := make([]uint32, alignedSize)
	for i := 0; i < len(i8Weights); i += 4 {
		var p uint32
		p = uint32(uint8(i8Weights[i]))
		if i+1 < len(i8Weights) {
			p |= uint32(uint8(i8Weights[i+1])) << 8
		}
		if i+2 < len(i8Weights) {
			p |= uint32(uint8(i8Weights[i+2])) << 16
		}
		if i+3 < len(i8Weights) {
			p |= uint32(uint8(i8Weights[i+3])) << 24
		}
		packed[i/4] = p
	}

	pBuf, _ := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    label + " INT8",
		Contents: wgpu.ToBytes(packed),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})

	l.WeightStore.GPUWeights[DTypeInt8] = pBuf
}

func (l *VolumetricLayer) syncQuantizedSwiGLU_I8(ctx *WGPUContext, h, inter int) {
	ws := l.WeightStore
	ws.Morph(DTypeInt8)
	i8Weights := ws.Int8Slice(DTypeInt8)

	wSize := h * inter
	gateW := i8Weights[0:wSize]
	upW := i8Weights[wSize : 2*wSize]
	downW := i8Weights[2*wSize : 3*wSize]

	l.syncQuantizedComponentI8(ctx, gateW, "Gate", DType(100))
	l.syncQuantizedComponentI8(ctx, upW, "Up", DType(101))
	l.syncQuantizedComponentI8(ctx, downW, "Down", DType(102))

	// Biases stay FP32 for precision
	master := ws.Master
	gateB := master[3*wSize : 3*wSize+inter]
	upB := master[3*wSize+inter : 3*wSize+2*inter]
	downB := master[3*wSize+2*inter : 3*wSize+2*inter+h]

	gBBuf, _ := ctx.CreatePersistentBuffer(gateB, "Gate Bias")
	uBBuf, _ := ctx.CreatePersistentBuffer(upB, "Up Bias")
	dBBuf, _ := ctx.CreatePersistentBuffer(downB, "Down Bias")

	l.WeightStore.GPUWeights[DType(110)] = gBBuf
	l.WeightStore.GPUWeights[DType(111)] = uBBuf
	l.WeightStore.GPUWeights[DType(112)] = dBBuf
}

func (l *VolumetricLayer) syncQuantizedComponentI8(ctx *WGPUContext, data []int8, label string, weightDType DType) {
	// Ensure packed size is a multiple of 64 uint32s to satisfy the shader workgroup dispatch
	// and minimum driver-specific binding ranges (e.g. 2048 bytes).
	numElements := len(data)
	packedSize := (numElements + 3) / 4
	alignedSize := (packedSize + 63) &^ 63
	if alignedSize < 512 {
		alignedSize = 512
	}

	packed := make([]uint32, alignedSize)
	for i := 0; i < len(data); i += 4 {
		var p uint32
		p = uint32(uint8(data[i]))
		if i+1 < len(data) {
			p |= uint32(uint8(data[i+1])) << 8
		}
		if i+2 < len(data) {
			p |= uint32(uint8(data[i+2])) << 16
		}
		if i+3 < len(data) {
			p |= uint32(uint8(data[i+3])) << 24
		}
		packed[i/4] = p
	}
	pBuf, _ := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    label + " INT8",
		Contents: wgpu.ToBytes(packed),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	l.WeightStore.GPUWeights[weightDType] = pBuf
}

func (l *VolumetricLayer) syncQuantizedMHA_I8(ctx *WGPUContext) {
	ws := l.WeightStore
	ws.Morph(DTypeInt8)
	i8Weights := ws.Int8Slice(DTypeInt8)

	d := l.DModel
	q := l.QueryDim
	if q == 0 {
		q = d
	}
	kv := l.NumKVHeads * l.HeadDim
	qwSize := q * d
	kwSize := d * kv
	vwSize := d * kv
	owSize := d * q

	l.syncQuantizedComponentI8(ctx, i8Weights[0:qwSize], "Q", WeightMHAQuery)
	l.syncQuantizedComponentI8(ctx, i8Weights[qwSize:qwSize+kwSize], "K", WeightMHAKey)
	l.syncQuantizedComponentI8(ctx, i8Weights[qwSize+kwSize:qwSize+kwSize+vwSize], "V", WeightMHAValue)
	l.syncQuantizedComponentI8(ctx, i8Weights[qwSize+kwSize+vwSize:qwSize+kwSize+vwSize+owSize], "O", WeightMHAProjection)
	if len(l.QNormWeight) > 0 {
		if buf, err := ctx.CreatePersistentBuffer(l.QNormWeight, "QNorm Weights"); err == nil {
			l.WeightStore.GPUWeights[WeightMHAQNorm] = buf
		}
	}
	if len(l.KNormWeight) > 0 {
		if buf, err := ctx.CreatePersistentBuffer(l.KNormWeight, "KNorm Weights"); err == nil {
			l.WeightStore.GPUWeights[WeightMHAKNorm] = buf
		}
	}
}

func (l *VolumetricLayer) syncQuantizedSwiGLU(ctx *WGPUContext, h, inter int) {
	w := l.WeightStore.Master
	gateW := w[0 : h*inter]
	upW := w[h*inter : 2*h*inter]
	downW := w[2*h*inter : 3*h*inter]

	gateB := w[3*h*inter : 3*h*inter+inter]
	upB := w[3*h*inter+inter : 3*h*inter+2*inter]
	downB := w[3*h*inter+2*inter : 3*h*inter+2*inter+h]

	// Weights & Scales
	l.syncQuantizedComponent(ctx, gateW, "Gate", DType(1100), DType(100))
	l.syncQuantizedComponent(ctx, upW, "Up", DType(1101), DType(101))
	l.syncQuantizedComponent(ctx, downW, "Down", DType(1102), DType(102))

	// Biases (typically kept in FP32 on GPU for precision)
	gBBuf, _ := ctx.CreatePersistentBuffer(gateB, "Gate Bias")
	uBBuf, _ := ctx.CreatePersistentBuffer(upB, "Up Bias")
	dBBuf, _ := ctx.CreatePersistentBuffer(downB, "Down Bias")

	l.WeightStore.GPUWeights[DType(110)] = gBBuf
	l.WeightStore.GPUWeights[DType(111)] = uBBuf
	l.WeightStore.GPUWeights[DType(112)] = dBBuf
}

func (l *VolumetricLayer) syncQuantizedComponent(ctx *WGPUContext, data []float32, label string, scaleDType, weightDType DType) {
	blocks := QuantizeQ4_0(data)
	// Ensure packed size is a multiple of tile/workgroup alignment
	numBlocks := len(blocks)
	packedSize := numBlocks * 4
	alignedSize := (packedSize + 63) &^ 63 // Match 64-thread workgroups
	if alignedSize < 512 {
		alignedSize = 512
	} // 2048 byte minimum

	scales := make([]float32, len(blocks))
	packed := make([]uint32, alignedSize)
	for i, b := range blocks {
		scales[i] = b.Scale
		for j := 0; j < 4; j++ {
			packed[i*4+j] = uint32(b.Weights[j*4]) | (uint32(b.Weights[j*4+1]) << 8) | (uint32(b.Weights[j*4+2]) << 16) | (uint32(b.Weights[j*4+3]) << 24)
		}
	}
	sBuf, _ := ctx.CreatePersistentBuffer(scales, label+" Scales")
	pBuf, _ := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    label + " Packed",
		Contents: wgpu.ToBytes(packed),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	l.WeightStore.GPUScales[weightDType] = sBuf
	l.WeightStore.GPUWeights[weightDType] = pBuf
}

func (l *VolumetricLayer) syncFP32MHA(ctx *WGPUContext) {
	d := l.DModel
	q := l.QueryDim
	if q == 0 {
		q = d
	}
	kv := l.NumKVHeads * l.HeadDim
	qwSize := q * d
	kwSize := d * kv
	vwSize := d * kv
	owSize := d * q

	w := l.WeightStore.Master
	qBuf, err := ctx.CreatePersistentBuffer(w[0:qwSize], "Q Weights")
	if err != nil {
		fmt.Printf("Q err: %v\n", err)
	}
	kBuf, err := ctx.CreatePersistentBuffer(w[qwSize:qwSize+kwSize], "K Weights")
	if err != nil {
		fmt.Printf("K err: %v\n", err)
	}
	vBuf, err := ctx.CreatePersistentBuffer(w[qwSize+kwSize:qwSize+kwSize+vwSize], "V Weights")
	if err != nil {
		fmt.Printf("V err: %v\n", err)
	}
	oBuf, err := ctx.CreatePersistentBuffer(w[qwSize+kwSize+vwSize:qwSize+kwSize+vwSize+owSize], "O Weights")
	if err != nil {
		fmt.Printf("O err: %v\n", err)
	}

	l.WeightStore.GPUWeights[WeightMHAQuery] = qBuf
	l.WeightStore.GPUWeights[WeightMHAKey] = kBuf
	l.WeightStore.GPUWeights[WeightMHAValue] = vBuf
	l.WeightStore.GPUWeights[WeightMHAProjection] = oBuf
	if len(l.QNormWeight) > 0 {
		if buf, err := ctx.CreatePersistentBuffer(l.QNormWeight, "QNorm Weights"); err == nil {
			l.WeightStore.GPUWeights[WeightMHAQNorm] = buf
		}
	}
	if len(l.KNormWeight) > 0 {
		if buf, err := ctx.CreatePersistentBuffer(l.KNormWeight, "KNorm Weights"); err == nil {
			l.WeightStore.GPUWeights[WeightMHAKNorm] = buf
		}
	}
}

func (l *VolumetricLayer) syncFP32SwiGLU(ctx *WGPUContext) {
	h := l.InputHeight
	inter := l.OutputHeight
	wSize := h * inter

	w := l.WeightStore.Master
	if len(w) < 3*wSize+2*inter+h {
		return
	}

	gBuf, _ := ctx.CreatePersistentBuffer(w[0:wSize], "Gate Weights")
	uBuf, _ := ctx.CreatePersistentBuffer(w[wSize:2*wSize], "Up Weights")
	dBuf, _ := ctx.CreatePersistentBuffer(w[2*wSize:3*wSize], "Down Weights")

	gBBuf, _ := ctx.CreatePersistentBuffer(w[3*wSize:3*wSize+inter], "Gate Bias")
	uBBuf, _ := ctx.CreatePersistentBuffer(w[3*wSize+inter:3*wSize+2*inter], "Up Bias")
	dBBuf, _ := ctx.CreatePersistentBuffer(w[3*wSize+2*inter:3*wSize+2*inter+h], "Down Bias")

	l.WeightStore.GPUWeights[DType(100)] = gBuf
	l.WeightStore.GPUWeights[DType(101)] = uBuf
	l.WeightStore.GPUWeights[DType(102)] = dBuf

	l.WeightStore.GPUWeights[DType(110)] = gBBuf
	l.WeightStore.GPUWeights[DType(111)] = uBBuf
	l.WeightStore.GPUWeights[DType(112)] = dBBuf
}

func (l *VolumetricLayer) syncQuantizedMHA(ctx *WGPUContext) {
	d := l.DModel
	q := l.QueryDim
	if q == 0 {
		q = d
	}
	kv := l.NumKVHeads * l.HeadDim
	qwSize := q * d
	kwSize := d * kv
	vwSize := d * kv
	owSize := d * q

	w := l.WeightStore.Master
	l.syncQuantizedComponent(ctx, w[0:qwSize], "Q", WeightMHAQuery, WeightMHAQuery)
	l.syncQuantizedComponent(ctx, w[qwSize:qwSize+kwSize], "K", WeightMHAKey, WeightMHAKey)
	l.syncQuantizedComponent(ctx, w[qwSize+kwSize:qwSize+kwSize+vwSize], "V", WeightMHAValue, WeightMHAValue)
	l.syncQuantizedComponent(ctx, w[qwSize+kwSize+vwSize:qwSize+kwSize+vwSize+owSize], "O", WeightMHAProjection, WeightMHAProjection)
	if len(l.QNormWeight) > 0 {
		if buf, err := ctx.CreatePersistentBuffer(l.QNormWeight, "QNorm Weights"); err == nil {
			l.WeightStore.GPUWeights[WeightMHAQNorm] = buf
		}
	}
	if len(l.KNormWeight) > 0 {
		if buf, err := ctx.CreatePersistentBuffer(l.KNormWeight, "KNorm Weights"); err == nil {
			l.WeightStore.GPUWeights[WeightMHAKNorm] = buf
		}
	}
}

// SyncToCPU releases GPU resources and prepares the individual layer for CPU tiling optimizations.
// Per-dtype CPU tile sizes are computed for all 21 numerical types.
func (l *VolumetricLayer) SyncToCPU() {
	l.EnableMultiCoreTiling = l.Network.EnableMultiCoreTiling

	if l.UseTiling {
		l.refreshRuntimeCPUTileSizes()
	}

	if l.WeightStore != nil {
		for dtype, buf := range l.WeightStore.GPUWeights {
			_ = dtype
			_ = buf
		}
		l.WeightStore.GPUWeights = make(map[DType]any)
	}
	l.IsGPUResident = false
	l.IsKVCacheGPUResident = false

	for i := range l.ParallelBranches {
		l.ParallelBranches[i].SyncToCPU()
	}
	for i := range l.SequentialLayers {
		l.SequentialLayers[i].SyncToCPU()
	}
}
