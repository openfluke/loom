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
	default:
		return fmt.Sprintf("LayerType(%d)", t)
	}
}

// ActivationType defines the activation function
type ActivationType int

const (
	ActivationReLU    ActivationType = 0
	ActivationSilu    ActivationType = 1
	ActivationGELU    ActivationType = 2
	ActivationTanh    ActivationType = 3
	ActivationSigmoid ActivationType = 4
	ActivationLinear  ActivationType = -1
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
numerical metamorphosis.

I. MULTI-NUMERICAL ARCHITECTURE (The "M" in M-POLY)
--------------------------------------------------
This engine supports native forward/backward passes across diverse numerical
types (FP32, FP16, INT8, and FP4 E2M1).

1. Bandwidth Optimization (The 192 GB/s Wall):
   - Targets a 75-80% reduction in weight size via low-bit quantization.
   - Specifically optimized for Turing (GTX 1650 Super) memory constraints,
     where global memory reads are the primary bottleneck for SmolLM2-135M.

2. Numerical Switching "On Cue":
   - Supports mid-stream precision shifts. A layer can be "Morphed" via
     QAT (Quantization-Aware Training) logic on-the-fly, allowing the
     dispatcher to move from high-precision accumulation to low-bit
     throughput based on the model's state or "command."

3. Hardware-Aware Emulation:
   - Since Turing lacks native FP4 Tensor Cores, the "Multi-Numerical"
     bus handles vectorized unpacking (Stage 3 optimization). It treats
     low-bit types as "packed payloads" to be expanded in-shader,
     mimicking the efficiency of native hardware.

II. POLYMORPHIC LAYER-MORPHING (The "POLY")
-------------------------------------------
- Compartmentalization: Every layer is treated as a polymorphic processing
  unit that can transform its weight-store (e.g., FP32 -> FP4) and
  re-compartmentalize its state for the next step in the 3D grid.
- Dynamic DType Management: Uses a WeightStore versioning system to
  instantly swap between active numerical representations without
  re-allocating buffers.

III. VOLUMETRIC TENSOR DISPATCH (The "VTD")
-------------------------------------------
- 3D Grid Representation: Replaces the 1D sequential stack with a
  (Row, Col, Layer) coordinate system.
- Spatial Hopping: Enables (0,0,0) recursive passing. This allows
  multi-pass inference at varying intervals to simulate the recursive
  feedback loops of the human brain.
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
	UseTiling              bool
	EnableMultiCoreTiling bool
	UseGPU                 bool

	// GPU Acceleration context
	GPUContext *WGPUContext

	// Persistent GPU buffers to avoid allocations
	GPUHiddenState []any // map[DType]wgpu.Buffer or similar, use any for now
	GPULogits      any   // wgpu.Buffer

	GPUEmbeddings any // *wgpu.Buffer
	GPULMHead     any // *wgpu.Buffer
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
	DModel       int
	SeqLength    int
	RoPEFreqBase float64

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
	UseTiling              bool
	EnableMultiCoreTiling bool
	TileSize               int
	UseGPU                 bool

	IsGPUResident        bool
	IsKVCacheGPUResident bool

	Observer PolyObserver

	// KV Cache (for MHA)
	KVCacheK  *Tensor[float32]
	KVCacheV  *Tensor[float32]
	KVOffset  int
	MaxSeqLen int

	// Persistent GPU KV buffers
	GPUKVCacheK any // *wgpu.Buffer
	GPUKVCacheV any // *wgpu.Buffer
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

// CalculateTotalMemory returns the total size of all layers in bytes.
func (n *VolumetricNetwork) CalculateTotalMemory() int {
	total := 0
	for _, l := range n.Layers {
		if l.WeightStore != nil {
			total += l.WeightStore.SizeInBytes(l.DType)
		}
	}
	return total
}

// MorphLayer performs an on-the-fly conversion of a layer's weights to a new DType.
func MorphLayer(layer *VolumetricLayer, target DType) error {
	if layer.WeightStore == nil {
		return fmt.Errorf("layer has no WeightStore to morph")
	}
	// Conversion logic would go here
	layer.DType = target
	return nil
}

// SimulatePrecision handles the numerical simulation of low-bit and non-standard types.
// It is the universal "Metamorphosis" engine used across Dense, CNN, and RNN layers.
func SimulatePrecision(wVal float32, dtype DType, scale float32) float32 {
	switch dtype {
	case DTypeFloat64, DTypeInt64, DTypeUint64, DTypeInt32, DTypeUint32:
		return wVal
	case DTypeBFloat16:
		u32 := math.Float32bits(wVal)
		u32 &= 0xFFFF0000
		return math.Float32frombits(u32)
	case DTypeFP8E4M3, DTypeFP8E5M2, DTypeInt8, DTypeUint8, DTypeInt16, DTypeUint16:
		return float32(int8(wVal/scale)) * scale
	case DTypeInt4, DTypeUint4, DTypeFP4:
		return float32(int(wVal/scale)) * scale
	case DTypeInt2, DTypeUint2:
		// 2-bit simulation (4 levels)
		return float32(int(wVal*2/scale)) * scale / 2
	case DTypeTernary:
		// Ternary (-1, 0, 1)
		if wVal > 0.5*scale {
			return scale
		} else if wVal < -0.5*scale {
			return -scale
		} else {
			return 0
		}
	case DTypeBinary:
		if wVal > 0 {
			return scale
		} else {
			return -scale
		}
	case DTypeFloat16:
		// Simulated truncation (float32 to float16)
		// For now, identity simulation
		return wVal
	default:
		return wVal
	}
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
		if l.Type == LayerRMSNorm {
			// RMSNorm MUST stay in FP32 for numerical stability.
			// 4-bit quantization would destroy the normalization precision.
			if _, ok := l.WeightStore.GPUWeights[DTypeFloat32]; !ok {
				buf, err := ctx.CreatePersistentBuffer(l.WeightStore.Master, "Norm Weights")
				if err != nil {
					return err
				}
				l.WeightStore.GPUWeights[DTypeFloat32] = buf
			}
		} else if l.Type == LayerSwiGLU && l.DType != DTypeInt4 {
			// Split Gate, Up, and Down weights for SwiGLU
			h, inter := l.InputHeight, l.OutputHeight
			gateSlice := l.WeightStore.Master[0 : h*inter]
			upSlice := l.WeightStore.Master[h*inter : 2*h*inter]
			downSlice := l.WeightStore.Master[2*h*inter : 2*h*inter+inter*h]

			gBuf, _ := ctx.CreatePersistentBuffer(gateSlice, "Gate Weights")
			uBuf, _ := ctx.CreatePersistentBuffer(upSlice, "Up Weights")
			dBuf, _ := ctx.CreatePersistentBuffer(downSlice, "Down Weights")
			l.WeightStore.GPUWeights[DType(100)] = gBuf
			l.WeightStore.GPUWeights[DType(101)] = uBuf
			l.WeightStore.GPUWeights[DType(102)] = dBuf
		} else if l.DType == DTypeInt4 {
			// --- Q4_0 Quantized Sync ---
			if l.Type == LayerSwiGLU {
				h, inter := l.InputHeight, l.OutputHeight
				l.syncQuantizedSwiGLU(ctx, h, inter)
			} else if l.Type == LayerMultiHeadAttention {
				l.syncQuantizedMHA(ctx)
			} else {
				l.syncQuantizedDense(ctx, "Layer Weights")
			}
		} else {
			if l.Type == LayerMultiHeadAttention {
				l.syncFP32MHA(ctx)
			} else {
				// Mirror Master (FP32) to GPU if no specific version is requested
				if _, ok := l.WeightStore.GPUWeights[DTypeFloat32]; !ok {
					buf, err := ctx.CreatePersistentBuffer(l.WeightStore.Master, "Layer Weights")
					if err != nil {
						return err
					}
					l.WeightStore.GPUWeights[DTypeFloat32] = buf
				}
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

	l.IsGPUResident = true
	return nil
}


// syncQuantizedDense handles the quantization and upload of a single weight buffer.
func (l *VolumetricLayer) syncQuantizedDense(ctx *WGPUContext, label string) {
	blocks := QuantizeQ4_0(l.WeightStore.Master)

	scales := make([]float32, len(blocks))
	packed := make([]uint32, len(blocks)*4) // 4 u32s per block (16 bytes)

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

func (l *VolumetricLayer) syncQuantizedSwiGLU(ctx *WGPUContext, h, inter int) {
	w := l.WeightStore.Master
	gateSlice := w[0 : h*inter]
	upSlice := w[h*inter : 2*h*inter]
	downSlice := w[2*h*inter : 2*h*inter+inter*h]

	// We'll use special internal DTypes for the SwiGLU components to avoid collisions
	// 1100 = Gate scales, 1101 = Gate weights, etc.
	l.syncQuantizedComponent(ctx, gateSlice, "Gate", DType(1100), DType(100))
	l.syncQuantizedComponent(ctx, upSlice, "Up", DType(1101), DType(101))
	l.syncQuantizedComponent(ctx, downSlice, "Down", DType(1102), DType(102))
}

func (l *VolumetricLayer) syncQuantizedComponent(ctx *WGPUContext, data []float32, label string, scaleDType, weightDType DType) {
	blocks := QuantizeQ4_0(data)
	scales := make([]float32, len(blocks))
	packed := make([]uint32, len(blocks)*4)
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
	kv := l.NumKVHeads * l.HeadDim
	qwSize := d * d
	kwSize := d * kv
	vwSize := d * kv
	owSize := d * d

	w := l.WeightStore.Master
	qBuf, err := ctx.CreatePersistentBuffer(w[0:qwSize], "Q Weights")
	if err != nil { fmt.Printf("Q err: %v\n", err) }
	kBuf, err := ctx.CreatePersistentBuffer(w[qwSize:qwSize+kwSize], "K Weights")
	if err != nil { fmt.Printf("K err: %v\n", err) }
	vBuf, err := ctx.CreatePersistentBuffer(w[qwSize+kwSize:qwSize+kwSize+vwSize], "V Weights")
	if err != nil { fmt.Printf("V err: %v\n", err) }
	oBuf, err := ctx.CreatePersistentBuffer(w[qwSize+kwSize+vwSize:qwSize+kwSize+vwSize+owSize], "O Weights")
	if err != nil { fmt.Printf("O err: %v\n", err) }

	l.WeightStore.GPUWeights[DType(200)] = qBuf
	l.WeightStore.GPUWeights[DType(201)] = kBuf
	l.WeightStore.GPUWeights[DType(202)] = vBuf
	l.WeightStore.GPUWeights[DType(203)] = oBuf
}

func (l *VolumetricLayer) syncQuantizedMHA(ctx *WGPUContext) {
	d := l.DModel
	kv := l.NumKVHeads * l.HeadDim
	qwSize := d * d
	kwSize := d * kv
	vwSize := d * kv
	owSize := d * d

	w := l.WeightStore.Master
	l.syncQuantizedComponent(ctx, w[0:qwSize], "Q", DType(200), DType(200))
	l.syncQuantizedComponent(ctx, w[qwSize:qwSize+kwSize], "K", DType(201), DType(201))
	l.syncQuantizedComponent(ctx, w[qwSize+kwSize:qwSize+kwSize+vwSize], "V", DType(202), DType(202))
	l.syncQuantizedComponent(ctx, w[qwSize+kwSize+vwSize:qwSize+kwSize+vwSize+owSize], "O", DType(203), DType(203))
}

// SyncToCPU releases GPU resources and prepares the individual layer for CPU tiling optimizations.
func (l *VolumetricLayer) SyncToCPU() {
	l.EnableMultiCoreTiling = l.Network.EnableMultiCoreTiling
	if l.UseTiling && l.TileSize <= 0 {
		if l.Type == LayerCNN3 {
			l.TileSize = CalculateOptimalCNN3TileSize(l.InputChannels, l.DType)
		} else if l.Type == LayerMultiHeadAttention {
			l.TileSize = CalculateOptimalTileSize(l.HeadDim)
		} else {
			l.TileSize = 8 // Generic fallback
		}
	}

	if l.WeightStore != nil {
		for dtype, buf := range l.WeightStore.GPUWeights {
			// In a real implementation we'd call Destroy() here
			// For now, we'll clear the map and let the pool/GC handle it if applicable
			_ = dtype
			_ = buf
		}
		l.WeightStore.GPUWeights = make(map[DType]any)
	}
	l.IsGPUResident = false
	l.IsKVCacheGPUResident = false
}
