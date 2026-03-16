package poly

import (
	"fmt"
	"math/rand"
	"time"
)

// =============================================================================
// Brain Types (Polymorphic)
// =============================================================================

type BrainType int

const (
	BrainDense BrainType = iota
	BrainMHA
	BrainSwiGLU
	BrainRMSNorm
	BrainRNN
	BrainLSTM
	BrainLayerNorm
	BrainEmbedding
	BrainKMeans
	BrainSoftmax
	BrainParallel
	BrainSequential
)

var BrainTypeNames = []string{
	"Dense", "MHA", "SwiGLU", "RMSNorm", "RNN", "LSTM", "LayerNorm",
	"Embedding", "KMeans", "Softmax", "Parallel", "Sequential",
}

func (bt BrainType) String() string {
	if int(bt) < len(BrainTypeNames) { return BrainTypeNames[bt] }
	return fmt.Sprintf("BrainType(%d)", bt)
}

func (n *VolumetricNetwork) InitCNNCell(z, y, x, l int, ltype LayerType, inChannels, filters, kSize int, dtype DType, scale float32) {
	layer := n.GetLayer(z, y, x, l)
	layer.Type = ltype
	layer.InputChannels = inChannels
	layer.Filters = filters
	layer.KernelSize = kSize
	layer.DType = dtype
	
	wCount := filters * inChannels * kSize * kSize
	if ltype == LayerCNN3 { wCount *= kSize }
	layer.WeightStore = NewWeightStore(wCount)
	layer.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
	scaleWeights(layer.WeightStore.Master, scale)
}

func (n *VolumetricNetwork) InitConvTransposedCell(z, y, x, l int, ltype LayerType, inChannels, filters, kSize int, dtype DType, scale float32) {
	layer := n.GetLayer(z, y, x, l)
	layer.Type = ltype
	layer.InputChannels = inChannels
	layer.Filters = filters
	layer.KernelSize = kSize
	layer.DType = dtype
	
	wCount := inChannels * filters * kSize * kSize
	if ltype == LayerConvTransposed3D { wCount *= kSize }
	layer.WeightStore = NewWeightStore(wCount)
	layer.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
	scaleWeights(layer.WeightStore.Master, scale)
}

func (n *VolumetricNetwork) InitLayerNormCell(z, y, x, l int, size int, dtype DType) {
	layer := n.GetLayer(z, y, x, l)
	layer.Type = LayerLayerNorm
	layer.InputHeight = size
	layer.OutputHeight = size
	layer.DType = dtype
	layer.WeightStore = NewWeightStore(size * 2) // Gamma + Beta
	for i := range layer.WeightStore.Master { layer.WeightStore.Master[i] = 1.0 }
}

func (n *VolumetricNetwork) InitEmbeddingCell(z, y, x, l int, vocabSize, dModel int, dtype DType) {
	layer := n.GetLayer(z, y, x, l)
	layer.Type = LayerEmbedding
	layer.InputHeight = vocabSize
	layer.OutputHeight = dModel
	layer.DType = dtype
	layer.WeightStore = NewWeightStore(vocabSize * dModel)
	layer.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
}

func (n *VolumetricNetwork) InitKMeansCell(z, y, x, l int, numClusters, dModel int, dtype DType) {
	layer := n.GetLayer(z, y, x, l)
	layer.Type = LayerKMeans
	layer.InputHeight = dModel
	layer.OutputHeight = numClusters
	layer.DType = dtype
	layer.WeightStore = NewWeightStore(numClusters * dModel)
	layer.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
}

// =============================================================================
// Architecture Configuration
// =============================================================================

type ArchConfig struct {
	ID           int            `json:"id"`
	Name         string         `json:"name"`
	GridDepth    int            `json:"gridDepth"`
	GridRows     int            `json:"gridRows"`
	GridCols     int            `json:"gridCols"`
	LayersPerCell int           `json:"layersPerCell"`
	DModel       int            `json:"dModel"`
	NumHeads     int            `json:"numHeads"`
	Activation   ActivationType `json:"activation"`
	DType        DType          `json:"dtype"`
	InitScale    float32        `json:"initScale"`
}

// =============================================================================
// Brain Initializers
// =============================================================================

func (n *VolumetricNetwork) InitDenseCell(z, y, x, l int, dModel int, act ActivationType, scale float32) {
	layer := n.GetLayer(z, y, x, l)
	layer.Type = LayerDense
	layer.Activation = act
	layer.InputHeight = dModel
	layer.OutputHeight = dModel
	
	wCount := dModel * dModel
	layer.WeightStore = NewWeightStore(wCount)
	layer.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
	scaleWeights(layer.WeightStore.Master, scale)
}

func (n *VolumetricNetwork) InitMHACell(z, y, x, l int, dModel, numHeads int, scale float32) {
	layer := n.GetLayer(z, y, x, l)
	layer.Type = LayerMultiHeadAttention
	layer.DModel = dModel
	layer.NumHeads = numHeads
	layer.NumKVHeads = numHeads
	layer.HeadDim = dModel / numHeads
	layer.SeqLength = 1
	layer.InputHeight = dModel
	layer.OutputHeight = dModel

	kv := layer.NumKVHeads * layer.HeadDim
	wCount := 2*dModel*dModel + 2*dModel*kv + 2*dModel + 2*kv
	layer.WeightStore = NewWeightStore(wCount)
	layer.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
	scaleWeights(layer.WeightStore.Master, scale)
}

func (n *VolumetricNetwork) InitRNNCell(z, y, x, l int, dModel int, scale float32) {
	layer := n.GetLayer(z, y, x, l)
	layer.Type = LayerRNN
	layer.InputChannels = dModel
	layer.InputHeight = dModel
	layer.OutputHeight = dModel
	layer.SeqLength = 1
	
	wCount := dModel*dModel + dModel*dModel + dModel
	layer.WeightStore = NewWeightStore(wCount)
	layer.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
	scaleWeights(layer.WeightStore.Master, scale)
}

func (n *VolumetricNetwork) InitLSTMCell(z, y, x, l int, dModel int, scale float32) {
	layer := n.GetLayer(z, y, x, l)
	layer.Type = LayerLSTM
	layer.InputChannels = dModel
	layer.InputHeight = dModel
	layer.OutputHeight = dModel
	layer.SeqLength = 1
	
	gate := dModel*dModel + dModel*dModel + dModel
	layer.WeightStore = NewWeightStore(4 * gate)
	layer.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
	scaleWeights(layer.WeightStore.Master, scale)
}

// =============================================================================
// Network Builders
// =============================================================================

func BuildSequentialNetwork(numLayers int, dModel int, act ActivationType, dtype DType) *VolumetricNetwork {
	n := NewVolumetricNetwork(1, 1, 1, numLayers)
	for l := 0; l < numLayers; l++ {
		layer := n.GetLayer(0, 0, 0, l)
		layer.DType = dtype
		n.InitDenseCell(0, 0, 0, l, dModel, act, 0.5)
	}
	return n
}

// BuildTransformerNetwork creates a stack of Transformer blocks.
func BuildTransformerNetwork(numBlocks int, dModel int, numHeads int, dtype DType) *VolumetricNetwork {
	// Each block: MHA -> RMSNorm -> SwiGLU -> RMSNorm
	layersPerBlock := 4
	n := NewVolumetricNetwork(1, 1, 1, numBlocks*layersPerBlock)
	
	for b := 0; b < numBlocks; b++ {
		base := b * layersPerBlock
		n.InitMHACell(0, 0, 0, base, dModel, numHeads, 0.02)
		
		l1 := n.GetLayer(0, 0, 0, base+1)
		l1.Type = LayerRMSNorm
		l1.InputHeight = dModel
		l1.DType = dtype
		
		l2 := n.GetLayer(0, 0, 0, base+2)
		l2.Type = LayerSwiGLU
		l2.InputHeight = dModel
		l2.OutputHeight = dModel * 4
		l2.DType = dtype
		inter := l2.OutputHeight
		l2.WeightStore = NewWeightStore(dModel*inter*3 + inter*2 + dModel)
		l2.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
		
		l3 := n.GetLayer(0, 0, 0, base+3)
		l3.Type = LayerRMSNorm
		l3.InputHeight = dModel
		l3.DType = dtype
	}
	return n
}

// BuildCNN creates a simple convolutional network.
func BuildCNN(inputSize, numClasses int, dtype DType) *VolumetricNetwork {
	n := NewVolumetricNetwork(1, 1, 1, 4)
	
	// Conv1
	l0 := n.GetLayer(0, 0, 0, 0)
	l0.Type = LayerCNN2
	l0.InputChannels = 1
	l0.Filters = 16
	l0.KernelSize = 3
	l0.DType = dtype
	l0.WeightStore = NewWeightStore(16 * 1 * 3 * 3)
	l0.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
	
	// Conv2
	l1 := n.GetLayer(0, 0, 0, 1)
	l1.Type = LayerCNN2
	l1.InputChannels = 16
	l1.Filters = 32
	l1.KernelSize = 3
	l1.DType = dtype
	l1.WeightStore = NewWeightStore(32 * 16 * 3 * 3)
	l1.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
	
	// Flatten + Dense
	l2 := n.GetLayer(0, 0, 0, 2)
	l2.Type = LayerDense
	l2.InputHeight = 32 * (inputSize - 4) * (inputSize - 4)
	l2.OutputHeight = 128
	l2.DType = dtype
	l2.WeightStore = NewWeightStore(l2.InputHeight * 128)
	l2.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
	
	// Output
	l3 := n.GetLayer(0, 0, 0, 3)
	l3.Type = LayerSoftmax
	l3.InputHeight = 128
	l3.OutputHeight = numClasses
	l3.DType = dtype
	
	return n
}

// BuildRandomNetwork generates a diverse VolumetricNetwork.
func BuildRandomNetwork(depth, rows, cols, lpc int, dModel int) *VolumetricNetwork {
	n := NewVolumetricNetwork(depth, rows, cols, lpc)
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	for i := range n.Layers {
		l := &n.Layers[i]
		l.DType = DTypeFloat32
		
		bt := r.Intn(10) 
		switch bt {
		case 0: // MHA
			n.InitMHACell(l.Z, l.Y, l.X, l.L, dModel, 4, 0.02)
		case 1: // RNN
			n.InitRNNCell(l.Z, l.Y, l.X, l.L, dModel, 0.02)
		case 2: // LSTM
			n.InitLSTMCell(l.Z, l.Y, l.X, l.L, dModel, 0.02)
		case 3: // SwiGLU
			l.Type = LayerSwiGLU
			l.InputHeight = dModel
			inter := dModel * 2
			l.OutputHeight = inter
			l.WeightStore = NewWeightStore(dModel*inter*3 + inter*2 + dModel)
			l.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
		case 4: // RMSNorm
			l.Type = LayerRMSNorm
			l.InputHeight = dModel
		case 5: // CNN2
			n.InitCNNCell(l.Z, l.Y, l.X, l.L, LayerCNN2, 1, 16, 3, DTypeFloat32, 0.02)
		case 6: // LayerNorm
			n.InitLayerNormCell(l.Z, l.Y, l.X, l.L, dModel, DTypeFloat32)
		case 7: // Embedding
			n.InitEmbeddingCell(l.Z, l.Y, l.X, l.L, 100, dModel, DTypeFloat32)
		case 8: // ConvTransposed2D
			n.InitConvTransposedCell(l.Z, l.Y, l.X, l.L, LayerConvTransposed2D, 16, 8, 3, DTypeFloat32, 0.02)
		default: // Dense
			n.InitDenseCell(l.Z, l.Y, l.X, l.L, dModel, ActivationReLU, 0.02)
		}
	}
	return n
}

// Helper to scale weights.
func scaleWeights(w []float32, scale float32) {
	for i := range w { w[i] *= scale }
}
