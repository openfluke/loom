package poly

// HFDecoderDims holds decoded HF config dimensions for a Llama-style decoder stack.
type HFDecoderDims struct {
	NumLayers        int
	HiddenSize       int
	NumHeads         int
	NumKVHeads       int
	HeadDim          int
	QueryDim         int
	KVDim            int
	IntermediateSize int
	RMSNormEps       float64
	RoPEFreqBase     float64
	Activation       ActivationType
}

// InitHFDecoderBlocks wires numLayers transformer blocks (4 poly layers each):
// RMSNorm → MHA → RMSNorm → SwiGLU. net must be NewVolumetricNetwork(1,1,1, numLayers*4).
func InitHFDecoderBlocks(net *VolumetricNetwork, d HFDecoderDims) {
	numLayers := d.NumLayers
	hiddenSize := d.HiddenSize
	numHeads := d.NumHeads
	numKVHeads := d.NumKVHeads
	headDim := d.HeadDim
	queryDim := d.QueryDim
	kvDim := d.KVDim
	intermediateSize := d.IntermediateSize
	rmsNormEps := d.RMSNormEps
	ropeFreqBase := d.RoPEFreqBase

	for b := 0; b < numLayers; b++ {
		base := b * 4
		mhaSize := queryDim*hiddenSize + kvDim*hiddenSize + kvDim*hiddenSize + hiddenSize*queryDim + queryDim + kvDim + kvDim + hiddenSize
		mlpSize := (3 * hiddenSize * intermediateSize) + (2 * intermediateSize) + hiddenSize

		l0 := &net.Layers[base+0]
		l0.Type = LayerRMSNorm
		l0.InputHeight = hiddenSize
		l0.OutputHeight = hiddenSize
		l0.RMSNormEps = rmsNormEps
		l0.WeightStore = NewWeightStore(hiddenSize)

		l1 := &net.Layers[base+1]
		l1.Type = LayerMultiHeadAttention
		l1.DModel = hiddenSize
		l1.NumHeads = numHeads
		l1.NumKVHeads = numKVHeads
		l1.HeadDim = headDim
		l1.QueryDim = queryDim
		l1.RoPEFreqBase = ropeFreqBase
		l1.WeightStore = NewWeightStore(mhaSize)

		l2 := &net.Layers[base+2]
		l2.Type = LayerRMSNorm
		l2.InputHeight = hiddenSize
		l2.OutputHeight = hiddenSize
		l2.RMSNormEps = rmsNormEps
		l2.WeightStore = NewWeightStore(hiddenSize)

		l3 := &net.Layers[base+3]
		l3.Type = LayerSwiGLU
		l3.InputHeight = hiddenSize
		l3.OutputHeight = intermediateSize
		l3.Activation = d.Activation
		l3.WeightStore = NewWeightStore(mlpSize)
	}
}
