package poly

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"
	"strconv"
)

// InitSeededNetwork fills every layer from init seed (He-init per DeriveLayerSeed).
func InitSeededNetwork(net *VolumetricNetwork, initSeed uint64) {
	if net == nil {
		return
	}
	net.InitSeed = initSeed
	walkSeedLayers(net, func(l *VolumetricLayer, idx int, path string) {
		InitLayerWeightsSeeded(l, DeriveLayerSeed(initSeed, idx, path), 0)
	})
}

// InitLayerWeightsSeeded He-inits one layer; entityQuant applies MHA/SwiGLU morph when set.
func InitLayerWeightsSeeded(l *VolumetricLayer, layerSeed uint64, entityQuant DType) {
	if l == nil {
		return
	}
	switch l.Type {
	case LayerParallel, LayerSequential, LayerResidual, LayerSoftmax, LayerMetacognition:
		return
	}
	seedEnsureLayerWeightStore(l)
	if l.WeightStore == nil || len(l.WeightStore.Master) == 0 {
		return
	}
	switch l.Type {
	case LayerRMSNorm, LayerLayerNorm:
		for i := range l.WeightStore.Master {
			l.WeightStore.Master[i] = 1
		}
	default:
		InitWeightStoreHeSeeded(l.WeightStore, seedLayerInputSize(l), layerSeed)
	}
	seedApplyLayerDType(l, entityQuant)
}

// InitSeededEntity fills decoder layers + transformer globals from init seed.
func InitSeededEntity(et *EntityTransformer, initSeed uint64) error {
	if et == nil || et.Network == nil {
		return fmt.Errorf("seed: nil entity")
	}
	et.Network.InitSeed = initSeed
	walkSeedLayers(et.Network, func(l *VolumetricLayer, idx int, path string) {
		InitLayerWeightsSeeded(l, DeriveLayerSeed(initSeed, idx, path), et.WeightDType)
	})
	hidden := et.HiddenSize
	if hidden <= 0 {
		hidden = et.Dims.HiddenSize
	}
	if len(et.Embeddings) > 0 && hidden > 0 {
		InitFloat32HeSeeded(et.Embeddings, hidden, DeriveLayerSeed(initSeed, 0, "transformer.embeddings"))
	}
	if !et.LMHeadTied && len(et.LMHead) > 0 && hidden > 0 {
		InitFloat32HeSeeded(et.LMHead, hidden, DeriveLayerSeed(initSeed, 0, "transformer.lm_head"))
	} else if et.LMHeadTied {
		et.LMHead = et.Embeddings
	}
	if et.HasFinalNorm && len(et.FinalNorm) > 0 {
		for i := range et.FinalNorm {
			et.FinalNorm[i] = 1
		}
	}
	if et.WeightDType != 0 && et.WeightDType != DTypeFloat32 {
		MorphHFDecoderWeights(et.Network, et.WeightDType)
	}
	return nil
}

// EntityTransformerFingerprint hashes layer weights + transformer globals.
func EntityTransformerFingerprint(et *EntityTransformer) uint64 {
	if et == nil {
		return 0
	}
	h := fnv.New64a()
	if et.Network != nil {
		fp := NetworkWeightFingerprint(et.Network)
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], fp)
		_, _ = h.Write(buf[:])
	}
	seedWriteFloat32Hash(h, et.Embeddings)
	if !et.LMHeadTied {
		seedWriteFloat32Hash(h, et.LMHead)
	}
	seedWriteFloat32Hash(h, et.FinalNorm)
	return h.Sum64()
}

// NetworkWeightFingerprint hashes every layer master weight in order.
func NetworkWeightFingerprint(net *VolumetricNetwork) uint64 {
	if net == nil {
		return 0
	}
	h := fnv.New64a()
	for i := range net.Layers {
		fp := weightStoreFingerprint(net.Layers[i].WeightStore)
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], fp)
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}

func weightStoreFingerprint(ws *WeightStore) uint64 {
	if ws == nil {
		return 0
	}
	h := fnv.New64a()
	var buf [4]byte
	for _, v := range ws.Master {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}

func walkSeedLayers(net *VolumetricNetwork, fn func(l *VolumetricLayer, idx int, path string)) {
	if net == nil {
		return
	}
	for i := range net.Layers {
		walkSeedLayer(&net.Layers[i], i, fmt.Sprintf("layers.%d", i), fn)
	}
}

func walkSeedLayer(l *VolumetricLayer, idx int, path string, fn func(l *VolumetricLayer, idx int, path string)) {
	if l == nil {
		return
	}
	fn(l, idx, path)
	for i := range l.ParallelBranches {
		walkSeedLayer(&l.ParallelBranches[i], idx, path+".parallel_branches."+strconv.Itoa(i), fn)
	}
	for i := range l.SequentialLayers {
		walkSeedLayer(&l.SequentialLayers[i], idx, path+".sequential_layers."+strconv.Itoa(i), fn)
	}
	if l.MetaObservedLayer != nil {
		walkSeedLayer(l.MetaObservedLayer, idx, path+".meta_observed_layer", fn)
	}
}

func seedApplyLayerDType(l *VolumetricLayer, entityQuant DType) {
	dt := l.DType
	if dt == 0 {
		dt = DTypeFloat32
	}
	if entityQuant != 0 && entityQuant != DTypeFloat32 {
		if l.Type == LayerMultiHeadAttention || l.Type == LayerSwiGLU {
			dt = entityQuant
		}
	}
	if l.Type == LayerRMSNorm {
		dt = DTypeFloat32
	}
	l.DType = dt
	if dt != DTypeFloat32 && l.WeightStore != nil {
		l.WeightStore.Morph(dt)
	}
}

func seedEnsureLayerWeightStore(l *VolumetricLayer) {
	if l == nil || l.WeightStore != nil {
		return
	}
	wCount := seedLayerWeightCount(l)
	if wCount > 0 {
		l.WeightStore = NewWeightStore(wCount)
	}
}

func seedLayerWeightCount(l *VolumetricLayer) int {
	switch l.Type {
	case LayerDense:
		return l.InputHeight * l.OutputHeight
	case LayerRMSNorm:
		return l.InputHeight
	case LayerLayerNorm:
		return 2 * l.InputHeight
	case LayerMultiHeadAttention:
		q := l.QueryDim
		if q == 0 {
			q = l.DModel
		}
		kv := l.NumKVHeads * l.HeadDim
		if kv == 0 {
			kv = l.DModel
		}
		return q*l.DModel + kv*l.DModel + kv*l.DModel + l.DModel*q + q + kv + kv + l.DModel
	case LayerRNN:
		return l.InputHeight*l.OutputHeight + l.OutputHeight*l.OutputHeight + l.OutputHeight
	case LayerLSTM:
		gate := l.InputHeight*l.OutputHeight + l.OutputHeight*l.OutputHeight + l.OutputHeight
		return 4 * gate
	case LayerSwiGLU:
		return 3*l.InputHeight*l.OutputHeight + 2*l.OutputHeight + l.InputHeight
	case LayerCNN1, LayerCNN2, LayerCNN3:
		k := l.KernelSize
		if k == 0 {
			k = 1
		}
		n := l.Filters * l.InputChannels * k
		if l.Type == LayerCNN2 {
			n *= k
		}
		if l.Type == LayerCNN3 {
			n *= k * k
		}
		return n
	case LayerConvTransposed1D, LayerConvTransposed2D, LayerConvTransposed3D:
		k := l.KernelSize
		if k == 0 {
			k = 1
		}
		n := l.InputChannels * l.Filters * k
		if l.Type == LayerConvTransposed2D {
			n *= k
		}
		if l.Type == LayerConvTransposed3D {
			n *= k * k
		}
		return n
	case LayerEmbedding:
		return l.VocabSize * l.EmbeddingDim
	case LayerKMeans:
		return l.NumClusters * l.InputHeight
	default:
		return 0
	}
}

func seedLayerInputSize(l *VolumetricLayer) int {
	switch l.Type {
	case LayerDense, LayerSwiGLU, LayerRNN, LayerLSTM:
		if l.InputHeight > 0 {
			return l.InputHeight
		}
	case LayerMultiHeadAttention:
		if l.DModel > 0 {
			return l.DModel
		}
	case LayerCNN1, LayerCNN2, LayerCNN3:
		ic := l.InputChannels
		if ic <= 0 {
			ic = 1
		}
		return ic
	case LayerConvTransposed1D, LayerConvTransposed2D, LayerConvTransposed3D:
		if l.InputChannels > 0 {
			return l.InputChannels
		}
	case LayerEmbedding:
		if l.EmbeddingDim > 0 {
			return l.EmbeddingDim
		}
	case LayerKMeans:
		if l.InputHeight > 0 {
			return l.InputHeight
		}
	}
	return 1
}

func seedWriteFloat32Hash(h hashWriter, data []float32) {
	var buf [4]byte
	for _, v := range data {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
		_, _ = h.Write(buf[:])
	}
}

type hashWriter interface {
	Write([]byte) (int, error)
}
