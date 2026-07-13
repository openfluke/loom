package poly

import (
	"encoding/binary"
	"fmt"
	"math"
)

// PackNetworkWeights serializes every trainable host parameter of net into one
// little-endian FP32 byte blob (Loom’s Master / persistence space).
//
// Walks the full layer tree:
//   top-level Layers → ParallelBranches → SequentialLayers → FilterGateConfig → MetaObservedLayer
// and for each layer packs:
//   WeightStore.Master, WeightStore.Scale, QNormWeight, KNormWeight, InnerNormWeight
//
// After specialize/exact-dtype train, Call EnsureTrainingWeights (and SyncToCPU when needed)
// so Masters reflect current natives. Unpack writes Masters then ForceMorph(layer.DType)
// so any of the 21 numerical types can be restored for forward/exact train.
func PackNetworkWeights(net *VolumetricNetwork) ([]byte, error) {
	if net == nil {
		return nil, fmt.Errorf("poly: PackNetworkWeights nil network")
	}
	net.EnsureTrainingWeights()
	net.SyncToCPU()

	var floats []float32
	for i := range net.Layers {
		collectLayerFloats(&net.Layers[i], &floats)
	}
	if len(floats) == 0 {
		return nil, fmt.Errorf("poly: PackNetworkWeights no trainable floats found")
	}
	out := make([]byte, len(floats)*4)
	for i, v := range floats {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out, nil
}

// UnpackNetworkWeights restores a PackNetworkWeights blob onto an equal-layout net,
// then ForceMorph each layer to its DType (any supported numerical type).
func UnpackNetworkWeights(net *VolumetricNetwork, blob []byte) error {
	if net == nil {
		return fmt.Errorf("poly: UnpackNetworkWeights nil network")
	}
	if len(blob)%4 != 0 {
		return fmt.Errorf("poly: UnpackNetworkWeights blob length %d not multiple of 4", len(blob))
	}
	net.EnsureTrainingWeights()
	floats := make([]float32, len(blob)/4)
	for i := range floats {
		floats[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
	}
	off := 0
	for i := range net.Layers {
		n, err := scatterLayerFloats(&net.Layers[i], floats[off:])
		if err != nil {
			return fmt.Errorf("poly: UnpackNetworkWeights layer %d: %w", i, err)
		}
		off += n
	}
	if off != len(floats) {
		return fmt.Errorf("poly: UnpackNetworkWeights length mismatch used=%d have=%d", off, len(floats))
	}
	MorphNetworkToLayerDTypes(net)
	return nil
}

// MorphNetworkToLayerDTypes ForceMorphs every WeightStore to its layer.DType (recursive).
func MorphNetworkToLayerDTypes(net *VolumetricNetwork) {
	if net == nil {
		return
	}
	for i := range net.Layers {
		morphLayerTree(&net.Layers[i])
	}
}

// SetNetworkUseExactDType enables/disables native-dtype train/forward on the whole net.
func SetNetworkUseExactDType(net *VolumetricNetwork, exact bool) {
	if net == nil {
		return
	}
	net.UseExactDType = exact
}

// ApplyUniformDType sets every layer’s DType (recursive) and morphs Masters into that dtype.
func ApplyUniformDType(net *VolumetricNetwork, dtype DType) {
	if net == nil {
		return
	}
	net.EnsureTrainingWeights()
	for i := range net.Layers {
		applyLayerDType(&net.Layers[i], dtype)
	}
}

// WireNetworkLayers sets Network back-pointers on the full layer tree.
func WireNetworkLayers(net *VolumetricNetwork) {
	if net == nil {
		return
	}
	for i := range net.Layers {
		wireLayerTree(&net.Layers[i], net)
	}
}

func collectLayerFloats(l *VolumetricLayer, dst *[]float32) {
	if l == nil {
		return
	}
	if l.WeightStore != nil {
		ws := l.WeightStore
		ws.EnsureFP32Master(l.DType)
		*dst = append(*dst, ws.Master...)
		*dst = append(*dst, ws.Scale)
	}
	*dst = append(*dst, l.QNormWeight...)
	*dst = append(*dst, l.KNormWeight...)
	*dst = append(*dst, l.InnerNormWeight...)

	for i := range l.ParallelBranches {
		collectLayerFloats(&l.ParallelBranches[i], dst)
	}
	for i := range l.SequentialLayers {
		collectLayerFloats(&l.SequentialLayers[i], dst)
	}
	if l.FilterGateConfig != nil {
		collectLayerFloats(l.FilterGateConfig, dst)
	}
	if l.MetaObservedLayer != nil {
		collectLayerFloats(l.MetaObservedLayer, dst)
	}
}

func scatterLayerFloats(l *VolumetricLayer, src []float32) (int, error) {
	if l == nil {
		return 0, nil
	}
	off := 0
	if l.WeightStore != nil {
		ws := l.WeightStore
		ws.EnsureFP32Master(l.DType)
		n := len(ws.Master)
		if off+n+1 > len(src) {
			return 0, fmt.Errorf("short for WeightStore master+scale need=%d have=%d", n+1, len(src)-off)
		}
		copy(ws.Master, src[off:off+n])
		off += n
		ws.Scale = src[off]
		off++
	}
	if n := len(l.QNormWeight); n > 0 {
		if off+n > len(src) {
			return 0, fmt.Errorf("short for QNormWeight")
		}
		copy(l.QNormWeight, src[off:off+n])
		off += n
	}
	if n := len(l.KNormWeight); n > 0 {
		if off+n > len(src) {
			return 0, fmt.Errorf("short for KNormWeight")
		}
		copy(l.KNormWeight, src[off:off+n])
		off += n
	}
	if n := len(l.InnerNormWeight); n > 0 {
		if off+n > len(src) {
			return 0, fmt.Errorf("short for InnerNormWeight")
		}
		copy(l.InnerNormWeight, src[off:off+n])
		off += n
	}

	for i := range l.ParallelBranches {
		n, err := scatterLayerFloats(&l.ParallelBranches[i], src[off:])
		if err != nil {
			return 0, err
		}
		off += n
	}
	for i := range l.SequentialLayers {
		n, err := scatterLayerFloats(&l.SequentialLayers[i], src[off:])
		if err != nil {
			return 0, err
		}
		off += n
	}
	if l.FilterGateConfig != nil {
		n, err := scatterLayerFloats(l.FilterGateConfig, src[off:])
		if err != nil {
			return 0, err
		}
		off += n
	}
	if l.MetaObservedLayer != nil {
		n, err := scatterLayerFloats(l.MetaObservedLayer, src[off:])
		if err != nil {
			return 0, err
		}
		off += n
	}
	return off, nil
}

func morphLayerTree(l *VolumetricLayer) {
	if l == nil {
		return
	}
	if l.WeightStore != nil && len(l.WeightStore.Master) > 0 {
		l.WeightStore.ForceMorph(l.DType)
	}
	for i := range l.ParallelBranches {
		morphLayerTree(&l.ParallelBranches[i])
	}
	for i := range l.SequentialLayers {
		morphLayerTree(&l.SequentialLayers[i])
	}
	if l.FilterGateConfig != nil {
		morphLayerTree(l.FilterGateConfig)
	}
	if l.MetaObservedLayer != nil {
		morphLayerTree(l.MetaObservedLayer)
	}
}

func applyLayerDType(l *VolumetricLayer, dtype DType) {
	if l == nil {
		return
	}
	l.DType = dtype
	if l.WeightStore != nil {
		l.WeightStore.EnsureFP32Master(dtype)
		l.WeightStore.ForceMorph(dtype)
	}
	for i := range l.ParallelBranches {
		applyLayerDType(&l.ParallelBranches[i], dtype)
	}
	for i := range l.SequentialLayers {
		applyLayerDType(&l.SequentialLayers[i], dtype)
	}
	if l.FilterGateConfig != nil {
		applyLayerDType(l.FilterGateConfig, dtype)
	}
	if l.MetaObservedLayer != nil {
		applyLayerDType(l.MetaObservedLayer, dtype)
	}
}

func wireLayerTree(l *VolumetricLayer, net *VolumetricNetwork) {
	if l == nil {
		return
	}
	l.Network = net
	for i := range l.ParallelBranches {
		wireLayerTree(&l.ParallelBranches[i], net)
	}
	for i := range l.SequentialLayers {
		wireLayerTree(&l.SequentialLayers[i], net)
	}
	if l.FilterGateConfig != nil {
		wireLayerTree(l.FilterGateConfig, net)
	}
	if l.MetaObservedLayer != nil {
		wireLayerTree(l.MetaObservedLayer, net)
	}
}
