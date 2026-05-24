package poly

// EnsureTrainingWeights allocates or restores FP32 Master slices for every layer
// so training, ForceMorph, and Master-based optimizers can run. Clears
// inference-only mode for the duration of training.
func (n *VolumetricNetwork) EnsureTrainingWeights() {
	if n == nil {
		return
	}
	n.ReleaseFP32MasterWhenIdle = false
	for i := range n.Layers {
		ensureTrainingWeightsLayer(&n.Layers[i])
	}
}

// SyncInferenceWeights morphs each layer to its active DType and, when
// ReleaseFP32MasterWhenIdle is set, drops FP32 Master so forward-only RAM
// reflects native Versions only.
func (n *VolumetricNetwork) SyncInferenceWeights() {
	if n == nil || !n.ReleaseFP32MasterWhenIdle {
		return
	}
	for i := range n.Layers {
		syncInferenceWeightsLayer(&n.Layers[i])
	}
}

func ensureTrainingWeightsLayer(l *VolumetricLayer) {
	if l == nil {
		return
	}
	if l.WeightStore != nil {
		l.WeightStore.EnsureFP32Master(l.DType)
	}
	for i := range l.ParallelBranches {
		ensureTrainingWeightsLayer(&l.ParallelBranches[i])
	}
	for i := range l.SequentialLayers {
		ensureTrainingWeightsLayer(&l.SequentialLayers[i])
	}
	if l.MetaObservedLayer != nil {
		ensureTrainingWeightsLayer(l.MetaObservedLayer)
	}
}

func syncInferenceWeightsLayer(l *VolumetricLayer) {
	if l == nil {
		return
	}
	if l.WeightStore != nil {
		ws := l.WeightStore
		if len(ws.Master) > 0 {
			ws.Morph(l.DType)
			ws.commitFP32Snapshot()
		}
		ws.ReleaseFP32Master()
	}
	for i := range l.ParallelBranches {
		syncInferenceWeightsLayer(&l.ParallelBranches[i])
	}
	for i := range l.SequentialLayers {
		syncInferenceWeightsLayer(&l.SequentialLayers[i])
	}
	if l.MetaObservedLayer != nil {
		syncInferenceWeightsLayer(l.MetaObservedLayer)
	}
}

// EnsureFP32Master rebuilds Master from the native Versions entry for dtype when
// Master was released for inference-only execution.
func (ws *WeightStore) EnsureFP32Master(dtype DType) {
	if ws == nil || len(ws.Master) > 0 {
		return
	}
	n := ws.nativeWeightCount(dtype)
	if n == 0 {
		n = ws.nativeWeightCount(DTypeFloat32)
		dtype = DTypeFloat32
	}
	if n == 0 {
		return
	}
	ws.Master = AlignedFloat32(n)
	ws.Unpack(dtype)
}

// ReleaseFP32Master frees the FP32 Master slice. Forward continues via
// native Versions (see GetActive).
func (ws *WeightStore) ReleaseFP32Master() {
	if ws == nil {
		return
	}
	ws.Master = nil
}

func (ws *WeightStore) commitFP32Snapshot() {
	if ws == nil || len(ws.Master) == 0 {
		return
	}
	if ws.Versions == nil {
		ws.Versions = make(map[DType]any)
	}
	snap := make([]float32, len(ws.Master))
	copy(snap, ws.Master)
	ws.Versions[DTypeFloat32] = snap
}

// WeightCount returns the number of scalar weights (Master if retained, else native).
func (ws *WeightStore) WeightCount(dtype DType) int {
	if ws == nil {
		return 0
	}
	if n := len(ws.Master); n > 0 {
		return n
	}
	return ws.nativeWeightCount(dtype)
}

func (ws *WeightStore) nativeWeightCount(dtype DType) int {
	if ws == nil {
		return 0
	}
	if v := ws.Versions[dtype]; v != nil {
		return weightValueLen(v)
	}
	if v := ws.Versions[DTypeFloat32]; v != nil {
		return weightValueLen(v)
	}
	for _, v := range ws.Versions {
		if n := weightValueLen(v); n > 0 {
			return n
		}
	}
	return 0
}

func weightValueLen(v any) int {
	switch w := v.(type) {
	case []float32:
		return len(w)
	case []float64:
		return len(w)
	case []int64:
		return len(w)
	case []uint64:
		return len(w)
	case []int32:
		return len(w)
	case []uint32:
		return len(w)
	case []int16:
		return len(w)
	case []uint16:
		return len(w)
	case []uint8:
		return len(w)
	case []int8:
		return len(w)
	default:
		return 0
	}
}

// AccountingWeightBytes returns bytes charged to weight storage: FP32 Master when
// retained for training, otherwise the active native Versions slice.
func (ws *WeightStore) AccountingWeightBytes(dtype DType) uint64 {
	if ws == nil {
		return 0
	}
	if len(ws.Master) > 0 {
		return uint64(len(ws.Master)) * 4
	}
	if b := nativeValueBytes(ws.Versions[dtype]); b > 0 {
		return b
	}
	return nativeValueBytes(ws.Versions[DTypeFloat32])
}

func nativeValueBytes(v any) uint64 {
	switch w := v.(type) {
	case []float32:
		return uint64(len(w)) * 4
	case []float64:
		return uint64(len(w)) * 8
	case []int64:
		return uint64(len(w)) * 8
	case []uint64:
		return uint64(len(w)) * 8
	case []int32:
		return uint64(len(w)) * 4
	case []uint32:
		return uint64(len(w)) * 4
	case []int16:
		return uint64(len(w)) * 2
	case []uint16:
		return uint64(len(w)) * 2
	case []uint8:
		return uint64(len(w))
	case []int8:
		return uint64(len(w))
	default:
		return 0
	}
}

// NetworkAccountingWeightBytes sums per-layer weight storage (Master or native).
func NetworkAccountingWeightBytes(net *VolumetricNetwork) uint64 {
	if net == nil {
		return 0
	}
	var total uint64
	for i := range net.Layers {
		total += layerAccountingWeightBytes(&net.Layers[i])
	}
	return total
}

func layerAccountingWeightBytes(l *VolumetricLayer) uint64 {
	var n uint64
	if l.WeightStore != nil {
		n += l.WeightStore.AccountingWeightBytes(l.DType)
	}
	for i := range l.ParallelBranches {
		n += layerAccountingWeightBytes(&l.ParallelBranches[i])
	}
	for i := range l.SequentialLayers {
		n += layerAccountingWeightBytes(&l.SequentialLayers[i])
	}
	if l.MetaObservedLayer != nil {
		n += layerAccountingWeightBytes(l.MetaObservedLayer)
	}
	return n
}
