package poly

// mhaLayout describes [batch, seq, d_model] (or [seq, d_model]) tensor indexing for MHA.
type mhaLayout struct {
	batch, seqLen, dModel int
	elemStride            int // seqLen * dModel
}

func mhaParseLayout[T Numeric](layer *VolumetricLayer, input *Tensor[T]) mhaLayout {
	lay := mhaLayout{dModel: layer.DModel}
	switch len(input.Shape) {
	case 3:
		lay.batch = input.Shape[0]
		lay.seqLen = input.Shape[1]
		// Only use layer.SeqLength when the last dim is not d_model (legacy flat layouts).
		if input.Shape[2] != layer.DModel {
			if layer.SeqLength > 0 {
				lay.seqLen = layer.SeqLength
			} else if lay.batch > 0 {
				lay.seqLen = len(input.Data) / (lay.batch * layer.DModel)
			}
		}
	case 2:
		lay.batch = 1
		if input.Shape[1] == layer.DModel {
			lay.seqLen = input.Shape[0]
		} else if layer.SeqLength > 0 {
			lay.seqLen = layer.SeqLength
		} else {
			lay.seqLen = len(input.Data) / layer.DModel
		}
	default:
		lay.batch = 1
		if layer.SeqLength > 0 {
			lay.seqLen = layer.SeqLength
		} else {
			lay.seqLen = len(input.Data) / layer.DModel
		}
	}
	if lay.seqLen <= 0 {
		lay.seqLen = 1
	}
	lay.elemStride = lay.seqLen * lay.dModel
	return lay
}

func (lay mhaLayout) base(batch int) int { return batch * lay.elemStride }

func (lay mhaLayout) inIdx(batch, seq, j int) int { return lay.base(batch) + seq*lay.dModel + j }

func (lay mhaLayout) outIdx(batch, seq, i int) int { return lay.inIdx(batch, seq, i) }

// mhaResetKVCache clears KV state at the start of a feed-forward MHA pass (one layer, one batch).
func mhaResetKVCache(layer *VolumetricLayer) {
	layer.KVOffset = 0
	layer.KVCacheK = nil
	layer.KVCacheV = nil
}

// mhaPrepareKVForForward resets KV for full-sequence / multi-batch training passes,
// but keeps the cache when doing autoregressive decode (batch=1, seq=1, cache warm).
func mhaPrepareKVForForward[T Numeric](layer *VolumetricLayer, lay mhaLayout, msl, kvDim int) {
	incremental := lay.batch == 1 && lay.seqLen == 1 && layer.KVCacheK != nil && layer.KVOffset > 0
	if incremental {
		return
	}
	layer.KVCacheK = NewTensor[T](msl, kvDim)
	layer.KVCacheV = NewTensor[T](msl, kvDim)
	layer.KVOffset = 0
}
