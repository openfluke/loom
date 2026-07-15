package poly

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// entityPayloadAcc tracks blob offsets while streaming ENTITY payload bytes to disk.
type entityPayloadAcc struct {
	w   io.Writer
	off int
}

func newEntityPayloadAcc(w io.Writer) *entityPayloadAcc {
	return &entityPayloadAcc{w: w}
}

func (a *entityPayloadAcc) Len() int { return a.off }

func (a *entityPayloadAcc) Write(p []byte) (int, error) {
	n, err := a.w.Write(p)
	a.off += n
	return n, err
}

func collectEntityGlobalBlobAcc(name string, weights []float32, acc *entityPayloadAcc, blobs *[]EntityWeightBlob) {
	if len(weights) == 0 {
		return
	}
	raw := EncodeWeightsRaw(weights)
	offset := acc.Len()
	_, _ = acc.Write(raw)
	*blobs = append(*blobs, EntityWeightBlob{
		Path:   "transformer." + name,
		Offset: uint64(offset),
		Length: uint64(len(raw)),
		DType:  DTypeFloat32.String(),
		Native: false,
	})
}

// collectEntityLMHeadQ4Acc bakes a Q4 logits matrix for CPU fused GEMV (alongside FP32 emb/head).
func collectEntityLMHeadQ4Acc(weights []float32, acc *entityPayloadAcc, blobs *[]EntityWeightBlob) {
	if len(weights) == 0 {
		return
	}
	scales, packed := PackQ4_0GPUParallel(weights)
	raw := encodeEntityQ4_0Blob(scales, packed)
	offset := acc.Len()
	_, _ = acc.Write(raw)
	*blobs = append(*blobs, EntityWeightBlob{
		Path:   "transformer.lm_head.q4_0",
		Offset: uint64(offset),
		Length: uint64(len(raw)),
		DType:  entityBlobDTypeQ4_0,
		Native: true,
	})
}

func collectEntityWeightBlobsAcc(l *VolumetricLayer, path string, acc *entityPayloadAcc, blobs *[]EntityWeightBlob, entityQuant DType) {
	if entityQuant == DTypeInt4 && (l.Type == LayerMultiHeadAttention || l.Type == LayerSwiGLU) {
		if l.WeightStore != nil && len(l.WeightStore.Master) > 0 {
			collectEntityQ4_0LayerAcc(l, path, acc, blobs)
		}
	} else if l.WeightStore != nil {
		dt := l.DType
		if l.Type == LayerRMSNorm || entityQuant == DTypeInt4 || entityQuant == DTypeTernary {
			dt = DTypeFloat32
		}
		scale := l.WeightStore.Scale
		active := l.WeightStore.Versions[dt]
		if active == nil && len(l.WeightStore.Master) > 0 {
			delete(l.WeightStore.Versions, dt)
			l.WeightStore.Morph(dt)
			active = l.WeightStore.Versions[dt]
		}
		if active == nil {
			active = l.WeightStore.GetNative(dt)
		}
		if active != nil {
			raw := EncodeNativeWeightsRaw(active, dt)
			if len(raw) > 0 {
				offset := acc.Len()
				_, _ = acc.Write(raw)
				*blobs = append(*blobs, EntityWeightBlob{
					Path:   path,
					Offset: uint64(offset),
					Length: uint64(len(raw)),
					DType:  dt.String(),
					Scale:  scale,
					Native: true,
				})
			}
		} else if len(l.WeightStore.Master) > 0 {
			raw := EncodeWeightsRaw(l.WeightStore.Master)
			offset := acc.Len()
			_, _ = acc.Write(raw)
			*blobs = append(*blobs, EntityWeightBlob{
				Path:   path,
				Offset: uint64(offset),
				Length: uint64(len(raw)),
				DType:  DTypeFloat32.String(),
				Scale:  scale,
				Native: false,
			})
		}
	}
	for i := range l.ParallelBranches {
		collectEntityWeightBlobsAcc(&l.ParallelBranches[i], fmt.Sprintf("%s.parallel_branches.%d", path, i), acc, blobs, entityQuant)
	}
	for i := range l.SequentialLayers {
		collectEntityWeightBlobsAcc(&l.SequentialLayers[i], fmt.Sprintf("%s.sequential_layers.%d", path, i), acc, blobs, entityQuant)
	}
	if l.MetaObservedLayer != nil {
		collectEntityWeightBlobsAcc(l.MetaObservedLayer, path+".meta_observed_layer", acc, blobs, entityQuant)
	}
}

func appendEntityQ4_0BlobAcc(path string, key DType, data []float32, acc *entityPayloadAcc, blobs *[]EntityWeightBlob) {
	if len(data) == 0 {
		return
	}
	scales, packed := PackQ4_0GPU(data)
	raw := encodeEntityQ4_0Blob(scales, packed)
	offset := acc.Len()
	_, _ = acc.Write(raw)
	*blobs = append(*blobs, EntityWeightBlob{
		Path:   fmt.Sprintf("%s.q4_0.%d", path, int(key)),
		Offset: uint64(offset),
		Length: uint64(len(raw)),
		DType:  entityBlobDTypeQ4_0,
		Native: true,
	})
}

func collectEntityQ4_0LayerAcc(l *VolumetricLayer, path string, acc *entityPayloadAcc, blobs *[]EntityWeightBlob) {
	if l.WeightStore == nil || len(l.WeightStore.Master) == 0 {
		return
	}
	switch l.Type {
	case LayerMultiHeadAttention:
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
		if len(w) < qwSize+kwSize+vwSize+owSize {
			return
		}
		appendEntityQ4_0BlobAcc(path, WeightMHAQuery, w[0:qwSize], acc, blobs)
		appendEntityQ4_0BlobAcc(path, WeightMHAKey, w[qwSize:qwSize+kwSize], acc, blobs)
		appendEntityQ4_0BlobAcc(path, WeightMHAValue, w[qwSize+kwSize:qwSize+kwSize+vwSize], acc, blobs)
		appendEntityQ4_0BlobAcc(path, WeightMHAProjection, w[qwSize+kwSize+vwSize:qwSize+kwSize+vwSize+owSize], acc, blobs)
		collectEntityMHANormBlobsAcc(l, path, acc, blobs)
	case LayerSwiGLU:
		h, inter := l.InputHeight, l.OutputHeight
		wSize := h * inter
		w := l.WeightStore.Master
		if len(w) < 3*wSize+2*inter+h {
			return
		}
		appendEntityQ4_0BlobAcc(path, DType(100), w[0:wSize], acc, blobs)
		appendEntityQ4_0BlobAcc(path, DType(101), w[wSize:2*wSize], acc, blobs)
		appendEntityQ4_0BlobAcc(path, DType(102), w[2*wSize:3*wSize], acc, blobs)
		bias := append([]float32(nil), w[3*wSize:3*wSize+2*inter+h]...)
		raw := EncodeWeightsRaw(bias)
		offset := acc.Len()
		_, _ = acc.Write(raw)
		*blobs = append(*blobs, EntityWeightBlob{
			Path:   path + ".biases",
			Offset: uint64(offset),
			Length: uint64(len(raw)),
			DType:  DTypeFloat32.String(),
			Native: false,
		})
	}
}

func collectEntityMHANormBlobsAcc(l *VolumetricLayer, path string, acc *entityPayloadAcc, blobs *[]EntityWeightBlob) {
	appendEntityFP32AuxBlobAcc(path+".q_norm", l.QNormWeight, acc, blobs)
	appendEntityFP32AuxBlobAcc(path+".k_norm", l.KNormWeight, acc, blobs)
}

func appendEntityFP32AuxBlobAcc(path string, weights []float32, acc *entityPayloadAcc, blobs *[]EntityWeightBlob) {
	if len(weights) == 0 {
		return
	}
	raw := EncodeWeightsRaw(weights)
	offset := acc.Len()
	_, _ = acc.Write(raw)
	*blobs = append(*blobs, EntityWeightBlob{
		Path:   path,
		Offset: uint64(offset),
		Length: uint64(len(raw)),
		DType:  DTypeFloat32.String(),
		Native: false,
	})
}

// releaseEntityConvertLayerWeights drops encoded layer weights so convert peak stays ~one block.
func releaseEntityConvertLayerWeights(l *VolumetricLayer) {
	if l == nil || l.WeightStore == nil {
		return
	}
	ws := l.WeightStore
	ws.Master = nil
	ws.Versions = nil
	ws.Q4_0Scales = nil
	ws.Q4_0Packed = nil
	ws.CPUPacked = nil
}

func buildEntityTransformerSpecFromImport(
	arch HFArchitectureKind,
	dims HFDecoderDims,
	embeddings, lmHead, finalNorm []float32,
	hasFinalNorm, lmHeadTied bool,
	weightDType DType,
) *EntityTransformerSpec {
	if weightDType == 0 {
		weightDType = DTypeFloat32
	}
	hiddenSize := dims.HiddenSize
	if hiddenSize <= 0 && len(finalNorm) > 0 {
		hiddenSize = len(finalNorm)
	}
	vocabSize := 0
	if hiddenSize > 0 && len(embeddings) > 0 {
		vocabSize = len(embeddings) / hiddenSize
	}
	return &EntityTransformerSpec{
		Architecture: arch.String(),
		HiddenSize:   hiddenSize,
		VocabSize:    vocabSize,
		LMHeadTied:   lmHeadTied,
		HasFinalNorm: hasFinalNorm,
		WeightDType:  weightDType.String(),
		Dims: &EntityTransformerDimsSpec{
			NumLayers:        dims.NumLayers,
			NumHeads:         dims.NumHeads,
			NumKVHeads:       dims.NumKVHeads,
			HeadDim:          dims.HeadDim,
			QueryDim:         dims.QueryDim,
			KVDim:            dims.KVDim,
			IntermediateSize: dims.IntermediateSize,
			RMSNormEps:       dims.RMSNormEps,
			RoPEFreqBase:     dims.RoPEFreqBase,
			Activation:       dims.Activation.String(),
		},
	}
}

func writeEntityWireStreaming(path string, net *VolumetricNetwork, trSpec *EntityTransformerSpec, blobs []EntityWeightBlob, payloadPath string) error {
	spec := BuildPersistenceNetworkSpec(net)
	canonicalEntityTopology(&spec)
	stripPersistenceWeights(&spec)
	header := entityHeaderDoc{
		FormatVersion: entityFormatVersion,
		Network:       spec,
		Transformer:   trSpec,
		Blobs:         blobs,
	}
	headerJSON, err := json.Marshal(header)
	if err != nil {
		return err
	}
	if len(headerJSON) > entityHeaderMaxSize {
		return fmt.Errorf("entity header too large: %d bytes", len(headerJSON))
	}
	payloadF, err := os.Open(payloadPath)
	if err != nil {
		return err
	}
	defer payloadF.Close()

	out, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			_ = out.Close()
			_ = os.Remove(path)
		}
	}()

	if _, err = out.Write([]byte(entityMagic)); err != nil {
		return err
	}
	var ver [2]byte
	binary.LittleEndian.PutUint16(ver[:], entityFormatVersion)
	if _, err = out.Write(ver[:]); err != nil {
		return err
	}
	if _, err = out.Write([]byte{0, 0}); err != nil {
		return err
	}
	var hlen [8]byte
	binary.LittleEndian.PutUint64(hlen[:], uint64(len(headerJSON)))
	if _, err = out.Write(hlen[:]); err != nil {
		return err
	}
	if _, err = out.Write(headerJSON); err != nil {
		return err
	}
	if _, err = io.Copy(out, payloadF); err != nil {
		return err
	}
	return out.Close()
}
