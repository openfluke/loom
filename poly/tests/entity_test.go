package poly_test

import (
	"bytes"
	"math"
	"strings"
	"testing"

	. "github.com/openfluke/loom/poly"
)

func TestEntityRoundTripMatchesJSONPersistence(t *testing.T) {
	for _, dtype := range []DType{
		DTypeFloat32, DTypeFloat16, DTypeBFloat16,
		DTypeInt8, DTypeInt4, DTypeBinary, DTypeTernary,
	} {
		t.Run(dtype.String(), func(t *testing.T) {
			net := persistenceDenseNetwork(dtype)
			layer := net.GetLayer(0, 0, 0, 0)
			layer.WeightStore.Morph(dtype)

			entityWire, err := SerializeEntity(net)
			if err != nil {
				t.Fatal(err)
			}
			jsonWire, err := SerializeNetwork(net)
			if err != nil {
				t.Fatal(err)
			}
			if len(entityWire) >= len(jsonWire) {
				t.Fatalf("entity size %d should be smaller than JSON %d", len(entityWire), len(jsonWire))
			}

			reloaded, err := DeserializeEntity(entityWire)
			if err != nil {
				t.Fatal(err)
			}
			got := reloaded.GetLayer(0, 0, 0, 0)
			if got.DType != dtype {
				t.Fatalf("dtype = %v, want %v", got.DType, dtype)
			}
			wantActive := layer.WeightStore.Versions[dtype]
			if wantActive == nil {
				wantActive = layer.WeightStore.GetNative(dtype)
			}
			gotActive := got.WeightStore.Versions[dtype]
			if gotActive == nil {
				gotActive = got.WeightStore.GetNative(dtype)
			}
			if !NativeWeightsEncoded(wantActive, gotActive, dtype) {
				t.Fatal("native weight blob mismatch after entity round-trip")
			}
			if dtype == DTypeInt8 {
				reloaded.UseGPU = false
				in := NewTensor[float32](1, 2)
				in.Data[0], in.Data[1] = 0.2, -0.1
				ForwardPolymorphic(reloaded, in)
				if got.WeightStore.GetActive(dtype) == nil {
					t.Fatal("GetActive returned nil after int8 entity reload forward")
				}
			}
		})
	}
}

func TestEntityIdempotentBytes(t *testing.T) {
	net := persistenceDenseNetwork(DTypeInt4)
	net.GetLayer(0, 0, 0, 0).WeightStore.Morph(DTypeInt4)

	first, err := SerializeEntity(net)
	if err != nil {
		t.Fatal(err)
	}
	reloaded, err := DeserializeEntity(first)
	if err != nil {
		t.Fatal(err)
	}
	second, err := SerializeEntity(reloaded)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("entity save→load→save changed bytes (%d vs %d)", len(first), len(second))
	}
}

func TestEntityLayerSelectiveLoad(t *testing.T) {
	net := NewVolumetricNetwork(1, 1, 2, 1)
	for i := range net.Layers {
		l := &net.Layers[i]
		l.Type = LayerDense
		l.Activation = ActivationLinear
		l.DType = DTypeFloat32
		l.InputHeight = 2
		l.OutputHeight = 2
		l.WeightStore = NewWeightStore(4)
		copy(l.WeightStore.Master, deterministicWeights(4))
	}

	wire, err := SerializeEntity(net)
	if err != nil {
		t.Fatal(err)
	}
	partial, err := DeserializeEntityLayer(wire, 1)
	if err != nil {
		t.Fatal(err)
	}
	l0 := partial.GetLayer(0, 0, 0, 0)
	l1 := partial.GetLayer(0, 0, 1, 0)
	if l0.WeightStore != nil && (len(l0.WeightStore.Master) > 0 || l0.WeightStore.Versions[DTypeFloat32] != nil) {
		t.Fatal("layer 0 should not have weights when selectively loading layer 1 only")
	}
	if l1.WeightStore == nil || l1.WeightStore.Versions[DTypeFloat32] == nil && len(l1.WeightStore.Master) == 0 {
		t.Fatal("layer 1 should have loaded entity blob")
	}
}

func TestEntityTransformerRoundTrip(t *testing.T) {
	dims := HFDecoderDims{
		NumLayers:        1,
		HiddenSize:       4,
		NumHeads:         2,
		NumKVHeads:       2,
		HeadDim:          2,
		QueryDim:         4,
		KVDim:            4,
		IntermediateSize: 8,
		RMSNormEps:       1e-5,
		RoPEFreqBase:     10000,
		Activation:       ActivationSilu,
	}
	net := NewVolumetricNetwork(1, 1, 1, 4)
	InitHFDecoderBlocks(net, dims)
	for i := range net.Layers {
		if net.Layers[i].WeightStore != nil {
			copy(net.Layers[i].WeightStore.Master, deterministicWeights(len(net.Layers[i].WeightStore.Master)))
		}
	}

	hidden := dims.HiddenSize
	vocab := 8
	embeddings := deterministicWeights(vocab * hidden)
	finalNorm := deterministicWeights(hidden)
	et := NewEntityTransformer(net, HFArchLlamaStyleDecoder, dims, embeddings, embeddings, finalNorm, true)
	et.WeightDType = DTypeInt4
	wire, err := SerializeEntityTransformer(et)
	if err != nil {
		t.Fatal(err)
	}
	reloaded, err := DeserializeEntityTransformer(wire)
	if err != nil {
		t.Fatal(err)
	}
	if len(reloaded.Embeddings) != len(embeddings) {
		t.Fatalf("embeddings len = %d, want %d", len(reloaded.Embeddings), len(embeddings))
	}
	for i := range embeddings {
		if reloaded.Embeddings[i] != embeddings[i] {
			t.Fatalf("embeddings[%d] = %v, want %v", i, reloaded.Embeddings[i], embeddings[i])
		}
	}
	if !reloaded.LMHeadTied {
		t.Fatal("expected tied lm_head")
	}
	if &reloaded.LMHead[0] != &reloaded.Embeddings[0] {
		t.Fatal("tied lm_head should share embeddings backing array")
	}
	if len(reloaded.FinalNorm) != len(finalNorm) {
		t.Fatalf("final_norm len = %d, want %d", len(reloaded.FinalNorm), len(finalNorm))
	}
	for i := range finalNorm {
		if reloaded.FinalNorm[i] != finalNorm[i] {
			t.Fatalf("final_norm[%d] = %v, want %v", i, reloaded.FinalNorm[i], finalNorm[i])
		}
	}
	if len(reloaded.Network.Layers) != 4 {
		t.Fatalf("layers = %d, want 4", len(reloaded.Network.Layers))
	}
	if reloaded.WeightDType != DTypeInt4 {
		t.Fatalf("WeightDType = %v, want INT4", reloaded.WeightDType)
	}
	l1 := reloaded.Network.Layers[1] // MHA
	if l1.WeightStore == nil || !l1.WeightStore.HasAnyQ4_0() {
		t.Fatal("MHA layer missing baked Q4_0 after INT4 entity load")
	}
	for _, key := range []DType{WeightMHAQuery, WeightMHAKey, WeightMHAValue, WeightMHAProjection} {
		if !l1.WeightStore.HasQ4_0Component(key) {
			t.Fatalf("MHA missing Q4_0 component %v", key)
		}
	}
	swiglu := reloaded.Network.Layers[3]
	if !swiglu.WeightStore.HasAnyQ4_0() {
		t.Fatal("SwiGLU layer missing baked Q4_0 after INT4 entity load")
	}
	if len(swiglu.WeightStore.Master) == 0 {
		t.Fatal("SwiGLU missing bias tail in Master after INT4 entity load")
	}
	hdr, err := ParseEntityHeader(wire)
	if err != nil {
		t.Fatal(err)
	}
	var q4blobs int
	for _, b := range hdr.Blobs {
		if strings.Contains(b.Path, ".q4_0.") {
			q4blobs++
		}
	}
	if q4blobs == 0 {
		t.Fatal("INT4 entity wire missing q4_0 weight blobs")
	}
	norm := reloaded.Network.Layers[0]
	if norm.Type != LayerRMSNorm {
		t.Fatalf("layer 0 type = %v, want RMSNorm", norm.Type)
	}
	if norm.DType != DTypeFloat32 {
		t.Fatalf("RMSNorm DType = %v, want FP32", norm.DType)
	}
	wantNorm := net.Layers[0].WeightStore.Master
	for i := range wantNorm {
		if norm.WeightStore.Master[i] != wantNorm[i] {
			t.Fatalf("RMSNorm weight[%d] corrupted by INT4 entity round-trip", i)
		}
	}
	PrepareEntityTransformerInference(reloaded)

	netOnly, err := DeserializeEntity(wire)
	if err != nil {
		t.Fatal(err)
	}
	if len(netOnly.Layers) != 4 {
		t.Fatalf("network-only layers = %d, want 4", len(netOnly.Layers))
	}
}

func TestQ4_0GPUPackedRoundTrip(t *testing.T) {
	data := make([]float32, 128)
	for i := range data {
		data[i] = float32(i)*0.01 - 0.5
	}
	scales, packed := PackQ4_0GPU(data)
	out := DequantizeQ4_0GPUPacked(scales, packed)
	if len(out) != len(data) {
		t.Fatalf("dequant len = %d, want %d", len(out), len(data))
	}
	var maxErr float32
	for i := range data {
		d := float32(math.Abs(float64(data[i] - out[i])))
		if d > maxErr {
			maxErr = d
		}
	}
	if maxErr > 0.2 {
		t.Fatalf("max Q4_0 round-trip error %v too large", maxErr)
	}
}

func TestEntityQ4_0CPUMaterialize(t *testing.T) {
	dims := HFDecoderDims{
		NumLayers:        1,
		HiddenSize:       4,
		NumHeads:         2,
		NumKVHeads:       2,
		HeadDim:          2,
		QueryDim:         4,
		KVDim:            4,
		IntermediateSize: 8,
		RMSNormEps:       1e-5,
		RoPEFreqBase:     10000,
		Activation:       ActivationSilu,
	}
	net := NewVolumetricNetwork(1, 1, 1, 4)
	InitHFDecoderBlocks(net, dims)
	for i := range net.Layers {
		if net.Layers[i].WeightStore != nil {
			copy(net.Layers[i].WeightStore.Master, deterministicWeights(len(net.Layers[i].WeightStore.Master)))
		}
	}
	et := NewEntityTransformer(net, HFArchLlamaStyleDecoder, dims,
		deterministicWeights(8*4), deterministicWeights(8*4), deterministicWeights(4), true)
	et.WeightDType = DTypeInt4
	wire, err := SerializeEntityTransformer(et)
	if err != nil {
		t.Fatal(err)
	}
	reloaded, err := DeserializeEntityTransformer(wire)
	if err != nil {
		t.Fatal(err)
	}
	PrepareEntityTransformerInference(reloaded)
	mha := &reloaded.Network.Layers[1]
	if !mha.WeightStore.HasAnyQ4_0() {
		t.Fatal("expected baked Q4_0 on MHA")
	}
	mha.Network = reloaded.Network
	mha.SyncToCPU()
	if mha.DType != DTypeFloat32 {
		t.Fatalf("MHA DType = %v, want FP32 after CPU materialize", mha.DType)
	}
	q := mha.QueryDim
	if q == 0 {
		q = mha.DModel
	}
	qwSize := q * mha.DModel
	if len(mha.WeightStore.Master) < qwSize {
		t.Fatalf("MHA Master len = %d, want >= %d", len(mha.WeightStore.Master), qwSize)
	}
	weights := mha.WeightStore.GetActive(DTypeFloat32)
	if weights == nil {
		t.Fatal("MHA GetActive(FP32) nil after materialize")
	}
	w := weights.([]float32)
	var sumAbs float32
	for _, v := range w[:qwSize] {
		if v < 0 {
			sumAbs -= v
		} else {
			sumAbs += v
		}
	}
	if sumAbs == 0 {
		t.Fatal("MHA Q weights all zero after Q4_0 CPU materialize")
	}
}

func TestEntityBitNetTernaryBlobRoundTrip(t *testing.T) {
	m := &BitNetTernaryMatrix{
		Rows:     8,
		Cols:     4,
		RowWords: 1,
		Offset:   0,
		Scale:    0.05,
		Words:    []uint32{0x49249249, 0x92492492, 0x24924924, 0x49249249, 0x92492492, 0x24924924, 0x49249249, 0x92492492},
	}
	raw := EncodeEntityBitNetTernaryBlob(m)
	got, err := DecodeEntityBitNetTernaryBlob(raw)
	if err != nil {
		t.Fatal(err)
	}
	if got.Rows != m.Rows || got.Cols != m.Cols || got.Offset != m.Offset || got.Scale != m.Scale || len(got.Words) != len(m.Words) {
		t.Fatalf("round-trip mismatch: %+v vs %+v", got, m)
	}
}

// Large HF BitNet layouts use bitNetPackedKey(offset)=10000+offset with offset in the
// millions, so matrix keys exceed 20000. Collect must persist every *BitNetTernaryMatrix.
func TestEntityBitNetCollectLargeOffsetKeys(t *testing.T) {
	net := NewVolumetricNetwork(1, 1, 1, 4)
	InitHFDecoderBlocks(net, HFDecoderDims{
		NumLayers:        1,
		HiddenSize:       8,
		NumHeads:         2,
		NumKVHeads:       1,
		HeadDim:          4,
		QueryDim:         8,
		KVDim:            4,
		IntermediateSize: 16,
		Activation:       ActivationReLU2,
	})
	mha := &net.Layers[1]
	mha.WeightStore = NewWeightStore(1)
	mha.WeightStore.CPUPacked = map[DType]any{
		DType(10000): &BitNetTernaryMatrix{Rows: 8, Cols: 8, RowWords: 1, Offset: 0, Scale: 1, Words: []uint32{1, 2, 3, 4, 5, 6, 7, 8}},
		DType(6563600): &BitNetTernaryMatrix{Rows: 4, Cols: 8, RowWords: 1, Offset: 6553600, Scale: 2, Words: []uint32{9, 10, 11, 12}},
		DType(20000):       float32(1),
		DType(6573600): float32(2),
	}
	wire, err := SerializeEntityTransformer(&EntityTransformer{
		Network:      net,
		Architecture: HFArchLlamaStyleDecoder,
		HiddenSize:   8,
		VocabSize:    4,
		Dims: HFDecoderDims{
			NumLayers: 1, HiddenSize: 8, NumHeads: 2, NumKVHeads: 1, HeadDim: 4,
			QueryDim: 8, KVDim: 4, IntermediateSize: 16, Activation: ActivationReLU2,
		},
		WeightDType: DTypeTernary,
		Embeddings:  deterministicWeights(32),
		LMHead:      deterministicWeights(32),
		FinalNorm:   deterministicWeights(8),
		HasFinalNorm: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	hdr, err := ParseEntityHeader(wire)
	if err != nil {
		t.Fatal(err)
	}
	var keys []string
	for _, b := range hdr.Blobs {
		if strings.Contains(b.Path, "layers.1.bitnet_ternary.") {
			keys = append(keys, b.Path)
		}
	}
	if len(keys) != 2 {
		t.Fatalf("expected 2 bitnet blobs for MHA (Q+K offsets), got %d: %v", len(keys), keys)
	}
	reloaded, err := DeserializeEntityTransformer(wire)
	if err != nil {
		t.Fatal(err)
	}
	rl := &reloaded.Network.Layers[1]
	if _, ok := rl.WeightStore.GetBitNetTernaryMatrix(0, 8, 8); !ok {
		t.Fatal("missing Q matrix after entity round-trip")
	}
	if _, ok := rl.WeightStore.GetBitNetTernaryMatrix(6553600, 4, 8); !ok {
		t.Fatal("missing K matrix after entity round-trip (large offset key was dropped)")
	}
}

func TestEntityTopologyOnlySkipsWeightAllocation(t *testing.T) {
	dims := HFDecoderDims{
		NumLayers:        1,
		HiddenSize:       64,
		NumHeads:         4,
		NumKVHeads:       4,
		HeadDim:          16,
		QueryDim:         64,
		KVDim:            64,
		IntermediateSize: 128,
		RMSNormEps:       1e-5,
		RoPEFreqBase:     10000,
		Activation:       ActivationSilu,
	}
	net := NewVolumetricNetwork(1, 1, 1, 4)
	InitHFDecoderBlocks(net, dims)
	for i := range net.Layers {
		if net.Layers[i].WeightStore != nil {
			copy(net.Layers[i].WeightStore.Master, deterministicWeights(len(net.Layers[i].WeightStore.Master)))
		}
	}
	hidden := dims.HiddenSize
	vocab := 32
	embeddings := deterministicWeights(vocab * hidden)
	et := NewEntityTransformer(net, HFArchLlamaStyleDecoder, dims, embeddings, embeddings, deterministicWeights(hidden), true)
	et.WeightDType = DTypeTernary
	wire, err := SerializeEntityTransformer(et)
	if err != nil {
		t.Fatal(err)
	}
	topology, err := DeserializeEntityWithOptions(wire, &EntityLoadOptions{SkipLayerWeights: true})
	if err != nil {
		t.Fatal(err)
	}
	for i, l := range topology.Layers {
		if l.Type != LayerMultiHeadAttention && l.Type != LayerSwiGLU {
			continue
		}
		if l.WeightStore != nil && len(l.WeightStore.Master) > 0 {
			t.Fatalf("layer %d (%v): topology-only load allocated Master len=%d", i, l.Type, len(l.WeightStore.Master))
		}
	}
}
