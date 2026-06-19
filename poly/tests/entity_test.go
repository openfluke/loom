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
	if l0.WeightStore == nil || len(l0.WeightStore.Master) == 0 {
		t.Fatal("layer 0 should have initialized weights from topology init")
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
