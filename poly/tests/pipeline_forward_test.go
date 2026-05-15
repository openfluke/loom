package poly_test

import (
	"testing"

	. "github.com/openfluke/loom/poly"
)

func newTinyHFTransformer(numLayers, hidden int) *Transformer[float32] {
	net := NewVolumetricNetwork(1, 1, 1, numLayers*4)
	InitHFDecoderBlocks(net, HFDecoderDims{
		NumLayers:        numLayers,
		HiddenSize:       hidden,
		NumHeads:         2,
		NumKVHeads:       2,
		HeadDim:          hidden / 2,
		QueryDim:         hidden,
		KVDim:            hidden,
		IntermediateSize: hidden * 2,
		RMSNormEps:       1e-5,
		RoPEFreqBase:     10000,
		Activation:       ActivationSilu,
	})
	for i := range net.Layers {
		l := &net.Layers[i]
		if l.WeightStore == nil {
			continue
		}
		n := len(l.WeightStore.Master)
		if n == 0 {
			continue
		}
		w := deterministicWeights(n)
		copy(l.WeightStore.Master, w)
	}
	embeddings := deterministicWeights(32 * hidden)
	finalNorm := deterministicWeights(hidden)
	return NewTransformer[float32](net, embeddings, embeddings, finalNorm, Template{})
}

func TestPipelineDecodeAfterPrefillNoStall(t *testing.T) {
	const hidden = 8
	const numLayers = 2
	tr := newTinyHFTransformer(numLayers, hidden)
	tr.ForwardMode = TransformerForwardPipelineCPU

	prefillToks := []uint32{1, 2, 3}
	prefill := tr.TokensToTensor(prefillToks)

	tr.Reset()
	tr.ForwardMode = TransformerForwardNormal
	_ = tr.ForwardFull(prefill)

	kv := tr.Network.Layers[1].KVOffset
	if kv != len(prefillToks) {
		t.Fatalf("KVOffset after prefill: got %d want %d", kv, len(prefillToks))
	}

	decodeIn := tr.TokensToTensor([]uint32{4})
	tr.ForwardMode = TransformerForwardPipelineCPU
	got := tr.ForwardFull(decodeIn)
	if got == nil || len(got.Data) != hidden {
		t.Fatalf("pipeline decode output: len=%d", len(got.Data))
	}
	if !allFinite(got.Data) {
		t.Fatalf("pipeline decode non-finite: %v", got.Data)
	}

	tr.Reset()
	tr.ForwardMode = TransformerForwardNormal
	_ = tr.ForwardFull(prefill)
	want := tr.ForwardFull(decodeIn)

	tr.Reset()
	tr.ForwardMode = TransformerForwardNormal
	_ = tr.ForwardFull(prefill)
	got2 := tr.ForwardFull(decodeIn)
	if maxAbsDiffSlice(got2.Data, want.Data) > 1e-3 {
		t.Fatalf("fused decode mismatch after pipeline run")
	}
}
