package poly_test

import (
	"path/filepath"
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestSaveEntityTransformerBakesLMHeadQ4(t *testing.T) {
	const V, H = 64, 32
	net := &poly.VolumetricNetwork{
		Layers: []poly.VolumetricLayer{{
			Type:         poly.LayerRMSNorm,
			InputHeight:  H,
			OutputHeight: H,
			WeightStore:  poly.NewWeightStore(H),
		}},
	}
	emb := make([]float32, V*H)
	for i := range emb {
		emb[i] = float32((i%7)-3) * 0.1
	}
	et := &poly.EntityTransformer{
		Network:      net,
		HiddenSize:   H,
		VocabSize:    V,
		LMHeadTied:   true,
		HasFinalNorm: false,
		WeightDType:  poly.DTypeInt4,
		Embeddings:   emb,
		LMHead:       emb,
	}
	path := filepath.Join(t.TempDir(), "tiny.entity")
	if err := poly.SaveEntityTransformer(path, et); err != nil {
		t.Fatalf("save: %v", err)
	}
	if !poly.EntityHasLMHeadQ4(path) {
		t.Fatal("expected transformer.lm_head.q4_0 after Int4 SaveEntityTransformer")
	}
}
