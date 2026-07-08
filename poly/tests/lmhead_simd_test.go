package poly_test

import (
	"math"
	"math/rand"
	"testing"

	. "github.com/openfluke/loom/poly"
)

// TestLMHeadFP32SimdMatchesScalar checks the FP32 LM head (used when the head is
// tied to FP32 embeddings, e.g. BitNet weight tying) produces the same logits with
// the NEON/AVX2 DotTile kernel on as with the scalar float64 loop. Both accumulate
// in float64; the SIMD reduction order differs, so we assert closeness rather than
// bit-identity — the same contract the Dense/SwiGLU/MHA SIMD paths already follow.
func TestLMHeadFP32SimdMatchesScalar(t *testing.T) {
	const H, V = 320, 700
	net := NewVolumetricNetwork(1, 1, 1, 1)
	tr := &Transformer[float32]{
		Network:    net,
		HiddenSize: H,
		VocabSize:  V,
		LMHead:     make([]float32, V*H),
	}
	rng := rand.New(rand.NewSource(3))
	for i := range tr.LMHead {
		tr.LMHead[i] = float32(rng.NormFloat64()) * 0.05
	}
	hidden := make([]float32, H)
	for i := range hidden {
		hidden[i] = float32(rng.NormFloat64()) * 0.1
	}

	net.UseSimdForward = false
	scalar := tr.ApplyLMHead(hidden)

	net.SetSimdForwardRecursive(true)
	defer net.SetSimdForwardRecursive(false)
	if !Plan9SimdEnabled() {
		t.Skip("SIMD not linked for this GOARCH")
	}
	simd := tr.ApplyLMHead(hidden)

	if len(scalar) != V || len(simd) != V {
		t.Fatalf("logit lengths: scalar=%d simd=%d want %d", len(scalar), len(simd), V)
	}
	var maxDiff float64
	for i := range scalar {
		d := math.Abs(float64(scalar[i] - simd[i]))
		if d > maxDiff {
			maxDiff = d
		}
	}
	// float64 accumulation both ways → differences are last-bit rounding only.
	if maxDiff > 1e-3 {
		t.Fatalf("LM head SIMD vs scalar max diff = %g, want < 1e-3", maxDiff)
	}
}
