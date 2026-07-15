package poly_test

import (
	"math"
	"testing"

	. "github.com/openfluke/loom/poly"
	"github.com/openfluke/loom/poly/simd"
)

func TestApplyLMHeadQ4MatchesFP32Roughly(t *testing.T) {
	const V, H = 128, 64
	net := &VolumetricNetwork{UsePackedQ4CPU: true, UseSimdForward: true}
	emb := make([]float32, V*H)
	head := make([]float32, V*H)
	for i := range head {
		head[i] = float32((i%19)-9) * 0.04
		emb[i] = head[i] * 0.5
	}
	tr := &Transformer[float32]{
		Network:    net,
		Embeddings: emb,
		LMHead:     head,
		HiddenSize: H,
		VocabSize:  V,
	}
	hidden := make([]float32, H)
	for i := range hidden {
		hidden[i] = float32((i%7)-3) * 0.1
	}

	// FP32 reference before packing frees LMHead.
	fp32 := make([]float32, V)
	for v := 0; v < V; v++ {
		var sum float64
		off := v * H
		for d := 0; d < H; d++ {
			sum += float64(hidden[d]) * float64(head[off+d])
		}
		fp32[v] = float32(sum)
	}

	tr.EnsurePackedQ4LMHead()
	if len(tr.LMHead) != 0 {
		t.Fatal("expected untied FP32 LMHead released after pack")
	}
	got := tr.ApplyLMHead(hidden)
	if len(got) != V {
		t.Fatalf("len got %d", len(got))
	}

	var maxDiff float64
	for i := 0; i < V; i++ {
		d := math.Abs(float64(fp32[i] - got[i]))
		if d > maxDiff {
			maxDiff = d
		}
	}
	// Q4 approximates FP32; allow generous absolute error on this toy scale.
	if maxDiff > 0.5 {
		t.Fatalf("Q4 vs FP32 maxDiff=%g (too large)", maxDiff)
	}
	if simd.SimdEnabled() {
		t.Logf("maxDiff=%g (simd on)", maxDiff)
	}
}

func TestPackQ4_0GPUParallelMatchesSerial(t *testing.T) {
	n := 4096
	data := make([]float32, n)
	for i := range data {
		data[i] = float32((i%23)-11) * 0.03
	}
	s1, p1 := PackQ4_0GPU(data)
	s2, p2 := PackQ4_0GPUParallel(data)
	if len(s1) != len(s2) {
		t.Fatalf("scales len %d vs %d", len(s1), len(s2))
	}
	for i := range s1 {
		if s1[i] != s2[i] {
			t.Fatalf("scale[%d] %v vs %v", i, s1[i], s2[i])
		}
	}
	for i := 0; i < len(s1)*4; i++ {
		if p1[i] != p2[i] {
			t.Fatalf("packed[%d]", i)
		}
	}
}
