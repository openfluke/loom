package poly_test

import (
	"testing"

	. "github.com/openfluke/loom/poly"
)

func TestDenseManifestRoundTrip(t *testing.T) {
	topo := DenseTopologySeed("test", []int{4, 8, 2})
	m, err := BuildDenseManifest(topo, []int{4, 8, 2}, []string{"float32", "int8"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := RebuildDenseManifest(m); err != nil {
		t.Fatal(err)
	}
}

func TestBuildSeededEntityTransformer(t *testing.T) {
	seed := SeedFrom("test", "lm")
	dims := HFDecoderDims{
		NumLayers: 1, HiddenSize: 8, NumHeads: 2, NumKVHeads: 2, HeadDim: 4,
		QueryDim: 8, KVDim: 8, IntermediateSize: 16, Activation: ActivationSilu,
	}
	et := BuildSeededEntityTransformer(seed, dims, 16, DTypeFloat32, true, true)
	if et == nil {
		t.Fatal("nil et")
	}
	fp := EntityTransformerFingerprint(et)
	again := BuildSeededEntityTransformer(seed, dims, 16, DTypeFloat32, true, true)
	if EntityTransformerFingerprint(again) != fp {
		t.Fatal("fingerprint mismatch")
	}
}

func TestSeedFromStable(t *testing.T) {
	a := SeedFrom("loom", "dense", 4, 8)
	b := SeedFrom("loom", "dense", 4, 8)
	if a != b {
		t.Fatalf("unstable: %x %x", a, b)
	}
}
