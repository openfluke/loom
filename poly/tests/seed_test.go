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

func TestSwiGLUManifestRoundTrip(t *testing.T) {
	specs := []SwiGLUSpec{
		{Hidden: 8, Intermediate: 16},
		{Hidden: 8, Intermediate: 12},
	}
	topo := SwiGLUTopologySeed("test", specs)
	m, err := BuildSwiGLUManifest(topo, specs, []string{"float32", "int8"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := RebuildSwiGLUManifest(m); err != nil {
		t.Fatal(err)
	}
	net, err := BuildSwiGLUVolumetricFromManifest(m)
	if err != nil {
		t.Fatal(err)
	}
	extracted, err := ManifestFromSwiGLUNetwork(net, topo, specs, []string{"float32", "int8"})
	if err != nil {
		t.Fatal(err)
	}
	if extracted.NetworkFP != m.NetworkFP {
		t.Fatalf("network fp mismatch: %x %x", extracted.NetworkFP, m.NetworkFP)
	}
}

func TestMHAManifestRoundTrip(t *testing.T) {
	specs := []MHASpec{
		{DModel: 8, NumHeads: 2, NumKVHeads: 2, HeadDim: 4, QueryDim: 8},
		{DModel: 8, NumHeads: 4, NumKVHeads: 2, HeadDim: 2, QueryDim: 8},
	}
	topo := MHATopologySeed("test", specs)
	m, err := BuildMHAManifest(topo, specs, []string{"float32", "int8"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := RebuildMHAManifest(m); err != nil {
		t.Fatal(err)
	}
	net, err := BuildMHAVolumetricFromManifest(m)
	if err != nil {
		t.Fatal(err)
	}
	extracted, err := ManifestFromMHANetwork(net, topo, specs, []string{"float32", "int8"})
	if err != nil {
		t.Fatal(err)
	}
	if extracted.NetworkFP != m.NetworkFP {
		t.Fatalf("network fp mismatch: %x %x", extracted.NetworkFP, m.NetworkFP)
	}
}

func TestRNNManifestRoundTrip(t *testing.T) {
	sizes := []int{4, 8, 3}
	topo := RNNTopologySeed("test", sizes)
	m, err := BuildRNNManifest(topo, sizes, []string{"float32", "int8"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := RebuildRNNManifest(m); err != nil {
		t.Fatal(err)
	}
	net, err := BuildRNNVolumetricFromManifest(m)
	if err != nil {
		t.Fatal(err)
	}
	extracted, err := ManifestFromRNNNetwork(net, topo, sizes, []string{"float32", "int8"})
	if err != nil {
		t.Fatal(err)
	}
	if extracted.NetworkFP != m.NetworkFP {
		t.Fatalf("network fp mismatch: %x %x", extracted.NetworkFP, m.NetworkFP)
	}
}

func TestLSTMManifestRoundTrip(t *testing.T) {
	sizes := []int{4, 8, 3}
	topo := LSTMTopologySeed("test", sizes)
	m, err := BuildLSTMManifest(topo, sizes, []string{"float32", "int8"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := RebuildLSTMManifest(m); err != nil {
		t.Fatal(err)
	}
	net, err := BuildLSTMVolumetricFromManifest(m)
	if err != nil {
		t.Fatal(err)
	}
	extracted, err := ManifestFromLSTMNetwork(net, topo, sizes, []string{"float32", "int8"})
	if err != nil {
		t.Fatal(err)
	}
	if extracted.NetworkFP != m.NetworkFP {
		t.Fatalf("network fp mismatch: %x %x", extracted.NetworkFP, m.NetworkFP)
	}
}

func TestCNNManifestRoundTrip(t *testing.T) {
	cases := []struct {
		name  string
		specs []CNNSpec
	}{
		{
			"cnn1",
			[]CNNSpec{
				{Dim: 1, InputChannels: 2, Filters: 4, Spatial: 8, KernelSize: 3},
				{Dim: 1, InputChannels: 4, Filters: 2, Spatial: 8, KernelSize: 3},
			},
		},
		{
			"cnn2",
			[]CNNSpec{
				{Dim: 2, InputChannels: 2, Filters: 4, Spatial: 6, KernelSize: 3},
				{Dim: 2, InputChannels: 4, Filters: 2, Spatial: 6, KernelSize: 3},
			},
		},
		{
			"cnn3",
			[]CNNSpec{
				{Dim: 3, InputChannels: 2, Filters: 4, Spatial: 4, KernelSize: 3},
				{Dim: 3, InputChannels: 4, Filters: 2, Spatial: 4, KernelSize: 3},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			topo := CNNTopologySeed("test", tc.specs)
			m, err := BuildCNNManifest(topo, tc.specs, []string{"float32", "int8"})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := RebuildCNNManifest(m); err != nil {
				t.Fatal(err)
			}
			net, err := BuildCNNVolumetricFromManifest(m)
			if err != nil {
				t.Fatal(err)
			}
			extracted, err := ManifestFromCNNNetwork(net, topo, tc.specs, []string{"float32", "int8"})
			if err != nil {
				t.Fatal(err)
			}
			if extracted.NetworkFP != m.NetworkFP {
				t.Fatalf("network fp mismatch: %x %x", extracted.NetworkFP, m.NetworkFP)
			}
		})
	}
}

func TestEmbeddingManifestRoundTrip(t *testing.T) {
	specs := []EmbeddingSpec{
		{VocabSize: 32, EmbeddingDim: 8, SeqLen: 8},
		{VocabSize: 32, EmbeddingDim: 8, SeqLen: 8},
	}
	topo := EmbeddingTopologySeed("test", specs)
	m, err := BuildEmbeddingManifest(topo, specs, []string{"float32", "int8"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := RebuildEmbeddingManifest(m); err != nil {
		t.Fatal(err)
	}
	net, err := BuildEmbeddingVolumetricFromManifest(m)
	if err != nil {
		t.Fatal(err)
	}
	extracted, err := ManifestFromEmbeddingNetwork(net, topo, specs, []string{"float32", "int8"})
	if err != nil {
		t.Fatal(err)
	}
	if extracted.NetworkFP != m.NetworkFP {
		t.Fatalf("network fp mismatch: %x %x", extracted.NetworkFP, m.NetworkFP)
	}
}

func TestResidualManifestRoundTrip(t *testing.T) {
	spec := ResidualSpec{In: 8, Out: 8}
	topo := ResidualTopologySeed("test", spec)
	m, err := BuildResidualManifest(topo, spec, "float32")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := RebuildResidualManifest(m); err != nil {
		t.Fatal(err)
	}
	net, err := BuildResidualVolumetricFromManifest(m)
	if err != nil {
		t.Fatal(err)
	}
	extracted, err := ManifestFromResidualNetwork(net, topo, spec, "float32")
	if err != nil {
		t.Fatal(err)
	}
	if extracted.DenseSeed != m.DenseSeed {
		t.Fatalf("dense seed mismatch: %x %x", extracted.DenseSeed, m.DenseSeed)
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
