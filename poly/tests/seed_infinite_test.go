package poly_test

import (
	"fmt"
	"testing"

	. "github.com/openfluke/loom/poly"
)

func TestInfiniteLayerDenseRoundTrip(t *testing.T) {
	topo := DenseTopologySeed("inf", []int{4, 8})
	seed := DenseLayerWeightSeed(topo, 0)
	layer, err := BuildLayerFromSeed("dense", seed, DTypeFloat32, &InfiniteLayerManifest{In: 4, Out: 8})
	if err != nil {
		t.Fatal(err)
	}
	m, err := ManifestFromLayer(layer, seed)
	if err != nil {
		t.Fatal(err)
	}
	if m.Kind != "dense" || m.OverrideCount() != 0 {
		t.Fatalf("kind=%q overrides=%d", m.Kind, m.OverrideCount())
	}
	back, err := BuildLayerFromManifest(m)
	if err != nil {
		t.Fatal(err)
	}
	if WeightStoreFingerprint(layer.WeightStore) != WeightStoreFingerprint(back.WeightStore) {
		t.Fatal("fp mismatch")
	}
}

func TestInfiniteLayerEditedAllKinds(t *testing.T) {
	cases := []struct {
		kind string
		seed uint64
		build func(uint64) (*VolumetricLayer, error)
	}{
		{"dense", DenseLayerWeightSeed(DenseTopologySeed("t", []int{6, 6}), 0), func(s uint64) (*VolumetricLayer, error) {
			return BuildLayerFromSeed("dense", s, DTypeInt8, &InfiniteLayerManifest{In: 6, Out: 6})
		}},
		{"swiglu", SwiGLULayerWeightSeed(SwiGLUTopologySeed("t", []SwiGLUSpec{{8, 16}}), 0), func(s uint64) (*VolumetricLayer, error) {
			return BuildLayerFromSeed("swiglu", s, DTypeFloat32, &InfiniteLayerManifest{Hidden: 8, Intermediate: 16})
		}},
		{"mha", MHALayerWeightSeed(MHATopologySeed("t", []MHASpec{{DModel: 8, NumHeads: 2, NumKVHeads: 2, HeadDim: 4, QueryDim: 8}}), 0), func(s uint64) (*VolumetricLayer, error) {
			spec := MHASpec{DModel: 8, NumHeads: 2, NumKVHeads: 2, HeadDim: 4, QueryDim: 8}
			return BuildLayerFromSeed("mha", s, DTypeFloat32, &InfiniteLayerManifest{MHA: &spec})
		}},
		{"rnn", RNNLayerWeightSeed(RNNTopologySeed("t", []int{4, 6}), 0), func(s uint64) (*VolumetricLayer, error) {
			return BuildLayerFromSeed("rnn", s, DTypeFloat32, &InfiniteLayerManifest{In: 4, Out: 6})
		}},
		{"lstm", LSTMLayerWeightSeed(LSTMTopologySeed("t", []int{4, 6}), 0), func(s uint64) (*VolumetricLayer, error) {
			return BuildLayerFromSeed("lstm", s, DTypeFloat32, &InfiniteLayerManifest{In: 4, Out: 6})
		}},
	}
	for _, tc := range cases {
		t.Run(tc.kind, func(t *testing.T) {
			layer, err := tc.build(tc.seed)
			if err != nil {
				t.Fatal(err)
			}
			layer.WeightStore.Master[0] = 42
			m, err := ManifestFromLayer(layer, tc.seed)
			if err != nil {
				t.Fatal(err)
			}
			if len(m.Overrides) == 0 {
				t.Fatal("expected weight overrides")
			}
			back, err := BuildLayerFromManifest(m)
			if err != nil {
				t.Fatal(err)
			}
			if WeightStoreFingerprint(layer.WeightStore) != WeightStoreFingerprint(back.WeightStore) {
				t.Fatal("fp mismatch")
			}
		})
	}
	for _, spec := range []CNNSpec{
		{Dim: 1, InputChannels: 2, Filters: 4, Spatial: 8, KernelSize: 3},
		{Dim: 2, InputChannels: 2, Filters: 4, Spatial: 8, KernelSize: 3},
		{Dim: 3, InputChannels: 2, Filters: 4, Spatial: 4, KernelSize: 3},
	} {
		name := fmt.Sprintf("cnn%d", spec.Dim)
		t.Run(name, func(t *testing.T) {
			seed := CNNLayerWeightSeed(CNNTopologySeed("t", []CNNSpec{spec}), spec, 0)
			layer, err := BuildLayerFromSeed(fmt.Sprintf("cnn%d", spec.Dim), seed, DTypeFloat32, &InfiniteLayerManifest{CNN: &spec})
			if err != nil {
				t.Fatal(err)
			}
			layer.WeightStore.Master[1] = 7
			m, err := ManifestFromLayer(layer, seed)
			if err != nil {
				t.Fatal(err)
			}
			back, err := BuildLayerFromManifest(m)
			if err != nil {
				t.Fatal(err)
			}
			if WeightStoreFingerprint(layer.WeightStore) != WeightStoreFingerprint(back.WeightStore) {
				t.Fatal("fp mismatch")
			}
		})
	}
}

func TestInfiniteLayerAllDTypesDense(t *testing.T) {
	for _, dt := range SeedDTypesAll() {
		t.Run(dt.String(), func(t *testing.T) {
			topo := DenseDTypeTopologySeed("inf-all", []int{4, 8}, dt)
			seed := DenseLayerWeightSeed(topo, 0)
			layer, err := BuildLayerFromSeed("dense", seed, dt, &InfiniteLayerManifest{In: 4, Out: 8})
			if err != nil {
				t.Fatal(err)
			}
			m, err := ManifestFromLayer(layer, seed)
			if err != nil {
				t.Fatal(err)
			}
			back, err := BuildLayerFromManifest(m)
			if err != nil {
				t.Fatal(err)
			}
			if WeightStoreFingerprint(layer.WeightStore) != WeightStoreFingerprint(back.WeightStore) {
				t.Fatal("fp mismatch")
			}
		})
	}
}

func TestInfiniteLayerDTypeMatrix(t *testing.T) {
	matrix := RunAllInfiniteLayerDTypeMatrix("test")
	for _, block := range matrix {
		pass, fail, report := DTypeRoundTripSummary(block.Results)
		if fail > 0 {
			t.Errorf("%s: %d/%d failed: %s", block.Family, fail, pass+fail, report)
		}
	}
}
