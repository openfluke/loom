package poly

import "testing"

func TestInfiniteDenseLayerRoundTrip(t *testing.T) {
	topo := DenseTopologySeed("inf-layer", []int{4, 8})
	seed := DenseLayerWeightSeed(topo, 0)
	layer, err := BuildDenseLayerFromSeed(seed, 4, 8, DTypeFloat32)
	if err != nil {
		t.Fatal(err)
	}
	m, err := EncodeInfiniteDenseLayer(layer.WeightStore, 4, 8, DTypeFloat32, seed)
	if err != nil {
		t.Fatal(err)
	}
	if m.OverrideCount() != 0 {
		t.Fatalf("procedural layer overrides=%d", m.OverrideCount())
	}
	back, err := BuildDenseLayerFromInfiniteManifest(m)
	if err != nil {
		t.Fatal(err)
	}
	if weightStoreFingerprint(layer.WeightStore) != weightStoreFingerprint(back.WeightStore) {
		t.Fatal("fp mismatch")
	}
	extracted, err := ManifestFromDenseLayer(layer, seed)
	if err != nil {
		t.Fatal(err)
	}
	if extracted.LayerSeed != seed {
		t.Fatalf("seed 0x%x vs 0x%x", extracted.LayerSeed, seed)
	}
}

func TestInfiniteDenseLayerEditedRoundTrip(t *testing.T) {
	topo := DenseTopologySeed("inf-edit", []int{6, 6})
	seed := DenseLayerWeightSeed(topo, 0)
	layer, err := BuildDenseLayerFromSeed(seed, 6, 6, DTypeInt8)
	if err != nil {
		t.Fatal(err)
	}
	layer.WeightStore.Master[0] = 99.0
	m, err := EncodeInfiniteDenseLayer(layer.WeightStore, 6, 6, DTypeInt8, seed)
	if err != nil {
		t.Fatal(err)
	}
	if m.OverrideCount() == 0 {
		t.Fatal("expected overrides for edited weight")
	}
	back, err := BuildDenseLayerFromInfiniteManifest(m)
	if err != nil {
		t.Fatal(err)
	}
	if weightStoreFingerprint(layer.WeightStore) != weightStoreFingerprint(back.WeightStore) {
		t.Fatal("edited round-trip fp mismatch")
	}
}

func TestInfiniteDenseLayerAllDTypes(t *testing.T) {
	sizes := []int{4, 8}
	for _, dt := range SeedDTypesAll() {
		topo := DenseDTypeTopologySeed("inf-all", sizes, dt)
		seed := DenseLayerWeightSeed(topo, 0)
		layer, err := BuildDenseLayerFromSeed(seed, 4, 8, dt)
		if err != nil {
			t.Fatalf("%s build: %v", dt, err)
		}
		m, err := ManifestFromDenseLayer(layer, seed)
		if err != nil {
			t.Fatalf("%s manifest: %v", dt, err)
		}
		back, err := BuildDenseLayerFromInfiniteManifest(m)
		if err != nil {
			t.Fatalf("%s decode: %v", dt, err)
		}
		if weightStoreFingerprint(layer.WeightStore) != weightStoreFingerprint(back.WeightStore) {
			t.Fatalf("%s fp mismatch", dt)
		}
	}
}
