package poly

import (
	"fmt"
)

func infiniteLayerSeedForKind(tag, kind string, dt DType) uint64 {
	return SeedFrom("infinite-layer-dtype", tag, kind, dt.String())
}

// RunInfiniteLayerDTypeRoundTrip builds a procedural layer, edits weight[0],
// round-trips through InfiniteLayerManifest, and verifies weight fingerprint.
func RunInfiniteLayerDTypeRoundTrip(kind string, dt DType) DTypeRoundTripResult {
	return runInfiniteLayerDTypeRoundTrip("roundtrip", kind, dt)
}

func runInfiniteLayerDTypeRoundTrip(tag, kind string, dt DType) DTypeRoundTripResult {
	res := DTypeRoundTripResult{Family: kind, DType: dt, DTypeName: dt.String()}
	kind = normalizeInfiniteKind(kind)
	layerSeed := infiniteLayerSeedForKind(tag, kind, dt)
	res.LayerSeed = layerSeed

	l, err := BuildLayerFromSeed(kind, layerSeed, dt)
	if err != nil {
		res.Err = "build: " + err.Error()
		return res
	}
	if l.WeightStore == nil || len(l.WeightStore.Master) == 0 {
		res.Err = "empty weight store"
		return res
	}
	l.WeightStore.Master[0] += 0.001
	wantFP := weightStoreFingerprint(l.WeightStore)

	m, err := ManifestFromLayer(l, layerSeed)
	if err != nil {
		res.Err = "manifest: " + err.Error()
		return res
	}
	res.WeightFP = m.WeightFP
	if m.WeightFP != wantFP {
		res.Err = fmt.Sprintf("manifest weight fp 0x%x want 0x%x", m.WeightFP, wantFP)
		return res
	}
	if m.OverrideCount() == 0 {
		res.Err = "expected sparse overrides after edit"
		return res
	}

	rebuilt, err := BuildLayerFromManifest(m)
	if err != nil {
		res.Err = "rebuild: " + err.Error()
		return res
	}
	gotFP := weightStoreFingerprint(rebuilt.WeightStore)
	if gotFP != wantFP {
		res.Err = fmt.Sprintf("rebuilt weight fp 0x%x want 0x%x", gotFP, wantFP)
		return res
	}

	raw, err := MarshalInfiniteLayer(m)
	if err != nil {
		res.Err = "marshal: " + err.Error()
		return res
	}
	parsed, err := ParseInfiniteLayer(raw)
	if err != nil {
		res.Err = "parse: " + err.Error()
		return res
	}
	fromJSON, err := BuildLayerFromManifest(parsed)
	if err != nil {
		res.Err = "json rebuild: " + err.Error()
		return res
	}
	if weightStoreFingerprint(fromJSON.WeightStore) != wantFP {
		res.Err = "json round-trip weight fp mismatch"
		return res
	}

	res.OK = true
	return res
}

// RunAllInfiniteLayerDTypeMatrix runs 21 dtypes across all infinite layer kinds.
func RunAllInfiniteLayerDTypeMatrix(tag string) []LayerDTypeMatrix {
	kinds := []string{"dense", "swiglu", "mha", "cnn1", "cnn2", "cnn3", "rnn", "lstm", "embedding"}
	out := make([]LayerDTypeMatrix, 0, len(kinds))
	for _, kind := range kinds {
		results := make([]DTypeRoundTripResult, 0, 21)
		for _, dt := range SeedDTypesAll() {
			results = append(results, runInfiniteLayerDTypeRoundTrip(tag, kind, dt))
		}
		out = append(out, LayerDTypeMatrix{Family: kind, Results: results})
	}
	return out
}
