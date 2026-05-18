package poly_test

import (
	"testing"

	. "github.com/openfluke/loom/poly"
)

func TestLayerNativePersistenceSnapshotRoundTrip(t *testing.T) {
	for _, dtype := range []DType{DTypeInt4, DTypeInt2, DTypeFP8E4M3} {
		t.Run(dtype.String(), func(t *testing.T) {
			net := persistenceDenseNetwork(dtype)
			ws := net.GetLayer(0, 0, 0, 0).WeightStore
			b1, s1, ok := LayerNativePersistenceSnapshot(ws, dtype)
			if !ok || b1 == "" {
				t.Fatal("snapshot failed")
			}
			data, err := SerializeNetwork(net)
			if err != nil {
				t.Fatal(err)
			}
			net2, err := DeserializeNetwork(data)
			if err != nil {
				t.Fatal(err)
			}
			ws2 := net2.GetLayer(0, 0, 0, 0).WeightStore
			b2, s2, ok2 := LayerNativePersistenceSnapshot(ws2, dtype)
			if !ok2 || b2 != b1 || s2 != s1 {
				t.Fatalf("blob mismatch ok=%v scale %v vs %v len %d vs %d", ok2, s1, s2, len(b1), len(b2))
			}
		})
	}
}
