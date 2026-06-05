package poly_test

import (
	"bytes"
	"testing"

	. "github.com/openfluke/loom/poly"
)

func TestEntityRoundTripMatchesJSONPersistence(t *testing.T) {
	for _, dtype := range []DType{
		DTypeFloat32, DTypeFloat16, DTypeBFloat16,
		DTypeInt8, DTypeInt4, DTypeBinary, DTypeTernary,
	} {
		t.Run(dtype.String(), func(t *testing.T) {
			net := persistenceDenseNetwork(dtype)
			layer := net.GetLayer(0, 0, 0, 0)
			layer.WeightStore.Morph(dtype)

			entityWire, err := SerializeEntity(net)
			if err != nil {
				t.Fatal(err)
			}
			jsonWire, err := SerializeNetwork(net)
			if err != nil {
				t.Fatal(err)
			}
			if len(entityWire) >= len(jsonWire) {
				t.Fatalf("entity size %d should be smaller than JSON %d", len(entityWire), len(jsonWire))
			}

			reloaded, err := DeserializeEntity(entityWire)
			if err != nil {
				t.Fatal(err)
			}
			got := reloaded.GetLayer(0, 0, 0, 0)
			if got.DType != dtype {
				t.Fatalf("dtype = %v, want %v", got.DType, dtype)
			}
			wantActive := layer.WeightStore.Versions[dtype]
			if wantActive == nil {
				wantActive = layer.WeightStore.GetNative(dtype)
			}
			gotActive := got.WeightStore.Versions[dtype]
			if gotActive == nil {
				gotActive = got.WeightStore.GetNative(dtype)
			}
			if !NativeWeightsEncoded(wantActive, gotActive, dtype) {
				t.Fatal("native weight blob mismatch after entity round-trip")
			}
		})
	}
}

func TestEntityIdempotentBytes(t *testing.T) {
	net := persistenceDenseNetwork(DTypeInt4)
	net.GetLayer(0, 0, 0, 0).WeightStore.Morph(DTypeInt4)

	first, err := SerializeEntity(net)
	if err != nil {
		t.Fatal(err)
	}
	reloaded, err := DeserializeEntity(first)
	if err != nil {
		t.Fatal(err)
	}
	second, err := SerializeEntity(reloaded)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("entity save→load→save changed bytes (%d vs %d)", len(first), len(second))
	}
}

func TestEntityLayerSelectiveLoad(t *testing.T) {
	net := NewVolumetricNetwork(1, 1, 2, 1)
	for i := range net.Layers {
		l := &net.Layers[i]
		l.Type = LayerDense
		l.Activation = ActivationLinear
		l.DType = DTypeFloat32
		l.InputHeight = 2
		l.OutputHeight = 2
		l.WeightStore = NewWeightStore(4)
		copy(l.WeightStore.Master, deterministicWeights(4))
	}

	wire, err := SerializeEntity(net)
	if err != nil {
		t.Fatal(err)
	}
	partial, err := DeserializeEntityLayer(wire, 1)
	if err != nil {
		t.Fatal(err)
	}
	l0 := partial.GetLayer(0, 0, 0, 0)
	l1 := partial.GetLayer(0, 0, 1, 0)
	if l0.WeightStore == nil || len(l0.WeightStore.Master) == 0 {
		t.Fatal("layer 0 should have initialized weights from topology init")
	}
	if l1.WeightStore == nil || l1.WeightStore.Versions[DTypeFloat32] == nil && len(l1.WeightStore.Master) == 0 {
		t.Fatal("layer 1 should have loaded entity blob")
	}
}
