package poly_test

import (
	"testing"

	. "github.com/openfluke/loom/poly"
)

func persistenceDenseNetwork(dtype DType) *VolumetricNetwork {
	net := NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = LayerDense
	l.Activation = ActivationLinear
	l.DType = dtype
	l.InputHeight = 4
	l.OutputHeight = 3
	l.WeightStore = NewWeightStore(l.InputHeight * l.OutputHeight)
	copy(l.WeightStore.Master, deterministicWeights(len(l.WeightStore.Master)))
	return net
}

func TestNativePersistenceRoundTripRestoresCompactDTypes(t *testing.T) {
	cases := []struct {
		name      string
		dtype     DType
		tolerance float64
	}{
		{"float16", DTypeFloat16, 1e-2},
		{"bfloat16", DTypeBFloat16, 1e-2},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			net := persistenceDenseNetwork(tc.dtype)
			want := append([]float32(nil), net.GetLayer(0, 0, 0, 0).WeightStore.Master...)

			data, err := SerializeNetwork(net)
			if err != nil {
				t.Fatal(err)
			}
			roundTrip, err := DeserializeNetwork(data)
			if err != nil {
				t.Fatal(err)
			}
			gotLayer := roundTrip.GetLayer(0, 0, 0, 0)
			if gotLayer.DType != tc.dtype {
				t.Fatalf("dtype = %v, want %v", gotLayer.DType, tc.dtype)
			}
			if _, ok := gotLayer.WeightStore.Versions[tc.dtype].([]uint16); !ok {
				t.Fatalf("version type = %T, want []uint16", gotLayer.WeightStore.Versions[tc.dtype])
			}
			if diff := maxAbsDiffSlice(gotLayer.WeightStore.Master, want); diff > tc.tolerance {
				t.Fatalf("round-trip diff = %.6f, want <= %.6f", diff, tc.tolerance)
			}
		})
	}
}

func TestMasterPersistenceRoundTripKeepsTrainedWeights(t *testing.T) {
	for _, dtype := range []DType{
		DTypeFP8E4M3, DTypeFP8E5M2,
		DTypeUint64, DTypeUint32, DTypeUint16, DTypeUint8,
	} {
		t.Run(dtype.String(), func(t *testing.T) {
			net := persistenceDenseNetwork(dtype)
			want := append([]float32(nil), net.GetLayer(0, 0, 0, 0).WeightStore.Master...)

			data, err := SerializeNetwork(net)
			if err != nil {
				t.Fatal(err)
			}
			roundTrip, err := DeserializeNetwork(data)
			if err != nil {
				t.Fatal(err)
			}
			got := roundTrip.GetLayer(0, 0, 0, 0).WeightStore.Master
			if diff := maxAbsDiffSlice(got, want); diff != 0 {
				t.Fatalf("round-trip diff = %.6f, want exact master preservation", diff)
			}
		})
	}
}
