package poly_test

import (
	"testing"

	. "github.com/openfluke/loom/poly"
)

// Simulates lucy training matrix save/reload: Morph → serialize native → load → compare Master.
func TestTrainingStyleNativeSaveReload(t *testing.T) {
	cases := []struct {
		name      string
		dtype     DType
		tolerance float64
	}{
		{"float32", DTypeFloat32, 1e-6},
		{"float16", DTypeFloat16, 1e-2},
		{"bfloat16", DTypeBFloat16, 1e-2},
		{"fp8e4m3", DTypeFP8E4M3, 0},
		{"fp8e5m2", DTypeFP8E5M2, 0},
		{"int64", DTypeInt64, 0},
		{"uint8", DTypeUint8, 0},
		{"int8", DTypeInt8, 0},
		{"int4", DTypeInt4, 0},
		{"fp4", DTypeFP4, 0},
		{"ternary", DTypeTernary, 1e-2},
		{"binary", DTypeBinary, 0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			net := persistenceDenseNetwork(tc.dtype)
			layer := net.GetLayer(0, 0, 0, 0)
			layer.WeightStore.Morph(tc.dtype)
			layer.WeightStore.Unpack(tc.dtype)
			before := append([]float32(nil), layer.WeightStore.Master...)

			data, err := SerializeNetwork(net)
			if err != nil {
				t.Fatal(err)
			}
			net2, err := DeserializeNetwork(data)
			if err != nil {
				t.Fatal(err)
			}
			l2 := net2.GetLayer(0, 0, 0, 0)
			after := append([]float32(nil), l2.WeightStore.Master...)

			tol := tc.tolerance
			if tol == 0 {
				if l2.WeightStore.Scale != 0 {
					tol = float64(l2.WeightStore.Scale) * 0.51
				} else {
					tol = 1e-4
				}
			}
			if diff := maxAbsDiffSlice(after, before); diff > tol {
				t.Fatalf("master diff %.6e > tol %.6e (scale=%v)", diff, tol, l2.WeightStore.Scale)
			}
		})
	}
}
