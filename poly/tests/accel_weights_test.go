package poly_test

import (
	"encoding/json"
	"testing"

	"github.com/openfluke/loom/poly"
	"github.com/openfluke/loom/poly/accel"
)

func TestLayerWeightBytesForAccelDTypes(t *testing.T) {
	for _, tc := range []struct {
		dtype string
		label string
	}{
		{"FLOAT32", "FP32"},
		{"FLOAT16", "FP16"},
		{"INT8", "INT8"},
	} {
		t.Run(tc.label, func(t *testing.T) {
			spec := []byte(`{
				"id":"t","depth":1,"rows":1,"cols":1,"layers_per_cell":1,
				"layers":[{"z":0,"y":0,"x":0,"l":0,"type":"DENSE","activation":"LINEAR",
					"dtype":"` + tc.dtype + `","input_height":32,"output_height":32}]
			}`)
			net, err := poly.BuildNetworkFromJSON(spec)
			if err != nil {
				t.Fatal(err)
			}
			if err := poly.ConfigureNetworkForMode(net, poly.TrainingModeCPUMC); err != nil {
				t.Fatal(err)
			}
			l := &net.Layers[0]
			l.WeightStore.Randomize(42, 0.1)
			got := poly.LayerWeightBytesForAccel(l)
			if len(got) == 0 {
				t.Fatal("empty weight bytes")
			}
			wantElem := 4
			if tc.label == "FP16" {
				wantElem = 2
			}
			if len(got) != 32*32*wantElem {
				t.Fatalf("len=%d want=%d", len(got), 32*32*wantElem)
			}
			if tc.label != "FP32" {
				master := append([]float32(nil), l.WeightStore.Master...)
				if len(master) == 0 {
					t.Fatal("missing master")
				}
				same := true
				active := poly.CastWeights[float32](l.WeightStore.GetActive(l.DType))
				for i := range active {
					if active[i] != master[i] {
						same = false
						break
					}
				}
				if same {
					t.Fatalf("%s GetActive should differ from Master after Morph", tc.label)
				}
			}
		})
	}
}

func TestLayerWeightBytesMatchesPlugin(t *testing.T) {
	path := accel.DefaultIntelPath()
	reg, err := poly.DiscoverAccel(accel.AccelConfig{IntelSO: path})
	if err != nil {
		t.Skip(err)
	}
	defer reg.Close()
	plug := reg.PluginFor(accel.ExecIntelCPU)
	if plug == nil {
		t.Skip("no intel cpu plugin")
	}

	spec, _ := json.Marshal(map[string]any{
		"id": "t", "depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": []any{map[string]any{
			"z": 0, "y": 0, "x": 0, "l": 0,
			"type": "DENSE", "activation": "LINEAR", "dtype": "FLOAT16",
			"input_height": 32, "output_height": 32,
		}},
	})
	net, err := poly.BuildNetworkFromJSON(spec)
	if err != nil {
		t.Fatal(err)
	}
	_ = poly.ConfigureNetworkForMode(net, poly.TrainingModeCPUMC)
	net.Layers[0].WeightStore.Randomize(7, 0.1)

	desc := accel.LayerDesc{LayerName: "MatMul", DType: "FP16", SizeLabel: "small"}
	want, err := plug.WeightBytes(desc)
	if err != nil {
		t.Fatal(err)
	}
	got := poly.LayerWeightBytesForAccel(&net.Layers[0])
	if uintptr(len(got)) != want {
		t.Fatalf("bytes len=%d plugin want=%d", len(got), want)
	}
}
