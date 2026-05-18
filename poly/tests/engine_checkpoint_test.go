package poly_test

import (
	"math"
	"testing"

	. "github.com/openfluke/loom/poly"
)

// Engine truth: train → save native checkpoint → load → forward matches (not just blob equality).
func TestCNN1TrainSaveLoadForwardParity(t *testing.T) {
	cases := []struct {
		name      string
		dtype     DType
		scale     float32
		tolerance float64
	}{
		{"fp8e4m3", DTypeFP8E4M3, 0.01, 0.05},
		{"int4", DTypeInt4, 0.01, 0.55},
		{"int8", DTypeInt8, 0.01, 0.05},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			net := NewVolumetricNetwork(1, 1, 1, 1)
			l := net.GetLayer(0, 0, 0, 0)
			l.Type = LayerCNN1
			l.DType = tc.dtype
			l.Filters = 16
			l.InputChannels = 3
			l.InputHeight = 32
			l.OutputHeight = 32
			l.KernelSize = 3
			l.Stride = 1
			l.Padding = 1
			l.Activation = ActivationReLU
			l.WeightStore = NewWeightStore(16 * 3 * 3)
			l.WeightStore.Scale = tc.scale
			for i := range l.WeightStore.Master {
				l.WeightStore.Master[i] = float32(i%17)*0.03 - 0.2
			}
			l.EnableMultiCoreTiling = true
			net.EnableMultiCoreTiling = true

			batch := 8
			input := NewTensor[float32](batch, 3, 32)
			for i := range input.Data {
				input.Data[i] = float32(i%13)*0.1 - 0.6
			}
			target := NewTensor[float32](batch, 16, 32)
			for i := range target.Data {
				target.Data[i] = float32(i%7) * 0.05
			}

			cfg := DefaultTrainingConfig()
			cfg.Epochs = 3
			cfg.Mode = TrainingModeCPUMC
			cfg.LearningRate = 0.01
			cfg.Verbose = false
			_, err := Train(net, []TrainingBatch[float32]{{Input: input, Target: target}}, cfg)
			if err != nil {
				t.Fatal(err)
			}

			l.ResetState()
			_, before := DispatchLayer(l, input, nil)

			data, err := SerializeNetwork(net)
			if err != nil {
				t.Fatal(err)
			}
			fileB64, fileScale, fileNative, err := LayerPersistenceFromJSON(data, 0)
			if err != nil || !fileNative || fileB64 == "" {
				t.Fatalf("checkpoint: native=%v err=%v", fileNative, err)
			}

			net2, err := DeserializeNetwork(data)
			if err != nil {
				t.Fatal(err)
			}
			l2 := net2.GetLayer(0, 0, 0, 0)
			if l2.WeightStore == nil {
				t.Fatal("no weight store after load")
			}
			decoded, err := DecodeNativeWeights(fileB64, tc.dtype)
			if err != nil {
				t.Fatal(err)
			}
			loaded := l2.WeightStore.Versions[tc.dtype]
			if loaded == nil || l2.WeightStore.Scale != fileScale || !NativeWeightsEncoded(decoded, loaded, tc.dtype) {
				t.Fatalf("loaded native mismatch scale=%v fileScale=%v", l2.WeightStore.Scale, fileScale)
			}

			l2.ResetState()
			_, after := DispatchLayer(l2, input, nil)
			if len(before.Data) != len(after.Data) {
				t.Fatalf("shape mismatch %v vs %v", before.Shape, after.Shape)
			}
			diff := 0.0
			for i := range before.Data {
				if d := math.Abs(float64(before.Data[i] - after.Data[i])); d > diff {
					diff = d
				}
			}
			if diff > tc.tolerance {
				t.Fatalf("forward diff %.4e > tol %.4e", diff, tc.tolerance)
			}
		})
	}
}
