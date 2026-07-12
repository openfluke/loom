package poly

import (
	"fmt"
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestDenseTrueNativeInt8UpdatesWeights(t *testing.T) {
	net := buildDenseNativeTestNet(t, poly.DTypeInt8, "INT8", 8, 4)
	layer := &net.Layers[0]
	if !poly.DenseUsesTrueNative(layer) {
		t.Fatal("expected true-native path for Int8 dense")
	}
	ws := layer.WeightStore
	before := append([]uint8(nil), ws.Versions[poly.DTypeInt8].([]uint8)...)

	input := poly.NewTensor[float32](1, 8)
	for i := range input.Data {
		input.Data[i] = 0.1 * float32(i+1)
	}
	target := poly.NewTensor[float32](1, 4)
	for i := range target.Data {
		target.Data[i] = 0.05 * float32(i+1)
	}
	cfg := poly.DefaultTrainingConfig()
	cfg.Epochs = 5
	cfg.LearningRate = 0.01
	cfg.Mode = poly.TrainingModeCPUSC
	cfg.Verbose = false
	_, err := poly.Train(net, []poly.TrainingBatch[float32]{{Input: input, Target: target}}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	after := ws.Versions[poly.DTypeInt8].([]uint8)
	changed := 0
	for i := range before {
		if before[i] != after[i] {
			changed++
		}
	}
	if changed == 0 {
		t.Fatal("true-native Int8 training did not change native weight storage")
	}
}

func TestDenseNativeExactPathInt8(t *testing.T) {
	net := buildDenseNativeTestNet(t, poly.DTypeInt8, "INT8", 8, 4)
	layer := &net.Layers[0]
	if !poly.DenseUsesNativeExact(layer) {
		t.Fatal("expected native-exact path for Int8 dense")
	}
	runDenseNativeFwdBwd(t, net, 4)
}

func TestDenseNativeExactFloat16Rounds(t *testing.T) {
	net := buildDenseNativeTestNet(t, poly.DTypeFloat16, "FLOAT16", 4, 4)
	if !poly.DenseUsesNativeExact(&net.Layers[0]) {
		t.Fatal("expected native-exact for Float16")
	}
	runDenseNativeFwdBwd(t, net, 4)
}

func TestDenseNativeExactFloat32UsesMaster(t *testing.T) {
	net := buildDenseNativeTestNet(t, poly.DTypeFloat32, "FLOAT32", 8, 4)
	if !poly.DenseUsesNativeExact(&net.Layers[0]) {
		t.Fatal("Float32 dense should use native-exact when UseExactDType")
	}
	runDenseNativeFwdBwd(t, net, 4)
}

func buildDenseNativeTestNet(t *testing.T, dt poly.DType, jsonName string, inH, outH int) *poly.VolumetricNetwork {
	t.Helper()
	json := []byte(fmt.Sprintf(
		`{"id":"test-dense-native","depth":1,"rows":1,"cols":1,"layers_per_cell":1,"layers":[`+
			`{"z":0,"y":0,"x":0,"l":0,"type":"DENSE","activation":"LINEAR","dtype":"%s","input_height":%d,"output_height":%d}`+
			`]}`,
		jsonName, inH, outH,
	))
	net, err := poly.BuildNetworkFromJSON(json)
	if err != nil {
		t.Fatal(err)
	}
	net.UseExactDType = poly.IsDenseNativeExactDType(dt)
	for i := range net.Layers {
		net.Layers[i].DType = dt
		if net.Layers[i].WeightStore != nil {
			net.Layers[i].WeightStore.Scale = 0.01
			for j := range net.Layers[i].WeightStore.Master {
				net.Layers[i].WeightStore.Master[j] = 0.01 * float32(j%5+1)
			}
			net.Layers[i].WeightStore.ForceMorph(dt)
		}
	}
	return net
}

func runDenseNativeFwdBwd(t *testing.T, net *poly.VolumetricNetwork, outH int) {
	t.Helper()
	inH := net.Layers[0].InputHeight
	input := poly.NewTensor[float32](1, inH)
	for i := range input.Data {
		input.Data[i] = 0.1 * float32(i+1)
	}
	out, _, _ := poly.ForwardPolymorphic(net, input)
	pre := out // LINEAR activation: preAct == postAct
	gradOut := poly.NewTensor[float32](1, outH)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.01
	}
	_, grads, _ := poly.BackwardPolymorphic(net, gradOut, []*poly.Tensor[float32]{input}, []*poly.Tensor[float32]{pre})
	if len(grads) == 0 || grads[0][1] == nil || len(grads[0][1].Data) == 0 {
		t.Fatal("empty weight grad")
	}
}
