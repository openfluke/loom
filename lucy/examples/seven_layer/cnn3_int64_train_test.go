package sevenlayer

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func cnn3NativeMenuTrain(t *testing.T, exact bool) (lossInit, lossFinal float64, trainOK bool) {
	t.Helper()
	g := grid1()
	s := buildCNN3NativeSuite(g)
	var tc dtypeCase
	for _, c := range allDTypes {
		if c.dtype == poly.DTypeInt64 {
			tc = c
			break
		}
	}
	net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
	if err != nil {
		t.Fatal(err)
	}
	applyDType(net, tc)
	if exact {
		configureNativeNet(net, tc)
	} else {
		net.UseExactDType = false
	}
	prepareTrainingNet(net, tc.dtype)
	finalizeTrainingNet(net, tc)
	input := s.MakeInput()
	target := s.MakeTarget(net, input)
	setCPUMode(net, false)
	setSimdForward(net, false)
	if poly.Plan9SimdForwardForLayer(poly.LayerCNN3) {
		resetNetwork(net)
		_ = captureForwardSimd(net, input, true)
		resetNetwork(net)
		_ = captureBackwardSimd(net, input, target, true)
	}
	net.ReleaseFP32MasterWhenIdle = true
	cfg := poly.DefaultTrainingConfig()
	cfg.Epochs = 30
	cfg.LearningRate = trainingLearningRate(tc.dtype)
	cfg.GradientClip = 1.0
	cfg.Mode = poly.TrainingModeCPUSC
	cfg.Verbose = false
	res, err := poly.Train(net, []poly.TrainingBatch[float32]{{Input: input, Target: target}}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	li := res.LossHistory[0]
	lf := res.LossHistory[len(res.LossHistory)-1]
	return li, lf, trainingOK(li, lf, tc.dtype)
}

func TestCNN3Int64NativeMenuFlow(t *testing.T) {
	li, lf, ok := cnn3NativeMenuTrain(t, true)
	t.Logf("native exact Int64 loss %.4f -> %.4f trainOK=%v", li, lf, ok)
	if !ok {
		t.Fatalf("native Int64 train failed: %.4f -> %.4f", li, lf)
	}
}

func TestCNN3Int64TiledMenuFlow(t *testing.T) {
	li, lf, ok := cnn3NativeMenuTrain(t, false)
	t.Logf("tiled Int64 loss %.4f -> %.4f trainOK=%v", li, lf, ok)
}
