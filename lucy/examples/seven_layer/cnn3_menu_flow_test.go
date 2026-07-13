package sevenlayer

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestCNN3Int32NativeMenuFlow(t *testing.T) {
	g := grid1()
	s := buildCNN3NativeSuite(g)
	var tc dtypeCase
	for _, c := range allDTypes {
		if c.dtype == poly.DTypeInt32 {
			tc = c
			break
		}
	}
	net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
	if err != nil {
		t.Fatal(err)
	}
	applyDType(net, tc)
	configureNativeNet(net, tc)
	prepareTrainingNet(net, tc.dtype)
	finalizeTrainingNet(net, tc)
	input := s.MakeInput()
	target := s.MakeTarget(net, input)
	setCPUMode(net, false)
	setSimdForward(net, false)

	_ = captureForward(net, input, false)
	_ = captureBackward(net, input, target, false)

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
	ok := trainingOK(li, lf, tc.dtype)
	t.Logf("menu flow Int32 loss %.4f -> %.4f trainOK=%v", li, lf, ok)
	if !ok {
		t.Fatalf("failed like user saw")
	}
}
