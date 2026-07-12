package sevenlayer

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestMHANativeIntTrainStability(t *testing.T) {
	s := buildMHANativeSuite(grid1())
	for _, tc := range []dtypeCase{allDTypes[12], allDTypes[14]} { // Int8, Int4
		t.Run(tc.name, func(t *testing.T) {
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
			lossInit := res.LossHistory[0]
			lossFinal := res.FinalLoss
			if len(res.LossHistory) > 0 {
				lossFinal = res.LossHistory[len(res.LossHistory)-1]
			}
			if !lossFiniteOK(lossInit, lossFinal, layerRequiresLearn(poly.LayerMultiHeadAttention)) {
				t.Fatalf("non-finite loss init=%v final=%v", lossInit, lossFinal)
			}
			if !trainingOK(lossInit, lossFinal, tc.dtype) {
				t.Fatalf("training exploded or diverged init=%.4f final=%.4f", lossInit, lossFinal)
			}
		})
	}
}
