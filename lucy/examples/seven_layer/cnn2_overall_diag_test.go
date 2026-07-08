package sevenlayer

import (
	"fmt"
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestCNN2LossCompare3x3(t *testing.T) {
	g := GridSpec{3, 3, 3}
	input1 := cnn1SuiteForGrid(g).MakeInput()
	input2 := cnn2SuiteForGrid(g).MakeInput()
	tc := allDTypes[0] // Float64
	for _, pair := range []struct {
		name string
		s    LayerSuite
		in   *poly.Tensor[float32]
	}{
		{"CNN1", cnn1SuiteForGrid(g), input1},
		{"CNN2", cnn2SuiteForGrid(g), input2},
	} {
		net, err := poly.BuildNetworkFromJSON(pair.s.BuildJSON(tc.jsonName))
		if err != nil {
			t.Fatal(err)
		}
		applyDType(net, tc)
		target := pair.s.MakeTarget(net, pair.in)
		t.Logf("%s loss before train=%.4e", pair.name, forwardLoss(net, pair.in, target))
	}
}

func TestCNN2Overall3x3(t *testing.T) {
	if testing.Short() {
		t.Skip("slow full CNN2 3x3 suite")
	}
	g := GridSpec{3, 3, 3}
	s := cnn2SuiteForGrid(g)
	epochs := trainEpochsForGrid(g)
	input := s.MakeInput()
	requiresLearn := layerRequiresLearn(s.PrimaryType)

	var fails []string
	for _, tc := range allDTypes {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			applyDType(net, tc)
			target := s.MakeTarget(net, input)

			fwdSC := captureForward(net, input, false)
			fwdMC := captureForward(net, input, true)
			fwdSimd := captureForwardSimd(net, input, true)
			bwdSC := captureBackward(net, input, target, false)
			bwdMC := captureBackward(net, input, target, true)

			detTol := tc.tolerance
			if detTol < 1e-10 {
				detTol = 1e-10
			}
			detOK := maxAbsDiff(fwdSC.out, fwdMC.out) <= detTol &&
				maxAbsDiff(append(bwdSC.dx, bwdSC.dw...), append(bwdMC.dx, bwdMC.dw...)) <= detTol*10
			simdOK := maxAbsDiff(fwdSC.out, fwdSimd.out) <= simdParityTol(s.PrimaryType, tc)

			lossBefore := forwardLoss(net, input, target)
			if !lossFiniteOK(lossBefore, lossBefore, requiresLearn) {
				t.Fatalf("LOSS before non-finite %.4e", lossBefore)
			}

			before := checkSaveReload(net, input, target, tc, lossBefore, phaseBefore, s.PrimaryType)
			entityBefore := checkEntitySaveReload(net, input, target, tc, lossBefore, phaseBefore, s.PrimaryType)

			netMC, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
			applyDType(netMC, tc)
			configureTrainingNet(netMC, tc, s.PrimaryType)
			netMC.ReleaseFP32MasterWhenIdle = true
			resMC, _, err := trainCPU(netMC, input, target, poly.TrainingModeCPUMC, tc, s.PrimaryType, epochs)
			if err != nil {
				t.Fatalf("train: %v", err)
			}
			lossInit := resMC.LossHistory[0]
			lossFinal := resMC.FinalLoss
			if len(resMC.LossHistory) > 0 {
				lossFinal = resMC.LossHistory[len(resMC.LossHistory)-1]
			}
			learned := lossFiniteOK(lossInit, lossFinal, requiresLearn) &&
				(!requiresLearn || trainingOK(lossInit, lossFinal, tc.dtype))

			finalizeTrainingNet(netMC, tc)
			after := checkSaveReload(netMC, input, target, tc, lossFinal, phaseAfter, s.PrimaryType)
			entityAfter := checkEntitySaveReload(netMC, input, target, tc, lossFinal, phaseAfter, s.PrimaryType)

			overall := before.pass && after.pass && entityBefore.pass && entityAfter.pass &&
				learned && detOK && simdOK && lossFiniteOK(lossInit, lossFinal, requiresLearn)
			if !overall {
				msg := fmt.Sprintf("%s: overall=%v det=%v simd=%v learn=%v before=%v after=%v entityB=%v entityA=%v loss %.4e→%.4e reloadΔ=%.2e",
					tc.name, overall, detOK, simdOK, learned, before.pass, after.pass, entityBefore.pass, entityAfter.pass,
					lossInit, lossFinal, after.lossDelta)
				fails = append(fails, msg)
				t.Fatal(msg)
			}
		})
	}
	if len(fails) > 0 {
		t.Fatalf("%d failures: %v", len(fails), fails)
	}
}
