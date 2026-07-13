package sevenlayer

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestLSTMNativeSimdTrainStability(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerLSTM) {
		t.Skip("no Plan 9 SIMD")
	}
	s := lstmSuiteForGrid(GridSpec{Depth: 3, Rows: 3, Cols: 3})
	for _, tc := range []dtypeCase{allDTypes[14], allDTypes[15], allDTypes[18], allDTypes[20]} {
		// Int4, Uint4, Uint2, Binary
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
			if err != nil {
				t.Fatal(err)
			}
			applyDType(net, tc)
			configureNativeNet(net, tc)
			input := s.MakeInput()
			target := s.MakeTarget(net, input)
			epochs := trainEpochsForGrid(s.Grid)
			res, _, err := trainNativeExact(net, input, target, tc, poly.TrainingModeCPUSimd, epochs)
			if err != nil {
				t.Fatal(err)
			}
			lossInit := res.LossHistory[0]
			lossFinal := res.FinalLoss
			if len(res.LossHistory) > 0 {
				lossFinal = res.LossHistory[len(res.LossHistory)-1]
			}
			if !trainingOK(lossInit, lossFinal, tc.dtype) {
				t.Fatalf("%s Nat-SIMD train diverged init=%.4f final=%.4f", tc.name, lossInit, lossFinal)
			}
		})
	}
}
