package sevenlayer

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestEmbeddingNativeSimdTrainStability(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerEmbedding) {
		t.Skip("no Plan 9 SIMD")
	}
	s := buildEmbeddingNativeSuite(GridSpec{Depth: 1, Rows: 1, Cols: 1})
	for _, tc := range []dtypeCase{allDTypes[14], allDTypes[15], allDTypes[18], allDTypes[20]} {
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

func TestResidualNativeSimdForward3x3(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerResidual) {
		t.Skip("no Plan 9 SIMD")
	}
	s := buildResidualNativeSuite(GridSpec{Depth: 3, Rows: 3, Cols: 3})
	tc := allDTypes[1] // Float32
	net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
	if err != nil {
		t.Fatal(err)
	}
	applyDType(net, tc)
	configureNativeNet(net, tc)
	input := s.MakeInput()
	skip := sinInput(input.Shape...)
	setSimdForward(net, false)
	outSC, _, _ := poly.ForwardPolymorphic(net, input)
	_ = outSC
	setSimdForward(net, true)
	outSimd, _, _ := poly.ForwardPolymorphic(net, input)
	// Residual network: compare first layer residual add via direct layer call
	l := &net.Layers[0]
	l.Network = net
	in := input
	sk := skip
	if len(in.Data) != len(sk.Data) {
		t.Fatalf("input/skip size mismatch")
	}
	setSimdForward(net, false)
	_, postSC := poly.ResidualForwardPolymorphic(l, in, sk)
	setSimdForward(net, true)
	_, postSimd := poly.ResidualForwardPolymorphic(l, in, sk)
	if d := maxAbsDiff(postSC.Data, postSimd.Data); d > 1e-5 {
		t.Fatalf("residual native SC vs SIMD diff %g", d)
	}
	_ = outSimd
}
