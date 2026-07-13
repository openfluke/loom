package sevenlayer

import (
	"fmt"
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestCNN3Int32NativeTrainRepro(t *testing.T) {
	g := GridSpec{1, 1, 1}
	s := cnn3SuiteForGrid(g)
	tc := allDTypes[8] // Int32 - find index
	for _, c := range allDTypes {
		if c.dtype == poly.DTypeInt32 {
			tc = c
			break
		}
	}
	input := s.MakeInput()
	net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
	if err != nil {
		t.Fatal(err)
	}
	applyDType(net, tc)
	configureNativeNet(net, tc)
	target := s.MakeTarget(net, input)
	res, _, err := trainCPU(net, input, target, poly.TrainingModeCPUSC, tc, poly.LayerCNN3, 30)
	if err != nil {
		t.Fatal(err)
	}
	li := res.LossHistory[0]
	lf := res.LossHistory[len(res.LossHistory)-1]
	ok := trainingOK(li, lf, tc.dtype)
	t.Logf("Int32 loss %.4f -> %.4f trainOK=%v", li, lf, ok)
	if !ok {
		t.Fatalf("Int32 train failed")
	}
}

func TestCNN3Int32NativeVsTiledGrad(t *testing.T) {
	g := GridSpec{1, 1, 1}
	s := cnn3SuiteForGrid(g)
	var tc dtypeCase
	for _, c := range allDTypes {
		if c.dtype == poly.DTypeInt32 {
			tc = c
			break
		}
	}
	input := s.MakeInput()

	netN, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
	applyDType(netN, tc)
	configureNativeNet(netN, tc)
	target := s.MakeTarget(netN, input)
	bwdN := captureBackward(netN, input, target, false)

	netT, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
	applyDType(netT, tc)
	netT.UseExactDType = false
	targetT := s.MakeTarget(netT, input)
	bwdT := captureBackward(netT, input, targetT, false)

	dx := maxAbsDiff(bwdN.dx, bwdT.dx)
	dw := maxAbsDiff(bwdN.dw, bwdT.dw)
	fmt.Printf("native vs tiled dx=%.4e dw=%.4e\n", dx, dw)
	if dx > 1e-2 || dw > 1e-2 {
		t.Fatalf("native vs tiled mismatch dx=%.4e dw=%.4e", dx, dw)
	}
}
