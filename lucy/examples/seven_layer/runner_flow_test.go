package sevenlayer

import (
	"fmt"
	"strings"
	"testing"

	"github.com/openfluke/loom/poly"
)

// Replicates runner.go RunLayerSuite body for Dense 1x1 through Int8 (full bench path).
func TestRunnerFlowInt8Dense1x1(t *testing.T) {
	g := GridSpec{Depth: 1, Rows: 1, Cols: 1}
	dims := denseEndpoints(g)
	acts := []string{"LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR"}
	s := LayerSuite{
		Name: "Dense", Grid: g, PrimaryType: poly.LayerDense,
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "loom-seven-dense", g)
			first := true
			forEachGridCell(g, func(z, y, x int) {
				for i := 0; i < sevenLayersPerCell; i++ {
					appendLayerJSON(&b, &first, fmt.Sprintf(
						`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"DENSE","activation":"%s","dtype":"%s","input_height":%d,"output_height":%d}`,
						z, y, x, i, acts[i], jsonDType, dims[i], dims[i+1],
					))
				}
			})
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, dims[0]) },
		MakeTarget: sinTarget,
	}
	epochs := 2
	activeBenchIters = benchItersForGrid(g)
	input := s.MakeInput()

	for _, tc := range allDTypes {
		if tc.name == "Uint8" {
			break
		}
		net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		if err != nil {
			t.Fatal(err)
		}
		applyDType(net, tc)
		target := s.MakeTarget(net, input)
		fwdSC := captureForward(net, input, false)
		_ = fwdSC
		_ = captureForward(net, input, true)
		bwdSC := captureBackward(net, input, target, false)
		bwdMC := captureBackward(net, input, target, true)
		_ = bwdSC
		_ = bwdMC
		configureInferenceNet(net)
		lossBefore := forwardLoss(net, input, target)
		_ = checkSaveReload(net, input, target, tc, lossBefore, phaseBefore, s.PrimaryType)
		_ = checkEntitySaveReload(net, input, target, tc, lossBefore, phaseBefore, s.PrimaryType)
		netSC, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		applyDType(netSC, tc)
		configureTrainingNet(netSC, tc, s.PrimaryType)
		netSC.ReleaseFP32MasterWhenIdle = true
		trainCPU(netSC, input, target, poly.TrainingModeCPUSC, tc, s.PrimaryType, epochs)
		netMC, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		applyDType(netMC, tc)
		configureTrainingNet(netMC, tc, s.PrimaryType)
		netMC.ReleaseFP32MasterWhenIdle = true
		trainCPU(netMC, input, target, poly.TrainingModeCPUMC, tc, s.PrimaryType, epochs)
		readMemSnapshot()
	}

	tc := allDTypes[12]
	net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
	if err != nil {
		t.Fatal(err)
	}
	applyDType(net, tc)
	target := s.MakeTarget(net, input)
	captureForward(net, input, false)
	captureForward(net, input, true)
	captureBackward(net, input, target, false)
	captureBackward(net, input, target, true)
	configureInferenceNet(net)
}
