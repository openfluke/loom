package sevenlayer

import (
	"fmt"
	"strings"
	"testing"

	"github.com/openfluke/loom/poly"
)

// Regression: Int8 entity save/reload after Uint16 full suite flow (was []int8 vs []uint8 in GetActive).
func TestInt8EntitySaveReloadAfterUint16Flow(t *testing.T) {
	g := GridSpec{Depth: 1, Rows: 1, Cols: 1}
	dims := denseEndpoints(g)
	acts := []string{"LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR"}
	buildJSON := func(jsonDType string) []byte {
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
	}
	input := sinInput(4, dims[0])
	activeBenchIters = 3

	u16 := allDTypes[11]
	runPriorFlow(t, buildJSON, u16, input)

	tc := allDTypes[12]
	net, _ := poly.BuildNetworkFromJSON(buildJSON(tc.jsonName))
	applyDType(net, tc)
	target := sinTarget(net, input)
	captureForward(net, input, false)
	captureForward(net, input, true)
	captureBackward(net, input, target, false)
	captureBackward(net, input, target, true)
	configureInferenceNet(net)
	lossBefore := forwardLoss(net, input, target)
	if r := checkEntitySaveReload(net, input, target, tc, lossBefore, phaseBefore, poly.LayerDense); !r.pass && r.err != "" {
		t.Fatalf("entity save/reload: %s", r.err)
	}
}

func runPriorFlow(t *testing.T, buildJSON func(string) []byte, tc dtypeCase, input *poly.Tensor[float32]) {
	t.Helper()
	net, _ := poly.BuildNetworkFromJSON(buildJSON(tc.jsonName))
	applyDType(net, tc)
	target := sinTarget(net, input)
	captureForward(net, input, false)
	captureForward(net, input, true)
	captureBackward(net, input, target, false)
	captureBackward(net, input, target, true)
	configureInferenceNet(net)
	lossBefore := forwardLoss(net, input, target)
	_ = checkSaveReload(net, input, target, tc, lossBefore, phaseBefore, poly.LayerDense)
	_ = checkEntitySaveReload(net, input, target, tc, lossBefore, phaseBefore, poly.LayerDense)
	runOneDTypeTrain(t, buildJSON, tc, input, target, poly.LayerDense)
}

func runOneDTypeTrain(t *testing.T, buildJSON func(string) []byte, tc dtypeCase, input, target *poly.Tensor[float32], primary poly.LayerType) {
	t.Helper()
	netSC, _ := poly.BuildNetworkFromJSON(buildJSON(tc.jsonName))
	applyDType(netSC, tc)
	configureTrainingNet(netSC, tc, primary)
	netSC.ReleaseFP32MasterWhenIdle = true
	if _, _, err := trainCPU(netSC, input, target, poly.TrainingModeCPUSC, tc, primary, 2); err != nil {
		t.Fatal(err)
	}
	netMC, _ := poly.BuildNetworkFromJSON(buildJSON(tc.jsonName))
	applyDType(netMC, tc)
	configureTrainingNet(netMC, tc, primary)
	netMC.ReleaseFP32MasterWhenIdle = true
	if _, _, err := trainCPU(netMC, input, target, poly.TrainingModeCPUMC, tc, primary, 2); err != nil {
		t.Fatal(err)
	}
	readMemSnapshot()
}
