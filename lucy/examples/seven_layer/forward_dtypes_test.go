package sevenlayer

import (
	"fmt"
	"strings"
	"testing"

	"github.com/openfluke/loom/poly"
)

// Exercises the benchmark-net path: applyDType → forward MC for every dtype.
// configureInferenceNet must not run before forward (see runner.go).
func TestDenseForwardAllDTypes1x1(t *testing.T) {
	g := GridSpec{Depth: 1, Rows: 1, Cols: 1}
	dims := denseEndpoints(g)
	acts := []string{"LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR"}
	build := func(jsonDType string) []byte {
		var b strings.Builder
		writeNetworkHeader(&b, "test-dense", g)
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
	for _, tc := range allDTypes {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			net, err := poly.BuildNetworkFromJSON(build(tc.jsonName))
			if err != nil {
				t.Fatal(err)
			}
			applyDType(net, tc)
			target := sinTarget(net, input)
			_ = target
			if cap := captureForward(net, input, true); len(cap.out) == 0 {
				t.Fatal("empty forward output")
			}
			configureInferenceNet(net)
			if cap := captureForward(net, input, true); len(cap.out) == 0 {
				t.Fatal("empty forward after configureInferenceNet")
			}
		})
	}
}
