package sevenlayer

import (
	"fmt"
	"strings"
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestSwiGLUForwardSimdCapture1x1(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerSwiGLU) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}

	g := GridSpec{Depth: 1, Rows: 1, Cols: 1}
	dims := swigluEndpoints(g)
	build := func(jsonDType string) []byte {
		var b strings.Builder
		writeNetworkHeader(&b, "test-swiglu", g)
		first := true
		forEachGridCell(g, func(z, y, x int) {
			for i := 0; i < sevenLayersPerCell; i++ {
				appendLayerJSON(&b, &first, fmt.Sprintf(
					`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"SWIGLU","activation":"RELU","dtype":"%s","input_height":%d,"output_height":%d}`,
					z, y, x, i, jsonDType, dims[i], dims[i+1],
				))
			}
		})
		b.WriteString(`]}`)
		return []byte(b.String())
	}

	tc := allDTypes[1] // Float32
	net, err := poly.BuildNetworkFromJSON(build(tc.jsonName))
	if err != nil {
		t.Fatal(err)
	}
	applyDType(net, tc)
	input := sinInput(4, dims[0])

	fwdSC := captureForward(net, input, false)
	fwdMC := captureForward(net, input, true)
	fwdSimd := captureForwardSimd(net, input, true)

	if len(fwdSimd.out) == 0 {
		t.Fatal("empty SIMD forward output")
	}
	if maxAbsDiff(fwdSC.out, fwdSimd.out) > tc.tolerance {
		t.Fatalf("SC vs SIMD diff %g > tol %g", maxAbsDiff(fwdSC.out, fwdSimd.out), tc.tolerance)
	}
	t.Logf("SwiGLU 1x1 Float32: tiled SC=%s MC=%s SIMD=%s speedup=%s",
		formatDur(fwdSC.dur), formatDur(fwdMC.dur), formatDur(fwdSimd.dur),
		formatSimdSpeedup(fwdMC.dur, fwdSimd.dur))
}
