package sevenlayer

import (
	"fmt"
	"strings"
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestMHAForwardSimdCapture1x1(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerMultiHeadAttention) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}

	g := GridSpec{Depth: 1, Rows: 1, Cols: 1}
	m := mhaShapeFor(g)
	build := func(jsonDType string) []byte {
		var b strings.Builder
		writeNetworkHeader(&b, "test-mha", g)
		first := true
		forEachGridCell(g, func(z, y, x int) {
			for i := 0; i < sevenLayersPerCell; i++ {
				appendLayerJSON(&b, &first, fmt.Sprintf(
					`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"MHA","activation":"RELU","dtype":"%s","d_model":%d,"num_heads":%d,"seq_length":%d}`,
					z, y, x, i, jsonDType, m.dModel, m.heads, m.seq,
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
	input := sinInput(4, m.seq, m.dModel)

	fwdSC := captureForward(net, input, false)
	fwdMC := captureForward(net, input, true)
	fwdSimd := captureForwardSimd(net, input, true)

	if len(fwdSimd.out) == 0 {
		t.Fatal("empty SIMD forward output")
	}
	tol := tc.tolerance
	if tol < 1e-4 {
		tol = 1e-4
	}
	if maxAbsDiff(fwdSC.out, fwdSimd.out) > tol {
		t.Fatalf("SC vs SIMD diff %g > tol %g", maxAbsDiff(fwdSC.out, fwdSimd.out), tol)
	}
	t.Logf("MHA 1x1 Float32: tiled SC=%s MC=%s SIMD=%s speedup=%s",
		formatDur(fwdSC.dur), formatDur(fwdMC.dur), formatDur(fwdSimd.dur),
		formatSimdSpeedup(fwdMC.dur, fwdSimd.dur))
}
