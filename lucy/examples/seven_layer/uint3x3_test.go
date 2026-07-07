package sevenlayer

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestUintWideDense3x3Forward(t *testing.T) {
	g := GridSpec{Depth: 3, Rows: 3, Cols: 3}
	dims := denseEndpoints(g)
	acts := []string{"LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR", "LINEAR"}
	build := func(jsonDType string) *poly.VolumetricNetwork {
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
		net, err := poly.BuildNetworkFromJSON([]byte(b.String()))
		if err != nil {
			t.Fatal(err)
		}
		return net
	}
	input := sinInput(4, dims[0])
	for _, name := range []string{"UINT64", "UINT32", "UINT16", "UINT8", "FLOAT32"} {
		var tc dtypeCase
		for _, d := range allDTypes {
			if d.jsonName == name {
				tc = d
				break
			}
		}
		net := build(tc.jsonName)
		applyDType(net, tc)
		target := sinTarget(net, input)
		net.SetSimdForward(false)
		outSC, _, _ := poly.ForwardPolymorphic(net, input)
		net.SetSimdForward(true)
		outSimd, _, _ := poly.ForwardPolymorphic(net, input)
		loss := poly.CalculateLoss(outSC, target, "mse")
		var maxDiff, sumSC float64
		for i := range outSC.Data {
			sumSC += float64(outSC.Data[i])
			d := float64(outSC.Data[i] - outSimd.Data[i])
			if d < 0 {
				d = -d
			}
			if d > maxDiff {
				maxDiff = d
			}
		}
		t.Logf("%s dim=%d: loss=%.4e outSC_sum=%.4e max|sc-simd|=%.4e tol=%.4e",
			tc.name, dims[0], loss, sumSC, maxDiff, tc.tolerance)
	}
}

func TestUintWideDense3x3BackwardNaN(t *testing.T) {
	g := GridSpec{Depth: 3, Rows: 3, Cols: 3}
	dims := denseEndpoints(g)
	for _, tc := range allDTypes {
		if tc.name != "Uint64" && tc.name != "Uint32" && tc.name != "Uint16" {
			continue
		}
		// minimal build inline
		var b strings.Builder
		writeNetworkHeader(&b, "test", g)
		first := true
		forEachGridCell(g, func(z, y, x int) {
			for i := 0; i < sevenLayersPerCell; i++ {
				appendLayerJSON(&b, &first, fmt.Sprintf(
					`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"DENSE","activation":"LINEAR","dtype":"%s","input_height":%d,"output_height":%d}`,
					z, y, x, i, tc.jsonName, dims[i], dims[i+1],
				))
			}
		})
		b.WriteString(`]}`)
		net, _ := poly.BuildNetworkFromJSON([]byte(b.String()))
		applyDType(net, tc)
		input := sinInput(4, dims[0])
		target := sinTarget(net, input)
		bwd := captureBackward(net, input, target, false)
		var nan bool
		for _, v := range append(bwd.dx, bwd.dw...) {
			if math.IsNaN(float64(v)) {
				nan = true
				break
			}
		}
		t.Logf("%s backward nan=%v", tc.name, nan)
	}
}
