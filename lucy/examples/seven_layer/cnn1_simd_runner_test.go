package sevenlayer

import (
	"fmt"
	"strings"
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestCNN1ForwardSimdCapture1x1(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerCNN1) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}

	g := GridSpec{Depth: 1, Rows: 1, Cols: 1}
	ch := cnnChannelEndpoints(g)
	sp := cnnSpatial(g)
	build := func(jsonDType string) []byte {
		var b strings.Builder
		writeNetworkHeader(&b, "test-cnn1", g)
		first := true
		forEachGridCell(g, func(z, y, x int) {
			for i := 0; i < sevenLayersPerCell; i++ {
				appendLayerJSON(&b, &first, fmt.Sprintf(
					`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"CNN1","activation":"RELU","dtype":"%s","input_channels":%d,"filters":%d,"input_height":%d,"output_height":%d,"kernel_size":3,"stride":1,"padding":1}`,
					z, y, x, i, jsonDType, ch[i], ch[i+1], sp, sp,
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
	input := sinInput(4, ch[0], sp)

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
	t.Logf("CNN1 1x1 Float32: tiled SC=%s MC=%s SIMD=%s speedup=%s",
		formatDur(fwdSC.dur), formatDur(fwdMC.dur), formatDur(fwdSimd.dur),
		formatSimdSpeedup(fwdMC.dur, fwdSimd.dur))
}

func TestCNN1ForwardSimdCapture2x2(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerCNN1) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}

	g := GridSpec{Depth: 2, Rows: 2, Cols: 2}
	ch := cnnChannelEndpoints(g)
	sp := cnnSpatial(g)
	build := func(jsonDType string) []byte {
		var b strings.Builder
		writeNetworkHeader(&b, "test-cnn1-2x2", g)
		first := true
		forEachGridCell(g, func(z, y, x int) {
			for i := 0; i < sevenLayersPerCell; i++ {
				appendLayerJSON(&b, &first, fmt.Sprintf(
					`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"CNN1","activation":"RELU","dtype":"%s","input_channels":%d,"filters":%d,"input_height":%d,"output_height":%d,"kernel_size":3,"stride":1,"padding":1}`,
					z, y, x, i, jsonDType, ch[i], ch[i+1], sp, sp,
				))
			}
		})
		b.WriteString(`]}`)
		return []byte(b.String())
	}

	tc := allDTypes[1]
	net, err := poly.BuildNetworkFromJSON(build(tc.jsonName))
	if err != nil {
		t.Fatal(err)
	}
	applyDType(net, tc)
	input := sinInput(4, ch[0], sp)

	fwdMC := captureForward(net, input, true)
	fwdSimd := captureForwardSimd(net, input, true)
	tol := tc.tolerance
	if tol < 1e-4 {
		tol = 1e-4
	}
	fwdSC := captureForward(net, input, false)
	if maxAbsDiff(fwdSC.out, fwdSimd.out) > tol {
		t.Fatalf("SC vs SIMD diff %g > tol %g", maxAbsDiff(fwdSC.out, fwdSimd.out), tol)
	}
	t.Logf("CNN1 2x2 Float32: tiled MC=%s SIMD=%s speedup=%s",
		formatDur(fwdMC.dur), formatDur(fwdSimd.dur),
		formatSimdSpeedup(fwdMC.dur, fwdSimd.dur))
}

func TestCNN1SimdParityFloat32_1x1(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerCNN1) {
		t.Skip("no Plan 9 SIMD")
	}
	g := GridSpec{Depth: 1, Rows: 1, Cols: 1}
	s := cnn1SuiteForGrid(g)
	assertSCMCSimdParity(t, s, allDTypes[1])
}

func TestCNN1SimdParityAllGrids_Float32(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerCNN1) {
		t.Skip("no Plan 9 SIMD")
	}
	tc := allDTypes[1]
	for _, g := range StandardGrids {
		g := g
		t.Run(g.String(), func(t *testing.T) {
			if testing.Short() && g.StackLayers() > 7 {
				t.Skip("short mode")
			}
			s := cnn1SuiteForGrid(g)
			assertSCMCSimdParity(t, s, tc)
		})
	}
}

func TestCNN1ForwardSimdCapture3x3Float32(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerCNN1) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	if testing.Short() {
		t.Skip("3x3 CNN1 stack is slow")
	}

	g := GridSpec{Depth: 3, Rows: 3, Cols: 3}
	ch := cnnChannelEndpoints(g)
	sp := cnnSpatial(g)
	build := func(jsonDType string) []byte {
		var b strings.Builder
		writeNetworkHeader(&b, "test-cnn1-3x3", g)
		first := true
		forEachGridCell(g, func(z, y, x int) {
			for i := 0; i < sevenLayersPerCell; i++ {
				appendLayerJSON(&b, &first, fmt.Sprintf(
					`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"CNN1","activation":"RELU","dtype":"%s","input_channels":%d,"filters":%d,"input_height":%d,"output_height":%d,"kernel_size":3,"stride":1,"padding":1}`,
					z, y, x, i, jsonDType, ch[i], ch[i+1], sp, sp,
				))
			}
		})
		b.WriteString(`]}`)
		return []byte(b.String())
	}

	tc := allDTypes[1]
	net, err := poly.BuildNetworkFromJSON(build(tc.jsonName))
	if err != nil {
		t.Fatal(err)
	}
	applyDType(net, tc)
	input := sinInput(4, ch[0], sp)

	activeBenchIters = benchItersForGrid(g)
	fwdSC := captureForward(net, input, false)
	fwdMC := captureForward(net, input, true)
	fwdSimd := captureForwardSimd(net, input, true)
	if maxAbsDiff(fwdSC.out, fwdSimd.out) > 1e-4 {
		t.Fatalf("SC vs SIMD diff %g", maxAbsDiff(fwdSC.out, fwdSimd.out))
	}
	t.Logf("CNN1 3x3 Float32: tiled MC=%s SIMD=%s speedup=%s stack=%d",
		formatDur(fwdMC.dur), formatDur(fwdSimd.dur),
		formatSimdSpeedup(fwdMC.dur, fwdSimd.dur), g.StackLayers())
}
