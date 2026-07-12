package sevenlayer

import (
	"fmt"
	"strings"
	"testing"

	"github.com/openfluke/loom/poly"
)

func cnn2SuiteForGrid(g GridSpec) LayerSuite {
	ch := cnnChannelEndpoints(g)
	sp := cnnSpatial(g)
	return LayerSuite{
		Name:        "CNN2",
		Grid:        g,
		PrimaryType: poly.LayerCNN2,
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "diag-cnn2", g)
			first := true
			forEachGridCell(g, func(z, y, x int) {
				for i := 0; i < sevenLayersPerCell; i++ {
					appendLayerJSON(&b, &first, fmt.Sprintf(
						`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"CNN2","activation":"RELU","dtype":"%s","input_channels":%d,"filters":%d,"input_height":%d,"input_width":%d,"output_height":%d,"output_width":%d,"kernel_size":3,"stride":1,"padding":1}`,
						z, y, x, i, jsonDType, ch[i], ch[i+1], sp, sp, sp, sp,
					))
				}
			})
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, ch[0], sp, sp) },
		MakeTarget: sinTarget,
	}
}

func TestCNN2DiagAllDTypes3x3(t *testing.T) {
	if testing.Short() {
		t.Skip("slow 3x3 CNN2 stack")
	}
	g := GridSpec{3, 3, 3}
	s := cnn2SuiteForGrid(g)
	input := s.MakeInput()
	for _, tc := range allDTypes {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			applyDType(net, tc)
			target := s.MakeTarget(net, input)
			fwdSC := captureForward(net, input, false)
			fwdMC := captureForward(net, input, true)
			fwdSimd := captureForwardSimd(net, input, true)
			bwdSC := captureBackward(net, input, target, false)
			bwdMC := captureBackward(net, input, target, true)
			detTol := tc.tolerance
			if detTol < 1e-10 {
				detTol = 1e-10
			}
			simdTol := simdParityTol(s.PrimaryType, tc)
			scmc := maxAbsDiff(fwdSC.out, fwdMC.out)
			simdDiff := maxAbsDiff(fwdSC.out, fwdSimd.out)
			bwdDiff := maxAbsDiff(append(bwdSC.dx, bwdSC.dw...), append(bwdMC.dx, bwdMC.dw...))
			detOK := scmc <= detTol && bwdDiff <= detTol*10
			simdOK := simdDiff <= simdTol
			if !detOK || !simdOK {
				t.Fatalf("scmc=%.4e bwd=%.4e simd=%.4e simdTol=%.4e detOK=%v simdOK=%v", scmc, bwdDiff, simdDiff, simdTol, detOK, simdOK)
			}
		})
	}
}

func TestCNN2SimdParityFloat32_1x1(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerCNN2) {
		t.Skip("no Plan 9 SIMD")
	}
	g := GridSpec{Depth: 1, Rows: 1, Cols: 1}
	s := cnn2SuiteForGrid(g)
	assertSCMCSimdParity(t, s, allDTypes[1])
}

func TestCNN2SimdParityAllGrids_Float32(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerCNN2) {
		t.Skip("no Plan 9 SIMD")
	}
	tc := allDTypes[1]
	for _, g := range StandardGrids {
		g := g
		t.Run(g.String(), func(t *testing.T) {
			if testing.Short() && g.StackLayers() > 7 {
				t.Skip("short mode")
			}
			s := cnn2SuiteForGrid(g)
			assertSCMCSimdParity(t, s, tc)
		})
	}
}

func TestCNN2ForwardSimdCapture1x1(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerCNN2) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	g := GridSpec{1, 1, 1}
	ch := cnnChannelEndpoints(g)
	sp := cnnSpatial(g)
	s := cnn2SuiteForGrid(g)
	tc := allDTypes[1]
	net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
	if err != nil {
		t.Fatal(err)
	}
	applyDType(net, tc)
	input := sinInput(4, ch[0], sp, sp)
	fwdMC := captureForward(net, input, true)
	fwdSimd := captureForwardSimd(net, input, true)
	fwdSC := captureForward(net, input, false)
	if maxAbsDiff(fwdSC.out, fwdSimd.out) > 1e-4 {
		t.Fatalf("SC vs SIMD diff %g", maxAbsDiff(fwdSC.out, fwdSimd.out))
	}
	t.Logf("CNN2 1x1 Float32: MC=%s SIMD=%s speedup=%s",
		formatDur(fwdMC.dur), formatDur(fwdSimd.dur),
		formatSimdSpeedup(fwdMC.dur, fwdSimd.dur))
}
