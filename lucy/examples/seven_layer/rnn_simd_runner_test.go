package sevenlayer

import (
	"fmt"
	"strings"
	"testing"

	"github.com/openfluke/loom/poly"
)

func rnnSuiteForGrid(g GridSpec) LayerSuite {
	dims := rnnEndpoints(g)
	return LayerSuite{
		Name:        "RNN",
		Grid:        g,
		PrimaryType: poly.LayerRNN,
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "diag-rnn", g)
			first := true
			forEachGridCell(g, func(z, y, x int) {
				for i := 0; i < sevenLayersPerCell; i++ {
					appendLayerJSON(&b, &first, fmt.Sprintf(
						`{"z":%d,"y":%d,"x":%d,"l":%d,"type":"RNN","activation":"TANH","dtype":"%s","input_height":%d,"output_height":%d}`,
						z, y, x, i, jsonDType, dims[i], dims[i+1],
					))
				}
			})
			b.WriteString(`]}`)
			return []byte(b.String())
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, dims[0]) },
		MakeTarget: sinTarget,
	}
}

func TestRNNDiagAllDTypes1x1(t *testing.T) {
	if testing.Short() {
		t.Skip("slow RNN stack")
	}
	g := GridSpec{1, 1, 1}
	s := rnnSuiteForGrid(g)
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

func TestRNNSimdParityFloat32_1x1(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerRNN) {
		t.Skip("no Plan 9 SIMD")
	}
	g := GridSpec{Depth: 1, Rows: 1, Cols: 1}
	s := rnnSuiteForGrid(g)
	assertSCMCSimdParity(t, s, allDTypes[1])
}

func TestRNNSimdParityAllGrids_Float32(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerRNN) {
		t.Skip("no Plan 9 SIMD")
	}
	tc := allDTypes[1]
	for _, g := range StandardGrids {
		g := g
		t.Run(g.String(), func(t *testing.T) {
			if testing.Short() && g.StackLayers() > 7 {
				t.Skip("short mode")
			}
			s := rnnSuiteForGrid(g)
			assertSCMCSimdParity(t, s, tc)
		})
	}
}

// Verifies SIMD now actually runs (and matches tiled) on the narrow 2³/3³ grids,
// where the old width gate used to silently fall back to tiled.
func TestRNNSimdParityAllGrids(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerRNN) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	for _, g := range StandardGrids {
		g := g
		t.Run(g.String(), func(t *testing.T) {
			s := rnnSuiteForGrid(g)
			tc := allDTypes[1] // Float32
			net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			applyDType(net, tc)
			input := s.MakeInput()
			fwdSC := captureForward(net, input, false)
			fwdSimd := captureForwardSimd(net, input, true)
			if d := maxAbsDiff(fwdSC.out, fwdSimd.out); d > 1e-4 {
				t.Fatalf("grid %s SC vs SIMD diff %g > 1e-4", g, d)
			}
			t.Logf("RNN %s Float32: SC=%s SIMD=%s speedup=%s",
				g, formatDur(fwdSC.dur), formatDur(fwdSimd.dur),
				formatSimdSpeedup(fwdSC.dur, fwdSimd.dur))
		})
	}
}

func TestRNNForwardSimdCapture1x1(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerRNN) {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	g := GridSpec{1, 1, 1}
	dims := rnnEndpoints(g)
	s := rnnSuiteForGrid(g)
	tc := allDTypes[1]
	net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
	if err != nil {
		t.Fatal(err)
	}
	applyDType(net, tc)
	input := sinInput(4, dims[0])
	fwdMC := captureForward(net, input, true)
	fwdSimd := captureForwardSimd(net, input, true)
	fwdSC := captureForward(net, input, false)
	if maxAbsDiff(fwdSC.out, fwdSimd.out) > 1e-4 {
		t.Fatalf("SC vs SIMD diff %g", maxAbsDiff(fwdSC.out, fwdSimd.out))
	}
	t.Logf("RNN 1x1 Float32: MC=%s SIMD=%s speedup=%s",
		formatDur(fwdMC.dur), formatDur(fwdSimd.dur),
		formatSimdSpeedup(fwdMC.dur, fwdSimd.dur))
}
