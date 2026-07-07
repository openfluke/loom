package sevenlayer

import (
	"fmt"
	"strings"
	"testing"

	"github.com/openfluke/loom/poly"
)

func mhaSuiteForGrid(g GridSpec) LayerSuite {
	m := mhaShapeFor(g)
	return LayerSuite{
		Name:        "MHA",
		Grid:        g,
		PrimaryType: poly.LayerMultiHeadAttention,
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "diag-mha", g)
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
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, m.seq, m.dModel) },
		MakeTarget: sinTarget,
	}
}

func diagMHAForward(t *testing.T, g GridSpec) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerMultiHeadAttention) {
		t.Skip("no SIMD")
	}
	s := mhaSuiteForGrid(g)
	input := s.MakeInput()
	t.Logf("grid %s d_model=%d heads=%d seq=%d stack=%d",
		g, mhaShapeFor(g).dModel, mhaShapeFor(g).heads, mhaShapeFor(g).seq, g.StackLayers())

	for _, tc := range allDTypes {
		if tc.name != "Uint64" && tc.name != "Uint32" && tc.name != "Uint16" {
			continue
		}
		net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
		if err != nil {
			t.Fatalf("%s build: %v", tc.name, err)
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
		if s.PrimaryType == poly.LayerMultiHeadAttention && detTol < 1e-4 {
			detTol = 1e-4
		}
		simdTol := simdParityTol(s.PrimaryType, tc)

		scmc := maxAbsDiff(fwdSC.out, fwdMC.out)
		simdDiff := maxAbsDiff(fwdSC.out, fwdSimd.out)
		bwdDiff := maxAbsDiff(append(bwdSC.dx, bwdSC.dw...), append(bwdMC.dx, bwdMC.dw...))
		loss := forwardLoss(net, input, target)

		detOK := scmc <= detTol && bwdDiff <= detTol*10
		simdOK := simdDiff <= simdTol
		ok := detOK && simdOK

		if !ok {
			t.Errorf("%s grid %s: sc-mc=%.4e bwd=%.4e simd=%.4e simdTol=%.4e loss=%.4e detOK=%v simdOK=%v",
				tc.name, g, scmc, bwdDiff, simdDiff, simdTol, loss, detOK, simdOK)
		}
	}
}

func TestMHADiagForward1x1(t *testing.T) {
	diagMHAForward(t, GridSpec{1, 1, 1})
}

func TestMHADiagForward3x3(t *testing.T) {
	diagMHAForward(t, GridSpec{3, 3, 3})
}
