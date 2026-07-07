package sevenlayer

import (
	"fmt"
	"strings"
	"testing"

	"github.com/openfluke/loom/poly"
)

func cnn1SuiteForGrid(g GridSpec) LayerSuite {
	ch := cnnChannelEndpoints(g)
	sp := cnnSpatial(g)
	return LayerSuite{
		Name:        "CNN1",
		Grid:        g,
		PrimaryType: poly.LayerCNN1,
		BuildJSON: func(jsonDType string) []byte {
			var b strings.Builder
			writeNetworkHeader(&b, "diag-cnn1", g)
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
		},
		MakeInput:  func() *poly.Tensor[float32] { return sinInput(4, ch[0], sp) },
		MakeTarget: sinTarget,
	}
}

func TestCNN1DiagAllDTypes3x3(t *testing.T) {
	g := GridSpec{3, 3, 3}
	s := cnn1SuiteForGrid(g)
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
