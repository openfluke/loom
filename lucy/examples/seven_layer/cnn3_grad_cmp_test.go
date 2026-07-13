package sevenlayer

import (
	"fmt"
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestCNN3NativeVsTiledGradByDType(t *testing.T) {
	g := GridSpec{1, 1, 1}
	s := cnn3SuiteForGrid(g)
	input := s.MakeInput()
	for _, tc := range allDTypes {
		if tc.dtype != poly.DTypeInt32 && tc.dtype != poly.DTypeInt64 && tc.dtype != poly.DTypeInt16 {
			continue
		}
		t.Run(tc.name, func(t *testing.T) {
			netN, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
			applyDType(netN, tc)
			configureNativeNet(netN, tc)
			target := s.MakeTarget(netN, input)
			bwdN := captureBackward(netN, input, target, false)

			netT, _ := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
			applyDType(netT, tc)
			netT.UseExactDType = false
			targetT := s.MakeTarget(netT, input)
			bwdT := captureBackward(netT, input, targetT, false)

			dx := maxAbsDiff(bwdN.dx, bwdT.dx)
			dw := maxAbsDiff(bwdN.dw, bwdT.dw)
			fmt.Printf("%s native vs tiled dx=%.4e dw=%.4e\n", tc.name, dx, dw)
		})
	}
}
