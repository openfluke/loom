package sevenlayer

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

// assertSCMCSimdParity checks forward/backward SC·MC·SIMD parity for one dtype on one layer suite.
func assertSCMCSimdParity(t *testing.T, s LayerSuite, tc dtypeCase) {
	t.Helper()
	net, err := poly.BuildNetworkFromJSON(s.BuildJSON(tc.jsonName))
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	applyDType(net, tc)
	input := s.MakeInput()
	target := s.MakeTarget(net, input)

	fwdSC := captureForward(net, input, false)
	fwdMC := captureForward(net, input, true)
	bwdSC := captureBackward(net, input, target, false)
	bwdMC := captureBackward(net, input, target, true)

	detTol := tc.tolerance
	if detTol < 1e-10 {
		detTol = 1e-10
	}
	scmcFwd := maxAbsDiff(fwdSC.out, fwdMC.out)
	scmcBwd := maxAbsDiff(append(bwdSC.dx, bwdSC.dw...), append(bwdMC.dx, bwdMC.dw...))
	if scmcFwd > detTol || scmcBwd > detTol*10 {
		t.Fatalf("%s %s SC↔MC: fwd=%.4e bwd=%.4e", s.Name, tc.name, scmcFwd, scmcBwd)
	}

	if !poly.Plan9SimdForwardForLayer(s.PrimaryType) {
		return
	}

	fwdSimd := captureForwardSimd(net, input, true)
	bwdSimd := captureBackwardSimd(net, input, target, true)
	simdTol := simdParityTol(s.PrimaryType, tc)
	bwdSimdTol := simdTol * 10
	simdFwd := maxAbsDiff(fwdSC.out, fwdSimd.out)
	simdBwd := maxAbsDiff(append(bwdSC.dx, bwdSC.dw...), append(bwdSimd.dx, bwdSimd.dw...))
	if simdFwd > simdTol || simdBwd > bwdSimdTol {
		t.Fatalf("%s %s SC↔SIMD: fwd=%.4e bwd=%.4e tolFwd=%.4e tolBwd=%.4e",
			s.Name, tc.name, simdFwd, simdBwd, simdTol, bwdSimdTol)
	}
}
