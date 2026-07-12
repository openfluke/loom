package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func newDenseBwdNet(dim, batch int, multiCore bool) (*poly.VolumetricNetwork, *poly.Tensor[float32]) {
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = poly.LayerDense
	l.InputHeight = dim
	l.OutputHeight = dim
	l.Activation = poly.ActivationReLU
	l.DType = poly.DTypeFloat32
	l.TileSize = 32
	l.UseTiling = true
	l.EnableMultiCoreTiling = multiCore
	l.WeightStore = poly.NewWeightStore(dim * dim)
	for j := range l.WeightStore.Master {
		l.WeightStore.Master[j] = 0.001 * float32((j%19)+1)
	}
	input := poly.NewTensor[float32](batch, dim)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%13)
	}
	return net, input
}

func TestDenseBackwardSCMatchesMC(t *testing.T) {
	dim, batch := 128, 4
	netSC, input := newDenseBwdNet(dim, batch, false)
	netMC, _ := newDenseBwdNet(dim, batch, true)
	lSC := netSC.GetLayer(0, 0, 0, 0)
	lMC := netMC.GetLayer(0, 0, 0, 0)

	preSC, _ := poly.DenseForwardPolymorphic(lSC, input)
	preMC, _ := poly.DenseForwardPolymorphic(lMC, input)
	gradOut := poly.NewTensor[float32](batch, dim)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.02 * float32((i%7)+1)
	}

	dxSC, dwSC := poly.DenseBackwardPolymorphic(lSC, gradOut, input, preSC)
	dxMC, dwMC := poly.DenseBackwardPolymorphic(lMC, gradOut, input, preMC)

	assertMaxDiffF32(t, dxSC.Data, dxMC.Data, 1e-5, "dX SC vs MC")
	assertMaxDiffF32(t, dwSC.Data, dwMC.Data, 1e-5, "dW SC vs MC")
}

func TestDenseBackwardSimdMatchesTiled(t *testing.T) {
	if !poly.Plan9SimdEnabled() {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	dim, batch := 128, 4
	net, input := newDenseBwdNet(dim, batch, false)
	l := net.GetLayer(0, 0, 0, 0)

	net.SetSimdForward(false)
	preT, _ := poly.DenseForwardPolymorphic(l, input)
	gradOut := poly.NewTensor[float32](batch, dim)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.02 * float32((i%7)+1)
	}
	dxT, dwT := poly.DenseBackwardPolymorphic(l, gradOut, input, preT)

	net.SetSimdForward(true)
	preS, _ := poly.DenseForwardPolymorphic(l, input)
	dxS, dwS := poly.DenseBackwardPolymorphic(l, gradOut, input, preS)

	assertMaxDiffF32(t, dxT.Data, dxS.Data, 1e-5, "dX tiled vs simd")
	assertMaxDiffF32(t, dwT.Data, dwS.Data, 1e-5, "dW tiled vs simd")
}

func assertMaxDiffF32(t *testing.T, a, b []float32, tol float32, label string) {
	t.Helper()
	var max float32
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > max {
			max = d
		}
	}
	if max > tol {
		t.Fatalf("%s max diff %g > %g", label, max, tol)
	}
}
