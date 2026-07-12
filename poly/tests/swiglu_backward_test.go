package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func newSwiGLUBwdNet(inDim, interDim, seqLen int) (*poly.VolumetricNetwork, *poly.Tensor[float32]) {
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = poly.LayerSwiGLU
	l.InputHeight = inDim
	l.OutputHeight = interDim
	l.Activation = poly.ActivationSilu
	l.DType = poly.DTypeFloat32
	l.TileSize = 32
	l.UseTiling = true
	l.EnableMultiCoreTiling = true
	wSize := inDim * interDim
	total := 3*wSize + 2*interDim + inDim
	l.WeightStore = poly.NewWeightStore(total)
	for j := range l.WeightStore.Master {
		l.WeightStore.Master[j] = 0.001 * float32((j%23)+1)
	}
	input := poly.NewTensor[float32](seqLen, inDim)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%17)
	}
	return net, input
}

func TestSwiGLUBackwardSimdMatchesTiled(t *testing.T) {
	if !poly.Plan9SimdEnabled() {
		t.Skip("no Plan 9 SIMD on this GOARCH")
	}
	inDim, interDim, seqLen := 64, 32, 4
	net, input := newSwiGLUBwdNet(inDim, interDim, seqLen)
	l := net.GetLayer(0, 0, 0, 0)

	net.SetSimdForward(false)
	preT, _ := poly.SwiGLUForwardPolymorphic(l, input)
	gradOut := poly.NewTensor[float32](seqLen, inDim)
	for i := range gradOut.Data {
		gradOut.Data[i] = 0.02 * float32((i%9)+1)
	}
	dxT, dwT := poly.SwiGLUBackwardPolymorphic(l, gradOut, input, preT)

	net.SetSimdForward(true)
	preS, _ := poly.SwiGLUForwardPolymorphic(l, input)
	dxS, dwS := poly.SwiGLUBackwardPolymorphic(l, gradOut, input, preS)

	assertMaxDiffF32(t, dxT.Data, dxS.Data, 1e-4, "dX tiled vs simd")
	assertMaxDiffF32(t, dwT.Data, dwS.Data, 1e-4, "dW tiled vs simd")
}
