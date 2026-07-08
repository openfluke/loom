package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

// SmolLM2-135M decoder MLP dims. seqLen=1 is the decode (single-token) case.
func benchSwiGLULayer(inDim, interDim int) (*poly.VolumetricNetwork, *poly.VolumetricLayer) {
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = poly.LayerSwiGLU
	l.Activation = poly.ActivationSilu
	l.InputHeight = inDim
	l.OutputHeight = interDim
	l.DType = poly.DTypeFloat32
	wSize := inDim * interDim
	total := 3*wSize + 2*interDim + inDim
	l.WeightStore = poly.NewWeightStore(total)
	for i := range l.WeightStore.Master {
		l.WeightStore.Master[i] = 0.001 * float32((i%17)+1)
	}
	l.UseTiling = true
	l.EnableMultiCoreTiling = true
	l.EnsureRuntimeTileSizes()
	return net, l
}

func benchDecodeSwiGLU(b *testing.B, simdOn bool) {
	const inDim, interDim = 576, 1536
	net, l := benchSwiGLULayer(inDim, interDim)
	net.SetSimdForwardRecursive(simdOn)
	net.RefreshRuntimeTileSizes()
	input := poly.NewTensor[float32](1, inDim) // seqLen=1 → decode
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%11)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = poly.SwiGLUForwardPolymorphic(l, input)
	}
}

func BenchmarkDecodeSwiGLUScalar(b *testing.B) { benchDecodeSwiGLU(b, false) }
func BenchmarkDecodeSwiGLUSimd(b *testing.B)   { benchDecodeSwiGLU(b, true) }

// batch=32 mimics prefill of a 32-token prompt.
func benchPrefillSwiGLU(b *testing.B, simdOn bool) {
	const inDim, interDim, seqLen = 576, 1536, 32
	net, l := benchSwiGLULayer(inDim, interDim)
	net.SetSimdForwardRecursive(simdOn)
	net.RefreshRuntimeTileSizes()
	input := poly.NewTensor[float32](seqLen, inDim)
	for i := range input.Data {
		input.Data[i] = 0.01 * float32(i%11)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = poly.SwiGLUForwardPolymorphic(l, input)
	}
}

func BenchmarkPrefillSwiGLUScalar(b *testing.B) { benchPrefillSwiGLU(b, false) }
func BenchmarkPrefillSwiGLUSimd(b *testing.B)   { benchPrefillSwiGLU(b, true) }
