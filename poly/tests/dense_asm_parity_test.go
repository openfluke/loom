package poly_test

import (
	"testing"

	. "github.com/openfluke/loom/poly"
	"github.com/openfluke/loom/poly/asm"
)

func TestDenseAsmForwardMatchesGoTiled(t *testing.T) {
	if !asm.Enabled() {
		t.Skip("no assembly on this platform")
	}

	const batch, inDim, outDim = 2, 16, 8
	input := NewTensorFromSlice(makeF32Input(batch, inDim), batch, inDim)
	weights := makeF32Weights(outDim, inDim)

	layer := &VolumetricLayer{
		InputHeight:  inDim,
		OutputHeight: outDim,
		DType:        DTypeFloat32,
		UseTiling:    true,
		WeightStore:  &WeightStore{Master: weights},
	}
	layer.TileSize = 8
	layer.CPUTileSizes = map[DType]int{DTypeFloat32: 8}

	layer.UseAsmForward = false
	layer.EnableMultiCoreTiling = false
	goPre, _ := DenseForwardTiled(layer, input)

	layer.UseAsmForward = true
	asmPre, _ := DenseForwardPolymorphic(layer, input)

	for i := range goPre.Data {
		if goPre.Data[i] != asmPre.Data[i] {
			t.Fatalf("sc index %d: go %v asm %v", i, goPre.Data[i], asmPre.Data[i])
		}
	}

	layer.EnableMultiCoreTiling = true
	layer.UseAsmForward = false
	goPreMC, _ := DenseForwardTiled(layer, input)

	layer.UseAsmForward = true
	asmPreMC, _ := DenseForwardPolymorphic(layer, input)

	for i := range goPreMC.Data {
		if goPreMC.Data[i] != asmPreMC.Data[i] {
			t.Fatalf("mc index %d: go %v asm %v", i, goPreMC.Data[i], asmPreMC.Data[i])
		}
	}
}

func makeF32Input(batch, inDim int) []float32 {
	out := make([]float32, batch*inDim)
	for i := range out {
		out[i] = float32(i%13)*0.1 - 0.6
	}
	return out
}

func makeF32Weights(outDim, inDim int) []float32 {
	w := make([]float32, outDim*inDim)
	for o := 0; o < outDim; o++ {
		for i := 0; i < inDim; i++ {
			if o == i%outDim {
				w[o*inDim+i] = 1
			}
		}
	}
	return w
}
