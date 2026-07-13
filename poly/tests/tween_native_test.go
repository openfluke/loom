package poly_test

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func buildTweenTestDenseNet(t *testing.T, dtype poly.DType, inSz, outSz int) *poly.VolumetricNetwork {
	t.Helper()
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := net.GetLayer(0, 0, 0, 0)
	l.Type = poly.LayerDense
	l.InputHeight = inSz
	l.OutputHeight = outSz
	l.Activation = poly.ActivationLinear
	l.DType = dtype
	wCount := (inSz * outSz) + outSz
	l.WeightStore = poly.NewWeightStore(wCount)
	for j := range l.WeightStore.Master {
		l.WeightStore.Master[j] = 0.01 * float32((j%7)+1)
	}
	net.UseExactDType = true
	l.WeightStore.Morph(dtype)
	return net
}

func TestTweenNativeWeightReadInt8(t *testing.T) {
	net := buildTweenTestDenseNet(t, poly.DTypeInt8, 4, 2)
	w := poly.TweenWeightF32(&net.Layers[0], 0)
	if w == 0 && poly.TweenWeightCount(&net.Layers[0]) > 0 {
		t.Fatalf("expected non-zero native weight read, got %v", w)
	}
}

func TestTweenNativeLayerwiseGapsDense(t *testing.T) {
	net := buildTweenTestDenseNet(t, poly.DTypeInt8, 4, 2)
	cfg := poly.DefaultTweenConfig()
	cfg.UseChainRule = false
	state := poly.NewTweenState[float32](net, cfg)
	input := poly.NewTensor[float32](1, 4)
	for i := range input.Data {
		input.Data[i] = float32(i+1) * 0.1
	}
	target := poly.NewTensor[float32](1, 2)
	target.Data[0] = 1
	target.Data[1] = 0

	out := poly.TweenForward(net, state, input)
	poly.TweenBackward(net, state, target)
	state.CalculateLinkBudgets()
	poly.ApplyTweenGaps(net, state, 0.01)

	if out == nil || len(out.Data) == 0 {
		t.Fatal("empty forward")
	}
}

func TestTweenNativeChainRuleFloat32SimdParity(t *testing.T) {
	if !poly.Plan9SimdForwardForLayer(poly.LayerDense) {
		t.Skip("no Plan 9 SIMD")
	}
	netSC := buildTweenTestDenseNet(t, poly.DTypeFloat32, 16, 8)
	netSimd := buildTweenTestDenseNet(t, poly.DTypeFloat32, 16, 8)
	netSC.SetSimdForward(false)
	netSimd.SetSimdForward(true)

	cfg := poly.DefaultTweenConfig()
	cfg.UseChainRule = true
	input := poly.NewTensor[float32](1, 16)
	for i := range input.Data {
		input.Data[i] = float32(i) * 0.05
	}
	target := poly.NewTensor[float32](1, 8)
	for i := range target.Data {
		target.Data[i] = float32(i%3) * 0.1
	}

	stateSC := poly.NewTweenState[float32](netSC, cfg)
	stateSimd := poly.NewTweenState[float32](netSimd, cfg)
	outSC := poly.TweenForward(netSC, stateSC, input)
	outSimd := poly.TweenForward(netSimd, stateSimd, input)
	if d := maxAbsDiffF32(outSC.Data, outSimd.Data); d > 1e-4 {
		t.Fatalf("TweenForward SC vs SIMD diff %g", d)
	}

	poly.TweenBackward(netSC, stateSC, target)
	poly.TweenBackward(netSimd, stateSimd, target)
	if stateSC.Gradients[0] == nil || stateSimd.Gradients[0] == nil {
		t.Fatal("missing input gradients")
	}
	if d := maxAbsDiffF32(stateSC.Gradients[0].Data, stateSimd.Gradients[0].Data); d > 1e-3 {
		t.Fatalf("TweenBackward grad diff %g", d)
	}
}

func maxAbsDiffF32(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var m float64
	for i := 0; i < n; i++ {
		v := float64(a[i] - b[i])
		if v < 0 {
			v = -v
		}
		if v > m {
			m = v
		}
	}
	return m
}
