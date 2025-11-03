package pods

import (
	"errors"
	"math"
)

type SoftmaxIn struct{ Logits []float32 }
type SoftmaxOut struct{ Probs []float32 }

type SoftmaxPod struct{}

func (SoftmaxPod) Name() string { return "ml/softmax" }

func (SoftmaxPod) Run(x *ExecContext, in any) (any, error) {
	args, ok := in.(SoftmaxIn)
	if !ok {
		return nil, errors.New("SoftmaxIn expected")
	}
	n := len(args.Logits)
	out := make([]float32, n)
	// TODO GPU: x.GPU.DispatchSoftmaxF32(n, ptr)
	mx := args.Logits[0]
	for _, v := range args.Logits[1:] {
		if v > mx {
			mx = v
		}
	}
	var sum float64
	for i, v := range args.Logits {
		e := math.Exp(float64(v - mx))
		out[i] = float32(e)
		sum += e
	}
	inv := float32(1.0 / sum)
	for i := range out {
		out[i] *= inv
	}
	return SoftmaxOut{Probs: out}, nil
}

type LayerNormIn struct {
	X   []float32
	Eps float32
}
type LayerNormOut struct{ Y []float32 } // per-vector LN; extend with shape later

type LayerNormPod struct{}

func (LayerNormPod) Name() string { return "ml/layernorm" }

func (LayerNormPod) Run(_ *ExecContext, in any) (any, error) {
	args, ok := in.(LayerNormIn)
	if !ok {
		return nil, errors.New("LayerNormIn expected")
	}
	n := len(args.X)
	var mean, m2 float64
	for _, v := range args.X {
		mean += float64(v)
	}
	mean /= float64(n)
	for _, v := range args.X {
		d := float64(v) - mean
		m2 += d * d
	}
	varY := float32(m2 / float64(n))
	den := float32(1.0 / math.Sqrt(float64(varY+args.Eps)))
	Y := make([]float32, n)
	for i, v := range args.X {
		Y[i] = (v - float32(mean)) * den
	}
	return LayerNormOut{Y: Y}, nil
}
