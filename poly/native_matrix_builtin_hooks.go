package poly

import (
	"fmt"
	"math/rand"
)

const (
	builtinMatrixLayers = 4
	builtinMatrixDModel = 32
)

// BuiltinNativeMatrixHooks provides a self-contained hook set so RunNativeLayerMatrix
// can be invoked without an external harness (e.g. from C-ABI or tests).
func BuiltinNativeMatrixHooks() NativeMatrixHooks {
	return NativeMatrixHooks{
		BuildProof:             builtinNativeMatrixBuildProof,
		BuildSamples:           builtinNativeMatrixBuildSamples,
		ResolveBenchProfile:    builtinNativeMatrixResolveBenchProfile,
		GrowBenchShape:         builtinNativeMatrixGrowBenchShape,
		BuildBenchmarkInput:    builtinNativeMatrixBuildBenchmarkInput,
		BuildDefaultNetwork:    builtinNativeMatrixBuildDefaultNetwork,
		BuildNetworkForShape:   builtinNativeMatrixBuildNetworkForShape,
		SupportsNativeCPUExec:  func(DType) bool { return true },
		SupportsNativeCPUTrain: func(DType) bool { return true },
		SupportsNativeGPUExec:  func(DType) bool { return false },
		SupportsNativeGPUTrain: func(DType) bool { return false },
		UsesPackedGPU:          func(DType) bool { return false },
		CapabilityNotes:        nil,
	}
}

// RunNativeLayerMatrixBuiltin runs the native layer matrix using BuiltinNativeMatrixHooks.
func RunNativeLayerMatrixBuiltin(cfg NativeMatrixConfig) error {
	return RunNativeLayerMatrix(cfg, BuiltinNativeMatrixHooks())
}

func builtinNativeMatrixBuildProof() (NativeMatrixBuildProof, error) {
	n := BuildSequentialNetwork(builtinMatrixLayers, builtinMatrixDModel, ActivationReLU, DTypeFloat32)
	return NativeMatrixBuildProof{
		Layers:         len(n.Layers),
		LayersPerCell:  1,
		SequentialHits: 0,
	}, nil
}

func builtinNativeMatrixBuildSamples(trainPerClass, evalPerClass int) NativeMatrixSamples {
	batch := trainPerClass
	if batch < 2 {
		batch = 2
	}
	d := builtinMatrixDModel
	in := make([]float32, batch*d)
	tgt := make([]float32, batch*d)
	for i := range in {
		in[i] = float32(rand.Float64()*0.2 - 0.1)
		tgt[i] = in[i] * 0.95
	}
	trainIn := NewTensorFromSlice(in, batch, d, 1)
	trainTgt := NewTensorFromSlice(tgt, batch, d, 1)

	ev := evalPerClass
	if ev < 1 {
		ev = 1
	}
	evalInputs := make([]*Tensor[float32], ev)
	evalTgts := make([]*Tensor[float32], ev)
	labels := make([]int, ev)
	for i := 0; i < ev; i++ {
		ei := make([]float32, d)
		et := make([]float32, d)
		for j := range ei {
			ei[j] = 0.02 * float32(j+1)
			et[j] = ei[j]
		}
		evalInputs[i] = NewTensorFromSlice(ei, 1, d, 1)
		evalTgts[i] = NewTensorFromSlice(et, 1, d, 1)
		labels[i] = 0
	}
	return NativeMatrixSamples{
		Train:       TrainingBatch[float32]{Input: trainIn, Target: trainTgt},
		TrainCount:  batch,
		ParityIn:    trainIn,
		ParityTgt:   trainTgt,
		EvalInputs:  evalInputs,
		EvalTgts:    evalTgts,
		EvalLabels:  labels,
		EvalCount:   ev,
	}
}

func builtinNativeMatrixResolveBenchProfile(spec string) (string, any, string, error) {
	shape := map[string]int{"batch": 8, "dmodel": builtinMatrixDModel}
	summary := fmt.Sprintf("batch=%d dmodel=%d", shape["batch"], shape["dmodel"])
	return spec, shape, summary, nil
}

func builtinNativeMatrixGrowBenchShape(shape any, factor float64, gpuStress bool) (any, string, error) {
	_ = gpuStress
	m, ok := shape.(map[string]int)
	if !ok {
		return shape, "unchanged", nil
	}
	b := int(float64(m["batch"]) * factor)
	if b < 1 {
		b = 1
	}
	if b > 256 {
		b = 256
	}
	out := map[string]int{"batch": b, "dmodel": m["dmodel"]}
	return out, fmt.Sprintf("batch=%d", b), nil
}

func builtinNativeMatrixBuildBenchmarkInput(shape any) (*Tensor[float32], error) {
	batch := 8
	d := builtinMatrixDModel
	if m, ok := shape.(map[string]int); ok {
		if v, ok := m["batch"]; ok && v > 0 {
			batch = v
		}
		if v, ok := m["dmodel"]; ok && v > 0 {
			d = v
		}
	}
	data := make([]float32, batch*d)
	for i := range data {
		data[i] = float32(rand.Float64()*0.2 - 0.1)
	}
	return NewTensorFromSlice(data, batch, d, 1), nil
}

func builtinNativeMatrixBuildDefaultNetwork(dtype DType) (*VolumetricNetwork, error) {
	n := BuildSequentialNetwork(builtinMatrixLayers, builtinMatrixDModel, ActivationReLU, dtype)
	return n, nil
}

func builtinNativeMatrixBuildNetworkForShape(dtype DType, shape any) (*VolumetricNetwork, error) {
	_ = shape
	return builtinNativeMatrixBuildDefaultNetwork(dtype)
}
