package poly_test

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	. "github.com/openfluke/loom/poly"
	"github.com/openfluke/webgpu/wgpu"
)

func smolLM2EntityPath(t *testing.T) string {
	t.Helper()
	candidates := []string{
		filepath.Join("..", "..", "lucy", "lucy_entities", "HuggingFaceTB--SmolLM2-135M-Instruct.entity"),
		filepath.Join("..", "lucy", "lucy_entities", "HuggingFaceTB--SmolLM2-135M-Instruct.entity"),
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	t.Skip("SmolLM2 .entity not found (run from loom module with lucy_entities present)")
	return ""
}

func setupEntityTransformer(t *testing.T, path string) *Transformer[float32] {
	t.Helper()
	et, err := LoadEntityTransformer(path)
	if err != nil {
		t.Fatalf("LoadEntityTransformer: %v", err)
	}
	PrepareEntityTransformerInference(et)
	tr := BuildTransformerFromEntity[float32](et, Template{})
	tr.SetRMSNormEps(et.Dims.RMSNormEps)
	for i := range tr.Network.Layers {
		tr.Network.Layers[i].MaxSeqLen = 512
	}
	return tr
}

func cpuLogitsForToken(t *testing.T, tr *Transformer[float32], tok uint32) []float32 {
	t.Helper()
	tr.Network.UseGPU = false
	tr.SyncInferenceCPU()
	tr.Reset()
	input := tr.TokensToTensor([]uint32{tok})
	hidden := tr.ForwardFull(input)
	h := hidden.Data
	if len(h) < tr.HiddenSize {
		t.Fatal("hidden too short")
	}
	row := h[len(h)-tr.HiddenSize:]
	return tr.ApplyLMHead(row)
}

func gpuLogitsForToken(t *testing.T, tr *Transformer[float32], tok uint32, multiCore bool) []float32 {
	t.Helper()
	tr.Network.UseGPU = true
	tr.EnableTiling(-1)
	tr.Network.EnableMultiCoreTiling = multiCore
	if err := tr.Network.InitWGPU(); err != nil {
		t.Skipf("GPU unavailable: %v", err)
	}
	gpuDType := EntityGPUWeightDType(tr.Network.Layers[1].DType, true)
	for i := range tr.Network.Layers {
		l := &tr.Network.Layers[i]
		if l.Type == LayerRMSNorm {
			l.DType = DTypeFloat32
		} else {
			l.DType = gpuDType
		}
		if err := l.SyncToGPU(); err != nil {
			t.Fatalf("SyncToGPU layer %d: %v", i, err)
		}
	}
	if err := tr.SyncGlobalWeightsToGPUSequential(); err != nil {
		t.Fatalf("SyncGlobalWeightsToGPUSequential: %v", err)
	}
	if tr.Network.GPULMHead == nil {
		t.Fatalf("GPULMHead nil after global sync (GPUEmbeddings=%v lmHeadLen=%d)", tr.Network.GPUEmbeddings != nil, len(tr.LMHead))
	}
	tr.Reset()
	logitTensor, err := tr.ForwardTokenIDsWGPU([]uint32{tok}, nil, true, true)
	if err != nil {
		t.Fatalf("ForwardTokenIDsWGPU: %v", err)
	}
	if logitTensor == nil || len(logitTensor.Data) != tr.VocabSize {
		t.Fatalf("logits len = %d, want vocab %d", len(logitTensor.Data), tr.VocabSize)
	}
	out := make([]float32, tr.VocabSize)
	for i, v := range logitTensor.Data {
		out[i] = float32(v)
	}
	return out
}

// TestSmolLM2EntityCPUGPULogitParity compares last-token logits for one decode step.
func TestSmolLM2EntityCPUGPULogitParity(t *testing.T) {
	path := smolLM2EntityPath(t)
	tok := uint32(100)

	cpuTr := setupEntityTransformer(t, path)
	cpuLogits := cpuLogitsForToken(t, cpuTr, tok)

	for _, tc := range []struct {
		name string
		mc   bool
	}{
		{"GPU_SC", false},
		{"GPU_MC", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gpuTr := setupEntityTransformer(t, path)
			gpuLogits := gpuLogitsForToken(t, gpuTr, tok, tc.mc)

			var maxErr, cpuMax, gpuMax float64
			argmaxCPU, argmaxGPU := 0, 0
			for i := range cpuLogits {
				d := math.Abs(float64(cpuLogits[i] - gpuLogits[i]))
				if d > maxErr {
					maxErr = d
				}
				ca := math.Abs(float64(cpuLogits[i]))
				if ca > cpuMax {
					cpuMax = ca
					argmaxCPU = i
				}
				ga := math.Abs(float64(gpuLogits[i]))
				if ga > gpuMax {
					gpuMax = ga
					argmaxGPU = i
				}
			}
			t.Logf("max abs diff: %.4e | argmax cpu=%d (%.4f) gpu=%d (%.4f)", maxErr, argmaxCPU, cpuLogits[argmaxCPU], argmaxGPU, gpuLogits[argmaxGPU])
			if maxErr > 1.0 {
				t.Fatalf("CPU/GPU logit parity failed: max err %.4e", maxErr)
			}
			if argmaxCPU != argmaxGPU {
				t.Fatalf("argmax token mismatch: cpu=%d gpu=%d", argmaxCPU, argmaxGPU)
			}
		})
	}
}

// TestQ4GPUDenseParity compares CPU dequant matmul vs DispatchDenseQ4 (SC + MC tile sizes).
func TestQ4GPUDenseParity(t *testing.T) {
	const (
		batchSize  = 2
		inputSize  = 128 // multiple of 32 for Q4_0 blocks
		outputSize = 64
	)

	net := NewVolumetricNetwork(1, 1, 1, 1)
	net.Layers[0] = VolumetricLayer{
		Network:      net,
		Type:         LayerSwiGLU,
		DType:        DTypeInt4,
		InputHeight:  inputSize,
		OutputHeight: outputSize,
		Activation:   ActivationSilu,
		WeightStore:  NewWeightStore(0),
		UseTiling:    true,
	}
	l := &net.Layers[0]

	weights := make([]float32, inputSize*outputSize)
	for i := range weights {
		weights[i] = float32(math.Sin(float64(i)*0.17)) * 0.05
	}
	scales, packed := PackQ4_0GPU(weights)
	l.WeightStore.SetQ4_0Component(DType(100), scales, packed)

	input := NewTensor[float32](batchSize, inputSize)
	for i := range input.Data {
		input.Data[i] = float32(math.Cos(float64(i)*0.09)) * 0.1
	}

	// CPU reference: dequant baked Q4 weights and matmul
	dequant := DequantizeQ4_0GPUPacked(scales, packed)
	cpuOut := make([]float32, batchSize*outputSize)
	for b := 0; b < batchSize; b++ {
		inRow := input.Data[b*inputSize : (b+1)*inputSize]
		outRow := cpuOut[b*outputSize : (b+1)*outputSize]
		for o := 0; o < outputSize; o++ {
			var sum float32
			wRow := dequant[o*inputSize : (o+1)*inputSize]
			for i := 0; i < inputSize; i++ {
				sum += inRow[i] * wRow[i]
			}
			outRow[o] = sum
		}
	}

	if err := net.InitWGPU(); err != nil {
		t.Skipf("GPU unavailable: %v", err)
	}
	ctx := net.GPUContext
	if err := l.SyncToGPU(); err != nil {
		t.Fatalf("SyncToGPU: %v", err)
	}
	sBuf := l.WeightStore.GPUScales[DType(100)]
	wBuf, _ := l.WeightStore.GPUWeights[DType(100)].(*wgpu.Buffer)
	if sBuf == nil || wBuf == nil {
		t.Fatal("Q4 GPU buffers missing after SyncToGPU")
	}

	inBuf := ctx.GetActivationBuffer("q4_parity_in", uint64(len(input.Data)*4), wgpu.BufferUsageStorage)
	ctx.Queue.WriteBuffer(inBuf, 0, wgpu.ToBytes(input.Data))

	for _, tc := range []struct {
		name     string
		tileSize int
		mc       bool
	}{
		{"SC", l.GetGPUSCTileSize(DTypeInt4), false},
		{"MC", l.GetGPUMCTileSize(DTypeInt4), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			net.EnableMultiCoreTiling = tc.mc
			outBuf := ctx.GetActivationBuffer("q4_parity_out_"+tc.name, uint64(len(cpuOut)*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
			if err := ctx.DispatchDenseQ4(batchSize, inputSize, outputSize, inBuf, sBuf, wBuf, outBuf, tc.tileSize); err != nil {
				t.Fatalf("DispatchDenseQ4: %v", err)
			}
			ctx.Device.Poll(true, nil)
			gpuData, err := ctx.ReadBuffer(outBuf)
			if err != nil {
				t.Fatalf("ReadBuffer: %v", err)
			}
			var maxErr float64
			for i := range cpuOut {
				d := math.Abs(float64(cpuOut[i] - gpuData[i]))
				if d > maxErr {
					maxErr = d
				}
			}
			t.Logf("max abs diff vs CPU: %.6e (tile=%d mc=%v)", maxErr, tc.tileSize, tc.mc)
			if maxErr > 0.05 {
				t.Fatalf("GPU Q4 parity failed: max err %.6e", maxErr)
			}
		})
	}
}
