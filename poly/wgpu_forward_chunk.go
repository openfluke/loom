package poly

import (
	"encoding/binary"
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// ForwardSampleGreedyChunkWGPU records n decode steps into one GPU frame and maps back
// n uint32 token ids. seedToken is the first input token (usually the prefill ArgMax result).
// Works for Q4 / INT8 / FP32 / BitNet — same Dispatch* family as ForwardTokenIDsWGPU.
func (t *Transformer[T]) ForwardSampleGreedyChunkWGPU(seedToken uint32, n int) ([]uint32, error) {
	if n <= 0 {
		return nil, nil
	}
	if t.Network == nil || t.Network.GPUContext == nil {
		return nil, fmt.Errorf("GPU context not initialized")
	}
	hasQ4LM := t.Network.GPULMHeadQ4Packed != nil && t.Network.GPULMHeadQ4Scales != nil
	hasTernaryLM := t.Network.GPULMHeadTernaryPacked != nil
	if t.Network.GPUEmbeddings == nil || (t.Network.GPULMHead == nil && !hasQ4LM && !hasTernaryLM) {
		return nil, fmt.Errorf("GPU embeddings/LM head required for chunked greedy decode")
	}
	ctx := t.Network.GPUContext

	histBytes := uint64(n * 4)
	if histBytes < 64 {
		histBytes = 64
	}
	histBuf := ctx.GetActivationBuffer("greedy_hist", histBytes, wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	stepBuf := ctx.GetActivationBuffer("greedy_step", 64, wgpu.BufferUsageStorage)
	liveTok := ctx.GetActivationBuffer("decode_token", 64, wgpu.BufferUsageStorage)
	staging := ctx.GetActivationBuffer("greedy_hist_stage", histBytes, wgpu.BufferUsageMapRead)

	// step=[pos, outCount]; pos must match current KV cache length after prefill.
	pos := 0
	if len(t.Network.Layers) > 1 {
		pos = t.Network.Layers[1].KVOffset
	}
	ctx.Queue.WriteBuffer(stepBuf, 0, packU32LE(uint32(pos), 0))
	ctx.Queue.WriteBuffer(liveTok, 0, packU32LE(seedToken))

	if err := ctx.BeginFrame(); err != nil {
		return nil, err
	}
	t.gpuChunkRecording = true
	t.gpuReturnGreedyToken = true
	t.gpuUseDecodeTokenBuf = true
	t.gpuChunkHistCount = 0
	defer func() {
		t.gpuChunkRecording = false
		t.gpuReturnGreedyToken = false
		t.gpuUseDecodeTokenBuf = false
		t.gpuChunkHistCount = 0
	}()

	for i := 0; i < n; i++ {
		if _, err := t.ForwardTokenIDsWGPU(nil, nil, true, true); err != nil {
			ctx.FlushFrame()
			return nil, fmt.Errorf("chunk decode step %d: %w", i, err)
		}
	}

	ctx.EndComputePass()
	ctx.ActiveEncoder.CopyBufferToBuffer(histBuf, 0, staging, 0, uint64(n*4))
	ctx.FlushFrame()

	// Host KV offsets catch up to GPU step[0] after the chunk.
	for b := 0; b < len(t.Network.Layers)/4; b++ {
		t.Network.Layers[b*4+1].KVOffset = pos + n
	}

	raw, err := ctx.pollMapRead(staging, uint64(n*4))
	if err != nil {
		return nil, err
	}
	out := make([]uint32, n)
	for i := 0; i < n; i++ {
		out[i] = binary.LittleEndian.Uint32(raw[i*4 : i*4+4])
	}
	if n > 0 {
		t.lastGPUSampledToken = out[n-1]
	}
	return out, nil
}

func packU32LE(vals ...uint32) []byte {
	b := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:], v)
	}
	return b
}
