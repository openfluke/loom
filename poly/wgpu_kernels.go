package poly

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// WGPUDenseParams matches the WGSL struct
type WGPUDenseParams struct {
	BatchSize  uint32
	InputSize  uint32
	OutputSize uint32
	TileSize   uint32
}

// WGPUMHAParams matches the attention WGSL struct
type WGPUMHAParams struct {
	NumHeads   uint32
	NumKVHeads uint32
	HeadDim    uint32
	SeqLen     uint32
	KVOffset   uint32
	MaxSeqLen  uint32
	TileSize   uint32
	Padding    uint32
}

func (c *WGPUContext) CreateComputePipeline(shaderSource string) (*wgpu.ComputePipeline, error) {
	if p, ok := c.PipelineCache[shaderSource]; ok {
		return p, nil
	}

	shader, err := c.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shaderSource},
	})
	if err != nil {
		return nil, err
	}
	defer shader.Release()

	pipeline, err := c.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     shader,
			EntryPoint: "main",
		},
	})
	if err == nil {
		c.PipelineCache[shaderSource] = pipeline
	}
	return pipeline, err
}

// ctxEncoder returns the active shared encoder if one is open (BeginFrame was called),
// otherwise creates a new one-shot encoder.
// owned=true means the caller is responsible for Finish()+Submit().
func ctxEncoder(c *WGPUContext) (*wgpu.CommandEncoder, bool, error) {
	if c.ActiveEncoder != nil {
		return c.ActiveEncoder, false, nil
	}
	enc, err := c.Device.CreateCommandEncoder(nil)
	return enc, true, err
}

// ctxSubmit finishes and submits enc only when it is owned (not the shared frame encoder).
func ctxSubmit(c *WGPUContext, enc *wgpu.CommandEncoder, owned bool) {
	if !owned {
		return
	}
	cmd, _ := enc.Finish(nil)
	c.Queue.Submit(cmd)
}

// DispatchDense dispatches a tiled dense matrix-multiply kernel.
func (c *WGPUContext) DispatchDense(
	batchSize, inputSize, outputSize int,
	inputBuf, weightBuf, outputBuf *wgpu.Buffer,
	tileSize int,
) error {
	shaderSrc := ShaderTiledDenseN(tileSize)
	pipeline, err := c.CreateComputePipeline(shaderSrc)
	if err != nil {
		return err
	}

	params := WGPUDenseParams{
		BatchSize:  uint32(batchSize),
		InputSize:  uint32(inputSize),
		OutputSize: uint32(outputSize),
		TileSize:   uint32(tileSize),
	}
	paramBuf, _ := c.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Contents: wgpu.ToBytes([]WGPUDenseParams{params}),
		Usage:    wgpu.BufferUsageUniform,
	})
	// Queue for post-flush destruction when batching, or destroy now if standalone.
	defer c.deferOrDestroy(paramBuf)

	bindGroup, _ := c.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: paramBuf, Size: paramBuf.GetSize()},
			{Binding: 1, Buffer: inputBuf, Size: inputBuf.GetSize()},
			{Binding: 2, Buffer: weightBuf, Size: weightBuf.GetSize()},
			{Binding: 3, Buffer: outputBuf, Size: outputBuf.GetSize()},
		},
	})
	defer bindGroup.Release()

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups((uint32(outputSize)+uint32(tileSize)-1)/uint32(tileSize), uint32(batchSize), 1)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

// DispatchMHA dispatches the tiled multi-head attention kernel.
func (c *WGPUContext) DispatchMHA(
	numHeads, numKVHeads, headDim, seqLen, kvOffset, maxSeqLen int,
	qBuf, kBuf, vBuf, oBuf *wgpu.Buffer,
	tileSize int,
) error {
	shaderSrc := ShaderTiledMHAN(tileSize, headDim)
	pipeline, err := c.CreateComputePipeline(shaderSrc)
	if err != nil {
		return err
	}

	params := WGPUMHAParams{
		NumHeads:   uint32(numHeads),
		NumKVHeads: uint32(numKVHeads),
		HeadDim:    uint32(headDim),
		SeqLen:     uint32(seqLen),
		KVOffset:   uint32(kvOffset),
		MaxSeqLen:  uint32(maxSeqLen),
		TileSize:   uint32(tileSize),
	}
	pBuf, _ := c.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Contents: wgpu.ToBytes([]WGPUMHAParams{params}),
		Usage:    wgpu.BufferUsageUniform,
	})
	defer c.deferOrDestroy(pBuf)

	bindGroup, _ := c.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: qBuf, Size: qBuf.GetSize()},
			{Binding: 2, Buffer: kBuf, Size: kBuf.GetSize()},
			{Binding: 3, Buffer: vBuf, Size: vBuf.GetSize()},
			{Binding: 4, Buffer: oBuf, Size: oBuf.GetSize()},
		},
	})
	defer bindGroup.Release()

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups(uint32(numHeads), uint32(seqLen), 1)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

// DispatchSwiGLU dispatches the tiled SwiGLU MLP kernel.
func (c *WGPUContext) DispatchSwiGLU(
	batchSize, inputSize, outputSize int,
	inputBuf, gateBuf, upBuf, outputBuf *wgpu.Buffer,
	tileSize int,
) error {
	shaderSrc := ShaderTiledSwiGLUN(tileSize)
	pipeline, err := c.CreateComputePipeline(shaderSrc)
	if err != nil {
		return err
	}

	params := WGPUDenseParams{
		BatchSize:  uint32(batchSize),
		InputSize:  uint32(inputSize),
		OutputSize: uint32(outputSize),
		TileSize:   uint32(tileSize),
	}
	pBuf, _ := c.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Contents: wgpu.ToBytes([]WGPUDenseParams{params}),
		Usage:    wgpu.BufferUsageUniform,
	})
	defer c.deferOrDestroy(pBuf)

	bindGroup, _ := c.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: inputBuf, Size: inputBuf.GetSize()},
			{Binding: 2, Buffer: gateBuf, Size: gateBuf.GetSize()},
			{Binding: 3, Buffer: upBuf, Size: upBuf.GetSize()},
			{Binding: 4, Buffer: outputBuf, Size: outputBuf.GetSize()},
		},
	})
	defer bindGroup.Release()

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups((uint32(outputSize)+uint32(tileSize)-1)/uint32(tileSize), uint32(batchSize), 1)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

// WGPURMSNormParams matches the WGSL struct
type WGPURMSNormParams struct {
	Size    uint32
	Epsilon float32
}

// DispatchRMSNorm dispatches the RMSNorm kernel.
func (c *WGPUContext) DispatchRMSNorm(
	batchSize, size int, epsilon float32,
	inputBuf, weightBuf, outputBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderRMSNorm)
	if err != nil {
		return err
	}

	params := WGPURMSNormParams{Size: uint32(size), Epsilon: epsilon}
	pBuf, _ := c.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Contents: wgpu.ToBytes([]WGPURMSNormParams{params}),
		Usage:    wgpu.BufferUsageUniform,
	})
	defer c.deferOrDestroy(pBuf)

	bindGroup, _ := c.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: inputBuf, Size: inputBuf.GetSize()},
			{Binding: 2, Buffer: weightBuf, Size: weightBuf.GetSize()},
			{Binding: 3, Buffer: outputBuf, Size: outputBuf.GetSize()},
		},
	})
	defer bindGroup.Release()

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups(uint32(batchSize), 1, 1)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

// GetActivationBuffer retrieves or creates a persistent activation buffer.
func (c *WGPUContext) GetActivationBuffer(name string, size uint64, usage wgpu.BufferUsage) *wgpu.Buffer {
	if size%4 != 0 {
		size = (size + 3) &^ 3
	}

	if buf, ok := c.ActivationPool[name]; ok && buf != nil {
		if buf.GetSize() >= size && (buf.GetUsage()&usage == usage) {
			return buf
		}
		buf.Destroy()
	}
	actualUsage := usage
	if (usage & wgpu.BufferUsageMapRead) != 0 {
		actualUsage |= wgpu.BufferUsageCopyDst
	} else if (usage & wgpu.BufferUsageMapWrite) != 0 {
		actualUsage |= wgpu.BufferUsageCopySrc
	} else {
		actualUsage |= wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst
	}

	buf, err := c.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: name,
		Size:  size,
		Usage: actualUsage,
	})
	if err != nil {
		fmt.Printf("⚠️  Failed to allocate buffer %s (size %d): %v\n", name, size, err)
		return nil
	}
	c.ActivationPool[name] = buf
	return buf
}

type WGPUKVParams struct {
	Offset     uint32
	HeadDim    uint32
	MaxSeqLen  uint32
	NumKVHeads uint32
	NumTokens  uint32
}

func (c *WGPUContext) DispatchKVUpdate(
	offset, headDim, maxSeqLen, numKVHeads, numTokens int,
	kCache, vCache, newK, newV *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderKVUpdate)
	if err != nil {
		return err
	}

	p := WGPUKVParams{
		Offset:     uint32(offset),
		HeadDim:    uint32(headDim),
		MaxSeqLen:  uint32(maxSeqLen),
		NumKVHeads: uint32(numKVHeads),
		NumTokens:  uint32(numTokens),
	}
	pBuf, _ := c.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Contents: wgpu.ToBytes([]WGPUKVParams{p}),
		Usage:    wgpu.BufferUsageUniform,
	})
	defer c.deferOrDestroy(pBuf)

	bindGroup, _ := c.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: kCache, Size: kCache.GetSize()},
			{Binding: 1, Buffer: vCache, Size: vCache.GetSize()},
			{Binding: 2, Buffer: newK, Size: newK.GetSize()},
			{Binding: 3, Buffer: newV, Size: newV.GetSize()},
			{Binding: 4, Buffer: pBuf, Size: pBuf.GetSize()},
		},
	})
	defer bindGroup.Release()

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups((uint32(numKVHeads)*uint32(headDim)*uint32(numTokens)+63)/64, 1, 1)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

// DispatchResidual dispatches the element-wise addition kernel.
func (c *WGPUContext) DispatchResidual(
	size int,
	inputBuf, residualBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderResidualAdd)
	if err != nil {
		return err
	}

	pBuf, _ := c.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Contents: wgpu.ToBytes([]uint32{uint32(size)}),
		Usage:    wgpu.BufferUsageUniform,
	})
	defer c.deferOrDestroy(pBuf)

	bindGroup, _ := c.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: inputBuf, Size: inputBuf.GetSize()},
			{Binding: 2, Buffer: residualBuf, Size: residualBuf.GetSize()},
		},
	})
	defer bindGroup.Release()

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups((uint32(size)+63)/64, 1, 1)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

type WGPURoPEParams struct {
	SeqLen   uint32
	HeadDim  uint32
	NumHeads uint32
	Offset   uint32
	Theta    float32
}

func (c *WGPUContext) DispatchRoPE(
	seqLen, headDim, numHeads, offset int, theta float32,
	targetBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderRoPE)
	if err != nil {
		return err
	}

	p := WGPURoPEParams{
		SeqLen:   uint32(seqLen),
		HeadDim:  uint32(headDim),
		NumHeads: uint32(numHeads),
		Offset:   uint32(offset),
		Theta:    theta,
	}
	pBuf, _ := c.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Contents: wgpu.ToBytes([]WGPURoPEParams{p}),
		Usage:    wgpu.BufferUsageUniform,
	})
	defer c.deferOrDestroy(pBuf)

	bindGroup, _ := c.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: targetBuf, Size: targetBuf.GetSize()},
		},
	})
	defer bindGroup.Release()

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)

	halfDim := headDim / 2
	totalPairs := seqLen * numHeads * halfDim
	pass.DispatchWorkgroups((uint32(totalPairs)+63)/64, 1, 1)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

type WGPUEmbeddingParams struct {
	VocabSize  uint32
	HiddenSize uint32
	NumTokens  uint32
	Padding    uint32
}

func (c *WGPUContext) DispatchEmbedding(
	vocabSize, hiddenSize, numTokens int,
	indicesBuf, weightsBuf, outputBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderEmbedding)
	if err != nil {
		return err
	}

	p := WGPUEmbeddingParams{
		VocabSize:  uint32(vocabSize),
		HiddenSize: uint32(hiddenSize),
		NumTokens:  uint32(numTokens),
	}
	pBuf, _ := c.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Contents: wgpu.ToBytes([]WGPUEmbeddingParams{p}),
		Usage:    wgpu.BufferUsageUniform,
	})
	defer c.deferOrDestroy(pBuf)

	bindGroup, _ := c.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: indicesBuf, Size: indicesBuf.GetSize()},
			{Binding: 2, Buffer: weightsBuf, Size: weightsBuf.GetSize()},
			{Binding: 3, Buffer: outputBuf, Size: outputBuf.GetSize()},
		},
	})
	defer bindGroup.Release()

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups((uint32(numTokens*hiddenSize)+63)/64, 1, 1)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}
