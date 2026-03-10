package poly

import (
	"fmt"
	"unsafe"

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

// DispatchDenseQ4 dispatches a tiled dense kernel that dequantizes Q4_0 weights.
func (c *WGPUContext) DispatchDenseQ4(
	batchSize, inputSize, outputSize int,
	inputBuf, scaleBuf, weightBuf, outputBuf *wgpu.Buffer,
	tileSize int,
) error {
	shaderSrc := ShaderTiledDenseQ4(tileSize)
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
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(params)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUDenseParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, scaleBuf, weightBuf, outputBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups((uint32(outputSize)+uint32(tileSize)-1)/uint32(tileSize), uint32(batchSize), 1)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
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
	paramBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(params)))
	c.Queue.WriteBuffer(paramBuf, 0, wgpu.ToBytes([]WGPUDenseParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, paramBuf, inputBuf, weightBuf, outputBuf)
	if err != nil {
		return err
	}

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
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(params)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUMHAParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, qBuf, kBuf, vBuf, oBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups(uint32(numHeads), uint32(seqLen), 1)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

// DispatchSwiGLUQ4 dispatches a tiled SwiGLU kernel with Q4_0 weights.
func (c *WGPUContext) DispatchSwiGLUQ4(
	batchSize, inputSize, outputSize int,
	inputBuf, gateScaleBuf, gateWeightBuf, upScaleBuf, upWeightBuf, outputBuf *wgpu.Buffer,
	tileSize int,
) error {
	shaderSrc := ShaderTiledSwiGLUQ4(tileSize)
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
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(params)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUDenseParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, gateScaleBuf, gateWeightBuf, upScaleBuf, upWeightBuf, outputBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups((uint32(outputSize)+uint32(tileSize)-1)/uint32(tileSize), uint32(batchSize), 1)
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
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(params)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUDenseParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, gateBuf, upBuf, outputBuf)
	if err != nil {
		return err
	}

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
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(params)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPURMSNormParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, weightBuf, outputBuf)
	if err != nil {
		return err
	}

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
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUKVParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, kCache, vCache, newK, newV, pBuf)
	if err != nil {
		return err
	}

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

	pBuf := c.GetUniformBuffer(4)
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]uint32{uint32(size)}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, residualBuf)
	if err != nil {
		return err
	}

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

	pROPE := WGPURoPEParams{
		SeqLen:   uint32(seqLen),
		HeadDim:  uint32(headDim),
		NumHeads: uint32(numHeads),
		Offset:   uint32(offset),
		Theta:    theta,
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(pROPE)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPURoPEParams{pROPE}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, targetBuf)
	if err != nil {
		return err
	}

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

	pEmbed := WGPUEmbeddingParams{
		VocabSize:  uint32(vocabSize),
		HiddenSize: uint32(hiddenSize),
		NumTokens:  uint32(numTokens),
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(pEmbed)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUEmbeddingParams{pEmbed}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, indicesBuf, weightsBuf, outputBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups((uint32(numTokens*hiddenSize)+63)/64, 1, 1)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}
