package poly

import (
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// WGPUDenseScaleParams is the uniform struct for Dense tiled shaders.
// 12 × uint32/float32 = 48 bytes (multiple of 16 for WebGPU uniform alignment).
type WGPUDenseScaleParams struct {
	BatchSize  uint32
	InputSize  uint32
	OutputSize uint32
	Activation uint32
	Scale      float32
	HasBias    uint32
	_pad       [6]uint32 // Padding to 48 bytes (multiple of 16)
}

func isTrulyNil(i any) bool {
	if i == nil { return true }
	switch v := i.(type) {
	case *wgpu.Buffer: return v == nil
	case *WGPUBufferBinding: return v == nil || v.Buffer == nil
	}
	return false
}

// DispatchDenseScaled dispatches a non-tiled Dense forward pass with a scale uniform.
func (c *WGPUContext) DispatchDenseScaled(
	batchSize, inputSize, outputSize int,
	activation uint32,
	scale float32,
	inputBuf, weightBuf, biasBuf, outputBuf any,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderDenseScaled)
	if err != nil {
		return err
	}

	hasBias := uint32(0)
	var bb any = biasBuf
	if isTrulyNil(biasBuf) {
		bb = c.BlankBuffer
	} else {
		hasBias = 1
	}

	params := WGPUDenseScaleParams{
		BatchSize:  uint32(batchSize),
		InputSize:  uint32(inputSize),
		OutputSize: uint32(outputSize),
		Activation: activation,
		Scale:      scale,
		HasBias:    hasBias,
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(params)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUDenseScaleParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, weightBuf, bb, outputBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups((uint32(outputSize)+63)/64, uint32(batchSize), 1)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

// DispatchDenseTiled dispatches a tiled Dense forward pass with workgroup shared-memory
// weight caching.
func (c *WGPUContext) DispatchDenseTiled(
	tileSize, batchSize, inputSize, outputSize int,
	activation uint32,
	scale float32,
	inputBuf, weightBuf, biasBuf, outputBuf any,
) error {
	if tileSize <= 0 {
		tileSize = 32
	}
	shaderSrc := ShaderTiledDense(tileSize)
	pipeline, err := c.CreateComputePipeline(shaderSrc)
	if err != nil {
		return err
	}

	hasBias := uint32(0)
	var bb any = biasBuf
	if isTrulyNil(biasBuf) {
		bb = c.BlankBuffer
	} else {
		hasBias = 1
	}

	params := WGPUDenseScaleParams{
		BatchSize:  uint32(batchSize),
		InputSize:  uint32(inputSize),
		OutputSize: uint32(outputSize),
		Activation: activation,
		Scale:      scale,
		HasBias:    hasBias,
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(params)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUDenseScaleParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, weightBuf, bb, outputBuf)
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

// DispatchDenseBackwardDXTiled dispatches a tiled Dense backward DX pass.
func (c *WGPUContext) DispatchDenseBackwardDXTiled(
	tileSize int,
	batchSize, inputSize, outputSize int,
	activation uint32,
	gradOutputBuf, weightBuf, preActBuf, gradInputBuf any,
) error {
	if tileSize <= 0 {
		tileSize = 64
	}
	pipeline, err := c.CreateComputePipeline(ShaderTiledDenseBackwardDX(tileSize))
	if err != nil {
		return err
	}

	p := WGPUDenseScaleParams{
		BatchSize:  uint32(batchSize),
		InputSize:  uint32(inputSize),
		OutputSize: uint32(outputSize),
		Activation: activation,
		Scale:      1.0, // DX doesn't use quantization scale like forward weights
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUDenseScaleParams{p}))

	// Bindings: Params, gradOutput, weights, gradInput, preAct
	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, weightBuf, gradInputBuf, preActBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups(
		(uint32(inputSize)+uint32(tileSize)-1)/uint32(tileSize),
		uint32(batchSize),
		1,
	)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

// DispatchDenseBackwardDWTiled dispatches a tiled Dense backward DW pass.
func (c *WGPUContext) DispatchDenseBackwardDWTiled(
	tileSize int,
	batchSize, inputSize, outputSize int,
	activation uint32,
	gradOutputBuf, inputBuf, preActBuf, gradWeightsBuf any,
) error {
	if tileSize <= 0 {
		tileSize = 64
	}
	pipeline, err := c.CreateComputePipeline(ShaderTiledDenseBackwardDW(tileSize))
	if err != nil {
		return err
	}

	p := WGPUDenseScaleParams{
		BatchSize:  uint32(batchSize),
		InputSize:  uint32(inputSize),
		OutputSize: uint32(outputSize),
		Activation: activation,
		Scale:      1.0,
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUDenseScaleParams{p}))

	// Bindings: Params, gradOutput, input, gradWeights, preAct
	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, inputBuf, gradWeightsBuf, preActBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups(
		(uint32(inputSize)+uint32(tileSize)-1)/uint32(tileSize),
		uint32(outputSize),
		1,
	)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}
