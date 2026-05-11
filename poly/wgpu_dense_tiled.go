package poly

import (
	"fmt"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// WGPUDenseScaleParams matches ShaderTiledDense / ShaderDenseScaled WGSL DenseScaleParams (48 bytes).
type WGPUDenseScaleParams struct {
	BatchSize      uint32
	InputSize      uint32
	OutputSize     uint32
	Activation     uint32
	Scale          float32
	HasBias        uint32
	TotalOutStride uint32 // 0 => stride is outputSize (legacy); else full output rows for logits layout
	OutputRowBase  uint32
	_pad           [4]uint32
}

// wgpuDenseBackwardUniform matches ShaderTiledDenseBackwardDX/DW (scale + seven padding words).
type wgpuDenseBackwardUniform struct {
	BatchSize  uint32
	InputSize  uint32
	OutputSize uint32
	Activation uint32
	Scale      float32
	Pad        [7]uint32
}

func maxDenseWeightBindBytes(c *WGPUContext) uint64 {
	x := c.Limits.MaxStorageBufferBindingSize
	if x > 512 {
		return x - 256
	}
	return x
}

func isTrulyNil(i any) bool {
	if i == nil {
		return true
	}
	switch v := i.(type) {
	case *wgpu.Buffer:
		return v == nil
	case *WGPUBufferBinding:
		return v == nil || v.Buffer == nil
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
		BatchSize:      uint32(batchSize),
		InputSize:      uint32(inputSize),
		OutputSize:     uint32(outputSize),
		Activation:     activation,
		Scale:          scale,
		HasBias:        hasBias,
		TotalOutStride: 0,
		OutputRowBase:  0,
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

// DispatchDenseTiled dispatches a tiled Dense forward pass. Splits weight bindings along the output
// dimension when vocab×hidden×4 exceeds maxStorageBufferBindingSize (common for large LM heads).
func (c *WGPUContext) DispatchDenseTiled(
	tileSize, batchSize, inputSize, outputSize int,
	activation uint32,
	scale float32,
	inputBuf, weightBuf, biasBuf, outputBuf any,
) error {
	if tileSize <= 0 {
		tileSize = 32
	}
	maxB := maxDenseWeightBindBytes(c)
	rowBytes := uint64(inputSize * 4)
	if rowBytes == 0 {
		return fmt.Errorf("DispatchDenseTiled: inputSize is zero")
	}
	maxChunkOut := int(maxB / rowBytes)
	if maxChunkOut < 1 {
		maxChunkOut = 1
	}

	switch wb := weightBuf.(type) {
	case *WGPUBufferBinding:
		return c.dispatchDenseTiledOnce(tileSize, batchSize, inputSize, outputSize, activation, scale, inputBuf, wb, biasBuf, outputBuf, 0, 0)

	case *wgpu.Buffer:
		wsz := wb.GetSize()
		if wsz <= maxB {
			bind := c.GetSubBuffer(wb, 0, wsz)
			return c.dispatchDenseTiledOnce(tileSize, batchSize, inputSize, outputSize, activation, scale, inputBuf, bind, biasBuf, outputBuf, 0, 0)
		}
		expected := uint64(outputSize) * rowBytes
		if wsz != expected {
			return fmt.Errorf("dense weights size mismatch: buffer=%d want=%d (output=%d input=%d)", wsz, expected, outputSize, inputSize)
		}
		for o0 := 0; o0 < outputSize; o0 += maxChunkOut {
			chunk := maxChunkOut
			if o0+chunk > outputSize {
				chunk = outputSize - o0
			}
			off := uint64(o0) * rowBytes
			sz := uint64(chunk) * rowBytes
			wbind := c.GetSubBuffer(wb, off, sz)
			var bb any = biasBuf
			if !isTrulyNil(biasBuf) {
				if bfull, ok := biasBuf.(*wgpu.Buffer); ok && bfull != nil {
					bb = c.GetSubBuffer(bfull, uint64(o0)*4, uint64(chunk)*4)
				}
			}
			if err := c.dispatchDenseTiledOnce(tileSize, batchSize, inputSize, chunk, activation, scale, inputBuf, wbind, bb, outputBuf, uint32(outputSize), uint32(o0)); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("DispatchDenseTiled: unsupported weights type %T", weightBuf)
	}
}

func (c *WGPUContext) dispatchDenseTiledOnce(
	tileSize, batchSize, inputSize, chunkOutputSize int,
	activation uint32,
	scale float32,
	inputBuf, weightBuf, biasBuf, outputBuf any,
	totalOutStride, outputRowBase uint32,
) error {
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
		BatchSize:      uint32(batchSize),
		InputSize:      uint32(inputSize),
		OutputSize:     uint32(chunkOutputSize),
		Activation:     activation,
		Scale:          scale,
		HasBias:        hasBias,
		TotalOutStride: totalOutStride,
		OutputRowBase:  outputRowBase,
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
	pass.DispatchWorkgroups((uint32(chunkOutputSize)+uint32(tileSize)-1)/uint32(tileSize), uint32(batchSize), 1)
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

	p := wgpuDenseBackwardUniform{
		BatchSize:  uint32(batchSize),
		InputSize:  uint32(inputSize),
		OutputSize: uint32(outputSize),
		Activation: activation,
		Scale:      1.0,
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]wgpuDenseBackwardUniform{p}))

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

	p := wgpuDenseBackwardUniform{
		BatchSize:  uint32(batchSize),
		InputSize:  uint32(inputSize),
		OutputSize: uint32(outputSize),
		Activation: activation,
		Scale:      1.0,
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]wgpuDenseBackwardUniform{p}))

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
