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

type WGPUDenseI8Params struct {
	BatchSize  uint32
	InputSize  uint32
	OutputSize uint32
	TileSize   uint32
	Scale      float32
	_          [3]uint32 // Padding for 16-byte alignment
}

type WGPUSwiGLUI8Params struct {
	BatchSize  uint32
	InputSize  uint32
	OutputSize uint32
	TileSize   uint32
	GScale     float32
	UScale     float32
	_          [2]uint32 // Padding
}

type WGPUDenseBitNetTernaryParams struct {
	BatchSize   uint32
	InputSize   uint32
	OutputSize  uint32
	RowWords    uint32
	WeightScale float32
	Activation  uint32
	HasBias     uint32
	_           uint32
}

type WGPUBitNetQuantizeActivationParams struct {
	BatchSize uint32
	InputSize uint32
	QWords    uint32
	_         uint32
}

type WGPUDenseBitNetTernaryQuantizedParams struct {
	BatchSize   uint32
	InputSize   uint32
	OutputSize  uint32
	RowWords    uint32
	QWords      uint32
	Activation  uint32
	HasBias     uint32
	_           uint32
	WeightScale float32
	_           [3]float32
}

type WGPUBitNetGateProductParams struct {
	BatchSize  uint32
	HiddenSize uint32
	Activation uint32
	_          uint32
}

type WGPUApplyGradientsParams struct {
	Size    uint32
	LR      float32
	ClipVal float32
	_       uint32 // Padding to 16 bytes
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

type WGPURNNParams struct {
	BatchSize  uint32
	InputSize  uint32
	HiddenSize uint32
	Padding    uint32
}

type WGPULSTMParams struct {
	BatchSize  uint32
	InputSize  uint32
	HiddenSize uint32
	Padding    uint32
}

type WGPUCNN1Params struct {
	BatchSize uint32
	InC       uint32
	InL       uint32
	OutC      uint32
	OutL      uint32
	KSize     uint32
	Stride    uint32
	Padding   uint32
}

type WGPUCNN2Params struct {
	BatchSize uint32
	InC       uint32
	InH       uint32
	InW       uint32
	OutC      uint32
	OutH      uint32
	OutW      uint32
	KH        uint32
	KW        uint32
	StrideH   uint32
	StrideW   uint32
	PadH      uint32
	PadW      uint32
}

type WGPUCNN3Params struct {
	BatchSize              uint32
	InC, InD, InH, InW     uint32
	OutC, OutD, OutH, OutW uint32
	KD, KH, KW             uint32
	SD, SH, SW             uint32
	PD, PH, PW             uint32
}

type WGPUCNN1BackwardParams struct {
	BatchSize  uint32
	InC        uint32
	InL        uint32
	Filters    uint32
	OutL       uint32
	KSize      uint32
	Stride     uint32
	Padding    uint32
	Activation uint32
}

type WGPUCNN2BackwardParams struct {
	BatchSize  uint32
	InC        uint32
	InH        uint32
	InW        uint32
	Filters    uint32
	OutH       uint32
	OutW       uint32
	KSize      uint32
	Stride     uint32
	Padding    uint32
	Activation uint32
}

type WGPUCNN3BackwardParams struct {
	BatchSize  uint32
	InC        uint32
	InD        uint32
	InH        uint32
	InW        uint32
	Filters    uint32
	OutD       uint32
	OutH       uint32
	OutW       uint32
	KSize      uint32
	Stride     uint32
	Padding    uint32
	Activation uint32
}

type WGPUMHABackwardParams struct {
	BatchSize  uint32
	NumHeads   uint32
	NumKVHeads uint32
	HeadDim    uint32
	SeqLen     uint32
	Scale      float32
	_          [2]uint32 // Padding to 32 bytes
}

type WGPUActivationParams struct {
	Size uint32
	Act  uint32
	_    [2]uint32 // Padding for 16-byte alignment
}

type WGPULossParams struct {
	Size uint32
	_    [3]uint32 // pad to 16 bytes for WebGPU uniform alignment
}

// WGPUMultiHeadSoftmaxCEParams configures ShaderMultiHeadSoftmaxCEGradPartialLoss (32-byte uniform).
type WGPUMultiHeadSoftmaxCEParams struct {
	Batch    uint32
	RowWidth uint32
	H0       uint32
	H1       uint32
	H2       uint32
	_        [3]uint32
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

// dispatchCompute records a compute dispatch. When BeginFrame is open, all dispatches
// share one ComputePass (the PoC win vs per-op BeginComputePass). Standalone calls
// keep one-shot create/submit behaviour.
func (c *WGPUContext) dispatchCompute(pipeline *wgpu.ComputePipeline, bindGroup *wgpu.BindGroup, x, y, z uint32) error {
	if c.ActiveEncoder != nil {
		if c.ActivePass == nil {
			c.ActivePass = c.ActiveEncoder.BeginComputePass(nil)
		}
		c.ActivePass.SetPipeline(pipeline)
		c.ActivePass.SetBindGroup(0, bindGroup, nil)
		c.ActivePass.DispatchWorkgroups(x, y, z)
		return nil
	}
	enc, err := c.Device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups(x, y, z)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	c.Queue.Submit(cmd)
	return nil
}

// DispatchDenseQ4 dispatches a Q4_0 dense kernel. batch=1 uses decode-shaped shared-mem GEMV.
func (c *WGPUContext) DispatchDenseQ4(
	batchSize, inputSize, outputSize int,
	inputBuf, scaleBuf, weightBuf, outputBuf *wgpu.Buffer,
	tileSize int,
) error {
	if batchSize == 1 && inputSize <= 2048 {
		type decodeParams struct {
			InputSize, OutputSize, _p0, _p1 uint32
		}
		params := decodeParams{uint32(inputSize), uint32(outputSize), 0, 0}
		pipeline, err := c.CreateComputePipeline(ShaderDecodeQ4GEMV)
		if err != nil {
			return err
		}
		pBuf := c.WriteUniformBytes(wgpu.ToBytes([]decodeParams{params}))
		bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, scaleBuf, weightBuf, outputBuf)
		if err != nil {
			return err
		}
		return c.dispatchCompute(pipeline, bindGroup, (uint32(outputSize)+63)/64, 1, 1)
	}

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
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUDenseParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, scaleBuf, weightBuf, outputBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(outputSize)+uint32(tileSize)-1)/uint32(tileSize), uint32(batchSize), 1); err != nil {
		return err
	}
	return nil
}

// DispatchDecodeQ4QKV fuses Q/K/V Q4 GEMVs with one shared-memory load of x (batch=1).
func (c *WGPUContext) DispatchDecodeQ4QKV(
	inputSize, qDim, kvDim int,
	inputBuf *wgpu.Buffer,
	qScales, qWeights, kScales, kWeights, vScales, vWeights *wgpu.Buffer,
	qOut, kOut, vOut *wgpu.Buffer,
) error {
	if inputSize > 2048 {
		return fmt.Errorf("DispatchDecodeQ4QKV: inputSize %d > shared-mem cap 2048", inputSize)
	}
	type p struct {
		InputSize, QDim, KVDim, _pad uint32
	}
	params := p{uint32(inputSize), uint32(qDim), uint32(kvDim), 0}
	pipeline, err := c.CreateComputePipeline(ShaderDecodeQ4GEMV_QKV)
	if err != nil {
		return err
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]p{params}))
	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, qScales, qWeights, kScales, kWeights, vScales, vWeights, qOut, kOut, vOut)
	if err != nil {
		return err
	}
	total := uint32(qDim + kvDim + kvDim)
	return c.dispatchCompute(pipeline, bindGroup, (total+63)/64, 1, 1)
}

// DispatchDenseI8 dispatches a tiled dense kernel that dequantizes INT8 weights.
func (c *WGPUContext) DispatchDenseI8(
	batchSize, inputSize, outputSize int,
	inputBuf, weightBuf, outputBuf *wgpu.Buffer,
	scale float32,
	tileSize int,
) error {
	shaderSrc := ShaderTiledDenseI8(tileSize)
	pipeline, err := c.CreateComputePipeline(shaderSrc)
	if err != nil {
		return err
	}

	params := WGPUDenseI8Params{
		BatchSize:  uint32(batchSize),
		InputSize:  uint32(inputSize),
		OutputSize: uint32(outputSize),
		TileSize:   uint32(tileSize),
		Scale:      scale,
	}
	data := wgpu.ToBytes([]WGPUDenseI8Params{params})
	pBuf := c.GetStickyUniform(fmt.Sprintf("di8_%d_%d_%d_%d_%g", params.BatchSize, params.InputSize, params.OutputSize, params.TileSize, params.Scale), uint64(len(data)), data)

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, weightBuf, outputBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(outputSize)+uint32(tileSize)-1)/uint32(tileSize), uint32(batchSize), 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchDenseBitNetTernary(
	batchSize, inputSize, outputSize int,
	inputBuf, weightBuf, biasBuf, outputBuf *wgpu.Buffer,
	weightScale float32,
	activation ActivationType,
	tileSize int,
) error {
	if tileSize <= 0 {
		tileSize = 32
	}
	if biasBuf == nil {
		biasBuf = c.BlankBuffer
	}
	shaderSrc := ShaderTiledDenseBitNetTernary(tileSize)
	pipeline, err := c.CreateComputePipeline(shaderSrc)
	if err != nil {
		return err
	}
	rowWords := (inputSize + 15) / 16
	params := WGPUDenseBitNetTernaryParams{
		BatchSize:   uint32(batchSize),
		InputSize:   uint32(inputSize),
		OutputSize:  uint32(outputSize),
		RowWords:    uint32(rowWords),
		WeightScale: weightScale,
		Activation:  mapActivation(activation),
	}
	if biasBuf != c.BlankBuffer {
		params.HasBias = 1
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUDenseBitNetTernaryParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, weightBuf, biasBuf, outputBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(outputSize)+uint32(tileSize)-1)/uint32(tileSize), uint32(batchSize), 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchBitNetQuantizeActivation(
	batchSize, inputSize int,
	inputBuf, qPackedBuf, scaleBuf *wgpu.Buffer,
) error {
	// Parallel absmax+pack (256 threads / batch row). Old serial WG(1) starved Turing.
	pipeline, err := c.CreateComputePipeline(ShaderBitNetQuantizeActivationParallel)
	if err != nil {
		return err
	}
	qWords := (inputSize + 3) / 4
	params := WGPUBitNetQuantizeActivationParams{
		BatchSize: uint32(batchSize),
		InputSize: uint32(inputSize),
		QWords:    uint32(qWords),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUBitNetQuantizeActivationParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, qPackedBuf, scaleBuf)
	if err != nil {
		return err
	}

	// One workgroup per batch row (@workgroup_size 256).
	if err := c.dispatchCompute(pipeline, bindGroup, uint32(batchSize), 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchDenseBitNetTernaryQuantized(
	batchSize, inputSize, outputSize int,
	qPackedBuf, scaleBuf, weightBuf, biasBuf, outputBuf *wgpu.Buffer,
	weightScale float32,
	activation ActivationType,
	tileSize int,
) error {
	if biasBuf == nil {
		biasBuf = c.BlankBuffer
	}
	// Decode-shaped path: shared qPacked, one output row per lane (Smol135-style).
	if batchSize == 1 && inputSize > 0 && inputSize <= 8192 {
		return c.DispatchDecodeBitNetGEMV(inputSize, outputSize, qPackedBuf, scaleBuf, weightBuf, biasBuf, outputBuf, weightScale, activation)
	}
	tileSize = bitNetReductionTileSize(tileSize)
	shaderSrc := ShaderTiledDenseBitNetTernaryQuantizedReduce(tileSize)
	pipeline, err := c.CreateComputePipeline(shaderSrc)
	if err != nil {
		return err
	}
	rowWords := (inputSize + 15) / 16
	qWords := (inputSize + 3) / 4
	params := WGPUDenseBitNetTernaryQuantizedParams{
		BatchSize:   uint32(batchSize),
		InputSize:   uint32(inputSize),
		OutputSize:  uint32(outputSize),
		RowWords:    uint32(rowWords),
		QWords:      uint32(qWords),
		Activation:  mapActivation(activation),
		WeightScale: weightScale,
	}
	if biasBuf != c.BlankBuffer {
		params.HasBias = 1
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUDenseBitNetTernaryQuantizedParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, qPackedBuf, scaleBuf, weightBuf, biasBuf, outputBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, uint32(outputSize), uint32(batchSize), 1); err != nil {
		return err
	}
	return nil
}

// DispatchDecodeBitNetGEMV is batch=1 ternary GEMV with shared int8 activations.
func (c *WGPUContext) DispatchDecodeBitNetGEMV(
	inputSize, outputSize int,
	qPackedBuf, scaleBuf, weightBuf, biasBuf, outputBuf *wgpu.Buffer,
	weightScale float32,
	activation ActivationType,
) error {
	if inputSize <= 0 || outputSize <= 0 {
		return fmt.Errorf("DispatchDecodeBitNetGEMV: bad dims %d→%d", inputSize, outputSize)
	}
	if inputSize > 8192 {
		return fmt.Errorf("DispatchDecodeBitNetGEMV: inputSize %d > shared-mem cap 8192", inputSize)
	}
	if biasBuf == nil {
		biasBuf = c.BlankBuffer
	}
	pipeline, err := c.CreateComputePipeline(ShaderDecodeBitNetGEMV)
	if err != nil {
		return err
	}
	rowWords := (inputSize + 15) / 16
	qWords := (inputSize + 3) / 4
	params := WGPUDenseBitNetTernaryQuantizedParams{
		BatchSize:   1,
		InputSize:   uint32(inputSize),
		OutputSize:  uint32(outputSize),
		RowWords:    uint32(rowWords),
		QWords:      uint32(qWords),
		Activation:  mapActivation(activation),
		WeightScale: weightScale,
	}
	if biasBuf != c.BlankBuffer {
		params.HasBias = 1
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUDenseBitNetTernaryQuantizedParams{params}))
	bindGroup, err := c.GetBindGroup(pipeline, pBuf, qPackedBuf, scaleBuf, weightBuf, biasBuf, outputBuf)
	if err != nil {
		return err
	}
	nWG := (uint32(outputSize) + 63) / 64
	return c.dispatchCompute(pipeline, bindGroup, nWG, 1, 1)
}

type wgpuDecodeBitNetQKVParams struct {
	InputSize uint32
	QDim      uint32
	KVDim     uint32
	RowWords  uint32
	QWords    uint32
	Pad0      uint32
	Pad1      uint32
	Pad2      uint32
	QScale    float32
	KScale    float32
	VScale    float32
	Pad3      float32
}

// DispatchDecodeBitNetQKV fuses Q/K/V ternary GEMVs after one shared int8 act load.
func (c *WGPUContext) DispatchDecodeBitNetQKV(
	inputSize, qDim, kvDim int,
	qPackedBuf, scaleBuf, qW, kW, vW, qOut, kOut, vOut *wgpu.Buffer,
	qScale, kScale, vScale float32,
) error {
	if inputSize <= 0 || inputSize > 8192 {
		return fmt.Errorf("DispatchDecodeBitNetQKV: inputSize %d out of range", inputSize)
	}
	pipeline, err := c.CreateComputePipeline(ShaderDecodeBitNetGEMV_QKV)
	if err != nil {
		return err
	}
	params := wgpuDecodeBitNetQKVParams{
		InputSize: uint32(inputSize),
		QDim:      uint32(qDim),
		KVDim:     uint32(kvDim),
		RowWords:  uint32((inputSize + 15) / 16),
		QWords:    uint32((inputSize + 3) / 4),
		QScale:    qScale,
		KScale:    kScale,
		VScale:    vScale,
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]wgpuDecodeBitNetQKVParams{params}))
	bindGroup, err := c.GetBindGroup(pipeline, pBuf, qPackedBuf, scaleBuf, qW, kW, vW, qOut, kOut, vOut)
	if err != nil {
		return err
	}
	total := uint32(qDim + kvDim + kvDim)
	nWG := (total + 63) / 64
	return c.dispatchCompute(pipeline, bindGroup, nWG, 1, 1)
}

func bitNetReductionTileSize(tileSize int) int {
	if tileSize <= 0 {
		return 64
	}
	if tileSize < 64 {
		return 64
	}
	if tileSize >= 128 {
		return 128
	}
	return 64
}

func (c *WGPUContext) DispatchBitNetGateProduct(
	batchSize, hiddenSize int,
	activation ActivationType,
	gateBuf, upBuf, outputBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderBitNetGateProduct)
	if err != nil {
		return err
	}
	params := WGPUBitNetGateProductParams{
		BatchSize:  uint32(batchSize),
		HiddenSize: uint32(hiddenSize),
		Activation: mapActivation(activation),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUBitNetGateProductParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gateBuf, upBuf, outputBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(batchSize*hiddenSize)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchSwiGLUI8(
	batchSize, inputSize, outputSize int,
	inputBuf, gateWeightBuf, upWeightBuf, gateBiasBuf, upBiasBuf, outputBuf *wgpu.Buffer,
	gScale, uScale float32,
	tileSize int,
) error {
	shaderSrc := ShaderTiledSwiGLUI8(tileSize)
	pipeline, err := c.CreateComputePipeline(shaderSrc)
	if err != nil {
		return err
	}

	params := WGPUSwiGLUI8Params{
		BatchSize:  uint32(batchSize),
		InputSize:  uint32(inputSize),
		OutputSize: uint32(outputSize),
		TileSize:   uint32(tileSize),
		GScale:     gScale,
		UScale:     uScale,
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUSwiGLUI8Params{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, gateWeightBuf, upWeightBuf, gateBiasBuf, upBiasBuf, outputBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(outputSize)+uint32(tileSize)-1)/uint32(tileSize), uint32(batchSize), 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchAdd(size int, a, b, res any) error {
	pipeline, err := c.CreateComputePipeline(ShaderAdd)
	if err != nil {
		return err
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]uint32{uint32(size)}))
	bindGroup, err := c.GetBindGroup(pipeline, pBuf, a, b, res)
	if err != nil {
		return err
	}
	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(size)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

// DispatchDense dispatches a tiled dense matrix-multiply kernel.
func (c *WGPUContext) DispatchDense(
	batchSize, inputSize, outputSize int,
	inputBuf, weightBuf, outputBuf any,
	tileSize int,
) error {
	// For backward compatibility, use a default scale and linear activation.
	// Map ActivationLinear (-1) to a safe positive uint32 (99) to avoid constant overflow.
	var act uint32
	if int(ActivationLinear) < 0 {
		act = 99
	} else {
		// Use a non-constant cast to satisfy the compiler
		v := int(ActivationLinear)
		act = uint32(v)
	}
	return c.DispatchDenseTiled(tileSize, batchSize, inputSize, outputSize, act, 1.0, inputBuf, weightBuf, nil, outputBuf)
}

// DispatchDenseBackwardDX calculates gradInput = gradOutput * weights
func (c *WGPUContext) DispatchDenseBackwardDX(
	batchSize, inputSize, outputSize int,
	gradOutputBuf, weightBuf, gradInputBuf any,
	tileSize int,
) error {
	// We lack preActBuf and activation here in the legacy signature.
	// We'll have to use the untiled/non-activation version or a dummy preAct.
	// BUT the premium implementation needs preAct.
	// For legacy compatibility, we'll restore the simple non-integrated DX shader.
	pipeline, err := c.CreateComputePipeline(ShaderDenseBackwardDX(tileSize))
	if err != nil {
		return err
	}

	params := WGPUDenseParams{
		BatchSize:  uint32(batchSize),
		InputSize:  uint32(inputSize),
		OutputSize: uint32(outputSize),
		TileSize:   uint32(tileSize),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUDenseParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, weightBuf, gradInputBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(inputSize)+uint32(tileSize)-1)/uint32(tileSize), uint32(batchSize), 1); err != nil {
		return err
	}
	return nil
}

// DispatchDenseBackwardDW calculates gradWeights = gradOutput^T * input
func (c *WGPUContext) DispatchDenseBackwardDW(
	batchSize, inputSize, outputSize int,
	gradOutputBuf, inputBuf, gradWeightBuf any,
	tileSize int,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderDenseBackwardDW(tileSize))
	if err != nil {
		return err
	}

	params := WGPUDenseParams{
		BatchSize:  uint32(batchSize),
		InputSize:  uint32(inputSize),
		OutputSize: uint32(outputSize),
		TileSize:   uint32(tileSize),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUDenseParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, inputBuf, gradWeightBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(inputSize)+uint32(tileSize)-1)/uint32(tileSize), uint32(outputSize), 1); err != nil {
		return err
	}
	return nil
}

// DispatchMHA dispatches the tiled multi-head attention kernel.
func (c *WGPUContext) DispatchMHA(
	numHeads, numKVHeads, headDim, seqLen, kvOffset, maxSeqLen int,
	qBuf, kBuf, vBuf, oBuf any,
	tileSize int,
) error {
	return c.dispatchMHA(numHeads, numKVHeads, headDim, seqLen, kvOffset, maxSeqLen, qBuf, kBuf, vBuf, oBuf, tileSize, nil)
}

// DispatchMHAStep is DispatchMHA with position from step[0] (chunked greedy decode).
func (c *WGPUContext) DispatchMHAStep(
	numHeads, numKVHeads, headDim, seqLen, maxSeqLen int,
	qBuf, kBuf, vBuf, oBuf any,
	tileSize int,
	stepBuf *wgpu.Buffer,
) error {
	return c.dispatchMHA(numHeads, numKVHeads, headDim, seqLen, 0, maxSeqLen, qBuf, kBuf, vBuf, oBuf, tileSize, stepBuf)
}

func (c *WGPUContext) dispatchMHA(
	numHeads, numKVHeads, headDim, seqLen, kvOffset, maxSeqLen int,
	qBuf, kBuf, vBuf, oBuf any,
	tileSize int,
	stepBuf *wgpu.Buffer,
) error {
	var shaderSrc string
	if stepBuf != nil {
		shaderSrc = ShaderTiledMHANStep(tileSize, headDim)
	} else {
		shaderSrc = ShaderTiledMHAN(tileSize, headDim)
	}
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
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUMHAParams{params}))

	var bindGroup *wgpu.BindGroup
	if stepBuf != nil {
		bindGroup, err = c.GetBindGroup(pipeline, pBuf, qBuf, kBuf, vBuf, oBuf, stepBuf)
	} else {
		bindGroup, err = c.GetBindGroup(pipeline, pBuf, qBuf, kBuf, vBuf, oBuf)
	}
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, uint32(numHeads), uint32(seqLen), 1); err != nil {
		return err
	}
	return nil
}

// DispatchSwiGLUQ4 dispatches a tiled SwiGLU kernel with Q4_0 weights.
func (c *WGPUContext) DispatchSwiGLUQ4(
	batchSize, inputSize, outputSize int,
	inputBuf, gateScaleBuf, gateWeightBuf, upScaleBuf, upWeightBuf, gateBiasBuf, upBiasBuf, outputBuf *wgpu.Buffer,
	tileSize int,
) error {
	if tileSize <= 0 {
		tileSize = 32
	}
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
	data := wgpu.ToBytes([]WGPUDenseParams{params})
	pBuf := c.GetStickyUniform(fmt.Sprintf("swq4_%d_%d_%d_%d", params.BatchSize, params.InputSize, params.OutputSize, params.TileSize), uint64(len(data)), data)

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, gateScaleBuf, gateWeightBuf, upScaleBuf, upWeightBuf, gateBiasBuf, upBiasBuf, outputBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(outputSize)+uint32(tileSize)-1)/uint32(tileSize), uint32(batchSize), 1); err != nil {
		return err
	}
	return nil
}

// DispatchSwiGLU dispatches the tiled SwiGLU MLP kernel.
func (c *WGPUContext) DispatchSwiGLU(
	batchSize, inputSize, outputSize int,
	inputBuf, gateBuf, upBuf, gateBiasBuf, upBiasBuf, outputBuf *wgpu.Buffer,
	tileSize int,
) error {
	if tileSize <= 0 {
		tileSize = 32
	}
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
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUDenseParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, gateBuf, upBuf, gateBiasBuf, upBiasBuf, outputBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(outputSize)+uint32(tileSize)-1)/uint32(tileSize), uint32(batchSize), 1); err != nil {
		return err
	}
	return nil
}

// DispatchSwiGLUWithActCache runs the SwiGLU forward pass and also stores the raw gate and
// up projections to gateOutBuf/upOutBuf so the backward pass can use them.
func (c *WGPUContext) DispatchSwiGLUWithActCache(
	batchSize, inputSize, outputSize int,
	inputBuf, gateBuf, upBuf, gateBiasBuf, upBiasBuf, outputBuf, gateOutBuf, upOutBuf *wgpu.Buffer,
	tileSize int,
) error {
	if tileSize <= 0 {
		tileSize = 32
	}
	pipeline, err := c.CreateComputePipeline(ShaderTiledSwiGLUActCache(tileSize))
	if err != nil {
		return err
	}
	params := WGPUDenseParams{
		BatchSize:  uint32(batchSize),
		InputSize:  uint32(inputSize),
		OutputSize: uint32(outputSize),
		TileSize:   uint32(tileSize),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUDenseParams{params}))
	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, gateBuf, upBuf, gateBiasBuf, upBiasBuf, outputBuf, gateOutBuf, upOutBuf)
	if err != nil {
		return err
	}
	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(outputSize)+uint32(tileSize)-1)/uint32(tileSize), uint32(batchSize), 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchSwiGLUBackward(
	batchSize, inputSize, outputSize int,
	gradOutputBuf, gateInBuf, upInBuf, gradGateBuf, gradUpBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderSwiGLUBackward)
	if err != nil {
		return err
	}

	params := WGPUDenseParams{
		BatchSize:  uint32(batchSize),
		InputSize:  uint32(inputSize),
		OutputSize: uint32(outputSize),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUDenseParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, gateInBuf, upInBuf, gradGateBuf, gradUpBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(batchSize*outputSize)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

// WGPURMSNormParams matches the WGSL struct
type WGPURMSNormParams struct {
	Size    uint32
	Epsilon float32
	_       [2]uint32 // Padding to 16 bytes
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
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPURMSNormParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, weightBuf, outputBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, uint32(batchSize), 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchRMSNormBackward(
	batchSize, size int, epsilon float32,
	gradOutputBuf, inputBuf, rmsBuf, weightBuf, gradInputBuf, gradWeightBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderRMSNormBackward)
	if err != nil {
		return err
	}

	params := WGPURMSNormParams{Size: uint32(size), Epsilon: epsilon}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPURMSNormParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, inputBuf, rmsBuf, weightBuf, gradInputBuf, gradWeightBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, uint32(batchSize), 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchEmbeddingBackward(
	vocabSize, hiddenSize, numTokens int,
	indicesBuf, gradOutputBuf, gradWeightBuf any,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderEmbeddingBackward)
	if err != nil {
		return err
	}

	pEmbed := WGPUEmbeddingParams{
		VocabSize:  uint32(vocabSize),
		HiddenSize: uint32(hiddenSize),
		NumTokens:  uint32(numTokens),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUEmbeddingParams{pEmbed}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, indicesBuf, gradOutputBuf, gradWeightBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(numTokens*hiddenSize)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchResidualBackward(
	size int,
	gradOutputBuf, gradInputBuf, gradResidualBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderResidualBackward)
	if err != nil {
		return err
	}

	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]uint32{uint32(size)}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, gradInputBuf, gradResidualBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(size)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

type WGPUKVParams struct {
	Offset     uint32
	HeadDim    uint32
	MaxSeqLen  uint32
	NumKVHeads uint32
	NumTokens  uint32
	_          [7]uint32 // Pad to 48 bytes (5 + 7 = 12 * 4)
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
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUKVParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, kCache, vCache, newK, newV, pBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(numKVHeads)*uint32(headDim)*uint32(numTokens)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

// DispatchKVUpdateStep writes K/V at GPU step[0] (stable uniforms; chunked decode).
func (c *WGPUContext) DispatchKVUpdateStep(
	headDim, maxSeqLen, numKVHeads, numTokens int,
	kCache, vCache, newK, newV, stepBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderKVUpdateStep)
	if err != nil {
		return err
	}
	type kvStepParams struct {
		HeadDim, MaxSeqLen, NumKVHeads, NumTokens uint32
	}
	p := kvStepParams{uint32(headDim), uint32(maxSeqLen), uint32(numKVHeads), uint32(numTokens)}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]kvStepParams{p}))
	bindGroup, err := c.GetBindGroup(pipeline, pBuf, stepBuf, kCache, vCache, newK, newV)
	if err != nil {
		return err
	}
	return c.dispatchCompute(pipeline, bindGroup, (uint32(numKVHeads)*uint32(headDim)*uint32(numTokens)+63)/64, 1, 1)
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

	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]uint32{uint32(size)}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, residualBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(size)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

type WGPURoPEParams struct {
	SeqLen   uint32
	HeadDim  uint32
	NumHeads uint32
	Offset   uint32
	Theta    float32
	_        [3]uint32 // Padding to 32 bytes
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
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPURoPEParams{pROPE}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, targetBuf)
	if err != nil {
		return err
	}

	halfDim := headDim / 2
	totalPairs := seqLen * numHeads * halfDim
	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(totalPairs)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

// DispatchRoPEStep applies RoPE at GPU step[0] (stable uniforms; chunked decode).
func (c *WGPUContext) DispatchRoPEStep(
	seqLen, headDim, numHeads int, theta float32,
	targetBuf, stepBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderRoPEStep)
	if err != nil {
		return err
	}
	type ropeStepParams struct {
		SeqLen, HeadDim, NumHeads, _pad uint32
		Theta                           float32
		_                               [3]uint32
	}
	p := ropeStepParams{SeqLen: uint32(seqLen), HeadDim: uint32(headDim), NumHeads: uint32(numHeads), Theta: theta}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]ropeStepParams{p}))
	bindGroup, err := c.GetBindGroup(pipeline, pBuf, stepBuf, targetBuf)
	if err != nil {
		return err
	}
	halfDim := headDim / 2
	totalPairs := seqLen * numHeads * halfDim
	return c.dispatchCompute(pipeline, bindGroup, (uint32(totalPairs)+63)/64, 1, 1)
}

type WGPUEmbeddingParams struct {
	VocabSize  uint32
	HiddenSize uint32
	NumTokens  uint32
	Padding    uint32
}

// WGPUEmbeddingShardParams matches ShaderEmbeddingShard WGSL Params.
type WGPUEmbeddingShardParams struct {
	VocabSize  uint32
	HiddenSize uint32
	NumTokens  uint32
	_pad       uint32
	RowOffset  uint32
	NumRows    uint32
	_pad2      uint32
}

func (c *WGPUContext) zeroWriteBuffer(dst *wgpu.Buffer, sizeBytes uint64) {
	const chunk = 4 * 1024 * 1024
	z := make([]byte, chunk)
	var off uint64
	for off < sizeBytes {
		n := chunk
		if uint64(n) > sizeBytes-off {
			n = int(sizeBytes - off)
			z = make([]byte, n)
		}
		c.Queue.WriteBuffer(dst, off, z[:n])
		off += uint64(n)
	}
}

// DispatchEmbedding gathers token embeddings. Splits weight bindings when the embedding matrix
// exceeds maxStorageBufferBindingSize (~1 GiB on many adapters).
func (c *WGPUContext) DispatchEmbedding(
	vocabSize, hiddenSize, numTokens int,
	indicesBuf, weightsBuf, outputBuf any,
) error {
	wb, ok := weightsBuf.(*wgpu.Buffer)
	if !ok || wb == nil {
		return fmt.Errorf("DispatchEmbedding: weights must be *wgpu.Buffer")
	}
	maxBind := c.Limits.MaxStorageBufferBindingSize
	if maxBind > 512 {
		maxBind -= 256
	}
	rowBytes := uint64(hiddenSize * 4)
	if rowBytes == 0 {
		return fmt.Errorf("DispatchEmbedding: hiddenSize is zero")
	}
	weightBytes := wb.GetSize()
	if weightBytes <= maxBind {
		pipeline, err := c.CreateComputePipeline(ShaderEmbedding)
		if err != nil {
			return err
		}
		pEmbed := WGPUEmbeddingParams{
			VocabSize:  uint32(vocabSize),
			HiddenSize: uint32(hiddenSize),
			NumTokens:  uint32(numTokens),
		}
		pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUEmbeddingParams{pEmbed}))
		bindGroup, err := c.GetBindGroup(pipeline, pBuf, indicesBuf, weightsBuf, outputBuf)
		if err != nil {
			return err
		}
		if err := c.dispatchCompute(pipeline, bindGroup, (uint32(numTokens*hiddenSize)+63)/64, 1, 1); err != nil {
			return err
		}
		return nil
	}

	outBytes := uint64(numTokens * hiddenSize * 4)
	ob, ok := outputBuf.(*wgpu.Buffer)
	if !ok || ob == nil {
		return fmt.Errorf("DispatchEmbedding: output must be *wgpu.Buffer for sharded path")
	}
	c.zeroWriteBuffer(ob, outBytes)

	maxRows := int(maxBind / rowBytes)
	if maxRows < 1 {
		maxRows = 1
	}

	pipeline, err := c.CreateComputePipeline(ShaderEmbeddingShard)
	if err != nil {
		return err
	}

	for rowOff := 0; rowOff < vocabSize; rowOff += maxRows {
		nRows := maxRows
		if rowOff+nRows > vocabSize {
			nRows = vocabSize - rowOff
		}
		off := uint64(rowOff * hiddenSize * 4)
		sz := uint64(nRows * hiddenSize * 4)
		wShard := c.GetSubBuffer(wb, off, sz)

		pEmbed := WGPUEmbeddingShardParams{
			VocabSize:  uint32(vocabSize),
			HiddenSize: uint32(hiddenSize),
			NumTokens:  uint32(numTokens),
			RowOffset:  uint32(rowOff),
			NumRows:    uint32(nRows),
		}
		pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUEmbeddingShardParams{pEmbed}))

		bindGroup, err := c.GetBindGroup(pipeline, pBuf, indicesBuf, wShard, outputBuf)
		if err != nil {
			return err
		}
		if err := c.dispatchCompute(pipeline, bindGroup, (uint32(numTokens*hiddenSize)+63)/64, 1, 1); err != nil {
			return err
		}
	}
	return nil
}

func (c *WGPUContext) DispatchRNNStep(
	batchSize, inputSize, hiddenSize int,
	inputBuf, hPrevBuf, wIHBuf, wHHBuf, biasBuf, hCurrBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderRNNStep)
	if err != nil {
		return err
	}

	p := WGPURNNParams{BatchSize: uint32(batchSize), InputSize: uint32(inputSize), HiddenSize: uint32(hiddenSize)}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPURNNParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, hPrevBuf, wIHBuf, wHHBuf, biasBuf, hCurrBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(hiddenSize)+63)/64, uint32(batchSize), 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchLSTMStep(
	batchSize, inputSize, hiddenSize int,
	inputBuf, hPrevBuf, cPrevBuf, weightBuf, hCurrBuf, cCurrBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderLSTMStep)
	if err != nil {
		return err
	}

	p := WGPULSTMParams{BatchSize: uint32(batchSize), InputSize: uint32(inputSize), HiddenSize: uint32(hiddenSize)}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPULSTMParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, hPrevBuf, cPrevBuf, weightBuf, hCurrBuf, cCurrBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(hiddenSize)+63)/64, uint32(batchSize), 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchLSTMStepPreAct(
	batchSize, inputSize, hiddenSize int,
	inputBuf, hPrevBuf, cPrevBuf, weightBuf, hCurrBuf, cCurrBuf, preActBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderLSTMStepPreAct)
	if err != nil {
		return err
	}
	p := WGPULSTMParams{BatchSize: uint32(batchSize), InputSize: uint32(inputSize), HiddenSize: uint32(hiddenSize)}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPULSTMParams{p}))
	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, hPrevBuf, cPrevBuf, weightBuf, hCurrBuf, cCurrBuf, preActBuf)
	if err != nil {
		return err
	}
	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(hiddenSize)+63)/64, uint32(batchSize), 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchRNNBackwardDX(
	tileSize, batchSize, inputSize, hiddenSize int,
	gradOutputBuf, wIHBuf, hCurrBuf, gradInputBuf *wgpu.Buffer,
) error {
	if tileSize <= 0 {
		tileSize = 64
	}
	pipeline, err := c.CreateComputePipeline(ShaderTiledRNNBackwardDX(tileSize))
	if err != nil {
		return err
	}
	p := WGPURNNParams{BatchSize: uint32(batchSize), InputSize: uint32(inputSize), HiddenSize: uint32(hiddenSize)}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPURNNParams{p}))
	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, wIHBuf, hCurrBuf, gradInputBuf)
	if err != nil {
		return err
	}
	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(inputSize)+uint32(tileSize)-1)/uint32(tileSize), uint32(batchSize), 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchRNNBackwardDW(
	tileSize, batchSize, inputSize, hiddenSize int,
	gradOutputBuf, inputBuf, hCurrBuf, hPrevBuf, gradWeightsBuf *wgpu.Buffer,
) error {
	if tileSize <= 0 {
		tileSize = 64
	}
	pipeline, err := c.CreateComputePipeline(ShaderTiledRNNBackwardDW(tileSize))
	if err != nil {
		return err
	}
	p := WGPURNNParams{BatchSize: uint32(batchSize), InputSize: uint32(inputSize), HiddenSize: uint32(hiddenSize)}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPURNNParams{p}))
	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, inputBuf, hCurrBuf, hPrevBuf, gradWeightsBuf)
	if err != nil {
		return err
	}
	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(hiddenSize)+uint32(tileSize)-1)/uint32(tileSize), 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchLSTMBackwardDX(
	tileSize, batchSize, inputSize, hiddenSize int,
	gradOutputBuf, weightBuf, preActBuf, gradInputBuf *wgpu.Buffer,
) error {
	if tileSize <= 0 {
		tileSize = 64
	}
	pipeline, err := c.CreateComputePipeline(ShaderTiledLSTMBackwardDX(tileSize))
	if err != nil {
		return err
	}
	p := WGPULSTMParams{BatchSize: uint32(batchSize), InputSize: uint32(inputSize), HiddenSize: uint32(hiddenSize)}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPULSTMParams{p}))
	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, weightBuf, preActBuf, gradInputBuf)
	if err != nil {
		return err
	}
	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(inputSize)+uint32(tileSize)-1)/uint32(tileSize), uint32(batchSize), 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchLSTMBackwardDW(
	tileSize, batchSize, inputSize, hiddenSize int,
	gradOutputBuf, inputBuf, preActBuf, hPrevBuf, gradWeightsBuf *wgpu.Buffer,
) error {
	if tileSize <= 0 {
		tileSize = 64
	}
	pipeline, err := c.CreateComputePipeline(ShaderTiledLSTMBackwardDW(tileSize))
	if err != nil {
		return err
	}
	p := WGPULSTMParams{BatchSize: uint32(batchSize), InputSize: uint32(inputSize), HiddenSize: uint32(hiddenSize)}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPULSTMParams{p}))
	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, inputBuf, preActBuf, hPrevBuf, gradWeightsBuf)
	if err != nil {
		return err
	}
	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(hiddenSize)+uint32(tileSize)-1)/uint32(tileSize), 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchCNN1(
	batchSize, inC, inL, outC, outL, kSize, stride, padding int,
	inputBuf, weightBuf, outputBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderCNN1)
	if err != nil {
		return err
	}

	p := WGPUCNN1Params{
		BatchSize: uint32(batchSize),
		InC:       uint32(inC), InL: uint32(inL),
		OutC: uint32(outC), OutL: uint32(outL),
		KSize: uint32(kSize), Stride: uint32(stride), Padding: uint32(padding),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUCNN1Params{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, weightBuf, outputBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(outL)+7)/8, (uint32(outC)+7)/8, uint32(batchSize)); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchCNN2(
	batchSize, inC, inH, inW, outC, outH, outW, kH, kW, strideH, strideW, padH, padW int,
	inputBuf, weightBuf, outputBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderCNN2)
	if err != nil {
		return err
	}

	p := WGPUCNN2Params{
		BatchSize: uint32(batchSize),
		InC:       uint32(inC), InH: uint32(inH), InW: uint32(inW),
		OutC: uint32(outC), OutH: uint32(outH), OutW: uint32(outW),
		KH: uint32(kH), KW: uint32(kW),
		StrideH: uint32(strideH), StrideW: uint32(strideW),
		PadH: uint32(padH), PadW: uint32(padW),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUCNN2Params{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, weightBuf, outputBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(outH*outW)+255)/16, (uint32(outC)+15)/16, uint32(batchSize)); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchCNN3(
	batchSize, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW int,
	inputBuf, weightBuf, outputBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderCNN3)
	if err != nil {
		return err
	}

	p := WGPUCNN3Params{
		BatchSize: uint32(batchSize),
		InC:       uint32(inC), InD: uint32(inD), InH: uint32(inH), InW: uint32(inW),
		OutC: uint32(outC), OutD: uint32(outD), OutH: uint32(outH), OutW: uint32(outW),
		KD: uint32(kD), KH: uint32(kH), KW: uint32(kW),
		SD: uint32(sD), SH: uint32(sH), SW: uint32(sW),
		PD: uint32(pD), PH: uint32(pH), PW: uint32(pW),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUCNN3Params{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, weightBuf, outputBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(outD*outH*outW)+63)/64, (uint32(outC)+1)/1, uint32(batchSize)); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchCNN1BackwardDX(
	batchSize, inC, inL, filters, outL, kSize, stride, padding int,
	activation ActivationType,
	gradOutputBuf, weightBuf, preActBuf, gradInputBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderCNN1BackwardDX)
	if err != nil {
		return err
	}

	p := WGPUCNN1BackwardParams{
		BatchSize: uint32(batchSize),
		InC:       uint32(inC), InL: uint32(inL),
		Filters: uint32(filters), OutL: uint32(outL),
		KSize: uint32(kSize), Stride: uint32(stride), Padding: uint32(padding),
		Activation: mapActivation(activation),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUCNN1BackwardParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, weightBuf, gradInputBuf, preActBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(batchSize*inC*inL)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchCNN1BackwardDW(
	batchSize, inC, inL, filters, outL, kSize, stride, padding int,
	activation ActivationType,
	gradOutputBuf, inputBuf, preActBuf, gradWeightBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderCNN1BackwardDW)
	if err != nil {
		return err
	}

	p := WGPUCNN1BackwardParams{
		BatchSize: uint32(batchSize),
		InC:       uint32(inC), InL: uint32(inL),
		Filters: uint32(filters), OutL: uint32(outL),
		KSize: uint32(kSize), Stride: uint32(stride), Padding: uint32(padding),
		Activation: mapActivation(activation),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUCNN1BackwardParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, inputBuf, gradWeightBuf, preActBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(filters*inC*kSize)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchCNN2BackwardDX(
	batchSize, inC, inH, inW, filters, outH, outW, kSize, stride, padding int,
	activation ActivationType,
	gradOutputBuf, weightBuf, preActBuf, gradInputBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderCNN2BackwardDX)
	if err != nil {
		return err
	}

	p := WGPUCNN2BackwardParams{
		BatchSize: uint32(batchSize),
		InC:       uint32(inC), InH: uint32(inH), InW: uint32(inW),
		Filters: uint32(filters), OutH: uint32(outH), OutW: uint32(outW),
		KSize: uint32(kSize), Stride: uint32(stride), Padding: uint32(padding),
		Activation: mapActivation(activation),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUCNN2BackwardParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, weightBuf, gradInputBuf, preActBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(batchSize*inC*inH*inW)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchCNN2BackwardDW(
	batchSize, inC, inH, inW, filters, outH, outW, kSize, stride, padding int,
	activation ActivationType,
	gradOutputBuf, inputBuf, preActBuf, gradWeightBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderCNN2BackwardDW)
	if err != nil {
		return err
	}

	p := WGPUCNN2BackwardParams{
		BatchSize: uint32(batchSize),
		InC:       uint32(inC), InH: uint32(inH), InW: uint32(inW),
		Filters: uint32(filters), OutH: uint32(outH), OutW: uint32(outW),
		KSize: uint32(kSize), Stride: uint32(stride), Padding: uint32(padding),
		Activation: mapActivation(activation),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUCNN2BackwardParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, inputBuf, gradWeightBuf, preActBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(filters*inC*kSize*kSize)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchCNN3BackwardDX(
	batchSize, inC, inD, inH, inW, filters, outD, outH, outW, kSize, stride, padding int,
	activation ActivationType,
	gradOutputBuf, weightBuf, preActBuf, gradInputBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderCNN3BackwardDX)
	if err != nil {
		return err
	}

	p := WGPUCNN3BackwardParams{
		BatchSize: uint32(batchSize),
		InC:       uint32(inC), InD: uint32(inD), InH: uint32(inH), InW: uint32(inW),
		Filters: uint32(filters), OutD: uint32(outD), OutH: uint32(outH), OutW: uint32(outW),
		KSize: uint32(kSize), Stride: uint32(stride), Padding: uint32(padding),
		Activation: mapActivation(activation),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUCNN3BackwardParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, weightBuf, gradInputBuf, preActBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(batchSize*inC*inD*inH*inW)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchCNN3BackwardDW(
	batchSize, inC, inD, inH, inW, filters, outD, outH, outW, kSize, stride, padding int,
	activation ActivationType,
	gradOutputBuf, inputBuf, preActBuf, gradWeightBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderCNN3BackwardDW)
	if err != nil {
		return err
	}

	p := WGPUCNN3BackwardParams{
		BatchSize: uint32(batchSize),
		InC:       uint32(inC), InD: uint32(inD), InH: uint32(inH), InW: uint32(inW),
		Filters: uint32(filters), OutD: uint32(outD), OutH: uint32(outH), OutW: uint32(outW),
		KSize: uint32(kSize), Stride: uint32(stride), Padding: uint32(padding),
		Activation: mapActivation(activation),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUCNN3BackwardParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, inputBuf, gradWeightBuf, preActBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(filters*inC*kSize*kSize*kSize)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchMHABackward(
	batchSize, numHeads, numKVHeads, headDim, seqLen int, scale float32,
	gradOutputBuf, qBuf, kBuf, vBuf, dQBuf, dKBuf, dVBuf any,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderMHABackward)
	if err != nil {
		return err
	}

	p := WGPUMHABackwardParams{
		BatchSize: uint32(batchSize),
		NumHeads:  uint32(numHeads), NumKVHeads: uint32(numKVHeads),
		HeadDim: uint32(headDim), SeqLen: uint32(seqLen),
		Scale: scale,
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUMHABackwardParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, qBuf, kBuf, vBuf, dQBuf, dKBuf, dVBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, uint32(batchSize*numHeads*seqLen+31)/32, 1, 1); err != nil {
		return err
	}
	return nil
}

func mapActivation(act ActivationType) uint32 {
	switch act {
	case ActivationReLU:
		return 0
	case ActivationSilu:
		return 1
	case ActivationTanh:
		return 3
	case ActivationSigmoid:
		return 4
	case ActivationLeakyReLU:
		return 5
	case ActivationReLU2:
		return 6
	default:
		return 99
	}
}

func (c *WGPUContext) DispatchApplyGradients(size int, lr float32, clipVal float32, weightBuf, gradBuf *wgpu.Buffer) error {
	pipeline, err := c.CreateComputePipeline(ShaderApplyGradients)
	if err != nil {
		return err
	}

	p := WGPUApplyGradientsParams{
		Size:    uint32(size),
		LR:      lr,
		ClipVal: clipVal,
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUApplyGradientsParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, weightBuf, gradBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(size)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchFillZero(size int, buf *wgpu.Buffer) error {
	pipeline, err := c.CreateComputePipeline(ShaderFillZero)
	if err != nil {
		return err
	}

	p := struct{ Size uint32 }{Size: uint32(size)}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]uint32{p.Size}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, buf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(size)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

type WGPUQuantizeParams struct {
	Size  uint32
	Scale float32
	_     [2]uint32
}

func (c *WGPUContext) DispatchQuantizeI8(size int, scale float32, masterBuf, nativeBuf *wgpu.Buffer) error {
	pipeline, err := c.CreateComputePipeline(ShaderQuantizeI8)
	if err != nil {
		return err
	}

	p := WGPUQuantizeParams{Size: uint32(size), Scale: scale}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUQuantizeParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, masterBuf, nativeBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(size/4)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchQuantizeI4(size int, scale float32, masterBuf, nativeBuf *wgpu.Buffer) error {
	pipeline, err := c.CreateComputePipeline(ShaderQuantizeI4)
	if err != nil {
		return err
	}

	p := WGPUQuantizeParams{Size: uint32(size), Scale: scale}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUQuantizeParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, masterBuf, nativeBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(size/8)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchQuantizeFP4(size int, scale float32, masterBuf, nativeBuf *wgpu.Buffer) error {
	pipeline, err := c.CreateComputePipeline(ShaderQuantizeFP4)
	if err != nil {
		return err
	}

	p := WGPUQuantizeParams{Size: uint32(size), Scale: scale}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUQuantizeParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, masterBuf, nativeBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(size/8)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchQuantizeTernary(size int, scale float32, masterBuf, nativeBuf *wgpu.Buffer) error {
	pipeline, err := c.CreateComputePipeline(ShaderQuantizeTernary)
	if err != nil {
		return err
	}

	p := WGPUQuantizeParams{Size: uint32(size), Scale: scale}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUQuantizeParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, masterBuf, nativeBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(size/16)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchQuantizeBinary(size int, scale float32, masterBuf, nativeBuf *wgpu.Buffer) error {
	pipeline, err := c.CreateComputePipeline(ShaderQuantizeBinary)
	if err != nil {
		return err
	}

	p := WGPUQuantizeParams{Size: uint32(size), Scale: scale}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUQuantizeParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, masterBuf, nativeBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(size/32)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

// DispatchMSEGradPartialLoss computes MSE gradients on GPU and writes partial loss sums.
// numWG = ceil(size/256) partial sums are written to partialsBuf. CPU sums them for total loss.
func (c *WGPUContext) DispatchMSEGradPartialLoss(
	size int,
	outputBuf, targetBuf, gradBuf, partialsBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderMSEGradPartialLoss)
	if err != nil {
		return err
	}

	p := WGPULossParams{Size: uint32(size)}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPULossParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, outputBuf, targetBuf, gradBuf, partialsBuf)
	if err != nil {
		return err
	}

	numWG := (uint32(size) + 255) / 256
	if err := c.dispatchCompute(pipeline, bindGroup, numWG, 1, 1); err != nil {
		return err
	}
	return nil
}

// DispatchCEGradPartialLoss computes Cross-Entropy gradients on GPU and writes partial loss sums.
func (c *WGPUContext) DispatchCEGradPartialLoss(
	size int,
	outputBuf, targetBuf, gradBuf, partialsBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderCEGradPartialLoss)
	if err != nil {
		return err
	}

	p := WGPULossParams{Size: uint32(size)}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPULossParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, outputBuf, targetBuf, gradBuf, partialsBuf)
	if err != nil {
		return err
	}

	numWG := (uint32(size) + 255) / 256
	if err := c.dispatchCompute(pipeline, bindGroup, numWG, 1, 1); err != nil {
		return err
	}
	return nil
}

// DispatchMultiHeadSoftmaxCEGradPartialLoss applies softmax + CE independently on three
// contiguous head slices per batch row (e.g. GPS|EACS|sent), writes dL/dlogit, and
// per-workgroup partial sums of total CE (caller should sum partials and divide by batchSize).
func (c *WGPUContext) DispatchMultiHeadSoftmaxCEGradPartialLoss(
	batchSize, rowWidth, h0, h1, h2 int,
	outputBuf, targetBuf, gradBuf, partialsBuf *wgpu.Buffer,
) error {
	if batchSize <= 0 || rowWidth <= 0 || h0 < 0 || h1 < 0 || h2 < 0 || h0+h1+h2 != rowWidth {
		return fmt.Errorf("multi-head softmax CE: invalid dims batch=%d row=%d heads=%d,%d,%d", batchSize, rowWidth, h0, h1, h2)
	}
	pipeline, err := c.CreateComputePipeline(ShaderMultiHeadSoftmaxCEGradPartialLoss)
	if err != nil {
		return err
	}

	p := WGPUMultiHeadSoftmaxCEParams{
		Batch:    uint32(batchSize),
		RowWidth: uint32(rowWidth),
		H0:       uint32(h0),
		H1:       uint32(h1),
		H2:       uint32(h2),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUMultiHeadSoftmaxCEParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, outputBuf, targetBuf, gradBuf, partialsBuf)
	if err != nil {
		return err
	}

	numWG := (uint32(batchSize) + 255) / 256
	if err := c.dispatchCompute(pipeline, bindGroup, numWG, 1, 1); err != nil {
		return err
	}
	return nil
}

// DispatchMultiHeadSoftmaxCEGradPartialLossMasked is like DispatchMultiHeadSoftmaxCEGradPartialLoss but reads
// per-row head masks from headMaskBuf: batchSize*3 floats, row-major [GPS, EACS, Sent], active if > 0.5.
func (c *WGPUContext) DispatchMultiHeadSoftmaxCEGradPartialLossMasked(
	batchSize, rowWidth, h0, h1, h2 int,
	outputBuf, targetBuf, gradBuf, partialsBuf, headMaskBuf *wgpu.Buffer,
) error {
	if batchSize <= 0 || rowWidth <= 0 || h0 < 0 || h1 < 0 || h2 < 0 || h0+h1+h2 != rowWidth {
		return fmt.Errorf("multi-head softmax CE (masked): invalid dims batch=%d row=%d heads=%d,%d,%d", batchSize, rowWidth, h0, h1, h2)
	}
	if headMaskBuf == nil {
		return fmt.Errorf("multi-head softmax CE (masked): headMaskBuf is nil")
	}
	pipeline, err := c.CreateComputePipeline(ShaderMultiHeadSoftmaxCEGradPartialLossMasked)
	if err != nil {
		return err
	}

	p := WGPUMultiHeadSoftmaxCEParams{
		Batch:    uint32(batchSize),
		RowWidth: uint32(rowWidth),
		H0:       uint32(h0),
		H1:       uint32(h1),
		H2:       uint32(h2),
	}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUMultiHeadSoftmaxCEParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, outputBuf, targetBuf, gradBuf, partialsBuf, headMaskBuf)
	if err != nil {
		return err
	}

	numWG := (uint32(batchSize) + 255) / 256
	if err := c.dispatchCompute(pipeline, bindGroup, numWG, 1, 1); err != nil {
		return err
	}
	return nil
}

// GetGPUWeightBuffer returns the GPU weight buffer for a layer using the layer's
// native DType first, then falling back to the Float32 master buffer.
// This lets forward kernels see PTQ-simulated weights (quantize→dequantize)
// while backward kernels can still request DTypeFloat32 explicitly.
func GetGPUWeightBuffer(l *VolumetricLayer) *wgpu.Buffer {
	if l.WeightStore == nil {
		return nil
	}
	// Float64 has no native GPU representation; treat as Float32.
	fwdDType := l.DType
	if fwdDType == DTypeFloat64 {
		fwdDType = DTypeFloat32
	}
	if buf, ok := l.WeightStore.GPUWeights[fwdDType].(*wgpu.Buffer); ok && buf != nil {
		return buf
	}
	if buf, ok := l.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer); ok && buf != nil {
		return buf
	}
	return nil
}

func (c *WGPUContext) DispatchForwardLayer(l *VolumetricLayer, batchSize int, inputBuf, outBuf *wgpu.Buffer) error {
	tileSize := c.GPUTileSize
	if tileSize <= 0 {
		tileSize = 32
	}
	switch l.Type {
	case LayerDense:
		if l.DType == DTypeTernary {
			wBuf, _ := l.WeightStore.GPUWeights[DTypeTernary].(*wgpu.Buffer)
			if wBuf == nil {
				return fmt.Errorf("layer %s at [%d,%d,%d,%d]: missing BitNet ternary GPU weights", l.Type.String(), l.Z, l.Y, l.X, l.L)
			}
			qBuf, scaleBuf, err := bitNetQuantizeActivationGPU(c, fmt.Sprintf("dense_bitnet_%p", l), batchSize, l.InputHeight, inputBuf)
			if err != nil {
				return err
			}
			return c.DispatchDenseBitNetTernaryQuantized(batchSize, l.InputHeight, l.OutputHeight, qBuf, scaleBuf, wBuf, nil, outBuf, bitNetGPUScaleValue(l.WeightStore, DTypeTernary, 0), l.Activation, tileSize)
		}
		if l.DType == DTypeInt4 {
			if scaleBuf, ok := l.WeightStore.GPUScales[DTypeInt4]; ok && scaleBuf != nil {
				if weightBuf, ok := l.WeightStore.GPUWeights[DTypeInt4].(*wgpu.Buffer); ok {
					return c.DispatchDenseQ4(batchSize, l.InputHeight, l.OutputHeight, inputBuf, scaleBuf, weightBuf, outBuf, tileSize)
				}
			}
		}
		wBuf := GetGPUWeightBuffer(l)
		if wBuf == nil {
			return fmt.Errorf("layer %s at [%d,%d,%d,%d]: missing GPU weights", l.Type.String(), l.Z, l.Y, l.X, l.L)
		}
		// Fetch bias buffer if it exists
		var bBuf *wgpu.Buffer
		if l.WeightStore != nil {
			if b, ok := l.WeightStore.GPUWeights[DType(1001)].(*wgpu.Buffer); ok {
				bBuf = b
			}
		}

		// Call the premium tiled dispatcher with the layer's actual scale.
		scale := float32(1.0)
		if l.WeightStore != nil && l.WeightStore.Scale != 0 {
			scale = l.WeightStore.Scale
		}
		if l.Activation == ActivationLinear {
			return c.DispatchDenseTiled(tileSize, batchSize, l.InputHeight, l.OutputHeight, uint32(l.Activation), scale, inputBuf, wBuf, bBuf, outBuf)
		}
		denseOutSize := batchSize * l.OutputHeight
		densePreBuf := c.GetActivationBuffer(fmt.Sprintf("dense_pre_%p", l), uint64(denseOutSize*4), wgpu.BufferUsageStorage)
		linAct := ActivationLinear
		if err := c.DispatchDenseTiled(tileSize, batchSize, l.InputHeight, l.OutputHeight, uint32(linAct), scale, inputBuf, wBuf, bBuf, densePreBuf); err != nil {
			return err
		}
		return c.DispatchActivation(denseOutSize, l.Activation, densePreBuf, outBuf)
	case LayerRMSNorm:
		wBuf := GetGPUWeightBuffer(l)
		if wBuf == nil {
			return fmt.Errorf("layer %s at [%d,%d,%d,%d]: missing GPU weights", l.Type.String(), l.Z, l.Y, l.X, l.L)
		}
		return c.DispatchRMSNorm(batchSize, l.InputHeight, 1e-5, inputBuf, wBuf, outBuf)
	case LayerCNN1:
		wBuf := GetGPUWeightBuffer(l)
		if wBuf == nil {
			return fmt.Errorf("layer %s at [%d,%d,%d,%d]: missing GPU weights", l.Type.String(), l.Z, l.Y, l.X, l.L)
		}
		if isCNN1NativeGPUQuantDType(l.DType) {
			scale := cnn1PackedGPUScale(l)
			if l.Activation == ActivationLinear {
				return c.DispatchCNN1Packed(l.DType, batchSize, l.InputChannels, l.InputHeight, l.Filters, l.OutputHeight, l.KernelSize, l.Stride, l.Padding, scale, inputBuf, wBuf, outBuf)
			}
			cnn1OutSize := batchSize * l.Filters * l.OutputHeight
			cnn1PreBuf := c.GetActivationBuffer(fmt.Sprintf("cnn1_pre_%p", l), uint64(cnn1OutSize*4), wgpu.BufferUsageStorage)
			if err := c.DispatchCNN1Packed(l.DType, batchSize, l.InputChannels, l.InputHeight, l.Filters, l.OutputHeight, l.KernelSize, l.Stride, l.Padding, scale, inputBuf, wBuf, cnn1PreBuf); err != nil {
				return err
			}
			return c.DispatchActivation(cnn1OutSize, l.Activation, cnn1PreBuf, outBuf)
		}
		if l.Activation == ActivationLinear {
			return c.DispatchCNN1(batchSize, l.InputChannels, l.InputHeight, l.Filters, l.OutputHeight, l.KernelSize, l.Stride, l.Padding, inputBuf, wBuf, outBuf)
		}
		cnn1OutSize := batchSize * l.Filters * l.OutputHeight
		cnn1PreBuf := c.GetActivationBuffer(fmt.Sprintf("cnn1_pre_%p", l), uint64(cnn1OutSize*4), wgpu.BufferUsageStorage)
		if err := c.DispatchCNN1(batchSize, l.InputChannels, l.InputHeight, l.Filters, l.OutputHeight, l.KernelSize, l.Stride, l.Padding, inputBuf, wBuf, cnn1PreBuf); err != nil {
			return err
		}
		return c.DispatchActivation(cnn1OutSize, l.Activation, cnn1PreBuf, outBuf)
	case LayerCNN2:
		wBuf := GetGPUWeightBuffer(l)
		if wBuf == nil {
			return fmt.Errorf("layer %s at [%d,%d,%d,%d]: missing GPU weights", l.Type.String(), l.Z, l.Y, l.X, l.L)
		}
		if l.Activation == ActivationLinear {
			return c.DispatchCNN2(batchSize, l.InputChannels, l.InputHeight, l.InputWidth, l.Filters, l.OutputHeight, l.OutputWidth, l.KernelSize, l.KernelSize, l.Stride, l.Stride, l.Padding, l.Padding, inputBuf, wBuf, outBuf)
		}
		cnn2OutSize := batchSize * l.Filters * l.OutputHeight * l.OutputWidth
		cnn2PreBuf := c.GetActivationBuffer(fmt.Sprintf("cnn2_pre_%p", l), uint64(cnn2OutSize*4), wgpu.BufferUsageStorage)
		if err := c.DispatchCNN2(batchSize, l.InputChannels, l.InputHeight, l.InputWidth, l.Filters, l.OutputHeight, l.OutputWidth, l.KernelSize, l.KernelSize, l.Stride, l.Stride, l.Padding, l.Padding, inputBuf, wBuf, cnn2PreBuf); err != nil {
			return err
		}
		return c.DispatchActivation(cnn2OutSize, l.Activation, cnn2PreBuf, outBuf)
	case LayerCNN3:
		wBuf := GetGPUWeightBuffer(l)
		if wBuf == nil {
			return fmt.Errorf("layer %s at [%d,%d,%d,%d]: missing GPU weights", l.Type.String(), l.Z, l.Y, l.X, l.L)
		}
		if l.Activation == ActivationLinear {
			return c.DispatchCNN3(batchSize, l.InputChannels, l.InputDepth, l.InputHeight, l.InputWidth, l.Filters, l.OutputDepth, l.OutputHeight, l.OutputWidth, l.KernelSize, l.KernelSize, l.KernelSize, l.Stride, l.Stride, l.Stride, l.Padding, l.Padding, l.Padding, inputBuf, wBuf, outBuf)
		}
		cnn3OutSize := batchSize * l.Filters * l.OutputDepth * l.OutputHeight * l.OutputWidth
		cnn3PreBuf := c.GetActivationBuffer(fmt.Sprintf("cnn3_pre_%p", l), uint64(cnn3OutSize*4), wgpu.BufferUsageStorage)
		if err := c.DispatchCNN3(batchSize, l.InputChannels, l.InputDepth, l.InputHeight, l.InputWidth, l.Filters, l.OutputDepth, l.OutputHeight, l.OutputWidth, l.KernelSize, l.KernelSize, l.KernelSize, l.Stride, l.Stride, l.Stride, l.Padding, l.Padding, l.Padding, inputBuf, wBuf, cnn3PreBuf); err != nil {
			return err
		}
		return c.DispatchActivation(cnn3OutSize, l.Activation, cnn3PreBuf, outBuf)
	case LayerRNN:
		wBuf := GetGPUWeightBuffer(l)
		ihSize := l.OutputHeight * l.InputHeight
		hhSize := l.OutputHeight * l.OutputHeight
		hPrev := c.GetActivationBuffer(fmt.Sprintf("rnn_hprev_%p", l), uint64(batchSize*l.OutputHeight*4), wgpu.BufferUsageStorage)
		// Inline RNN step with correct sub-buffer bindings for wIH, wHH, bias
		{
			pipeline, err := c.CreateComputePipeline(ShaderRNNStep)
			if err != nil {
				return err
			}
			p := WGPURNNParams{BatchSize: uint32(batchSize), InputSize: uint32(l.InputHeight), HiddenSize: uint32(l.OutputHeight)}
			pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPURNNParams{p}))
			wIHBind := c.GetSubBuffer(wBuf, 0, uint64(ihSize*4))
			wHHBind := c.GetSubBuffer(wBuf, uint64(ihSize*4), uint64(hhSize*4))
			biasBind := c.GetSubBuffer(wBuf, uint64((ihSize+hhSize)*4), uint64(l.OutputHeight*4))
			bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, hPrev, wIHBind, wHHBind, biasBind, outBuf)
			if err != nil {
				return err
			}
			if err := c.dispatchCompute(pipeline, bindGroup, (uint32(l.OutputHeight)+63)/64, uint32(batchSize), 1); err != nil {
				return err
			}
			return nil
		}
	case LayerLSTM:
		weights := GetGPUWeightBuffer(l)
		hPrev := c.GetActivationBuffer(fmt.Sprintf("lstm_hprev_%p", l), uint64(batchSize*l.OutputHeight*4), wgpu.BufferUsageStorage)
		cPrev := c.GetActivationBuffer(fmt.Sprintf("lstm_cprev_%p", l), uint64(batchSize*l.OutputHeight*4), wgpu.BufferUsageStorage)
		cCurr := c.GetActivationBuffer(fmt.Sprintf("lstm_ccurr_%p", l), uint64(batchSize*l.OutputHeight*4), wgpu.BufferUsageStorage)
		lstmPreAct := c.GetActivationBuffer(fmt.Sprintf("lstm_preact_%p", l), uint64(batchSize*5*l.OutputHeight*4), wgpu.BufferUsageStorage)
		return c.DispatchLSTMStepPreAct(batchSize, l.InputHeight, l.OutputHeight, inputBuf, hPrev, cPrev, weights, outBuf, cCurr, lstmPreAct)
	case LayerEmbedding:
		w := GetGPUWeightBuffer(l)
		sl := l.SeqLength
		if sl <= 0 {
			sl = 1
		}
		return c.DispatchEmbedding(l.VocabSize, l.EmbeddingDim, batchSize*sl, inputBuf, w, outBuf)
	case LayerSoftmax:
		temp := l.Temperature
		if temp == 0 {
			temp = 1.0
		}
		return c.DispatchSoftmaxForward(l, batchSize, inputBuf, outBuf)
	case LayerMultiHeadAttention:
		if err := c.partitionMHAWeights(l); err != nil {
			return err
		}
		qWeights, _ := l.WeightStore.GPUWeights[WeightMHAQuery].(*wgpu.Buffer)
		kWeights, _ := l.WeightStore.GPUWeights[WeightMHAKey].(*wgpu.Buffer)
		vWeights, _ := l.WeightStore.GPUWeights[WeightMHAValue].(*wgpu.Buffer)
		oWeights, _ := l.WeightStore.GPUWeights[WeightMHAProjection].(*wgpu.Buffer)
		sl := l.SeqLength
		if sl <= 0 {
			sl = 1
		}
		maxSL := l.MaxSeqLen
		if maxSL <= 0 {
			maxSL = 512
		}
		tileSize := l.GetGPUSCTileSize(l.DType)
		if tileSize <= 0 {
			tileSize = 32
		}

		kvDim := l.NumKVHeads * l.HeadDim
		qDim := l.QueryDim
		if qDim == 0 {
			qDim = l.DModel
		}
		qBuf := c.GetActivationBuffer(fmt.Sprintf("mha_q_%p", l), uint64(batchSize*sl*qDim*4), wgpu.BufferUsageStorage)
		kBuf := c.GetActivationBuffer(fmt.Sprintf("mha_k_%p", l), uint64(batchSize*sl*kvDim*4), wgpu.BufferUsageStorage)
		vBuf := c.GetActivationBuffer(fmt.Sprintf("mha_v_%p", l), uint64(batchSize*sl*kvDim*4), wgpu.BufferUsageStorage)

		// Q, K, V Projections
		if err := c.DispatchDense(batchSize*sl, l.DModel, qDim, inputBuf, qWeights, qBuf, tileSize); err != nil {
			return err
		}
		if err := c.DispatchDense(batchSize*sl, l.DModel, kvDim, inputBuf, kWeights, kBuf, tileSize); err != nil {
			return err
		}
		if err := c.DispatchDense(batchSize*sl, l.DModel, kvDim, inputBuf, vWeights, vBuf, tileSize); err != nil {
			return err
		}
		qkEps := float32(1e-6)
		if l.RMSNormEps > 0 {
			qkEps = float32(l.RMSNormEps)
		}
		if qNormBuf, ok := l.WeightStore.GPUWeights[WeightMHAQNorm].(*wgpu.Buffer); ok && qNormBuf != nil {
			qNormOut := c.GetActivationBuffer(fmt.Sprintf("mha_qn_%p", l), uint64(batchSize*sl*qDim*4), wgpu.BufferUsageStorage)
			if err := c.DispatchRMSNorm(batchSize*sl*l.NumHeads, l.HeadDim, qkEps, qBuf, qNormBuf, qNormOut); err != nil {
				return err
			}
			qBuf = qNormOut
		}
		if kNormBuf, ok := l.WeightStore.GPUWeights[WeightMHAKNorm].(*wgpu.Buffer); ok && kNormBuf != nil {
			kNormOut := c.GetActivationBuffer(fmt.Sprintf("mha_kn_%p", l), uint64(batchSize*sl*kvDim*4), wgpu.BufferUsageStorage)
			if err := c.DispatchRMSNorm(batchSize*sl*l.NumKVHeads, l.HeadDim, qkEps, kBuf, kNormBuf, kNormOut); err != nil {
				return err
			}
			kBuf = kNormOut
		}

		attnOut := c.GetActivationBuffer(fmt.Sprintf("attn_out_%p", l), uint64(batchSize*sl*qDim*4), wgpu.BufferUsageStorage)
		if err := c.DispatchMHA(l.NumHeads, l.NumKVHeads, l.HeadDim, sl, l.KVOffset, maxSL, qBuf, kBuf, vBuf, attnOut, tileSize); err != nil {
			return err
		}
		return c.DispatchDense(batchSize*sl, qDim, l.DModel, attnOut, oWeights, outBuf, tileSize)
	case LayerSwiGLU:
		g, _ := l.WeightStore.GPUWeights[DType(100)].(*wgpu.Buffer)
		u, _ := l.WeightStore.GPUWeights[DType(101)].(*wgpu.Buffer)
		wDown, _ := l.WeightStore.GPUWeights[DType(102)].(*wgpu.Buffer)

		gB, _ := l.WeightStore.GPUWeights[DType(110)].(*wgpu.Buffer)
		uB, _ := l.WeightStore.GPUWeights[DType(111)].(*wgpu.Buffer)
		dB, _ := l.WeightStore.GPUWeights[DType(112)].(*wgpu.Buffer)

		if gB == nil {
			gB = c.BlankBuffer
		}
		if uB == nil {
			uB = c.BlankBuffer
		}
		if dB == nil {
			dB = c.BlankBuffer
		}

		preOut := c.GetActivationBuffer(fmt.Sprintf("preOut_%p", l), uint64(batchSize*l.OutputHeight*4), wgpu.BufferUsageStorage)
		gatePreBuf := c.GetActivationBuffer(fmt.Sprintf("gateIn_%p", l), uint64(batchSize*l.OutputHeight*4), wgpu.BufferUsageStorage)
		upPreBuf := c.GetActivationBuffer(fmt.Sprintf("upIn_%p", l), uint64(batchSize*l.OutputHeight*4), wgpu.BufferUsageStorage)
		if err := c.DispatchSwiGLUWithActCache(batchSize, l.InputHeight, l.OutputHeight, inputBuf, g, u, gB, uB, preOut, gatePreBuf, upPreBuf, 32); err != nil {
			return err
		}
		var act uint32 = 99
		return c.DispatchDenseTiled(32, batchSize, l.OutputHeight, l.InputHeight, act, 1.0, preOut, wDown, dB, outBuf)
	case LayerResidual:
		totalSize := l.InputHeight * batchSize
		c.EndComputePass()
		enc, owned, _ := ctxEncoder(c)
		enc.CopyBufferToBuffer(inputBuf, 0, outBuf, 0, uint64(totalSize*4))
		ctxSubmit(c, enc, owned)
		return c.DispatchResidual(totalSize, outBuf, inputBuf)
	default:
		return fmt.Errorf("GPU forward not implemented for layer %v", l.Type)
	}
}

// propagateSplitWeights copies sub-ranges of the updated master weight buffer back
// to the per-projection split buffers used by SwiGLU and MHA forward passes.
// Must be called within a BeginFrame/FlushFrame block so ActiveEncoder is set.
func (c *WGPUContext) propagateSplitWeights(l *VolumetricLayer, masterBuf *wgpu.Buffer) {
	enc := c.ActiveEncoder
	if enc == nil || l.WeightStore == nil {
		return
	}
	c.EndComputePass()
	cp := func(key DType, srcOffset, size uint64) {
		if b, ok := l.WeightStore.GPUWeights[key].(*wgpu.Buffer); ok && b != nil {
			enc.CopyBufferToBuffer(masterBuf, srcOffset, b, 0, size)
		}
	}
	switch l.Type {
	case LayerSwiGLU:
		h := uint64(l.InputHeight)
		inter := uint64(l.OutputHeight)
		wSz := h * inter * 4
		cp(DType(100), 0, wSz)                 // gate weights
		cp(DType(101), wSz, wSz)               // up weights
		cp(DType(102), 2*wSz, wSz)             // down weights
		cp(DType(110), 3*wSz, inter*4)         // gate bias
		cp(DType(111), 3*wSz+inter*4, inter*4) // up bias
		cp(DType(112), 3*wSz+2*inter*4, h*4)   // down bias
	case LayerMultiHeadAttention:
		d := uint64(l.DModel)
		q := uint64(l.QueryDim)
		if q == 0 {
			q = d
		}
		kvH := uint64(l.NumKVHeads)
		if kvH == 0 {
			kvH = uint64(l.NumHeads)
		}
		kv := kvH * uint64(l.HeadDim)
		qSz := q * d * 4
		kSz := d * kv * 4
		vSz := d * kv * 4
		oSz := d * q * 4
		cp(WeightMHAQuery, 0, qSz)
		cp(WeightMHAKey, qSz, kSz)
		cp(WeightMHAValue, qSz+kSz, vSz)
		cp(WeightMHAProjection, qSz+kSz+vSz, oSz)
	}
}

func (c *WGPUContext) DispatchBackwardLayer(l *VolumetricLayer, batchSize int, gradOutBuf, inputBuf, preActBuf, dxBuf, dwBuf *wgpu.Buffer) error {
	tileSize := c.GPUTileSize
	if tileSize <= 0 {
		tileSize = 32
	}

	switch l.Type {
	case LayerDense:
		var wBuf *wgpu.Buffer
		if l.WeightStore != nil {
			if b, ok := l.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer); ok && b != nil {
				wBuf = b
			}
		}
		if wBuf == nil {
			return fmt.Errorf("layer %s at [%d,%d,%d,%d]: missing GPU weights (dense backward requires Float32 coef buffer)", l.Type.String(), l.Z, l.Y, l.X, l.L)
		}
		// Call the premium tiled backward dispatchers in wgpu_dense_tiled.go
		if err := c.DispatchDenseBackwardDXTiled(tileSize, batchSize, l.InputHeight, l.OutputHeight, uint32(l.Activation), gradOutBuf, wBuf, preActBuf, dxBuf); err != nil {
			return fmt.Errorf("dx: %w", err)
		}
		if err := c.DispatchDenseBackwardDWTiled(tileSize, batchSize, l.InputHeight, l.OutputHeight, uint32(l.Activation), gradOutBuf, inputBuf, preActBuf, dwBuf); err != nil {
			return fmt.Errorf("dw: %w", err)
		}
		return nil
	case LayerRMSNorm:
		rmsBuf := c.GetActivationBuffer(fmt.Sprintf("rms_%p", l), uint64(batchSize*4), wgpu.BufferUsageStorage)
		// Provide 1.0 for dummy validation
		c.Queue.WriteBuffer(rmsBuf, 0, wgpu.ToBytes([]float32{1.0}))
		wBuf := GetGPUWeightBuffer(l)
		if wBuf == nil {
			return fmt.Errorf("layer %s at [%d,%d,%d,%d]: missing GPU weights", l.Type.String(), l.Z, l.Y, l.X, l.L)
		}
		return c.DispatchRMSNormBackward(batchSize, l.InputHeight, 1e-5, gradOutBuf, inputBuf, rmsBuf, wBuf, dxBuf, dwBuf)
	case LayerCNN1:
		if isCNN1NativeGPUQuantDType(l.DType) {
			wBuf := GetGPUWeightBuffer(l)
			if wBuf == nil {
				return fmt.Errorf("layer %s at [%d,%d,%d,%d]: missing GPU weights", l.Type.String(), l.Z, l.Y, l.X, l.L)
			}
			if err := c.DispatchCNN1PackedBackwardDX(l.DType, batchSize, l.InputChannels, l.InputHeight, l.Filters, l.OutputHeight, l.KernelSize, l.Stride, l.Padding, l.Activation, cnn1PackedGPUScale(l), gradOutBuf, wBuf, preActBuf, dxBuf); err != nil {
				return err
			}
			return c.DispatchCNN1BackwardDW(batchSize, l.InputChannels, l.InputHeight, l.Filters, l.OutputHeight, l.KernelSize, l.Stride, l.Padding, l.Activation, gradOutBuf, inputBuf, preActBuf, dwBuf)
		}
		wBuf := GetGPUWeightBuffer(l)
		if wBuf == nil {
			return fmt.Errorf("layer %s at [%d,%d,%d,%d]: missing GPU weights", l.Type.String(), l.Z, l.Y, l.X, l.L)
		}
		if err := c.DispatchCNN1BackwardDX(batchSize, l.InputChannels, l.InputHeight, l.Filters, l.OutputHeight, l.KernelSize, l.Stride, l.Padding, l.Activation, gradOutBuf, wBuf, preActBuf, dxBuf); err != nil {
			return err
		}
		return c.DispatchCNN1BackwardDW(batchSize, l.InputChannels, l.InputHeight, l.Filters, l.OutputHeight, l.KernelSize, l.Stride, l.Padding, l.Activation, gradOutBuf, inputBuf, preActBuf, dwBuf)
	case LayerCNN2:
		wBuf := GetGPUWeightBuffer(l)
		if wBuf == nil {
			return fmt.Errorf("layer %s at [%d,%d,%d,%d]: missing GPU weights", l.Type.String(), l.Z, l.Y, l.X, l.L)
		}
		if err := c.DispatchCNN2BackwardDX(batchSize, l.InputChannels, l.InputHeight, l.InputWidth, l.Filters, l.OutputHeight, l.OutputWidth, l.KernelSize, l.Stride, l.Padding, l.Activation, gradOutBuf, wBuf, preActBuf, dxBuf); err != nil {
			return err
		}
		return c.DispatchCNN2BackwardDW(batchSize, l.InputChannels, l.InputHeight, l.InputWidth, l.Filters, l.OutputHeight, l.OutputWidth, l.KernelSize, l.Stride, l.Padding, l.Activation, gradOutBuf, inputBuf, preActBuf, dwBuf)
	case LayerCNN3:
		wBuf := GetGPUWeightBuffer(l)
		if wBuf == nil {
			return fmt.Errorf("layer %s at [%d,%d,%d,%d]: missing GPU weights", l.Type.String(), l.Z, l.Y, l.X, l.L)
		}
		if err := c.DispatchCNN3BackwardDX(batchSize, l.InputChannels, l.InputDepth, l.InputHeight, l.InputWidth, l.Filters, l.OutputDepth, l.OutputHeight, l.OutputWidth, l.KernelSize, l.Stride, l.Padding, l.Activation, gradOutBuf, wBuf, preActBuf, dxBuf); err != nil {
			return err
		}
		return c.DispatchCNN3BackwardDW(batchSize, l.InputChannels, l.InputDepth, l.InputHeight, l.InputWidth, l.Filters, l.OutputDepth, l.OutputHeight, l.OutputWidth, l.KernelSize, l.Stride, l.Padding, l.Activation, gradOutBuf, inputBuf, preActBuf, dwBuf)
	case LayerSwiGLU:
		return c.dispatchBackwardLayerCPUFallback(l, batchSize, gradOutBuf, inputBuf, dxBuf, dwBuf)
	case LayerRNN:
		wBuf := GetGPUWeightBuffer(l)
		if wBuf == nil {
			return fmt.Errorf("layer %s at [%d,%d,%d,%d]: missing GPU weights", l.Type.String(), l.Z, l.Y, l.X, l.L)
		}
		// hCurrBuf = histPreBuf[i] = post-tanh output; backward uses 1-hCurr^2 for tanh derivative.
		// wBuf starts with wIH at offset 0, so DX can read wIH[h*I+i] correctly from the full buffer.
		hPrevBuf := c.GetActivationBuffer(fmt.Sprintf("rnn_hprev_%p", l), uint64(batchSize*l.OutputHeight*4), wgpu.BufferUsageStorage)
		if err := c.DispatchRNNBackwardDX(tileSize, batchSize, l.InputHeight, l.OutputHeight, gradOutBuf, wBuf, preActBuf, dxBuf); err != nil {
			return fmt.Errorf("rnn dx: %w", err)
		}
		return c.DispatchRNNBackwardDW(tileSize, batchSize, l.InputHeight, l.OutputHeight, gradOutBuf, inputBuf, preActBuf, hPrevBuf, dwBuf)
	case LayerLSTM:
		wBuf := GetGPUWeightBuffer(l)
		if wBuf == nil {
			return fmt.Errorf("layer %s at [%d,%d,%d,%d]: missing GPU weights", l.Type.String(), l.Z, l.Y, l.X, l.L)
		}
		lstmPreAct := c.GetActivationBuffer(fmt.Sprintf("lstm_preact_%p", l), uint64(batchSize*5*l.OutputHeight*4), wgpu.BufferUsageStorage)
		hPrevBuf := c.GetActivationBuffer(fmt.Sprintf("lstm_hprev_%p", l), uint64(batchSize*l.OutputHeight*4), wgpu.BufferUsageStorage)
		if err := c.DispatchLSTMBackwardDX(tileSize, batchSize, l.InputHeight, l.OutputHeight, gradOutBuf, wBuf, lstmPreAct, dxBuf); err != nil {
			return fmt.Errorf("lstm dx: %w", err)
		}
		return c.DispatchLSTMBackwardDW(tileSize, batchSize, l.InputHeight, l.OutputHeight, gradOutBuf, inputBuf, lstmPreAct, hPrevBuf, dwBuf)
	case LayerEmbedding:
		// Embedding input (token indices) is already on GPU as uint32 if uploaded during trainBatchGPU.
		// No need to ReadBuffer/re-upload; just use inputBuf directly.
		sl := l.SeqLength
		if sl <= 0 {
			sl = 1
		}
		return c.DispatchEmbeddingBackward(l.VocabSize, l.EmbeddingDim, sl*batchSize, inputBuf, gradOutBuf, dwBuf)
	case LayerSoftmax:
		temp := l.Temperature
		if temp == 0 {
			temp = 1.0
		}
		// Softmax backward needs gradOutput and its own output (preActBuf)
		return c.DispatchSoftmaxBackward(batchSize, l.OutputHeight, float32(temp), gradOutBuf, preActBuf, dxBuf)
	case LayerResidual:
		return c.DispatchResidualBackward(l.InputHeight*batchSize, gradOutBuf, dxBuf, dwBuf)
	case LayerMultiHeadAttention:
		return c.dispatchBackwardLayerCPUFallback(l, batchSize, gradOutBuf, inputBuf, dxBuf, dwBuf)
	default:
		return fmt.Errorf("GPU backward not implemented for layer %v", l.Type)
	}
}

func (c *WGPUContext) dispatchBackwardLayerCPUFallback(l *VolumetricLayer, batchSize int, gradOutBuf, inputBuf, dxBuf, dwBuf *wgpu.Buffer) error {
	if c.ActiveEncoder != nil {
		return fmt.Errorf("cpu fallback unavailable while a GPU frame is active for layer %v", l.Type)
	}

	inputData, err := c.ReadBuffer(inputBuf)
	if err != nil {
		return err
	}
	gradData, err := c.ReadBuffer(gradOutBuf)
	if err != nil {
		return err
	}

	fallback := *l
	fallback.UseGPU = false
	fallback.IsGPUResident = false
	fallback.IsKVCacheGPUResident = false
	fallback.KVOffset = 0
	fallback.KVCacheK = nil
	fallback.KVCacheV = nil

	inputShape := fallbackInputShape(&fallback, batchSize, len(inputData))
	inputTensor := NewTensorFromSlice(inputData, inputShape...)
	gradTensor := NewTensorFromSlice(gradData, fallbackOutputShape(&fallback, batchSize, len(gradData))...)

	preAct, _ := DispatchLayer(&fallback, inputTensor, nil)
	gradInput, gradWeights := DispatchLayerBackward(&fallback, gradTensor, inputTensor, nil, preAct)
	if gradInput == nil || gradWeights == nil {
		return fmt.Errorf("cpu fallback produced nil gradients for layer %v", l.Type)
	}

	ctxDX := ConvertTensor[float32, float32](gradInput)
	ctxDW := ConvertTensor[float32, float32](gradWeights)
	c.Queue.WriteBuffer(dxBuf, 0, wgpu.ToBytes(ctxDX.Data))
	c.Queue.WriteBuffer(dwBuf, 0, wgpu.ToBytes(ctxDW.Data))
	return nil
}

func fallbackInputShape(l *VolumetricLayer, batchSize int, dataLen int) []int {
	switch l.Type {
	case LayerMultiHeadAttention:
		sl := l.SeqLength
		if sl <= 0 {
			sl = 1
		}
		if batchSize*sl*l.DModel == dataLen {
			return []int{batchSize, sl, l.DModel}
		}
	case LayerSwiGLU:
		if batchSize*l.InputHeight == dataLen {
			return []int{batchSize, l.InputHeight}
		}
	}
	return []int{batchSize, max(1, dataLen/max(1, batchSize))}
}

func fallbackOutputShape(l *VolumetricLayer, batchSize int, dataLen int) []int {
	switch l.Type {
	case LayerMultiHeadAttention:
		sl := l.SeqLength
		if sl <= 0 {
			sl = 1
		}
		if batchSize*sl*l.DModel == dataLen {
			return []int{batchSize, sl, l.DModel}
		}
	case LayerSwiGLU:
		if batchSize*l.InputHeight == dataLen {
			return []int{batchSize, l.InputHeight}
		}
	}
	return []int{batchSize, max(1, dataLen/max(1, batchSize))}
}

func (c *WGPUContext) DispatchActivation(size int, act ActivationType, inputBuf, outputBuf *wgpu.Buffer) error {
	if act == ActivationLinear {
		c.EndComputePass()
		enc, owned, _ := ctxEncoder(c)
		enc.CopyBufferToBuffer(inputBuf, 0, outputBuf, 0, uint64(size*4))
		ctxSubmit(c, enc, owned)
		return nil
	}
	pipeline, err := c.CreateComputePipeline(ShaderActivationForward)
	if err != nil {
		return err
	}

	p := WGPUActivationParams{Size: uint32(size), Act: uint32(act)}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUActivationParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, outputBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(size)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}

func (c *WGPUContext) DispatchActivationBackward(size int, act ActivationType, gradOutBuf, preActBuf, gradInBuf *wgpu.Buffer) error {
	if act == ActivationLinear {
		// gradInput = gradOutput
		// We could do a copy, but ideally we'd just reuse the buffer.
		// For now, let's skip as we assume linear doesn't need a kernel.
		return nil
	}
	pipeline, err := c.CreateComputePipeline(ShaderActivationBackward)
	if err != nil {
		return err
	}

	p := WGPUActivationParams{Size: uint32(size), Act: uint32(act)}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUActivationParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutBuf, preActBuf, gradInBuf)
	if err != nil {
		return err
	}

	if err := c.dispatchCompute(pipeline, bindGroup, (uint32(size)+63)/64, 1, 1); err != nil {
		return err
	}
	return nil
}
func (c *WGPUContext) partitionMHAWeights(l *VolumetricLayer) error {
	if l.WeightStore == nil {
		return nil
	}
	if _, ok := l.WeightStore.GPUWeights[WeightMHAQuery]; ok {
		return nil
	}

	dModel := l.DModel
	qDim := l.QueryDim
	if qDim == 0 {
		qDim = dModel
	}
	numKV := l.NumKVHeads
	if numKV == 0 {
		numKV = l.NumHeads
	}
	kvDim := numKV * l.HeadDim

	// offsets based on mha.go
	qwStart := 0
	kwStart := qDim * dModel
	vwStart := qDim*dModel + dModel*kvDim
	owStart := qDim*dModel + 2*dModel*kvDim

	data := l.WeightStore.Master
	if len(data) < owStart+dModel*qDim {
		return fmt.Errorf("insufficient MHA master weights: %d < %d", len(data), owStart+dModel*qDim)
	}

	l.WeightStore.GPUWeights[WeightMHAQuery], _ = c.CreatePersistentBuffer(data[qwStart:qwStart+qDim*dModel], "mha_q_w")
	l.WeightStore.GPUWeights[WeightMHAKey], _ = c.CreatePersistentBuffer(data[kwStart:kwStart+dModel*kvDim], "mha_k_w")
	l.WeightStore.GPUWeights[WeightMHAValue], _ = c.CreatePersistentBuffer(data[vwStart:vwStart+dModel*kvDim], "mha_v_w")
	l.WeightStore.GPUWeights[WeightMHAProjection], _ = c.CreatePersistentBuffer(data[owStart:owStart+dModel*qDim], "mha_o_w")
	if len(l.QNormWeight) > 0 {
		l.WeightStore.GPUWeights[WeightMHAQNorm], _ = c.CreatePersistentBuffer(l.QNormWeight, "mha_q_norm_w")
	}
	if len(l.KNormWeight) > 0 {
		l.WeightStore.GPUWeights[WeightMHAKNorm], _ = c.CreatePersistentBuffer(l.KNormWeight, "mha_k_norm_w")
	}

	return nil
}
