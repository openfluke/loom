package poly

import (
	"fmt"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

type WGPUCNN1BackwardScaleParams struct {
	BatchSize  uint32
	InC        uint32
	InL        uint32
	Filters    uint32
	OutL       uint32
	KSize      uint32
	Stride     uint32
	Padding    uint32
	Activation uint32
	Scale      float32
	Pad1       uint32
	Pad2       uint32
	Pad3       uint32
	Pad4       uint32
	Pad5       uint32
	Pad6       uint32
}

func cnn1PackedGPUScale(l *VolumetricLayer) float32 {
	if l == nil || l.WeightStore == nil {
		return 1.0
	}
	if l.WeightStore.Scale != 0 {
		return l.WeightStore.Scale
	}
	return 1.0
}

func cnn1PackedWGSLSpec(dtype DType) (bits int, weightsPerWord int, mask uint32, decodeFn string, err error) {
	switch dtype {
	case DTypeInt8:
		return 8, 4, 0xFF, `
fn decodeWeight(code: u32) -> f32 {
    var q = i32(code & 0xFFu);
    if (q > 127) { q -= 256; }
    return f32(q);
}
`, nil
	case DTypeInt4:
		return 4, 8, 0x0F, `
fn decodeWeight(code: u32) -> f32 {
    var q = i32(code & 0xFu);
    if (q > 7) { q -= 16; }
    return f32(q);
}
`, nil
	case DTypeInt2:
		return 2, 16, 0x03, `
fn decodeWeight(code: u32) -> f32 {
    var q = i32(code & 0x3u);
    if (q > 1) { q -= 4; }
    return f32(q);
}
`, nil
	case DTypeFP4:
		return 4, 8, 0x0F, `
fn decodeWeight(code: u32) -> f32 {
    let c = code & 0xFu;
    if (c == 0u || c == 8u) { return 0.0; }
    if (c == 1u) { return 0.75; }
    if (c == 2u) { return 1.0; }
    if (c == 3u) { return 1.5; }
    if (c == 4u) { return 2.0; }
    if (c == 5u) { return 3.0; }
    if (c == 9u) { return -0.75; }
    if (c == 10u) { return -1.0; }
    if (c == 11u) { return -1.5; }
    if (c == 12u) { return -2.0; }
    if (c == 13u) { return -3.0; }
    return 0.0;
}
`, nil
	case DTypeTernary:
		return 2, 16, 0x03, `
fn decodeWeight(code: u32) -> f32 {
    if (code == 0u) { return -1.0; }
    if (code == 1u) { return 0.0; }
    if (code == 2u) { return 1.0; }
    return 0.0;
}
`, nil
	case DTypeBinary:
		return 1, 32, 0x01, `
fn decodeWeight(code: u32) -> f32 {
    if ((code & 0x1u) == 0u) { return -1.0; }
    return 1.0;
}
`, nil
	case DTypeUint8:
		return 8, 4, 0xFF, `
fn decodeWeight(code: u32) -> f32 {
    return f32(code & 0xFFu);
}
`, nil
	case DTypeUint4:
		return 4, 8, 0x0F, `
fn decodeWeight(code: u32) -> f32 {
    return f32(code & 0xFu);
}
`, nil
	case DTypeUint2:
		return 2, 16, 0x03, `
fn decodeWeight(code: u32) -> f32 {
    return f32(code & 0x3u);
}
`, nil
	case DTypeInt16:
		return 16, 2, 0xFFFF, `
fn decodeWeight(code: u32) -> f32 {
    var q = i32(code & 0xFFFFu);
    if (q > 32767) { q -= 65536; }
    return f32(q);
}
`, nil
	case DTypeFloat16:
		return 16, 2, 0xFFFF, `
fn decodeWeight(code: u32) -> f32 {
    let bits = code & 0xFFFFu;
    let sign = (bits >> 15u) & 0x1u;
    let exp  = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;
    if (exp == 0u) {
        if (mant == 0u) { return select(0.0, -0.0, sign == 1u); }
        // Denormal (approximate as very small)
        return select(1.0, -1.0, sign == 1u) * f32(mant) * 5.9604644775390625e-8;
    }
    if (exp == 31u) { return 0.0; } // NaN/Inf to 0.0 for training stability
    let resExp = exp + 112u; // 127 - 15
    return bitcast<f32>((sign << 31u) | (resExp << 23u) | (mant << 13u));
}
`, nil
	case DTypeBFloat16:
		return 16, 2, 0xFFFF, `
fn decodeWeight(code: u32) -> f32 {
    return bitcast<f32>((code & 0xFFFFu) << 16u);
}
`, nil
	case DTypeInt32:
		return 32, 1, 0xFFFFFFFF, `
fn decodeWeight(code: u32) -> f32 {
    return f32(i32(code));
}
`, nil
	case DTypeUint32:
		return 32, 1, 0xFFFFFFFF, `
fn decodeWeight(code: u32) -> f32 {
    return f32(code);
}
`, nil
	case DTypeUint16:
		return 16, 2, 0xFFFF, `
fn decodeWeight(code: u32) -> f32 {
    return f32(code & 0xFFFFu);
}
`, nil
	case DTypeInt64, DTypeUint64:
		// 64-bit is special: handled via index doubling in unpackWeight
		return 64, 0, 0, `
fn decodeWeight64(low: u32, high: u32) -> f32 {
    // Reconstruct as much as possible into f32.
    // We only have 24 bits of mantissa in f32, so we lose precision for very large ints.
    // For weights, they are likely in a reasonable range.
    let is_negative = (high >> 31u) == 1u;
    if (high == 0u) { return f32(low); }
    if (high == 0xFFFFFFFFu && low != 0u) {
        // Simple negative handling
        return -f32(0xFFFFFFFFu - low + 1u);
    }
    // General case (approximate)
    return f32(i32(high)) * 4294967296.0 + f32(low);
}
`, nil
	case DTypeFP8E4M3:
		return 8, 4, 0xFF, `
fn decodeWeight(code: u32) -> f32 {
    let sign = (code >> 7u) & 0x1u;
    let exp  = (code >> 3u) & 0xFu;
    let mant = code & 0x7u;
    if (exp == 0u && mant == 0u) { return 0.0; }
    let resExp = exp + 127u - 7u;
    return bitcast<f32>((sign << 31u) | (resExp << 23u) | (mant << 20u));
}
`, nil
	case DTypeFP8E5M2:
		return 8, 4, 0xFF, `
fn decodeWeight(code: u32) -> f32 {
    let sign = (code >> 7u) & 0x1u;
    let exp  = (code >> 2u) & 0x1Fu;
    let mant = code & 0x3u;
    if (exp == 0u && mant == 0u) { return 0.0; }
    if (exp == 31u) { return 0.0; }
    let resExp = exp + 127u - 15u;
    return bitcast<f32>((sign << 31u) | (resExp << 23u) | (mant << 21u));
}
`, nil
	default:
		return 0, 0, 0, "", fmt.Errorf("cnn1 packed shader unsupported for dtype %v", dtype)
	}
}

func unpackWeightWGSL(bits, weightsPerWord int, mask uint32) string {
	if bits == 64 {
		return `fn unpackWeight(idx: u32) -> f32 {
    return decodeWeight64(weights[idx * 2u], weights[idx * 2u + 1u]);
}`
	}
	return fmt.Sprintf(`fn unpackWeight(idx: u32) -> f32 {
    let wordIdx = idx / %du;
    let shift = (idx %% %du) * %du;
    let code = (weights[wordIdx] >> shift) & 0x%Xu;
    return decodeWeight(code);
}`, weightsPerWord, weightsPerWord, bits, mask)
}

func shaderCNN1Packed(dtype DType) (string, error) {
	bits, weightsPerWord, mask, decodeFn, err := cnn1PackedWGSLSpec(dtype)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf(`
struct CNN1ScaleParams {
    batchSize: u32,
    inC: u32, inL: u32,
    outC: u32, outL: u32,
    kSize: u32, stride: u32, padding: u32,
    scale: f32,
    _p1: u32, _p2: u32, _p3: u32, _p4: u32, _p5: u32, _p6: u32, _p7: u32,
};

@group(0) @binding(0) var<uniform>             params:  CNN1ScaleParams;
@group(0) @binding(1) var<storage, read>       input:   array<f32>;
@group(0) @binding(2) var<storage, read>       weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> output:  array<f32>;

%s

%s

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outPos    = global_id.x;
    let filterIdx = global_id.y;
    let batchIdx  = global_id.z;
    if (outPos >= params.outL || filterIdx >= params.outC || batchIdx >= params.batchSize) { return; }

    var sum: f32 = 0.0;
    for (var ic: u32 = 0u; ic < params.inC; ic++) {
        for (var k: u32 = 0u; k < params.kSize; k++) {
            let inPos = i32(outPos * params.stride + k) - i32(params.padding);
            if (inPos >= 0 && u32(inPos) < params.inL) {
                let inIdx = batchIdx * params.inC * params.inL + ic * params.inL + u32(inPos);
                let wIdx = filterIdx * params.inC * params.kSize + ic * params.kSize + k;
                sum += input[inIdx] * unpackWeight(wIdx);
            }
        }
    }
    output[batchIdx * params.outC * params.outL + filterIdx * params.outL + outPos] = sum * params.scale;
}
`, decodeFn, unpackWeightWGSL(bits, weightsPerWord, mask)), nil
}

func shaderTiledCNN1Packed(dtype DType, tileSize, kernelVol int) (string, error) {
	bits, weightsPerWord, mask, decodeFn, err := cnn1PackedWGSLSpec(dtype)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf(`
struct CNN1ScaleParams {
    batchSize: u32,
    inC: u32, inL: u32,
    outC: u32, outL: u32,
    kSize: u32, stride: u32, padding: u32,
    scale: f32,
    _p1: u32, _p2: u32, _p3: u32, _p4: u32, _p5: u32, _p6: u32, _p7: u32,
};

@group(0) @binding(0) var<uniform>             params:  CNN1ScaleParams;
@group(0) @binding(1) var<storage, read>       input:   array<f32>;
@group(0) @binding(2) var<storage, read>       weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> output:  array<f32>;

var<workgroup> wCache: array<f32, %d>;

%s

%s

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id:  vec3<u32>,
) {
    let filterIdx = global_id.y;
    let batchIdx  = global_id.z;
    let kVol: u32 = %du;
    let wBase = filterIdx * kVol;

    var i: u32 = local_id.x;
    loop {
        if (i >= kVol) { break; }
        wCache[i] = unpackWeight(wBase + i);
        i += %du;
    }
    workgroupBarrier();

    let outPos = global_id.x;
    if (outPos >= params.outL || filterIdx >= params.outC || batchIdx >= params.batchSize) { return; }

    var sum: f32 = 0.0;
    for (var ic: u32 = 0u; ic < params.inC; ic++) {
        for (var k: u32 = 0u; k < params.kSize; k++) {
            let inPos = i32(outPos * params.stride + k) - i32(params.padding);
            if (inPos >= 0 && u32(inPos) < params.inL) {
                let inIdx = batchIdx * params.inC * params.inL + ic * params.inL + u32(inPos);
                let cacheIdx = ic * params.kSize + k;
                sum += input[inIdx] * wCache[cacheIdx];
            }
        }
    }
    output[batchIdx * params.outC * params.outL + filterIdx * params.outL + outPos] = sum * params.scale;
}
`, kernelVol, decodeFn, unpackWeightWGSL(bits, weightsPerWord, mask), tileSize, kernelVol, uint32(tileSize)), nil
}

func shaderCNN1PackedBackwardDX(dtype DType) (string, error) {
	bits, weightsPerWord, mask, decodeFn, err := cnn1PackedWGSLSpec(dtype)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf(`
struct Params {
    batchSize:  u32,
    inC:        u32,
    inL:        u32,
    filters:    u32,
    outL:       u32,
    kSize:      u32,
    stride:     u32,
    padding:    u32,
    activation: u32,
    scale:      f32,
    _p1:        u32,
    _p2:        u32,
    _p3:        u32,
    _p4:        u32,
    _p5:        u32,
    _p6:        u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> gradInput: array<f32>;
@group(0) @binding(4) var<storage, read> preAct: array<f32>;

%s

%s

`+wgslBwdActivateDeriv+`

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid >= params.batchSize * params.inC * params.inL) { return; }

    let b = tid / (params.inC * params.inL);
    let rem = tid %% (params.inC * params.inL);
    let ic = rem / params.inL;
    let ip = rem %% params.inL;

    var sum: f32 = 0.0;
    for (var f: u32 = 0u; f < params.filters; f++) {
        for (var k: u32 = 0u; k < params.kSize; k++) {
            let val = i32(ip) + i32(params.padding) - i32(k);
            if (val >= 0 && val %% i32(params.stride) == 0) {
                let o = u32(val / i32(params.stride));
                if (o < params.outL) {
                    let outIdx = b * params.filters * params.outL + f * params.outL + o;
                    let dy = gradOutput[outIdx] * activateDerivative(preAct[outIdx], params.activation);
                    let wIdx = f * params.inC * params.kSize + ic * params.kSize + k;
                    sum += dy * unpackWeight(wIdx);
                }
            }
        }
    }
    gradInput[tid] += sum * params.scale;
}
`, decodeFn, unpackWeightWGSL(bits, weightsPerWord, mask)), nil
}

func shaderTiledCNN1PackedBackwardDX(dtype DType, tileSize, kernelVol int, scale float32) (string, error) {
	bits, weightsPerWord, mask, decodeFn, err := cnn1PackedWGSLSpec(dtype)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf(wgslCNN1Bwd1DParamsStruct+`
@group(0) @binding(0) var<uniform>             params:     CNN1Bwd1DParams;
@group(0) @binding(1) var<storage, read>       gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read>       weights:    array<u32>;
@group(0) @binding(3) var<storage, read>       preAct:     array<f32>;
@group(0) @binding(4) var<storage, read_write> gradInput:  array<f32>;

var<workgroup> wCache: array<f32, %d>;
const PACKED_SCALE: f32 = %.8ff;

%s

%s

`+wgslBwdActivateDeriv+`

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id:  vec3<u32>,
) {
    let inElemFlat = global_id.x;
    let batchIdx = global_id.z;
    let kVol: u32 = %du;
    let inVol = params.inC * params.inL;
    if (batchIdx >= params.batchSize) { return; }

    let ic = inElemFlat / params.inL;
    let inPos = inElemFlat %% params.inL;
    var sum: f32 = 0.0;

    for (var f: u32 = 0u; f < params.filters; f++) {
        var i: u32 = local_id.x;
        loop {
            if (i >= kVol) { break; }
            wCache[i] = unpackWeight(f * kVol + i);
            i += %du;
        }
        workgroupBarrier();

        if (inElemFlat < inVol) {
            for (var k: u32 = 0u; k < params.kSize; k++) {
                let v = i32(inPos) + i32(params.padding) - i32(k);
                if (v >= 0 && v %% i32(params.stride) == 0) {
                    let outPos = u32(v / i32(params.stride));
                    if (outPos < params.outL) {
                        let outIdx = (batchIdx * params.filters + f) * params.outL + outPos;
                        let dy = gradOutput[outIdx] * activateDerivative(preAct[outIdx], params.activation);
                        let wCacheIdx = ic * params.kSize + k;
                        sum += dy * wCache[wCacheIdx];
                    }
                }
            }
        }
        workgroupBarrier();
    }

    if (inElemFlat < inVol) {
        gradInput[batchIdx * inVol + inElemFlat] += sum * PACKED_SCALE;
    }
}
`, tileSize, scale, decodeFn, unpackWeightWGSL(bits, weightsPerWord, mask), tileSize, kernelVol, uint32(tileSize)), nil
}

func (c *WGPUContext) DispatchCNN1Packed(
	dtype DType,
	batchSize, inC, inL, outC, outL, kSize, stride, padding int,
	scale float32,
	inputBuf, weightBuf, outputBuf *wgpu.Buffer,
) error {
	shaderSrc, err := shaderCNN1Packed(dtype)
	if err != nil {
		return err
	}
	pipeline, err := c.CreateComputePipeline(shaderSrc)
	if err != nil {
		return err
	}

	p := WGPUCNN1ScaleParams{
		BatchSize: uint32(batchSize),
		InC:       uint32(inC),
		InL:       uint32(inL),
		OutC:      uint32(outC),
		OutL:      uint32(outL),
		KSize:     uint32(kSize),
		Stride:    uint32(stride),
		Padding:   uint32(padding),
		Scale:     scale,
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUCNN1ScaleParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, weightBuf, outputBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups((uint32(outL)+63)/64, uint32(outC), uint32(batchSize))
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

func (c *WGPUContext) DispatchCNN1PackedTiled(
	dtype DType,
	tileSize, kernelVol int,
	batchSize, inC, inL, outC, outL, kSize, stride, padding int,
	scale float32,
	inputBuf, weightBuf, outputBuf *wgpu.Buffer,
) error {
	shaderSrc, err := shaderTiledCNN1Packed(dtype, tileSize, kernelVol)
	if err != nil {
		return err
	}
	pipeline, err := c.CreateComputePipeline(shaderSrc)
	if err != nil {
		return err
	}

	p := WGPUCNN1ScaleParams{
		BatchSize: uint32(batchSize),
		InC:       uint32(inC),
		InL:       uint32(inL),
		OutC:      uint32(outC),
		OutL:      uint32(outL),
		KSize:     uint32(kSize),
		Stride:    uint32(stride),
		Padding:   uint32(padding),
		Scale:     scale,
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUCNN1ScaleParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, weightBuf, outputBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups((uint32(outL)+uint32(tileSize)-1)/uint32(tileSize), uint32(outC), uint32(batchSize))
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

func (c *WGPUContext) DispatchCNN1PackedBackwardDX(
	dtype DType,
	batchSize, inC, inL, filters, outL, kSize, stride, padding int,
	activation ActivationType,
	scale float32,
	gradOutputBuf, weightBuf, preActBuf, gradInputBuf *wgpu.Buffer,
) error {
	shaderSrc, err := shaderCNN1PackedBackwardDX(dtype)
	if err != nil {
		return err
	}
	pipeline, err := c.CreateComputePipeline(shaderSrc)
	if err != nil {
		return err
	}

	p := WGPUCNN1BackwardScaleParams{
		BatchSize:  uint32(batchSize),
		InC:        uint32(inC),
		InL:        uint32(inL),
		Filters:    uint32(filters),
		OutL:       uint32(outL),
		KSize:      uint32(kSize),
		Stride:     uint32(stride),
		Padding:    uint32(padding),
		Activation: mapActivation(activation),
		Scale:      scale,
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUCNN1BackwardScaleParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, weightBuf, gradInputBuf, preActBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups((uint32(batchSize*inC*inL)+63)/64, 1, 1)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

func (c *WGPUContext) DispatchCNN1PackedBackwardDXTiled(
	dtype DType,
	tileSize, kernelVol int,
	batchSize, inC, inL, filters, outL, kSize, stride, padding int,
	activation ActivationType,
	scale float32,
	gradOutputBuf, weightBuf, preActBuf, gradInputBuf *wgpu.Buffer,
) error {
	shaderSrc, err := shaderTiledCNN1PackedBackwardDX(dtype, tileSize, kernelVol, scale)
	if err != nil {
		return err
	}
	pipeline, err := c.CreateComputePipeline(shaderSrc)
	if err != nil {
		return err
	}

	p := WGPUCNN1Bwd1DParams{
		BatchSize:  uint32(batchSize),
		InC:        uint32(inC),
		InL:        uint32(inL),
		Filters:    uint32(filters),
		OutL:       uint32(outL),
		KSize:      uint32(kSize),
		Stride:     uint32(stride),
		Padding:    uint32(padding),
		Activation: mapActivation(activation),
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUCNN1Bwd1DParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, weightBuf, preActBuf, gradInputBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups((uint32(inC*inL)+uint32(tileSize)-1)/uint32(tileSize), 1, uint32(batchSize))
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}
