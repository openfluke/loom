package poly

import (
	"fmt"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// WGPUCNN2Bwd2DParams is the uniform struct for the tiled CNN2 backward shaders.
// Uses separate kH/kW to support asymmetric kernels, matching the forward convention.
// 16 × uint32/float32 = 64 bytes (multiple of 16 for WebGPU alignment).
type WGPUCNN2Bwd2DParams struct {
	BatchSize        uint32
	InC, InH, InW   uint32
	Filters          uint32
	OutH, OutW      uint32
	KH, KW          uint32
	SH, SW          uint32
	PH, PW          uint32
	Activation      uint32
	Pad             uint32
}

// wgslCNN2Bwd2DParamsStruct is the WGSL struct definition shared by all tiled
// CNN2 backward shaders — avoids repeating field declarations in every generator.
const wgslCNN2Bwd2DParamsStruct = `
struct CNN2Bwd2DParams {
    batchSize: u32,
    inC: u32, inH: u32, inW: u32,
    filters: u32, outH: u32, outW: u32,
    kH: u32, kW: u32,
    sH: u32, sW: u32,
    pH: u32, pW: u32,
    activation: u32, _pad: u32,
};`

// ShaderTiledCNN2BackwardDX generates a tiled gradient-w.r.t-input (dX) WGSL shader.
//
// Tiling strategy — mirrors ShaderTiledCNN2 for the forward pass:
//
//	Each workgroup covers tileSize input elements (one per thread) for one batch item.
//	For every output filter the tileSize threads cooperatively load that filter's full
//	kernel (kernelVol = inC × kH × kW weights) into fast workgroup shared memory,
//	then each thread accumulates its input element's gradient contribution from that
//	filter.  The cooperative load reduces global weight reads by a factor of tileSize.
//
//	kernelVol — inC * kH * kW. Baked into var<workgroup> wCache array size.
//	tileSize  — @workgroup_size(tileSize, 1, 1).
//
// Dispatch: X=ceil(inC*inH*inW/tileSize), Y=1, Z=batchSize
func ShaderTiledCNN2BackwardDX(tileSize, kernelVol int) string {
	return fmt.Sprintf(wgslCNN2Bwd2DParamsStruct+`
@group(0) @binding(0) var<uniform>             params:     CNN2Bwd2DParams;
@group(0) @binding(1) var<storage, read>       gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read>       weights:    array<f32>;
@group(0) @binding(3) var<storage, read>       preAct:     array<f32>;
@group(0) @binding(4) var<storage, read_write> gradInput:  array<f32>;

// Workgroup kernel cache — holds one filter's weights.
// Size baked at generation time: kernelVol = inC * kH * kW.
var<workgroup> wCache: array<f32, %d>;
`+wgslBwdActivateDeriv+`

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id:  vec3<u32>,
) {
    let inElemFlat = global_id.x;
    let batchIdx   = global_id.z;
    let kVol: u32  = %du;
    let inHW       = params.inH * params.inW;
    let inVol      = params.inC * inHW;
    if (batchIdx >= params.batchSize) { return; }

    // Decode flat input position -> (ic, ih, iw)
    let ic  = inElemFlat / inHW;
    let rem = inElemFlat %% inHW;
    let ih  = rem / params.inW;
    let iw  = rem %% params.inW;

    let oArea = params.outH * params.outW;
    var sum: f32 = 0.0;

    // Loop over all filters; cooperative kernel cache load per filter.
    for (var f: u32 = 0u; f < params.filters; f++) {
        // Step 1 — all tileSize threads load filter f's kernel cooperatively.
        var i: u32 = local_id.x;
        loop {
            if (i >= kVol) { break; }
            wCache[i] = weights[f * kVol + i];
            i += %du;
        }
        workgroupBarrier();

        // Step 2 — each thread accumulates its input element's gradient from filter f.
        if (inElemFlat < inVol) {
            for (var kh: u32 = 0u; kh < params.kH; kh++) {
                for (var kw: u32 = 0u; kw < params.kW; kw++) {
                    let vh = i32(ih) + i32(params.pH) - i32(kh);
                    let vw = i32(iw) + i32(params.pW) - i32(kw);
                    if (vh >= 0 && vh %% i32(params.sH) == 0 &&
                        vw >= 0 && vw %% i32(params.sW) == 0) {
                        let oh = u32(vh / i32(params.sH));
                        let ow = u32(vw / i32(params.sW));
                        if (oh < params.outH && ow < params.outW) {
                            let outIdx = (batchIdx * params.filters + f) * oArea + oh * params.outW + ow;
                            let dy = gradOutput[outIdx] * activateDerivative(preAct[outIdx], params.activation);
                            // wCache holds weight[f, :, :, :]; index by (ic, kh, kw).
                            let wCacheIdx = ic * params.kH * params.kW + kh * params.kW + kw;
                            sum += dy * wCache[wCacheIdx];
                        }
                    }
                }
            }
        }
        workgroupBarrier(); // before wCache is reloaded for next filter
    }

    if (inElemFlat < inVol) {
        gradInput[batchIdx * inVol + inElemFlat] += sum;
    }
}
`, kernelVol, tileSize, kernelVol, tileSize)
}

// ShaderTiledCNN2BackwardDW generates a tiled gradient-w.r.t-weights (dW) WGSL shader.
//
// Tiling strategy:
//
//	Each workgroup covers tileSize weight elements (one per thread) for one filter.
//	Dispatch: X = ceil(kernelVol/tileSize),  Y = filters,  Z = 1.
//	The shader loops over all (batch, output spatial) positions to accumulate dW.
//	In each tile iteration, tileSize threads cooperatively load tileSize dy values
//	(gradOutput × activDeriv for filter f) into shared memory, then EVERY thread
//	uses ALL tileSize cached dy values to update its weight gradient.
//	This reduces global gradOutput reads by a factor of tileSize vs the naive shader.
//
//	tileSize — workgroup size and shared memory cache depth.
func ShaderTiledCNN2BackwardDW(tileSize int) string {
	return fmt.Sprintf(wgslCNN2Bwd2DParamsStruct+`
@group(0) @binding(0) var<uniform>             params:      CNN2Bwd2DParams;
@group(0) @binding(1) var<storage, read>       gradOutput:  array<f32>;
@group(0) @binding(2) var<storage, read>       input:       array<f32>;
@group(0) @binding(3) var<storage, read>       preAct:      array<f32>;
@group(0) @binding(4) var<storage, read_write> gradWeights: array<f32>;

// Shared cache: holds tileSize dy = (gradOutput * activDeriv) values for filter f.
var<workgroup> dyCache: array<f32, %d>;
`+wgslBwdActivateDeriv+`

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id:  vec3<u32>,
) {
    let kernelPos = global_id.x; // index into [0, inC*kH*kW)
    let f         = global_id.y; // filter index
    let kVol      = params.inC * params.kH * params.kW;
    if (f >= params.filters) { return; }

    // Decode kernelPos -> (ic, kh, kw)
    let kHW   = params.kH * params.kW;
    let ic    = kernelPos / kHW;
    let kRem  = kernelPos %% kHW;
    let kh    = kRem / params.kW;
    let kw    = kRem %% params.kW;

    let oArea        = params.outH * params.outW;
    let totalSpatial = params.batchSize * oArea;
    var sum: f32     = 0.0;

    // Tile over all (batch × output spatial) positions.
    var spatial: u32 = 0u;
    loop {
        if (spatial >= totalSpatial) { break; }

        // Step 1 — cooperatively load tileSize dy values for filter f.
        let loadIdx = spatial + local_id.x;
        if (loadIdx < totalSpatial) {
            let lb     = loadIdx / oArea;
            let loohow = loadIdx %% oArea;
            let lIdx   = lb * params.filters * oArea + f * oArea + loohow;
            dyCache[local_id.x] = gradOutput[lIdx] * activateDerivative(preAct[lIdx], params.activation);
        } else {
            dyCache[local_id.x] = 0.0;
        }
        workgroupBarrier();

        // Step 2 — each thread accumulates its weight gradient from cached dy values.
        if (kernelPos < kVol) {
            for (var ti: u32 = 0u; ti < %du; ti++) {
                let bSpatial = spatial + ti;
                if (bSpatial >= totalSpatial) { break; }
                let b     = bSpatial / oArea;
                let oohow = bSpatial %% oArea;
                let oh    = oohow / params.outW;
                let ow    = oohow %% params.outW;

                let ih_i = i32(oh * params.sH + kh) - i32(params.pH);
                let iw_i = i32(ow * params.sW + kw) - i32(params.pW);
                if (ih_i >= 0 && u32(ih_i) < params.inH &&
                    iw_i >= 0 && u32(iw_i) < params.inW) {
                    let inIdx = ((b * params.inC + ic) * params.inH + u32(ih_i)) * params.inW + u32(iw_i);
                    sum += dyCache[ti] * input[inIdx];
                }
            }
        }
        workgroupBarrier();
        spatial += %du;
    }

    if (kernelPos < kVol) {
        gradWeights[f * kVol + kernelPos] += sum;
    }
}
`, tileSize, tileSize, tileSize, tileSize)
}

// DispatchCNN2TiledBackwardDX dispatches a tiled DX backward pass for CNN2.
// weightBuf should contain raw (un-scaled) integer values stored as float32 to match
// CPU backward arithmetic (no scale applied in backward, same as DispatchCNN2BackwardDX).
// gradInputBuf must be pre-zeroed; the shader accumulates (+=) into it.
func (c *WGPUContext) DispatchCNN2TiledBackwardDX(
	tileSize, kernelVol int,
	batchSize, inC, inH, inW,
	filters, outH, outW,
	kH, kW, sH, sW, pH, pW int,
	activation ActivationType,
	gradOutputBuf, weightBuf, preActBuf, gradInputBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderTiledCNN2BackwardDX(tileSize, kernelVol))
	if err != nil {
		return err
	}

	p := WGPUCNN2Bwd2DParams{
		BatchSize: uint32(batchSize),
		InC: uint32(inC), InH: uint32(inH), InW: uint32(inW),
		Filters: uint32(filters),
		OutH: uint32(outH), OutW: uint32(outW),
		KH: uint32(kH), KW: uint32(kW),
		SH: uint32(sH), SW: uint32(sW),
		PH: uint32(pH), PW: uint32(pW),
		Activation: mapActivation(activation),
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUCNN2Bwd2DParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, weightBuf, preActBuf, gradInputBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups(
		(uint32(inC*inH*inW)+uint32(tileSize)-1)/uint32(tileSize),
		1,
		uint32(batchSize),
	)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

// DispatchCNN2TiledBackwardDW dispatches a tiled DW backward pass for CNN2.
// inputBuf contains the original forward-pass input (float32).
// gradWeightsBuf must be pre-zeroed; the shader accumulates (+=) into it.
func (c *WGPUContext) DispatchCNN2TiledBackwardDW(
	tileSize int,
	batchSize, inC, inH, inW,
	filters, outH, outW,
	kH, kW, sH, sW, pH, pW int,
	activation ActivationType,
	gradOutputBuf, inputBuf, preActBuf, gradWeightsBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderTiledCNN2BackwardDW(tileSize))
	if err != nil {
		return err
	}

	p := WGPUCNN2Bwd2DParams{
		BatchSize: uint32(batchSize),
		InC: uint32(inC), InH: uint32(inH), InW: uint32(inW),
		Filters: uint32(filters),
		OutH: uint32(outH), OutW: uint32(outW),
		KH: uint32(kH), KW: uint32(kW),
		SH: uint32(sH), SW: uint32(sW),
		PH: uint32(pH), PW: uint32(pW),
		Activation: mapActivation(activation),
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUCNN2Bwd2DParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, inputBuf, preActBuf, gradWeightsBuf)
	if err != nil {
		return err
	}

	kernelVol := inC * kH * kW
	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups(
		(uint32(kernelVol)+uint32(tileSize)-1)/uint32(tileSize),
		uint32(filters),
		1,
	)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}
