package poly

import (
	"fmt"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// WGPUCNN3Backward3DParams is the uniform struct for the tiled CNN3 backward shaders.
// Uses separate kD/kH/kW to support asymmetric kernels, matching the forward convention.
// 20 × uint32/float32 = 80 bytes (multiple of 16 for WebGPU alignment).
type WGPUCNN3Backward3DParams struct {
	BatchSize             uint32
	InC, InD, InH, InW   uint32
	Filters               uint32
	OutD, OutH, OutW      uint32
	KD, KH, KW            uint32
	SD, SH, SW            uint32
	PD, PH, PW            uint32
	Activation            uint32
	Pad                   uint32
}

// wgslBwdActivateDeriv is the WGSL activation derivative helper inlined into each
// generated backward shader.  Identical to wgslActivateDerivative in
// wgpu_backward_shaders.go but embedded as a raw string constant so the generated
// shader functions (which use fmt.Sprintf) remain self-contained.
const wgslBwdActivateDeriv = `
fn activateDerivative(v: f32, act: u32) -> f32 {
    if (act == 0u) { if (v <= 0.0) { return 0.0; } return 1.0; }
    if (act == 1u) { let sig = 1.0 / (1.0 + exp(-v)); return sig * (1.0 + v * (1.0 - sig)); }
    if (act == 3u) { let t = tanh(v); return 1.0 - t * t; }
    if (act == 4u) { let s = 1.0 / (1.0 + exp(-v)); return s * (1.0 - s); }
    return 1.0;
}`

// wgslCNN3Bwd3DParamsStruct is the WGSL struct definition shared by all tiled
// backward shaders — avoids repeating 20 field declarations in every generator.
const wgslCNN3Bwd3DParamsStruct = `
struct CNN3Bwd3DParams {
    batchSize: u32,
    inC: u32, inD: u32, inH: u32, inW: u32,
    filters: u32, outD: u32, outH: u32, outW: u32,
    kD: u32, kH: u32, kW: u32,
    sD: u32, sH: u32, sW: u32,
    pD: u32, pH: u32, pW: u32,
    activation: u32, _pad: u32,
};`

// ShaderTiledCNN3BackwardDX generates a tiled gradient-w.r.t-input (dX) WGSL shader.
//
// Tiling strategy — mirrors ShaderTiledCNN3 for the forward pass:
//   Each workgroup covers tileSize input elements (one per thread) for one batch item.
//   For every output filter the tileSize threads cooperatively load that filter's full
//   kernel (kernelVol = inC × kD × kH × kW weights) into fast workgroup shared memory,
//   then each thread accumulates its input element's gradient contribution from that
//   filter.  The cooperative load reduces global weight reads by a factor of tileSize.
//
//   kernelVol — inC * kD * kH * kW.  Baked into var<workgroup> wCache array size.
//   tileSize  — @workgroup_size(tileSize, 1, 1).
func ShaderTiledCNN3BackwardDX(tileSize, kernelVol int) string {
	return fmt.Sprintf(wgslCNN3Bwd3DParamsStruct+`
@group(0) @binding(0) var<uniform>           params:     CNN3Bwd3DParams;
@group(0) @binding(1) var<storage, read>       gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read>       weights:    array<f32>;
@group(0) @binding(3) var<storage, read>       preAct:     array<f32>;
@group(0) @binding(4) var<storage, read_write> gradInput:  array<f32>;

// Workgroup kernel cache — holds one filter's weights.
// Size baked at generation time: kernelVol = inC * kD * kH * kW.
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
    let inDHW      = params.inD * params.inH * params.inW;
    let inVol      = params.inC * inDHW;
    if (batchIdx >= params.batchSize) { return; }

    // Decode flat input position → (ic, id, ih, iw)
    let ic   = inElemFlat / inDHW;
    let rem1 = inElemFlat %% inDHW;
    let id   = rem1 / (params.inH * params.inW);
    let rem2 = rem1 %% (params.inH * params.inW);
    let ih   = rem2 / params.inW;
    let iw   = rem2 %% params.inW;

    let oArea = params.outD * params.outH * params.outW;
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
            for (var kd: u32 = 0u; kd < params.kD; kd++) {
                for (var kh: u32 = 0u; kh < params.kH; kh++) {
                    for (var kw: u32 = 0u; kw < params.kW; kw++) {
                        let vd = i32(id) + i32(params.pD) - i32(kd);
                        let vh = i32(ih) + i32(params.pH) - i32(kh);
                        let vw = i32(iw) + i32(params.pW) - i32(kw);
                        if (vd >= 0 && vd %% i32(params.sD) == 0 &&
                            vh >= 0 && vh %% i32(params.sH) == 0 &&
                            vw >= 0 && vw %% i32(params.sW) == 0) {
                            let od = u32(vd / i32(params.sD));
                            let oh = u32(vh / i32(params.sH));
                            let ow = u32(vw / i32(params.sW));
                            if (od < params.outD && oh < params.outH && ow < params.outW) {
                                let outIdx = (((batchIdx * params.filters + f) * params.outD + od) * params.outH + oh) * params.outW + ow;
                                let dy = gradOutput[outIdx] * activateDerivative(preAct[outIdx], params.activation);
                                // wCache holds weight[f, :, :, :, :]; index by (ic, kd, kh, kw).
                                let wCacheIdx = ic * params.kD * params.kH * params.kW
                                              + kd * params.kH * params.kW
                                              + kh * params.kW + kw;
                                sum += dy * wCache[wCacheIdx];
                            }
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

// ShaderTiledCNN3BackwardDW generates a tiled gradient-w.r.t-weights (dW) WGSL shader.
//
// Tiling strategy:
//   Each workgroup covers tileSize weight elements (one per thread) for one filter.
//   Dispatch: X = ceil(kernelVol/tileSize),  Y = filters,  Z = 1.
//   The shader loops over all (batch, output spatial) positions to accumulate dW.
//   In each tile iteration, tileSize threads cooperatively load tileSize dy values
//   (gradOutput × activDeriv for filter f) into shared memory, then EVERY thread
//   uses ALL tileSize cached dy values to update its weight gradient.
//   This reduces global gradOutput reads by a factor of tileSize vs the naive shader.
//
//   tileSize — workgroup size and shared memory cache depth.
func ShaderTiledCNN3BackwardDW(tileSize int) string {
	return fmt.Sprintf(wgslCNN3Bwd3DParamsStruct+`
@group(0) @binding(0) var<uniform>           params:      CNN3Bwd3DParams;
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
    let kernelPos = global_id.x; // index into [0, inC*kD*kH*kW)
    let f         = global_id.y; // filter index
    let kVol      = params.inC * params.kD * params.kH * params.kW;
    if (f >= params.filters) { return; }

    // Decode kernelPos → (ic, kd, kh, kw)
    let kDHW  = params.kD * params.kH * params.kW;
    let kHW   = params.kH * params.kW;
    let ic    = kernelPos / kDHW;
    let kRem  = kernelPos %% kDHW;
    let kd    = kRem / kHW;
    let kRem2 = kRem %% kHW;
    let kh    = kRem2 / params.kW;
    let kw    = kRem2 %% params.kW;

    let oArea        = params.outD * params.outH * params.outW;
    let totalSpatial = params.batchSize * oArea;
    var sum: f32     = 0.0;

    // Tile over all (batch × output spatial) positions.
    var spatial: u32 = 0u;
    loop {
        if (spatial >= totalSpatial) { break; }

        // Step 1 — cooperatively load tileSize dy values for filter f.
        let loadIdx = spatial + local_id.x;
        if (loadIdx < totalSpatial) {
            let lb      = loadIdx / oArea;
            let lodohow = loadIdx %% oArea;
            let lIdx    = lb * params.filters * oArea + f * oArea + lodohow;
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
                let b      = bSpatial / oArea;
                let odohow = bSpatial %% oArea;
                let od     = odohow / (params.outH * params.outW);
                let oh     = (odohow / params.outW) %% params.outH;
                let ow     = odohow %% params.outW;

                let id_i = i32(od * params.sD + kd) - i32(params.pD);
                let ih_i = i32(oh * params.sH + kh) - i32(params.pH);
                let iw_i = i32(ow * params.sW + kw) - i32(params.pW);
                if (id_i >= 0 && u32(id_i) < params.inD &&
                    ih_i >= 0 && u32(ih_i) < params.inH &&
                    iw_i >= 0 && u32(iw_i) < params.inW) {
                    let inIdx = (((b * params.inC + ic) * params.inD + u32(id_i)) * params.inH + u32(ih_i)) * params.inW + u32(iw_i);
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

// DispatchCNN3TiledBackwardDX dispatches a tiled DX backward pass.
// weightBuf should contain raw (un-scaled) integer values stored as float32 to match
// CPU backward arithmetic (no scale applied in backward, same as DispatchCNN3BackwardDX).
// gradInputBuf must be pre-zeroed; the shader accumulates (+=) into it.
func (c *WGPUContext) DispatchCNN3TiledBackwardDX(
	tileSize, kernelVol int,
	batchSize, inC, inD, inH, inW,
	filters, outD, outH, outW,
	kD, kH, kW, sD, sH, sW, pD, pH, pW int,
	activation ActivationType,
	gradOutputBuf, weightBuf, preActBuf, gradInputBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderTiledCNN3BackwardDX(tileSize, kernelVol))
	if err != nil {
		return err
	}

	p := WGPUCNN3Backward3DParams{
		BatchSize: uint32(batchSize),
		InC: uint32(inC), InD: uint32(inD), InH: uint32(inH), InW: uint32(inW),
		Filters: uint32(filters),
		OutD: uint32(outD), OutH: uint32(outH), OutW: uint32(outW),
		KD: uint32(kD), KH: uint32(kH), KW: uint32(kW),
		SD: uint32(sD), SH: uint32(sH), SW: uint32(sW),
		PD: uint32(pD), PH: uint32(pH), PW: uint32(pW),
		Activation: mapActivation(activation),
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUCNN3Backward3DParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, weightBuf, preActBuf, gradInputBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups(
		(uint32(inC*inD*inH*inW)+uint32(tileSize)-1)/uint32(tileSize),
		1,
		uint32(batchSize),
	)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

// DispatchCNN3TiledBackwardDW dispatches a tiled DW backward pass.
// inputBuf contains the original forward-pass input (float32).
// gradWeightsBuf must be pre-zeroed; the shader accumulates (+=) into it.
func (c *WGPUContext) DispatchCNN3TiledBackwardDW(
	tileSize int,
	batchSize, inC, inD, inH, inW,
	filters, outD, outH, outW,
	kD, kH, kW, sD, sH, sW, pD, pH, pW int,
	activation ActivationType,
	gradOutputBuf, inputBuf, preActBuf, gradWeightsBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderTiledCNN3BackwardDW(tileSize))
	if err != nil {
		return err
	}

	p := WGPUCNN3Backward3DParams{
		BatchSize: uint32(batchSize),
		InC: uint32(inC), InD: uint32(inD), InH: uint32(inH), InW: uint32(inW),
		Filters: uint32(filters),
		OutD: uint32(outD), OutH: uint32(outH), OutW: uint32(outW),
		KD: uint32(kD), KH: uint32(kH), KW: uint32(kW),
		SD: uint32(sD), SH: uint32(sH), SW: uint32(sW),
		PD: uint32(pD), PH: uint32(pH), PW: uint32(pW),
		Activation: mapActivation(activation),
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUCNN3Backward3DParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, inputBuf, preActBuf, gradWeightsBuf)
	if err != nil {
		return err
	}

	kernelVol := inC * kD * kH * kW
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
