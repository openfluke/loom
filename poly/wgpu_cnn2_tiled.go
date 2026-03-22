package poly

import (
	"fmt"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// WGPUCNN2ScaleParams is the uniform struct for CNN2 tiled shaders.
// 16 × uint32/float32 = 64 bytes (multiple of 16 for WebGPU uniform alignment).
type WGPUCNN2ScaleParams struct {
	BatchSize        uint32
	InC, InH, InW   uint32
	OutC, OutH, OutW uint32
	KH, KW          uint32
	SH, SW          uint32
	PH, PW          uint32
	Scale           float32
	Pad             uint32 // explicit padding → 64 bytes total
}

// ShaderCNN2Scaled is a non-tiled CNN2 WGSL shader that accepts a scale uniform and
// applies "sum * scale" at the end of accumulation. This matches the CPU integer-path
// arithmetic exactly (accumulate raw int values as f32, then multiply scale once) and
// produces 0.00e+00 diff vs the CPU reference for all integer types.
// Dispatch: X=ceil(outH*outW/64), Y=outC, Z=batchSize
const ShaderCNN2Scaled = `
struct CNN2ScaleParams {
    batchSize: u32,
    inC: u32, inH: u32, inW: u32,
    outC: u32, outH: u32, outW: u32,
    kH: u32, kW: u32,
    sH: u32, sW: u32,
    pH: u32, pW: u32,
    scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform>             params:  CNN2ScaleParams;
@group(0) @binding(1) var<storage, read>       input:   array<f32>;
@group(0) @binding(2) var<storage, read>       weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output:  array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outFlat   = global_id.x;
    let filterIdx = global_id.y;
    let batchIdx  = global_id.z;
    let oArea = params.outH * params.outW;
    if (outFlat >= oArea || filterIdx >= params.outC || batchIdx >= params.batchSize) { return; }

    let oh = outFlat / params.outW;
    let ow = outFlat % params.outW;
    var sum: f32 = 0.0;

    for (var ic: u32 = 0u; ic < params.inC; ic++) {
        for (var kh: u32 = 0u; kh < params.kH; kh++) {
            for (var kw: u32 = 0u; kw < params.kW; kw++) {
                let ih = i32(oh * params.sH + kh) - i32(params.pH);
                let iw = i32(ow * params.sW + kw) - i32(params.pW);
                if (ih >= 0 && u32(ih) < params.inH && iw >= 0 && u32(iw) < params.inW) {
                    let inIdx = batchIdx * params.inC * params.inH * params.inW
                              + ic * params.inH * params.inW
                              + u32(ih) * params.inW + u32(iw);
                    let wIdx = filterIdx * params.inC * params.kH * params.kW
                             + ic * params.kH * params.kW
                             + kh * params.kW + kw;
                    sum += input[inIdx] * weights[wIdx];
                }
            }
        }
    }
    // Apply scale at the end — identical to CPU integer-path arithmetic.
    output[batchIdx * params.outC * oArea + filterIdx * oArea + outFlat] = sum * params.scale;
}
`

// ShaderTiledCNN2 generates a tiled 2D convolution WGSL shader with shared-memory
// kernel caching — following the same pattern as ShaderTiledCNN3.
//
//   tileSize  — @workgroup_size(tileSize, 1, 1): threads per workgroup.
//               SC path: small value (e.g. 64). MC path: larger (e.g. 256).
//   kernelVol — inC * kH * kW: the number of weights to cache per filter.
//               Baked into var<workgroup> wCache array size.
//
// All tileSize threads cooperatively load one filter's kernelVol weights into fast
// shared memory, then each thread computes one spatial output element using the cache.
// Dispatch: X=ceil(outH*outW/tileSize), Y=outC, Z=batchSize
func ShaderTiledCNN2(tileSize, kernelVol int) string {
	return fmt.Sprintf(`
struct CNN2ScaleParams {
    batchSize: u32,
    inC: u32, inH: u32, inW: u32,
    outC: u32, outH: u32, outW: u32,
    kH: u32, kW: u32,
    sH: u32, sW: u32,
    pH: u32, pW: u32,
    scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform>             params:  CNN2ScaleParams;
@group(0) @binding(1) var<storage, read>       input:   array<f32>;
@group(0) @binding(2) var<storage, read>       weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output:  array<f32>;

// Workgroup kernel cache — holds one filter's weights.
// Size baked in at generation time: kernelVol = inC * kH * kW.
var<workgroup> wCache: array<f32, %d>;

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id:  vec3<u32>,
) {
    let filterIdx = global_id.y;
    let batchIdx  = global_id.z;
    let oArea     = params.outH * params.outW;
    let kVol: u32 = %du;
    let wBase     = filterIdx * kVol;

    // Step 1 — cooperatively load this filter's kernel into shared memory.
    var i: u32 = local_id.x;
    loop {
        if (i >= kVol) { break; }
        wCache[i] = weights[wBase + i];
        i += %du;
    }

    // Step 2 — all threads synchronise before reading from wCache.
    workgroupBarrier();

    // Step 3 — each thread computes one spatial output element.
    let outFlat = global_id.x;
    if (outFlat >= oArea || filterIdx >= params.outC || batchIdx >= params.batchSize) { return; }

    let oh = outFlat / params.outW;
    let ow = outFlat %% params.outW;
    var sum: f32 = 0.0;

    for (var ic: u32 = 0u; ic < params.inC; ic++) {
        for (var kh: u32 = 0u; kh < params.kH; kh++) {
            for (var kw: u32 = 0u; kw < params.kW; kw++) {
                let ih = i32(oh * params.sH + kh) - i32(params.pH);
                let iw = i32(ow * params.sW + kw) - i32(params.pW);
                if (ih >= 0 && u32(ih) < params.inH && iw >= 0 && u32(iw) < params.inW) {
                    let inIdx = batchIdx * params.inC * params.inH * params.inW
                              + ic * params.inH * params.inW
                              + u32(ih) * params.inW + u32(iw);
                    let cacheIdx = ic * params.kH * params.kW + kh * params.kW + kw;
                    sum += input[inIdx] * wCache[cacheIdx];
                }
            }
        }
    }
    // Apply scale at the end — identical to CPU integer-path arithmetic.
    output[batchIdx * params.outC * oArea + filterIdx * oArea + outFlat] = sum * params.scale;
}
`, kernelVol, tileSize, kernelVol, tileSize)
}

// DispatchCNN2Scaled dispatches a non-tiled CNN2 forward pass with a scale uniform.
// Use raw (un-scaled) integer values as float32 in weightBuf and pass the quantization
// scale here so the GPU applies "sum * scale" at the end — matching CPU arithmetic exactly.
// For float types pass scale=1.0.
func (c *WGPUContext) DispatchCNN2Scaled(
	batchSize, inC, inH, inW,
	outC, outH, outW,
	kH, kW, sH, sW, pH, pW int,
	scale float32,
	inputBuf, weightBuf, outputBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderCNN2Scaled)
	if err != nil {
		return err
	}

	p := WGPUCNN2ScaleParams{
		BatchSize: uint32(batchSize),
		InC: uint32(inC), InH: uint32(inH), InW: uint32(inW),
		OutC: uint32(outC), OutH: uint32(outH), OutW: uint32(outW),
		KH: uint32(kH), KW: uint32(kW),
		SH: uint32(sH), SW: uint32(sW),
		PH: uint32(pH), PW: uint32(pW),
		Scale: scale,
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUCNN2ScaleParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, weightBuf, outputBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups(
		(uint32(outH*outW)+63)/64,
		uint32(outC),
		uint32(batchSize),
	)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

// DispatchCNN2Tiled dispatches a tiled CNN2 forward pass with workgroup shared-memory
// kernel caching. The shader is generated and compiled once per (tileSize, kernelVol)
// pair; subsequent calls hit the PipelineCache instantly.
//
//   tileSize  — workgroup size (threads per workgroup).
//               Use scTile from CNN2GPUTileSizes for the SC path, mcTile for MC.
//   kernelVol — inC * kH * kW. Must match the actual weight buffer layout.
//   scale     — quantization scale. 1.0 for float types; cfg.scale for integer types.
//   weightBuf — raw (un-scaled) integer values stored as float32
//               (e.g. int8(10) uploaded as float32(10.0)), so the GPU applies scale
//               once at the end, matching CPU integer arithmetic exactly.
func (c *WGPUContext) DispatchCNN2Tiled(
	tileSize, kernelVol int,
	batchSize, inC, inH, inW,
	outC, outH, outW,
	kH, kW, sH, sW, pH, pW int,
	scale float32,
	inputBuf, weightBuf, outputBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderTiledCNN2(tileSize, kernelVol))
	if err != nil {
		return err
	}

	p := WGPUCNN2ScaleParams{
		BatchSize: uint32(batchSize),
		InC: uint32(inC), InH: uint32(inH), InW: uint32(inW),
		OutC: uint32(outC), OutH: uint32(outH), OutW: uint32(outW),
		KH: uint32(kH), KW: uint32(kW),
		SH: uint32(sH), SW: uint32(sW),
		PH: uint32(pH), PW: uint32(pW),
		Scale: scale,
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUCNN2ScaleParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, weightBuf, outputBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups(
		(uint32(outH*outW)+uint32(tileSize)-1)/uint32(tileSize),
		uint32(outC),
		uint32(batchSize),
	)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

// CNN2GPUTileSizes returns the SC and MC tile sizes for CNN2 tiling based on the
// GPU's auto-detected capabilities stored in ctx.
//
//   SC (single-core analog) — smallest warp-efficient workgroup size. Clamped to
//                             at least 64 so threads always fill at least one full
//                             wavefront; below that, SIMD lanes go idle.
//   MC (multi-core analog)  — larger workgroup from MaxComputeInvocationsPerWorkgroup,
//                             capped at 256 (8 NVIDIA warps / 4 AMD wavefronts).
//
// On a GPU with GPUTileSize=16 and MaxInvocations=256 → SC=64, MC=256.
func CNN2GPUTileSizes(ctx *WGPUContext) (scTile, mcTile int) {
	scTile = ctx.GPUTileSize * 4
	if scTile < 64 {
		scTile = 64
	}

	mcTile = int(ctx.Limits.MaxComputeInvocationsPerWorkgroup)
	if mcTile <= 0 || mcTile > 256 {
		mcTile = 256
	}
	// Align MC to a multiple of 64 (wavefront boundary).
	mcTile = (mcTile / 64) * 64
	if mcTile < 64 {
		mcTile = 64
	}
	// SC must be < MC to be a meaningful distinction.
	if scTile >= mcTile {
		scTile = mcTile / 4
		if scTile < 64 {
			scTile = 64
		}
	}
	return
}
