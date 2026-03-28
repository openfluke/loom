package poly

import (
	"fmt"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// WGPUCNN3ScaleParams is the uniform struct for the tiled CNN3 shaders.
// It extends the base CNN3 layout with a quantization scale factor so the
// GPU can apply scale-at-end arithmetic — matching the CPU integer path exactly.
// Padded to 80 bytes (multiple of 16) for WebGPU uniform alignment.
type WGPUCNN3ScaleParams struct {
	BatchSize             uint32
	InC, InD, InH, InW   uint32
	OutC, OutD, OutH, OutW uint32
	KD, KH, KW            uint32
	SD, SH, SW            uint32
	PD, PH, PW            uint32
	Scale                 float32
	Pad                   uint32 // explicit padding → 80 bytes total
}

// ShaderCNN3Scaled is a non-tiled CNN3 WGSL shader that accepts a scale uniform and
// applies "sum * scale" at the end of accumulation.  This matches the CPU integer-path
// arithmetic exactly (accumulate raw int values as f32, then multiply scale once) and
// produces 0.00e+00 diff vs the CPU reference for all integer types.
// The struct layout reuses CNN3ScaleParams (same as the tiled shader) so the same
// WGPUCNN3ScaleParams Go struct and DispatchCNN3Scaled helper can be shared.
const ShaderCNN3Scaled = `
struct CNN3ScaleParams {
    batchSize: u32,
    inC: u32, inD: u32, inH: u32, inW: u32,
    outC: u32, outD: u32, outH: u32, outW: u32,
    kD: u32, kH: u32, kW: u32,
    sD: u32, sH: u32, sW: u32,
    pD: u32, pH: u32, pW: u32,
    scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: CNN3ScaleParams;
@group(0) @binding(1) var<storage, read>       input:   array<f32>;
@group(0) @binding(2) var<storage, read>       weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output:  array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outIdx_flat = global_id.x;
    let filterIdx   = global_id.y;
    let batchIdx    = global_id.z;

    let oArea = params.outD * params.outH * params.outW;
    if (outIdx_flat >= oArea || filterIdx >= params.outC || batchIdx >= params.batchSize) { return; }

    let outW_pos  = outIdx_flat % params.outW;
    let remainder = outIdx_flat / params.outW;
    let outH_pos  = remainder % params.outH;
    let outD_pos  = remainder / params.outH;

    var sum: f32 = 0.0;
    for (var ic: u32 = 0u; ic < params.inC; ic++) {
        for (var kd: u32 = 0u; kd < params.kD; kd++) {
            for (var kh: u32 = 0u; kh < params.kH; kh++) {
                for (var kw: u32 = 0u; kw < params.kW; kw++) {
                    let inD_pos = i32(outD_pos * params.sD + kd) - i32(params.pD);
                    let inH_pos = i32(outH_pos * params.sH + kh) - i32(params.pH);
                    let inX_pos = i32(outW_pos * params.sW + kw) - i32(params.pW);

                    if (inD_pos >= 0 && u32(inD_pos) < params.inD &&
                        inH_pos >= 0 && u32(inH_pos) < params.inH &&
                        inX_pos >= 0 && u32(inX_pos) < params.inW) {

                        let inIdx = batchIdx * params.inC * params.inD * params.inH * params.inW
                                  + ic * params.inD * params.inH * params.inW
                                  + u32(inD_pos) * params.inH * params.inW
                                  + u32(inH_pos) * params.inW + u32(inX_pos);
                        let wIdx = filterIdx * params.inC * params.kD * params.kH * params.kW
                                 + ic * params.kD * params.kH * params.kW
                                 + kd * params.kH * params.kW
                                 + kh * params.kW + kw;
                        sum += input[inIdx] * weights[wIdx];
                    }
                }
            }
        }
    }
    // Apply scale at the end — identical to CPU integer-path arithmetic.
    output[batchIdx * params.outC * oArea + filterIdx * oArea + outIdx_flat] = sum * params.scale;
}
`

// DispatchCNN3Scaled dispatches a non-tiled CNN3 forward pass with a scale uniform.
// Use raw (un-scaled) integer values as float32 in weightBuf and pass the quantization
// scale here so the GPU applies "sum * scale" at the end — matching CPU arithmetic exactly.
// For float types pass scale=1.0.
func (c *WGPUContext) DispatchCNN3Scaled(
	batchSize, inC, inD, inH, inW,
	outC, outD, outH, outW,
	kD, kH, kW, sD, sH, sW, pD, pH, pW int,
	scale float32,
	inputBuf, weightBuf, outputBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderCNN3Scaled)
	if err != nil {
		return err
	}

	p := WGPUCNN3ScaleParams{
		BatchSize: uint32(batchSize),
		InC: uint32(inC), InD: uint32(inD), InH: uint32(inH), InW: uint32(inW),
		OutC: uint32(outC), OutD: uint32(outD), OutH: uint32(outH), OutW: uint32(outW),
		KD: uint32(kD), KH: uint32(kH), KW: uint32(kW),
		SD: uint32(sD), SH: uint32(sH), SW: uint32(sW),
		PD: uint32(pD), PH: uint32(pH), PW: uint32(pW),
		Scale: scale,
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUCNN3ScaleParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, weightBuf, outputBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups(
		(uint32(outD*outH*outW)+63)/64,
		uint32(outC),
		uint32(batchSize),
	)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

// ShaderTiledCNN3 generates a tiled 3D convolution WGSL shader with the workgroup
// tile size and kernel cache size baked in as compile-time constants — following the
// same pattern as ShaderTiledDenseN / ShaderTiledMHAN in wgpu_shaders.go.
//
//   tileSize  — @workgroup_size(tileSize, 1, 1): threads per workgroup.
//               SC path: ctx.GPUTileSize (small, e.g. 16 or 32).
//               MC path: larger value derived from GPU limits (e.g. 64 or 256).
//   kernelVol — inC * kD * kH * kW: the number of weights to cache per filter.
//               Baked into var<workgroup> wCache array size.
//
// All tileSize threads in a workgroup cooperatively load one filter's kernelVol
// weights into fast shared memory, then each thread computes one spatial output
// element using the cache.  The GPU applies scale at the end of accumulation,
// which matches the CPU integer-path arithmetic exactly (sum * scale, not per-element).
func ShaderTiledCNN3(tileSize, kernelVol int) string {
	return fmt.Sprintf(`
struct CNN3ScaleParams {
    batchSize: u32,
    inC: u32, inD: u32, inH: u32, inW: u32,
    outC: u32, outD: u32, outH: u32, outW: u32,
    kD: u32, kH: u32, kW: u32,
    sD: u32, sH: u32, sW: u32,
    pD: u32, pH: u32, pW: u32,
    scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: CNN3ScaleParams;
@group(0) @binding(1) var<storage, read>       input:   array<f32>;
@group(0) @binding(2) var<storage, read>       weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output:  array<f32>;

// Workgroup kernel cache — holds one filter's weights.
// Size baked in at generation time: kernelVol = inC * kD * kH * kW.
var<workgroup> wCache: array<f32, %d>;

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id:  vec3<u32>,
) {
    let filterIdx = global_id.y;
    let batchIdx  = global_id.z;
    let oArea     = params.outD * params.outH * params.outW;
    let kVol: u32 = %du;
    let wBase     = filterIdx * kVol;

    // Step 1 — cooperatively load this filter's kernel into shared memory.
    // Each thread strides by tileSize until the full kernel is loaded.
    var i: u32 = local_id.x;
    loop {
        if (i >= kVol) { break; }
        wCache[i] = weights[wBase + i];
        i += %du;
    }

    // Step 2 — all threads synchronise before reading from wCache.
    workgroupBarrier();

    // Step 3 — each thread computes one spatial output element.
    let outIdx_flat = global_id.x;
    if (outIdx_flat >= oArea || filterIdx >= params.outC || batchIdx >= params.batchSize) {
        return;
    }

    let outW_pos  = outIdx_flat %% params.outW;
    let remainder = outIdx_flat / params.outW;
    let outH_pos  = remainder %% params.outH;
    let outD_pos  = remainder / params.outH;

    var sum: f32 = 0.0;
    for (var ic: u32 = 0u; ic < params.inC; ic++) {
        for (var kd: u32 = 0u; kd < params.kD; kd++) {
            for (var kh: u32 = 0u; kh < params.kH; kh++) {
                for (var kw: u32 = 0u; kw < params.kW; kw++) {
                    let inD_pos = i32(outD_pos * params.sD + kd) - i32(params.pD);
                    let inH_pos = i32(outH_pos * params.sH + kh) - i32(params.pH);
                    let inX_pos = i32(outW_pos * params.sW + kw) - i32(params.pW);

                    if (inD_pos >= 0 && u32(inD_pos) < params.inD &&
                        inH_pos >= 0 && u32(inH_pos) < params.inH &&
                        inX_pos >= 0 && u32(inX_pos) < params.inW) {

                        let inIdx = batchIdx * params.inC * params.inD * params.inH * params.inW
                                  + ic * params.inD * params.inH * params.inW
                                  + u32(inD_pos) * params.inH * params.inW
                                  + u32(inH_pos) * params.inW + u32(inX_pos);

                        let cacheIdx = ic * params.kD * params.kH * params.kW
                                     + kd * params.kH * params.kW
                                     + kh * params.kW + kw;

                        sum += input[inIdx] * wCache[cacheIdx];
                    }
                }
            }
        }
    }
    // Apply scale at the end — identical to CPU integer path arithmetic.
    output[batchIdx * params.outC * oArea + filterIdx * oArea + outIdx_flat] = sum * params.scale;
}
`, kernelVol, tileSize, kernelVol, tileSize)
}

// DispatchCNN3Tiled dispatches a tiled CNN3 forward pass with workgroup shared-memory
// kernel caching.  The shader is generated and compiled once per (tileSize, kernelVol)
// pair; subsequent calls hit the PipelineCache instantly.
//
//   tileSize  — workgroup size (threads per workgroup).
//               Use ctx.GPUTileSize for the SC path, a larger multiple for MC.
//   kernelVol — inC * kD * kH * kW.  Must match the actual weight buffer layout.
//   scale     — quantization scale.  1.0 for float types; cfg.scale for integer types.
//   weightBuf — raw (un-scaled) integer values stored as float32
//               (e.g. int8(10) uploaded as float32(10.0)), so the GPU applies scale
//               once at the end, matching CPU integer arithmetic exactly.
func (c *WGPUContext) DispatchCNN3Tiled(
	tileSize, kernelVol int,
	batchSize, inC, inD, inH, inW,
	outC, outD, outH, outW,
	kD, kH, kW, sD, sH, sW, pD, pH, pW int,
	scale float32,
	inputBuf, weightBuf, outputBuf *wgpu.Buffer,
) error {
	// Each unique (tileSize, kernelVol) generates a distinct shader string →
	// CreateComputePipeline caches it by the source string key automatically.
	pipeline, err := c.CreateComputePipeline(ShaderTiledCNN3(tileSize, kernelVol))
	if err != nil {
		return err
	}

	p := WGPUCNN3ScaleParams{
		BatchSize: uint32(batchSize),
		InC: uint32(inC), InD: uint32(inD), InH: uint32(inH), InW: uint32(inW),
		OutC: uint32(outC), OutD: uint32(outD), OutH: uint32(outH), OutW: uint32(outW),
		KD: uint32(kD), KH: uint32(kH), KW: uint32(kW),
		SD: uint32(sD), SH: uint32(sH), SW: uint32(sW),
		PD: uint32(pD), PH: uint32(pH), PW: uint32(pW),
		Scale: scale,
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUCNN3ScaleParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, weightBuf, outputBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups(
		(uint32(outD*outH*outW)+uint32(tileSize)-1)/uint32(tileSize),
		uint32(outC),
		uint32(batchSize),
	)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

// CNN3GPUTileSizes returns the SC and MC tile sizes for CNN3 tiling based on the
// GPU's auto-detected capabilities stored in ctx.
//
//   SC (single-core analog) — smallest warp-efficient workgroup size.
//                             GPU warps are 32 wide (NVIDIA) or 64 wide (AMD).
//                             ctx.GPUTileSize is tuned for MHA (often 8–16) which is
//                             below warp size. We clamp SC to min 64 so threads always
//                             fill at least one full wavefront — below that, SIMD lanes
//                             go idle and per-thread loading overhead dominates.
//   MC (multi-core analog)  — larger workgroup from MaxComputeInvocationsPerWorkgroup,
//                             capped at 256 (8 NVIDIA warps / 4 AMD wavefronts).
//                             Each thread only loads ~4 kernel weights vs ~54 for SC=16.
//
// On a GPU with GPUTileSize=16 and MaxInvocations=256 → SC=64, MC=256.
func CNN3GPUTileSizes(ctx *WGPUContext) (scTile, mcTile int) {
	// SC: use GPUTileSize as a hint, but clamp to at least 64 (one full wavefront).
	// Values below 32 waste SIMD lanes and balloon per-thread kernel-load iterations.
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
