package poly

import (
	"fmt"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// WGPUCNN1ScaleParams is the uniform struct for CNN1 tiled shaders.
// 16 × uint32/float32 = 64 bytes (multiple of 16 for WebGPU uniform alignment).
type WGPUCNN1ScaleParams struct {
	BatchSize uint32
	InC       uint32
	InL       uint32
	OutC      uint32
	OutL      uint32
	KSize     uint32
	Stride    uint32
	Padding   uint32
	Scale     float32
	Pad1      uint32
	Pad2      uint32
	Pad3      uint32
	Pad4      uint32
	Pad5      uint32
	Pad6      uint32
	Pad7      uint32
}

// ShaderCNN1Scaled is a non-tiled CNN1 WGSL shader that accepts a scale uniform.
// Applies "sum * scale" at the end — matching CPU integer-path arithmetic exactly.
// Dispatch: X=ceil(outL/64), Y=outC, Z=batchSize
const ShaderCNN1Scaled = `
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
@group(0) @binding(2) var<storage, read>       weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output:  array<f32>;

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
                let inIdx  = batchIdx * params.inC * params.inL + ic * params.inL + u32(inPos);
                let wIdx   = filterIdx * params.inC * params.kSize + ic * params.kSize + k;
                sum += input[inIdx] * weights[wIdx];
            }
        }
    }
    output[batchIdx * params.outC * params.outL + filterIdx * params.outL + outPos] = sum * params.scale;
}
`

// ShaderTiledCNN1 generates a tiled 1D convolution WGSL shader with shared-memory
// kernel caching — analogous to ShaderTiledCNN2.
//
//	tileSize  — @workgroup_size(tileSize, 1, 1).
//	kernelVol — inC * kSize: weights to cache per filter.
//
// Dispatch: X=ceil(outL/tileSize), Y=outC, Z=batchSize
func ShaderTiledCNN1(tileSize, kernelVol int) string {
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
@group(0) @binding(2) var<storage, read>       weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output:  array<f32>;

// Workgroup kernel cache — holds one filter's weights.
// Size baked at generation time: kernelVol = inC * kSize.
var<workgroup> wCache: array<f32, %d>;

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id:  vec3<u32>,
) {
    let filterIdx = global_id.y;
    let batchIdx  = global_id.z;
    let kVol: u32 = %du;
    let wBase     = filterIdx * kVol;

    // Step 1 — cooperatively load this filter's kernel into shared memory.
    var i: u32 = local_id.x;
    loop {
        if (i >= kVol) { break; }
        wCache[i] = weights[wBase + i];
        i += %du;
    }
    workgroupBarrier();

    // Step 2 — each thread computes one output position.
    let outPos = global_id.x;
    if (outPos >= params.outL || filterIdx >= params.outC || batchIdx >= params.batchSize) { return; }

    var sum: f32 = 0.0;
    for (var ic: u32 = 0u; ic < params.inC; ic++) {
        for (var k: u32 = 0u; k < params.kSize; k++) {
            let inPos = i32(outPos * params.stride + k) - i32(params.padding);
            if (inPos >= 0 && u32(inPos) < params.inL) {
                let inIdx    = batchIdx * params.inC * params.inL + ic * params.inL + u32(inPos);
                let cacheIdx = ic * params.kSize + k;
                sum += input[inIdx] * wCache[cacheIdx];
            }
        }
    }
    output[batchIdx * params.outC * params.outL + filterIdx * params.outL + outPos] = sum * params.scale;
}
`, kernelVol, tileSize, kernelVol, tileSize)
}

// DispatchCNN1Scaled dispatches a non-tiled CNN1 forward pass with scale.
// For float types pass scale=1.0.
// Dispatch: X=ceil(outL/64), Y=outC, Z=batchSize
func (c *WGPUContext) DispatchCNN1Scaled(
	batchSize, inC, inL, outC, outL, kSize, stride, padding int,
	scale float32,
	inputBuf, weightBuf, outputBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderCNN1Scaled)
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
	pass.DispatchWorkgroups(
		(uint32(outL)+63)/64,
		uint32(outC),
		uint32(batchSize),
	)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

// DispatchCNN1Tiled dispatches a tiled CNN1 forward pass with shared-memory kernel caching.
//
//	tileSize  — workgroup size. Use scTile from CNN1GPUTileSizes for SC, mcTile for MC.
//	kernelVol — inC * kSize.
//	scale     — quantization scale; 1.0 for float types.
func (c *WGPUContext) DispatchCNN1Tiled(
	tileSize, kernelVol int,
	batchSize, inC, inL, outC, outL, kSize, stride, padding int,
	scale float32,
	inputBuf, weightBuf, outputBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderTiledCNN1(tileSize, kernelVol))
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
	pass.DispatchWorkgroups(
		(uint32(outL)+uint32(tileSize)-1)/uint32(tileSize),
		uint32(outC),
		uint32(batchSize),
	)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

// CNN1GPUTileSizes returns the SC and MC tile sizes for CNN1 tiling.
// Identical logic to CNN2GPUTileSizes — SC=max(GPUTileSize*4,64), MC capped at 256 and aligned to 64.
func CNN1GPUTileSizes(ctx *WGPUContext) (scTile, mcTile int) {
	scTile = ctx.GPUTileSize * 4
	if scTile < 64 {
		scTile = 64
	}

	mcTile = int(ctx.Limits.MaxComputeInvocationsPerWorkgroup)
	if mcTile <= 0 || mcTile > 256 {
		mcTile = 256
	}
	mcTile = (mcTile / 64) * 64
	if mcTile < 64 {
		mcTile = 64
	}
	if scTile >= mcTile {
		scTile = mcTile / 4
		if scTile < 64 {
			scTile = 64
		}
	}
	return
}
