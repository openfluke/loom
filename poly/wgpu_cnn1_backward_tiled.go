package poly

import (
	"fmt"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// WGPUCNN1Bwd1DParams is the uniform struct for tiled CNN1 backward shaders.
// 16 × uint32 = 64 bytes.
type WGPUCNN1Bwd1DParams struct {
	BatchSize  uint32
	InC        uint32
	InL        uint32
	Filters    uint32
	OutL       uint32
	KSize      uint32
	Stride     uint32
	Padding    uint32
	Activation uint32
	Pad1       uint32
	Pad2       uint32
	Pad3       uint32
	Pad4       uint32
	Pad5       uint32
	Pad6       uint32
	Pad7       uint32
}

const wgslCNN1Bwd1DParamsStruct = `
struct CNN1Bwd1DParams {
    batchSize: u32,
    inC: u32, inL: u32,
    filters: u32, outL: u32,
    kSize: u32, stride: u32, padding: u32,
    activation: u32,
    _p1: u32, _p2: u32, _p3: u32, _p4: u32, _p5: u32, _p6: u32, _p7: u32,
};`

// ShaderTiledCNN1BackwardDX generates a tiled dX WGSL shader for CNN1.
//
// Each workgroup covers tileSize input elements (ic, inPos) for one batch item.
// For every filter the threads cooperatively load kernelVol weights into shared memory,
// then each thread accumulates its gradient contribution.
//
// Dispatch: X=ceil(inC*inL/tileSize), Y=1, Z=batchSize
func ShaderTiledCNN1BackwardDX(tileSize, kernelVol int) string {
	return fmt.Sprintf(wgslCNN1Bwd1DParamsStruct+`
@group(0) @binding(0) var<uniform>             params:     CNN1Bwd1DParams;
@group(0) @binding(1) var<storage, read>       gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read>       weights:    array<f32>;
@group(0) @binding(3) var<storage, read>       preAct:     array<f32>;
@group(0) @binding(4) var<storage, read_write> gradInput:  array<f32>;

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
    let inVol      = params.inC * params.inL;
    if (batchIdx >= params.batchSize) { return; }

    // Decode flat input position -> (ic, inPos)
    let ic    = inElemFlat / params.inL;
    let inPos = inElemFlat %% params.inL;

    var sum: f32 = 0.0;

    // Loop over all filters; cooperative kernel cache load per filter.
    for (var f: u32 = 0u; f < params.filters; f++) {
        var i: u32 = local_id.x;
        loop {
            if (i >= kVol) { break; }
            wCache[i] = weights[f * kVol + i];
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
        gradInput[batchIdx * inVol + inElemFlat] += sum;
    }
}
`, kernelVol, tileSize, kernelVol, tileSize)
}

// ShaderTiledCNN1BackwardDW generates a tiled dW WGSL shader for CNN1.
//
// Each workgroup covers tileSize weight elements for one filter.
// Dispatch: X=ceil(kernelVol/tileSize), Y=filters, Z=1
func ShaderTiledCNN1BackwardDW(tileSize int) string {
	return fmt.Sprintf(wgslCNN1Bwd1DParamsStruct+`
@group(0) @binding(0) var<uniform>             params:      CNN1Bwd1DParams;
@group(0) @binding(1) var<storage, read>       gradOutput:  array<f32>;
@group(0) @binding(2) var<storage, read>       input:       array<f32>;
@group(0) @binding(3) var<storage, read>       preAct:      array<f32>;
@group(0) @binding(4) var<storage, read_write> gradWeights: array<f32>;

var<workgroup> dyCache: array<f32, %d>;
`+wgslBwdActivateDeriv+`

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id:  vec3<u32>,
) {
    let kernelPos  = global_id.x; // index into [0, inC*kSize)
    let f          = global_id.y;
    let kVol       = params.inC * params.kSize;
    if (f >= params.filters) { return; }

    // Decode kernelPos -> (ic, k)
    let ic = kernelPos / params.kSize;
    let k  = kernelPos %% params.kSize;

    let totalSpatial = params.batchSize * params.outL;
    var sum: f32     = 0.0;

    var spatial: u32 = 0u;
    loop {
        if (spatial >= totalSpatial) { break; }

        let loadIdx = spatial + local_id.x;
        if (loadIdx < totalSpatial) {
            let lb     = loadIdx / params.outL;
            let lOutPos = loadIdx %% params.outL;
            let lIdx   = lb * params.filters * params.outL + f * params.outL + lOutPos;
            dyCache[local_id.x] = gradOutput[lIdx] * activateDerivative(preAct[lIdx], params.activation);
        } else {
            dyCache[local_id.x] = 0.0;
        }
        workgroupBarrier();

        if (kernelPos < kVol) {
            for (var ti: u32 = 0u; ti < %du; ti++) {
                let bSpatial = spatial + ti;
                if (bSpatial >= totalSpatial) { break; }
                let b      = bSpatial / params.outL;
                let outPos = bSpatial %% params.outL;

                let inPos_i = i32(outPos * params.stride + k) - i32(params.padding);
                if (inPos_i >= 0 && u32(inPos_i) < params.inL) {
                    let inIdx = (b * params.inC + ic) * params.inL + u32(inPos_i);
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

// DispatchCNN1TiledBackwardDX dispatches a tiled dX backward for CNN1.
// gradInputBuf must be pre-zeroed; the shader accumulates (+=) into it.
func (c *WGPUContext) DispatchCNN1TiledBackwardDX(
	tileSize, kernelVol int,
	batchSize, inC, inL, filters, outL, kSize, stride, padding int,
	activation ActivationType,
	gradOutputBuf, weightBuf, preActBuf, gradInputBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderTiledCNN1BackwardDX(tileSize, kernelVol))
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
	pass.DispatchWorkgroups(
		(uint32(inC*inL)+uint32(tileSize)-1)/uint32(tileSize),
		1,
		uint32(batchSize),
	)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

// DispatchCNN1TiledBackwardDW dispatches a tiled dW backward for CNN1.
// gradWeightsBuf must be pre-zeroed; the shader accumulates (+=) into it.
func (c *WGPUContext) DispatchCNN1TiledBackwardDW(
	tileSize int,
	batchSize, inC, inL, filters, outL, kSize, stride, padding int,
	activation ActivationType,
	gradOutputBuf, inputBuf, preActBuf, gradWeightsBuf *wgpu.Buffer,
) error {
	pipeline, err := c.CreateComputePipeline(ShaderTiledCNN1BackwardDW(tileSize))
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

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, inputBuf, preActBuf, gradWeightsBuf)
	if err != nil {
		return err
	}

	kernelVol := inC * kSize
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
