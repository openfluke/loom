package poly

import (
	"fmt"
	"runtime"

	"github.com/openfluke/webgpu/wgpu"
)

// WGPUArgMaxParams matches ShaderArgMax.
type WGPUArgMaxParams struct {
	Length uint32
	_      [3]uint32 // 16-byte uniform alignment
}

// ShaderArgMax reduces a 1D logits vector to a single greedy token id (u32).
// One workgroup (256 lanes) scans the full vocab — enough for SmolLM (~49k) and Qwen-class vocabs.
const ShaderArgMax = `
struct Params {
    length: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_token: array<u32>;

var<workgroup> shared_val: array<f32, 256>;
var<workgroup> shared_idx: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    var best_v: f32 = -3.402823466e+38;
    var best_i: u32 = 0u;
    var i = lane;
    loop {
        if (i >= params.length) { break; }
        let v = logits[i];
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
        i += 256u;
    }
    shared_val[lane] = best_v;
    shared_idx[lane] = best_i;
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            let other = lane + stride;
            if (shared_val[other] > shared_val[lane]) {
                shared_val[lane] = shared_val[other];
                shared_idx[lane] = shared_idx[other];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lane == 0u) {
        out_token[0] = shared_idx[0];
    }
}
`

// ShaderAdvanceGreedy: step=[pos, outCount]. Store outTok→hist, token=outTok, pos++, outCount++.
const ShaderAdvanceGreedy = `
@group(0) @binding(0) var<storage, read_write> step: array<u32>;
@group(0) @binding(1) var<storage, read> outTok: array<u32>;
@group(0) @binding(2) var<storage, read_write> history: array<u32>;
@group(0) @binding(3) var<storage, read_write> token: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    let pos = step[0];
    let oc = step[1];
    let tok = outTok[0];
    history[oc] = tok;
    token[0] = tok;
    step[0] = pos + 1u;
    step[1] = oc + 1u;
}
`

// RoPE with GPU-resident position (step[0]) — stable uniforms for chunked decode.
const ShaderRoPEStep = `
struct RoPEParams {
    seqLen: u32,
    headDim: u32,
    numHeads: u32,
    _pad: u32,
    theta: f32,
};
@group(0) @binding(0) var<uniform> params: RoPEParams;
@group(0) @binding(1) var<storage, read> step: array<u32>;
@group(0) @binding(2) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let halfDim = params.headDim / 2u;
    let totalPairs = params.seqLen * params.numHeads * halfDim;
    if (tid >= totalPairs) { return; }
    let d = tid % halfDim;
    let h = (tid / halfDim) % params.numHeads;
    let s = tid / (halfDim * params.numHeads);
    let pos = f32(step[0] + s);
    let freq = 1.0 / pow(params.theta, f32(2u * d) / f32(params.headDim));
    let angle = pos * freq;
    let cos_val = cos(angle);
    let sin_val = sin(angle);
    let idx0 = (s * params.numHeads + h) * params.headDim + d;
    let idx1 = idx0 + halfDim;
    let v0 = data[idx0];
    let v1 = data[idx1];
    data[idx0] = v0 * cos_val - v1 * sin_val;
    data[idx1] = v0 * sin_val + v1 * cos_val;
}
`

const ShaderKVUpdateStep = `
struct KVParams {
    headDim: u32,
    maxSeqLen: u32,
    numKVHeads: u32,
    numTokens: u32,
};
@group(0) @binding(0) var<uniform> params: KVParams;
@group(0) @binding(1) var<storage, read> step: array<u32>;
@group(0) @binding(2) var<storage, read_write> kCache: array<f32>;
@group(0) @binding(3) var<storage, read_write> vCache: array<f32>;
@group(0) @binding(4) var<storage, read> newK: array<f32>;
@group(0) @binding(5) var<storage, read> newV: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let kvDim = params.numKVHeads * params.headDim;
    if (tid >= kvDim * params.numTokens) { return; }
    let tokenIdx = tid / kvDim;
    let dimIdx = tid % kvDim;
    let h = dimIdx / params.headDim;
    let d = dimIdx % params.headDim;
    let cacheIdx = (h * params.maxSeqLen + step[0] + tokenIdx) * params.headDim + d;
    kCache[cacheIdx] = newK[tid];
    vCache[cacheIdx] = newV[tid];
}
`

// DispatchAdvanceGreedy appends outTok to history and updates the live decode token on GPU.
func (c *WGPUContext) DispatchAdvanceGreedy(stepBuf, outTokBuf, histBuf, tokenBuf any) error {
	pipeline, err := c.CreateComputePipeline(ShaderAdvanceGreedy)
	if err != nil {
		return err
	}
	bindGroup, err := c.GetBindGroup(pipeline, stepBuf, outTokBuf, histBuf, tokenBuf)
	if err != nil {
		return err
	}
	return c.dispatchCompute(pipeline, bindGroup, 1, 1, 1)
}

// DispatchArgMax writes argmax(logits[0:length]) into outTokenBuf[0] (u32).
func (c *WGPUContext) DispatchArgMax(length int, logitsBuf, outTokenBuf any) error {
	if length <= 0 {
		return fmt.Errorf("DispatchArgMax: length must be > 0")
	}
	pipeline, err := c.CreateComputePipeline(ShaderArgMax)
	if err != nil {
		return err
	}
	params := WGPUArgMaxParams{Length: uint32(length)}
	pBuf := c.WriteUniformBytes(wgpu.ToBytes([]WGPUArgMaxParams{params}))
	bindGroup, err := c.GetBindGroup(pipeline, pBuf, logitsBuf, outTokenBuf)
	if err != nil {
		return err
	}
	return c.dispatchCompute(pipeline, bindGroup, 1, 1, 1)
}

// pollMapRead blocks until a MapAsync staging buffer is ready (same pattern as ForwardTokenIDsWGPU).
func (c *WGPUContext) pollMapRead(staging *wgpu.Buffer, size uint64) ([]byte, error) {
	done := make(chan struct{})
	if err := staging.MapAsync(wgpu.MapModeRead, 0, size, func(status wgpu.BufferMapAsyncStatus) {
		close(done)
	}); err != nil {
		return nil, err
	}
	for {
		c.Device.Poll(false, nil)
		select {
		case <-done:
			data := staging.GetMappedRange(0, uint(size))
			out := make([]byte, size)
			copy(out, data[:size])
			staging.Unmap()
			return out, nil
		default:
			runtime.Gosched()
		}
	}
}
