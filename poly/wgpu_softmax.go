package poly

import (
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

type WGPUSoftmaxParams struct {
	BatchSize   uint32
	Size        uint32
	Temp        float32
	Type        uint32
	Rows        uint32
	Cols        uint32
	Seed        uint32
	EntmaxAlpha float32
}

const ShaderSoftmaxForward = `
struct Params {
    batchSize: u32,
    size: u32,
    temp: f32,
    softmaxType: u32,
    rows: u32,
    cols: u32,
    seed: u32,
    entmaxAlpha: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<u32>;

var<workgroup> shared_reduce: array<f32, 256>;

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) / 4294967296.0;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let b = wg_id.x;
    let tid = local_id.x;
    let size = params.size;
    let base = b * size;
    var rng_seed = params.seed + b * 1337u + tid;

    // 1. Find max for stability
    var local_max: f32 = -1e38;
    for (var i = tid; i < size; i += 256u) {
        var val = input[base + i] / params.temp;
        
        // Gumbel Noise
        if (params.softmaxType == 4u) { // Gumbel
            var u = rand_f32(&rng_seed);
            if (u < 1e-10) { u = 1e-10; }
            val = val - log(-log(u));
        }

        // Masked
        if (params.softmaxType == 5u) { // Masked
             let maskIdx = i / 32u;
             let maskBit = i % 32u;
             let is_masked = (mask[maskIdx] & (1u << maskBit)) == 0u;
             if (is_masked) { val = -1e30; }
        }

        local_max = max(local_max, val);
    }
    shared_reduce[tid] = local_max;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + s]);
        }
        workgroupBarrier();
    }
    let global_max = shared_reduce[0];
    workgroupBarrier();

    // 2. Compute Sum of Exponentials
    var local_sum: f32 = 0.0;
    for (var i = tid; i < size; i += 256u) {
        var val = input[base + i] / params.temp;
        if (params.softmaxType == 4u) { // Gumbel
             var sum_seed = params.seed + b * 1337u + tid;
             var u = rand_f32(&sum_seed);
             if (u < 1e-10) { u = 1e-10; }
             val = val - log(-log(u));
        }
        if (params.softmaxType == 5u) {
             let maskIdx = i / 32u;
             let maskBit = i % 32u;
             if ((mask[maskIdx] & (1u << maskBit)) == 0u) { val = -1e30; }
        }
        local_sum += exp(val - global_max);
    }
    shared_reduce[tid] = local_sum;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_reduce[tid] += shared_reduce[tid + s];
        }
        workgroupBarrier();
    }
    let global_sum = shared_reduce[0];
    workgroupBarrier();

    // 3. Normalize and Write Output
    for (var i = tid; i < size; i += 256u) {
        var val = input[base + i] / params.temp;
        if (params.softmaxType == 4u) {
             var out_seed = params.seed + b * 1337u + tid;
             var u = rand_f32(&out_seed);
             if (u < 1e-10) { u = 1e-10; }
             val = val - log(-log(u));
        }
        if (params.softmaxType == 5u) {
             let maskIdx = i / 32u;
             let maskBit = i % 32u;
             if ((mask[maskIdx] & (1u << maskBit)) == 0u) { val = -1e30; }
        }
        output[base + i] = exp(val - global_max) / global_sum;
    }
}
`

const ShaderSoftmaxBackward = `
struct Params {
    batchSize: u32,
    size: u32,
    temp: f32,
    softmaxType: u32,
    rows: u32,
    cols: u32,
    seed: u32,
    entmaxAlpha: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read> softmaxOutput: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradInput: array<f32>;

var<workgroup> shared_reduce: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let b = wg_id.x;
    let tid = local_id.x;
    let size = params.size;
    let base = b * size;

    // 1. Compute dot product (sum_i p_i * dL/dy_i)
    var local_dot: f32 = 0.0;
    for (var i = tid; i < size; i += 256u) {
        local_dot += gradOutput[base + i] * softmaxOutput[base + i];
    }
    shared_reduce[tid] = local_dot;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_reduce[tid] += shared_reduce[tid + s];
        }
        workgroupBarrier();
    }
    let dot_prod = shared_reduce[0];
    workgroupBarrier();

    // 2. Final gradient calculation
    for (var i = tid; i < size; i += 256u) {
        let p = softmaxOutput[base + i];
        gradInput[base + i] = (p / params.temp) * (gradOutput[base + i] - dot_prod);
    }
}
`

func (c *WGPUContext) DispatchSoftmaxForward(l *VolumetricLayer, batchSize int, inputBuf, outBuf *wgpu.Buffer) error {
	pipeline, err := c.CreateComputePipeline(ShaderSoftmaxForward)
	if err != nil {
		return err
	}

	temp := l.Temperature
	if temp == 0 {
		temp = 1.0
	}

	params := WGPUSoftmaxParams{
		BatchSize:   uint32(batchSize),
		Size:        uint32(l.OutputHeight),
		Temp:        float32(temp),
		Type:        uint32(l.SoftmaxType),
		Rows:        uint32(l.SoftmaxRows),
		Cols:        uint32(l.SoftmaxCols),
		Seed:        uint32(c.FrameCount), // Use FrameCount as randomized seed base
		EntmaxAlpha: float32(l.EntmaxAlpha),
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(params)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUSoftmaxParams{params}))

	// Mask handling
	var maskBuf *wgpu.Buffer
	if len(l.Mask) > 0 {
		// Pack bool slice into bitmask uint32 slice
		uints := make([]uint32, (len(l.Mask)+31)/32)
		for i, b := range l.Mask {
			if b {
				uints[i/32] |= (1 << (i % 32))
			}
		}
		m, _ := c.CreatePersistentBufferUint32(uints, "Softmax Mask")
		maskBuf = m
	} else {
		// Empty mask buffer (identity)
		dummy := make([]uint32, (l.OutputHeight+31)/32)
		for i := range dummy {
			dummy[i] = 0xFFFFFFFF
		}
		m, _ := c.CreatePersistentBufferUint32(dummy, "Dummy Mask")
		maskBuf = m
	}

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, inputBuf, outBuf, maskBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups(uint32(batchSize), 1, 1)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}

func (c *WGPUContext) DispatchSoftmaxBackward(batchSize, size int, temp float32, gradOutputBuf, softmaxOutputBuf, gradInputBuf *wgpu.Buffer) error {
	pipeline, err := c.CreateComputePipeline(ShaderSoftmaxBackward)
	if err != nil {
		return err
	}

	params := WGPUSoftmaxParams{
		BatchSize: uint32(batchSize),
		Size:      uint32(size),
		Temp:      temp,
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(params)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUSoftmaxParams{params}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, gradOutputBuf, softmaxOutputBuf, gradInputBuf)
	if err != nil {
		return err
	}

	enc, owned, _ := ctxEncoder(c)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups(uint32(batchSize), 1, 1)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}
