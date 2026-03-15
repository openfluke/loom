package poly

import "fmt"

// WGSL Shaders for Training (Backward Pass)
//
// These shaders implement the gradient calculations (dx, dw) for core layers.

// ShaderDenseBackwardDX calculates gradInput = gradOutput * weights
// dx = dy * W^T  => dx[b, i] = sum_o dy[b, o] * W[o, i]
func ShaderDenseBackwardDX(tileSize int) string {
	ts2 := tileSize * tileSize
	return fmt.Sprintf(`
struct Params {
    batchSize: u32,
    inputSize: u32,
    outputSize: u32,
    tileSize: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradInput: array<f32>;

var<workgroup> dyTile: array<f32, %d>;
var<workgroup> wTile: array<f32, %d>;

@compute @workgroup_size(%d, %d, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let i = global_id.x;
    let b = global_id.y;
    let tx = local_id.x;
    let ty = local_id.y;

    var sum: f32 = 0.0;
    let numTiles = (params.outputSize + params.tileSize - 1) / params.tileSize;

    for (var t: u32 = 0u; t < numTiles; t++) {
        // Load dy[b, o] and W[o, i] into shared memory
        let o_idx = t * params.tileSize + tx;
        if (b < params.batchSize && o_idx < params.outputSize) {
            dyTile[ty * params.tileSize + tx] = gradOutput[b * params.outputSize + o_idx];
        } else {
            dyTile[ty * params.tileSize + tx] = 0.0;
        }

        let w_o_idx = t * params.tileSize + ty;
        if (i < params.inputSize && w_o_idx < params.outputSize) {
            wTile[ty * params.tileSize + tx] = weights[w_o_idx * params.inputSize + i];
        } else {
            wTile[ty * params.tileSize + tx] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < params.tileSize; k++) {
            sum += dyTile[ty * params.tileSize + k] * wTile[k * params.tileSize + tx];
        }

        workgroupBarrier();
    }

    if (b < params.batchSize && i < params.inputSize) {
        gradInput[b * params.inputSize + i] = sum;
    }
}
`, ts2, ts2, tileSize, tileSize)
}

// ShaderDenseBackwardDW calculates gradWeights = gradOutput^T * input
// dw = dy^T * x => dw[o, i] = sum_b dy[b, o] * x[b, i]
func ShaderDenseBackwardDW(tileSize int) string {
	ts2 := tileSize * tileSize
	return fmt.Sprintf(`
struct Params {
    batchSize: u32,
    inputSize: u32,
    outputSize: u32,
    tileSize: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradWeights: array<f32>;

var<workgroup> dyTile: array<f32, %d>;
var<workgroup> xTile: array<f32, %d>;

@compute @workgroup_size(%d, %d, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let i = global_id.x;
    let o = global_id.y;
    let tx = local_id.x;
    let ty = local_id.y;

    var sum: f32 = 0.0;
    let numTiles = (params.batchSize + params.tileSize - 1) / params.tileSize;

    for (var t: u32 = 0u; t < numTiles; t++) {
        let b_idx = t * params.tileSize + tx;
        if (o < params.outputSize && b_idx < params.batchSize) {
            dyTile[ty * params.tileSize + tx] = gradOutput[b_idx * params.outputSize + o];
        } else {
            dyTile[ty * params.tileSize + tx] = 0.0;
        }

        let bx_idx = t * params.tileSize + ty;
        if (i < params.inputSize && bx_idx < params.batchSize) {
            xTile[ty * params.tileSize + tx] = input[bx_idx * params.inputSize + i];
        } else {
            xTile[ty * params.tileSize + tx] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < params.tileSize; k++) {
            sum += dyTile[k * params.tileSize + ty] * xTile[k * params.tileSize + tx];
        }

        workgroupBarrier();
    }

    if (o < params.outputSize && i < params.inputSize) {
        gradWeights[o * params.inputSize + i] += sum;
    }
}
`, ts2, ts2, tileSize, tileSize)
}

const ShaderRMSNormBackward = `
struct Params {
    size: u32,
    epsilon: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read> rms: array<f32>;
@group(0) @binding(4) var<storage, read> weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> gradInput: array<f32>;
@group(0) @binding(6) var<storage, read_write> gradWeights: array<f32>;

var<workgroup> shared_sum: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let b = wg_id.x;
    let tid = local_id.x;
    let size = params.size;

    let b_rms = rms[b];
    let invRMS = 1.0 / b_rms;
    let invRMS3 = 1.0 / (b_rms * b_rms * b_rms);

    var local_sum_dxhat_x: f32 = 0.0;

    for (var i: u32 = tid; i < size; i += 64u) {
        let idx = b * size + i;
        let dy = gradOutput[idx];
        let x = input[idx];
        let xhat = x * invRMS;
        let g = weights[i];

        // Accumulate gradWeights (gamma)
        // Note: For multi-batch, this needs workgroup reduction + atomic add
        // For simplicity in this version, we assume single WG per batch or atomics
        gradWeights[i] += dy * xhat;
        
        local_sum_dxhat_x += dy * g * x;
    }
    shared_sum[tid] = local_sum_dxhat_x;
    workgroupBarrier();

    if (tid == 0u) {
        var total: f32 = 0.0;
        for (var idx_sum: u32 = 0u; idx_sum < 64u; idx_sum++) {
            total += shared_sum[idx_sum];
        }
        shared_sum[0] = total;
    }
    workgroupBarrier();

    let term2 = (shared_sum[0] * invRMS3) / f32(size);

    for (var i: u32 = tid; i < size; i += 64u) {
        let idx = b * size + i;
        let dy = gradOutput[idx];
        let x = input[idx];
        let g = weights[i];

        gradInput[idx] = (dy * g * invRMS) - (x * term2);
    }
}
`

const ShaderSwiGLUBackward = `
struct Params {
    batchSize: u32,
    inputSize: u32,
    outputSize: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read> gateIn: array<f32>;
@group(0) @binding(3) var<storage, read> upIn: array<f32>;
@group(0) @binding(4) var<storage, read_write> gradGate: array<f32>;
@group(0) @binding(5) var<storage, read_write> gradUp: array<f32>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.batchSize * params.outputSize) { return; }

    let dy = gradOutput[idx];
    let g = gateIn[idx];
    let u = upIn[idx];

    let sig = sigmoid(g);
    let silu = g * sig;
    let dSilu = sig * (1.0 + g * (1.0 - sig));

    gradUp[idx] = dy * silu;
    gradGate[idx] = dy * u * dSilu;
}
`

const ShaderEmbeddingBackward = `
struct Params {
    vocabSize: u32,
    hiddenSize: u32,
    numTokens: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradWeights: array<f32>;

// Note: This needs atomic additions to avoid race conditions when multiple
// tokens use the same vocab index. WebGPU supports atomicAdd on i32/u32,
// but not f32 (yet, in most backends).
// For now, we'll use a simple scatter which might have races, or handle it
// as a separate pass per vocab index.
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid >= params.numTokens * params.hiddenSize) { return; }
    
    let tokenIdx = tid / params.hiddenSize;
    let dimIdx = tid % params.hiddenSize;
    let vocabIdx = indices[tokenIdx];
    
    if (vocabIdx < params.vocabSize) {
        // RACE CONDITION: Multiple threads might write to the same weight
        // if multiple tokens have the same vocabIdx.
        // In a production kernel, we'd use atomics or a deterministic sort-reduce.
        gradWeights[vocabIdx * params.hiddenSize + dimIdx] += gradOutput[tid];
    }
}
`

const ShaderResidualBackward = `
struct Params {
    size: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradInput: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradResidual: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.size) { return; }
    let dy = gradOutput[i];
    gradInput[i] += dy;
    gradResidual[i] += dy;
}
`
const wgslActivateDerivative = `
fn activateDerivative(v: f32, act: u32) -> f32 {
    if (act == 0u) { // ReLU
        if (v <= 0.0) { return 0.0; }
        return 1.0;
    }
    if (act == 1u) { // SiLU
        let sig = 1.0 / (1.0 + exp(-v));
        return sig * (1.0 + v * (1.0 - sig));
    }
    if (act == 3u) { // Tanh
        let t = tanh(v);
        return 1.0 - t * t;
    }
    if (act == 4u) { // Sigmoid
        let s = 1.0 / (1.0 + exp(-v));
        return s * (1.0 - s);
    }
    return 1.0; // Linear/Default
}
`

const ShaderCNN1BackwardDX = `
struct Params {
    batchSize: u32,
    inC: u32,
    inL: u32,
    filters: u32,
    outL: u32,
    kSize: u32,
    stride: u32,
    padding: u32,
    activation: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> preAct: array<f32>;
@group(0) @binding(4) var<storage, read_write> gradInput: array<f32>;

` + wgslActivateDerivative + `

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid >= params.batchSize * params.inC * params.inL) { return; }

    let b = tid / (params.inC * params.inL);
    let rem = tid % (params.inC * params.inL);
    let ic = rem / params.inL;
    let ip = rem % params.inL;

    var sum: f32 = 0.0;
    for (var f: u32 = 0u; f < params.filters; f++) {
        for (var k: u32 = 0u; k < params.kSize; k++) {
            let val = i32(ip) + i32(params.padding) - i32(k);
            if (val >= 0 && val % i32(params.stride) == 0) {
                let o = u32(val / i32(params.stride));
                if (o < params.outL) {
                    let outIdx = b * params.filters * params.outL + f * params.outL + o;
                    let dy = gradOutput[outIdx] * activateDerivative(preAct[outIdx], params.activation);
                    let kWIdx = f * params.inC * params.kSize + ic * params.kSize + k;
                    sum += dy * weights[kWIdx];
                }
            }
        }
    }
    gradInput[tid] += sum;
}
`

const ShaderCNN1BackwardDW = `
struct Params {
    batchSize: u32,
    inC: u32,
    inL: u32,
    filters: u32,
    outL: u32,
    kSize: u32,
    stride: u32,
    padding: u32,
    activation: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read> preAct: array<f32>;
@group(0) @binding(4) var<storage, read_write> gradWeights: array<f32>;

` + wgslActivateDerivative + `

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid >= params.filters * params.inC * params.kSize) { return; }

    let f = tid / (params.inC * params.kSize);
    let rem = tid % (params.inC * params.kSize);
    let ic = rem / params.kSize;
    let k = rem % params.kSize;

    var sum: f32 = 0.0;
    for (var b: u32 = 0u; b < params.batchSize; b++) {
        for (var o: u32 = 0u; o < params.outL; o++) {
            let inPos = i32(o * params.stride) + i32(k) - i32(params.padding);
            if (inPos >= 0 && inPos < i32(params.inL)) {
                let outIdx = b * params.filters * params.outL + f * params.outL + o;
                let dy = gradOutput[outIdx] * activateDerivative(preAct[outIdx], params.activation);
                let inIdx = b * params.inC * params.inL + ic * params.inL + u32(inPos);
                sum += dy * input[inIdx];
            }
        }
    }
    gradWeights[tid] += sum;
}
`

const ShaderCNN2BackwardDX = `
struct Params {
    batchSize: u32,
    inC: u32,
    inH: u32,
    inW: u32,
    filters: u32,
    outH: u32,
    outW: u32,
    kSize: u32,
    stride: u32,
    padding: u32,
    activation: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> preAct: array<f32>;
@group(0) @binding(4) var<storage, read_write> gradInput: array<f32>;

` + wgslActivateDerivative + `

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let size = params.batchSize * params.inC * params.inH * params.inW;
    if (tid >= size) { return; }

    let b = tid / (params.inC * params.inH * params.inW);
    let rem = tid % (params.inC * params.inH * params.inW);
    let ic = rem / (params.inH * params.inW);
    let rem2 = rem % (params.inH * params.inW);
    let ih = rem2 / params.inW;
    let iw = rem2 % params.inW;

    var sum: f32 = 0.0;
    for (var f: u32 = 0u; f < params.filters; f++) {
        for (var kh: u32 = 0u; kh < params.kSize; kh++) {
            for (var kw: u32 = 0u; kw < params.kSize; kw++) {
                let vh = i32(ih) + i32(params.padding) - i32(kh);
                let vw = i32(iw) + i32(params.padding) - i32(kw);
                if (vh >= 0 && vh % i32(params.stride) == 0 && vw >= 0 && vw % i32(params.stride) == 0) {
                    let oh = u32(vh / i32(params.stride));
                    let ow = u32(vw / i32(params.stride));
                    if (oh < params.outH && ow < params.outW) {
                        let outIdx = ((b * params.filters + f) * params.outH + oh) * params.outW + ow;
                        let dy = gradOutput[outIdx] * activateDerivative(preAct[outIdx], params.activation);
                        let kWIdx = ((f * params.inC + ic) * params.kSize + kh) * params.kSize + kw;
                        sum += dy * weights[kWIdx];
                    }
                }
            }
        }
    }
    gradInput[tid] += sum;
}
`

const ShaderCNN2BackwardDW = `
struct Params {
    batchSize: u32,
    inC: u32,
    inH: u32,
    inW: u32,
    filters: u32,
    outH: u32,
    outW: u32,
    kSize: u32,
    stride: u32,
    padding: u32,
    activation: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read> preAct: array<f32>;
@group(0) @binding(4) var<storage, read_write> gradWeights: array<f32>;

` + wgslActivateDerivative + `

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let weightSize = params.filters * params.inC * params.kSize * params.kSize;
    if (tid >= weightSize) { return; }

    let f = tid / (params.inC * params.kSize * params.kSize);
    let rem = tid % (params.inC * params.kSize * params.kSize);
    let ic = rem / (params.kSize * params.kSize);
    let rem2 = rem % (params.kSize * params.kSize);
    let kh = rem2 / params.kSize;
    let kw = rem2 % params.kSize;

    var sum: f32 = 0.0;
    for (var b: u32 = 0u; b < params.batchSize; b++) {
        for (var oh: u32 = 0u; oh < params.outH; oh++) {
            for (var ow: u32 = 0u; ow < params.outW; ow++) {
                let ih = i32(oh * params.stride) + i32(kh) - i32(params.padding);
                let iw = i32(ow * params.stride) + i32(kw) - i32(params.padding);
                if (ih >= 0 && ih < i32(params.inH) && iw >= 0 && iw < i32(params.inW)) {
                    let outIdx = ((b * params.filters + f) * params.outH + oh) * params.outW + ow;
                    let dy = gradOutput[outIdx] * activateDerivative(preAct[outIdx], params.activation);
                    let inIdx = ((b * params.inC + ic) * params.inH + u32(ih)) * params.inW + u32(iw);
                    sum += dy * input[inIdx];
                }
            }
        }
    }
    gradWeights[tid] += sum;
}
`

const ShaderCNN3BackwardDX = `
struct Params {
    batchSize: u32,
    inC: u32,
    inD: u32,
    inH: u32,
    inW: u32,
    filters: u32,
    outD: u32,
    outH: u32,
    outW: u32,
    kSize: u32,
    stride: u32,
    padding: u32,
    activation: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> preAct: array<f32>;
@group(0) @binding(4) var<storage, read_write> gradInput: array<f32>;

` + wgslActivateDerivative + `

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let inVol = params.inD * params.inH * params.inW;
    let size = params.batchSize * params.inC * inVol;
    if (tid >= size) { return; }

    let b = tid / (params.inC * inVol);
    let rem = tid % (params.inC * inVol);
    let ic = rem / inVol;
    let rem2 = rem % inVol;
    let id = rem2 / (params.inH * params.inW);
    let rem3 = rem2 % (params.inH * params.inW);
    let ih = rem3 / params.inW;
    let iw = rem3 % params.inW;

    var sum: f32 = 0.0;
    for (var f: u32 = 0u; f < params.filters; f++) {
        for (var kd: u32 = 0u; kd < params.kSize; kd++) {
            for (var kh: u32 = 0u; kh < params.kSize; kh++) {
                for (var kw: u32 = 0u; kw < params.kSize; kw++) {
                    let vd = i32(id) + i32(params.padding) - i32(kd);
                    let vh = i32(ih) + i32(params.padding) - i32(kh);
                    let vw = i32(iw) + i32(params.padding) - i32(kw);
                    if (vd >= 0 && vd % i32(params.stride) == 0 && 
                        vh >= 0 && vh % i32(params.stride) == 0 && 
                        vw >= 0 && vw % i32(params.stride) == 0) {
                        let od = u32(vd / i32(params.stride));
                        let oh = u32(vh / i32(params.stride));
                        let ow = u32(vw / i32(params.stride));
                        if (od < params.outD && oh < params.outH && ow < params.outW) {
                            let outIdx = (((b * params.filters + f) * params.outD + od) * params.outH + oh) * params.outW + ow;
                            let dy = gradOutput[outIdx] * activateDerivative(preAct[outIdx], params.activation);
                            let kWIdx = (((f * params.inC + ic) * params.kSize + kd) * params.kSize + kh) * params.kSize + kw;
                            sum += dy * weights[kWIdx];
                        }
                    }
                }
            }
        }
    }
    gradInput[tid] += sum;
}
`

const ShaderCNN3BackwardDW = `
struct Params {
    batchSize: u32,
    inC: u32,
    inD: u32,
    inH: u32,
    inW: u32,
    filters: u32,
    outD: u32,
    outH: u32,
    outW: u32,
    kSize: u32,
    stride: u32,
    padding: u32,
    activation: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read> preAct: array<f32>;
@group(0) @binding(4) var<storage, read_write> gradWeights: array<f32>;

` + wgslActivateDerivative + `

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let kVol = params.kSize * params.kSize * params.kSize;
    let weightSize = params.filters * params.inC * kVol;
    if (tid >= weightSize) { return; }

    let f = tid / (params.inC * kVol);
    let rem = tid % (params.inC * kVol);
    let ic = rem / kVol;
    let rem2 = rem % kVol;
    let kd = rem2 / (params.kSize * params.kSize);
    let rem3 = rem2 % (params.kSize * params.kSize);
    let kh = rem3 / params.kSize;
    let kw = rem3 % params.kSize;

    var sum: f32 = 0.0;
    for (var b: u32 = 0u; b < params.batchSize; b++) {
        for (var od: u32 = 0u; od < params.outD; od++) {
            for (var oh: u32 = 0u; oh < params.outH; oh++) {
                for (var ow: u32 = 0u; ow < params.outW; ow++) {
                    let id = i32(od * params.stride) + i32(kd) - i32(params.padding);
                    let ih = i32(oh * params.stride) + i32(kh) - i32(params.padding);
                    let iw = i32(ow * params.stride) + i32(kw) - i32(params.padding);
                    if (id >= 0 && id < i32(params.inD) &&
                        ih >= 0 && ih < i32(params.inH) &&
                        iw >= 0 && iw < i32(params.inW)) {
                        let outIdx = (((b * params.filters + f) * params.outD + od) * params.outH + oh) * params.outW + ow;
                        let dy = gradOutput[outIdx] * activateDerivative(preAct[outIdx], params.activation);
                        let inIdx = (((b * params.inC + ic) * params.inD + u32(id)) * params.inH + u32(ih)) * params.inW + u32(iw);
                        sum += dy * input[inIdx];
                    }
                }
            }
        }
    }
    gradWeights[tid] += sum;
}
`

const ShaderMHABackward = `
struct Params {
    batchSize: u32,
    numHeads: u32,
    numKVHeads: u32,
    headDim: u32,
    seqLen: u32,
    scale: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> K: array<f32>;
@group(0) @binding(4) var<storage, read> V: array<f32>;
@group(0) @binding(5) var<storage, read_write> dQ: array<f32>;
@group(0) @binding(6) var<storage, read_write> dK: array<f32>;
@group(0) @binding(7) var<storage, read_write> dV: array<f32>;

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let q_idx = global_id.x; 
    let head_size = params.seqLen * params.numHeads;
    let b = q_idx / head_size;
    let rem = q_idx % head_size;
    let h = rem / params.seqLen;
    let s_q = rem % params.seqLen;

    if (b >= params.batchSize) { return; }

    let headDim = params.headDim;
    let kvHead = h / (params.numHeads / params.numKVHeads);
    
    // 1. Recompute scores and softmax for this query
    var scores: array<f32, 512>; 
    var max_score: f32 = -1e9;
    
    for (var s_k: u32 = 0u; s_k < params.seqLen; s_k++) {
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < headDim; d++) {
            dot += Q[((b * params.numHeads + h) * params.seqLen + s_q) * headDim + d] * 
                   K[((b * params.numKVHeads + kvHead) * params.seqLen + s_k) * headDim + d];
        }
        let score = dot * params.scale;
        scores[s_k] = score;
        if (score > max_score) { max_score = score; }
    }
    
    var exp_sum: f32 = 0.0;
    for (var s_k: u32 = 0u; s_k < params.seqLen; s_k++) {
        scores[s_k] = exp(scores[s_k] - max_score);
        exp_sum += scores[s_k];
    }
    for (var s_k: u32 = 0u; s_k < params.seqLen; s_k++) {
        scores[s_k] /= exp_sum;
    }
    
    // 2. Compute dScores / dV contributions
    var d_softmax: array<f32, 512>;
    for (var s_k: u32 = 0u; s_k < params.seqLen; s_k++) {
        var ds: f32 = 0.0;
        for (var d: u32 = 0u; d < headDim; d++) {
            ds += gradOutput[((b * params.numHeads + h) * params.seqLen + s_q) * headDim + d] *
                  V[((b * params.numKVHeads + kvHead) * params.seqLen + s_k) * headDim + d];
        }
        d_softmax[s_k] = ds;
    }
    
    var sum_ds_s: f32 = 0.0;
    for (var s_k: u32 = 0u; s_k < params.seqLen; s_k++) {
        sum_ds_s += d_softmax[s_k] * scores[s_k];
    }
    
    for (var s_k: u32 = 0u; s_k < params.seqLen; s_k++) {
        let d_logit = (d_softmax[s_k] - sum_ds_s) * scores[s_k];
        
        for (var d: u32 = 0u; d < headDim; d++) {
            let q_off = ((b * params.numHeads + h) * params.seqLen + s_q) * headDim + d;
            let k_off = ((b * params.numKVHeads + kvHead) * params.seqLen + s_k) * headDim + d;
            let v_off = ((b * params.numKVHeads + kvHead) * params.seqLen + s_k) * headDim + d;
            
            dQ[q_off] += d_logit * params.scale * K[k_off];
            dK[k_off] += d_logit * params.scale * Q[q_off];
            dV[v_off] += scores[s_k] * gradOutput[q_off];
        }
    }
}
`

const ShaderApplyGradients = `
struct Params {
    size: u32,
    lr: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<storage, read> gradients: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid >= params.size) { return; }
    
    // Simple SGD: w = w - lr * g
    weights[tid] -= params.lr * gradients[tid];
}
`

const ShaderActivationForward = `
struct Params {
    size: u32,
    act: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn silu(x: f32) -> f32 { return x * (1.0 / (1.0 + exp(-x))); }

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid >= params.size) { return; }
    let x = input[tid];
    var res: f32 = x;
    if (params.act == 0u) { res = max(0.0, x); }      // ReLU
    else if (params.act == 1u) { res = silu(x); }     // SiLU
    else if (params.act == 3u) { res = tanh(x); }     // Tanh
    else if (params.act == 4u) { res = 1.0 / (1.0 + exp(-x)); } // Sigmoid
    output[tid] = res;
}
`

const ShaderActivationBackward = `
struct Params {
    size: u32,
    act: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read> preAct: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradInput: array<f32>;

fn silu_deriv(x: f32) -> f32 {
    let s = 1.0 / (1.0 + exp(-x));
    return s + x * s * (1.0 - s);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid >= params.size) { return; }
    let x = preAct[tid];
    let g = gradOutput[tid];
    var d: f32 = 1.0;
    if (params.act == 0u) { if (x <= 0.0) { d = 0.0; } } // ReLU
    else if (params.act == 1u) { d = silu_deriv(x); }    // SiLU
    else if (params.act == 3u) { let t = tanh(x); d = 1.0 - t*t; } // Tanh
    else if (params.act == 4u) { let s = 1.0 / (1.0 + exp(-x)); d = s * (1.0 - s); } // Sigmoid
    gradInput[tid] = g * d;
}
`
