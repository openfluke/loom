package poly

import "fmt"

// WGSL Shaders for Training (Backward Pass)
//
// These shaders implement the gradient calculations (dx, dw) for core layers.

// ShaderDenseBackwardDX calculates gradInput = gradOutput * weights
// dx = dy * W^T  => dx[b, i] = sum_o dy[b, o] * W[o, i]
func ShaderDenseBackwardDX(tileSize int) string {
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

@compute @workgroup_size(%d, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = global_id.x;
    let b = wg_id.y;
    if (i >= params.inputSize || b >= params.batchSize) { return; }
    var sum: f32 = 0.0;
    for (var o: u32 = 0u; o < params.outputSize; o++) {
        sum += gradOutput[b * params.outputSize + o] * weights[o * params.inputSize + i];
    }
    gradInput[b * params.inputSize + i] = sum;
}
`, tileSize)
}

// ShaderDenseBackwardDW calculates gradWeights = gradOutput^T * input
// dw = dy^T * x => dw[o, i] = sum_b dy[b, o] * x[b, i]
func ShaderDenseBackwardDW(tileSize int) string {
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

@compute @workgroup_size(%d, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = global_id.x;
    let o = wg_id.y;
    if (i >= params.inputSize || o >= params.outputSize) { return; }
    var sum: f32 = 0.0;
    for (var b: u32 = 0u; b < params.batchSize; b++) {
        sum += gradOutput[b * params.outputSize + o] * input[b * params.inputSize + i];
    }
    gradWeights[o * params.inputSize + i] += sum;
}
`, tileSize)
}

func ShaderTiledDenseBackwardDX(tileSize int) string {
	return fmt.Sprintf(`
struct DenseScaleParams {
    batchSize: u32,
    inputSize: u32,
    outputSize: u32,
    activation: u32,
    scale: f32,
    p1: u32, p2: u32, p3: u32, p4: u32, p5: u32, p6: u32, p7: u32,
};
@group(0) @binding(0) var<uniform> params: DenseScaleParams;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;

@group(0) @binding(3) var<storage, read_write> gradInput: array<f32>;
@group(0) @binding(4) var<storage, read> preAct: array<f32>;

var<workgroup> gCache: array<f32, %d>;

` + wgslActivateDerivative + `

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         wg_id:      vec3<u32>
) {
    let i = global_id.x;
    let b = wg_id.y;
    let tid = local_id.x;
    let tileSize: u32 = %du;

    if (i >= params.inputSize || b >= params.batchSize) { return; }

    var sum: f32 = 0.0;
    let base_go = b * params.outputSize;

    // DX = GO * W^T  => sum_o GO[o] * W[o,i]
    // Threads in WG share the same input 'i' BUT GO[o] can be shared across different 'i'.
    // If we compute multiple 'i' per WG, we can cache GO[o].

    for (var oTile: u32 = 0u; oTile < params.outputSize; oTile += tileSize) {
        // Load GO tile into shared memory (GO[o] * activateDerivative(preAct[o]))
        let oIdx = oTile + tid;
        if (oIdx < params.outputSize) {
            let idx = base_go + oIdx;
            gCache[tid] = gradOutput[idx] * activateDerivative(preAct[idx], params.activation);
        } else {
            gCache[tid] = 0.0;
        }
        
        workgroupBarrier();
        
        let limit = min(tileSize, params.outputSize - oTile);
        for (var o: u32 = 0u; o < limit; o++) {
            // Weights is [outputSize, inputSize] => W[oTile + o, i]
            sum += gCache[o] * weights[(oTile + o) * params.inputSize + i];
        }
        
        workgroupBarrier();
    }
    
    gradInput[b * params.inputSize + i] = sum;
}
`, tileSize, tileSize, tileSize)
}

func ShaderTiledDenseBackwardDW(tileSize int) string {
	return fmt.Sprintf(`
struct DenseScaleParams {
    batchSize: u32,
    inputSize: u32,
    outputSize: u32,
    activation: u32,
    scale: f32,
    p1: u32, p2: u32, p3: u32, p4: u32, p5: u32, p6: u32, p7: u32,
};
@group(0) @binding(0) var<uniform> params: DenseScaleParams;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;

@group(0) @binding(3) var<storage, read_write> gradWeights: array<f32>;
@group(0) @binding(4) var<storage, read> preAct: array<f32>;

var<workgroup> inCache: array<f32, %d>;

` + wgslActivateDerivative + `

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         wg_id:      vec3<u32>
) {
    let i = global_id.x; // input
    let o = global_id.y; // output
    let tid = local_id.x;
    let tileSize: u32 = %du;

    if (i >= params.inputSize || o >= params.outputSize) { return; }

    var sum: f32 = 0.0;

    // DW = GO^T * IN => sum_b GO[b, o] * IN[b, i]
    // Threads in WG share output 'o', different 'i'.
    // We can't easily cache GO[b,o] across different 'i' because it's only one 'o'.
    // But we can cache inputs across batches.

    for (var bTile: u32 = 0u; bTile < params.batchSize; bTile += tileSize) {
        // Load input tile for this 'i' across batches? No, tid is 'i'.
        // Load input[bTile + tid, i]? No.
        // Let's cache GO[b, o] across batches.
        let bIdx = bTile + tid;
        if (bIdx < params.batchSize) {
            let outIdx = bIdx * params.outputSize + o;
            inCache[tid] = gradOutput[outIdx] * activateDerivative(preAct[outIdx], params.activation);
        } else {
            inCache[tid] = 0.0;
        }
        
        workgroupBarrier();
        
        let limit = min(tileSize, params.batchSize - bTile);
        for (var b: u32 = 0u; b < limit; b++) {
            sum += inCache[b] * input[(bTile + b) * params.inputSize + i];
        }
        
        workgroupBarrier();
    }
    
    gradWeights[o * params.inputSize + i] += sum;
}
`, tileSize, tileSize, tileSize)
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

@group(0) @binding(3) var<storage, read_write> gradInput: array<f32>;
@group(0) @binding(4) var<storage, read> preAct: array<f32>;

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

@group(0) @binding(3) var<storage, read_write> gradWeights: array<f32>;
@group(0) @binding(4) var<storage, read> preAct: array<f32>;

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

@group(0) @binding(3) var<storage, read_write> gradInput: array<f32>;
@group(0) @binding(4) var<storage, read> preAct: array<f32>;

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

@group(0) @binding(3) var<storage, read_write> gradWeights: array<f32>;
@group(0) @binding(4) var<storage, read> preAct: array<f32>;

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

@group(0) @binding(3) var<storage, read_write> gradInput: array<f32>;
@group(0) @binding(4) var<storage, read> preAct: array<f32>;

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
                            let outIdx = ((b * params.filters + f) * params.outD + od) * params.outH * params.outW + oh * params.outW + ow;
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

@group(0) @binding(3) var<storage, read_write> gradWeights: array<f32>;
@group(0) @binding(4) var<storage, read> preAct: array<f32>;

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
                        let outIdx = ((b * params.filters + f) * params.outD + od) * params.outH * params.outW + oh * params.outW + ow;
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
    var scores: array<f32, 2048>; 
    var max_score: f32 = -1e9;
    
    let seqLen = params.seqLen;
    if (seqLen > 2048u) { return; } // Safety limit

    for (var s_k: u32 = 0u; s_k < seqLen; s_k++) {
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
    for (var s_k: u32 = 0u; s_k < seqLen; s_k++) {
        scores[s_k] = exp(scores[s_k] - max_score);
        exp_sum += scores[s_k];
    }
    for (var s_k: u32 = 0u; s_k < seqLen; s_k++) {
        scores[s_k] /= exp_sum;
    }
    
    // 2. Compute dScores / dV contributions
    var d_softmax: array<f32, 2048>;
    for (var s_k: u32 = 0u; s_k < seqLen; s_k++) {
        var ds: f32 = 0.0;
        for (var d: u32 = 0u; d < headDim; d++) {
            ds += gradOutput[((b * params.numHeads + h) * params.seqLen + s_q) * headDim + d] *
                  V[((b * params.numKVHeads + kvHead) * params.seqLen + s_k) * headDim + d];
        }
        d_softmax[s_k] = ds;
    }
    
    var sum_ds_s: f32 = 0.0;
    for (var s_k: u32 = 0u; s_k < seqLen; s_k++) {
        sum_ds_s += d_softmax[s_k] * scores[s_k];
    }
    
    for (var s_k: u32 = 0u; s_k < seqLen; s_k++) {
        let d_logit = (d_softmax[s_k] - sum_ds_s) * scores[s_k];
        
        for (var d: u32 = 0u; d < headDim; d++) {
            let q_off = ((b * params.numHeads + h) * params.seqLen + s_q) * headDim + d;
            let k_off = ((b * params.numKVHeads + kvHead) * params.seqLen + s_k) * headDim + d;
            let v_off = ((b * params.numKVHeads + kvHead) * params.seqLen + s_k) * headDim + d;
            
            // Note: Parallel MHA backward needs atomics for dK/dV or separate passes
            dQ[q_off] += d_logit * params.scale * K[k_off];
            // Atomic operations are not supported for f32 in standard WGSL yet.
            // Using direct += will cause race conditions between heads/queries.
            // For bit-exact simulation on CPU, this shader is only a placeholder or 
            // used in specific non-training scenarios.
            dK[k_off] += d_logit * params.scale * Q[q_off];
            dV[v_off] += scores[s_k] * gradOutput[q_off];
        }
    }
}
`

// ShaderMSEGradPartialLoss computes MSE gradients and partial loss sums entirely on GPU.
// Each workgroup of 256 threads reduces its elements, writing one partial sum to partials[wg_id.x].
// CPU sums the partials array (ceil(N/256) floats) for the total loss — no full-output readback needed.
const ShaderMSEGradPartialLoss = `
struct Params {
    size: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       output:   array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:      array<f32>;
@group(0) @binding(3) var<storage, read_write> gradient: array<f32>;
@group(0) @binding(4) var<storage, read_write> partials: array<f32>;

var<workgroup> shared_sq: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id:   vec3<u32>,
    @builtin(workgroup_id)        wg_id:      vec3<u32>,
) {
    let tid = global_id.x;
    let lid = local_id.x;
    let N   = params.size;

    var local_sq: f32 = 0.0;
    if (tid < N) {
        let diff      = output[tid] - tgt[tid];
        gradient[tid] = (2.0 / f32(N)) * diff;
        local_sq      = diff * diff;
    }
    shared_sq[lid] = local_sq;
    workgroupBarrier();

    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { shared_sq[lid] += shared_sq[lid + s]; }
        workgroupBarrier();
    }

    if (lid == 0u) {
        partials[wg_id.x] = shared_sq[0] / f32(N);
    }
}
`

// ShaderMultiHeadSoftmaxCEGradPartialLoss: three disjoint softmax+CE heads per row (Mark union: 4+17+20).
// Each thread handles one batch row; writes dL/dlogit = (p-y)/B for logits in that row (B=batch).
// partials[wg] = sum of (CE0+CE1+CE2) over batch rows handled by this workgroup (CPU divides by B for mean loss).
const ShaderMultiHeadSoftmaxCEGradPartialLoss = `
struct Params {
    batch: u32,
    row_width: u32,
    h0: u32,
    h1: u32,
    h2: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       output:   array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:      array<f32>;
@group(0) @binding(3) var<storage, read_write> gradient: array<f32>;
@group(0) @binding(4) var<storage, read_write> partials: array<f32>;

var<workgroup> shared_ce: array<f32, 256>;

fn head_ce_grad(base: u32, C: u32, B: u32) -> f32 {
    var ce: f32 = 0.0;
    if (C == 0u) { return ce; }
    var maxv: f32 = output[base];
    for (var i: u32 = 1u; i < C; i++) {
        maxv = max(maxv, output[base + i]);
    }
    var sume: f32 = 0.0;
    for (var i: u32 = 0u; i < C; i++) {
        sume += exp(output[base + i] - maxv);
    }
    let invB = 1.0 / f32(B);
    for (var i: u32 = 0u; i < C; i++) {
        let p = exp(output[base + i] - maxv) / sume;
        let y = tgt[base + i];
        ce -= y * log(p + 1e-7);
        gradient[base + i] = (p - y) * invB;
    }
    return ce;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let b = global_id.x;
    let lid = local_id.x;
    let B = params.batch;
    let R = params.row_width;

    var local_ce: f32 = 0.0;
    if (b < B) {
        let row0 = b * R;
        let h0 = params.h0;
        let h1 = params.h1;
        let h2 = params.h2;
        local_ce += head_ce_grad(row0, h0, B);
        local_ce += head_ce_grad(row0 + h0, h1, B);
        local_ce += head_ce_grad(row0 + h0 + h1, h2, B);
    }

    shared_ce[lid] = local_ce;
    workgroupBarrier();

    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            shared_ce[lid] += shared_ce[lid + s];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        partials[wg_id.x] = shared_ce[0];
    }
}
`

// ShaderMultiHeadSoftmaxCEGradPartialLossMasked: same as ShaderMultiHeadSoftmaxCEGradPartialLoss, but each batch row
// has three mask scalars in head_mask[b*3+{0,1,2}] (active if value > 0.5). Inactive heads contribute 0 CE and get
// zero dL/dlogit on that head's logit slice (matches CPU multiHeadMaskActive).
const ShaderMultiHeadSoftmaxCEGradPartialLossMasked = `
struct Params {
    batch: u32,
    row_width: u32,
    h0: u32,
    h1: u32,
    h2: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       output:   array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:      array<f32>;
@group(0) @binding(3) var<storage, read_write> gradient: array<f32>;
@group(0) @binding(4) var<storage, read_write> partials: array<f32>;
@group(0) @binding(5) var<storage, read>         head_mask: array<f32>;

var<workgroup> shared_ce: array<f32, 256>;

fn head_ce_grad_masked(base: u32, C: u32, B: u32, m: f32) -> f32 {
    if (m < 0.5) {
        if (C == 0u) { return 0.0; }
        for (var i: u32 = 0u; i < C; i++) {
            gradient[base + i] = 0.0;
        }
        return 0.0;
    }
    var ce: f32 = 0.0;
    if (C == 0u) { return ce; }
    var maxv: f32 = output[base];
    for (var i: u32 = 1u; i < C; i++) {
        maxv = max(maxv, output[base + i]);
    }
    var sume: f32 = 0.0;
    for (var i: u32 = 0u; i < C; i++) {
        sume += exp(output[base + i] - maxv);
    }
    let invB = 1.0 / f32(B);
    for (var i: u32 = 0u; i < C; i++) {
        let p = exp(output[base + i] - maxv) / sume;
        let y = tgt[base + i];
        ce -= y * log(p + 1e-7);
        gradient[base + i] = (p - y) * invB;
    }
    return ce;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let b = global_id.x;
    let lid = local_id.x;
    let B = params.batch;
    let R = params.row_width;

    var local_ce: f32 = 0.0;
    if (b < B) {
        let row0 = b * R;
        let h0 = params.h0;
        let h1 = params.h1;
        let h2 = params.h2;
        let m0 = head_mask[b * 3u + 0u];
        let m1 = head_mask[b * 3u + 1u];
        let m2 = head_mask[b * 3u + 2u];
        local_ce += head_ce_grad_masked(row0, h0, B, m0);
        local_ce += head_ce_grad_masked(row0 + h0, h1, B, m1);
        local_ce += head_ce_grad_masked(row0 + h0 + h1, h2, B, m2);
    }

    shared_ce[lid] = local_ce;
    workgroupBarrier();

    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            shared_ce[lid] += shared_ce[lid + s];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        partials[wg_id.x] = shared_ce[0];
    }
}
`

const ShaderApplyGradients = `
struct Params {
    size: u32,
    lr: f32,
    clipVal: f32,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<storage, read> gradients: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid >= params.size) { return; }
    
    var g = gradients[tid];
    if (params.clipVal > 0.0) {
        g = clamp(g, -params.clipVal, params.clipVal);
    }
    weights[tid] -= params.lr * g;
}
`

const ShaderQuantizeI8 = `
struct Params {
    size: u32,
    scale: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>  master: array<f32>;
@group(0) @binding(2) var<storage, read_write> native: array<u32>; // packed i8

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid * 4u >= params.size) { return; }
    
    var packed: u32 = 0u;
    for (var i: u32 = 0u; i < 4u; i++) {
        let idx = tid * 4u + i;
        if (idx < params.size) {
            let val = master[idx] / params.scale;
            let q = u32(clamp(f32(i32(round(val))), -128.0, 127.0) + 128.0);
            packed |= (q & 0xFFu) << (i * 8u);
        }
    }
    native[tid] = packed;
}
`

const ShaderQuantizeI4 = `
struct Params {
    size: u32,
    scale: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>  master: array<f32>;
@group(0) @binding(2) var<storage, read_write> native: array<u32>; // packed i4

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid * 8u >= params.size) { return; }
    
    var packed: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i++) {
        let idx = tid * 8u + i;
        if (idx < params.size) {
            let val = master[idx] / params.scale;
            let q = u32(clamp(f32(i32(round(val))), -8.0, 7.0) + 8.0);
            packed |= (q & 0xFu) << (i * 4u);
        }
    }
    native[tid] = packed;
}
`

const ShaderQuantizeFP4 = `
struct Params {
    size: u32,
    scale: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>  master: array<f32>;
@group(0) @binding(2) var<storage, read_write> native: array<u32>; // packed fp4

fn encodeFP4(v: f32) -> u32 {
    var bestCode: u32 = 0u;
    var bestDiff: f32 = abs(v);
    let c0 = 0.0; if (abs(v - c0) < bestDiff) { bestDiff = abs(v - c0); bestCode = 0u; }
    let c1 = 0.75; if (abs(v - c1) < bestDiff) { bestDiff = abs(v - c1); bestCode = 1u; }
    let c2 = 1.0; if (abs(v - c2) < bestDiff) { bestDiff = abs(v - c2); bestCode = 2u; }
    let c3 = 1.5; if (abs(v - c3) < bestDiff) { bestDiff = abs(v - c3); bestCode = 3u; }
    let c4 = 2.0; if (abs(v - c4) < bestDiff) { bestDiff = abs(v - c4); bestCode = 4u; }
    let c5 = 3.0; if (abs(v - c5) < bestDiff) { bestDiff = abs(v - c5); bestCode = 5u; }
    let c8 = 0.0; if (abs(v - c8) < bestDiff) { bestDiff = abs(v - c8); bestCode = 8u; }
    let c9 = -0.75; if (abs(v - c9) < bestDiff) { bestDiff = abs(v - c9); bestCode = 9u; }
    let c10 = -1.0; if (abs(v - c10) < bestDiff) { bestDiff = abs(v - c10); bestCode = 10u; }
    let c11 = -1.5; if (abs(v - c11) < bestDiff) { bestDiff = abs(v - c11); bestCode = 11u; }
    let c12 = -2.0; if (abs(v - c12) < bestDiff) { bestDiff = abs(v - c12); bestCode = 12u; }
    let c13 = -3.0; if (abs(v - c13) < bestDiff) { bestDiff = abs(v - c13); bestCode = 13u; }
    return bestCode;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid * 8u >= params.size) { return; }
    var packed: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i++) {
        let idx = tid * 8u + i;
        if (idx < params.size) {
            let code = encodeFP4(master[idx]);
            packed |= (code & 0xFu) << (i * 4u);
        }
    }
    native[tid] = packed;
}
`

const ShaderQuantizeTernary = `
struct Params {
    size: u32,
    scale: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>  master: array<f32>;
@group(0) @binding(2) var<storage, read_write> native: array<u32>; // packed ternary (2 bits)

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid * 16u >= params.size) { return; }
    var packed: u32 = 0u;
    for (var i: u32 = 0u; i < 16u; i++) {
        let idx = tid * 16u + i;
        if (idx < params.size) {
            let val = master[idx] / params.scale;
            var q = i32(round(val));
            if (q < -1) { q = -1; }
            if (q > 1) { q = 1; }
            var code: u32 = 1u;
            if (q < 0) { code = 0u; }
            else if (q > 0) { code = 2u; }
            packed |= (code & 0x3u) << (i * 2u);
        }
    }
    native[tid] = packed;
}
`

const ShaderQuantizeBinary = `
struct Params {
    size: u32,
    scale: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>  master: array<f32>;
@group(0) @binding(2) var<storage, read_write> native: array<u32>; // packed binary (1 bit)

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid * 32u >= params.size) { return; }
    var packed: u32 = 0u;
    for (var i: u32 = 0u; i < 32u; i++) {
        let idx = tid * 32u + i;
        if (idx < params.size) {
            if (master[idx] >= 0.0) {
                packed |= (1u << i);
            }
        }
    }
    native[tid] = packed;
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

// ShaderTiledRNNBackwardDX computes gradInput = gPre @ wIH^T for a single RNN step.
// hCurr (binding 3) holds the post-tanh output; tanh' = 1 - hCurr^2.
// Thread grid: ((inputSize+tileSize-1)/tileSize, batchSize, 1).
func ShaderTiledRNNBackwardDX(tileSize int) string {
	return fmt.Sprintf(`
struct RNNParams {
    batchSize:  u32,
    inputSize:  u32,
    hiddenSize: u32,
    padding:    u32,
};
@group(0) @binding(0) var<uniform>             params:     RNNParams;
@group(0) @binding(1) var<storage, read>       gradOutput: array<f32>; // [batchSize, hiddenSize]
@group(0) @binding(2) var<storage, read>       wIH:        array<f32>; // [hiddenSize, inputSize]
@group(0) @binding(3) var<storage, read>       hCurr:      array<f32>; // [batchSize, hiddenSize] post-tanh
@group(0) @binding(4) var<storage, read_write> gradInput:  array<f32>; // [batchSize, inputSize]

var<workgroup> shGPre: array<f32, %d>;

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         wg_id:     vec3<u32>,
) {
    let i   = global_id.x;
    let b   = wg_id.y;
    let tid = local_id.x;
    let H   = params.hiddenSize;
    let I   = params.inputSize;
    let TS: u32 = %du;

    var grad: f32 = 0.0;

    for (var hTile: u32 = 0u; hTile < H; hTile += TS) {
        let h = hTile + tid;
        if (h < H) {
            let hc = hCurr[b * H + h];
            shGPre[tid] = gradOutput[b * H + h] * (1.0 - hc * hc);
        } else {
            shGPre[tid] = 0.0;
        }
        workgroupBarrier();

        if (i < I) {
            let limit = min(TS, H - hTile);
            for (var k: u32 = 0u; k < limit; k++) {
                grad += wIH[(hTile + k) * I + i] * shGPre[k];
            }
        }
        workgroupBarrier();
    }

    if (i < I) {
        gradInput[b * I + i] = grad;
    }
}
`, tileSize, tileSize, tileSize)
}

// ShaderTiledRNNBackwardDW computes gradWeights for a single RNN step.
// Layout: gradWeights = [gradWIH (H×I), gradWHH (H×H), gradBias (H)].
// Thread grid: ((hiddenSize+tileSize-1)/tileSize, 1, 1).
func ShaderTiledRNNBackwardDW(tileSize int) string {
	return fmt.Sprintf(`
struct RNNParams {
    batchSize:  u32,
    inputSize:  u32,
    hiddenSize: u32,
    padding:    u32,
};
@group(0) @binding(0) var<uniform>             params:      RNNParams;
@group(0) @binding(1) var<storage, read>       gradOutput:  array<f32>; // [batchSize, hiddenSize]
@group(0) @binding(2) var<storage, read>       input:       array<f32>; // [batchSize, inputSize]
@group(0) @binding(3) var<storage, read>       hCurr:       array<f32>; // [batchSize, hiddenSize] post-tanh
@group(0) @binding(4) var<storage, read>       hPrev:       array<f32>; // [batchSize, hiddenSize]
@group(0) @binding(5) var<storage, read_write> gradWeights: array<f32>; // [ihSize + hhSize + hiddenSize]

@compute @workgroup_size(%d, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let h = global_id.x;
    if (h >= params.hiddenSize) { return; }

    let H      = params.hiddenSize;
    let I      = params.inputSize;
    let ihSize = H * I;
    let hhSize = H * H;

    var biasGrad: f32 = 0.0;

    for (var b: u32 = 0u; b < params.batchSize; b++) {
        let hc   = hCurr[b * H + h];
        let gPre = gradOutput[b * H + h] * (1.0 - hc * hc);
        biasGrad += gPre;

        for (var i: u32 = 0u; i < I; i++) {
            gradWeights[h * I + i] += gPre * input[b * I + i];
        }
        for (var hp: u32 = 0u; hp < H; hp++) {
            gradWeights[ihSize + h * H + hp] += gPre * hPrev[b * H + hp];
        }
    }
    gradWeights[ihSize + hhSize + h] += biasGrad;
}
`, tileSize)
}

// ShaderTiledLSTMBackwardDX computes gradInput for a single LSTM step.
// preAct holds [iS, fS, gS, oS, cC] per (b, h) — 5*hiddenSize floats per batch item.
// cPrev is assumed 0 (single step), so dfP = 0 and only diP/dgP/doP are non-zero.
// Thread grid: ((inputSize+tileSize-1)/tileSize, batchSize, 1).
func ShaderTiledLSTMBackwardDX(tileSize int) string {
	return fmt.Sprintf(`
struct LSTMParams {
    batchSize:  u32,
    inputSize:  u32,
    hiddenSize: u32,
    padding:    u32,
};
@group(0) @binding(0) var<uniform>             params:    LSTMParams;
@group(0) @binding(1) var<storage, read>       gradOutput: array<f32>; // [batchSize, hiddenSize]
@group(0) @binding(2) var<storage, read>       weights:    array<f32>; // [wI,wF,wG,wO] (4*gateSize)
@group(0) @binding(3) var<storage, read>       preAct:     array<f32>; // [batchSize, 5*hiddenSize]
@group(0) @binding(4) var<storage, read_write> gradInput:  array<f32>; // [batchSize, inputSize]

var<workgroup> shDI: array<f32, %d>;
var<workgroup> shDG: array<f32, %d>;
var<workgroup> shDO: array<f32, %d>;

fn lstm_sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         wg_id:     vec3<u32>,
) {
    let i   = global_id.x;
    let b   = wg_id.y;
    let tid = local_id.x;
    let H   = params.hiddenSize;
    let I   = params.inputSize;
    let TS: u32 = %du;

    let ihSize   = H * I;
    let hhSize   = H * H;
    let gateSize = ihSize + hhSize + H;

    var grad: f32 = 0.0;

    for (var hTile: u32 = 0u; hTile < H; hTile += TS) {
        let h = hTile + tid;
        if (h < H) {
            let pIdx = b * 5u * H;
            let iS = preAct[pIdx + h];
            let gS = preAct[pIdx + 2u * H + h];
            let oS = preAct[pIdx + 3u * H + h];
            let cC = preAct[pIdx + 4u * H + h];

            let iG = lstm_sigmoid(iS);
            let gG = tanh(gS);
            let oG = lstm_sigmoid(oS);
            let cT = tanh(cC);

            let dh = gradOutput[b * H + h];
            let dc = dh * oG * (1.0 - cT * cT);

            shDI[tid] = dc * gG * iG * (1.0 - iG);
            shDG[tid] = dc * iG * (1.0 - gG * gG);
            shDO[tid] = dh * cT * oG * (1.0 - oG);
        } else {
            shDI[tid] = 0.0;
            shDG[tid] = 0.0;
            shDO[tid] = 0.0;
        }
        workgroupBarrier();

        if (i < I) {
            let limit = min(TS, H - hTile);
            for (var k: u32 = 0u; k < limit; k++) {
                let hh = hTile + k;
                grad += weights[hh * I + i]                 * shDI[k]
                      + weights[2u * gateSize + hh * I + i] * shDG[k]
                      + weights[3u * gateSize + hh * I + i] * shDO[k];
            }
        }
        workgroupBarrier();
    }

    if (i < I) {
        gradInput[b * I + i] = grad;
    }
}
`, tileSize, tileSize, tileSize, tileSize, tileSize)
}

// ShaderTiledLSTMBackwardDW computes weight gradients for a single LSTM step.
// Weight layout: [wI gateSize, wF gateSize, wG gateSize, wO gateSize],
// gateSize = hiddenSize*inputSize + hiddenSize*hiddenSize + hiddenSize.
// Thread grid: ((hiddenSize+tileSize-1)/tileSize, 1, 1).
func ShaderTiledLSTMBackwardDW(tileSize int) string {
	return fmt.Sprintf(`
struct LSTMParams {
    batchSize:  u32,
    inputSize:  u32,
    hiddenSize: u32,
    padding:    u32,
};
@group(0) @binding(0) var<uniform>             params:      LSTMParams;
@group(0) @binding(1) var<storage, read>       gradOutput:  array<f32>; // [batchSize, hiddenSize]
@group(0) @binding(2) var<storage, read>       input:       array<f32>; // [batchSize, inputSize]
@group(0) @binding(3) var<storage, read>       preAct:      array<f32>; // [batchSize, 5*hiddenSize]
@group(0) @binding(4) var<storage, read>       hPrev:       array<f32>; // [batchSize, hiddenSize]
@group(0) @binding(5) var<storage, read_write> gradWeights: array<f32>;

fn lstm_sigmoid_dw(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

@compute @workgroup_size(%d, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let h = global_id.x;
    if (h >= params.hiddenSize) { return; }

    let H        = params.hiddenSize;
    let I        = params.inputSize;
    let ihSize   = H * I;
    let hhSize   = H * H;
    let gateSize = ihSize + hhSize + H;

    var gBI: f32 = 0.0;
    var gBF: f32 = 0.0;
    var gBG: f32 = 0.0;
    var gBO: f32 = 0.0;

    for (var b: u32 = 0u; b < params.batchSize; b++) {
        let pIdx = b * 5u * H;
        let iS = preAct[pIdx + h];
        let fS = preAct[pIdx + H + h];
        let gS = preAct[pIdx + 2u * H + h];
        let oS = preAct[pIdx + 3u * H + h];
        let cC = preAct[pIdx + 4u * H + h];

        let iG = lstm_sigmoid_dw(iS);
        let fG = lstm_sigmoid_dw(fS);
        let gG = tanh(gS);
        let oG = lstm_sigmoid_dw(oS);
        let cT = tanh(cC);

        let dh = gradOutput[b * H + h];
        let dc = dh * oG * (1.0 - cT * cT);
        // cPrev = 0 for single-step, so dfP = 0
        let diP = dc * gG * iG * (1.0 - iG);
        let dfP = 0.0;
        let dgP = dc * iG * (1.0 - gG * gG);
        let doP = dh * cT * oG * (1.0 - oG);

        gBI += diP; gBF += dfP; gBG += dgP; gBO += doP;

        for (var i: u32 = 0u; i < I; i++) {
            let x = input[b * I + i];
            gradWeights[h * I + i]                         += diP * x;
            gradWeights[gateSize + h * I + i]              += dfP * x;
            gradWeights[2u * gateSize + h * I + i]         += dgP * x;
            gradWeights[3u * gateSize + h * I + i]         += doP * x;
        }
        for (var hp: u32 = 0u; hp < H; hp++) {
            let hv = hPrev[b * H + hp];
            gradWeights[ihSize + h * H + hp]                         += diP * hv;
            gradWeights[gateSize + ihSize + h * H + hp]              += dfP * hv;
            gradWeights[2u * gateSize + ihSize + h * H + hp]         += dgP * hv;
            gradWeights[3u * gateSize + ihSize + h * H + hp]         += doP * hv;
        }
    }
    gradWeights[ihSize + hhSize + h]                         += gBI;
    gradWeights[gateSize + ihSize + hhSize + h]              += gBF;
    gradWeights[2u * gateSize + ihSize + hhSize + h]         += gBG;
    gradWeights[3u * gateSize + ihSize + hhSize + h]         += gBO;
}
`, tileSize)
}

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

const ShaderFillZero = `
struct Params {
    size: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.size) { return; }
    data[global_id.x] = 0.0;
}
`

const ShaderCEGradPartialLoss = ` 
struct Params {
    size: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       output:   array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:      array<f32>;
@group(0) @binding(3) var<storage, read_write> gradient: array<f32>;
@group(0) @binding(4) var<storage, read_write> partials: array<f32>;

var<workgroup> shared_ce: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id)        wg_id:    vec3<u32>
) {
    let tid = local_id.x + wg_id.x * 256u;
    let lid = local_id.x;
    let N   = params.size;
    let eps = 1e-10;

    var local_ce: f32 = 0.0;
    if (tid < N) {
        let p = output[tid];
        let y = tgt[tid];
        gradient[tid] = -(y / (p + eps)) / f32(N);
        local_ce      = -y * log(p + eps);
    }
    shared_ce[lid] = local_ce;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            shared_ce[lid] += shared_ce[lid + s];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        partials[wg_id.x] = shared_ce[0] / f32(N);
    }
}
` 
