package poly

import "fmt"

// WGSL Shaders for FlashPoly Tiling Acceleration
//
// ShaderTiledDense, ShaderTiledSwiGLU, and ShaderTiledMHA are generated
// dynamically so the WGSL workgroup array sizes always match the runtime
// tile size (WGSL doesn't allow runtime-sized workgroup arrays).

// ShaderTiledDenseN generates a tiled dense (matmul) shader for the given tile size.
// The tile size is baked into the WGSL workgroup array and @workgroup_size.
func ShaderTiledDenseN(tileSize int) string {
	return fmt.Sprintf(`
struct Params {
    batchSize: u32,
    inputSize: u32,
    outputSize: u32,
    tileSize: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> tile_input: array<f32, %d>;

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let o = global_id.x;
    let b = global_id.y;

    if (o >= params.outputSize || b >= params.batchSize) { return; }

    var sum: f32 = 0.0;
    let tileSize = params.tileSize;

    for (var iTile: u32 = 0u; iTile < params.inputSize; iTile += tileSize) {
        let i = iTile + local_id.x;
        if (i < params.inputSize) {
            tile_input[local_id.x] = input[b * params.inputSize + i];
        } else {
            tile_input[local_id.x] = 0.0;
        }

        workgroupBarrier();

        let rowOff = o * params.inputSize + iTile;
        for (var k: u32 = 0u; k < tileSize; k++) {
            if (iTile + k < params.inputSize) {
                sum += tile_input[k] * weights[rowOff + k];
            }
        }

        workgroupBarrier();
    }

    output[b * params.outputSize + o] = sum;
}
`, tileSize, tileSize)
}

// ShaderTiledSwiGLUN generates a tiled SwiGLU shader for the given tile size.
func ShaderTiledSwiGLUN(tileSize int) string {
	return fmt.Sprintf(`
struct Params {
    batchSize: u32,
    inputSize: u32,
    outputSize: u32,
    tileSize: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gateWeights: array<f32>;
@group(0) @binding(3) var<storage, read> upWeights: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> tile_input: array<f32, %d>;

fn silu(x: f32) -> f32 {
    return x * (1.0 / (1.0 + exp(-x)));
}

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let o = global_id.x;
    let b = global_id.y;

    if (o >= params.outputSize || b >= params.batchSize) { return; }

    var gateSum: f32 = 0.0;
    var upSum: f32 = 0.0;
    let tileSize = params.tileSize;

    for (var iTile: u32 = 0u; iTile < params.inputSize; iTile += tileSize) {
        let i = iTile + local_id.x;
        if (i < params.inputSize) {
            tile_input[local_id.x] = input[b * params.inputSize + i];
        } else {
            tile_input[local_id.x] = 0.0;
        }

        workgroupBarrier();

        let rowOff = o * params.inputSize + iTile;
        for (var k: u32 = 0u; k < tileSize; k++) {
            if (iTile + k < params.inputSize) {
                gateSum += tile_input[k] * gateWeights[rowOff + k];
                upSum   += tile_input[k] * upWeights[rowOff + k];
            }
        }

        workgroupBarrier();
    }

    output[b * params.outputSize + o] = silu(gateSum) * upSum;
}
`, tileSize, tileSize)
}

// ShaderTiledMHAN generates a tiled MHA shader for the given tile size and headDim.
// Both are baked in as WGSL compile-time constants.
func ShaderTiledMHAN(tileSize, headDim int) string {
	// tile_k and tile_v each hold tileSize rows of headDim floats
	kvArraySize := tileSize * headDim
	return fmt.Sprintf(`
struct Params {
    numHeads: u32,
    numKVHeads: u32,
    headDim: u32,
    seqLen: u32,
    kvOffset: u32,
    maxSeqLen: u32,
    tileSize: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> kCache: array<f32>;
@group(0) @binding(3) var<storage, read> vCache: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> tile_q: array<f32, %d>;   // headDim
var<workgroup> tile_k: array<f32, %d>;   // tileSize * headDim
var<workgroup> tile_v: array<f32, %d>;   // tileSize * headDim

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let h = wg_id.x;
    let s = wg_id.y;
    if (h >= params.numHeads || s >= params.seqLen) { return; }

    let tid = local_id.x;
    let headDim = params.headDim;
    let kvGroupSize = params.numHeads / params.numKVHeads;
    let kvH = h / kvGroupSize;
    let currentTotalPos = params.kvOffset + s;
    let totalKLen = currentTotalPos + 1u;

    let scale = 1.0 / sqrt(f32(headDim));

    if (tid < headDim) {
        tile_q[tid] = q[(s * params.numHeads + h) * headDim + tid];
    }
    workgroupBarrier();

    var max_score: f32 = -1e38;
    var denom: f32 = 0.0;
    var local_v_acc: f32 = 0.0;

    for (var kTile: u32 = 0u; kTile < totalKLen; kTile += params.tileSize) {
        let currentKSize = min(params.tileSize, totalKLen - kTile);

        for (var i: u32 = tid; i < currentKSize * headDim; i += 64u) {
            let row = i / headDim;
            let col = i %% headDim;
            let kvIdx = (kvH * params.maxSeqLen + (kTile + row)) * headDim + col;
            tile_k[i] = kCache[kvIdx];
            tile_v[i] = vCache[kvIdx];
        }
        workgroupBarrier();

        for (var j: u32 = 0u; j < currentKSize; j++) {
            let globalKPos = kTile + j;
            if (globalKPos > currentTotalPos) { continue; }

            var score: f32 = 0.0;
            for (var d: u32 = 0u; d < headDim; d++) {
                score += tile_q[d] * tile_k[j * headDim + d];
            }
            score *= scale;

            let old_max = max_score;
            if (score > max_score) {
                max_score = score;
                let exp_factor = exp(old_max - max_score);
                denom = denom * exp_factor + 1.0;
                local_v_acc = local_v_acc * exp_factor + tile_v[j * headDim + tid];
            } else {
                let exp_val = exp(score - max_score);
                denom += exp_val;
                local_v_acc += tile_v[j * headDim + tid] * exp_val;
            }
        }
        workgroupBarrier();
    }

    if (tid < headDim) {
        output[(s * params.numHeads + h) * headDim + tid] = local_v_acc / denom;
    }
}
`, headDim, kvArraySize, kvArraySize)
}

// Legacy alias constants (kept for non-tiled paths if used elsewhere)

const ShaderRMSNorm = `
struct Params {
    size: u32,
    epsilon: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> shared_sums: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let b = wg_id.x;
    let tid = local_id.x;
    let size = params.size;

    var local_sum: f32 = 0.0;
    for (var i: u32 = tid; i < size; i += 64u) {
        let val = input[b * size + i];
        local_sum += val * val;
    }
    shared_sums[tid] = local_sum;
    workgroupBarrier();

    if (tid == 0u) {
        var total_sum: f32 = 0.0;
        for (var i: u32 = 0u; i < 64u; i++) {
            total_sum += shared_sums[i];
        }
        let rms = 1.0 / sqrt(total_sum / f32(size) + params.epsilon);
        shared_sums[0] = rms;
    }
    workgroupBarrier();

    let rms_val = shared_sums[0];

    for (var i: u32 = tid; i < size; i += 64u) {
        output[b * size + i] = input[b * size + i] * rms_val * weights[i];
    }
}
`

const ShaderResidualAdd = `
struct Params {
    size: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let i = global_id.x;
    if (i >= params.size) { return; }
    input[i] = input[i] + residual[i];
}
`

const ShaderKVUpdate = `
struct KVParams {
    offset: u32,
    headDim: u32,
    maxSeqLen: u32,
    numKVHeads: u32,
    numTokens: u32,
};
@group(0) @binding(0) var<storage, read_write> kCache: array<f32>;
@group(0) @binding(1) var<storage, read_write> vCache: array<f32>;
@group(0) @binding(2) var<storage, read> newK: array<f32>;
@group(0) @binding(3) var<storage, read> newV: array<f32>;
@group(0) @binding(4) var<uniform> params: KVParams;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let kvDim = params.numKVHeads * params.headDim;
    if (tid >= kvDim * params.numTokens) { return; }

    let tokenIdx = tid / kvDim;
    let dimIdx = tid % kvDim;
    let h = dimIdx / params.headDim;
    let d = dimIdx % params.headDim;

    let cacheIdx = (h * params.maxSeqLen + params.offset + tokenIdx) * params.headDim + d;
    kCache[cacheIdx] = newK[tid];
    vCache[cacheIdx] = newV[tid];
}
`

const ShaderRoPE = `
struct RoPEParams {
    seqLen: u32,
    headDim: u32,
    numHeads: u32,
    offset: u32,
    theta: f32,
};

@group(0) @binding(0) var<uniform> params: RoPEParams;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let halfDim = params.headDim / 2u;

    let totalPairs = params.seqLen * params.numHeads * halfDim;
    if (tid >= totalPairs) { return; }

    let d = tid % halfDim;
    let h = (tid / halfDim) % params.numHeads;
    let s = tid / (halfDim * params.numHeads);

    let pos = f32(params.offset + s);
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

const ShaderEmbedding = `
struct Params {
    vocabSize: u32,
    hiddenSize: u32,
    numTokens: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid >= params.numTokens * params.hiddenSize) { return; }
    
    let tokenIdx = tid / params.hiddenSize;
    let dimIdx = tid % params.hiddenSize;
    let vocabIdx = indices[tokenIdx];
    
    if (vocabIdx >= params.vocabSize) {
        output[tid] = 0.0;
    } else {
        output[tid] = weights[vocabIdx * params.hiddenSize + dimIdx];
    }
}
`
