package poly

import "fmt"

// WGSL Shaders for FlashPoly Tiling Acceleration
//
// ShaderTiledDense, ShaderTiledSwiGLU, and ShaderTiledMHA are generated
// dynamically so the WGSL workgroup array sizes always match the runtime
// tile size (WGSL doesn't allow runtime-sized workgroup arrays).

// ShaderTiledDenseN generates a tiled dense (matmul) shader for the given tile size.
// The tile size is baked into the WGSL workgroup array and @workgroup_size.
// ShaderTiledDenseQ4 generates a tiled dense shader that dequantizes 4-bit weights on the fly.
// Block size is 32: 1 f32 scale + 16 bytes (32 nibbles).
func ShaderTiledDenseQ4(tileSize int) string {
	// A pure global-memory based, register-unrolled kernel for Q4_0.
	// We avoid shared memory entirely to eliminate barrier overhead.
	// We unroll by 8 since each u32 word contains 8 Q4 weights.
	return fmt.Sprintf(`
struct Params {
    batchSize: u32,
    inputSize: u32,
    outputSize: u32,
    tileSize: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let o = global_id.x;
    let b = wg_id.y;

    if (o >= params.outputSize || b >= params.batchSize) { return; }

    var sum: f32 = 0.0;
    
    // We process 8 elements at a time (one u32 word)
    let limit = params.inputSize / 8u;
    let rem = params.inputSize %% 8u;
    
    let base_in = b * params.inputSize;
    let base_w = o * params.inputSize;

    var i: u32 = 0u;
    for (var k: u32 = 0u; k < limit; k++) {
        let globalIdx = base_w + i;
        let blockIdx = globalIdx / 32u;
        let scale = scales[blockIdx];
        
        let wordIdx = globalIdx / 8u;
        let packed = weights[wordIdx];
        
        let in0 = input[base_in + i];
        let in1 = input[base_in + i + 1u];
        let in2 = input[base_in + i + 2u];
        let in3 = input[base_in + i + 3u];
        let in4 = input[base_in + i + 4u];
        let in5 = input[base_in + i + 5u];
        let in6 = input[base_in + i + 6u];
        let in7 = input[base_in + i + 7u];

        // Nibble 0
        var q0 = i32(packed & 0xFu);
        if (q0 > 7) { q0 -= 16; }
        
        // Nibble 1
        var q1 = i32((packed >> 4u) & 0xFu);
        if (q1 > 7) { q1 -= 16; }
        
        // Nibble 2
        var q2 = i32((packed >> 8u) & 0xFu);
        if (q2 > 7) { q2 -= 16; }
        
        // Nibble 3
        var q3 = i32((packed >> 12u) & 0xFu);
        if (q3 > 7) { q3 -= 16; }

        // Nibble 4
        var q4 = i32((packed >> 16u) & 0xFu);
        if (q4 > 7) { q4 -= 16; }
        
        // Nibble 5
        var q5 = i32((packed >> 20u) & 0xFu);
        if (q5 > 7) { q5 -= 16; }
        
        // Nibble 6
        var q6 = i32((packed >> 24u) & 0xFu);
        if (q6 > 7) { q6 -= 16; }
        
        // Nibble 7
        var q7 = i32((packed >> 28u) & 0xFu);
        if (q7 > 7) { q7 -= 16; }

        sum += (in0 * f32(q0) + in1 * f32(q1) + in2 * f32(q2) + in3 * f32(q3) +
               in4 * f32(q4) + in5 * f32(q5) + in6 * f32(q6) + in7 * f32(q7)) * scale;
        
        i += 8u;
    }

    // Remainder should ideally be 0 if sizes are multiples of 32
    for (var k: u32 = 0u; k < rem; k++) {
        let globalIdx = base_w + i + k;
        let blockIdx = globalIdx / 32u;
        let scale = scales[blockIdx];
        
        let wordIdx = globalIdx / 8u;
        let nibbleIdx = globalIdx %% 8u;
        let packed = weights[wordIdx];
        
        var q = i32((packed >> (nibbleIdx * 4u)) & 0xFu);
        if (q > 7) { q -= 16; }
        
        sum += input[base_in + i + k] * (f32(q) * scale);
    }

    output[b * params.outputSize + o] = sum;
}
`, tileSize)
}

func ShaderTiledDenseN(tileSize int) string {
	// A pure global-memory based, register-unrolled kernel.
	// We avoid shared memory entirely to eliminate barrier overhead,
	// because some WebGPU backends emulate workgroup memory poorly.
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

// tileSize is typically 32 or 64. Workgroup processes 'tileSize' outputs.
@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let o = global_id.x;
    let b = wg_id.y;

    if (o >= params.outputSize || b >= params.batchSize) { return; }

    var sum: f32 = 0.0;
    
    // We unroll the loop by 4 (a typical SIMD width)
    let limit = params.inputSize / 4u;
    let rem = params.inputSize %% 4u;
    
    let base_in = b * params.inputSize;
    let base_w = o * params.inputSize;

    var i: u32 = 0u;
    for (var k: u32 = 0u; k < limit; k++) {
        let in0 = input[base_in + i];
        let in1 = input[base_in + i + 1u];
        let in2 = input[base_in + i + 2u];
        let in3 = input[base_in + i + 3u];

        let w0 = weights[base_w + i];
        let w1 = weights[base_w + i + 1u];
        let w2 = weights[base_w + i + 2u];
        let w3 = weights[base_w + i + 3u];

        sum += in0 * w0 + in1 * w1 + in2 * w2 + in3 * w3;
        i += 4u;
    }

    for (var k: u32 = 0u; k < rem; k++) {
        sum += input[base_in + i + k] * weights[base_w + i + k];
    }

    output[b * params.outputSize + o] = sum;
}
`, tileSize)
}

// ShaderTiledSwiGLUQ4 generates a tiled SwiGLU shader with Q4_0 weights.
func ShaderTiledSwiGLUQ4(tileSize int) string {
	// Unrolled Q4 kernel for SwiGLU: processes 8 weights per u32 directly from global.
	return fmt.Sprintf(`
struct Params {
    batchSize: u32,
    inputSize: u32,
    outputSize: u32,
    tileSize: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gateScales: array<f32>;
@group(0) @binding(3) var<storage, read> gateWeights: array<u32>;
@group(0) @binding(4) var<storage, read> upScales: array<f32>;
@group(0) @binding(5) var<storage, read> upWeights: array<u32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;

fn silu(x: f32) -> f32 {
    return x * (1.0 / (1.0 + exp(-x)));
}

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let o = global_id.x;
    let b = wg_id.y;

    if (o >= params.outputSize || b >= params.batchSize) { return; }

    var gateSum: f32 = 0.0;
    var upSum: f32 = 0.0;
    
    // We process 8 elements at a time (one u32 word)
    let limit = params.inputSize / 8u;
    let rem = params.inputSize %% 8u;
    
    let base_in = b * params.inputSize;
    let base_w = o * params.inputSize;

    var i: u32 = 0u;
    for (var k: u32 = 0u; k < limit; k++) {
        let globalIdx = base_w + i;
        let blockIdx = globalIdx / 32u;
        
        let gScale = gateScales[blockIdx];
        let uScale = upScales[blockIdx];
        
        let wordIdx = globalIdx / 8u;
        let gPacked = gateWeights[wordIdx];
        let uPacked = upWeights[wordIdx];
        
        let in0 = input[base_in + i];
        let in1 = input[base_in + i + 1u];
        let in2 = input[base_in + i + 2u];
        let in3 = input[base_in + i + 3u];
        let in4 = input[base_in + i + 4u];
        let in5 = input[base_in + i + 5u];
        let in6 = input[base_in + i + 6u];
        let in7 = input[base_in + i + 7u];

        // Process Gate
        var gq0 = i32(gPacked & 0xFu); if (gq0 > 7) { gq0 -= 16; }
        var gq1 = i32((gPacked >> 4u) & 0xFu); if (gq1 > 7) { gq1 -= 16; }
        var gq2 = i32((gPacked >> 8u) & 0xFu); if (gq2 > 7) { gq2 -= 16; }
        var gq3 = i32((gPacked >> 12u) & 0xFu); if (gq3 > 7) { gq3 -= 16; }
        var gq4 = i32((gPacked >> 16u) & 0xFu); if (gq4 > 7) { gq4 -= 16; }
        var gq5 = i32((gPacked >> 20u) & 0xFu); if (gq5 > 7) { gq5 -= 16; }
        var gq6 = i32((gPacked >> 24u) & 0xFu); if (gq6 > 7) { gq6 -= 16; }
        var gq7 = i32((gPacked >> 28u) & 0xFu); if (gq7 > 7) { gq7 -= 16; }

        gateSum += (in0 * f32(gq0) + in1 * f32(gq1) + in2 * f32(gq2) + in3 * f32(gq3) +
                   in4 * f32(gq4) + in5 * f32(gq5) + in6 * f32(gq6) + in7 * f32(gq7)) * gScale;

        // Process Up
        var uq0 = i32(uPacked & 0xFu); if (uq0 > 7) { uq0 -= 16; }
        var uq1 = i32((uPacked >> 4u) & 0xFu); if (uq1 > 7) { uq1 -= 16; }
        var uq2 = i32((uPacked >> 8u) & 0xFu); if (uq2 > 7) { uq2 -= 16; }
        var uq3 = i32((uPacked >> 12u) & 0xFu); if (uq3 > 7) { uq3 -= 16; }
        var uq4 = i32((uPacked >> 16u) & 0xFu); if (uq4 > 7) { uq4 -= 16; }
        var uq5 = i32((uPacked >> 20u) & 0xFu); if (uq5 > 7) { uq5 -= 16; }
        var uq6 = i32((uPacked >> 24u) & 0xFu); if (uq6 > 7) { uq6 -= 16; }
        var uq7 = i32((uPacked >> 28u) & 0xFu); if (uq7 > 7) { uq7 -= 16; }

        upSum += (in0 * f32(uq0) + in1 * f32(uq1) + in2 * f32(uq2) + in3 * f32(uq3) +
                 in4 * f32(uq4) + in5 * f32(uq5) + in6 * f32(uq6) + in7 * f32(uq7)) * uScale;
        
        i += 8u;
    }

    for (var k: u32 = 0u; k < rem; k++) {
        let globalIdx = base_w + i + k;
        let blockIdx = globalIdx / 32u;
        
        let gScale = gateScales[blockIdx];
        let uScale = upScales[blockIdx];
        
        let wordIdx = globalIdx / 8u;
        let nibbleIdx = globalIdx %% 8u;
        let gPacked = gateWeights[wordIdx];
        let uPacked = upWeights[wordIdx];
        
        var gQ = i32((gPacked >> (nibbleIdx * 4u)) & 0xFu); if (gQ > 7) { gQ -= 16; }
        var uQ = i32((uPacked >> (nibbleIdx * 4u)) & 0xFu); if (uQ > 7) { uQ -= 16; }
        
        let val = input[base_in + i + k];
        gateSum += val * (f32(gQ) * gScale);
        upSum   += val * (f32(uQ) * uScale);
    }

    output[b * params.outputSize + o] = silu(gateSum) * upSum;
}
`, tileSize)
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

fn silu(x: f32) -> f32 {
    return x * (1.0 / (1.0 + exp(-x)));
}

// tileSize is typically 32 or 64
@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let o = global_id.x;
    let b = wg_id.y;

    if (o >= params.outputSize || b >= params.batchSize) { return; }

    var gateSum: f32 = 0.0;
    var upSum: f32 = 0.0;
    
    let limit = params.inputSize / 4u;
    let rem = params.inputSize %% 4u;
    
    let base_in = b * params.inputSize;
    let base_w = o * params.inputSize;

    var i: u32 = 0u;
    for (var k: u32 = 0u; k < limit; k++) {
        let in0 = input[base_in + i];
        let in1 = input[base_in + i + 1u];
        let in2 = input[base_in + i + 2u];
        let in3 = input[base_in + i + 3u];

        gateSum += in0 * gateWeights[base_w + i]     + in1 * gateWeights[base_w + i + 1u] +
                   in2 * gateWeights[base_w + i + 2u] + in3 * gateWeights[base_w + i + 3u];

        upSum   += in0 * upWeights[base_w + i]       + in1 * upWeights[base_w + i + 1u] +
                   in2 * upWeights[base_w + i + 2u]   + in3 * upWeights[base_w + i + 3u];
        
        i += 4u;
    }

    for (var k: u32 = 0u; k < rem; k++) {
        let in_val = input[base_in + i + k];
        gateSum += in_val * gateWeights[base_w + i + k];
        upSum   += in_val * upWeights[base_w + i + k];
    }

    output[b * params.outputSize + o] = silu(gateSum) * upSum;
}
`, tileSize)
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

const ShaderRNNStep = `
struct RNNParams {
    batchSize: u32,
    inputSize: u32,
    hiddenSize: u32,
};

@group(0) @binding(0) var<uniform> params: RNNParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> hPrev: array<f32>;
@group(0) @binding(3) var<storage, read> wIH: array<f32>;
@group(0) @binding(4) var<storage, read> wHH: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> hCurr: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let h = global_id.x;
    let b = global_id.y;
    if (h >= params.hiddenSize || b >= params.batchSize) { return; }

    var sum: f32 = bias[h];
    
    // Input to Hidden
    let base_in = b * params.inputSize;
    let base_w_ih = h * params.inputSize;
    for (var i: u32 = 0u; i < params.inputSize; i++) {
        sum += input[base_in + i] * wIH[base_w_ih + i];
    }
    
    // Hidden to Hidden
    let base_h_prev = b * params.hiddenSize;
    let base_w_hh = h * params.hiddenSize;
    for (var i: u32 = 0u; i < params.hiddenSize; i++) {
        sum += hPrev[base_h_prev + i] * wHH[base_w_hh + i];
    }
    
    hCurr[b * params.hiddenSize + h] = tanh(sum);
}
`

const ShaderLSTMStep = `
struct LSTMParams {
    batchSize: u32,
    inputSize: u32,
    hiddenSize: u32,
};

@group(0) @binding(0) var<uniform> params: LSTMParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> hPrev: array<f32>;
@group(0) @binding(3) var<storage, read> cPrev: array<f32>;
@group(0) @binding(4) var<storage, read> weights: array<f32>; // [wI, wF, wG, wO] concatenated
@group(0) @binding(5) var<storage, read_write> hCurr: array<f32>;
@group(0) @binding(6) var<storage, read_write> cCurr: array<f32>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let h = global_id.x;
    let b = global_id.y;
    if (h >= params.hiddenSize || b >= params.batchSize) { return; }

    let ihSize = params.hiddenSize * params.inputSize;
    let hhSize = params.hiddenSize * params.hiddenSize;
    let gateSize = ihSize + hhSize + params.hiddenSize;
    
    // Weight offsets for gates
    let wF_off = gateSize;
    let wG_off = 2u * gateSize;
    let wO_off = 3u * gateSize;

    var iSum: f32 = weights[ihSize + hhSize + h];
    var fSum: f32 = weights[wF_off + ihSize + hhSize + h];
    var gSum: f32 = weights[wG_off + ihSize + hhSize + h];
    var oSum: f32 = weights[wO_off + ihSize + hhSize + h];

    let base_in = b * params.inputSize;
    let base_h_prev = b * params.hiddenSize;

    for (var i: u32 = 0u; i < params.inputSize; i++) {
        let x = input[base_in + i];
        let w_idx = h * params.inputSize + i;
        iSum += x * weights[w_idx];
        fSum += x * weights[wF_off + w_idx];
        gSum += x * weights[wG_off + w_idx];
        oSum += x * weights[wO_off + w_idx];
    }

    for (var hp: u32 = 0u; hp < params.hiddenSize; hp++) {
        let hv = hPrev[base_h_prev + hp];
        let w_idx = ihSize + h * params.hiddenSize + hp;
        iSum += hv * weights[w_idx];
        fSum += hv * weights[wF_off + w_idx];
        gSum += hv * weights[wG_off + w_idx];
        oSum += hv * weights[wO_off + w_idx];
    }

    let iG = sigmoid(iSum);
    let fG = sigmoid(fSum);
    let gG = tanh(gSum);
    let oG = sigmoid(oSum);

    let cell = fG * cPrev[base_h_prev + h] + iG * gG;
    cCurr[base_h_prev + h] = cell;
    hCurr[base_h_prev + h] = oG * tanh(cell);
}
`

const ShaderCNN1 = `
struct CNN1Params {
    batchSize: u32,
    inC: u32,
    inL: u32,
    outC: u32,
    outL: u32,
    kSize: u32,
    stride: u32,
    padding: u32,
};

@group(0) @binding(0) var<uniform> params: CNN1Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let o = global_id.x; // outL
    let f = global_id.y; // outC (filters)
    let b = global_id.z; // batch
    
    if (o >= params.outL || f >= params.outC || b >= params.batchSize) { return; }

    var sum: f32 = 0.0;
    for (var ic: u32 = 0u; ic < params.inC; ic++) {
        for (var k: u32 = 0u; k < params.kSize; k++) {
            let inPos = i32(o * params.stride + k) - i32(params.padding);
            if (inPos >= 0 && u32(inPos) < params.inL) {
                let inIdx = b * params.inC * params.inL + ic * params.inL + u32(inPos);
                let wIdx = f * params.inC * params.kSize + ic * params.kSize + k;
                sum += input[inIdx] * weights[wIdx];
            }
        }
    }
    
    output[b * params.outC * params.outL + f * params.outL + o] = sum;
}
`

const ShaderCNN2 = `
struct CNN2Params {
    batchSize: u32,
    inC: u32, inH: u32, inW: u32,
    outC: u32, outH: u32, outW: u32,
    kH: u32, kW: u32,
    strideH: u32, strideW: u32,
    padH: u32, padW: u32,
};

@group(0) @binding(0) var<uniform> params: CNN2Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let w = global_id.x; // outW
    let h = global_id.y; // outH
    let f_batch = global_id.z; // filter + batch? No, lets use z for batch, and loop filter or vice versa
    
    // Better: x=outW*outH, y=outC, z=batch
    let oArea = params.outW * params.outH;
    let outIdx_flat = global_id.x;
    let filterIdx = global_id.y;
    let batchIdx = global_id.z;
    
    if (outIdx_flat >= oArea || filterIdx >= params.outC || batchIdx >= params.batchSize) { return; }
    
    let outX = outIdx_flat % params.outW;
    let outY = outIdx_flat / params.outW;

    var sum: f32 = 0.0;
    for (var ic: u32 = 0u; ic < params.inC; ic++) {
        for (var kh: u32 = 0u; kh < params.kH; kh++) {
            for (var kw: u32 = 0u; kw < params.kW; kw++) {
                let inY = i32(outY * params.strideH + kh) - i32(params.padH);
                let inX = i32(outX * params.strideW + kw) - i32(params.padW);
                
                if (inY >= 0 && u32(inY) < params.inH && inX >= 0 && u32(inX) < params.inW) {
                    let inIdx = batchIdx * params.inC * params.inH * params.inW +
                               ic * params.inH * params.inW +
                               u32(inY) * params.inW + u32(inX);
                    let wIdx = filterIdx * params.inC * params.kH * params.kW +
                               ic * params.kH * params.kW +
                               kh * params.kW + kw;
                    sum += input[inIdx] * weights[wIdx];
                }
            }
        }
    }
    output[batchIdx * params.outC * oArea + filterIdx * oArea + outIdx_flat] = sum;
}
`

const ShaderCNN3 = `
struct CNN3Params {
    batchSize: u32,
    inC: u32, inD: u32, inH: u32, inW: u32,
    outC: u32, outD: u32, outH: u32, outW: u32,
    kD: u32, kH: u32, kW: u32,
    sD: u32, sH: u32, sW: u32,
    pD: u32, pH: u32, pW: u32,
};

@group(0) @binding(0) var<uniform> params: CNN3Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outIdx_flat = global_id.x;
    let filterIdx = global_id.y;
    let batchIdx = global_id.z;
    
    let oArea = params.outD * params.outH * params.outW;
    if (outIdx_flat >= oArea || filterIdx >= params.outC || batchIdx >= params.batchSize) { return; }
    
    let outW = outIdx_flat % params.outW;
    let remainder = outIdx_flat / params.outW;
    let outH = remainder % params.outH;
    let outD = remainder / params.outH;

    var sum: f32 = 0.0;
    for (var ic: u32 = 0u; ic < params.inC; ic++) {
        for (var kd: u32 = 0u; kd < params.kD; kd++) {
            for (var kh: u32 = 0u; kh < params.kH; kh++) {
                for (var kw: u32 = 0u; kw < params.kW; kw++) {
                    let inD = i32(outD * params.sD + kd) - i32(params.pD);
                    let inH = i32(outH * params.sH + kh) - i32(params.pH);
                    let inX = i32(outW * params.sW + kw) - i32(params.pW);
                    
                    if (inD >= 0 && u32(inD) < params.inD &&
                        inH >= 0 && u32(inH) < params.inH &&
                        inX >= 0 && u32(inX) < params.inW) {
                        
                        let inIdx = batchIdx * params.inC * params.inD * params.inH * params.inW +
                                   ic * params.inD * params.inH * params.inW +
                                   u32(inD) * params.inH * params.inW +
                                   u32(inH) * params.inW + u32(inX);
                        let wIdx = filterIdx * params.inC * params.kD * params.kH * params.kW +
                                   ic * params.kD * params.kH * params.kW +
                                   kd * params.kH * params.kW +
                                   kh * params.kW + kw;
                        sum += input[inIdx] * weights[wIdx];
                    }
                }
            }
        }
    }
    output[batchIdx * params.outC * oArea + filterIdx * oArea + outIdx_flat] = sum;
}
`
