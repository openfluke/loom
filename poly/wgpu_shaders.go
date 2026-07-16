package poly

import "fmt"

// WGSL Shaders for FlashPoly Tiling Acceleration
//
// ShaderTiledDense, ShaderTiledSwiGLU, and ShaderTiledMHA are generated
// dynamically so the WGSL workgroup array sizes always match the runtime
// tile size (WGSL doesn't allow runtime-sized workgroup arrays).

const wgslActivate = `
fn activate(v: f32, act: u32) -> f32 {
    if (act == 0u) { return max(0.0, v); } // ReLU
    if (act == 1u) { return v * (1.0 / (1.0 + exp(-v))); } // SiLU
    if (act == 2u) {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return 0.5 * v * (1.0 + tanh(0.7978845608 * (v + 0.044715 * v * v * v)));
    }
    if (act == 3u) { return tanh(v); } // Tanh
    if (act == 4u) { return 1.0 / (1.0 + exp(-v)); } // Sigmoid
    if (act == 5u) {
        if (v < 0.0) { return v * 0.01; } // LeakyReLU
        return v;
    }
    if (act == 6u) {
        let r = max(0.0, v);
        return r * r; // ReLU2
    }
    return v; // Linear/Default
}
`

const ShaderDenseScaled = `
struct DenseScaleParams {
    batchSize: u32,
    inputSize: u32,
    outputSize: u32,
    activation: u32,
    scale: f32,
    hasBias: u32,
    totalOutStride: u32,
    outputRowBase: u32,
    p3: u32, p4: u32, p5: u32, p6: u32,
};

@group(0) @binding(0) var<uniform>             params:  DenseScaleParams;
@group(0) @binding(1) var<storage, read>       input:   array<f32>;
@group(0) @binding(2) var<storage, read>       weights: array<f32>;
@group(0) @binding(3) var<storage, read>       bias:    array<f32>;
@group(0) @binding(4) var<storage, read_write> output:  array<f32>;

` + wgslActivate + `

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) wg_id: vec3<u32>) {
    let o = global_id.x;
    let b = wg_id.y;
    if (o >= params.outputSize || b >= params.batchSize) { return; }

    let outStride = select(params.outputSize, params.totalOutStride, params.totalOutStride != 0u);

    var sum: f32 = 0.0;
    let base_in = b * params.inputSize;
    let base_w = o * params.inputSize;

    for (var i: u32 = 0u; i < params.inputSize; i++) {
        sum += input[base_in + i] * weights[base_w + i];
    }
    
    var res = sum * params.scale;
    if (params.hasBias != 0u) {
        res += bias[o];
    }
    
    output[b * outStride + params.outputRowBase + o] = activate(res, params.activation);
}
`

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

// ShaderTiledDenseI8 generates a tiled dense shader that dequantizes 8-bit weights on the fly.
// 4 weights per u32 word.
func ShaderTiledDenseI8(tileSize int) string {
	return fmt.Sprintf(`
struct Params {
    batchSize: u32,
    inputSize: u32,
    outputSize: u32,
    tileSize: u32,
    scale: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

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
    
    // We process 4 elements at a time (one u32 word)
    let limit = params.inputSize / 4u;
    let rem = params.inputSize %% 4u;
    
    let base_in = b * params.inputSize;
    let base_w = o * params.inputSize;

    var i: u32 = 0u;
    for (var k: u32 = 0u; k < limit; k++) {
        let wordIdx = (base_w + i) / 4u;
        let packed = weights[wordIdx];
        
        let in0 = input[base_in + i];
        let in1 = input[base_in + i + 1u];
        let in2 = input[base_in + i + 2u];
        let in3 = input[base_in + i + 3u];

        // Byte 0
        var q0 = i32(packed & 0xFFu); if (q0 > 127) { q0 -= 256; }
        // Byte 1
        var q1 = i32((packed >> 8u) & 0xFFu); if (q1 > 127) { q1 -= 256; }
        // Byte 2
        var q2 = i32((packed >> 16u) & 0xFFu); if (q2 > 127) { q2 -= 256; }
        // Byte 3
        var q3 = i32((packed >> 24u) & 0xFFu); if (q3 > 127) { q3 -= 256; }

        sum += in0 * f32(q0) + in1 * f32(q1) + in2 * f32(q2) + in3 * f32(q3);
        i += 4u;
    }

    for (var k: u32 = 0u; k < rem; k++) {
        let globalIdx = base_w + i + k;
        let wordIdx = globalIdx / 4u;
        let byteIdx = globalIdx %% 4u;
        let packed = weights[wordIdx];
        
        var q = i32((packed >> (byteIdx * 8u)) & 0xFFu);
        if (q > 127) { q -= 256; }
        
        sum += input[base_in + i + k] * f32(q);
    }

    output[b * params.outputSize + o] = sum * params.scale;
}
`, tileSize)
}

// ShaderTiledDenseBitNetTernary generates a dense kernel for BitNet packed
// ternary matrices. Each u32 stores 16 weights as 2-bit codes:
// 0 -> -1, 1 -> 0, 2 -> +1. Activations are quantized per token/row.
func ShaderTiledDenseBitNetTernary(tileSize int) string {
	return fmt.Sprintf(`
struct Params {
    batchSize: u32,
    inputSize: u32,
    outputSize: u32,
    rowWords: u32,
    weightScale: f32,
    activation: u32,
    hasBias: u32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

`+wgslActivate+`

fn ternary_value(code: u32) -> f32 {
    if (code == 0u) { return -1.0; }
    if (code == 2u) { return 1.0; }
    return 0.0;
}

fn quantize_activation(v: f32, maxAbs: f32) -> f32 {
    let scale = 127.0 / max(maxAbs, 0.00001);
    return clamp(round(v * scale), -128.0, 127.0);
}

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let o = global_id.x;
    let b = wg_id.y;
    if (o >= params.outputSize || b >= params.batchSize) { return; }

    let baseIn = b * params.inputSize;
    var maxAbs = 0.0;
    for (var i: u32 = 0u; i < params.inputSize; i++) {
        maxAbs = max(maxAbs, abs(input[baseIn + i]));
    }
    maxAbs = max(maxAbs, 0.00001);

    let rowBase = o * params.rowWords;
    var sum = 0.0;
    for (var i: u32 = 0u; i < params.inputSize; i++) {
        let word = weights[rowBase + (i / 16u)];
        let shift = (i %% 16u) * 2u;
        let code = (word >> shift) & 0x3u;
        let q = quantize_activation(input[baseIn + i], maxAbs);
        sum += q * ternary_value(code);
    }

    var res = sum * params.weightScale * (maxAbs / 127.0);
    if (params.hasBias != 0u) {
        res += bias[o];
    }
    output[b * params.outputSize + o] = activate(res, params.activation);
}
`, tileSize)
}

const ShaderBitNetQuantizeActivation = `
struct Params {
    batchSize: u32,
    inputSize: u32,
    qWords: u32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> qPacked: array<u32>;
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;

fn quant_byte(v: f32, maxAbs: f32) -> u32 {
    let scale = 127.0 / max(maxAbs, 0.00001);
    let q = i32(clamp(round(v * scale), -128.0, 127.0));
    return u32(q) & 0xFFu;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= params.batchSize) { return; }

    let baseIn = b * params.inputSize;
    var maxAbs = 0.0;
    for (var i: u32 = 0u; i < params.inputSize; i++) {
        maxAbs = max(maxAbs, abs(input[baseIn + i]));
    }
    maxAbs = max(maxAbs, 0.00001);
    scales[b] = maxAbs / 127.0;

    let baseQ = b * params.qWords;
    for (var w: u32 = 0u; w < params.qWords; w++) {
        var packed = 0u;
        for (var lane: u32 = 0u; lane < 4u; lane++) {
            let i = w * 4u + lane;
            if (i < params.inputSize) {
                let qb = quant_byte(input[baseIn + i], maxAbs);
                packed |= qb << (lane * 8u);
            }
        }
        qPacked[baseQ + w] = packed;
    }
}
`

// ShaderTiledDenseBitNetTernaryQuantized consumes activations already quantized
// and packed as 4 signed int8 lanes per u32, one activation scale per row.
func ShaderTiledDenseBitNetTernaryQuantized(tileSize int) string {
	return fmt.Sprintf(`
struct Params {
    batchSize: u32,
    inputSize: u32,
    outputSize: u32,
    rowWords: u32,
    qWords: u32,
    activation: u32,
    hasBias: u32,
    pad0: u32,
    weightScale: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> qPacked: array<u32>;
@group(0) @binding(2) var<storage, read> inputScales: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<u32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

`+wgslActivate+`

fn ternary_value(code: u32) -> f32 {
    if (code == 0u) { return -1.0; }
    if (code == 2u) { return 1.0; }
    return 0.0;
}

fn q_i8(word: u32, lane: u32) -> i32 {
    var q = i32((word >> (lane * 8u)) & 0xFFu);
    if (q > 127) { q -= 256; }
    return q;
}

fn add_lane(acc: i32, qWord: u32, qLane: u32, wWord: u32, wLane: u32) -> i32 {
    let q = q_i8(qWord, qLane);
    let code = (wWord >> (wLane * 2u)) & 0x3u;
    if (code == 0u) {
        return acc - q;
    }
    if (code == 2u) {
        return acc + q;
    }
    return acc;
}

fn dot16(acc0: i32, q0: u32, q1: u32, q2: u32, q3: u32, w: u32) -> i32 {
    var acc = acc0;
    acc = add_lane(acc, q0, 0u, w, 0u);
    acc = add_lane(acc, q0, 1u, w, 1u);
    acc = add_lane(acc, q0, 2u, w, 2u);
    acc = add_lane(acc, q0, 3u, w, 3u);
    acc = add_lane(acc, q1, 0u, w, 4u);
    acc = add_lane(acc, q1, 1u, w, 5u);
    acc = add_lane(acc, q1, 2u, w, 6u);
    acc = add_lane(acc, q1, 3u, w, 7u);
    acc = add_lane(acc, q2, 0u, w, 8u);
    acc = add_lane(acc, q2, 1u, w, 9u);
    acc = add_lane(acc, q2, 2u, w, 10u);
    acc = add_lane(acc, q2, 3u, w, 11u);
    acc = add_lane(acc, q3, 0u, w, 12u);
    acc = add_lane(acc, q3, 1u, w, 13u);
    acc = add_lane(acc, q3, 2u, w, 14u);
    acc = add_lane(acc, q3, 3u, w, 15u);
    return acc;
}

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let o = global_id.x;
    let b = wg_id.y;
    if (o >= params.outputSize || b >= params.batchSize) { return; }

    let qBase = b * params.qWords;
    let wBase = o * params.rowWords;
    var sum: i32 = 0;
    let fullWords = params.inputSize / 16u;
    for (var wIdx: u32 = 0u; wIdx < fullWords; wIdx++) {
        let qOff = wIdx * 4u;
        sum = dot16(
            sum,
            qPacked[qBase + qOff],
            qPacked[qBase + qOff + 1u],
            qPacked[qBase + qOff + 2u],
            qPacked[qBase + qOff + 3u],
            weights[wBase + wIdx],
        );
    }

    for (var i: u32 = fullWords * 16u; i < params.inputSize; i++) {
        let q = q_i8(qPacked[qBase + (i / 4u)], i %% 4u);
        let code = (weights[wBase + (i / 16u)] >> ((i %% 16u) * 2u)) & 0x3u;
        if (code == 0u) {
            sum -= q;
        } else if (code == 2u) {
            sum += q;
        }
    }

    var res = f32(sum) * params.weightScale * inputScales[b];
    if (params.hasBias != 0u) {
        res += bias[o];
    }
    output[b * params.outputSize + o] = activate(res, params.activation);
}
`, tileSize)
}

// ShaderTiledDenseBitNetTernaryQuantizedReduce uses a full workgroup to compute
// one output row. Each lane walks a strided subset of packed 16-column ternary
// words, then a workgroup reduction combines the partial int32 dot products.
func ShaderTiledDenseBitNetTernaryQuantizedReduce(tileSize int) string {
	return fmt.Sprintf(`
struct Params {
    batchSize: u32,
    inputSize: u32,
    outputSize: u32,
    rowWords: u32,
    qWords: u32,
    activation: u32,
    hasBias: u32,
    pad0: u32,
    weightScale: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> qPacked: array<u32>;
@group(0) @binding(2) var<storage, read> inputScales: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<u32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

var<workgroup> partials: array<i32, %d>;

`+wgslActivate+`

fn q_i8(word: u32, lane: u32) -> i32 {
    var q = i32((word >> (lane * 8u)) & 0xFFu);
    if (q > 127) { q -= 256; }
    return q;
}

fn add_lane(acc: i32, qWord: u32, qLane: u32, wWord: u32, wLane: u32) -> i32 {
    let q = q_i8(qWord, qLane);
    let code = (wWord >> (wLane * 2u)) & 0x3u;
    if (code == 0u) {
        return acc - q;
    }
    if (code == 2u) {
        return acc + q;
    }
    return acc;
}

fn dot16(acc0: i32, q0: u32, q1: u32, q2: u32, q3: u32, w: u32) -> i32 {
    var acc = acc0;
    acc = add_lane(acc, q0, 0u, w, 0u);
    acc = add_lane(acc, q0, 1u, w, 1u);
    acc = add_lane(acc, q0, 2u, w, 2u);
    acc = add_lane(acc, q0, 3u, w, 3u);
    acc = add_lane(acc, q1, 0u, w, 4u);
    acc = add_lane(acc, q1, 1u, w, 5u);
    acc = add_lane(acc, q1, 2u, w, 6u);
    acc = add_lane(acc, q1, 3u, w, 7u);
    acc = add_lane(acc, q2, 0u, w, 8u);
    acc = add_lane(acc, q2, 1u, w, 9u);
    acc = add_lane(acc, q2, 2u, w, 10u);
    acc = add_lane(acc, q2, 3u, w, 11u);
    acc = add_lane(acc, q3, 0u, w, 12u);
    acc = add_lane(acc, q3, 1u, w, 13u);
    acc = add_lane(acc, q3, 2u, w, 14u);
    acc = add_lane(acc, q3, 3u, w, 15u);
    return acc;
}

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let o = wg_id.x;
    let b = wg_id.y;
    let lid = local_id.x;
    if (o >= params.outputSize || b >= params.batchSize) { return; }

    let qBase = b * params.qWords;
    let wBase = o * params.rowWords;
    let fullWords = params.inputSize / 16u;
    var sum: i32 = 0;

    for (var wIdx = lid; wIdx < fullWords; wIdx += %du) {
        let qOff = wIdx * 4u;
        sum = dot16(
            sum,
            qPacked[qBase + qOff],
            qPacked[qBase + qOff + 1u],
            qPacked[qBase + qOff + 2u],
            qPacked[qBase + qOff + 3u],
            weights[wBase + wIdx]
        );
    }

    if (lid == 0u) {
        for (var i: u32 = fullWords * 16u; i < params.inputSize; i++) {
            let q = q_i8(qPacked[qBase + (i / 4u)], i %% 4u);
            let code = (weights[wBase + (i / 16u)] >> ((i %% 16u) * 2u)) & 0x3u;
            if (code == 0u) {
                sum -= q;
            } else if (code == 2u) {
                sum += q;
            }
        }
    }

    partials[lid] = sum;
    workgroupBarrier();

    var stride = %du / 2u;
    loop {
        if (lid < stride) {
            partials[lid] = partials[lid] + partials[lid + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lid == 0u) {
        var res = f32(partials[0]) * params.weightScale * inputScales[b];
        if (params.hasBias != 0u) {
            res += bias[o];
        }
        output[b * params.outputSize + o] = activate(res, params.activation);
    }
}
`, tileSize, tileSize, tileSize, tileSize)
}

const ShaderBitNetGateProduct = `
struct Params {
    batchSize: u32,
    hiddenSize: u32,
    activation: u32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

` + wgslActivate + `

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.batchSize * params.hiddenSize;
    if (idx >= total) { return; }
    output[idx] = activate(gate[idx], params.activation) * up[idx];
}
`

func ShaderTiledDenseN(tileSize int) string {
	return ShaderTiledDense(tileSize)
}

func ShaderTiledDense(tileSize int) string {
	return fmt.Sprintf(`
struct DenseScaleParams {
    batchSize: u32,
    inputSize: u32,
    outputSize: u32,
    activation: u32,
    scale: f32,
    hasBias: u32,
    totalOutStride: u32,
    outputRowBase: u32,
    p3: u32, p4: u32, p5: u32, p6: u32,
};

@group(0) @binding(0) var<uniform>             params:  DenseScaleParams;
@group(0) @binding(1) var<storage, read>       input:   array<f32>;
@group(0) @binding(2) var<storage, read>       weights: array<f32>;
@group(0) @binding(3) var<storage, read>       bias:    array<f32>;
@group(0) @binding(4) var<storage, read_write> output:  array<f32>;

// Workgroup input cache — holds one row's input values.
// Since inputSize can be large, we cache it in chunks of tileSize.
var<workgroup> iCache: array<f32, %d>;

`+wgslActivate+`

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id:  vec3<u32>,
    @builtin(workgroup_id)        wg_id:     vec3<u32>
) {
    let o = global_id.x;
    let b = wg_id.y;
    let tid = local_id.x;
    let tileSize: u32 = %du;

    if (o >= params.outputSize || b >= params.batchSize) { return; }

    let outStride = select(params.outputSize, params.totalOutStride, params.totalOutStride != 0u);

    var sum: f32 = 0.0;
    let base_in = b * params.inputSize;
    let base_w = o * params.inputSize;

    // Process input in tiles
    for (var iTile: u32 = 0u; iTile < params.inputSize; iTile += tileSize) {
        let iIdx = iTile + tid;
        if (iIdx < params.inputSize) {
            iCache[tid] = input[base_in + iIdx];
        } else {
            iCache[tid] = 0.0;
        }
        
        workgroupBarrier();
        
        let limit = min(tileSize, params.inputSize - iTile);
        for (var i: u32 = 0u; i < limit; i++) {
            sum += iCache[i] * weights[base_w + iTile + i];
        }
        
        workgroupBarrier();
    }

    var res = sum * params.scale;
    if (params.hasBias != 0u) {
        res += bias[o];
    }
    
    output[b * outStride + params.outputRowBase + o] = activate(res, params.activation);
}
`, tileSize, tileSize, tileSize)
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
@group(0) @binding(6) var<storage, read> gateBiases: array<f32>;
@group(0) @binding(7) var<storage, read> upBiases: array<f32>;
@group(0) @binding(8) var<storage, read_write> output: array<f32>;

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

    gateSum += gateBiases[o];
    upSum   += upBiases[o];

    output[b * params.outputSize + o] = silu(gateSum) * upSum;
}
`, tileSize)
}

// ShaderTiledSwiGLUI8 generates a tiled SwiGLU shader with INT8 weights.
func ShaderTiledSwiGLUI8(tileSize int) string {
	return fmt.Sprintf(`
struct Params {
    batchSize: u32,
    inputSize: u32,
    outputSize: u32,
    tileSize: u32,
    gScale: f32,
    uScale: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gateWeights: array<u32>;
@group(0) @binding(3) var<storage, read> upWeights: array<u32>;
@group(0) @binding(4) var<storage, read> gateBiases: array<f32>;
@group(0) @binding(5) var<storage, read> upBiases: array<f32>;
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
    
    let limit = params.inputSize / 4u;
    let rem = params.inputSize %% 4u;
    
    let base_in = b * params.inputSize;
    let base_w = o * params.inputSize;

    var i: u32 = 0u;
    for (var k: u32 = 0u; k < limit; k++) {
        let wordIdx = (base_w + i) / 4u;
        let gPacked = gateWeights[wordIdx];
        let uPacked = upWeights[wordIdx];
        
        let in0 = input[base_in + i];
        let in1 = input[base_in + i + 1u];
        let in2 = input[base_in + i + 2u];
        let in3 = input[base_in + i + 3u];

        // Unpack Gate
        var gq0 = i32(gPacked & 0xFFu); if (gq0 > 127) { gq0 -= 256; }
        var gq1 = i32((gPacked >> 8u) & 0xFFu); if (gq1 > 127) { gq1 -= 256; }
        var gq2 = i32((gPacked >> 16u) & 0xFFu); if (gq2 > 127) { gq2 -= 256; }
        var gq3 = i32((gPacked >> 24u) & 0xFFu); if (gq3 > 127) { gq3 -= 256; }
        gateSum += in0 * f32(gq0) + in1 * f32(gq1) + in2 * f32(gq2) + in3 * f32(gq3);

        // Unpack Up
        var uq0 = i32(uPacked & 0xFFu); if (uq0 > 127) { uq0 -= 256; }
        var uq1 = i32((uPacked >> 8u) & 0xFFu); if (uq1 > 127) { uq1 -= 256; }
        var uq2 = i32((uPacked >> 16u) & 0xFFu); if (uq2 > 127) { uq2 -= 256; }
        var uq3 = i32((uPacked >> 24u) & 0xFFu); if (uq3 > 127) { uq3 -= 256; }
        upSum += in0 * f32(uq0) + in1 * f32(uq1) + in2 * f32(uq2) + in3 * f32(uq3);

        i += 4u;
    }

    for (var k: u32 = 0u; k < rem; k++) {
        let globalIdx = base_w + i + k;
        let wordIdx = globalIdx / 4u;
        let byteIdx = globalIdx %% 4u;
        
        var gQ = i32((gateWeights[wordIdx] >> (byteIdx * 8u)) & 0xFFu); if (gQ > 127) { gQ -= 256; }
        var uQ = i32((upWeights[wordIdx] >> (byteIdx * 8u)) & 0xFFu); if (uQ > 127) { uQ -= 256; }
        
        let val = input[base_in + i + k];
        gateSum += val * f32(gQ);
        upSum   += val * f32(uQ);
    }

    gateSum = gateSum * params.gScale + gateBiases[o];
    upSum   = upSum * params.uScale + upBiases[o];

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
@group(0) @binding(4) var<storage, read> gateBiases: array<f32>;
@group(0) @binding(5) var<storage, read> upBiases: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;

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

    gateSum += gateBiases[o];
    upSum   += upBiases[o];

    output[b * params.outputSize + o] = silu(gateSum) * upSum;
}
`, tileSize)
}

// ShaderTiledSwiGLUActCache is like ShaderTiledSwiGLUN but also stores the raw gate and up
// projections (before SiLU) to separate buffers needed for the backward pass.
func ShaderTiledSwiGLUActCache(tileSize int) string {
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
@group(0) @binding(4) var<storage, read> gateBiases: array<f32>;
@group(0) @binding(5) var<storage, read> upBiases: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;
@group(0) @binding(7) var<storage, read_write> gateOut: array<f32>;
@group(0) @binding(8) var<storage, read_write> upOut: array<f32>;

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

    gateSum += gateBiases[o];
    upSum   += upBiases[o];

    let idx = b * params.outputSize + o;
    gateOut[idx] = gateSum;
    upOut[idx]   = upSum;
    output[idx]  = silu(gateSum) * upSum;
}
`, tileSize)
}

// ShaderTiledMHAN generates a tiled MHA shader for the given tile size and headDim.
// Both are baked in as WGSL compile-time constants.
func ShaderTiledMHAN(tileSize, headDim int) string {
	return shaderTiledMHAN(tileSize, headDim, false)
}

// ShaderTiledMHANStep reads decode position from step[0] (stable uniforms for chunked decode).
func ShaderTiledMHANStep(tileSize, headDim int) string {
	return shaderTiledMHAN(tileSize, headDim, true)
}

func shaderTiledMHAN(tileSize, headDim int, useStep bool) string {
	// tile_k and tile_v each hold tileSize rows of headDim floats
	kvArraySize := tileSize * headDim
	wgSize := 64
	if headDim > 64 {
		wgSize = 128
	}
	if headDim > 128 {
		wgSize = 256
	}
	stepBind := ""
	posExpr := "params.kvOffset + s"
	if useStep {
		stepBind = `
@group(0) @binding(5) var<storage, read> step: array<u32>;`
		posExpr = "step[0] + s"
	}
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
%s

var<workgroup> tile_q: array<f32, %d>;   // headDim
var<workgroup> tile_k: array<f32, %d>;   // tileSize * headDim
var<workgroup> tile_v: array<f32, %d>;   // tileSize * headDim

@compute @workgroup_size(%d, 1, 1)
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
    let currentTotalPos = %s;
    let totalKLen = currentTotalPos + 1u;

    let scale = 1.0 / sqrt(f32(headDim));

    for (var d: u32 = tid; d < headDim; d += %du) {
        tile_q[d] = q[(s * params.numHeads + h) * headDim + d];
    }
    workgroupBarrier();

    var max_score: f32 = -1e38;
    var denom: f32 = 0.0;
    var local_v_acc: f32 = 0.0;

    for (var kTile: u32 = 0u; kTile < totalKLen; kTile += params.tileSize) {
        let currentKSize = min(params.tileSize, totalKLen - kTile);

        for (var i: u32 = tid; i < currentKSize * headDim; i += %du) {
            let row = i / headDim;
            let col = i %% headDim;
            let kvIdx = (kvH * params.maxSeqLen + (kTile + row)) * params.headDim + col;
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
                if (tid < headDim) {
                    local_v_acc = local_v_acc * exp_factor + tile_v[j * headDim + tid];
                }
            } else {
                let exp_val = exp(score - max_score);
                denom += exp_val;
                if (tid < headDim) {
                    local_v_acc += tile_v[j * headDim + tid] * exp_val;
                }
            }
        }
        workgroupBarrier();
    }

    if (tid < headDim) {
        output[(s * params.numHeads + h) * headDim + tid] = local_v_acc / denom;
    }
}
`, stepBind, headDim, kvArraySize, kvArraySize, wgSize, posExpr, wgSize, wgSize)
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
    pad0: u32, pad1: u32, pad2: u32, pad3: u32, pad4: u32, pad5: u32, pad6: u32,
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

// ShaderEmbeddingShard loads embeddings when the weight matrix is split across bindings
// (WebGPU maxStorageBufferBindingSize). Each dispatch binds rows [rowOffset, rowOffset + numRows).
const ShaderEmbeddingShard = `
struct Params {
    vocabSize: u32,
    hiddenSize: u32,
    numTokens: u32,
    _pad: u32,
    rowOffset: u32,
    numRows: u32,
    _pad2: u32,
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
        return;
    }
    if (vocabIdx < params.rowOffset || vocabIdx >= params.rowOffset + params.numRows) {
        return;
    }

    let lr = vocabIdx - params.rowOffset;
    output[tid] = weights[lr * params.hiddenSize + dimIdx];
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

// ShaderLSTMStepPreAct is like ShaderLSTMStep but also writes
// [iS, fS, gS, oS, cC] per (batch, hidden) to preAct for use in the backward pass.
// preAct layout: batchSize × 5 × hiddenSize.
const ShaderLSTMStepPreAct = `
struct LSTMParams {
    batchSize: u32,
    inputSize: u32,
    hiddenSize: u32,
};

@group(0) @binding(0) var<uniform>             params:  LSTMParams;
@group(0) @binding(1) var<storage, read>       input:   array<f32>;
@group(0) @binding(2) var<storage, read>       hPrev:   array<f32>;
@group(0) @binding(3) var<storage, read>       cPrev:   array<f32>;
@group(0) @binding(4) var<storage, read>       weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> hCurr:   array<f32>;
@group(0) @binding(6) var<storage, read_write> cCurr:   array<f32>;
@group(0) @binding(7) var<storage, read_write> preAct:  array<f32>; // [batchSize, 5*hiddenSize]

fn lstmpa_sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let h = global_id.x;
    let b = global_id.y;
    if (h >= params.hiddenSize || b >= params.batchSize) { return; }

    let ihSize   = params.hiddenSize * params.inputSize;
    let hhSize   = params.hiddenSize * params.hiddenSize;
    let gateSize = ihSize + hhSize + params.hiddenSize;

    let wF_off = gateSize;
    let wG_off = 2u * gateSize;
    let wO_off = 3u * gateSize;

    var iSum: f32 = weights[ihSize + hhSize + h];
    var fSum: f32 = weights[wF_off + ihSize + hhSize + h];
    var gSum: f32 = weights[wG_off + ihSize + hhSize + h];
    var oSum: f32 = weights[wO_off + ihSize + hhSize + h];

    let base_in     = b * params.inputSize;
    let base_h_prev = b * params.hiddenSize;

    for (var i: u32 = 0u; i < params.inputSize; i++) {
        let x     = input[base_in + i];
        let w_idx = h * params.inputSize + i;
        iSum += x * weights[w_idx];
        fSum += x * weights[wF_off + w_idx];
        gSum += x * weights[wG_off + w_idx];
        oSum += x * weights[wO_off + w_idx];
    }
    for (var hp: u32 = 0u; hp < params.hiddenSize; hp++) {
        let hv    = hPrev[base_h_prev + hp];
        let w_idx = ihSize + h * params.hiddenSize + hp;
        iSum += hv * weights[w_idx];
        fSum += hv * weights[wF_off + w_idx];
        gSum += hv * weights[wG_off + w_idx];
        oSum += hv * weights[wO_off + w_idx];
    }

    let iG   = lstmpa_sigmoid(iSum);
    let fG   = lstmpa_sigmoid(fSum);
    let gG   = tanh(gSum);
    let oG   = lstmpa_sigmoid(oSum);
    let cell = fG * cPrev[base_h_prev + h] + iG * gG;

    cCurr[base_h_prev + h] = cell;
    hCurr[base_h_prev + h] = oG * tanh(cell);

    let pIdx = b * 5u * params.hiddenSize;
    preAct[pIdx + h]                              = iSum;
    preAct[pIdx + params.hiddenSize + h]          = fSum;
    preAct[pIdx + 2u * params.hiddenSize + h]     = gSum;
    preAct[pIdx + 3u * params.hiddenSize + h]     = oSum;
    preAct[pIdx + 4u * params.hiddenSize + h]     = cell;
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

const ShaderAdd = `
struct Params {
    size: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> res: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.size) { return; }
    res[global_id.x] = a[global_id.x] + b[global_id.x];
}
`
