package poly

// Decode-shaped BitNet kernels (Microsoft-style int8 acts × packed ternary weights).
// Modeled on ShaderDecodeQ4GEMV: one shared load of quantized x per workgroup.

// ShaderBitNetQuantizeActivationParallel: workgroup-parallel absmax + pack.
// Replaces the old @workgroup_size(1) serial ShaderBitNetQuantizeActivation.
const ShaderBitNetQuantizeActivationParallel = `
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

var<workgroup> partialMax: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let b = wg_id.x;
    if (b >= params.batchSize) { return; }
    let tid = lid.x;
    let baseIn = b * params.inputSize;

    var localMax = 0.0;
    for (var i = tid; i < params.inputSize; i += 256u) {
        localMax = max(localMax, abs(input[baseIn + i]));
    }
    partialMax[tid] = localMax;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (tid < stride) {
            partialMax[tid] = max(partialMax[tid], partialMax[tid + stride]);
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride / 2u;
    }

    let maxAbs = max(partialMax[0], 0.00001);
    if (tid == 0u) {
        scales[b] = maxAbs / 127.0;
    }
    workgroupBarrier();

    let scale = 127.0 / maxAbs;
    let baseQ = b * params.qWords;
    for (var w = tid; w < params.qWords; w += 256u) {
        var packed = 0u;
        for (var lane = 0u; lane < 4u; lane++) {
            let i = w * 4u + lane;
            if (i < params.inputSize) {
                let q = i32(clamp(round(input[baseIn + i] * scale), -128.0, 127.0));
                packed |= (u32(q) & 0xFFu) << (lane * 8u);
            }
        }
        qPacked[baseQ + w] = packed;
    }
}
`

// ShaderDecodeBitNetGEMV: batch=1, shared qPacked int8 words, one output row per lane.
// Supports inputSize up to 8192 (2048 qWords × 4).
// Params layout matches WGPUDenseBitNetTernaryQuantizedParams (batchSize ignored=1).
const ShaderDecodeBitNetGEMV = `
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

var<workgroup> qLocal: array<u32, 2048>;

` + wgslActivate + `

fn q_i8(word: u32, lane: u32) -> i32 {
    var q = i32((word >> (lane * 8u)) & 0xFFu);
    if (q > 127) { q -= 256; }
    return q;
}

fn add_lane(acc: i32, qWord: u32, qLane: u32, wWord: u32, wLane: u32) -> i32 {
    let q = q_i8(qWord, qLane);
    let code = (wWord >> (wLane * 2u)) & 0x3u;
    if (code == 0u) { return acc - q; }
    if (code == 2u) { return acc + q; }
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

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let qn = params.qWords;
    for (var i = tid; i < qn; i += 64u) {
        qLocal[i] = qPacked[i];
    }
    workgroupBarrier();

    let o = wg_id.x * 64u + tid;
    if (o >= params.outputSize) { return; }

    let wBase = o * params.rowWords;
    let fullWords = params.inputSize / 16u;
    var sum: i32 = 0;
    for (var wIdx = 0u; wIdx < fullWords; wIdx++) {
        let qOff = wIdx * 4u;
        sum = dot16(
            sum,
            qLocal[qOff],
            qLocal[qOff + 1u],
            qLocal[qOff + 2u],
            qLocal[qOff + 3u],
            weights[wBase + wIdx],
        );
    }
    for (var i = fullWords * 16u; i < params.inputSize; i++) {
        let q = q_i8(qLocal[i / 4u], i % 4u);
        let code = (weights[wBase + (i / 16u)] >> ((i % 16u) * 2u)) & 0x3u;
        if (code == 0u) { sum -= q; }
        else if (code == 2u) { sum += q; }
    }

    var res = f32(sum) * params.weightScale * inputScales[0];
    if (params.hasBias != 0u) {
        res += bias[o];
    }
    output[o] = activate(res, params.activation);
}
`

// ShaderDecodeBitNetGEMV_QKV: one shared q load → Q|K|V (decode, batch=1).
const ShaderDecodeBitNetGEMV_QKV = `
struct Params {
    inputSize: u32,
    qDim: u32,
    kvDim: u32,
    rowWords: u32,
    qWords: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    qScale: f32,
    kScale: f32,
    vScale: f32,
    pad3: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> qPacked: array<u32>;
@group(0) @binding(2) var<storage, read> inputScales: array<f32>;
@group(0) @binding(3) var<storage, read> qWeights: array<u32>;
@group(0) @binding(4) var<storage, read> kWeights: array<u32>;
@group(0) @binding(5) var<storage, read> vWeights: array<u32>;
@group(0) @binding(6) var<storage, read_write> qOut: array<f32>;
@group(0) @binding(7) var<storage, read_write> kOut: array<f32>;
@group(0) @binding(8) var<storage, read_write> vOut: array<f32>;

var<workgroup> qLocal: array<u32, 2048>;

fn q_i8(word: u32, lane: u32) -> i32 {
    var q = i32((word >> (lane * 8u)) & 0xFFu);
    if (q > 127) { q -= 256; }
    return q;
}

fn add_lane(acc: i32, qWord: u32, qLane: u32, wWord: u32, wLane: u32) -> i32 {
    let q = q_i8(qWord, qLane);
    let code = (wWord >> (wLane * 2u)) & 0x3u;
    if (code == 0u) { return acc - q; }
    if (code == 2u) { return acc + q; }
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

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let qn = params.qWords;
    for (var i = tid; i < qn; i += 64u) {
        qLocal[i] = qPacked[i];
    }
    workgroupBarrier();

    let total = params.qDim + params.kvDim + params.kvDim;
    let o = wg_id.x * 64u + tid;
    if (o >= total) { return; }

    let fullWords = params.inputSize / 16u;
    let actScale = inputScales[0];
    var sum: i32 = 0;
    var wBase: u32 = 0u;
    var wScale: f32 = 0.0;

    if (o < params.qDim) {
        wBase = o * params.rowWords;
        wScale = params.qScale;
        for (var wIdx = 0u; wIdx < fullWords; wIdx++) {
            let qOff = wIdx * 4u;
            sum = dot16(sum, qLocal[qOff], qLocal[qOff+1u], qLocal[qOff+2u], qLocal[qOff+3u], qWeights[wBase + wIdx]);
        }
        for (var i = fullWords * 16u; i < params.inputSize; i++) {
            let q = q_i8(qLocal[i / 4u], i % 4u);
            let code = (qWeights[wBase + (i / 16u)] >> ((i % 16u) * 2u)) & 0x3u;
            if (code == 0u) { sum -= q; } else if (code == 2u) { sum += q; }
        }
        qOut[o] = f32(sum) * wScale * actScale;
    } else if (o < params.qDim + params.kvDim) {
        let row = o - params.qDim;
        wBase = row * params.rowWords;
        wScale = params.kScale;
        for (var wIdx = 0u; wIdx < fullWords; wIdx++) {
            let qOff = wIdx * 4u;
            sum = dot16(sum, qLocal[qOff], qLocal[qOff+1u], qLocal[qOff+2u], qLocal[qOff+3u], kWeights[wBase + wIdx]);
        }
        for (var i = fullWords * 16u; i < params.inputSize; i++) {
            let q = q_i8(qLocal[i / 4u], i % 4u);
            let code = (kWeights[wBase + (i / 16u)] >> ((i % 16u) * 2u)) & 0x3u;
            if (code == 0u) { sum -= q; } else if (code == 2u) { sum += q; }
        }
        kOut[row] = f32(sum) * wScale * actScale;
    } else {
        let row = o - params.qDim - params.kvDim;
        wBase = row * params.rowWords;
        wScale = params.vScale;
        for (var wIdx = 0u; wIdx < fullWords; wIdx++) {
            let qOff = wIdx * 4u;
            sum = dot16(sum, qLocal[qOff], qLocal[qOff+1u], qLocal[qOff+2u], qLocal[qOff+3u], vWeights[wBase + wIdx]);
        }
        for (var i = fullWords * 16u; i < params.inputSize; i++) {
            let q = q_i8(qLocal[i / 4u], i % 4u);
            let code = (vWeights[wBase + (i / 16u)] >> ((i % 16u) * 2u)) & 0x3u;
            if (code == 0u) { sum -= q; } else if (code == 2u) { sum += q; }
        }
        vOut[row] = f32(sum) * wScale * actScale;
    }
}
`
