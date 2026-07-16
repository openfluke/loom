package poly

// Decode-shaped Q4 GEMV: one shared-memory load of x per workgroup (batch=1).
// From poc/entity_wgpu_fast — much faster than ShaderTiledDenseQ4 on GTX 1650 SUPER decode.
const ShaderDecodeQ4GEMV = `
struct Params {
    inputSize: u32,
    outputSize: u32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> xin: array<f32, 2048>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let inN = params.inputSize;
    for (var i = tid; i < inN; i += 64u) {
        xin[i] = input[i];
    }
    workgroupBarrier();

    let o = wg_id.x * 64u + tid;
    if (o >= params.outputSize) { return; }

    var sum: f32 = 0.0;
    let base_w = o * inN;
    let nBlocks = inN / 32u;
    for (var b = 0u; b < nBlocks; b++) {
        let scale = scales[(base_w / 32u) + b];
        let wBase = (base_w / 8u) + b * 4u;
        let iBase = b * 32u;
        for (var w = 0u; w < 4u; w++) {
            let packed = weights[wBase + w];
            let i0 = iBase + w * 8u;
            var acc: f32 = 0.0;
            for (var n = 0u; n < 8u; n++) {
                var q = i32((packed >> (n * 4u)) & 0xFu);
                if (q > 7) { q -= 16; }
                acc += xin[i0 + n] * f32(q);
            }
            sum += acc * scale;
        }
    }
    output[o] = sum;
}
`

// Fused QKV Q4 decode GEMV — one shared-memory x load → [Q|K|V] packed out.
const ShaderDecodeQ4GEMV_QKV = `
struct Params {
    inputSize: u32,
    qDim: u32,
    kvDim: u32,
    _pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> qScales: array<f32>;
@group(0) @binding(3) var<storage, read> qWeights: array<u32>;
@group(0) @binding(4) var<storage, read> kScales: array<f32>;
@group(0) @binding(5) var<storage, read> kWeights: array<u32>;
@group(0) @binding(6) var<storage, read> vScales: array<f32>;
@group(0) @binding(7) var<storage, read> vWeights: array<u32>;
@group(0) @binding(8) var<storage, read_write> qOut: array<f32>;
@group(0) @binding(9) var<storage, read_write> kOut: array<f32>;
@group(0) @binding(10) var<storage, read_write> vOut: array<f32>;

var<workgroup> xin: array<f32, 2048>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let inN = params.inputSize;
    for (var i = tid; i < inN; i += 64u) {
        xin[i] = input[i];
    }
    workgroupBarrier();

    let total = params.qDim + params.kvDim + params.kvDim;
    let o = wg_id.x * 64u + tid;
    if (o >= total) { return; }

    var row: u32;
    var sum: f32 = 0.0;
    let nBlocks = inN / 32u;
    if (o < params.qDim) {
        row = o;
        let base_w = row * inN;
        for (var b = 0u; b < nBlocks; b++) {
            let scale = qScales[(base_w / 32u) + b];
            let wBase = (base_w / 8u) + b * 4u;
            let iBase = b * 32u;
            for (var w = 0u; w < 4u; w++) {
                let packed = qWeights[wBase + w];
                let i0 = iBase + w * 8u;
                var acc: f32 = 0.0;
                for (var n = 0u; n < 8u; n++) {
                    var q = i32((packed >> (n * 4u)) & 0xFu);
                    if (q > 7) { q -= 16; }
                    acc += xin[i0 + n] * f32(q);
                }
                sum += acc * scale;
            }
        }
        qOut[o] = sum;
    } else if (o < params.qDim + params.kvDim) {
        row = o - params.qDim;
        let base_w = row * inN;
        for (var b = 0u; b < nBlocks; b++) {
            let scale = kScales[(base_w / 32u) + b];
            let wBase = (base_w / 8u) + b * 4u;
            let iBase = b * 32u;
            for (var w = 0u; w < 4u; w++) {
                let packed = kWeights[wBase + w];
                let i0 = iBase + w * 8u;
                var acc: f32 = 0.0;
                for (var n = 0u; n < 8u; n++) {
                    var q = i32((packed >> (n * 4u)) & 0xFu);
                    if (q > 7) { q -= 16; }
                    acc += xin[i0 + n] * f32(q);
                }
                sum += acc * scale;
            }
        }
        kOut[row] = sum;
    } else {
        row = o - params.qDim - params.kvDim;
        let base_w = row * inN;
        for (var b = 0u; b < nBlocks; b++) {
            let scale = vScales[(base_w / 32u) + b];
            let wBase = (base_w / 8u) + b * 4u;
            let iBase = b * 32u;
            for (var w = 0u; w < 4u; w++) {
                let packed = vWeights[wBase + w];
                let i0 = iBase + w * 8u;
                var acc: f32 = 0.0;
                for (var n = 0u; n < 8u; n++) {
                    var q = i32((packed >> (n * 4u)) & 0xFu);
                    if (q > 7) { q -= 16; }
                    acc += xin[i0 + n] * f32(q);
                }
                sum += acc * scale;
            }
        }
        vOut[row] = sum;
    }
}
`
