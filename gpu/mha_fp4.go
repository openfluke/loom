package gpu

import (
	"fmt"
	"math"

	"github.com/openfluke/webgpu/wgpu"
)

// FP4MHASpec defines configuration for FP4 Multi-Head Attention layer
type FP4MHASpec struct {
	DModel       int
	NumHeads     int
	NumKVHeads   int
	SeqLen       int
	HeadDim      int
	QData        []uint8
	QScales      []float32
	KData        []uint8
	KScales      []float32
	VData        []uint8
	VScales      []float32
	OData        []uint8
	OScales      []float32
	QBias        []float32
	KBias        []float32
	VBias        []float32
	OBias        []float32
	RoPEFreqBase float32
	MaxSeq       int
}

// FP4MHALayer holds GPU resources for FP4 Multi-Head Attention
type FP4MHALayer struct {
	Spec      FP4MHASpec
	BatchSize int

	pipelineQKV   *wgpu.ComputePipeline
	bindGroupQKV  *wgpu.BindGroup
	pipelineAttn  *wgpu.ComputePipeline
	bindGroupAttn *wgpu.BindGroup
	pipelineOut   *wgpu.ComputePipeline
	bindGroupOut  *wgpu.BindGroup

	InputBuffer   *wgpu.Buffer
	OutputBuffer  *wgpu.Buffer
	StagingBuffer *wgpu.Buffer

	// Weight Bundles
	QKVPackedBuf *wgpu.Buffer
	QKVScaleBuf  *wgpu.Buffer
	OPackedBuf   *wgpu.Buffer
	OScaleBuf    *wgpu.Buffer
	QKVBiasBuf   *wgpu.Buffer
	OBiasBuf     *wgpu.Buffer

	// Intermediates
	QKVOutBuf    *wgpu.Buffer
	AttnBuffer   *wgpu.Buffer
	ParamsBuffer *wgpu.Buffer

	// Cache
	KVCacheBuffer *wgpu.Buffer
	CachePos      int

	InputAliased bool
}

func (l *FP4MHALayer) GetInputBuffer() *wgpu.Buffer         { return l.InputBuffer }
func (l *FP4MHALayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *FP4MHALayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *FP4MHALayer) GetInputGradientBuffer() *wgpu.Buffer { return nil }

func (l *FP4MHALayer) SetInputBuffer(buf *wgpu.Buffer) {
	l.InputBuffer = buf
	l.InputAliased = true
}

func nz(n int) uint64 {
	if n <= 0 {
		return 4
	}
	return uint64(n)
}

func (l *FP4MHALayer) AllocateBuffers(ctx *Context, labelPrefix string) error {
	var err error
	batchSize := l.BatchSize
	if batchSize < 1 {
		batchSize = 1
	}
	seqLen := l.Spec.SeqLen
	if seqLen < 1 {
		seqLen = 1
	}
	seqDim := seqLen * l.Spec.DModel * batchSize
	numKVHeads := l.Spec.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = l.Spec.NumHeads
	}
	dKV := numKVHeads * l.Spec.HeadDim
	dModel := l.Spec.DModel

	if !l.InputAliased {
		l.InputBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
			Label: labelPrefix + "_In",
			Size:  nz(seqDim * 4),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
		})
		if err != nil {
			return err
		}
	}

	l.OutputBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Out",
		Size:  nz(seqDim * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	l.QKVOutBuf, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_QOut",
		Size:  nz(seqDim * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	l.AttnBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Attn",
		Size:  nz(seqDim * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	l.QKVPackedBuf, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_QKVPacked",
		Size:  nz(packBytesToU32Size(len(l.Spec.QData)+len(l.Spec.KData)+len(l.Spec.VData)) * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}

	l.QKVScaleBuf, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_QKVScale",
		Size:  nz((len(l.Spec.QScales) + len(l.Spec.KScales) + len(l.Spec.VScales)) * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}

	l.OPackedBuf, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_OPacked",
		Size:  nz(packBytesToU32Size(len(l.Spec.OData)) * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}

	l.OScaleBuf, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_OScale",
		Size:  nz(len(l.Spec.OScales) * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}

	l.QKVBiasBuf, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_QKVBias",
		Size:  nz((l.Spec.DModel + 2*dKV) * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}

	l.OBiasBuf, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_OBias",
		Size:  nz(dModel * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}

	l.StagingBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Staging",
		Size:  nz(seqDim * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}

	l.ParamsBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Params",
		Size:  128,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}

	maxSeq := l.Spec.MaxSeq
	if maxSeq <= 0 {
		maxSeq = seqLen
	}
	l.KVCacheBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_KVCache",
		Size:  nz(2 * maxSeq * dKV * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	return err
}

func packBytesToU32Size(byteLen int) int { return (byteLen + 3) / 4 }

func (l *FP4MHALayer) UploadWeights(ctx *Context) {
	qData, kData, vData := l.Spec.QData, l.Spec.KData, l.Spec.VData
	combinedData := make([]uint8, len(qData)+len(kData)+len(vData))
	copy(combinedData, qData)
	copy(combinedData[len(qData):], kData)
	copy(combinedData[len(qData)+len(kData):], vData)
	if len(combinedData) > 0 {
		ctx.Queue.WriteBuffer(l.QKVPackedBuf, 0, wgpu.ToBytes(packBytesToU32(combinedData)))
	}

	qScale, kScale, vScale := l.Spec.QScales, l.Spec.KScales, l.Spec.VScales
	combinedScales := append(append(append([]float32{}, qScale...), kScale...), vScale...)
	if len(combinedScales) > 0 {
		ctx.Queue.WriteBuffer(l.QKVScaleBuf, 0, wgpu.ToBytes(combinedScales))
	}

	dModel := l.Spec.DModel
	numKVHeads := l.Spec.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = l.Spec.NumHeads
	}
	dKV := numKVHeads * l.Spec.HeadDim

	qBias, kBias, vBias := l.Spec.QBias, l.Spec.KBias, l.Spec.VBias
	if len(qBias) == 0 {
		qBias = make([]float32, dModel)
	}
	if len(kBias) == 0 {
		kBias = make([]float32, dKV)
	}
	if len(vBias) == 0 {
		vBias = make([]float32, dKV)
	}
	combinedBias := append(append(append([]float32{}, qBias...), kBias...), vBias...)
	ctx.Queue.WriteBuffer(l.QKVBiasBuf, 0, wgpu.ToBytes(combinedBias))

	if len(l.Spec.OData) > 0 {
		ctx.Queue.WriteBuffer(l.OPackedBuf, 0, wgpu.ToBytes(packBytesToU32(l.Spec.OData)))
	}
	if len(l.Spec.OScales) > 0 {
		ctx.Queue.WriteBuffer(l.OScaleBuf, 0, wgpu.ToBytes(l.Spec.OScales))
	}
	oBias := l.Spec.OBias
	if len(oBias) == 0 {
		oBias = make([]float32, dModel)
	}
	ctx.Queue.WriteBuffer(l.OBiasBuf, 0, wgpu.ToBytes(oBias))
}

func (l *FP4MHALayer) Cleanup() {
	if l.InputBuffer != nil && !l.InputAliased {
		l.InputBuffer.Destroy()
	}
	for _, b := range []*wgpu.Buffer{l.OutputBuffer, l.StagingBuffer, l.QKVOutBuf, l.AttnBuffer, l.QKVPackedBuf, l.QKVScaleBuf, l.OPackedBuf, l.OScaleBuf, l.QKVBiasBuf, l.OBiasBuf, l.ParamsBuffer, l.KVCacheBuffer} {
		if b != nil {
			b.Destroy()
		}
	}
	for _, p := range []*wgpu.ComputePipeline{l.pipelineQKV, l.pipelineAttn, l.pipelineOut} {
		if p != nil {
			p.Release()
		}
	}
	for _, bg := range []*wgpu.BindGroup{l.bindGroupQKV, l.bindGroupAttn, l.bindGroupOut} {
		if bg != nil {
			bg.Release()
		}
	}
}

func (l *FP4MHALayer) AllocateBackwardBuffers(ctx *Context, label string) error { return nil }
func (l *FP4MHALayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	return nil, nil, nil
}
func (l *FP4MHALayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	return nil, nil, nil, nil
}
func (l *FP4MHALayer) ZeroGradients(ctx *Context)                {}
func (l *FP4MHALayer) DispatchBackward(enc *wgpu.CommandEncoder) {}
func (l *FP4MHALayer) CreateBackwardBindGroup(ctx *Context, label string, dOut *wgpu.Buffer) error {
	return nil
}
func (l *FP4MHALayer) CompileBackward(ctx *Context, label string) error { return nil }

func (l *FP4MHALayer) GenerateQKVShader() string {
	numKVHeads := l.Spec.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = l.Spec.NumHeads
	}
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read> w_pack : array<u32>;
		@group(0) @binding(2) var<storage, read> w_scale : array<f32>;
		@group(0) @binding(3) var<storage, read> biases : array<f32>;
		@group(0) @binding(4) var<storage, read_write> q_out : array<f32>;
		@group(0) @binding(5) var<storage, read_write> kv_cache : array<f32>;

		struct LayerParams {
			input_len: u32, cache_pos: u32,
			q_w_offset: u32, k_w_offset: u32, v_w_offset: u32,
			q_s_offset: u32, k_s_offset: u32, v_s_offset: u32,
			q_b_offset: u32, k_b_offset: u32, v_b_offset: u32,
			q_res_offset: u32, k_res_offset: u32, v_res_offset: u32,
			k_cache_offset: u32, v_cache_offset: u32,
		};
		@group(0) @binding(6) var<uniform> params : LayerParams;

		const SEQ_LEN: u32 = %du;
		const D_MODEL: u32 = %du;
		const NUM_HEADS: u32 = %du;
		const NUM_KV_HEADS: u32 = %du;
		const HEAD_DIM: u32 = %du;
		const ROPE_BASE: f32 = %f;
		const MICRO_SCALE_GROUP : u32 = 16u;
		const NUM_ROW_GROUPS : u32 = D_MODEL / MICRO_SCALE_GROUP;

		fn unpack_fp4(n: u32) -> f32 {
			let s = (n >> 3u) & 1u; var v: f32 = 0.0;
			switch(n & 7u) {
				case 1u: { v = 0.5; } case 2u: { v = 1.0; } case 3u: { v = 1.5; }
				case 4u: { v = 2.0; } case 5u: { v = 3.0; } case 6u: { v = 4.0; }
				case 7u: { v = 6.0; } default: { v = 0.0; }
			}
			return select(v, -v, s == 1u);
		}

		fn get_nibble(w_off: u32, k: u32) -> u32 {
			let t = w_off + k;
			return (w_pack[t >> 3u] >> ((t & 7u) * 4u)) & 0xFu;
		}

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			if (gid.x >= params.input_len * D_MODEL) { return; }
			let seq = gid.x / D_MODEL; let d = gid.x %% D_MODEL;
			let D_KV = NUM_KV_HEADS * HEAD_DIM; let in_off = seq * D_MODEL;

			// 1. Q projection
			var q_val = biases[params.q_b_offset + d];
			for (var g: u32 = 0u; g < NUM_ROW_GROUPS; g++) {
				var acc: f32 = 0.0;
				for (var m: u32 = 0u; m < MICRO_SCALE_GROUP; m++) {
					let r = g * MICRO_SCALE_GROUP + m;
					acc += unpack_fp4(get_nibble(params.q_w_offset, r * D_MODEL + d)) * input[in_off + r];
				}
				q_val += acc * w_scale[params.q_s_offset + d * NUM_ROW_GROUPS + g];
			}

			// 2. K/V projection (if within D_KV)
			if (d < D_KV) {
				var k_val = biases[params.k_b_offset + d];
				var v_val = biases[params.v_b_offset + d];
				for (var g: u32 = 0u; g < NUM_ROW_GROUPS; g++) {
					var k_acc: f32 = 0.0; var v_acc: f32 = 0.0;
					for (var m: u32 = 0u; m < MICRO_SCALE_GROUP; m++) {
						let r = g * MICRO_SCALE_GROUP + m;
						k_acc += unpack_fp4(get_nibble(params.k_w_offset, r*D_KV + d)) * input[in_off + r];
						v_acc += unpack_fp4(get_nibble(params.v_w_offset, r*D_KV + d)) * input[in_off + r];
					}
					k_val += k_acc * w_scale[params.k_s_offset + d * NUM_ROW_GROUPS + g];
					v_val += v_acc * w_scale[params.v_s_offset + d * NUM_ROW_GROUPS + g];
				}

				// RoPE for Key
				let head_d = d %% HEAD_DIM; let half = HEAD_DIM / 2u;
				let theta = pow(ROPE_BASE, -2.0 * f32(head_d %% half) / f32(HEAD_DIM));
				let angle = f32(params.cache_pos + seq) * theta;
				let c = cos(angle); let s = sin(angle);

				var k_pair: f32 = 0.0;
				let d_pair = select(d + half, d - half, head_d >= half);
				var k_p_val = biases[params.k_b_offset + d_pair];
				for (var g: u32 = 0u; g < NUM_ROW_GROUPS; g++) {
					var acc: f32 = 0.0;
					for (var m: u32 = 0u; m < MICRO_SCALE_GROUP; m++) {
						let r = g * MICRO_SCALE_GROUP + m;
						acc += unpack_fp4(get_nibble(params.k_w_offset, r*D_KV + d_pair)) * input[in_off + r];
					}
					k_p_val += acc * w_scale[params.k_s_offset + d_pair * NUM_ROW_GROUPS + g];
				}
				
				let k_rot = select(k_val * c - k_p_val * s, k_val * c + k_p_val * s, head_d >= half);
				kv_cache[params.k_cache_offset + (params.cache_pos + seq)*D_KV + d] = k_rot;
				kv_cache[params.v_cache_offset + (params.cache_pos + seq)*D_KV + d] = v_val;
			}

			// RoPE for Query
			let head_d_q = d %% HEAD_DIM; let half_q = HEAD_DIM / 2u;
			let theta_q = pow(ROPE_BASE, -2.0 * f32(head_d_q %% half_q) / f32(HEAD_DIM));
			let angle_q = f32(params.cache_pos + seq) * theta_q;
			let cq = cos(angle_q); let sq = sin(angle_q);
			
			let d_p_q = select(d + half_q, d - half_q, head_d_q >= half_q);
			var q_p_val = biases[params.q_b_offset + d_p_q];
			for (var g: u32 = 0u; g < NUM_ROW_GROUPS; g++) {
				var acc: f32 = 0.0;
				for (var m: u32 = 0u; m < MICRO_SCALE_GROUP; m++) {
					let r = g * MICRO_SCALE_GROUP + m;
					acc += unpack_fp4(get_nibble(params.q_w_offset, r * D_MODEL + d_p_q)) * input[in_off + r];
				}
				q_p_val += acc * w_scale[params.q_s_offset + d_p_q * NUM_ROW_GROUPS + g];
			}
			
			q_out[gid.x] = select(q_val * cq - q_p_val * sq, q_val * cq + q_p_val * sq, head_d_q >= half_q);
		}
	`, l.Spec.SeqLen, l.Spec.DModel, l.Spec.NumHeads, numKVHeads, l.Spec.HeadDim, l.Spec.RoPEFreqBase)
}

func (l *FP4MHALayer) GenerateAttnShader() string {
	headDim := l.Spec.DModel / l.Spec.NumHeads
	scale := 1.0 / float32(math.Sqrt(float64(headDim)))
	numKVHeads := l.Spec.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = l.Spec.NumHeads
	}

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> q_in : array<f32>;
		@group(0) @binding(1) var<storage, read> kv_cache : array<f32>;
		@group(0) @binding(2) var<storage, read_write> attn_out : array<f32>;

		struct LayerParams {
			input_len: u32, cache_pos: u32,
			q_w_offset: u32, k_w_offset: u32, v_w_offset: u32,
			q_s_offset: u32, k_s_offset: u32, v_s_offset: u32,
			q_b_offset: u32, k_b_offset: u32, v_b_offset: u32,
			q_res_offset: u32, k_res_offset: u32, v_res_offset: u32,
			k_cache_offset: u32, v_cache_offset: u32,
		};
		@group(0) @binding(3) var<uniform> params : LayerParams;

		const SEQ_LEN: u32 = %du;
		const D_MODEL: u32 = %du;
		const NUM_HEADS: u32 = %du;
		const NUM_KV_HEADS: u32 = %du;
		const HEAD_DIM: u32 = %du;
		const SCALE: f32 = %f;

		var<workgroup> shared_q : array<f32, 128>; // Covers HEAD_DIM up to 128

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
			if (gid.x >= params.input_len * D_MODEL) { return; }
			let seq_i = gid.x / D_MODEL; let d = gid.x %% D_MODEL;
			let head = d / HEAD_DIM; let head_d = d %% HEAD_DIM;
			let heads_per_kv = NUM_HEADS / NUM_KV_HEADS; let kv_head = head / heads_per_kv;
			let total_len = params.input_len + params.cache_pos;
			let D_KV = NUM_KV_HEADS * HEAD_DIM;

			// Shared Memory Optimization: Load Query head once
            // (Note: with 256 threads, many threads share the same head)
			if (head_d < HEAD_DIM) {
				shared_q[head_d] = q_in[seq_i*D_MODEL + head*HEAD_DIM + head_d];
			}
			workgroupBarrier();

			var max_s: f32 = -1e10;
			// 1. Max pass (Stable softmax)
			for (var j: u32 = 0u; j < total_len; j++) {
				if (j > (params.cache_pos + seq_i)) { continue; }
				var score: f32 = 0.0;
				for (var hd: u32 = 0u; hd < HEAD_DIM; hd++) {
                    let qi = head*HEAD_DIM + hd;
					score += q_in[seq_i*D_MODEL + qi] * kv_cache[params.k_cache_offset + j*D_KV + kv_head*HEAD_DIM + hd];
				}
				max_s = max(max_s, score * SCALE);
			}

			// 2. Sum and V pass
			var sum_v: f32 = 0.0; var exp_sum: f32 = 0.0;
			for (var j: u32 = 0u; j < total_len; j++) {
				if (j > (params.cache_pos + seq_i)) { continue; }
				var score: f32 = 0.0;
				for (var hd: u32 = 0u; hd < HEAD_DIM; hd++) {
                    let qi = head*HEAD_DIM + hd;
					score += q_in[seq_i*D_MODEL + qi] * kv_cache[params.k_cache_offset + j*D_KV + kv_head*HEAD_DIM + hd];
				}
				let e = exp(clamp(score * SCALE - max_s, -80.0, 0.0));
				exp_sum += e;
				sum_v += e * kv_cache[params.v_cache_offset + j*D_KV + kv_head*HEAD_DIM + head_d];
			}
			attn_out[gid.x] = select(0.0, sum_v / (exp_sum + 1e-9), exp_sum > 0.0);
		}
	`, l.Spec.SeqLen, l.Spec.DModel, l.Spec.NumHeads, numKVHeads, headDim, scale)
}

func (l *FP4MHALayer) GenerateOutShader() string {
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> attn_out : array<f32>;
		@group(0) @binding(1) var<storage, read> w_p : array<u32>;
		@group(0) @binding(2) var<storage, read> w_s : array<f32>;
		@group(0) @binding(3) var<storage, read> biases : array<f32>;
		@group(0) @binding(4) var<storage, read_write> output : array<f32>;

		struct LayerParams {
			input_len: u32, cache_pos: u32,
		};
		@group(0) @binding(5) var<uniform> params : LayerParams;

		const SEQ_LEN: u32 = %du;
		const D_MODEL: u32 = %du;
		const MICRO_SCALE_GROUP : u32 = 16u;
		const NUM_ROW_GROUPS : u32 = D_MODEL / MICRO_SCALE_GROUP;

		fn unpack_fp4(n: u32) -> f32 {
			let s = (n >> 3u) & 1u; var v: f32 = 0.0;
			switch(n & 7u) {
				case 1u: { v = 0.5; } case 2u: { v = 1.0; } case 3u: { v = 1.5; }
				case 4u: { v = 2.0; } case 5u: { v = 3.0; } case 6u: { v = 4.0; }
				case 7u: { v = 6.0; } default: { v = 0.0; }
			}
			return select(v, -v, s == 1u);
		}

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			if (gid.x >= params.input_len * D_MODEL) { return; }
			let s = gid.x / D_MODEL; let d = gid.x %% D_MODEL;
			var sum = biases[d];
			for (var g: u32 = 0u; g < NUM_ROW_GROUPS; g++) {
				var acc: f32 = 0.0;
				for (var m: u32 = 0u; m < MICRO_SCALE_GROUP; m++) {
					let r = g * MICRO_SCALE_GROUP + m;
                    let t = r * D_MODEL + d;
					acc += unpack_fp4((w_p[t >> 3u] >> ((t & 7u) * 4u)) & 0xFu) * attn_out[s*D_MODEL + r];
				}
				sum += acc * w_s[d * NUM_ROW_GROUPS + g];
			}
			output[gid.x] = sum;
		}
	`, l.Spec.SeqLen, l.Spec.DModel)
}

func (l *FP4MHALayer) Compile(ctx *Context, labelPrefix string) error {
	var err error
	m1, _ := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: labelPrefix + "_QKV", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateQKVShader()}})
	bgl1, _ := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{Entries: []wgpu.BindGroupLayoutEntry{{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, {Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, {Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, {Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, {Binding: 4, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}}, {Binding: 5, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}}, {Binding: 6, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}}}})
	pl1, _ := ctx.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{BindGroupLayouts: []*wgpu.BindGroupLayout{bgl1}})
	l.pipelineQKV, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Layout: pl1, Compute: wgpu.ProgrammableStageDescriptor{Module: m1, EntryPoint: "main"}})
	if err != nil {
		return err
	}

	m2, _ := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: labelPrefix + "_Attn", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateAttnShader()}})
	l.pipelineAttn, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Layout: nil, Compute: wgpu.ProgrammableStageDescriptor{Module: m2, EntryPoint: "main"}})
	if err != nil {
		return err
	}

	m3, _ := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: labelPrefix + "_Out", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateOutShader()}})
	l.pipelineOut, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Layout: nil, Compute: wgpu.ProgrammableStageDescriptor{Module: m3, EntryPoint: "main"}})
	return err
}

func (l *FP4MHALayer) CreateBindGroup(ctx *Context, labelPrefix string) error {
	l.bindGroupQKV, _ = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: l.pipelineQKV.GetBindGroupLayout(0), Entries: []wgpu.BindGroupEntry{{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()}, {Binding: 1, Buffer: l.QKVPackedBuf, Size: l.QKVPackedBuf.GetSize()}, {Binding: 2, Buffer: l.QKVScaleBuf, Size: l.QKVScaleBuf.GetSize()}, {Binding: 3, Buffer: l.QKVBiasBuf, Size: l.QKVBiasBuf.GetSize()}, {Binding: 4, Buffer: l.QKVOutBuf, Size: l.QKVOutBuf.GetSize()}, {Binding: 5, Buffer: l.KVCacheBuffer, Size: l.KVCacheBuffer.GetSize()}, {Binding: 6, Buffer: l.ParamsBuffer, Size: l.ParamsBuffer.GetSize()}}})
	l.bindGroupAttn, _ = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: l.pipelineAttn.GetBindGroupLayout(0), Entries: []wgpu.BindGroupEntry{{Binding: 0, Buffer: l.QKVOutBuf, Size: l.QKVOutBuf.GetSize()}, {Binding: 1, Buffer: l.KVCacheBuffer, Size: l.KVCacheBuffer.GetSize()}, {Binding: 2, Buffer: l.AttnBuffer, Size: l.AttnBuffer.GetSize()}, {Binding: 3, Buffer: l.ParamsBuffer, Size: l.ParamsBuffer.GetSize()}}})
	l.bindGroupOut, _ = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: l.pipelineOut.GetBindGroupLayout(0), Entries: []wgpu.BindGroupEntry{{Binding: 0, Buffer: l.AttnBuffer, Size: l.AttnBuffer.GetSize()}, {Binding: 1, Buffer: l.OPackedBuf, Size: l.OPackedBuf.GetSize()}, {Binding: 2, Buffer: l.OScaleBuf, Size: l.OScaleBuf.GetSize()}, {Binding: 3, Buffer: l.OBiasBuf, Size: l.OBiasBuf.GetSize()}, {Binding: 4, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()}, {Binding: 5, Buffer: l.ParamsBuffer, Size: l.ParamsBuffer.GetSize()}}})
	return nil
}

func (l *FP4MHALayer) UpdateParams(ctx *Context, inputLen int, cachePos int) {
	dModel := l.Spec.DModel
	numKVHeads := l.Spec.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = l.Spec.NumHeads
	}
	dKV := numKVHeads * l.Spec.HeadDim
	maxSeq := l.Spec.MaxSeq
	if maxSeq <= 0 {
		maxSeq = l.Spec.SeqLen
	}
	if maxSeq <= 0 {
		maxSeq = 1
	}
	l.BatchSize = inputLen
	paramsData := wgpu.ToBytes([]uint32{uint32(inputLen), uint32(cachePos), 0, uint32(len(l.Spec.QData) * 2), uint32((len(l.Spec.QData) + len(l.Spec.KData)) * 2), 0, uint32(len(l.Spec.QScales)), uint32(len(l.Spec.QScales) + len(l.Spec.KScales)), 0, uint32(dModel), uint32(dModel + dKV), 0, uint32(dModel * l.Spec.SeqLen), uint32((dModel + dKV) * l.Spec.SeqLen), 0, uint32(dKV * maxSeq)})
	ctx.Queue.WriteBuffer(l.ParamsBuffer, 0, paramsData)
}

func (l *FP4MHALayer) Dispatch(pass *wgpu.ComputePassEncoder) {}
func (l *FP4MHALayer) DispatchFull(enc *wgpu.CommandEncoder) {
	b := l.BatchSize
	if b < 1 {
		b = 1
	}
	p1 := enc.BeginComputePass(nil)
	p1.SetPipeline(l.pipelineQKV)
	p1.SetBindGroup(0, l.bindGroupQKV, nil)
	p1.DispatchWorkgroups(uint32((b*l.Spec.DModel+255)/256), 1, 1)
	p1.End()

	p2 := enc.BeginComputePass(nil)
	p2.SetPipeline(l.pipelineAttn)
	p2.SetBindGroup(0, l.bindGroupAttn, nil)
	p2.DispatchWorkgroups(uint32((b*l.Spec.DModel+255)/256), 1, 1)
	p2.End()

	p3 := enc.BeginComputePass(nil)
	p3.SetPipeline(l.pipelineOut)
	p3.SetBindGroup(0, l.bindGroupOut, nil)
	p3.DispatchWorkgroups(uint32((b*l.Spec.DModel+255)/256), 1, 1)
	p3.End()
}
