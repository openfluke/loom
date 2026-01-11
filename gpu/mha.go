package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// MHASpec defines configuration for Multi-Head Attention layer
type MHASpec struct {
	DModel       int       // Model dimension (embedding size)
	NumHeads     int       // Number of attention heads
	NumKVHeads   int       // Number of key/value heads (for GQA)
	SeqLen       int       // Sequence length
	HeadDim      int       // Dimension per head (DModel / NumHeads)
	QWeights     []float32 // Query projection [DModel * DModel]
	KWeights     []float32 // Key projection [DModel * DModel]
	VWeights     []float32 // Value projection [DModel * DModel]
	OWeights     []float32 // Output projection [DModel * DModel]
	QBias        []float32 // [DModel]
	KBias        []float32 // [DModel]
	VBias        []float32 // [DModel]
	OBias        []float32 // [DModel]
	RoPEFreqBase float32   // Base frequency for RoPE (default 10000.0)
}

// MHALayer holds GPU resources for Multi-Head Attention
type MHALayer struct {
	Spec MHASpec

	// Q/K/V projection pipeline (combined)
	pipelineQKV  *wgpu.ComputePipeline
	bindGroupQKV *wgpu.BindGroup

	// Attention scores pipeline
	pipelineAttn  *wgpu.ComputePipeline
	bindGroupAttn *wgpu.BindGroup

	// Output projection pipeline
	pipelineOut  *wgpu.ComputePipeline
	bindGroupOut *wgpu.BindGroup

	// Buffers
	InputBuffer   *wgpu.Buffer
	OutputBuffer  *wgpu.Buffer
	StagingBuffer *wgpu.Buffer

	CombinedWeightsQKV *wgpu.Buffer
	CombinedBiasesQKV  *wgpu.Buffer
	OWeightBuffer      *wgpu.Buffer
	OBiasBuffer        *wgpu.Buffer

	QBuffer    *wgpu.Buffer // Projected queries
	KBuffer    *wgpu.Buffer // Projected keys
	VBuffer    *wgpu.Buffer // Projected values
	AttnBuffer *wgpu.Buffer // Attention output

	InputGradientBuffer *wgpu.Buffer
	bwPipeline          *wgpu.ComputePipeline
	bwBindGroup         *wgpu.BindGroup
}

func (l *MHALayer) GetInputBuffer() *wgpu.Buffer         { return l.InputBuffer }
func (l *MHALayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *MHALayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *MHALayer) GetInputGradientBuffer() *wgpu.Buffer { return l.InputGradientBuffer }

func (l *MHALayer) AllocateBuffers(ctx *Context, labelPrefix string) error {
	var err error
	seqDim := l.Spec.SeqLen * l.Spec.DModel
	dimSq := l.Spec.DModel * l.Spec.DModel

	// Input/Output
	l.InputBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_In",
		Size:  uint64(seqDim * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	l.OutputBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Out",
		Size:  uint64(seqDim * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Q/K/V intermediate buffers
	l.QBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Q",
		Size:  uint64(seqDim * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}
	l.KBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_K",
		Size:  uint64(seqDim * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}
	l.VBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_V",
		Size:  uint64(seqDim * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}
	l.AttnBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Attn",
		Size:  uint64(seqDim * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Combined weights Q, K, V
	combine := func(w1, w2, w3 []float32, size int) []float32 {
		res := make([]float32, size*3)
		if len(w1) > 0 {
			copy(res[0:size], w1)
		}
		if len(w2) > 0 {
			copy(res[size:2*size], w2)
		}
		if len(w3) > 0 {
			copy(res[2*size:3*size], w3)
		}
		return res
	}

	qkvWeights := combine(l.Spec.QWeights, l.Spec.KWeights, l.Spec.VWeights, dimSq)
	l.CombinedWeightsQKV, err = NewFloatBuffer(qkvWeights, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}

	qkvBiases := combine(l.Spec.QBias, l.Spec.KBias, l.Spec.VBias, l.Spec.DModel)
	l.CombinedBiasesQKV, err = NewFloatBuffer(qkvBiases, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}

	// Output weights/biases remain separate
	allocW := func(data []float32, size int) (*wgpu.Buffer, error) {
		if len(data) > 0 {
			return NewFloatBuffer(data, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
		}
		return NewFloatBuffer(make([]float32, size), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	}
	l.OWeightBuffer, err = allocW(l.Spec.OWeights, dimSq)
	if err != nil {
		return err
	}

	allocB := func(data []float32, size int) (*wgpu.Buffer, error) {
		if len(data) > 0 {
			return NewFloatBuffer(data, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
		}
		return NewFloatBuffer(make([]float32, size), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	}
	l.OBiasBuffer, err = allocB(l.Spec.OBias, l.Spec.DModel)
	if err != nil {
		return err
	}

	l.StagingBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Staging",
		Size:  uint64(seqDim * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	return err
}

func (l *MHALayer) AllocateBackwardBuffers(ctx *Context, labelPrefix string) error {
	var err error
	l.InputGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_InGrad",
		Size:  uint64(l.Spec.SeqLen * l.Spec.DModel * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	return err
}

func (l *MHALayer) GenerateQKVShader() string {
	// Combined Q/K/V linear projections + RoPE
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read> w_qkv : array<f32>;
		@group(0) @binding(2) var<storage, read> b_qkv : array<f32>;
		@group(0) @binding(3) var<storage, read_write> q_out : array<f32>;
		@group(0) @binding(4) var<storage, read_write> k_out : array<f32>;
		@group(0) @binding(5) var<storage, read_write> v_out : array<f32>;

		const SEQ_LEN: u32 = %du;
		const D_MODEL: u32 = %du;
		const HEAD_DIM: u32 = %du;
		const ROPE_BASE: f32 = %f;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let total = SEQ_LEN * D_MODEL;
			if (idx >= total) { return; }

			let seq = idx / D_MODEL;
			let d = idx %% D_MODEL;
			let in_offset = seq * D_MODEL;

			let offset_k = D_MODEL * D_MODEL;
			let offset_v = 2u * D_MODEL * D_MODEL;
			let bias_offset = D_MODEL;

			// ---------------------------------------------------------
			// 1. Compute Raw Q, K, V for this position 'd'
			// ---------------------------------------------------------

			// V projection (No RoPE)
			var v_sum: f32 = b_qkv[2u*bias_offset + d];
			for (var j: u32 = 0u; j < D_MODEL; j++) {
				v_sum += input[in_offset + j] * w_qkv[offset_v + j * D_MODEL + d];
			}
			v_out[idx] = v_sum;

			// Q projection (Raw)
			var q_sum: f32 = b_qkv[d];
			for (var j: u32 = 0u; j < D_MODEL; j++) {
				q_sum += input[in_offset + j] * w_qkv[j * D_MODEL + d];
			}

			// K projection (Raw)
			var k_sum: f32 = b_qkv[bias_offset + d];
			for (var j: u32 = 0u; j < D_MODEL; j++) {
				k_sum += input[in_offset + j] * w_qkv[offset_k + j * D_MODEL + d];
			}

			// ---------------------------------------------------------
			// 2. Apply RoPE to Q and K (Rotate Half Strategy)
			// ---------------------------------------------------------
			
			let d_in_head = d %% HEAD_DIM;
			let half_dim = HEAD_DIM / 2u;
			var pair_d: u32;
			var theta_idx: u32;
			var sign: f32; // -1 for low half (x), +1 for high half (y)

			if (d_in_head < half_dim) {
				pair_d = d + half_dim;
				theta_idx = d_in_head;
				sign = -1.0; 
			} else {
				pair_d = d - half_dim;
				theta_idx = d_in_head - half_dim;
				sign = 1.0;
			}

			// Compute angle
			let theta = pow(ROPE_BASE, -2.0 * f32(theta_idx) / f32(HEAD_DIM));
			let angle = f32(seq) * theta;
			let c = cos(angle);
			let s = sin(angle);

			// We need the value of the PAIR to apply rotation
			var q_pair: f32 = b_qkv[pair_d];
			for (var j: u32 = 0u; j < D_MODEL; j++) {
				q_pair += input[in_offset + j] * w_qkv[j * D_MODEL + pair_d];
			}

			var k_pair: f32 = b_qkv[bias_offset + pair_d];
			for (var j: u32 = 0u; j < D_MODEL; j++) {
				k_pair += input[in_offset + j] * w_qkv[offset_k + j * D_MODEL + pair_d];
			}

			// Apply rotation
			// x' = x cos - y sin  (for low half, sign=-1: x cos + y*sign*sin => x cos - y sin)
			// y' = y cos + x sin  (for high half, sign=+1: y cos + x*sign*sin => y cos + x sin)
			// Wait, simple logic:
			if (d_in_head < half_dim) {
				q_out[idx] = q_sum * c - q_pair * s;
				k_out[idx] = k_sum * c - k_pair * s;
			} else {
				q_out[idx] = q_sum * c + q_pair * s;
				k_out[idx] = k_sum * c + k_pair * s;
			}
		}
	`, l.Spec.SeqLen, l.Spec.DModel, l.Spec.HeadDim, l.Spec.RoPEFreqBase)
}

func (l *MHALayer) GenerateAttnShader() string {
	// Simplified attention: for each position, attend to all positions
	headDim := l.Spec.DModel / l.Spec.NumHeads
	scale := 1.0 / float32(headDim)
	numKVHeads := l.Spec.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = l.Spec.NumHeads
	}

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> q : array<f32>;
		@group(0) @binding(1) var<storage, read> k : array<f32>;
		@group(0) @binding(2) var<storage, read> v : array<f32>;
		@group(0) @binding(3) var<storage, read_write> attn_out : array<f32>;

		const SEQ_LEN: u32 = %du;
		const D_MODEL: u32 = %du;
		const NUM_HEADS: u32 = %du;
		const NUM_KV_HEADS: u32 = %du;
		const HEAD_DIM: u32 = %du;
		const SCALE: f32 = %f;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let total = SEQ_LEN * D_MODEL;
			if (idx >= total) { return; }

			let seq_i = idx / D_MODEL;
			let d = idx %% D_MODEL;
			let head = d / HEAD_DIM;
			let head_d = d %% HEAD_DIM;
			
			// GQA: Map Query head to KV head
			let heads_per_kv = NUM_HEADS / NUM_KV_HEADS;
			let kv_head = head / heads_per_kv;

			// Compute attention weights and weighted sum of V
			var sum: f32 = 0.0;
			var max_score: f32 = -1e10;

			// First pass: find max for stability
			for (var seq_j: u32 = 0u; seq_j < SEQ_LEN; seq_j++) {
				if (seq_j > seq_i) { continue; } // Causal mask
				var score: f32 = 0.0;
				for (var hd: u32 = 0u; hd < HEAD_DIM; hd++) {
					let q_idx = seq_i * D_MODEL + head * HEAD_DIM + hd;
					// Read K from the correct mapped KV head
					let k_idx = seq_j * D_MODEL + kv_head * HEAD_DIM + hd;
					score += q[q_idx] * k[k_idx];
				}
				score *= SCALE;
				max_score = max(max_score, score);
			}

			// Second pass: compute softmax and weighted V sum
			var exp_sum: f32 = 0.0;
			for (var seq_j: u32 = 0u; seq_j < SEQ_LEN; seq_j++) {
				if (seq_j > seq_i) { continue; }
				var score: f32 = 0.0;
				for (var hd: u32 = 0u; hd < HEAD_DIM; hd++) {
					let q_idx = seq_i * D_MODEL + head * HEAD_DIM + hd;
					let k_idx = seq_j * D_MODEL + kv_head * HEAD_DIM + hd;
					score += q[q_idx] * k[k_idx];
				}
				score = exp(score * SCALE - max_score);
				exp_sum += score;
				
				// Read V from the correct mapped KV head
				let v_idx = seq_j * D_MODEL + kv_head * HEAD_DIM + head_d;
				sum += score * v[v_idx];
			}

			attn_out[idx] = sum / exp_sum;
		}
	`, l.Spec.SeqLen, l.Spec.DModel, l.Spec.NumHeads, numKVHeads, headDim, scale)
}

func (l *MHALayer) GenerateOutShader() string {
	// Output projection
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> attn_out : array<f32>;
		@group(0) @binding(1) var<storage, read> o_w : array<f32>;
		@group(0) @binding(2) var<storage, read> o_b : array<f32>;
		@group(0) @binding(3) var<storage, read_write> output : array<f32>;

		const SEQ_LEN: u32 = %du;
		const D_MODEL: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let total = SEQ_LEN * D_MODEL;
			if (idx >= total) { return; }

			let seq = idx / D_MODEL;
			let d = idx %% D_MODEL;
			let in_offset = seq * D_MODEL;

			var sum: f32 = o_b[d];
			for (var j: u32 = 0u; j < D_MODEL; j++) {
				sum += attn_out[in_offset + j] * o_w[j * D_MODEL + d];
			}
			output[idx] = sum;
		}
	`, l.Spec.SeqLen, l.Spec.DModel)
}

func (l *MHALayer) GenerateBackwardShader() string {
	// Backward through all 3 stages (simplified - computes input gradient only)
	// 1. d_attn = d_output @ O_w.T
	// 2. For each position: backprop through attention (complex - simplified here)
	// 3. d_input = d_q @ Q_w.T + d_k @ K_w.T + d_v @ V_w.T
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> d_output : array<f32>;
		@group(0) @binding(1) var<storage, read> o_w : array<f32>;
		@group(0) @binding(2) var<storage, read> w_qkv : array<f32>;
		@group(0) @binding(3) var<storage, read_write> d_input : array<f32>;


		const SEQ_LEN: u32 = %du;
		const D_MODEL: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let total = SEQ_LEN * D_MODEL;
			if (idx >= total) { return; }

			let seq = idx / D_MODEL;
			let j = idx %% D_MODEL;

			// Step 1: d_attn = d_output @ O_w.T
			// Then simplified: assume attention is identity for gradient flow
			// d_qkv = d_attn (this is a simplification)
			// d_input = d_qkv @ (Q_w + K_w + V_w).T

			var d_in: f32 = 0.0;

			for (var d: u32 = 0u; d < D_MODEL; d++) {
				// d_attn[seq, d] = sum_k(d_output[seq, k] * o_w[d, k])
				var d_attn: f32 = 0.0;
				for (var k: u32 = 0u; k < D_MODEL; k++) {
					d_attn += d_output[seq * D_MODEL + k] * o_w[d * D_MODEL + k];
				}
				
				// Use d_attn as d_q, d_k, d_v (simplified)
				// d_input[seq, j] += d_attn * (q_w[j,d] + k_w[j,d] + v_w[j,d])
				// With combined weights:
				// q_w = w_qkv[...]
				// k_w = w_qkv[offset + ...]
				let offset_k = D_MODEL * D_MODEL;
				let offset_v = 2u * D_MODEL * D_MODEL;
				
				let w_q = w_qkv[j * D_MODEL + d];
				let w_k = w_qkv[offset_k + j * D_MODEL + d];
				let w_v = w_qkv[offset_v + j * D_MODEL + d];
				
				d_in += d_attn * (w_q + w_k + w_v);
			}

			d_input[idx] = d_in;
		}
	`, l.Spec.SeqLen, l.Spec.DModel)
}

func (l *MHALayer) Compile(ctx *Context, labelPrefix string) error {
	var err error

	// QKV pipeline
	mod1, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_QKV",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateQKVShader()},
	})
	if err != nil {
		return err
	}
	l.pipelineQKV, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_QKVPipe",
		Compute: wgpu.ProgrammableStageDescriptor{Module: mod1, EntryPoint: "main"},
	})
	if err != nil {
		return err
	}

	// Attention pipeline
	mod2, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_Attn",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateAttnShader()},
	})
	if err != nil {
		return err
	}
	l.pipelineAttn, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_AttnPipe",
		Compute: wgpu.ProgrammableStageDescriptor{Module: mod2, EntryPoint: "main"},
	})
	if err != nil {
		return err
	}

	// Output pipeline
	mod3, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_Out",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateOutShader()},
	})
	if err != nil {
		return err
	}
	l.pipelineOut, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_OutPipe",
		Compute: wgpu.ProgrammableStageDescriptor{Module: mod3, EntryPoint: "main"},
	})
	return err
}

func (l *MHALayer) CompileBackward(ctx *Context, labelPrefix string) error {
	mod, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_Bwd",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateBackwardShader()},
	})
	if err != nil {
		return err
	}
	l.bwPipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_BwdPipe",
		Compute: wgpu.ProgrammableStageDescriptor{Module: mod, EntryPoint: "main"},
	})
	return err
}

func (l *MHALayer) CreateBindGroup(ctx *Context, labelPrefix string) error {
	var err error

	// QKV bind group
	l.bindGroupQKV, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_QKVBind",
		Layout: l.pipelineQKV.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
			{Binding: 1, Buffer: l.CombinedWeightsQKV, Size: l.CombinedWeightsQKV.GetSize()},
			{Binding: 2, Buffer: l.CombinedBiasesQKV, Size: l.CombinedBiasesQKV.GetSize()},
			{Binding: 3, Buffer: l.QBuffer, Size: l.QBuffer.GetSize()},
			{Binding: 4, Buffer: l.KBuffer, Size: l.KBuffer.GetSize()},
			{Binding: 5, Buffer: l.VBuffer, Size: l.VBuffer.GetSize()},
		},
	})
	if err != nil {
		return err
	}

	// Attention bind group
	l.bindGroupAttn, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_AttnBind",
		Layout: l.pipelineAttn.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: l.QBuffer, Size: l.QBuffer.GetSize()},
			{Binding: 1, Buffer: l.KBuffer, Size: l.KBuffer.GetSize()},
			{Binding: 2, Buffer: l.VBuffer, Size: l.VBuffer.GetSize()},
			{Binding: 3, Buffer: l.AttnBuffer, Size: l.AttnBuffer.GetSize()},
		},
	})
	if err != nil {
		return err
	}

	// Output bind group
	l.bindGroupOut, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_OutBind",
		Layout: l.pipelineOut.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: l.AttnBuffer, Size: l.AttnBuffer.GetSize()},
			{Binding: 1, Buffer: l.OWeightBuffer, Size: l.OWeightBuffer.GetSize()},
			{Binding: 2, Buffer: l.OBiasBuffer, Size: l.OBiasBuffer.GetSize()},
			{Binding: 3, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
		},
	})
	return err
}

func (l *MHALayer) CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error {
	var err error
	l.bwBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_BwdBind",
		Layout: l.bwPipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
			{Binding: 1, Buffer: l.OWeightBuffer, Size: l.OWeightBuffer.GetSize()},
			{Binding: 2, Buffer: l.CombinedWeightsQKV, Size: l.CombinedWeightsQKV.GetSize()},
			{Binding: 3, Buffer: l.InputGradientBuffer, Size: l.InputGradientBuffer.GetSize()},
		},
	})
	return err
}

func (l *MHALayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	total := l.Spec.SeqLen * l.Spec.DModel
	wg := uint32((total + 255) / 256)

	// Stage 1: QKV projection
	pass.SetPipeline(l.pipelineQKV)
	pass.SetBindGroup(0, l.bindGroupQKV, nil)
	pass.DispatchWorkgroups(wg, 1, 1)

	// Stage 2: Attention
	// Note: Ideally this should be a separate pass for memory barriers,
	// but within the current interface we must use the same pass.
	// We rely on implicit synchronization or lack of overlapping hazards for now.
	pass.SetPipeline(l.pipelineAttn)
	pass.SetBindGroup(0, l.bindGroupAttn, nil)
	pass.DispatchWorkgroups(wg, 1, 1)

	// Stage 3: Output projection
	pass.SetPipeline(l.pipelineOut)
	pass.SetBindGroup(0, l.bindGroupOut, nil)
	pass.DispatchWorkgroups(wg, 1, 1)
}

func (l *MHALayer) DispatchFull(enc *wgpu.CommandEncoder) {
	total := l.Spec.SeqLen * l.Spec.DModel
	wg := uint32((total + 255) / 256)

	// Stage 1: QKV
	pass1 := enc.BeginComputePass(nil)
	pass1.SetPipeline(l.pipelineQKV)
	pass1.SetBindGroup(0, l.bindGroupQKV, nil)
	pass1.DispatchWorkgroups(wg, 1, 1)
	pass1.End()

	// Stage 2: Attention
	pass2 := enc.BeginComputePass(nil)
	pass2.SetPipeline(l.pipelineAttn)
	pass2.SetBindGroup(0, l.bindGroupAttn, nil)
	pass2.DispatchWorkgroups(wg, 1, 1)
	pass2.End()

	// Stage 3: Output projection
	pass3 := enc.BeginComputePass(nil)
	pass3.SetPipeline(l.pipelineOut)
	pass3.SetBindGroup(0, l.bindGroupOut, nil)
	pass3.DispatchWorkgroups(wg, 1, 1)
	pass3.End()
}

func (l *MHALayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(l.bwPipeline)
	pass.SetBindGroup(0, l.bwBindGroup, nil)
	total := l.Spec.SeqLen * l.Spec.DModel
	pass.DispatchWorkgroups(uint32((total+255)/256), 1, 1)
	pass.End()
}

func (l *MHALayer) UploadWeights(ctx *Context) {
	upload := func(buf *wgpu.Buffer, data []float32) {
		if len(data) > 0 {
			ctx.Queue.WriteBuffer(buf, 0, wgpu.ToBytes(data))
		}
	}
	upload(l.CombinedWeightsQKV, nil) // Already uploaded
	upload(l.OWeightBuffer, l.Spec.OWeights)
}

func (l *MHALayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	return nil, nil, nil // Complex multi-weight
}

func (l *MHALayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	iGrad, err := ReadBuffer(l.InputGradientBuffer, l.Spec.SeqLen*l.Spec.DModel)
	return nil, nil, iGrad, err
}

func (l *MHALayer) Cleanup() {
	bufs := []*wgpu.Buffer{
		l.InputBuffer, l.OutputBuffer, l.StagingBuffer,
		l.InputBuffer, l.OutputBuffer, l.StagingBuffer,
		l.CombinedWeightsQKV, l.CombinedBiasesQKV, l.OWeightBuffer,
		l.OBiasBuffer,
		l.QBuffer, l.KBuffer, l.VBuffer, l.AttnBuffer,
		l.InputGradientBuffer,
	}
	for _, b := range bufs {
		if b != nil {
			b.Destroy()
		}
	}

	pipes := []*wgpu.ComputePipeline{l.pipelineQKV, l.pipelineAttn, l.pipelineOut, l.bwPipeline}
	for _, p := range pipes {
		if p != nil {
			p.Release()
		}
	}

	bgs := []*wgpu.BindGroup{l.bindGroupQKV, l.bindGroupAttn, l.bindGroupOut, l.bwBindGroup}
	for _, bg := range bgs {
		if bg != nil {
			bg.Release()
		}
	}
}
