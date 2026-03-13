package gpu

// lmhead_fp4.go — GPU-accelerated Language Model Head
//
// Runs the final two steps of transformer inference entirely on GPU:
//   1. RMSNorm(hidden; gamma)
//   2. Linear: logits = norm_hidden @ lm_head.T  (vocab × hiddenSize matmul)
//
// This eliminates the #1 per-token CPU bottleneck in fp4_quicktalk:
// a 49152 × 576 float32 matmul that was previously executed on CPU.
//
// Usage:
//   lmh, err := NewGPULMHead(ctx, hiddenSize, vocabSize, normGamma, lmWeights)
//   logits, err := lmh.Infer(ctx, hidden)   // hidden = []float32 len=hiddenSize
//   lmh.Cleanup()

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// GPULMHead holds the GPU resources for RMSNorm + LM head projection.
type GPULMHead struct {
	HiddenSize int
	VocabSize  int

	// RMSNorm pipeline
	pipelineNorm *wgpu.ComputePipeline
	bgNorm       *wgpu.BindGroup

	// Linear (lm_head) pipeline
	pipelineLM *wgpu.ComputePipeline
	bgLM       *wgpu.BindGroup

	// Buffers
	HiddenBuf  *wgpu.Buffer // input: hidden state [hiddenSize] float32
	NormBuf    *wgpu.Buffer // intermediate: RMSnorm output [hiddenSize]
	LogitsBuf  *wgpu.Buffer // output: logits [vocabSize]
	StagingBuf *wgpu.Buffer // CPU-readable copy of logits

	GammaBuf   *wgpu.Buffer // RMSNorm gamma [hiddenSize]
	LMBuf      *wgpu.Buffer // LM head weights [vocabSize × hiddenSize] float32
	EpsilonBuf *wgpu.Buffer // uniform: epsilon (f32)
}

// NewGPULMHead creates, compiles, and uploads an LM head onto the GPU.
//   - gamma: RMSNorm scale weights (len = hiddenSize), nil = skip norm
//   - lmWeights: [vocabSize × hiddenSize] row-major, i.e. lmWeights[v*h + d]
func NewGPULMHead(ctx *Context, hiddenSize, vocabSize int, gamma, lmWeights []float32) (*GPULMHead, error) {
	l := &GPULMHead{
		HiddenSize: hiddenSize,
		VocabSize:  vocabSize,
	}
	if err := l.allocate(ctx, gamma, lmWeights); err != nil {
		l.Cleanup()
		return nil, err
	}
	if err := l.compile(ctx); err != nil {
		l.Cleanup()
		return nil, err
	}
	if err := l.bindGroups(ctx); err != nil {
		l.Cleanup()
		return nil, err
	}
	return l, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Cleanup
// ─────────────────────────────────────────────────────────────────────────────

func (l *GPULMHead) Cleanup() {
	for _, b := range []*wgpu.Buffer{
		l.HiddenBuf, l.NormBuf, l.LogitsBuf, l.StagingBuf,
		l.GammaBuf, l.LMBuf, l.EpsilonBuf,
	} {
		if b != nil {
			b.Destroy()
		}
	}
	for _, p := range []*wgpu.ComputePipeline{l.pipelineNorm, l.pipelineLM} {
		if p != nil {
			p.Release()
		}
	}
	for _, bg := range []*wgpu.BindGroup{l.bgNorm, l.bgLM} {
		if bg != nil {
			bg.Release()
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Allocate buffers
// ─────────────────────────────────────────────────────────────────────────────

func (l *GPULMHead) allocate(ctx *Context, gamma, lmWeights []float32) error {
	var err error
	sz := func(n int) uint64 { return uint64(n * 4) }
	rw := wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc

	l.HiddenBuf, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "LMH_hidden", Size: sz(l.HiddenSize), Usage: rw,
	})

	if err != nil {
		return fmt.Errorf("LMH hidden buf: %w", err)
	}

	l.NormBuf, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "LMH_norm", Size: sz(l.HiddenSize), Usage: rw,
	})

	if err != nil {
		return fmt.Errorf("LMH norm buf: %w", err)
	}

	l.LogitsBuf, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "LMH_logits", Size: sz(l.VocabSize), Usage: rw,
	})

	if err != nil {
		return fmt.Errorf("LMH logits buf: %w", err)
	}

	l.StagingBuf, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "LMH_staging",
		Size:  sz(l.VocabSize),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})

	if err != nil {
		return fmt.Errorf("LMH staging buf: %w", err)
	}

	// Gamma — fill with 1s if not provided (identity norm)
	g := gamma
	if len(g) == 0 {
		g = make([]float32, l.HiddenSize)
		for i := range g {
			g[i] = 1.0
		}
	}
	l.GammaBuf, err = NewFloatBuffer(g, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return fmt.Errorf("LMH gamma buf: %w", err)
	}

	// LM head weights
	l.LMBuf, err = NewFloatBuffer(lmWeights, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return fmt.Errorf("LMH lm buf: %w", err)
	}

	// Epsilon as a 1-element uniform buffer
	eps := []float32{1e-6}
	l.EpsilonBuf, err = NewFloatBuffer(eps, wgpu.BufferUsageUniform|wgpu.BufferUsageCopyDst)
	if err != nil {
		return fmt.Errorf("LMH eps buf: %w", err)
	}

	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Shaders
// ─────────────────────────────────────────────────────────────────────────────

// rmsNormShader: one thread per hidden dimension, parallel reduce via shared memory.
// Each workgroup handles one entire vector reduction.
func (l *GPULMHead) rmsNormShaderCode() string {
	return fmt.Sprintf(`
const H : u32 = %du;   // hidden size

@group(0) @binding(0) var<storage, read>       hidden_in : array<f32>;  // [H]
@group(0) @binding(1) var<storage, read>       gamma     : array<f32>;  // [H]
@group(0) @binding(2) var<storage, read_write> norm_out  : array<f32>;  // [H]
@group(0) @binding(3) var<uniform>             epsilon   : f32;

var<workgroup> shared_sq : array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>,
        @builtin(workgroup_id)         wkid: vec3<u32>) {

    let local_id = lid.x;

    // Each thread sums a contiguous stripe of squares
    var sq_sum : f32 = 0.0;
    var i : u32 = local_id;
    loop {
        if i >= H { break; }
        let v = hidden_in[i];
        sq_sum += v * v;
        i += 256u;
        continuing { }
    }
    shared_sq[local_id] = sq_sum;
    workgroupBarrier();

    // Parallel reduction within workgroup (256 → 1)
    var stride : u32 = 128u;
    loop {
        if stride == 0u { break; }
        if local_id < stride {
            shared_sq[local_id] += shared_sq[local_id + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
        continuing { }
    }

    // Compute RMS scale factor (thread 0 writes, others read after barrier)
    let rms_scale = 1.0 / sqrt(shared_sq[0] / f32(H) + epsilon);
    workgroupBarrier();

    // Apply scale + gamma, each thread handles a stripe
    var j : u32 = local_id;
    loop {
        if j >= H { break; }
        norm_out[j] = hidden_in[j] * rms_scale * gamma[j];
        j += 256u;
        continuing { }
    }
}
`, l.HiddenSize)
}

// lmHeadShaderCode: vocab × hidden matmul. One thread per vocab token.
func (l *GPULMHead) lmHeadShaderCode() string {
	return fmt.Sprintf(`
const H    : u32 = %du;  // hidden size
const VOCAB: u32 = %du;  // vocab size

@group(0) @binding(0) var<storage, read>       norm_in  : array<f32>;  // [H]
@group(0) @binding(1) var<storage, read>       lm_w     : array<f32>;  // [VOCAB * H], row-major: w[v*H+d]
@group(0) @binding(2) var<storage, read_write> logits   : array<f32>;  // [VOCAB]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= VOCAB) { return; }

    let base = v * H;
    var acc : f32 = 0.0;

    // Unrolled 4-wide inner loop
    var d : u32 = 0u;
    loop {
        if (d + 3u >= H) { break; }
        acc += norm_in[d    ] * lm_w[base + d    ];
        acc += norm_in[d + 1u] * lm_w[base + d + 1u];
        acc += norm_in[d + 2u] * lm_w[base + d + 2u];
        acc += norm_in[d + 3u] * lm_w[base + d + 3u];
        d += 4u;
        continuing { }
    }
    // Tail
    loop {
        if (d >= H) { break; }
        acc += norm_in[d] * lm_w[base + d];
        d += 1u;
        continuing { }
    }

    logits[v] = acc;
}
`, l.HiddenSize, l.VocabSize)
}

// ─────────────────────────────────────────────────────────────────────────────
// Compile pipelines
// ─────────────────────────────────────────────────────────────────────────────

func (l *GPULMHead) compile(ctx *Context) error {
	// ── RMSNorm pipeline ──────────────────────────────────────────────────────
	normMod, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "LMH_NormShader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.rmsNormShaderCode()},
	})
	if err != nil {
		return fmt.Errorf("LMH rmsnorm shader: %w", err)
	}
	defer normMod.Release()

	normBGL, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "LMH_NormBGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, HasDynamicOffset: false, MinBindingSize: 4}},
		},
	})
	if err != nil {
		return fmt.Errorf("LMH norm bgl: %w", err)
	}
	normPL, err := ctx.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "LMH_NormPL",
		BindGroupLayouts: []*wgpu.BindGroupLayout{normBGL},
	})
	if err != nil {
		return fmt.Errorf("LMH norm pl: %w", err)
	}
	l.pipelineNorm, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  "LMH_NormPipe",
		Layout: normPL,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     normMod,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return fmt.Errorf("LMH norm pipeline: %w", err)
	}

	// ── LM head linear pipeline ───────────────────────────────────────────────
	lmMod, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "LMH_LMShader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.lmHeadShaderCode()},
	})
	if err != nil {
		return fmt.Errorf("LMH lm shader: %w", err)
	}
	defer lmMod.Release()

	lmBGL, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "LMH_LMBGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
		},
	})
	if err != nil {
		return fmt.Errorf("LMH lm bgl: %w", err)
	}
	lmPL, err := ctx.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "LMH_LMPL",
		BindGroupLayouts: []*wgpu.BindGroupLayout{lmBGL},
	})
	if err != nil {
		return fmt.Errorf("LMH lm pl: %w", err)
	}
	l.pipelineLM, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  "LMH_LMPipe",
		Layout: lmPL,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     lmMod,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return fmt.Errorf("LMH lm pipeline: %w", err)
	}

	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Bind groups
// ─────────────────────────────────────────────────────────────────────────────

func (l *GPULMHead) bindGroups(ctx *Context) error {
	var err error

	// RMSNorm bind group
	l.bgNorm, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "LMH_NormBG",
		Layout: l.pipelineNorm.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: l.HiddenBuf, Size: l.HiddenBuf.GetSize()},
			{Binding: 1, Buffer: l.GammaBuf, Size: l.GammaBuf.GetSize()},
			{Binding: 2, Buffer: l.NormBuf, Size: l.NormBuf.GetSize()},
			{Binding: 3, Buffer: l.EpsilonBuf, Size: 4},
		},
	})
	if err != nil {
		return fmt.Errorf("LMH norm bg: %w", err)
	}

	// LM head bind group
	l.bgLM, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "LMH_LMBG",
		Layout: l.pipelineLM.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: l.NormBuf, Size: l.NormBuf.GetSize()},
			{Binding: 1, Buffer: l.LMBuf, Size: l.LMBuf.GetSize()},
			{Binding: 2, Buffer: l.LogitsBuf, Size: l.LogitsBuf.GetSize()},
		},
	})
	if err != nil {
		return fmt.Errorf("LMH lm bg: %w", err)
	}

	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Infer — the hot path called once per generated token
// ─────────────────────────────────────────────────────────────────────────────

// Infer runs RMSNorm + LM head on the GPU for the given hidden state slice.
// Returns logits as []float32 of length VocabSize.
// hidden must be exactly len=HiddenSize.
func (l *GPULMHead) Infer(ctx *Context, hidden []float32) ([]float32, error) {
	if len(hidden) < l.HiddenSize {
		return nil, fmt.Errorf("hidden too short: got %d, need %d", len(hidden), l.HiddenSize)
	}

	// Upload the hidden state (only the last hidden vector)
	ctx.Queue.WriteBuffer(l.HiddenBuf, 0, wgpu.ToBytes(hidden[:l.HiddenSize]))

	// Encode both passes
	enc, err := ctx.Device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("LMH enc: %w", err)
	}

	pass := enc.BeginComputePass(nil)

	// Pass 1: RMSNorm — single workgroup (256 threads), handles any hidden size
	pass.SetPipeline(l.pipelineNorm)
	pass.SetBindGroup(0, l.bgNorm, nil)
	pass.DispatchWorkgroups(1, 1, 1) // one workgroup = one vector

	// Pass 2: LM head linear — ceil(VocabSize/256) workgroups
	wgLM := (uint32(l.VocabSize) + 255) / 256
	pass.SetPipeline(l.pipelineLM)
	pass.SetBindGroup(0, l.bgLM, nil)
	pass.DispatchWorkgroups(wgLM, 1, 1)

	pass.End()

	// Copy logits → staging
	enc.CopyBufferToBuffer(l.LogitsBuf, 0, l.StagingBuf, 0, l.LogitsBuf.GetSize())

	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, fmt.Errorf("LMH enc finish: %w", err)
	}
	ctx.Queue.Submit(cmd)

	// Readback
	logits, err := readStagingBuffer(ctx, l.StagingBuf, l.VocabSize)
	if err != nil {
		return nil, fmt.Errorf("LMH readback: %w", err)
	}
	return logits, nil
}
