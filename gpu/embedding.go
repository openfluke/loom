package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// EmbeddingSpec defines configuration for Embedding layer
type EmbeddingSpec struct {
	VocabSize    int       // Number of tokens in vocabulary
	EmbeddingDim int       // Dimension of each embedding vector
	SeqLength    int       // Sequence length (number of tokens to lookup)
	Weights      []float32 // [VocabSize * EmbeddingDim] - flattened embedding table
}

// EmbeddingLayer holds GPU resources for Embedding lookup
type EmbeddingLayer struct {
	Spec EmbeddingSpec

	pipeline  *wgpu.ComputePipeline
	bindGroup *wgpu.BindGroup

	TokenBuffer   *wgpu.Buffer // Input token IDs (u32)
	OutputBuffer  *wgpu.Buffer // Output embeddings
	WeightBuffer  *wgpu.Buffer // Embedding weights
	StagingBuffer *wgpu.Buffer

	// Backward
	WeightGradientBuffer *wgpu.Buffer

	bwPipeline  *wgpu.ComputePipeline
	bwBindGroup *wgpu.BindGroup
}

// Interface implementations
func (l *EmbeddingLayer) GetInputBuffer() *wgpu.Buffer         { return l.TokenBuffer }
func (l *EmbeddingLayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *EmbeddingLayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *EmbeddingLayer) GetInputGradientBuffer() *wgpu.Buffer { return nil } // No input grad for embeddings

func (l *EmbeddingLayer) AllocateBuffers(ctx *Context, labelPrefix string) error {
	var err error

	// Token input buffer (u32 token IDs)
	l.TokenBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Tokens",
		Size:  uint64(l.Spec.SeqLength * 4), // u32 per token
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}

	outputSize := l.Spec.SeqLength * l.Spec.EmbeddingDim
	l.OutputBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Out",
		Size:  uint64(outputSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Weight buffer
	if len(l.Spec.Weights) > 0 {
		l.WeightBuffer, err = NewFloatBuffer(l.Spec.Weights, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
	} else {
		// Placeholder
		placeholder := make([]float32, l.Spec.VocabSize*l.Spec.EmbeddingDim)
		l.WeightBuffer, err = NewFloatBuffer(placeholder, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
	}
	if err != nil {
		return err
	}

	l.StagingBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Staging",
		Size:  uint64(outputSize * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	return err
}

func (l *EmbeddingLayer) AllocateBackwardBuffers(ctx *Context, labelPrefix string) error {
	var err error
	size := l.Spec.VocabSize * l.Spec.EmbeddingDim
	l.WeightGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_WGrad",
		Size:  uint64(size * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	return err
}

func (l *EmbeddingLayer) GenerateShader() string {
	// Each thread handles one output position (token x embedding dimension)
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> tokens : array<u32>;
		@group(0) @binding(1) var<storage, read> weights : array<f32>;
		@group(0) @binding(2) var<storage, read_write> output : array<f32>;

		const SEQ_LEN: u32 = %du;
		const EMB_DIM: u32 = %du;
		const VOCAB_SIZE: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let total = SEQ_LEN * EMB_DIM;
			if (idx >= total) { return; }

			let pos = idx / EMB_DIM;  // Which token position
			let dim = idx %% EMB_DIM;  // Which dimension

			let token_id = tokens[pos];
			if (token_id < VOCAB_SIZE) {
				output[idx] = weights[token_id * EMB_DIM + dim];
			} else {
				output[idx] = 0.0;
			}
		}
	`, l.Spec.SeqLength, l.Spec.EmbeddingDim, l.Spec.VocabSize)
}

func (l *EmbeddingLayer) GenerateBackwardShader() string {
	// Backward: Pull gradient from d_output to d_weights (Atomic-Free)
	// Iterate over all tokens for each weight element to find matches
	// O(Vocab * Dim * Batch * Seq) - simpler/safer for MSL than atomics
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> tokens : array<u32>;
		@group(0) @binding(1) var<storage, read> d_output : array<f32>;
		@group(0) @binding(2) var<storage, read_write> d_weights : array<f32>;

		const SEQ_LEN: u32 = %du;
		const EMB_DIM: u32 = %du;
		const VOCAB_SIZE: u32 = %du;
		const INPUT_TOTAL: u32 = %du; // SeqLen * BatchSize (conceptually, though here just seqlen)

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x; // Weight index (Vocab * Dim)
			let total_weights = VOCAB_SIZE * EMB_DIM;
			if (idx >= total_weights) { return; }

			let vocab_id = idx / EMB_DIM;
			let dim = idx %% EMB_DIM;

			var grad_acc: f32 = 0.0;
			
			// Iterate over all input tokens to see if they contributed to this weight
			// Note: l.Spec.SeqLength acts as total input tokens here
			for (var i: u32 = 0u; i < SEQ_LEN; i++) {
				if (tokens[i] == vocab_id) {
					// This token used this weight, accumulate gradient
					grad_acc += d_output[i * EMB_DIM + dim];
				}
			}
			
			// Write accumulated gradient (assuming buffer zeroed or overwriting)
			// Since we account for all inputs, strictly overwriting is correct if we assume standard backprop
			// But usually we accumulate into existing gradients? 
			// Standard Loom usually zeroes before backward or accumulates.
			// The atomic version added. Here we calculate the WHOLE delta for this weight from this batch.
			// So we should ADD to existing d_weights? 
			// If d_weights accumulates across batches or optimizers, read-modify-write is safest.
			
			// However, without atomics, read-add-write is only safe if one thread owns this address.
			// Here, exactly one thread owns 'idx' in d_weights. So race-free!
			
			// But we need to load current value if we want to support accumulation?
			// Usually ZeroGradients is called before Backward.
			// Loom convention: Backward typically *adds* to gradient buffer (to support branching/sharing).
			// So we will add.
			
			let current = d_weights[idx];
			d_weights[idx] = current + grad_acc;
		}
	`, l.Spec.SeqLength, l.Spec.EmbeddingDim, l.Spec.VocabSize, l.Spec.SeqLength)
}

func (l *EmbeddingLayer) Compile(ctx *Context, labelPrefix string) error {
	shader := l.GenerateShader()
	module, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_Shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shader},
	})
	if err != nil {
		return err
	}
	defer module.Release()

	// Explicit Layout
	bgl, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: labelPrefix + "_BGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // tokens
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // weights
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // output
		},
	})
	if err != nil {
		return err
	}
	pl, err := ctx.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            labelPrefix + "_PL",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgl},
	})
	if err != nil {
		return err
	}

	l.pipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_Pipe",
		Layout:  pl,
		Compute: wgpu.ProgrammableStageDescriptor{Module: module, EntryPoint: "main"},
	})
	return err
}

func (l *EmbeddingLayer) CompileBackward(ctx *Context, labelPrefix string) error {
	shader := l.GenerateBackwardShader()
	module, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_BwdShader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shader},
	})
	if err != nil {
		return err
	}
	defer module.Release()

	// Explicit Layout Backward
	bglBwd, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: labelPrefix + "_BwdBGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // tokens
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // d_output
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // d_weights (changed from atomic)
		},
	})
	if err != nil {
		return err
	}
	plBwd, err := ctx.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            labelPrefix + "_BwdPL",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bglBwd},
	})
	if err != nil {
		return err
	}

	l.bwPipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_BwdPipe",
		Layout:  plBwd,
		Compute: wgpu.ProgrammableStageDescriptor{Module: module, EntryPoint: "main"},
	})
	return err
}

func (l *EmbeddingLayer) CreateBindGroup(ctx *Context, labelPrefix string) error {
	var err error
	entries := []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: l.TokenBuffer, Size: l.TokenBuffer.GetSize()},
		{Binding: 1, Buffer: l.WeightBuffer, Size: l.WeightBuffer.GetSize()},
		{Binding: 2, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
	}
	l.bindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   labelPrefix + "_Bind",
		Layout:  l.pipeline.GetBindGroupLayout(0),
		Entries: entries,
	})
	return err
}

func (l *EmbeddingLayer) CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error {
	var err error
	entries := []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: l.TokenBuffer, Size: l.TokenBuffer.GetSize()},
		{Binding: 1, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
		{Binding: 2, Buffer: l.WeightGradientBuffer, Size: l.WeightGradientBuffer.GetSize()},
	}
	l.bwBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   labelPrefix + "_BwdBind",
		Layout:  l.bwPipeline.GetBindGroupLayout(0),
		Entries: entries,
	})
	return err
}

func (l *EmbeddingLayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	pass.SetPipeline(l.pipeline)
	pass.SetBindGroup(0, l.bindGroup, nil)
	total := l.Spec.SeqLength * l.Spec.EmbeddingDim
	wgx := (total + 255) / 256
	pass.DispatchWorkgroups(uint32(wgx), 1, 1)
}

func (l *EmbeddingLayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(l.bwPipeline)
	pass.SetBindGroup(0, l.bwBindGroup, nil)
	total := l.Spec.SeqLength * l.Spec.EmbeddingDim
	wgx := (total + 255) / 256
	pass.DispatchWorkgroups(uint32(wgx), 1, 1)
	pass.End()
}

func (l *EmbeddingLayer) UploadWeights(ctx *Context) {
	if len(l.Spec.Weights) > 0 {
		ctx.Queue.WriteBuffer(l.WeightBuffer, 0, wgpu.ToBytes(l.Spec.Weights))
	}
}

func (l *EmbeddingLayer) ZeroGradients(ctx *Context) {
	size := l.Spec.VocabSize * l.Spec.EmbeddingDim
	zeros := make([]float32, size)
	ctx.Queue.WriteBuffer(l.WeightGradientBuffer, 0, wgpu.ToBytes(zeros))
}

func (l *EmbeddingLayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	weights, err := ReadBuffer(l.WeightBuffer, l.Spec.VocabSize*l.Spec.EmbeddingDim)
	return weights, nil, err
}

func (l *EmbeddingLayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	wGrad, err := ReadBuffer(l.WeightGradientBuffer, l.Spec.VocabSize*l.Spec.EmbeddingDim)
	return wGrad, nil, nil, err
}

func (l *EmbeddingLayer) Cleanup() {
	if l.TokenBuffer != nil {
		l.TokenBuffer.Destroy()
	}
	if l.OutputBuffer != nil {
		l.OutputBuffer.Destroy()
	}
	if l.WeightBuffer != nil {
		l.WeightBuffer.Destroy()
	}
	if l.StagingBuffer != nil {
		l.StagingBuffer.Destroy()
	}
	if l.WeightGradientBuffer != nil {
		l.WeightGradientBuffer.Destroy()
	}
	if l.pipeline != nil {
		l.pipeline.Release()
	}
	if l.bindGroup != nil {
		l.bindGroup.Release()
	}
	if l.bwPipeline != nil {
		l.bwPipeline.Release()
	}
	if l.bwBindGroup != nil {
		l.bwBindGroup.Release()
	}
}
