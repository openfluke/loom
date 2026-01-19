package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// RMSNormSpec defines configuration for RMS Normalization layer
// RMSNorm is simpler than LayerNorm - only gamma, no beta, no mean subtraction
type RMSNormSpec struct {
	NormSize  int
	BatchSize int // Number of vectors to normalize (default 1)
	Epsilon   float32
	Gamma     []float32 // [NormSize] - scale parameters
}

// RMSNormLayer holds GPU resources for RMS Normalization
type RMSNormLayer struct {
	Spec RMSNormSpec

	pipeline  *wgpu.ComputePipeline
	bindGroup *wgpu.BindGroup

	InputBuffer   *wgpu.Buffer
	OutputBuffer  *wgpu.Buffer
	StagingBuffer *wgpu.Buffer
	GammaBuffer   *wgpu.Buffer

	// Backward
	GammaGradientBuffer *wgpu.Buffer
	InputGradientBuffer *wgpu.Buffer

	// NEW: Batch Gradients for atomic-free reduction
	GammaBatchGradientBuffer *wgpu.Buffer

	bwPipeline  *wgpu.ComputePipeline
	bwBindGroup *wgpu.BindGroup

	reducePipeline  *wgpu.ComputePipeline
	reduceBindGroup *wgpu.BindGroup
}

// Interface implementations
func (l *RMSNormLayer) GetInputBuffer() *wgpu.Buffer         { return l.InputBuffer }
func (l *RMSNormLayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *RMSNormLayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *RMSNormLayer) GetInputGradientBuffer() *wgpu.Buffer { return l.InputGradientBuffer }

func (l *RMSNormLayer) AllocateBuffers(ctx *Context, labelPrefix string) error {
	var err error

	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	totalSize := batch * l.Spec.NormSize

	// Input/Output buffers
	l.InputBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_In",
		Size:  uint64(totalSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	l.OutputBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Out",
		Size:  uint64(totalSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Gamma buffer
	if len(l.Spec.Gamma) > 0 {
		l.GammaBuffer, err = NewFloatBuffer(l.Spec.Gamma, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
	} else {
		l.GammaBuffer, err = NewFloatBuffer([]float32{1.0}, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
	}
	if err != nil {
		return err
	}

	// Staging buffer
	l.StagingBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Staging",
		Size:  uint64(totalSize * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	return err
}

func (l *RMSNormLayer) AllocateBackwardBuffers(ctx *Context, labelPrefix string) error {
	var err error

	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	totalSize := batch * l.Spec.NormSize

	// Input gradient buffer
	l.InputGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_InGrad",
		Size:  uint64(totalSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Gamma gradient buffer (atomic for accumulation)
	sz := len(l.Spec.Gamma)
	if sz == 0 {
		sz = 1
	}
	l.GammaGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_GammaGrad",
		Size:  uint64(sz * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// NEW: Batch Gradients (Batch * NormSize)
	batchTotal := batch * l.Spec.NormSize
	l.GammaBatchGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_GammaBatchGrad",
		Size:  uint64(batchTotal * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})

	return err
}

func (l *RMSNormLayer) GenerateShader() string {
	// Optimized RMSNorm using workgroup_size(256) with shared memory reductions
	// RMSNorm: y = x * gamma / sqrt(mean(x^2) + eps)

	elemsPerThread := (l.Spec.NormSize + 255) / 256

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read_write> output : array<f32>;
		@group(0) @binding(2) var<storage, read> gamma : array<f32>;

		const N: u32 = %du;
		const EPS: f32 = %f;
		const ELEMS_PER_THREAD: u32 = %du;

		var<workgroup> shared_sq: array<f32, 256>;
		var<workgroup> wg_inv_rms: f32;

		@compute @workgroup_size(256)
		fn main(
			@builtin(workgroup_id) wg_id: vec3<u32>,
			@builtin(local_invocation_id) local_id: vec3<u32>
		) {
			let batch = wg_id.x;
			let tid = local_id.x;
			let offset = batch * N;

			// Phase 1: Each thread sums squares of its elements
			var local_sq: f32 = 0.0;
			for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
				let idx = tid + i * 256u;
				if (idx < N) {
					let val = input[offset + idx];
					local_sq += val * val;
				}
			}
			shared_sq[tid] = local_sq;
			workgroupBarrier();

			// Parallel reduction for sum of squares
			for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
				if (tid < s) {
					shared_sq[tid] = shared_sq[tid] + shared_sq[tid + s];
				}
				workgroupBarrier();
			}

			// Thread 0 computes inverse RMS
			if (tid == 0u) {
				let mean_sq = shared_sq[0] / f32(N);
				wg_inv_rms = 1.0 / sqrt(mean_sq + EPS);
			}
			workgroupBarrier();

			let inv_rms = wg_inv_rms;

			// Phase 2: Normalize each element
			for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
				let idx = tid + i * 256u;
				if (idx < N) {
					let val = input[offset + idx];
					let norm = val * inv_rms;
					
					var g: f32 = 1.0;
					if (arrayLength(&gamma) >= N) { g = gamma[idx]; }

					output[offset + idx] = norm * g;
				}
			}
		}
	`, l.Spec.NormSize, l.Spec.Epsilon, elemsPerThread)
}

func (l *RMSNormLayer) GenerateBackwardShader() string {
	// RMSNorm backward: compute dGamma and dInput
	// Uses atomic-free partial reduction (Batch * NormSize)
	elemsPerThread := (l.Spec.NormSize + 255) / 256

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read> d_output : array<f32>;
		@group(0) @binding(2) var<storage, read> gamma : array<f32>;
		
		@group(0) @binding(3) var<storage, read_write> d_input : array<f32>;
		@group(0) @binding(4) var<storage, read_write> d_gamma_batch : array<f32>;

		const N: u32 = %du;
		const EPS: f32 = %f;
		const ELEMS_PER_THREAD: u32 = %du;

		var<workgroup> shared_sq: array<f32, 256>;
		var<workgroup> wg_inv_rms: f32;
		var<workgroup> wg_sum_dxhat_x: f32;

		@compute @workgroup_size(256)
		fn main(
			@builtin(workgroup_id) wg_id: vec3<u32>,
			@builtin(local_invocation_id) local_id: vec3<u32>
		) {
			let batch = wg_id.x;
			let tid = local_id.x;
			let offset = batch * N;
			
			// Phase 1: Compute RMS
			var local_sq: f32 = 0.0;
			for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
				let idx = tid + i * 256u;
				if (idx < N) {
					let val = input[offset + idx];
					local_sq += val * val;
				}
			}
			shared_sq[tid] = local_sq;
			workgroupBarrier();

			for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
				if (tid < s) { shared_sq[tid] = shared_sq[tid] + shared_sq[tid + s]; }
				workgroupBarrier();
			}
			if (tid == 0u) {
				let mean_sq = shared_sq[0] / f32(N);
				wg_inv_rms = 1.0 / sqrt(mean_sq + EPS);
			}
			workgroupBarrier();
			let inv_rms = wg_inv_rms;
			
			// Phase 2: Compute d_gamma (partial) and term2
			var local_sum_dxhat_x: f32 = 0.0;
			
			for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
				let idx = tid + i * 256u;
				if (idx < N) {
					let val = input[offset + idx];
					let dout = d_output[offset + idx];
					let x_hat = val * inv_rms;
					
					// Store partial gradient directly
					let out_idx = offset + idx;
					d_gamma_batch[out_idx] = dout * x_hat;
					
					var g: f32 = 1.0;
					if (arrayLength(&gamma) >= N) { g = gamma[idx]; }
					
					local_sum_dxhat_x += dout * g * val;
				}
			}
			
			// Reduce sum_dxhat_x
			shared_sq[tid] = local_sum_dxhat_x;
			workgroupBarrier();
			for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
				if (tid < s) { shared_sq[tid] = shared_sq[tid] + shared_sq[tid + s]; }
				workgroupBarrier();
			}
			if (tid == 0u) { wg_sum_dxhat_x = shared_sq[0]; }
			workgroupBarrier();
			
			let inv_rms3 = inv_rms * inv_rms * inv_rms;
			let term2 = wg_sum_dxhat_x * inv_rms3 / f32(N);
			
			// Phase 3: Compute d_input
			for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
				let idx = tid + i * 256u;
				if (idx < N) {
					let val = input[offset + idx];
					let dout = d_output[offset + idx];
					
					var g: f32 = 1.0;
					if (arrayLength(&gamma) >= N) { g = gamma[idx]; }
					
					let dx = (dout * g * inv_rms) - (val * term2);
					d_input[offset + idx] = dx;
				}
			}
		}
	`, l.Spec.NormSize, l.Spec.Epsilon, elemsPerThread)
}

func (l *RMSNormLayer) GenerateReduceShader() string {
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> batch_gamma : array<f32>;
		@group(0) @binding(1) var<storage, read_write> final_gamma : array<f32>;

		const N: u32 = %du;
		const BATCH: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			if (idx >= N) { return; }

			var sum_g: f32 = 0.0;

			for (var b: u32 = 0u; b < BATCH; b++) {
				let offset = b * N + idx;
				sum_g += batch_gamma[offset];
			}

			// Accumulate
			final_gamma[idx] = final_gamma[idx] + sum_g;
		}
	`, l.Spec.NormSize, l.Spec.BatchSize)
}

func (l *RMSNormLayer) Compile(ctx *Context, labelPrefix string) error {
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
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Input
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // Output
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Gamma
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

func (l *RMSNormLayer) CompileBackward(ctx *Context, labelPrefix string) error {
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
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Input
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // dOutput
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Gamma
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // dInput
			{Binding: 4, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // dGamma
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
	if err != nil {
		return err
	}

	// 2. Reduce Pipeline
	redShader := l.GenerateReduceShader()
	redModule, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_RedShader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: redShader},
	})
	if err != nil {
		return err
	}
	defer redModule.Release()

	bglRed, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: labelPrefix + "_RedBGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // batchG
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // finalG
		},
	})
	if err != nil {
		return err
	}
	plRed, err := ctx.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            labelPrefix + "_RedPL",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bglRed},
	})
	if err != nil {
		return err
	}
	l.reducePipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_RedPipe",
		Layout:  plRed,
		Compute: wgpu.ProgrammableStageDescriptor{Module: redModule, EntryPoint: "main"},
	})

	return err
}

func (l *RMSNormLayer) CreateBindGroup(ctx *Context, labelPrefix string) error {
	var err error
	entries := []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
		{Binding: 1, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
		{Binding: 2, Buffer: l.GammaBuffer, Size: l.GammaBuffer.GetSize()},
	}
	l.bindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   labelPrefix + "_Bind",
		Layout:  l.pipeline.GetBindGroupLayout(0),
		Entries: entries,
	})
	return err
}

func (l *RMSNormLayer) CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error {
	var err error
	entries := []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
		{Binding: 1, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
		{Binding: 2, Buffer: l.GammaBuffer, Size: l.GammaBuffer.GetSize()},
		{Binding: 3, Buffer: l.InputGradientBuffer, Size: l.InputGradientBuffer.GetSize()},
		{Binding: 4, Buffer: l.GammaBatchGradientBuffer, Size: l.GammaBatchGradientBuffer.GetSize()}, // Batch Grads
	}
	l.bwBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   labelPrefix + "_BwdBind",
		Layout:  l.bwPipeline.GetBindGroupLayout(0),
		Entries: entries,
	})
	if err != nil {
		return err
	}

	// Reduce BindGroup
	redEntries := []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: l.GammaBatchGradientBuffer, Size: l.GammaBatchGradientBuffer.GetSize()},
		{Binding: 1, Buffer: l.GammaGradientBuffer, Size: l.GammaGradientBuffer.GetSize()},
	}
	l.reduceBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   labelPrefix + "_RedBind",
		Layout:  l.reducePipeline.GetBindGroupLayout(0),
		Entries: redEntries,
	})

	return err
}

func (l *RMSNormLayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	pass.SetPipeline(l.pipeline)
	pass.SetBindGroup(0, l.bindGroup, nil)
	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	pass.DispatchWorkgroups(uint32(batch), 1, 1)
}

func (l *RMSNormLayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(l.bwPipeline)
	pass.SetBindGroup(0, l.bwBindGroup, nil)
	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	pass.DispatchWorkgroups(uint32(batch), 1, 1)
	pass.End()

	// Reduce Pass
	passRed := enc.BeginComputePass(nil)
	passRed.SetPipeline(l.reducePipeline)
	passRed.SetBindGroup(0, l.reduceBindGroup, nil)

	wgx := (l.Spec.NormSize + 255) / 256
	passRed.DispatchWorkgroups(uint32(wgx), 1, 1)
	passRed.End()
}

func (l *RMSNormLayer) UploadWeights(ctx *Context) {
	if len(l.Spec.Gamma) > 0 {
		ctx.Queue.WriteBuffer(l.GammaBuffer, 0, wgpu.ToBytes(l.Spec.Gamma))
	}
}

func (l *RMSNormLayer) ZeroGradients(ctx *Context) {
	sz := len(l.Spec.Gamma)
	if sz == 0 {
		sz = 1
	}
	zeros := make([]float32, sz)
	ctx.Queue.WriteBuffer(l.GammaGradientBuffer, 0, wgpu.ToBytes(zeros))
}

func (l *RMSNormLayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	sz := len(l.Spec.Gamma)
	if sz == 0 {
		sz = 1
	}
	gamma, err := ReadBuffer(l.GammaBuffer, sz)
	if err != nil {
		return nil, nil, err
	}
	return gamma, nil, nil // RMSNorm has no beta
}

func (l *RMSNormLayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	sz := len(l.Spec.Gamma)
	if sz == 0 {
		sz = 1
	}
	gGrad, err := ReadBuffer(l.GammaGradientBuffer, sz)
	if err != nil {
		return nil, nil, nil, err
	}

	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	iGrad, err := ReadBuffer(l.InputGradientBuffer, batch*l.Spec.NormSize)
	if err != nil {
		return nil, nil, nil, err
	}

	return gGrad, nil, iGrad, nil // No beta gradients
}

func (l *RMSNormLayer) Cleanup() {
	if l.InputBuffer != nil {
		l.InputBuffer.Destroy()
	}
	if l.OutputBuffer != nil {
		l.OutputBuffer.Destroy()
	}
	if l.StagingBuffer != nil {
		l.StagingBuffer.Destroy()
	}
	if l.GammaBuffer != nil {
		l.GammaBuffer.Destroy()
	}
	if l.InputGradientBuffer != nil {
		l.InputGradientBuffer.Destroy()
	}
	if l.GammaGradientBuffer != nil {
		l.GammaGradientBuffer.Destroy()
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
