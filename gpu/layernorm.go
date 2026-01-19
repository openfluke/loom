package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

type LayerNormSpec struct {
	NormSize  int
	BatchSize int // Number of vectors to normalize (default 1)
	Epsilon   float32
	Gamma     []float32 // [NormSize]
	Beta      []float32 // [NormSize]
}

type LayerNormLayer struct {
	Spec LayerNormSpec

	pipeline  *wgpu.ComputePipeline
	bindGroup *wgpu.BindGroup

	InputBuffer   *wgpu.Buffer
	OutputBuffer  *wgpu.Buffer
	StagingBuffer *wgpu.Buffer
	GammaBuffer   *wgpu.Buffer
	BetaBuffer    *wgpu.Buffer

	// Backward
	GammaGradientBuffer *wgpu.Buffer
	BetaGradientBuffer  *wgpu.Buffer
	InputGradientBuffer *wgpu.Buffer

	// Intermediate buffers for atomic-free reduction
	GammaBatchGradientBuffer *wgpu.Buffer
	BetaBatchGradientBuffer  *wgpu.Buffer

	bwPipeline  *wgpu.ComputePipeline
	bwBindGroup *wgpu.BindGroup

	reducePipeline  *wgpu.ComputePipeline
	reduceBindGroup *wgpu.BindGroup // Binds BatchGrad -> FinalGrad

	WorkgroupsX uint32
}

// Implement GPULayer Interface
func (l *LayerNormLayer) GetInputBuffer() *wgpu.Buffer         { return l.InputBuffer }
func (l *LayerNormLayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *LayerNormLayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *LayerNormLayer) GetInputGradientBuffer() *wgpu.Buffer { return l.InputGradientBuffer }

func (l *LayerNormLayer) AllocateBuffers(ctx *Context, labelPrefix string) error {
	var err error

	// Ensure batch size is at least 1
	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	totalSize := batch * l.Spec.NormSize

	// Input: batch * normSize elements
	l.InputBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_In",
		Size:  uint64(totalSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Output: batch * normSize elements
	l.OutputBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Out",
		Size:  uint64(totalSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Gamma: shared across batch (just normSize)
	if len(l.Spec.Gamma) > 0 {
		l.GammaBuffer, err = NewFloatBuffer(l.Spec.Gamma, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
		if err != nil {
			return err
		}
	} else {
		// Provide dummy 1-element buffer to satisfy binding layout if missing
		l.GammaBuffer, err = NewFloatBuffer([]float32{1.0}, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
		if err != nil {
			return err
		}
	}

	// Beta: shared across batch (just normSize)
	if len(l.Spec.Beta) > 0 {
		l.BetaBuffer, err = NewFloatBuffer(l.Spec.Beta, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
		if err != nil {
			return err
		}
	} else {
		l.BetaBuffer, err = NewFloatBuffer([]float32{0.0}, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
		if err != nil {
			return err
		}
	}

	// Staging: batch * normSize elements
	l.StagingBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Staging",
		Size:  uint64(totalSize * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	return err
}

func (l *LayerNormLayer) AllocateBackwardBuffers(ctx *Context, labelPrefix string) error {
	var err error

	// Ensure batch size is at least 1
	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	totalSize := batch * l.Spec.NormSize

	// Input Grad: batch * normSize elements
	l.InputGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_InGrad",
		Size:  uint64(totalSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Gamma Grad: shared across batch (just normSize)
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

	// Beta Grad: shared across batch (just normSize)
	sz = len(l.Spec.Beta)
	if sz == 0 {
		sz = 1
	}
	l.BetaGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_BetaGrad",
		Size:  uint64(sz * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// NEW: Batch Gradients (Batch * NormSize) for reduction
	// If gamma/beta are scalars per feature (NormSize), then batch grads are (Batch * NormSize)
	batchTotal := batch * l.Spec.NormSize

	l.GammaBatchGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_GammaBatchGrad",
		Size:  uint64(batchTotal * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc, // Written by compute, read by reduce
	})
	if err != nil {
		return err
	}

	l.BetaBatchGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_BetaBatchGrad",
		Size:  uint64(batchTotal * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	return nil
}

func (l *LayerNormLayer) GenerateShader() string {
	// Optimized LayerNorm using workgroup_size(256) with shared memory reductions
	// Each workgroup handles one batch sample
	// 256 threads collaborate to compute mean/variance using parallel reduction

	// Calculate iterations per thread (ceiling division)
	elemsPerThread := (l.Spec.NormSize + 255) / 256

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read_write> output : array<f32>;
		@group(0) @binding(2) var<storage, read> gamma : array<f32>;
		@group(0) @binding(3) var<storage, read> beta : array<f32>;

		const N: u32 = %du;
		const EPS: f32 = %f;
		const ELEMS_PER_THREAD: u32 = %du;

		// Shared memory for parallel reductions
		var<workgroup> shared_sum: array<f32, 256>;
		var<workgroup> shared_sq: array<f32, 256>;
		var<workgroup> wg_mean: f32;
		var<workgroup> wg_invstd: f32;

		@compute @workgroup_size(256)
		fn main(
			@builtin(workgroup_id) wg_id: vec3<u32>,
			@builtin(local_invocation_id) local_id: vec3<u32>
		) {
			let batch = wg_id.x;
			let tid = local_id.x;
			let offset = batch * N;

			// Phase 1: Each thread sums its chunk of elements
			var local_sum: f32 = 0.0;
			for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
				let idx = tid + i * 256u;
				if (idx < N) {
					local_sum += input[offset + idx];
				}
			}
			shared_sum[tid] = local_sum;
			workgroupBarrier();

			// Parallel reduction for sum (log2 steps)
			for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
				if (tid < s) {
					shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
				}
				workgroupBarrier();
			}

			// Thread 0 computes mean
			if (tid == 0u) {
				wg_mean = shared_sum[0] / f32(N);
			}
			workgroupBarrier();

			let mean = wg_mean;

			// Phase 2: Each thread sums squared differences
			var local_sq: f32 = 0.0;
			for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
				let idx = tid + i * 256u;
				if (idx < N) {
					let diff = input[offset + idx] - mean;
					local_sq += diff * diff;
				}
			}
			shared_sq[tid] = local_sq;
			workgroupBarrier();

			// Parallel reduction for variance
			for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
				if (tid < s) {
					shared_sq[tid] = shared_sq[tid] + shared_sq[tid + s];
				}
				workgroupBarrier();
			}

			// Thread 0 computes inverse std
			if (tid == 0u) {
				let variance = shared_sq[0] / f32(N);
				wg_invstd = 1.0 / sqrt(variance + EPS);
			}
			workgroupBarrier();

			let invstd = wg_invstd;

			// Phase 3: All threads normalize their elements in parallel
			for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
				let idx = tid + i * 256u;
				if (idx < N) {
					let val = input[offset + idx];
					let norm = (val - mean) * invstd;
					
					var g: f32 = 1.0;
					if (arrayLength(&gamma) >= N) { g = gamma[idx]; }
					
					var bt: f32 = 0.0;
					if (arrayLength(&beta) >= N) { bt = beta[idx]; }

					output[offset + idx] = norm * g + bt;
				}
			}
		}
	`, l.Spec.NormSize, l.Spec.Epsilon, elemsPerThread)
}

func (l *LayerNormLayer) GenerateBackwardShader() string {
	// Optimized backward pass using workgroup_size(256) with shared memory reductions
	// Each workgroup handles one batch sample.
	// We write ONE gradient per batch sample per feature (Batch * NormSize).
	// Then a reduction shader sums them up. ATOMIC-FREE.

	elemsPerThread := (l.Spec.NormSize + 255) / 256

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read> d_output : array<f32>;
		@group(0) @binding(2) var<storage, read> gamma : array<f32>;
		
		@group(0) @binding(3) var<storage, read_write> d_input : array<f32>;
		@group(0) @binding(4) var<storage, read_write> d_gamma_batch : array<f32>;
		@group(0) @binding(5) var<storage, read_write> d_beta_batch : array<f32>;

		const N: u32 = %du;
		const EPS: f32 = %f;
		const ELEMS_PER_THREAD: u32 = %du;

		// Shared memory for reductions
		var<workgroup> shared_sum: array<f32, 256>;
		var<workgroup> shared_sq: array<f32, 256>;
		var<workgroup> wg_mean: f32;
		var<workgroup> wg_invstd: f32;
		var<workgroup> wg_sum_dxhat: f32;
		var<workgroup> wg_sum_dxhat_xhat: f32;

		@compute @workgroup_size(256)
		fn main(
			@builtin(workgroup_id) wg_id: vec3<u32>,
			@builtin(local_invocation_id) local_id: vec3<u32>
		) {
			let batch = wg_id.x;
			let tid = local_id.x;
			let offset = batch * N;
			
			// Phase 1: Compute mean using parallel reduction
			var local_sum: f32 = 0.0;
			for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
				let idx = tid + i * 256u;
				if (idx < N) {
					local_sum += input[offset + idx];
				}
			}
			shared_sum[tid] = local_sum;
			workgroupBarrier();

			for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
				if (tid < s) {
					shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
				}
				workgroupBarrier();
			}
			if (tid == 0u) { wg_mean = shared_sum[0] / f32(N); }
			workgroupBarrier();
			let mean = wg_mean;

			// Phase 2: Compute variance using parallel reduction
			var local_sq: f32 = 0.0;
			for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
				let idx = tid + i * 256u;
				if (idx < N) {
					let diff = input[offset + idx] - mean;
					local_sq += diff * diff;
				}
			}
			shared_sq[tid] = local_sq;
			workgroupBarrier();

			for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
				if (tid < s) {
					shared_sq[tid] = shared_sq[tid] + shared_sq[tid + s];
				}
				workgroupBarrier();
			}
			if (tid == 0u) {
				let variance = shared_sq[0] / f32(N);
				wg_invstd = 1.0 / sqrt(variance + EPS);
			}
			workgroupBarrier();
			let invStd = wg_invstd;
			
			// Phase 3: Compute d_gamma/d_beta (per-batch storage) and sum_dxhat
			var local_sum_dxhat: f32 = 0.0;
			var local_sum_dxhat_xhat: f32 = 0.0;
			
			for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
				let idx = tid + i * 256u;
				if (idx < N) {
					let val = input[offset + idx];
					let dout = d_output[offset + idx];
					let x_hat = (val - mean) * invStd;
					
					// Store partial gradients directly (No Atomics needed)
					// Each idx is unique per batch.
					let out_idx = offset + idx;
					d_beta_batch[out_idx] = dout;
					d_gamma_batch[out_idx] = dout * x_hat;
					
					var g: f32 = 1.0;
					if (arrayLength(&gamma) >= N) { g = gamma[idx]; }
					let d_xhat = dout * g;
					
					local_sum_dxhat += d_xhat;
					local_sum_dxhat_xhat += d_xhat * x_hat;
				}
			}
			
			// Reduce sum_dxhat
			shared_sum[tid] = local_sum_dxhat;
			workgroupBarrier();
			for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
				if (tid < s) { shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s]; }
				workgroupBarrier();
			}
			if (tid == 0u) { wg_sum_dxhat = shared_sum[0]; }
			workgroupBarrier();
			
			// Reduce sum_dxhat_xhat
			shared_sq[tid] = local_sum_dxhat_xhat;
			workgroupBarrier();
			for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
				if (tid < s) { shared_sq[tid] = shared_sq[tid] + shared_sq[tid + s]; }
				workgroupBarrier();
			}
			if (tid == 0u) { wg_sum_dxhat_xhat = shared_sq[0]; }
			workgroupBarrier();
			
			let sum_dxhat = wg_sum_dxhat;
			let sum_dxhat_xhat = wg_sum_dxhat_xhat;
			
			// Phase 4: Compute d_input
			for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
				let idx = tid + i * 256u;
				if (idx < N) {
					let val = input[offset + idx];
					let x_hat = (val - mean) * invStd;
					
					var g: f32 = 1.0;
					if (arrayLength(&gamma) >= N) { g = gamma[idx]; }
					
					let dout = d_output[offset + idx];
					let d_xhat = dout * g;
					
					let dx = invStd * (d_xhat - (sum_dxhat / f32(N)) - x_hat * (sum_dxhat_xhat / f32(N)));
					d_input[offset + idx] = dx;
				}
			}
		}
	`, l.Spec.NormSize, l.Spec.Epsilon, elemsPerThread)
}

func (l *LayerNormLayer) GenerateReduceShader() string {
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> batch_gamma : array<f32>;
		@group(0) @binding(1) var<storage, read> batch_beta : array<f32>;
		@group(0) @binding(2) var<storage, read_write> final_gamma : array<f32>;
		@group(0) @binding(3) var<storage, read_write> final_beta : array<f32>;

		const N: u32 = %du;
		const BATCH: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			if (idx >= N) { return; }

			var sum_g: f32 = 0.0;
			var sum_b: f32 = 0.0;

			for (var b: u32 = 0u; b < BATCH; b++) {
				let offset = b * N + idx;
				sum_g += batch_gamma[offset];
				sum_b += batch_beta[offset];
			}

			// Accumulate into final gradients (standard Add)
			final_gamma[idx] = final_gamma[idx] + sum_g;
			final_beta[idx]  = final_beta[idx]  + sum_b;
		}
	`, l.Spec.NormSize, l.Spec.BatchSize)
}

func (l *LayerNormLayer) Compile(ctx *Context, labelPrefix string) error {
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
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Beta
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

func (l *LayerNormLayer) CompileBackward(ctx *Context, labelPrefix string) error {
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
			{Binding: 5, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // dBeta
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

	// 2. Reduce Shader
	redShader := l.GenerateReduceShader()
	redModule, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_RedShader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: redShader},
	})
	if err != nil {
		return err
	}
	defer redModule.Release()

	// Layout for Reduce: 2 inputs (batch grads), 2 outputs (final grads)
	bglRed, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: labelPrefix + "_RedBGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // batchG
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // batchB
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // finalG
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // finalB
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

func (l *LayerNormLayer) CreateBindGroup(ctx *Context, labelPrefix string) error {
	var err error

	/*fmt.Printf("[DEBUG] LN CreateBindGroup: In=%d Out=%d Gam=%d Bet=%d\n",
	l.InputBuffer.GetSize(), l.OutputBuffer.GetSize(),
	l.GammaBuffer.GetSize(), l.BetaBuffer.GetSize())*/

	entries := []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
		{Binding: 1, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
		{Binding: 2, Buffer: l.GammaBuffer, Size: l.GammaBuffer.GetSize()},
		{Binding: 3, Buffer: l.BetaBuffer, Size: l.BetaBuffer.GetSize()},
	}
	l.bindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   labelPrefix + "_Bind",
		Layout:  l.pipeline.GetBindGroupLayout(0),
		Entries: entries,
	})
	return err
}

func (l *LayerNormLayer) CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error {
	var err error
	entries := []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
		{Binding: 1, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
		{Binding: 2, Buffer: l.GammaBuffer, Size: l.GammaBuffer.GetSize()},
		{Binding: 3, Buffer: l.InputGradientBuffer, Size: l.InputGradientBuffer.GetSize()},
		{Binding: 4, Buffer: l.GammaBatchGradientBuffer, Size: l.GammaBatchGradientBuffer.GetSize()}, // Batch Grads
		{Binding: 5, Buffer: l.BetaBatchGradientBuffer, Size: l.BetaBatchGradientBuffer.GetSize()},   // Batch Grads
	}
	l.bwBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   labelPrefix + "_BwdBind",
		Layout:  l.bwPipeline.GetBindGroupLayout(0),
		Entries: entries,
	})
	if err != nil {
		return err
	}

	// Create Reduce BindGroup (always strictly internal buffers)
	reduceEntries := []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: l.GammaBatchGradientBuffer, Size: l.GammaBatchGradientBuffer.GetSize()},
		{Binding: 1, Buffer: l.BetaBatchGradientBuffer, Size: l.BetaBatchGradientBuffer.GetSize()},
		{Binding: 2, Buffer: l.GammaGradientBuffer, Size: l.GammaGradientBuffer.GetSize()},
		{Binding: 3, Buffer: l.BetaGradientBuffer, Size: l.BetaGradientBuffer.GetSize()},
	}
	l.reduceBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   labelPrefix + "_RedBind",
		Layout:  l.reducePipeline.GetBindGroupLayout(0),
		Entries: reduceEntries,
	})

	return err
}

func (l *LayerNormLayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	pass.SetPipeline(l.pipeline)
	pass.SetBindGroup(0, l.bindGroup, nil)
	// Launch one workgroup per batch sample
	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	pass.DispatchWorkgroups(uint32(batch), 1, 1)
}

func (l *LayerNormLayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	pass := enc.BeginComputePass(nil)

	// Pass 1: Backward Compute (Partial Gradients)
	pass.SetPipeline(l.bwPipeline)
	pass.SetBindGroup(0, l.bwBindGroup, nil)
	// Launch one workgroup per batch sample
	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	pass.DispatchWorkgroups(uint32(batch), 1, 1)
	pass.End()

	// Pass 2: Reduction (Final Gradients)
	// Dispatch (NormSize + 255) / 256 workgroups to cover all features
	// Each thread reduces over Batch.
	passRed := enc.BeginComputePass(nil)
	passRed.SetPipeline(l.reducePipeline)
	passRed.SetBindGroup(0, l.reduceBindGroup, nil)

	wgx := (l.Spec.NormSize + 255) / 256
	passRed.DispatchWorkgroups(uint32(wgx), 1, 1)
	passRed.End()
}

func (l *LayerNormLayer) UploadWeights(ctx *Context) {
	if len(l.Spec.Gamma) > 0 {
		ctx.Queue.WriteBuffer(l.GammaBuffer, 0, wgpu.ToBytes(l.Spec.Gamma))
	}
	if len(l.Spec.Beta) > 0 {
		ctx.Queue.WriteBuffer(l.BetaBuffer, 0, wgpu.ToBytes(l.Spec.Beta))
	}
}

// ZeroGradients zeroes gradient buffers before backward pass (required for atomic accumulation)
func (l *LayerNormLayer) ZeroGradients(ctx *Context) {
	// Zero gamma gradient buffer
	sz := len(l.Spec.Gamma)
	if sz == 0 {
		sz = 1
	}
	zeros := make([]float32, sz)
	ctx.Queue.WriteBuffer(l.GammaGradientBuffer, 0, wgpu.ToBytes(zeros))

	// Zero beta gradient buffer
	sz = len(l.Spec.Beta)
	if sz == 0 {
		sz = 1
	}
	zeros = make([]float32, sz)
	ctx.Queue.WriteBuffer(l.BetaGradientBuffer, 0, wgpu.ToBytes(zeros))
}

func (l *LayerNormLayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	// Gamma
	sz := len(l.Spec.Gamma)
	if sz == 0 {
		sz = 1
	}
	gamma, err := ReadBuffer(l.GammaBuffer, sz)
	if err != nil {
		return nil, nil, err
	}

	// Beta
	sz = len(l.Spec.Beta)
	if sz == 0 {
		sz = 1
	}
	beta, err := ReadBuffer(l.BetaBuffer, sz)
	if err != nil {
		return nil, nil, err
	}

	return gamma, beta, nil
}

func (l *LayerNormLayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	// Verify signatures match main.go expectation
	// wGrad, bGrad, iGrad

	// Gamma Grad
	sz := len(l.Spec.Gamma)
	if sz == 0 {
		sz = 1
	}
	gGrad, err := ReadBuffer(l.GammaGradientBuffer, sz)
	if err != nil {
		return nil, nil, nil, err
	}

	// Beta Grad
	sz = len(l.Spec.Beta)
	if sz == 0 {
		sz = 1
	}
	bGrad, err := ReadBuffer(l.BetaGradientBuffer, sz)
	if err != nil {
		return nil, nil, nil, err
	}

	// Input Grad
	iGrad, err := ReadBuffer(l.InputGradientBuffer, l.Spec.NormSize) // Batch=1 assumed
	if err != nil {
		return nil, nil, nil, err
	}

	return gGrad, bGrad, iGrad, nil
}

func (l *LayerNormLayer) Cleanup() {
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
	if l.BetaBuffer != nil {
		l.BetaBuffer.Destroy()
	}
	if l.InputGradientBuffer != nil {
		l.InputGradientBuffer.Destroy()
	}
	if l.GammaGradientBuffer != nil {
		l.GammaGradientBuffer.Destroy()
	}
	if l.BetaGradientBuffer != nil {
		l.BetaGradientBuffer.Destroy()
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
