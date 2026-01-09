package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// SoftmaxSpec defines configuration for Softmax layer
type SoftmaxSpec struct {
	Size        int     // Number of elements per softmax operation
	BatchSize   int     // Number of independent softmax operations
	Temperature float32 // Temperature scaling (default 1.0)
}

// SoftmaxLayer holds GPU resources for Softmax
type SoftmaxLayer struct {
	Spec SoftmaxSpec

	pipeline  *wgpu.ComputePipeline
	bindGroup *wgpu.BindGroup

	InputBuffer   *wgpu.Buffer
	OutputBuffer  *wgpu.Buffer
	StagingBuffer *wgpu.Buffer

	// Backward
	InputGradientBuffer *wgpu.Buffer

	bwPipeline  *wgpu.ComputePipeline
	bwBindGroup *wgpu.BindGroup
}

// Interface implementations
func (l *SoftmaxLayer) GetInputBuffer() *wgpu.Buffer         { return l.InputBuffer }
func (l *SoftmaxLayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *SoftmaxLayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *SoftmaxLayer) GetInputGradientBuffer() *wgpu.Buffer { return l.InputGradientBuffer }

func (l *SoftmaxLayer) AllocateBuffers(ctx *Context, labelPrefix string) error {
	var err error

	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	totalSize := batch * l.Spec.Size

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

	l.StagingBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Staging",
		Size:  uint64(totalSize * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	return err
}

func (l *SoftmaxLayer) AllocateBackwardBuffers(ctx *Context, labelPrefix string) error {
	var err error

	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	totalSize := batch * l.Spec.Size

	l.InputGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_InGrad",
		Size:  uint64(totalSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	return err
}

func (l *SoftmaxLayer) GenerateShader() string {
	// Softmax: y_i = exp(x_i - max) / sum(exp(x_j - max))
	// Uses parallel reduction for max and sum

	elemsPerThread := (l.Spec.Size + 255) / 256
	temp := l.Spec.Temperature
	if temp <= 0 {
		temp = 1.0
	}

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read_write> output : array<f32>;

		const N: u32 = %du;
		const TEMP: f32 = %f;
		const ELEMS_PER_THREAD: u32 = %du;

		var<workgroup> shared_val: array<f32, 256>;
		var<workgroup> wg_max: f32;
		var<workgroup> wg_sum: f32;

		@compute @workgroup_size(256)
		fn main(
			@builtin(workgroup_id) wg_id: vec3<u32>,
			@builtin(local_invocation_id) local_id: vec3<u32>
		) {
			let batch = wg_id.x;
			let tid = local_id.x;
			let offset = batch * N;

			// Phase 1: Find max for numerical stability
			var local_max: f32 = -1e38;
			for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
				let idx = tid + i * 256u;
				if (idx < N) {
					local_max = max(local_max, input[offset + idx] / TEMP);
				}
			}
			shared_val[tid] = local_max;
			workgroupBarrier();

			// Parallel reduction for max
			for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
				if (tid < s) {
					shared_val[tid] = max(shared_val[tid], shared_val[tid + s]);
				}
				workgroupBarrier();
			}
			if (tid == 0u) { wg_max = shared_val[0]; }
			workgroupBarrier();
			let max_val = wg_max;

			// Phase 2: Compute sum of exp(x - max)
			var local_sum: f32 = 0.0;
			for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
				let idx = tid + i * 256u;
				if (idx < N) {
					local_sum += exp(input[offset + idx] / TEMP - max_val);
				}
			}
			shared_val[tid] = local_sum;
			workgroupBarrier();

			// Parallel reduction for sum
			for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
				if (tid < s) {
					shared_val[tid] = shared_val[tid] + shared_val[tid + s];
				}
				workgroupBarrier();
			}
			if (tid == 0u) { wg_sum = shared_val[0]; }
			workgroupBarrier();
			let sum_exp = wg_sum;

			// Phase 3: Compute softmax output
			for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
				let idx = tid + i * 256u;
				if (idx < N) {
					output[offset + idx] = exp(input[offset + idx] / TEMP - max_val) / sum_exp;
				}
			}
		}
	`, l.Spec.Size, temp, elemsPerThread)
}

func (l *SoftmaxLayer) GenerateBackwardShader() string {
	// Softmax backward: d_input = output * (d_output - sum(d_output * output))
	elemsPerThread := (l.Spec.Size + 255) / 256

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> output : array<f32>;
		@group(0) @binding(1) var<storage, read> d_output : array<f32>;
		@group(0) @binding(2) var<storage, read_write> d_input : array<f32>;

		const N: u32 = %du;
		const ELEMS_PER_THREAD: u32 = %du;

		var<workgroup> shared_val: array<f32, 256>;
		var<workgroup> wg_sum_dy_y: f32;

		@compute @workgroup_size(256)
		fn main(
			@builtin(workgroup_id) wg_id: vec3<u32>,
			@builtin(local_invocation_id) local_id: vec3<u32>
		) {
			let batch = wg_id.x;
			let tid = local_id.x;
			let offset = batch * N;

			// Phase 1: Compute sum(d_output * output)
			var local_sum: f32 = 0.0;
			for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
				let idx = tid + i * 256u;
				if (idx < N) {
					local_sum += d_output[offset + idx] * output[offset + idx];
				}
			}
			shared_val[tid] = local_sum;
			workgroupBarrier();

			for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
				if (tid < s) { shared_val[tid] = shared_val[tid] + shared_val[tid + s]; }
				workgroupBarrier();
			}
			if (tid == 0u) { wg_sum_dy_y = shared_val[0]; }
			workgroupBarrier();
			let sum_dy_y = wg_sum_dy_y;

			// Phase 2: Compute d_input
			for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
				let idx = tid + i * 256u;
				if (idx < N) {
					let y = output[offset + idx];
					let dy = d_output[offset + idx];
					d_input[offset + idx] = y * (dy - sum_dy_y);
				}
			}
		}
	`, l.Spec.Size, elemsPerThread)
}

func (l *SoftmaxLayer) Compile(ctx *Context, labelPrefix string) error {
	shader := l.GenerateShader()
	module, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_Shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shader},
	})
	if err != nil {
		return err
	}

	l.pipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_Pipe",
		Compute: wgpu.ProgrammableStageDescriptor{Module: module, EntryPoint: "main"},
	})
	return err
}

func (l *SoftmaxLayer) CompileBackward(ctx *Context, labelPrefix string) error {
	shader := l.GenerateBackwardShader()
	module, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_BwdShader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shader},
	})
	if err != nil {
		return err
	}

	l.bwPipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_BwdPipe",
		Compute: wgpu.ProgrammableStageDescriptor{Module: module, EntryPoint: "main"},
	})
	return err
}

func (l *SoftmaxLayer) CreateBindGroup(ctx *Context, labelPrefix string) error {
	var err error
	entries := []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
		{Binding: 1, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
	}
	l.bindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   labelPrefix + "_Bind",
		Layout:  l.pipeline.GetBindGroupLayout(0),
		Entries: entries,
	})
	return err
}

func (l *SoftmaxLayer) CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error {
	var err error
	entries := []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
		{Binding: 1, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
		{Binding: 2, Buffer: l.InputGradientBuffer, Size: l.InputGradientBuffer.GetSize()},
	}
	l.bwBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   labelPrefix + "_BwdBind",
		Layout:  l.bwPipeline.GetBindGroupLayout(0),
		Entries: entries,
	})
	return err
}

func (l *SoftmaxLayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	pass.SetPipeline(l.pipeline)
	pass.SetBindGroup(0, l.bindGroup, nil)
	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	pass.DispatchWorkgroups(uint32(batch), 1, 1)
}

func (l *SoftmaxLayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(l.bwPipeline)
	pass.SetBindGroup(0, l.bwBindGroup, nil)
	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	pass.DispatchWorkgroups(uint32(batch), 1, 1)
	pass.End()
}

func (l *SoftmaxLayer) UploadWeights(ctx *Context) {
	// Softmax has no learnable weights
}

func (l *SoftmaxLayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	return nil, nil, nil // Softmax has no weights
}

func (l *SoftmaxLayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	iGrad, err := ReadBuffer(l.InputGradientBuffer, batch*l.Spec.Size)
	return nil, nil, iGrad, err
}

func (l *SoftmaxLayer) Cleanup() {
	if l.InputBuffer != nil {
		l.InputBuffer.Destroy()
	}
	if l.OutputBuffer != nil {
		l.OutputBuffer.Destroy()
	}
	if l.StagingBuffer != nil {
		l.StagingBuffer.Destroy()
	}
	if l.InputGradientBuffer != nil {
		l.InputGradientBuffer.Destroy()
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
