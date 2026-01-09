package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// ResidualSpec defines configuration for Residual (skip connection) layer
type ResidualSpec struct {
	Size int // Number of elements
}

// ResidualLayer holds GPU resources for Residual addition
type ResidualLayer struct {
	Spec ResidualSpec

	pipeline  *wgpu.ComputePipeline
	bindGroup *wgpu.BindGroup

	InputBuffer   *wgpu.Buffer // Primary input
	SkipBuffer    *wgpu.Buffer // Residual/skip input
	OutputBuffer  *wgpu.Buffer
	StagingBuffer *wgpu.Buffer

	// Backward
	InputGradientBuffer *wgpu.Buffer
	SkipGradientBuffer  *wgpu.Buffer

	bwPipeline  *wgpu.ComputePipeline
	bwBindGroup *wgpu.BindGroup
}

// Interface implementations
func (l *ResidualLayer) GetInputBuffer() *wgpu.Buffer         { return l.InputBuffer }
func (l *ResidualLayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *ResidualLayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *ResidualLayer) GetInputGradientBuffer() *wgpu.Buffer { return l.InputGradientBuffer }
func (l *ResidualLayer) GetSkipGradientBuffer() *wgpu.Buffer  { return l.SkipGradientBuffer }

func (l *ResidualLayer) AllocateBuffers(ctx *Context, labelPrefix string) error {
	var err error
	size := l.Spec.Size

	l.InputBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_In",
		Size:  uint64(size * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	l.SkipBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Skip",
		Size:  uint64(size * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	l.OutputBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Out",
		Size:  uint64(size * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	l.StagingBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Staging",
		Size:  uint64(size * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	return err
}

func (l *ResidualLayer) AllocateBackwardBuffers(ctx *Context, labelPrefix string) error {
	var err error
	size := l.Spec.Size

	l.InputGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_InGrad",
		Size:  uint64(size * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	l.SkipGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_SkipGrad",
		Size:  uint64(size * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	return err
}

func (l *ResidualLayer) GenerateShader() string {
	// Simple element-wise addition: output = input + skip
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read> skip : array<f32>;
		@group(0) @binding(2) var<storage, read_write> output : array<f32>;

		const SIZE: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			if (idx >= SIZE) { return; }
			output[idx] = input[idx] + skip[idx];
		}
	`, l.Spec.Size)
}

func (l *ResidualLayer) GenerateBackwardShader() string {
	// Backward: d_input = d_output, d_skip = d_output (gradient flows to both)
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> d_output : array<f32>;
		@group(0) @binding(1) var<storage, read_write> d_input : array<f32>;
		@group(0) @binding(2) var<storage, read_write> d_skip : array<f32>;

		const SIZE: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			if (idx >= SIZE) { return; }
			d_input[idx] = d_output[idx];
			d_skip[idx] = d_output[idx];
		}
	`, l.Spec.Size)
}

func (l *ResidualLayer) Compile(ctx *Context, labelPrefix string) error {
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

func (l *ResidualLayer) CompileBackward(ctx *Context, labelPrefix string) error {
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

func (l *ResidualLayer) CreateBindGroup(ctx *Context, labelPrefix string) error {
	var err error
	entries := []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
		{Binding: 1, Buffer: l.SkipBuffer, Size: l.SkipBuffer.GetSize()},
		{Binding: 2, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
	}
	l.bindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   labelPrefix + "_Bind",
		Layout:  l.pipeline.GetBindGroupLayout(0),
		Entries: entries,
	})
	return err
}

func (l *ResidualLayer) CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error {
	var err error
	entries := []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
		{Binding: 1, Buffer: l.InputGradientBuffer, Size: l.InputGradientBuffer.GetSize()},
		{Binding: 2, Buffer: l.SkipGradientBuffer, Size: l.SkipGradientBuffer.GetSize()},
	}
	l.bwBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   labelPrefix + "_BwdBind",
		Layout:  l.bwPipeline.GetBindGroupLayout(0),
		Entries: entries,
	})
	return err
}

func (l *ResidualLayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	pass.SetPipeline(l.pipeline)
	pass.SetBindGroup(0, l.bindGroup, nil)
	wgx := (l.Spec.Size + 255) / 256
	pass.DispatchWorkgroups(uint32(wgx), 1, 1)
}

func (l *ResidualLayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(l.bwPipeline)
	pass.SetBindGroup(0, l.bwBindGroup, nil)
	wgx := (l.Spec.Size + 255) / 256
	pass.DispatchWorkgroups(uint32(wgx), 1, 1)
	pass.End()
}

func (l *ResidualLayer) UploadWeights(ctx *Context) {
	// Residual has no learnable weights
}

func (l *ResidualLayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	return nil, nil, nil
}

func (l *ResidualLayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	iGrad, err := ReadBuffer(l.InputGradientBuffer, l.Spec.Size)
	return nil, nil, iGrad, err
}

func (l *ResidualLayer) Cleanup() {
	if l.InputBuffer != nil {
		l.InputBuffer.Destroy()
	}
	if l.SkipBuffer != nil {
		l.SkipBuffer.Destroy()
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
	if l.SkipGradientBuffer != nil {
		l.SkipGradientBuffer.Destroy()
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
