package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// Conv2DSpec defines configuration for 2D Convolution layer
type Conv2DSpec struct {
	InChannels  int       // Input channels
	OutChannels int       // Output channels (filters)
	KernelSize  int       // Kernel size (squared)
	Stride      int       // Stride (default 1)
	Padding     int       // Padding (default 0)
	InputHeight int       // Input height
	InputWidth  int       // Input width
	Weights     []float32 // [OutChannels * InChannels * KernelSize * KernelSize]
	Bias        []float32 // [OutChannels]
}

// Conv2DLayer holds GPU resources for 2D Convolution
type Conv2DLayer struct {
	Spec Conv2DSpec

	pipeline  *wgpu.ComputePipeline
	bindGroup *wgpu.BindGroup

	InputBuffer   *wgpu.Buffer
	OutputBuffer  *wgpu.Buffer
	StagingBuffer *wgpu.Buffer
	WeightBuffer  *wgpu.Buffer
	BiasBuffer    *wgpu.Buffer

	InputGradientBuffer *wgpu.Buffer

	bwPipeline  *wgpu.ComputePipeline
	bwBindGroup *wgpu.BindGroup

	outputH, outputW int
}

func (l *Conv2DLayer) GetInputBuffer() *wgpu.Buffer         { return l.InputBuffer }
func (l *Conv2DLayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *Conv2DLayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *Conv2DLayer) GetInputGradientBuffer() *wgpu.Buffer { return l.InputGradientBuffer }

func (l *Conv2DLayer) computeOutputSize() (int, int) {
	stride := l.Spec.Stride
	if stride < 1 {
		stride = 1
	}
	h := (l.Spec.InputHeight+2*l.Spec.Padding-l.Spec.KernelSize)/stride + 1
	w := (l.Spec.InputWidth+2*l.Spec.Padding-l.Spec.KernelSize)/stride + 1
	return h, w
}

func (l *Conv2DLayer) AllocateBuffers(ctx *Context, labelPrefix string) error {
	var err error

	l.outputH, l.outputW = l.computeOutputSize()
	inputSize := l.Spec.InputHeight * l.Spec.InputWidth * l.Spec.InChannels
	outputSize := l.outputH * l.outputW * l.Spec.OutChannels

	l.InputBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_In",
		Size:  uint64(inputSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	l.OutputBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Out",
		Size:  uint64(outputSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	weightSize := l.Spec.OutChannels * l.Spec.InChannels * l.Spec.KernelSize * l.Spec.KernelSize
	if len(l.Spec.Weights) > 0 {
		l.WeightBuffer, err = NewFloatBuffer(l.Spec.Weights, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	} else {
		placeholder := make([]float32, weightSize)
		l.WeightBuffer, err = NewFloatBuffer(placeholder, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	}
	if err != nil {
		return err
	}

	bias := l.Spec.Bias
	if len(bias) == 0 {
		bias = make([]float32, l.Spec.OutChannels)
	}
	l.BiasBuffer, err = NewFloatBuffer(bias, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
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

func (l *Conv2DLayer) AllocateBackwardBuffers(ctx *Context, labelPrefix string) error {
	var err error
	inputSize := l.Spec.InputHeight * l.Spec.InputWidth * l.Spec.InChannels
	l.InputGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_InGrad",
		Size:  uint64(inputSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	return err
}

func (l *Conv2DLayer) GenerateShader() string {
	stride := l.Spec.Stride
	if stride < 1 {
		stride = 1
	}
	outH, outW := l.computeOutputSize()

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read> weights : array<f32>;
		@group(0) @binding(2) var<storage, read> bias : array<f32>;
		@group(0) @binding(3) var<storage, read_write> output : array<f32>;

		const IN_H: u32 = %du;
		const IN_W: u32 = %du;
		const IN_CH: u32 = %du;
		const OUT_CH: u32 = %du;
		const K: u32 = %du;
		const STRIDE: u32 = %du;
		const PADDING: u32 = %du;
		const OUT_H: u32 = %du;
		const OUT_W: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let total = OUT_H * OUT_W * OUT_CH;
			if (idx >= total) { return; }

			// Output layout: [H, W, C]
			let out_c = idx %% OUT_CH;
			let out_w = (idx / OUT_CH) %% OUT_W;
			let out_h = idx / (OUT_CH * OUT_W);

			var sum: f32 = bias[out_c];

			for (var kh: u32 = 0u; kh < K; kh++) {
				for (var kw: u32 = 0u; kw < K; kw++) {
					let in_h_signed = i32(out_h * STRIDE + kh) - i32(PADDING);
					let in_w_signed = i32(out_w * STRIDE + kw) - i32(PADDING);
					
					if (in_h_signed >= 0 && u32(in_h_signed) < IN_H &&
					    in_w_signed >= 0 && u32(in_w_signed) < IN_W) {
						let in_h = u32(in_h_signed);
						let in_w = u32(in_w_signed);
						
						for (var in_c: u32 = 0u; in_c < IN_CH; in_c++) {
							// Input: [H, W, C]
							let i_idx = in_h * IN_W * IN_CH + in_w * IN_CH + in_c;
							// Weights: [OUT_CH, IN_CH, K, K]
							let w_idx = out_c * IN_CH * K * K + in_c * K * K + kh * K + kw;
							sum += input[i_idx] * weights[w_idx];
						}
					}
				}
			}

			output[idx] = sum;
		}
	`, l.Spec.InputHeight, l.Spec.InputWidth, l.Spec.InChannels, l.Spec.OutChannels,
		l.Spec.KernelSize, stride, l.Spec.Padding, outH, outW)
}

func (l *Conv2DLayer) GenerateBackwardShader() string {
	stride := l.Spec.Stride
	if stride < 1 {
		stride = 1
	}
	outH, outW := l.computeOutputSize()

	// Backward pass: compute dInput via transposed convolution
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> d_output : array<f32>;
		@group(0) @binding(1) var<storage, read> weights : array<f32>;
		@group(0) @binding(2) var<storage, read_write> d_input : array<f32>;

		const IN_H: u32 = %du;
		const IN_W: u32 = %du;
		const IN_CH: u32 = %du;
		const OUT_CH: u32 = %du;
		const K: u32 = %du;
		const STRIDE: u32 = %du;
		const PADDING: u32 = %du;
		const OUT_H: u32 = %du;
		const OUT_W: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let in_total = IN_H * IN_W * IN_CH;
			if (idx >= in_total) { return; }

			// Input layout: [H, W, C]
			let in_c = idx %% IN_CH;
			let in_w = (idx / IN_CH) %% IN_W;
			let in_h = idx / (IN_CH * IN_W);

			var grad: f32 = 0.0;

			// For each output position that this input contributed to
			for (var out_h: u32 = 0u; out_h < OUT_H; out_h++) {
				for (var out_w: u32 = 0u; out_w < OUT_W; out_w++) {
					for (var kh: u32 = 0u; kh < K; kh++) {
						for (var kw: u32 = 0u; kw < K; kw++) {
							let in_h_check = i32(out_h * STRIDE + kh) - i32(PADDING);
							let in_w_check = i32(out_w * STRIDE + kw) - i32(PADDING);
							
							if (u32(in_h_check) == in_h && u32(in_w_check) == in_w &&
							    in_h_check >= 0 && in_w_check >= 0) {
								for (var out_c: u32 = 0u; out_c < OUT_CH; out_c++) {
									// d_output: [OUT_H, OUT_W, OUT_CH]
									let do_idx = out_h * OUT_W * OUT_CH + out_w * OUT_CH + out_c;
									// weights: [OUT_CH, IN_CH, K, K]
									let w_idx = out_c * IN_CH * K * K + in_c * K * K + kh * K + kw;
									grad += d_output[do_idx] * weights[w_idx];
								}
							}
						}
					}
				}
			}

			d_input[idx] = grad;
		}
	`, l.Spec.InputHeight, l.Spec.InputWidth, l.Spec.InChannels, l.Spec.OutChannels,
		l.Spec.KernelSize, stride, l.Spec.Padding, outH, outW)
}

func (l *Conv2DLayer) Compile(ctx *Context, labelPrefix string) error {
	mod, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_Shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateShader()},
	})
	if err != nil {
		return err
	}
	l.pipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_Pipe",
		Compute: wgpu.ProgrammableStageDescriptor{Module: mod, EntryPoint: "main"},
	})
	return err
}

func (l *Conv2DLayer) CompileBackward(ctx *Context, labelPrefix string) error {
	mod, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_BwdShader",
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

func (l *Conv2DLayer) CreateBindGroup(ctx *Context, labelPrefix string) error {
	var err error
	l.bindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_Bind",
		Layout: l.pipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
			{Binding: 1, Buffer: l.WeightBuffer, Size: l.WeightBuffer.GetSize()},
			{Binding: 2, Buffer: l.BiasBuffer, Size: l.BiasBuffer.GetSize()},
			{Binding: 3, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
		},
	})
	return err
}

func (l *Conv2DLayer) CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error {
	var err error
	l.bwBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_BwdBind",
		Layout: l.bwPipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
			{Binding: 1, Buffer: l.WeightBuffer, Size: l.WeightBuffer.GetSize()},
			{Binding: 2, Buffer: l.InputGradientBuffer, Size: l.InputGradientBuffer.GetSize()},
		},
	})
	return err
}

func (l *Conv2DLayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	pass.SetPipeline(l.pipeline)
	pass.SetBindGroup(0, l.bindGroup, nil)
	total := l.outputH * l.outputW * l.Spec.OutChannels
	pass.DispatchWorkgroups(uint32((total+255)/256), 1, 1)
}

func (l *Conv2DLayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(l.bwPipeline)
	pass.SetBindGroup(0, l.bwBindGroup, nil)
	total := l.Spec.InputHeight * l.Spec.InputWidth * l.Spec.InChannels
	pass.DispatchWorkgroups(uint32((total+255)/256), 1, 1)
	pass.End()
}

func (l *Conv2DLayer) UploadWeights(ctx *Context) {
	if len(l.Spec.Weights) > 0 {
		ctx.Queue.WriteBuffer(l.WeightBuffer, 0, wgpu.ToBytes(l.Spec.Weights))
	}
	if len(l.Spec.Bias) > 0 {
		ctx.Queue.WriteBuffer(l.BiasBuffer, 0, wgpu.ToBytes(l.Spec.Bias))
	}
}

func (l *Conv2DLayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	wSize := l.Spec.OutChannels * l.Spec.InChannels * l.Spec.KernelSize * l.Spec.KernelSize
	w, err := ReadBuffer(l.WeightBuffer, wSize)
	if err != nil {
		return nil, nil, err
	}
	b, err := ReadBuffer(l.BiasBuffer, l.Spec.OutChannels)
	return w, b, err
}

func (l *Conv2DLayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	iGrad, err := ReadBuffer(l.InputGradientBuffer, l.Spec.InputHeight*l.Spec.InputWidth*l.Spec.InChannels)
	return nil, nil, iGrad, err
}

func (l *Conv2DLayer) Cleanup() {
	bufs := []*wgpu.Buffer{l.InputBuffer, l.OutputBuffer, l.StagingBuffer, l.WeightBuffer, l.BiasBuffer, l.InputGradientBuffer}
	for _, b := range bufs {
		if b != nil {
			b.Destroy()
		}
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
