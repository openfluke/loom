package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// Conv1DSpec defines configuration for 1D Convolution layer
type Conv1DSpec struct {
	InChannels  int       // Input channels
	OutChannels int       // Output channels (filters)
	KernelSize  int       // Kernel/filter size
	Stride      int       // Stride (default 1)
	Padding     int       // Padding (default 0)
	SeqLen      int       // Input sequence length
	Weights     []float32 // [OutChannels * InChannels * KernelSize]
	Bias        []float32 // [OutChannels]
	Activation  string    // "relu", "sigmoid", etc.
}

// Conv1DLayer holds GPU resources for 1D Convolution
type Conv1DLayer struct {
	Spec Conv1DSpec

	pipeline  *wgpu.ComputePipeline
	bindGroup *wgpu.BindGroup

	InputBuffer   *wgpu.Buffer
	OutputBuffer  *wgpu.Buffer
	StagingBuffer *wgpu.Buffer
	WeightBuffer  *wgpu.Buffer
	BiasBuffer    *wgpu.Buffer

	InputGradientBuffer  *wgpu.Buffer
	WeightGradientBuffer *wgpu.Buffer
	BiasGradientBuffer   *wgpu.Buffer

	bwPipeline  *wgpu.ComputePipeline
	bwBindGroup *wgpu.BindGroup

	outputLen int
}

func (l *Conv1DLayer) GetInputBuffer() *wgpu.Buffer         { return l.InputBuffer }
func (l *Conv1DLayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *Conv1DLayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *Conv1DLayer) GetInputGradientBuffer() *wgpu.Buffer { return l.InputGradientBuffer }

func (l *Conv1DLayer) computeOutputLen() int {
	stride := l.Spec.Stride
	if stride < 1 {
		stride = 1
	}
	return (l.Spec.SeqLen+2*l.Spec.Padding-l.Spec.KernelSize)/stride + 1
}

func (l *Conv1DLayer) AllocateBuffers(ctx *Context, labelPrefix string) error {
	var err error

	// Validate parameters
	if l.Spec.SeqLen <= 0 {
		l.Spec.SeqLen = 32 // Default sequence length
	}
	if l.Spec.InChannels <= 0 {
		l.Spec.InChannels = 1
	}
	if l.Spec.OutChannels <= 0 {
		l.Spec.OutChannels = 1
	}
	if l.Spec.KernelSize <= 0 {
		l.Spec.KernelSize = 3
	}
	if l.Spec.Stride <= 0 {
		l.Spec.Stride = 1
	}

	l.outputLen = l.computeOutputLen()
	if l.outputLen <= 0 {
		l.outputLen = 1
	}

	inputSize := l.Spec.SeqLen * l.Spec.InChannels
	if inputSize < 1 {
		inputSize = 1
	}
	outputSize := l.outputLen * l.Spec.OutChannels
	if outputSize < 1 {
		outputSize = 1
	}

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

	weightSize := l.Spec.OutChannels * l.Spec.InChannels * l.Spec.KernelSize
	if weightSize < 1 {
		weightSize = 1
	}

	// Ensure weights array matches expected size
	if len(l.Spec.Weights) != weightSize {
		l.Spec.Weights = make([]float32, weightSize)
		for i := range l.Spec.Weights {
			l.Spec.Weights[i] = float32(i%100) * 0.01
		}
	}
	l.WeightBuffer, err = NewFloatBuffer(l.Spec.Weights, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
	if err != nil {
		return err
	}

	// Ensure bias array matches expected size
	if len(l.Spec.Bias) != l.Spec.OutChannels {
		l.Spec.Bias = make([]float32, l.Spec.OutChannels)
	}
	l.BiasBuffer, err = NewFloatBuffer(l.Spec.Bias, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
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

func (l *Conv1DLayer) AllocateBackwardBuffers(ctx *Context, labelPrefix string) error {
	var err error

	inputSize := l.Spec.SeqLen * l.Spec.InChannels
	l.InputGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_InGrad",
		Size:  uint64(inputSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	weightSize := l.Spec.OutChannels * l.Spec.InChannels * l.Spec.KernelSize
	l.WeightGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_WGrad",
		Size:  uint64(weightSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	l.BiasGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_BGrad",
		Size:  uint64(l.Spec.OutChannels * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	return err
}

func (l *Conv1DLayer) getActivationCode(varName string) string {
	switch l.Spec.Activation {
	case "relu":
		// Match CPU "ScaledReLU" behavior (1.1x scaling)
		return fmt.Sprintf("max(%s * 1.1, 0.0)", varName)
	case "sigmoid":
		return fmt.Sprintf("1.0 / (1.0 + exp(-%s))", varName)
	case "tanh":
		return fmt.Sprintf("tanh(%s)", varName)
	case "leaky_relu":
		return fmt.Sprintf("select(%s, %s * 0.01, %s < 0.0)", varName, varName, varName)
	default:
		return varName
	}
}

func (l *Conv1DLayer) GenerateShader() string {
	stride := l.Spec.Stride
	if stride < 1 {
		stride = 1
	}
	outputLen := l.computeOutputLen()

	// CPU layout: input[ic * seqLen + pos], output[f * outLen + o]
	// Weight layout: kernel[f * inChannels * kernelSize + ic * kernelSize + k]
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read> weights : array<f32>;
		@group(0) @binding(2) var<storage, read> bias : array<f32>;
		@group(0) @binding(3) var<storage, read_write> output : array<f32>;

		const SEQ_LEN: u32 = %du;
		const IN_CH: u32 = %du;
		const OUT_CH: u32 = %du;
		const KERNEL_SIZE: u32 = %du;
		const STRIDE: u32 = %du;
		const PADDING: u32 = %du;
		const OUT_LEN: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let total = OUT_LEN * OUT_CH;
			if (idx >= total) { return; }

			// Output layout: [filters][outLen] (CPU-compatible)
			let out_c = idx / OUT_LEN;
			let out_pos = idx %% OUT_LEN;

			var sum: f32 = bias[out_c];
			
			for (var k: u32 = 0u; k < KERNEL_SIZE; k++) {
				let in_pos_signed = i32(out_pos * STRIDE + k) - i32(PADDING);
				if (in_pos_signed >= 0 && u32(in_pos_signed) < SEQ_LEN) {
					let in_pos = u32(in_pos_signed);
					for (var in_c: u32 = 0u; in_c < IN_CH; in_c++) {
						// Weight layout: [out_c][in_c][k]
						let w_idx = out_c * IN_CH * KERNEL_SIZE + in_c * KERNEL_SIZE + k;
						// Input layout: [in_c][seqLen] (CPU-compatible)
						let i_idx = in_c * SEQ_LEN + in_pos;
						sum += input[i_idx] * weights[w_idx];
					}
				}
			}

			output[idx] = %s;
		}
	`, l.Spec.SeqLen, l.Spec.InChannels, l.Spec.OutChannels, l.Spec.KernelSize, stride, l.Spec.Padding, outputLen, l.getActivationCode("sum"))
}

func (l *Conv1DLayer) GenerateBackwardShader() string {
	stride := l.Spec.Stride
	if stride < 1 {
		stride = 1
	}
	outputLen := l.computeOutputLen()

	// Backward pass computes:
	// 1. dInput via transposed convolution
	// 2. dWeights and dBias via accumulation (using atomics)
	// CPU layout: input[in_c * seqLen + pos], output[out_c * outLen + pos]
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> d_output : array<f32>;
		@group(0) @binding(1) var<storage, read> output : array<f32>; // Added for activation derivative
		@group(0) @binding(2) var<storage, read> input : array<f32>;
		@group(0) @binding(3) var<storage, read> weights : array<f32>;
		@group(0) @binding(4) var<storage, read_write> d_input : array<f32>;
		@group(0) @binding(5) var<storage, read_write> d_weights : array<atomic<u32>>;
		@group(0) @binding(6) var<storage, read_write> d_bias : array<atomic<u32>>;

		const SEQ_LEN: u32 = %du;
		const IN_CH: u32 = %du;
		const OUT_CH: u32 = %du;
		const KERNEL_SIZE: u32 = %du;
		const STRIDE: u32 = %du;
		const PADDING: u32 = %du;
		const OUT_LEN: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let in_total = SEQ_LEN * IN_CH;
			if (idx >= in_total) { return; }

			// Input/dInput layout: [in_c][seqLen] (CPU-compatible)
			let in_c = idx / SEQ_LEN;
			let in_pos = idx %% SEQ_LEN;

			// Compute d_input[in_c, in_pos]
			var grad: f32 = 0.0;

			// For each output position that this input contributed to
			for (var out_pos: u32 = 0u; out_pos < OUT_LEN; out_pos++) {
				for (var k: u32 = 0u; k < KERNEL_SIZE; k++) {
					let in_pos_check = i32(out_pos * STRIDE + k) - i32(PADDING);
					if (u32(in_pos_check) == in_pos && in_pos_check >= 0) {
						for (var out_c: u32 = 0u; out_c < OUT_CH; out_c++) {
							let w_idx = out_c * IN_CH * KERNEL_SIZE + in_c * KERNEL_SIZE + k;
							// Output layout: [out_c][outLen] (CPU-compatible)
							let do_idx = out_c * OUT_LEN + out_pos;
							// Apply activation derivative: d_act = 1.1 if output[do_idx] > 0 else 0
							let out_val = output[do_idx];
							var d_act: f32 = 0.0;
							if (out_val > 0.0) { d_act = 1.1; }

							grad += d_output[do_idx] * d_act * weights[w_idx];
						}
					}
				}
			}
			d_input[idx] = grad;

			// Also accumulate weight gradients (each thread contributes)
			// For efficiency, we only accumulate for one weight per thread
			if (in_pos < OUT_LEN) {
				let out_pos = in_pos;
				for (var k: u32 = 0u; k < KERNEL_SIZE; k++) {
					let src_pos = i32(out_pos * STRIDE + k) - i32(PADDING);
					if (src_pos >= 0 && u32(src_pos) < SEQ_LEN) {
						for (var out_c: u32 = 0u; out_c < OUT_CH; out_c++) {
							let w_idx = out_c * IN_CH * KERNEL_SIZE + in_c * KERNEL_SIZE + k;
							// Output layout: [out_c][outLen]
							let do_idx = out_c * OUT_LEN + out_pos;
							
							// Activation derivative
							let out_val = output[do_idx];
							var d_act: f32 = 0.0;
							if (out_val > 0.0) { d_act = 1.1; }

							// Input layout: [in_c][seqLen]
							let contrib = d_output[do_idx] * d_act * input[in_c * SEQ_LEN + u32(src_pos)];
							
							var old_val: u32 = atomicLoad(&d_weights[w_idx]);
							loop {
								let old_f32 = bitcast<f32>(old_val);
								let new_f32 = old_f32 + contrib;
								let new_val = bitcast<u32>(new_f32);
								let result = atomicCompareExchangeWeak(&d_weights[w_idx], old_val, new_val);
								if (result.exchanged) { break; }
								old_val = result.old_value;
							}
						}
					}
				}

				// Bias gradient
				if (in_c == 0u) {
					for (var out_c: u32 = 0u; out_c < OUT_CH; out_c++) {
						// Output layout: [out_c][outLen]
						let do_idx = out_c * OUT_LEN + out_pos;
						// Activation derivative
						let out_val = output[do_idx];
						var d_act: f32 = 0.0;
						if (out_val > 0.0) { d_act = 1.1; }

						let bias_contrib = d_output[do_idx] * d_act;
						var old_val: u32 = atomicLoad(&d_bias[out_c]);
						loop {
							let old_f32 = bitcast<f32>(old_val);
							let new_f32 = old_f32 + bias_contrib;
							let new_val = bitcast<u32>(new_f32);
							let result = atomicCompareExchangeWeak(&d_bias[out_c], old_val, new_val);
							if (result.exchanged) { break; }
							old_val = result.old_value;
						}
					}
				}
			}
		}
	`, l.Spec.SeqLen, l.Spec.InChannels, l.Spec.OutChannels, l.Spec.KernelSize, stride, l.Spec.Padding, outputLen)
}

func (l *Conv1DLayer) Compile(ctx *Context, labelPrefix string) error {
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

func (l *Conv1DLayer) CompileBackward(ctx *Context, labelPrefix string) error {
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

func (l *Conv1DLayer) CreateBindGroup(ctx *Context, labelPrefix string) error {
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

func (l *Conv1DLayer) CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error {
	var err error
	l.bwBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_BwdBind",
		Layout: l.bwPipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
			{Binding: 1, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
			{Binding: 2, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
			{Binding: 3, Buffer: l.WeightBuffer, Size: l.WeightBuffer.GetSize()},
			{Binding: 4, Buffer: l.InputGradientBuffer, Size: l.InputGradientBuffer.GetSize()},
			{Binding: 5, Buffer: l.WeightGradientBuffer, Size: l.WeightGradientBuffer.GetSize()},
			{Binding: 6, Buffer: l.BiasGradientBuffer, Size: l.BiasGradientBuffer.GetSize()},
		},
	})
	return err
}

func (l *Conv1DLayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	pass.SetPipeline(l.pipeline)
	pass.SetBindGroup(0, l.bindGroup, nil)
	total := l.outputLen * l.Spec.OutChannels
	pass.DispatchWorkgroups(uint32((total+255)/256), 1, 1)
}

func (l *Conv1DLayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(l.bwPipeline)
	pass.SetBindGroup(0, l.bwBindGroup, nil)
	total := l.Spec.SeqLen * l.Spec.InChannels
	pass.DispatchWorkgroups(uint32((total+255)/256), 1, 1)
	pass.End()
}

func (l *Conv1DLayer) UploadWeights(ctx *Context) {
	if len(l.Spec.Weights) > 0 {
		ctx.Queue.WriteBuffer(l.WeightBuffer, 0, wgpu.ToBytes(l.Spec.Weights))
	}
	if len(l.Spec.Bias) > 0 {
		ctx.Queue.WriteBuffer(l.BiasBuffer, 0, wgpu.ToBytes(l.Spec.Bias))
	}
}

func (l *Conv1DLayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	wSize := l.Spec.OutChannels * l.Spec.InChannels * l.Spec.KernelSize
	w, err := ReadBuffer(l.WeightBuffer, wSize)
	if err != nil {
		return nil, nil, err
	}
	b, err := ReadBuffer(l.BiasBuffer, l.Spec.OutChannels)
	return w, b, err
}

func (l *Conv1DLayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	wSize := l.Spec.OutChannels * l.Spec.InChannels * l.Spec.KernelSize
	wGrad, err := ReadBuffer(l.WeightGradientBuffer, wSize)
	if err != nil {
		return nil, nil, nil, err
	}
	bGrad, err := ReadBuffer(l.BiasGradientBuffer, l.Spec.OutChannels)
	if err != nil {
		return nil, nil, nil, err
	}
	iGrad, err := ReadBuffer(l.InputGradientBuffer, l.Spec.SeqLen*l.Spec.InChannels)
	return wGrad, bGrad, iGrad, err
}

func (l *Conv1DLayer) Cleanup() {
	bufs := []*wgpu.Buffer{l.InputBuffer, l.OutputBuffer, l.StagingBuffer, l.WeightBuffer, l.BiasBuffer, l.InputGradientBuffer, l.WeightGradientBuffer, l.BiasGradientBuffer}
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
