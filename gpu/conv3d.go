package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// Conv3DSpec defines configuration for 3D Convolution layer
type Conv3DSpec struct {
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int
	InputDepth  int
	InputHeight int
	InputWidth  int
	Weights     []float32 // [OutChannels * InChannels * KernelSize * KernelSize * KernelSize]
	Bias        []float32 // [OutChannels]
	Activation  string
}

// Conv3DLayer holds GPU resources for 3D Convolution
type Conv3DLayer struct {
	Spec Conv3DSpec

	BatchSize int

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

	bwPipeline      *wgpu.ComputePipeline
	bwBindGroup     *wgpu.BindGroup
	bwGradPipeline  *wgpu.ComputePipeline
	bwGradBindGroup *wgpu.BindGroup

	outputD, outputH, outputW int
	InputAliased              bool
}

func (l *Conv3DLayer) GetInputBuffer() *wgpu.Buffer         { return l.InputBuffer }
func (l *Conv3DLayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *Conv3DLayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *Conv3DLayer) GetInputGradientBuffer() *wgpu.Buffer { return l.InputGradientBuffer }

func (l *Conv3DLayer) SetInputBuffer(buf *wgpu.Buffer) {
	l.InputBuffer = buf
	l.InputAliased = true
}

func (l *Conv3DLayer) computeOutputSize() (int, int, int) {
	stride := l.Spec.Stride
	if stride < 1 {
		stride = 1
	}
	d := (l.Spec.InputDepth+2*l.Spec.Padding-l.Spec.KernelSize)/stride + 1
	h := (l.Spec.InputHeight+2*l.Spec.Padding-l.Spec.KernelSize)/stride + 1
	w := (l.Spec.InputWidth+2*l.Spec.Padding-l.Spec.KernelSize)/stride + 1
	return d, h, w
}

func (l *Conv3DLayer) AllocateBuffers(ctx *Context, labelPrefix string) error {
	var err error

	if l.Spec.InputDepth <= 0 {
		l.Spec.InputDepth = 4
	}
	if l.Spec.InputHeight <= 0 {
		l.Spec.InputHeight = 8
	}
	if l.Spec.InputWidth <= 0 {
		l.Spec.InputWidth = 8
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
	if l.BatchSize <= 0 {
		l.BatchSize = 1
	}

	l.outputD, l.outputH, l.outputW = l.computeOutputSize()
	if l.outputD <= 0 {
		l.outputD = 1
	}
	if l.outputH <= 0 {
		l.outputH = 1
	}
	if l.outputW <= 0 {
		l.outputW = 1
	}

	inputSize := l.Spec.InputDepth * l.Spec.InputHeight * l.Spec.InputWidth * l.Spec.InChannels * l.BatchSize
	if inputSize < 1 {
		inputSize = 1
	}
	outputSize := l.outputD * l.outputH * l.outputW * l.Spec.OutChannels * l.BatchSize
	if outputSize < 1 {
		outputSize = 1
	}

	if !l.InputAliased {
		l.InputBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
			Label: labelPrefix + "_In",
			Size:  uint64(inputSize * 4),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
		})
		if err != nil {
			return err
		}
	}

	l.OutputBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Out",
		Size:  uint64(outputSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	weightSize := l.Spec.OutChannels * l.Spec.InChannels * l.Spec.KernelSize * l.Spec.KernelSize * l.Spec.KernelSize
	if weightSize < 1 {
		weightSize = 1
	}

	if len(l.Spec.Weights) != weightSize {
		l.Spec.Weights = make([]float32, weightSize)
		for i := range l.Spec.Weights {
			l.Spec.Weights[i] = float32(i%100) * 0.01
		}
	}
	l.WeightBuffer, err = NewFloatBuffer(l.Spec.Weights, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}

	if len(l.Spec.Bias) != l.Spec.OutChannels {
		l.Spec.Bias = make([]float32, l.Spec.OutChannels)
	}
	l.BiasBuffer, err = NewFloatBuffer(l.Spec.Bias, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}

	l.StagingBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Staging",
		Size:  uint64(outputSize * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})

	return err
}

func (l *Conv3DLayer) AllocateBackwardBuffers(ctx *Context, labelPrefix string) error {
	var err error
	if l.BatchSize <= 0 {
		l.BatchSize = 1
	}
	inputSize := l.Spec.InputDepth * l.Spec.InputHeight * l.Spec.InputWidth * l.Spec.InChannels * l.BatchSize
	l.InputGradientBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_InGrad",
		Size:  uint64(inputSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	weightSize := l.Spec.OutChannels * l.Spec.InChannels * l.Spec.KernelSize * l.Spec.KernelSize * l.Spec.KernelSize
	l.WeightGradientBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_WGrad",
		Size:  uint64(weightSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	l.BiasGradientBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_BGrad",
		Size:  uint64(l.Spec.OutChannels * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})

	return err
}

func (l *Conv3DLayer) getActivationCode(varName string) string {
	switch l.Spec.Activation {
	case "relu":
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

func (l *Conv3DLayer) getActivationDerivative(varName string) string {
	switch l.Spec.Activation {
	case "relu":
		return fmt.Sprintf("select(1.1, 0.0, %s > 0.0)", varName)
	case "sigmoid":
		return fmt.Sprintf("(%s * (1.0 - %s))", varName, varName)
	case "tanh":
		return fmt.Sprintf("(1.0 - %s * %s)", varName, varName)
	case "leaky_relu":
		return fmt.Sprintf("select(1.0, 0.01, %s > 0.0)", varName)
	default:
		return "1.0"
	}
}

func (l *Conv3DLayer) GenerateShader() string {
	stride := l.Spec.Stride
	if stride < 1 {
		stride = 1
	}
	outD, outH, outW := l.computeOutputSize()

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read> weights : array<f32>;
		@group(0) @binding(2) var<storage, read> bias : array<f32>;
		@group(0) @binding(3) var<storage, read_write> output : array<f32>;

		const IN_D: u32 = %du;
		const IN_H: u32 = %du;
		const IN_W: u32 = %du;
		const IN_CH: u32 = %du;
		const OUT_CH: u32 = %du;
		const K: u32 = %du;
		const STRIDE: u32 = %du;
		const PADDING: u32 = %du;
		const OUT_D: u32 = %du;
		const OUT_H: u32 = %du;
		const OUT_W: u32 = %du;
		const BATCH_SIZE: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let total_per_sample = OUT_D * OUT_H * OUT_W * OUT_CH;
			let total = total_per_sample * BATCH_SIZE;
			
			if (idx >= total) { return; }

			let batch = idx / total_per_sample;
			let sample_idx = idx %% total_per_sample;
			
			// Output Layout: [Batch, OutCH, OutD, OutH, OutW] (aligned with CPU)
			// Wait, if CPU is [Batch][Filters][OutD][OutH][OutW]:
			let out_w = sample_idx %% OUT_W;
			let out_h = (sample_idx / OUT_W) %% OUT_H;
			let out_d = (sample_idx / (OUT_W * OUT_H)) %% OUT_D;
			let out_c = sample_idx / (OUT_W * OUT_H * OUT_D);

			var sum: f32 = bias[out_c];

			let input_batch_offset = batch * (IN_CH * IN_D * IN_H * IN_W);

			for (var in_c: u32 = 0u; in_c < IN_CH; in_c++) {
				for (var kd: u32 = 0u; kd < K; kd++) {
					for (var kh: u32 = 0u; kh < K; kh++) {
						for (var kw: u32 = 0u; kw < K; kw++) {
							let in_d_signed = i32(out_d * STRIDE + kd) - i32(PADDING);
							let in_h_signed = i32(out_h * STRIDE + kh) - i32(PADDING);
							let in_w_signed = i32(out_w * STRIDE + kw) - i32(PADDING);
							
							if (in_d_signed >= 0 && u32(in_d_signed) < IN_D &&
							    in_h_signed >= 0 && u32(in_h_signed) < IN_H &&
							    in_w_signed >= 0 && u32(in_w_signed) < IN_W) {
								
								let in_d = u32(in_d_signed);
								let in_h = u32(in_h_signed);
								let in_w = u32(in_w_signed);
								
								// Input Layout: [InCH, InD, InH, InW]
								let i_local = in_c * (IN_D * IN_H * IN_W) + in_d * (IN_H * IN_W) + in_h * IN_W + in_w;
								let i_idx = input_batch_offset + i_local;
								
								// Weights: [OUT_CH, IN_CH, K, K, K]
								let w_idx = out_c * (IN_CH * K * K * K) + in_c * (K * K * K) + kd * (K * K) + kh * K + kw;
								
								sum += input[i_idx] * weights[w_idx];
							}
						}
					}
				}
			}
			output[idx] = %s;
		}
	`, l.Spec.InputDepth, l.Spec.InputHeight, l.Spec.InputWidth, l.Spec.InChannels, l.Spec.OutChannels,
		l.Spec.KernelSize, stride, l.Spec.Padding, outD, outH, outW, l.BatchSize, l.getActivationCode("sum"))
}

func (l *Conv3DLayer) GenerateBackwardShader() string {
	stride := l.Spec.Stride
	if stride < 1 {
		stride = 1
	}
	outD, outH, outW := l.computeOutputSize()

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> d_output : array<f32>;
		@group(0) @binding(1) var<storage, read> output : array<f32>;
		@group(0) @binding(2) var<storage, read> weights : array<f32>;
		@group(0) @binding(3) var<storage, read_write> d_input : array<f32>;

		const IN_D: u32 = %du;
		const IN_H: u32 = %du;
		const IN_W: u32 = %du;
		const IN_CH: u32 = %du;
		const OUT_CH: u32 = %du;
		const K: u32 = %du;
		const STRIDE: u32 = %du;
		const PADDING: u32 = %du;
		const OUT_D: u32 = %du;
		const OUT_H: u32 = %du;
		const OUT_W: u32 = %du;
		const BATCH_SIZE: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let in_total_per_sample = IN_CH * IN_D * IN_H * IN_W;
			let total = in_total_per_sample * BATCH_SIZE;

			if (idx >= total) { return; }

			let batch = idx / in_total_per_sample;
			let sample_idx = idx %% in_total_per_sample;

			let in_w = sample_idx %% IN_W;
			let in_h = (sample_idx / IN_W) %% IN_H;
			let in_d = (sample_idx / (IN_W * IN_H)) %% IN_D;
			let in_c = sample_idx / (IN_W * IN_H * IN_D);

			var grad: f32 = 0.0;
			let d_out_batch_offset = batch * (OUT_CH * OUT_D * OUT_H * OUT_W);

			for (var out_d: u32 = 0u; out_d < OUT_D; out_d++) {
				for (var out_h: u32 = 0u; out_h < OUT_H; out_h++) {
					for (var out_w: u32 = 0u; out_w < OUT_W; out_w++) {
						for (var kd: u32 = 0u; kd < K; kd++) {
							for (var kh: u32 = 0u; kh < K; kh++) {
								for (var kw: u32 = 0u; kw < K; kw++) {
									let in_d_check = i32(out_d * STRIDE + kd) - i32(PADDING);
									let in_h_check = i32(out_h * STRIDE + kh) - i32(PADDING);
									let in_w_check = i32(out_w * STRIDE + kw) - i32(PADDING);
									
									if (u32(in_d_check) == in_d && u32(in_h_check) == in_h && u32(in_w_check) == in_w &&
									    in_d_check >= 0 && in_h_check >= 0 && in_w_check >= 0) {
										
										for (var out_c: u32 = 0u; out_c < OUT_CH; out_c++) {
											let do_local = out_c * (OUT_D * OUT_H * OUT_W) + out_d * (OUT_H * OUT_W) + out_h * OUT_W + out_w;
											let do_idx = d_out_batch_offset + do_local;

											let w_idx = out_c * (IN_CH * K * K * K) + in_c * (K * K * K) + kd * (K * K) + kh * K + kw;
											
											let out_val = output[do_idx];
											let d_act: f32 = %s;
											
											grad += d_output[do_idx] * d_act * weights[w_idx];
										}
									}
								}
							}
						}
					}
				}
			}

			d_input[idx] = grad;
		}
	`, l.Spec.InputDepth, l.Spec.InputHeight, l.Spec.InputWidth, l.Spec.InChannels, l.Spec.OutChannels,
		l.Spec.KernelSize, stride, l.Spec.Padding, outD, outH, outW, l.BatchSize, l.getActivationDerivative("out_val"))
}

func (l *Conv3DLayer) GenerateBackwardGradsShader() string {
	stride := l.Spec.Stride
	if stride < 1 {
		stride = 1
	}
	outD, outH, outW := l.computeOutputSize()

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read> d_output : array<f32>;
		@group(0) @binding(2) var<storage, read_write> d_weights : array<f32>;
		@group(0) @binding(3) var<storage, read_write> d_bias : array<f32>;
		@group(0) @binding(4) var<storage, read> output : array<f32>;

		const IN_D: u32 = %du;
		const IN_H: u32 = %du;
		const IN_W: u32 = %du;
		const IN_CH: u32 = %du;
		const OUT_CH: u32 = %du;
		const K: u32 = %du;
		const STRIDE: u32 = %du;
		const PADDING: u32 = %du;
		const OUT_D: u32 = %du;
		const OUT_H: u32 = %du;
		const OUT_W: u32 = %du;
		const BATCH_SIZE: u32 = %du;
		
		const WEIGHT_SIZE: u32 = %du; 
		const BIAS_SIZE: u32 = %du;   

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			
			// --- Bias Gradients ---
			if (idx < BIAS_SIZE) {
				let out_c = idx;
				var sum: f32 = 0.0;
				
				for (var b: u32 = 0u; b < BATCH_SIZE; b++) {
					let batch_offset = b * (OUT_CH * OUT_D * OUT_H * OUT_W);
					for (var d: u32 = 0u; d < OUT_D; d++) {
						for (var h: u32 = 0u; h < OUT_H; h++) {
							for (var w: u32 = 0u; w < OUT_W; w++) {
								let do_idx = batch_offset + out_c * (OUT_D * OUT_H * OUT_W) + d * (OUT_H * OUT_W) + h * OUT_W + w;
								
								let out_val = output[do_idx];
								let d_act: f32 = %s;
	
								sum += d_output[do_idx] * d_act;
							}
						}
					}
				}
				d_bias[idx] = sum;
			}
			
			// --- Weight Gradients ---
			if (idx < WEIGHT_SIZE) {
				let kw = idx %% K;
				let kh = (idx / K) %% K;
				let kd = (idx / (K * K)) %% K;
				let in_c = (idx / (K * K * K)) %% IN_CH;
				let out_c = idx / (IN_CH * K * K * K);
				
				var dw: f32 = 0.0;
				
				for (var b: u32 = 0u; b < BATCH_SIZE; b++) {
					let input_batch_offset = b * (IN_CH * IN_D * IN_H * IN_W);
					let do_batch_offset = b * (OUT_CH * OUT_D * OUT_H * OUT_W);
					
					for (var out_d: u32 = 0u; out_d < OUT_D; out_d++) {
						for (var out_h: u32 = 0u; out_h < OUT_H; out_h++) {
							for (var out_w: u32 = 0u; out_w < OUT_W; out_w++) {
								
								let in_d_signed = i32(out_d * STRIDE + kd) - i32(PADDING);
								let in_h_signed = i32(out_h * STRIDE + kh) - i32(PADDING);
								let in_w_signed = i32(out_w * STRIDE + kw) - i32(PADDING);
								
								if (in_d_signed >= 0 && u32(in_d_signed) < IN_D &&
									in_h_signed >= 0 && u32(in_h_signed) < IN_H &&
									in_w_signed >= 0 && u32(in_w_signed) < IN_W) {
										
									let in_d = u32(in_d_signed);
									let in_h = u32(in_h_signed);
									let in_w = u32(in_w_signed);
									
									let in_idx = input_batch_offset + in_c * (IN_D * IN_H * IN_W) + in_d * (IN_H * IN_W) + in_h * IN_W + in_w;
									let do_idx = do_batch_offset + out_c * (OUT_D * OUT_H * OUT_W) + out_d * (OUT_H * OUT_W) + out_h * OUT_W + out_w;
									
									let out_val = output[do_idx];
									let d_act: f32 = %s; 
	
									dw += d_output[do_idx] * d_act * input[in_idx];
								}
							}
						}
					}
				}
				d_weights[idx] = dw;
			}
		}
	`, l.Spec.InputDepth, l.Spec.InputHeight, l.Spec.InputWidth, l.Spec.InChannels, l.Spec.OutChannels,
		l.Spec.KernelSize, stride, l.Spec.Padding, outD, outH, outW, l.BatchSize,
		l.Spec.OutChannels*l.Spec.InChannels*l.Spec.KernelSize*l.Spec.KernelSize*l.Spec.KernelSize,
		l.Spec.OutChannels,
		l.getActivationDerivative("out_val"),
		l.getActivationDerivative("out_val"))
}

func (l *Conv3DLayer) Compile(ctx *Context, labelPrefix string) error {
	mod, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_Shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateShader()},
	})
	if err != nil {
		return err
	}
	defer mod.Release()

	bgl, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: labelPrefix + "_BGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
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
		Compute: wgpu.ProgrammableStageDescriptor{Module: mod, EntryPoint: "main"},
	})
	return err
}

func (l *Conv3DLayer) CompileBackward(ctx *Context, labelPrefix string) error {
	var err error
	mod, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_BwdShader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateBackwardShader()},
	})
	if err != nil {
		return err
	}
	defer mod.Release()

	bglBackward, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: labelPrefix + "_BwdBGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
		},
	})
	if err != nil {
		return err
	}

	plBackward, err := ctx.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            labelPrefix + "_BwdPL",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bglBackward},
	})
	if err != nil {
		return err
	}

	l.bwPipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_BwdPipe",
		Layout:  plBackward,
		Compute: wgpu.ProgrammableStageDescriptor{Module: mod, EntryPoint: "main"},
	})
	if err != nil {
		return err
	}

	modGrad, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_BwdGradShader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateBackwardGradsShader()},
	})
	if err != nil {
		return err
	}
	defer modGrad.Release()

	bglGrad, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: labelPrefix + "_BwdGradBGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
			{Binding: 4, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
		},
	})
	if err != nil {
		return err
	}

	plGrad, err := ctx.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            labelPrefix + "_BwdGradPL",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bglGrad},
	})
	if err != nil {
		return err
	}

	l.bwGradPipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_BwdGradPipe",
		Layout:  plGrad,
		Compute: wgpu.ProgrammableStageDescriptor{Module: modGrad, EntryPoint: "main"},
	})
	return err
}

func (l *Conv3DLayer) CreateBindGroup(ctx *Context, labelPrefix string) error {
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

func (l *Conv3DLayer) CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error {
	var err error
	l.bwBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_BwdBind",
		Layout: l.bwPipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
			{Binding: 1, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
			{Binding: 2, Buffer: l.WeightBuffer, Size: l.WeightBuffer.GetSize()},
			{Binding: 3, Buffer: l.InputGradientBuffer, Size: l.InputGradientBuffer.GetSize()},
		},
	})
	if err != nil {
		return err
	}

	l.bwGradBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_BwdGradBind",
		Layout: l.bwGradPipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
			{Binding: 1, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
			{Binding: 2, Buffer: l.WeightGradientBuffer, Size: l.WeightGradientBuffer.GetSize()},
			{Binding: 3, Buffer: l.BiasGradientBuffer, Size: l.BiasGradientBuffer.GetSize()},
			{Binding: 4, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
		},
	})
	return err
}

func (l *Conv3DLayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	pass.SetPipeline(l.pipeline)
	pass.SetBindGroup(0, l.bindGroup, nil)
	total := l.outputD * l.outputH * l.outputW * l.Spec.OutChannels * l.BatchSize
	pass.DispatchWorkgroups(uint32((total+255)/256), 1, 1)
}

func (l *Conv3DLayer) UpdateParams(ctx *Context, inputLen int, cachePos int) {
	if inputLen > 0 {
		l.BatchSize = inputLen
	}
}

func (l *Conv3DLayer) ZeroGradients(ctx *Context) {}

func (l *Conv3DLayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(l.bwPipeline)
	pass.SetBindGroup(0, l.bwBindGroup, nil)
	totalInput := l.Spec.InputDepth * l.Spec.InputHeight * l.Spec.InputWidth * l.Spec.InChannels * l.BatchSize
	pass.DispatchWorkgroups(uint32((totalInput+255)/256), 1, 1)
	pass.End()

	passGrad := enc.BeginComputePass(nil)
	passGrad.SetPipeline(l.bwGradPipeline)
	passGrad.SetBindGroup(0, l.bwGradBindGroup, nil)
	weightSize := l.Spec.OutChannels * l.Spec.InChannels * l.Spec.KernelSize * l.Spec.KernelSize * l.Spec.KernelSize
	passGrad.DispatchWorkgroups(uint32((weightSize+255)/256), 1, 1)
	passGrad.End()
}

func (l *Conv3DLayer) UploadWeights(ctx *Context) {
	if len(l.Spec.Weights) > 0 {
		ctx.Queue.WriteBuffer(l.WeightBuffer, 0, wgpu.ToBytes(l.Spec.Weights))
	}
	if len(l.Spec.Bias) > 0 {
		ctx.Queue.WriteBuffer(l.BiasBuffer, 0, wgpu.ToBytes(l.Spec.Bias))
	}
}

func (l *Conv3DLayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	wSize := l.Spec.OutChannels * l.Spec.InChannels * l.Spec.KernelSize * l.Spec.KernelSize * l.Spec.KernelSize
	w, err := ReadBuffer(l.WeightBuffer, wSize)
	if err != nil {
		return nil, nil, err
	}
	b, err := ReadBuffer(l.BiasBuffer, l.Spec.OutChannels)
	return w, b, err
}

func (l *Conv3DLayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	iSize := l.Spec.InputDepth * l.Spec.InputHeight * l.Spec.InputWidth * l.Spec.InChannels * l.BatchSize
	iGrad, err := ReadBuffer(l.InputGradientBuffer, iSize)
	if err != nil {
		return nil, nil, nil, err
	}

	wSize := l.Spec.OutChannels * l.Spec.InChannels * l.Spec.KernelSize * l.Spec.KernelSize * l.Spec.KernelSize
	wGrad, err := ReadBuffer(l.WeightGradientBuffer, wSize)
	if err != nil {
		return nil, nil, iGrad, err
	}

	bGrad, err := ReadBuffer(l.BiasGradientBuffer, l.Spec.OutChannels)
	return wGrad, bGrad, iGrad, err
}

func (l *Conv3DLayer) Cleanup() {
	if l.InputBuffer != nil && !l.InputAliased {
		l.InputBuffer.Destroy()
	}
	if l.OutputBuffer != nil {
		l.OutputBuffer.Destroy()
	}
	bufs := []*wgpu.Buffer{
		l.StagingBuffer, l.WeightBuffer, l.BiasBuffer,
		l.InputGradientBuffer, l.WeightGradientBuffer, l.BiasGradientBuffer,
	}
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
	if l.bwGradPipeline != nil {
		l.bwGradPipeline.Release()
	}
	if l.bwGradBindGroup != nil {
		l.bwGradBindGroup.Release()
	}
}
