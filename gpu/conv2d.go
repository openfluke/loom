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
	Activation  string    // "relu", "sigmoid", etc.
}

// Conv2DLayer holds GPU resources for 2D Convolution
type Conv2DLayer struct {
	Spec Conv2DSpec

	BatchSize int // Number of samples per batch

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

	// Validate parameters
	if l.Spec.InputHeight <= 0 {
		l.Spec.InputHeight = 32
	}
	if l.Spec.InputWidth <= 0 {
		l.Spec.InputWidth = 32
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

	l.outputH, l.outputW = l.computeOutputSize()
	if l.outputH <= 0 {
		l.outputH = 1
	}
	if l.outputW <= 0 {
		l.outputW = 1
	}

	inputSize := l.Spec.InputHeight * l.Spec.InputWidth * l.Spec.InChannels * l.BatchSize
	if inputSize < 1 {
		inputSize = 1
	}
	outputSize := l.outputH * l.outputW * l.Spec.OutChannels * l.BatchSize
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

	weightSize := l.Spec.OutChannels * l.Spec.InChannels * l.Spec.KernelSize * l.Spec.KernelSize
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
	l.WeightBuffer, err = NewFloatBuffer(l.Spec.Weights, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}

	// Ensure bias array matches expected size
	if len(l.Spec.Bias) != l.Spec.OutChannels {
		l.Spec.Bias = make([]float32, l.Spec.OutChannels)
	}
	l.BiasBuffer, err = NewFloatBuffer(l.Spec.Bias, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
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
	if l.BatchSize <= 0 {
		l.BatchSize = 1
	}
	inputSize := l.Spec.InputHeight * l.Spec.InputWidth * l.Spec.InChannels * l.BatchSize
	l.InputGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_InGrad",
		Size:  uint64(inputSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Weight Gradients
	weightSize := l.Spec.OutChannels * l.Spec.InChannels * l.Spec.KernelSize * l.Spec.KernelSize
	l.WeightGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_WGrad",
		Size:  uint64(weightSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Bias Gradients
	l.BiasGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_BGrad",
		Size:  uint64(l.Spec.OutChannels * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	return err
}

func (l *Conv2DLayer) getActivationCode(varName string) string {
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

func (l *Conv2DLayer) getActivationDerivative(varName string) string {
	switch l.Spec.Activation {
	case "relu":
		// Derivative of ReLU with 1.1x scaling: 1.1 if x > 0, else 0
		return fmt.Sprintf("select(1.1, 0.0, %s > 0.0)", varName)
	case "sigmoid":
		// Derivative of sigmoid: s(x) * (1 - s(x)), where varName is already sigmoid output
		return fmt.Sprintf("(%s * (1.0 - %s))", varName, varName)
	case "tanh":
		// Derivative of tanh: 1 - tanh^2(x), where varName is already tanh output
		return fmt.Sprintf("(1.0 - %s * %s)", varName, varName)
	case "leaky_relu":
		// Derivative of leaky ReLU: 1 if x > 0, else 0.01
		return fmt.Sprintf("select(1.0, 0.01, %s > 0.0)", varName)
	default:
		// No activation, derivative is 1
		return "1.0"
	}
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
		const BATCH_SIZE: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let total_per_sample = OUT_H * OUT_W * OUT_CH;
			let total = total_per_sample * BATCH_SIZE;
			
			if (idx >= total) { return; }

			// Global Index -> [Batch, OutH, OutW, OutCh]
			let batch = idx / total_per_sample;
			let sample_idx = idx %% total_per_sample;
			
			// Sample Layout: [H, W, C]
			let out_c = sample_idx %% OUT_CH;
			let out_w = (sample_idx / OUT_CH) %% OUT_W;
			let out_h = sample_idx / (OUT_CH * OUT_W);

			var sum: f32 = bias[out_c];

			// Input Base Offset for this batch
			let input_batch_offset = batch * (IN_H * IN_W * IN_CH);

			for (var kh: u32 = 0u; kh < K; kh++) {
				for (var kw: u32 = 0u; kw < K; kw++) {
					let in_h_signed = i32(out_h * STRIDE + kh) - i32(PADDING);
					let in_w_signed = i32(out_w * STRIDE + kw) - i32(PADDING);
					
					if (in_h_signed >= 0 && u32(in_h_signed) < IN_H &&
					    in_w_signed >= 0 && u32(in_w_signed) < IN_W) {
						let in_h = u32(in_h_signed);
						let in_w = u32(in_w_signed);
						
						for (var in_c: u32 = 0u; in_c < IN_CH; in_c++) {
							// Input: [Batch, H, W, C]
							// Local index in sample: [H, W, C]
							let i_local = in_h * IN_W * IN_CH + in_w * IN_CH + in_c;
							let i_idx = input_batch_offset + i_local;
							
							// Weights: [OUT_CH, IN_CH, K, K]
							let w_idx = out_c * IN_CH * K * K + in_c * K * K + kh * K + kw;
							sum += input[i_idx] * weights[w_idx];
						}
					}
				}
			}

			output[idx] = %s;
		}
	`, l.Spec.InputHeight, l.Spec.InputWidth, l.Spec.InChannels, l.Spec.OutChannels,
		l.Spec.KernelSize, stride, l.Spec.Padding, outH, outW, l.BatchSize, l.getActivationCode("sum"))
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
		@group(0) @binding(1) var<storage, read> output : array<f32>;
		@group(0) @binding(2) var<storage, read> weights : array<f32>;
		@group(0) @binding(3) var<storage, read_write> d_input : array<f32>;

		const IN_H: u32 = %du;
		const IN_W: u32 = %du;
		const IN_CH: u32 = %du;
		const OUT_CH: u32 = %du;
		const K: u32 = %du;
		const STRIDE: u32 = %du;
		const PADDING: u32 = %du;
		const OUT_H: u32 = %du;
		const OUT_W: u32 = %du;
		const BATCH_SIZE: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let in_total_per_sample = IN_H * IN_W * IN_CH;
			let total = in_total_per_sample * BATCH_SIZE;

			if (idx >= total) { return; }

			// Global Index -> [Batch, InH, InW, InCh]
			let batch = idx / in_total_per_sample;
			let sample_idx = idx %% in_total_per_sample;

			// Input layout: [H, W, C]
			let in_c = sample_idx %% IN_CH;
			let in_w = (sample_idx / IN_CH) %% IN_W;
			let in_h = sample_idx / (IN_CH * IN_W);

			var grad: f32 = 0.0;
			
			// dOutput Batch Offset
			let d_out_batch_offset = batch * (OUT_H * OUT_W * OUT_CH);

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
									// d_output: [Batch, OUT_H, OUT_W, OUT_CH]
									// Local: [OUT_H, OUT_W, OUT_CH]
									let do_local = out_h * OUT_W * OUT_CH + out_w * OUT_CH + out_c;
									let do_idx = d_out_batch_offset + do_local;

									// weights: [OUT_CH, IN_CH, K, K]
									let w_idx = out_c * IN_CH * K * K + in_c * K * K + kh * K + kw;
									
									// Apply activation derivative
									let out_val = output[do_idx];
									let d_act: f32 = %s;
									
									grad += d_output[do_idx] * d_act * weights[w_idx];
								}
							}
						}
					}
				}
			}

			d_input[idx] = grad;
		}
	`, l.Spec.InputHeight, l.Spec.InputWidth, l.Spec.InChannels, l.Spec.OutChannels,
		l.Spec.KernelSize, stride, l.Spec.Padding, outH, outW, l.BatchSize, l.getActivationDerivative("out_val"))
}

func (l *Conv2DLayer) GenerateBackwardGradsShader() string {
	stride := l.Spec.Stride
	if stride < 1 {
		stride = 1
	}
	outH, outW := l.computeOutputSize()

	// Gradients Shader: Compute dWeights and dBias
	// We dispatch threads for weights and bias separately or together
	// Let's do a single dispatch over largest dimension, but they are different sizes.
	// Weights: OutCh * InCh * K * K
	// Bias: OutCh
	// To keep it simple, let's use a single shader that handles both if we can map IDs,
	// or just optimize for Weights since they are the heaviest, and do Bias in the same kernel or use branching.

	// Actually, let's just make the total threads cover (WeightSize + BiasSize).
	// WeightSize = OUT_CH * IN_CH * K * K
	// BiasSize = OUT_CH

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read> d_output : array<f32>;
		@group(0) @binding(2) var<storage, read_write> d_weights : array<f32>;
		@group(0) @binding(3) var<storage, read_write> d_bias : array<f32>;
		@group(0) @binding(4) var<storage, read> output : array<f32>;

		const IN_H: u32 = %du;
		const IN_W: u32 = %du;
		const IN_CH: u32 = %du;
		const OUT_CH: u32 = %du;
		const K: u32 = %du;
		const STRIDE: u32 = %du;
		const PADDING: u32 = %du;
		const OUT_H: u32 = %du;
		const OUT_W: u32 = %du;
		const BATCH_SIZE: u32 = %du;
		
		const WEIGHT_SIZE: u32 = %du; // OUT_CH * IN_CH * K * K
		const BIAS_SIZE: u32 = %du;   // OUT_CH

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			
			// --- Bias Gradients ---
			if (idx < BIAS_SIZE) {
				// idx maps to out_c
				let out_c = idx;
				var sum: f32 = 0.0;
				
				// Sum d_output[b, out_h, out_w, out_c] over all b, h, w
				for (var b: u32 = 0u; b < BATCH_SIZE; b++) {
					let batch_offset = b * (OUT_H * OUT_W * OUT_CH);
					for (var h: u32 = 0u; h < OUT_H; h++) {
						for (var w: u32 = 0u; w < OUT_W; w++) {
							let do_idx = batch_offset + h * (OUT_W * OUT_CH) + w * OUT_CH + out_c;
							
							// Apply activation derivative
							let out_val = output[do_idx];
							let d_act: f32 = %s;

							sum += d_output[do_idx] * d_act;
						}
					}
				}
				d_bias[idx] = sum;
			}
			
			// --- Weight Gradients ---
			if (idx < WEIGHT_SIZE) {
				// Map idx to [out_c, in_c, kh, kw]
				let kw = idx %% K;
				let kh = (idx / K) %% K;
				let in_c = (idx / (K * K)) %% IN_CH;
				let out_c = idx / (IN_CH * K * K);
				
				var dw: f32 = 0.0;
				
				for (var b: u32 = 0u; b < BATCH_SIZE; b++) {
					let input_batch_offset = b * (IN_H * IN_W * IN_CH);
					let do_batch_offset = b * (OUT_H * OUT_W * OUT_CH);
					
					for (var out_h: u32 = 0u; out_h < OUT_H; out_h++) {
						for (var out_w: u32 = 0u; out_w < OUT_W; out_w++) {
							
							// Corresponding input position
							let in_h_signed = i32(out_h * STRIDE + kh) - i32(PADDING);
							let in_w_signed = i32(out_w * STRIDE + kw) - i32(PADDING);
							
							if (in_h_signed >= 0 && u32(in_h_signed) < IN_H &&
								in_w_signed >= 0 && u32(in_w_signed) < IN_W) {
									
								let in_h = u32(in_h_signed);
								let in_w = u32(in_w_signed);
								
								let in_idx = input_batch_offset + in_h * (IN_W * IN_CH) + in_w * IN_CH + in_c;
								let do_idx = do_batch_offset + out_h * (OUT_W * OUT_CH) + out_w * OUT_CH + out_c;
								
								// Apply activation derivative
								let out_val = output[do_idx];
								let d_act: f32 = %s; // Reusing same format string

								dw += d_output[do_idx] * d_act * input[in_idx];
							}
						}
					}
				}
				d_weights[idx] = dw;
			}
		}
	`, l.Spec.InputHeight, l.Spec.InputWidth, l.Spec.InChannels, l.Spec.OutChannels,
		l.Spec.KernelSize, stride, l.Spec.Padding, outH, outW, l.BatchSize,
		l.Spec.OutChannels*l.Spec.InChannels*l.Spec.KernelSize*l.Spec.KernelSize,
		l.Spec.OutChannels,
		l.getActivationDerivative("out_val"),
		l.getActivationDerivative("out_val"))
}

func (l *Conv2DLayer) Compile(ctx *Context, labelPrefix string) error {
	mod, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_Shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateShader()},
	})
	if err != nil {
		return err
	}
	defer mod.Release()

	// Explicit Layout
	bgl, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: labelPrefix + "_BGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Input
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Weights
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Bias
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // Output
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

func (l *Conv2DLayer) CompileBackward(ctx *Context, labelPrefix string) error {
	var err error
	// 1. Input Gradients Shader
	mod, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_BwdShader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateBackwardShader()},
	})
	if err != nil {
		return err
	}
	defer mod.Release()

	// dInput Layout: 0:dOut, 1:Out, 2:Writes, 3:dIn
	bglBackward, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: labelPrefix + "_BwdBGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // dOutput
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Output
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Weights
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // dInput
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

	// 2. Weight/Bias Gradients Shader
	modGrad, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_BwdGradShader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateBackwardGradsShader()},
	})
	if err != nil {
		return err
	}
	defer modGrad.Release()

	// dGrad Layout: 0:In, 1:dOut, 2:dW, 3:dB, 4:Out
	bglGrad, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: labelPrefix + "_BwdGradBGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Input
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // dOutput
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // dWeights
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // dBias
			{Binding: 4, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Output
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
	// bind group for dInput
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

	// bind group for dWeights/dBias
	l.bwGradBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_BwdGradBind",
		Layout: l.bwGradPipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},                   // input
			{Binding: 1, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},                   // d_output
			{Binding: 2, Buffer: l.WeightGradientBuffer, Size: l.WeightGradientBuffer.GetSize()}, // d_weights
			{Binding: 3, Buffer: l.BiasGradientBuffer, Size: l.BiasGradientBuffer.GetSize()},     // d_bias
			{Binding: 4, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},                 // output (for activation deriv)
		},
	})
	return err
}

func (l *Conv2DLayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	pass.SetPipeline(l.pipeline)
	pass.SetBindGroup(0, l.bindGroup, nil)
	total := l.outputH * l.outputW * l.Spec.OutChannels * l.BatchSize
	pass.DispatchWorkgroups(uint32((total+255)/256), 1, 1)
}

func (l *Conv2DLayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	// 1. Compute dInput
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(l.bwPipeline)
	pass.SetBindGroup(0, l.bwBindGroup, nil)
	totalInput := l.Spec.InputHeight * l.Spec.InputWidth * l.Spec.InChannels * l.BatchSize
	pass.DispatchWorkgroups(uint32((totalInput+255)/256), 1, 1)
	pass.End()

	// 2. Compute dWeights / dBias
	passGrad := enc.BeginComputePass(nil)
	passGrad.SetPipeline(l.bwGradPipeline)
	passGrad.SetBindGroup(0, l.bwGradBindGroup, nil)

	weightSize := l.Spec.OutChannels * l.Spec.InChannels * l.Spec.KernelSize * l.Spec.KernelSize
	// We dispatch enough threads to cover WeightSize (which is larger than BiasSize typically)
	// The shader checks bounds for both.
	passGrad.DispatchWorkgroups(uint32((weightSize+255)/256), 1, 1)
	passGrad.End()
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
	iGrad, err := ReadBuffer(l.InputGradientBuffer, l.Spec.InputHeight*l.Spec.InputWidth*l.Spec.InChannels*l.BatchSize)
	if err != nil {
		return nil, nil, nil, err
	}

	wSize := l.Spec.OutChannels * l.Spec.InChannels * l.Spec.KernelSize * l.Spec.KernelSize
	wGrad, err := ReadBuffer(l.WeightGradientBuffer, wSize)
	if err != nil {
		return nil, nil, iGrad, err
	}

	bGrad, err := ReadBuffer(l.BiasGradientBuffer, l.Spec.OutChannels)
	return wGrad, bGrad, iGrad, err
}

func (l *Conv2DLayer) Cleanup() {
	bufs := []*wgpu.Buffer{
		l.InputBuffer, l.OutputBuffer, l.StagingBuffer, l.WeightBuffer, l.BiasBuffer,
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
