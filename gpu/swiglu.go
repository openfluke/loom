package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// SwiGLUSpec defines configuration for SwiGLU gated activation layer
// SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
type SwiGLUSpec struct {
	InputSize        int       // Hidden size (e.g., 768)
	IntermediateSize int       // Intermediate size (e.g., 3072)
	SeqLen           int       // Sequence length / batch size
	GateWeights      []float32 // [InputSize * IntermediateSize]
	UpWeights        []float32 // [InputSize * IntermediateSize]
	DownWeights      []float32 // [IntermediateSize * InputSize]
	GateBias         []float32 // [IntermediateSize]
	UpBias           []float32 // [IntermediateSize]
	DownBias         []float32 // [InputSize]
}

// SwiGLULayer holds GPU resources for SwiGLU
type SwiGLULayer struct {
	Spec      SwiGLUSpec
	BatchSize int // Number of vectors to process

	// Forward pipelines (3 stages)
	pipelineGateUp   *wgpu.ComputePipeline // Combined gate+up projection
	pipelineActivate *wgpu.ComputePipeline // SiLU + element-wise multiply
	pipelineDown     *wgpu.ComputePipeline // Down projection

	bindGroupGateUp   *wgpu.BindGroup
	bindGroupActivate *wgpu.BindGroup
	bindGroupDown     *wgpu.BindGroup

	// Buffers
	InputBuffer        *wgpu.Buffer
	OutputBuffer       *wgpu.Buffer
	StagingBuffer      *wgpu.Buffer
	GateWeightBuffer   *wgpu.Buffer
	UpWeightBuffer     *wgpu.Buffer
	DownWeightBuffer   *wgpu.Buffer
	GateBiasBuffer     *wgpu.Buffer
	UpBiasBuffer       *wgpu.Buffer
	DownBiasBuffer     *wgpu.Buffer
	GateOutBuffer      *wgpu.Buffer // Intermediate: gate projection output
	UpOutBuffer        *wgpu.Buffer // Intermediate: up projection output
	IntermediateBuffer *wgpu.Buffer // After activation
	ParamsBuffer       *wgpu.Buffer // Uniforms: [BatchSize, ...]

	// Backward (simplified - just input gradient for now)
	InputGradientBuffer *wgpu.Buffer
	bwPipeline          *wgpu.ComputePipeline
	bwBindGroup         *wgpu.BindGroup

	InputAliased bool
}

func (l *SwiGLULayer) GetInputBuffer() *wgpu.Buffer         { return l.InputBuffer }
func (l *SwiGLULayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *SwiGLULayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *SwiGLULayer) GetInputGradientBuffer() *wgpu.Buffer { return l.InputGradientBuffer }

func (l *SwiGLULayer) SetInputBuffer(buf *wgpu.Buffer) {
	l.InputBuffer = buf
	l.InputAliased = true
}

func (l *SwiGLULayer) AllocateBuffers(ctx *Context, labelPrefix string) error {
	var err error

	batch := l.BatchSize
	if batch < 1 {
		batch = 1
	}

	inputTotal := batch * l.Spec.InputSize
	interTotal := batch * l.Spec.IntermediateSize

	// Input/Output
	if !l.InputAliased {
		l.InputBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
			Label: labelPrefix + "_In",
			Size:  uint64(inputTotal * 4),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
		})

		if err != nil {
			return err
		}
	}

	l.OutputBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Out",
		Size:  uint64(inputTotal * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})

	if err != nil {
		return err
	}

	// Intermediate buffers
	l.GateOutBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_GateOut",
		Size:  uint64(interTotal * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})

	if err != nil {
		return err
	}

	l.UpOutBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_UpOut",
		Size:  uint64(interTotal * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})

	if err != nil {
		return err
	}

	l.IntermediateBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Intermediate",
		Size:  uint64(interTotal * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})

	if err != nil {
		return err
	}

	// Weight buffers
	gateUpSize := l.Spec.InputSize * l.Spec.IntermediateSize
	downSize := l.Spec.IntermediateSize * l.Spec.InputSize

	l.GateWeightBuffer, err = NewFloatBuffer(l.Spec.GateWeights, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}
	l.UpWeightBuffer, err = NewFloatBuffer(l.Spec.UpWeights, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}
	l.DownWeightBuffer, err = NewFloatBuffer(l.Spec.DownWeights, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}

	// Bias buffers
	gateBias := l.Spec.GateBias
	if len(gateBias) == 0 {
		gateBias = make([]float32, l.Spec.IntermediateSize)
	}
	l.GateBiasBuffer, err = NewFloatBuffer(gateBias, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}

	upBias := l.Spec.UpBias
	if len(upBias) == 0 {
		upBias = make([]float32, l.Spec.IntermediateSize)
	}
	l.UpBiasBuffer, err = NewFloatBuffer(upBias, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}

	downBias := l.Spec.DownBias
	if len(downBias) == 0 {
		downBias = make([]float32, l.Spec.InputSize)
	}
	l.DownBiasBuffer, err = NewFloatBuffer(downBias, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}

	// Params Buffer
	l.ParamsBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Params",
		Size:  16,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})

	if err != nil {
		return err
	}

	// Staging
	l.StagingBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Staging",
		Size:  uint64(inputTotal * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})

	if err != nil {
		return err
	}

	_ = gateUpSize
	_ = downSize
	return err
}

func (l *SwiGLULayer) AllocateBackwardBuffers(ctx *Context, labelPrefix string) error {
	var err error
	batch := l.BatchSize
	if batch < 1 {
		batch = 1
	}
	l.InputGradientBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_InGrad",
		Size:  uint64(batch * l.Spec.InputSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})

	return err
}

func (l *SwiGLULayer) GenerateGateUpShader() string {
	// Combined kernel: compute both gate and up projections
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read> gate_w : array<f32>;
		@group(0) @binding(2) var<storage, read> up_w : array<f32>;
		@group(0) @binding(3) var<storage, read> gate_b : array<f32>;
		@group(0) @binding(4) var<storage, read> up_b : array<f32>;
		@group(0) @binding(5) var<storage, read_write> gate_out : array<f32>;
		@group(0) @binding(6) var<storage, read_write> up_out : array<f32>;
		@group(0) @binding(7) var<uniform> params : LayerParams;
		struct LayerParams {
			batch: u32,
		};

		const INPUT_SIZE: u32 = %du;
		const INTER_SIZE: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let total = params.batch * INTER_SIZE;
			if (idx >= total) { return; }

			let s = idx / INTER_SIZE;
			let i = idx %% INTER_SIZE;
			let input_offset = s * INPUT_SIZE;

			// Gate projection
			var gate_sum: f32 = gate_b[i];
			for (var i: u32 = 0u; i < INPUT_SIZE; i++) {
				gate_sum += input[input_offset + i] * gate_w[i * INTER_SIZE + idx %% INTER_SIZE];
			}
			gate_out[idx] = gate_sum;

			// Up projection
			var up_sum: f32 = up_b[i];
			for (var i: u32 = 0u; i < INPUT_SIZE; i++) {
				up_sum += input[input_offset + i] * up_w[i * INTER_SIZE + idx %% INTER_SIZE];
			}
			up_out[idx] = up_sum;
		}
	`, l.Spec.InputSize, l.Spec.IntermediateSize)
}

func (l *SwiGLULayer) GenerateActivateShader() string {
	// SiLU activation on gate, then multiply with up
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> gate_out : array<f32>;
		@group(0) @binding(1) var<storage, read> up_out : array<f32>;
		@group(0) @binding(2) var<storage, read_write> intermediate : array<f32>;
		@group(0) @binding(3) var<uniform> params : LayerParams;
		struct LayerParams {
			batch: u32,
		};

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let total = params.batch * %du;
			if (idx >= total) { return; }

			let x = gate_out[idx];
			let sigmoid = 1.0 / (1.0 + exp(-x));
			let silu = x * sigmoid;
			intermediate[idx] = silu * up_out[idx];
		}
	`, l.Spec.IntermediateSize)
}

func (l *SwiGLULayer) GenerateDownShader() string {
	// Down projection
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> intermediate : array<f32>;
		@group(0) @binding(1) var<storage, read> down_w : array<f32>;
		@group(0) @binding(2) var<storage, read> down_b : array<f32>;
		@group(0) @binding(3) var<storage, read_write> output : array<f32>;
		@group(0) @binding(4) var<uniform> params : LayerParams;
		struct LayerParams {
			batch: u32,
		};

		const INPUT_SIZE: u32 = %du;
		const INTER_SIZE: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let total = params.batch * INPUT_SIZE;
			if (idx >= total) { return; }

			let s = idx / INPUT_SIZE;
			let i = idx %% INPUT_SIZE;
			let inter_offset = s * INTER_SIZE;

			var sum: f32 = down_b[i];
			for (var j: u32 = 0u; j < INTER_SIZE; j++) {
				sum += intermediate[inter_offset + j] * down_w[j * INPUT_SIZE + i];
			}
			output[idx] = sum;
		}
	`, l.Spec.InputSize, l.Spec.IntermediateSize)
}

func (l *SwiGLULayer) GenerateBackwardShader() string {
	// Full backward pass through the 3 stages:
	// 1. d_intermediate = d_output @ down_w.T
	// 2. d_gate_out = d_intermediate * up_out * d_silu(gate_out)
	//    d_up_out = d_intermediate * silu(gate_out)
	// 3. d_input = d_gate_out @ gate_w.T + d_up_out @ up_w.T
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> d_output : array<f32>;
		@group(0) @binding(1) var<storage, read> gate_w : array<f32>;
		@group(0) @binding(2) var<storage, read> up_w : array<f32>;
		@group(0) @binding(3) var<storage, read> down_w : array<f32>;
		@group(0) @binding(4) var<storage, read> gate_out : array<f32>;
		@group(0) @binding(5) var<storage, read> up_out : array<f32>;
		@group(0) @binding(6) var<storage, read_write> d_input : array<f32>;

		const SEQ_LEN: u32 = %du;
		const INPUT_SIZE: u32 = %du;
		const INTER_SIZE: u32 = %du;

		fn sigmoid(x: f32) -> f32 {
			return 1.0 / (1.0 + exp(-x));
		}

		fn silu(x: f32) -> f32 {
			return x * sigmoid(x);
		}

		fn d_silu(x: f32) -> f32 {
			let sig = sigmoid(x);
			return sig * (1.0 + x * (1.0 - sig));
		}

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let total = SEQ_LEN * INPUT_SIZE;
			if (idx >= total) { return; }

			let s = idx / INPUT_SIZE;
			let j = idx %% INPUT_SIZE;

			// First: compute d_intermediate from d_output
			// d_intermediate[s,i] = sum_j(d_output[s,j] * down_w[i,j])
			// Then compute d_gate and d_up from d_intermediate
			// Finally d_input from d_gate and d_up

			var d_in: f32 = 0.0;

			for (var i: u32 = 0u; i < INTER_SIZE; i++) {
				// d_intermediate[s,i] = sum_k(d_output[s,k] * down_w[i,k])
				var d_inter: f32 = 0.0;
				for (var k: u32 = 0u; k < INPUT_SIZE; k++) {
					d_inter += d_output[s * INPUT_SIZE + k] * down_w[i * INPUT_SIZE + k];
				}

				// d_gate = d_inter * up_out * d_silu(gate_out)
				let gate_val = gate_out[s * INTER_SIZE + i];
				let up_val = up_out[s * INTER_SIZE + i];
				let d_gate = d_inter * up_val * d_silu(gate_val);
				let d_up = d_inter * silu(gate_val);

				// d_input[s,j] += d_gate * gate_w[j,i] + d_up * up_w[j,i]
				d_in += d_gate * gate_w[j * INTER_SIZE + i];
				d_in += d_up * up_w[j * INTER_SIZE + i];
			}

			d_input[idx] = d_in;
		}
	`, l.BatchSize, l.Spec.InputSize, l.Spec.IntermediateSize)
}

func (l *SwiGLULayer) Compile(ctx *Context, labelPrefix string) error {
	var err error

	// Gate+Up pipeline
	mod1, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_GateUp",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateGateUpShader()},
	})
	if err != nil {
		return err
	}
	defer mod1.Release()

	// Explicit Layout 1
	bgl1, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: labelPrefix + "_GateUpBGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Input
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // GateW
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // UpW
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // GateB
			{Binding: 4, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // UpB
			{Binding: 5, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // GateOut
			{Binding: 6, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // UpOut
			{Binding: 7, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},         // Params
		},
	})
	if err != nil {
		return err
	}
	pl1, err := ctx.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            labelPrefix + "_GateUpPL",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgl1},
	})
	if err != nil {
		return err
	}
	l.pipelineGateUp, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_GateUpPipe",
		Layout:  pl1,
		Compute: wgpu.ProgrammableStageDescriptor{Module: mod1, EntryPoint: "main"},
	})
	if err != nil {
		return err
	}

	// Activate pipeline
	mod2, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_Activate",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateActivateShader()},
	})
	if err != nil {
		return err
	}
	defer mod2.Release()

	// Explicit Layout 2
	bgl2, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: labelPrefix + "_ActivateBGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // GateOut
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // UpOut
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // Intermediate
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},         // Params
		},
	})
	if err != nil {
		return err
	}
	pl2, err := ctx.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            labelPrefix + "_ActivatePL",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgl2},
	})
	if err != nil {
		return err
	}
	l.pipelineActivate, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_ActivatePipe",
		Layout:  pl2,
		Compute: wgpu.ProgrammableStageDescriptor{Module: mod2, EntryPoint: "main"},
	})
	if err != nil {
		return err
	}

	// Down pipeline
	mod3, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_Down",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateDownShader()},
	})
	if err != nil {
		return err
	}
	defer mod3.Release()

	// Explicit Layout 3
	bgl3, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: labelPrefix + "_DownBGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Intermediate
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // DownW
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // DownB
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // Output
			{Binding: 4, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},         // Params
		},
	})
	if err != nil {
		return err
	}
	pl3, err := ctx.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            labelPrefix + "_DownPL",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgl3},
	})
	if err != nil {
		return err
	}
	l.pipelineDown, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_DownPipe",
		Layout:  pl3,
		Compute: wgpu.ProgrammableStageDescriptor{Module: mod3, EntryPoint: "main"},
	})
	return err
}

func (l *SwiGLULayer) CompileBackward(ctx *Context, labelPrefix string) error {
	mod, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_Bwd",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateBackwardShader()},
	})
	if err != nil {
		return err
	}
	defer mod.Release()

	// Explicit Layout Backward
	bglBwd, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: labelPrefix + "_BwdBGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // dOutput
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // GateW
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // UpW
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // DownW
			{Binding: 4, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // GateOut
			{Binding: 5, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // UpOut
			{Binding: 6, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // dInput
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
		Compute: wgpu.ProgrammableStageDescriptor{Module: mod, EntryPoint: "main"},
	})
	return err
}

func (l *SwiGLULayer) CreateBindGroup(ctx *Context, labelPrefix string) error {
	var err error

	// Gate+Up bind group
	l.bindGroupGateUp, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_GateUpBind",
		Layout: l.pipelineGateUp.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
			{Binding: 1, Buffer: l.GateWeightBuffer, Size: l.GateWeightBuffer.GetSize()},
			{Binding: 2, Buffer: l.UpWeightBuffer, Size: l.UpWeightBuffer.GetSize()},
			{Binding: 3, Buffer: l.GateBiasBuffer, Size: l.GateBiasBuffer.GetSize()},
			{Binding: 4, Buffer: l.UpBiasBuffer, Size: l.UpBiasBuffer.GetSize()},
			{Binding: 5, Buffer: l.GateOutBuffer, Size: l.GateOutBuffer.GetSize()},
			{Binding: 6, Buffer: l.UpOutBuffer, Size: l.UpOutBuffer.GetSize()},
			{Binding: 7, Buffer: l.ParamsBuffer, Size: l.ParamsBuffer.GetSize()},
		},
	})
	if err != nil {
		return err
	}

	// Activate bind group
	l.bindGroupActivate, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_ActivateBind",
		Layout: l.pipelineActivate.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: l.GateOutBuffer, Size: l.GateOutBuffer.GetSize()},
			{Binding: 1, Buffer: l.UpOutBuffer, Size: l.UpOutBuffer.GetSize()},
			{Binding: 2, Buffer: l.IntermediateBuffer, Size: l.IntermediateBuffer.GetSize()},
			{Binding: 3, Buffer: l.ParamsBuffer, Size: l.ParamsBuffer.GetSize()},
		},
	})
	if err != nil {
		return err
	}

	// Down bind group
	l.bindGroupDown, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_DownBind",
		Layout: l.pipelineDown.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: l.IntermediateBuffer, Size: l.IntermediateBuffer.GetSize()},
			{Binding: 1, Buffer: l.DownWeightBuffer, Size: l.DownWeightBuffer.GetSize()},
			{Binding: 2, Buffer: l.DownBiasBuffer, Size: l.DownBiasBuffer.GetSize()},
			{Binding: 3, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
			{Binding: 4, Buffer: l.ParamsBuffer, Size: l.ParamsBuffer.GetSize()},
		},
	})
	return err
}

func (l *SwiGLULayer) CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error {
	var err error
	l.bwBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_BwdBind",
		Layout: l.bwPipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
			{Binding: 1, Buffer: l.GateWeightBuffer, Size: l.GateWeightBuffer.GetSize()},
			{Binding: 2, Buffer: l.UpWeightBuffer, Size: l.UpWeightBuffer.GetSize()},
			{Binding: 3, Buffer: l.DownWeightBuffer, Size: l.DownWeightBuffer.GetSize()},
			{Binding: 4, Buffer: l.GateOutBuffer, Size: l.GateOutBuffer.GetSize()},
			{Binding: 5, Buffer: l.UpOutBuffer, Size: l.UpOutBuffer.GetSize()},
			{Binding: 6, Buffer: l.InputGradientBuffer, Size: l.InputGradientBuffer.GetSize()},
		},
	})
	return err
}

func (l *SwiGLULayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	batch := l.BatchSize
	if batch < 1 {
		batch = 1
	}
	interTotal := batch * l.Spec.IntermediateSize
	inputTotal := batch * l.Spec.InputSize

	// Stage 1: Gate + Up projections
	pass.SetPipeline(l.pipelineGateUp)
	pass.SetBindGroup(0, l.bindGroupGateUp, nil)
	pass.DispatchWorkgroups(uint32((interTotal+255)/256), 1, 1)

	// Stage 2: Activate
	pass.SetPipeline(l.pipelineActivate)
	pass.SetBindGroup(0, l.bindGroupActivate, nil)
	pass.DispatchWorkgroups(uint32((interTotal+255)/256), 1, 1)

	// Stage 3: Down projection
	pass.SetPipeline(l.pipelineDown)
	pass.SetBindGroup(0, l.bindGroupDown, nil)
	pass.DispatchWorkgroups(uint32((inputTotal+255)/256), 1, 1)
}

func (l *SwiGLULayer) UpdateParams(ctx *Context, inputLen int, cachePos int) {
	if inputLen > 0 {
		l.BatchSize = inputLen
		if l.ParamsBuffer != nil {
			ctx.Queue.WriteBuffer(l.ParamsBuffer, 0, wgpu.ToBytes([]uint32{uint32(inputLen)}))
		}
	}
}

func (l *SwiGLULayer) DispatchFull(enc *wgpu.CommandEncoder) {
	batch := l.BatchSize
	if batch < 1 {
		batch = 1
	}
	interTotal := batch * l.Spec.IntermediateSize
	inputTotal := batch * l.Spec.InputSize

	// Stage 1
	pass1 := enc.BeginComputePass(nil)
	pass1.SetPipeline(l.pipelineGateUp)
	pass1.SetBindGroup(0, l.bindGroupGateUp, nil)
	pass1.DispatchWorkgroups(uint32((interTotal+255)/256), 1, 1)
	pass1.End()

	// Stage 2
	pass2 := enc.BeginComputePass(nil)
	pass2.SetPipeline(l.pipelineActivate)
	pass2.SetBindGroup(0, l.bindGroupActivate, nil)
	pass2.DispatchWorkgroups(uint32((interTotal+255)/256), 1, 1)
	pass2.End()

	// Stage 3
	pass3 := enc.BeginComputePass(nil)
	pass3.SetPipeline(l.pipelineDown)
	pass3.SetBindGroup(0, l.bindGroupDown, nil)
	pass3.DispatchWorkgroups(uint32((inputTotal+255)/256), 1, 1)
	pass3.End()
}

func (l *SwiGLULayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(l.bwPipeline)
	pass.SetBindGroup(0, l.bwBindGroup, nil)
	batch := l.BatchSize
	if batch < 1 {
		batch = 1
	}
	total := batch * l.Spec.InputSize
	pass.DispatchWorkgroups(uint32((total+255)/256), 1, 1)
	pass.End()
}

func (l *SwiGLULayer) UploadWeights(ctx *Context) {
	if len(l.Spec.GateWeights) > 0 {
		ctx.Queue.WriteBuffer(l.GateWeightBuffer, 0, wgpu.ToBytes(l.Spec.GateWeights))
	}
	if len(l.Spec.UpWeights) > 0 {
		ctx.Queue.WriteBuffer(l.UpWeightBuffer, 0, wgpu.ToBytes(l.Spec.UpWeights))
	}
	if len(l.Spec.DownWeights) > 0 {
		ctx.Queue.WriteBuffer(l.DownWeightBuffer, 0, wgpu.ToBytes(l.Spec.DownWeights))
	}
}

func (l *SwiGLULayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	return nil, nil, nil // Complex multi-weight, skip for now
}

func (l *SwiGLULayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	batch := l.BatchSize
	if batch < 1 {
		batch = 1
	}
	iGrad, err := ReadBuffer(l.InputGradientBuffer, batch*l.Spec.InputSize)
	return nil, nil, iGrad, err
}

func (l *SwiGLULayer) Cleanup() {
	if l.InputBuffer != nil && !l.InputAliased {
		l.InputBuffer.Destroy()
	}
	if l.OutputBuffer != nil {
		l.OutputBuffer.Destroy()
	}
	bufs := []*wgpu.Buffer{
		l.StagingBuffer,
		l.GateWeightBuffer, l.UpWeightBuffer, l.DownWeightBuffer,
		l.GateBiasBuffer, l.UpBiasBuffer, l.DownBiasBuffer,
		l.GateOutBuffer, l.UpOutBuffer, l.IntermediateBuffer,
		l.InputGradientBuffer, l.ParamsBuffer,
	}
	for _, b := range bufs {
		if b != nil {
			b.Destroy()
		}
	}

	pipes := []*wgpu.ComputePipeline{l.pipelineGateUp, l.pipelineActivate, l.pipelineDown, l.bwPipeline}
	for _, p := range pipes {
		if p != nil {
			p.Release()
		}
	}

	bgs := []*wgpu.BindGroup{l.bindGroupGateUp, l.bindGroupActivate, l.bindGroupDown, l.bwBindGroup}
	for _, bg := range bgs {
		if bg != nil {
			bg.Release()
		}
	}
}
