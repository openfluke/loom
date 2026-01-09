package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// RNNSpec defines configuration for RNN layer
type RNNSpec struct {
	InputSize  int       // Input feature size
	HiddenSize int       // Hidden state size
	SeqLen     int       // Sequence length
	WeightIH   []float32 // Input-to-hidden weights [HiddenSize * InputSize]
	WeightHH   []float32 // Hidden-to-hidden weights [HiddenSize * HiddenSize]
	BiasH      []float32 // Hidden bias [HiddenSize]
}

// RNNLayer holds GPU resources for RNN
// Note: RNN is inherently sequential across time steps but parallel within each step
// We process one time step per dispatch to avoid cross-workgroup synchronization issues
type RNNLayer struct {
	Spec RNNSpec

	pipeline   *wgpu.ComputePipeline
	bindGroups []*wgpu.BindGroup // One bind group per time step

	InputBuffer    *wgpu.Buffer // [SeqLen * InputSize]
	OutputBuffer   *wgpu.Buffer // [SeqLen * HiddenSize]
	StagingBuffer  *wgpu.Buffer
	HiddenBuffer   *wgpu.Buffer   // Current hidden state [HiddenSize]
	StepBuffers    []*wgpu.Buffer // Uniform buffers for each step
	WeightIHBuffer *wgpu.Buffer
	WeightHHBuffer *wgpu.Buffer
	BiasBuffer     *wgpu.Buffer

	InputGradientBuffer *wgpu.Buffer
	bwPipeline          *wgpu.ComputePipeline
	bwBindGroup         *wgpu.BindGroup
}

func (l *RNNLayer) GetInputBuffer() *wgpu.Buffer         { return l.InputBuffer }
func (l *RNNLayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *RNNLayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *RNNLayer) GetInputGradientBuffer() *wgpu.Buffer { return l.InputGradientBuffer }

func (l *RNNLayer) AllocateBuffers(ctx *Context, labelPrefix string) error {
	var err error

	// Validate parameters
	if l.Spec.SeqLen <= 0 {
		l.Spec.SeqLen = 32
	}
	if l.Spec.InputSize <= 0 {
		l.Spec.InputSize = 64
	}
	if l.Spec.HiddenSize <= 0 {
		l.Spec.HiddenSize = 64
	}

	inputTotal := l.Spec.SeqLen * l.Spec.InputSize
	if inputTotal < 1 {
		inputTotal = 1
	}
	outputTotal := l.Spec.SeqLen * l.Spec.HiddenSize
	if outputTotal < 1 {
		outputTotal = 1
	}

	hiddenSize := l.Spec.HiddenSize
	if hiddenSize < 1 {
		hiddenSize = 1
	}

	l.InputBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_In",
		Size:  uint64(inputTotal * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	l.OutputBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Out",
		Size:  uint64(outputTotal * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	l.HiddenBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Hidden",
		Size:  uint64(hiddenSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Weights
	ihSize := l.Spec.HiddenSize * l.Spec.InputSize
	if ihSize < 1 {
		ihSize = 1
	}
	hhSize := l.Spec.HiddenSize * l.Spec.HiddenSize
	if hhSize < 1 {
		hhSize = 1
	}

	// Ensure weight arrays match expected sizes
	if len(l.Spec.WeightIH) != ihSize {
		l.Spec.WeightIH = make([]float32, ihSize)
		for i := range l.Spec.WeightIH {
			l.Spec.WeightIH[i] = float32(i%100) * 0.01
		}
	}
	l.WeightIHBuffer, err = NewFloatBuffer(l.Spec.WeightIH, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}

	if len(l.Spec.WeightHH) != hhSize {
		l.Spec.WeightHH = make([]float32, hhSize)
		for i := range l.Spec.WeightHH {
			l.Spec.WeightHH[i] = float32(i%100) * 0.01
		}
	}
	l.WeightHHBuffer, err = NewFloatBuffer(l.Spec.WeightHH, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}

	// Ensure bias array matches expected size
	if len(l.Spec.BiasH) != l.Spec.HiddenSize {
		l.Spec.BiasH = make([]float32, l.Spec.HiddenSize)
	}
	l.BiasBuffer, err = NewFloatBuffer(l.Spec.BiasH, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}

	// Create step uniform buffers - one per time step
	l.StepBuffers = make([]*wgpu.Buffer, l.Spec.SeqLen)
	for step := 0; step < l.Spec.SeqLen; step++ {
		stepData := []uint32{uint32(step)}
		l.StepBuffers[step], err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: fmt.Sprintf("%s_Step%d", labelPrefix, step),
			Size:  4, // u32
			Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			return err
		}
		ctx.Queue.WriteBuffer(l.StepBuffers[step], 0, wgpu.ToBytes(stepData))
	}

	l.StagingBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Staging",
		Size:  uint64(outputTotal * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	return err
}

func (l *RNNLayer) AllocateBackwardBuffers(ctx *Context, labelPrefix string) error {
	var err error
	l.InputGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_InGrad",
		Size:  uint64(l.Spec.SeqLen * l.Spec.InputSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	return err
}

func (l *RNNLayer) GenerateShader() string {
	// RNN: h' = tanh(W_ih * x + W_hh * h + b)
	// Process ONE time step per dispatch - this avoids the cross-workgroup sync issue
	// The Go code will dispatch SeqLen times, once per time step
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read> w_ih : array<f32>;
		@group(0) @binding(2) var<storage, read> w_hh : array<f32>;
		@group(0) @binding(3) var<storage, read> bias : array<f32>;
		@group(0) @binding(4) var<storage, read_write> hidden : array<f32>;
		@group(0) @binding(5) var<storage, read_write> output : array<f32>;
		@group(0) @binding(6) var<uniform> step : u32;

		const INPUT_SIZE: u32 = %du;
		const HIDDEN_SIZE: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let h_idx = gid.x;
			if (h_idx >= HIDDEN_SIZE) { return; }

			let input_offset = step * INPUT_SIZE;
			
			// W_ih * x
			var sum: f32 = bias[h_idx];
			for (var i: u32 = 0u; i < INPUT_SIZE; i++) {
				sum += input[input_offset + i] * w_ih[h_idx * INPUT_SIZE + i];
			}

			// W_hh * h (read from hidden buffer - synchronized between dispatches)
			for (var i: u32 = 0u; i < HIDDEN_SIZE; i++) {
				sum += hidden[i] * w_hh[h_idx * HIDDEN_SIZE + i];
			}

			// tanh activation
			let h_val = tanh(sum);
			
			// Update hidden state and output
			hidden[h_idx] = h_val;
			output[step * HIDDEN_SIZE + h_idx] = h_val;
		}
	`, l.Spec.InputSize, l.Spec.HiddenSize)
}

func (l *RNNLayer) GenerateBackwardShader() string {
	// Simplified BPTT: d_input = d_h @ W_ih.T where d_h = d_output * (1 - h^2)
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> d_output : array<f32>;
		@group(0) @binding(1) var<storage, read> output : array<f32>;
		@group(0) @binding(2) var<storage, read> w_ih : array<f32>;
		@group(0) @binding(3) var<storage, read_write> d_input : array<f32>;

		const SEQ_LEN: u32 = %du;
		const INPUT_SIZE: u32 = %du;
		const HIDDEN_SIZE: u32 = %du;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let total = SEQ_LEN * INPUT_SIZE;
			if (idx >= total) { return; }

			let t = idx / INPUT_SIZE;
			let j = idx %% INPUT_SIZE;

			var grad: f32 = 0.0;

			// For each hidden unit, backprop through tanh
			for (var h: u32 = 0u; h < HIDDEN_SIZE; h++) {
				let h_val = output[t * HIDDEN_SIZE + h];
				let d_tanh = 1.0 - h_val * h_val;  // derivative of tanh
				let d_h = d_output[t * HIDDEN_SIZE + h] * d_tanh;
				
				// d_input[t, j] += d_h * w_ih[h, j]
				grad += d_h * w_ih[h * INPUT_SIZE + j];
			}

			d_input[idx] = grad;
		}
	`, l.Spec.SeqLen, l.Spec.InputSize, l.Spec.HiddenSize)
}

func (l *RNNLayer) Compile(ctx *Context, labelPrefix string) error {
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

func (l *RNNLayer) CompileBackward(ctx *Context, labelPrefix string) error {
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

func (l *RNNLayer) CreateBindGroup(ctx *Context, labelPrefix string) error {
	// Create one bind group per time step, each with its own step uniform buffer
	l.bindGroups = make([]*wgpu.BindGroup, l.Spec.SeqLen)
	var err error
	for step := 0; step < l.Spec.SeqLen; step++ {
		l.bindGroups[step], err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  fmt.Sprintf("%s_Bind%d", labelPrefix, step),
			Layout: l.pipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
				{Binding: 1, Buffer: l.WeightIHBuffer, Size: l.WeightIHBuffer.GetSize()},
				{Binding: 2, Buffer: l.WeightHHBuffer, Size: l.WeightHHBuffer.GetSize()},
				{Binding: 3, Buffer: l.BiasBuffer, Size: l.BiasBuffer.GetSize()},
				{Binding: 4, Buffer: l.HiddenBuffer, Size: l.HiddenBuffer.GetSize()},
				{Binding: 5, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
				{Binding: 6, Buffer: l.StepBuffers[step], Size: 4},
			},
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func (l *RNNLayer) CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error {
	var err error
	l.bwBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_BwdBind",
		Layout: l.bwPipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
			{Binding: 1, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
			{Binding: 2, Buffer: l.WeightIHBuffer, Size: l.WeightIHBuffer.GetSize()},
			{Binding: 3, Buffer: l.InputGradientBuffer, Size: l.InputGradientBuffer.GetSize()},
		},
	})
	return err
}

func (l *RNNLayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	// Dispatch once per time step - each step processes in parallel within hidden units
	// but steps must be sequential due to hidden state dependency
	wg := uint32((l.Spec.HiddenSize + 255) / 256)
	for step := 0; step < l.Spec.SeqLen; step++ {
		pass.SetPipeline(l.pipeline)
		pass.SetBindGroup(0, l.bindGroups[step], nil)
		pass.DispatchWorkgroups(wg, 1, 1)
	}
}

func (l *RNNLayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(l.bwPipeline)
	pass.SetBindGroup(0, l.bwBindGroup, nil)
	total := l.Spec.SeqLen * l.Spec.InputSize
	pass.DispatchWorkgroups(uint32((total+255)/256), 1, 1)
	pass.End()
}

func (l *RNNLayer) UploadWeights(ctx *Context) {
	if len(l.Spec.WeightIH) > 0 {
		ctx.Queue.WriteBuffer(l.WeightIHBuffer, 0, wgpu.ToBytes(l.Spec.WeightIH))
	}
	if len(l.Spec.WeightHH) > 0 {
		ctx.Queue.WriteBuffer(l.WeightHHBuffer, 0, wgpu.ToBytes(l.Spec.WeightHH))
	}
}

func (l *RNNLayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	return nil, nil, nil
}

func (l *RNNLayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	iGrad, err := ReadBuffer(l.InputGradientBuffer, l.Spec.SeqLen*l.Spec.InputSize)
	return nil, nil, iGrad, err
}

func (l *RNNLayer) Cleanup() {
	bufs := []*wgpu.Buffer{l.InputBuffer, l.OutputBuffer, l.StagingBuffer, l.HiddenBuffer, l.WeightIHBuffer, l.WeightHHBuffer, l.BiasBuffer, l.InputGradientBuffer}
	for _, b := range bufs {
		if b != nil {
			b.Destroy()
		}
	}
	// Clean up step buffers
	for _, b := range l.StepBuffers {
		if b != nil {
			b.Destroy()
		}
	}
	if l.pipeline != nil {
		l.pipeline.Release()
	}
	// Clean up per-step bind groups
	for _, bg := range l.bindGroups {
		if bg != nil {
			bg.Release()
		}
	}
	if l.bwPipeline != nil {
		l.bwPipeline.Release()
	}
	if l.bwBindGroup != nil {
		l.bwBindGroup.Release()
	}
}
