package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// LSTMSpec defines configuration for LSTM layer
type LSTMSpec struct {
	InputSize  int // Input feature size
	HiddenSize int // Hidden state size
	SeqLen     int // Sequence length

	// 4 gates: input (i), forget (f), cell (g), output (o)
	WeightIH_i []float32 // [HiddenSize * InputSize]
	WeightHH_i []float32 // [HiddenSize * HiddenSize]
	BiasH_i    []float32 // [HiddenSize]

	WeightIH_f []float32
	WeightHH_f []float32
	BiasH_f    []float32

	WeightIH_g []float32
	WeightHH_g []float32
	BiasH_g    []float32

	WeightIH_o []float32
	WeightHH_o []float32
	BiasH_o    []float32
}

// LSTMLayer holds GPU resources for LSTM
// Note: LSTM is inherently sequential across time steps but parallel within each step
// We process one time step per dispatch to avoid cross-workgroup synchronization issues
type LSTMLayer struct {
	Spec LSTMSpec

	pipeline   *wgpu.ComputePipeline
	bindGroups []*wgpu.BindGroup // One bind group per time step

	InputBuffer   *wgpu.Buffer // [SeqLen * InputSize]
	OutputBuffer  *wgpu.Buffer // [SeqLen * HiddenSize]
	StagingBuffer *wgpu.Buffer
	HiddenBuffer  *wgpu.Buffer   // Hidden state [HiddenSize]
	CellBuffer    *wgpu.Buffer   // Cell state [HiddenSize]
	StepBuffers   []*wgpu.Buffer // Uniform buffers for each step

	// Combined Weight buffers (concatenated [i, f, g, o])
	CombinedWeightsIH *wgpu.Buffer
	CombinedWeightsHH *wgpu.Buffer
	CombinedBiases    *wgpu.Buffer

	InputGradientBuffer *wgpu.Buffer
	bwPipeline          *wgpu.ComputePipeline
	bwBindGroup         *wgpu.BindGroup
}

func (l *LSTMLayer) GetInputBuffer() *wgpu.Buffer         { return l.InputBuffer }
func (l *LSTMLayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *LSTMLayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *LSTMLayer) GetInputGradientBuffer() *wgpu.Buffer { return l.InputGradientBuffer }

func (l *LSTMLayer) AllocateBuffers(ctx *Context, labelPrefix string) error {
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

	l.CellBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Cell",
		Size:  uint64(hiddenSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	l.StagingBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Staging",
		Size:  uint64(outputTotal * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}

	ihSize := l.Spec.HiddenSize * l.Spec.InputSize
	if ihSize < 1 {
		ihSize = 1
	}
	hhSize := l.Spec.HiddenSize * l.Spec.HiddenSize
	if hhSize < 1 {
		hhSize = 1
	}

	// Combine weights: i, f, g, o
	combine := func(w1, w2, w3, w4 []float32, size int) []float32 {
		res := make([]float32, size*4)
		s := size
		if len(w1) > 0 {
			copy(res[0:s], w1)
		}
		if len(w2) > 0 {
			copy(res[s:2*s], w2)
		}
		if len(w3) > 0 {
			copy(res[2*s:3*s], w3)
		}
		if len(w4) > 0 {
			copy(res[3*s:4*s], w4)
		}
		return res
	}

	ihData := combine(l.Spec.WeightIH_i, l.Spec.WeightIH_f, l.Spec.WeightIH_g, l.Spec.WeightIH_o, ihSize)
	l.CombinedWeightsIH, err = NewFloatBuffer(ihData, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}

	hhData := combine(l.Spec.WeightHH_i, l.Spec.WeightHH_f, l.Spec.WeightHH_g, l.Spec.WeightHH_o, hhSize)
	l.CombinedWeightsHH, err = NewFloatBuffer(hhData, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}

	biasData := combine(l.Spec.BiasH_i, l.Spec.BiasH_f, l.Spec.BiasH_g, l.Spec.BiasH_o, l.Spec.HiddenSize)
	l.CombinedBiases, err = NewFloatBuffer(biasData, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
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

	return nil
}

func (l *LSTMLayer) AllocateBackwardBuffers(ctx *Context, labelPrefix string) error {
	var err error
	l.InputGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_InGrad",
		Size:  uint64(l.Spec.SeqLen * l.Spec.InputSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	return err
}

func (l *LSTMLayer) GenerateShader() string {
	// LSTM: computes all 4 gates and updates h, c for ONE time step
	// Process ONE time step per dispatch - this avoids the cross-workgroup sync issue
	// The Go code will dispatch SeqLen times, once per time step
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read_write> hidden : array<f32>;
		@group(0) @binding(2) var<storage, read_write> cell : array<f32>;
		@group(0) @binding(3) var<storage, read_write> output : array<f32>;
		
		// Gate weights (4 concatenated weight sets: i, f, g, o)
		@group(0) @binding(4) var<storage, read> w_ih : array<f32>;
		@group(0) @binding(5) var<storage, read> w_hh : array<f32>;
		@group(0) @binding(6) var<storage, read> bias : array<f32>;
		@group(0) @binding(7) var<uniform> step : u32;

		const INPUT_SIZE: u32 = %du;
		const HIDDEN_SIZE: u32 = %du;

		fn sigmoid(x: f32) -> f32 {
			return 1.0 / (1.0 + exp(-x));
		}

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let h_idx = gid.x;
			if (h_idx >= HIDDEN_SIZE) { return; }

			// Read current hidden and cell states
			let h_val: f32 = hidden[h_idx];
			let c_val: f32 = cell[h_idx];

			// Weight offsets for each gate
			let IH_STRIDE: u32 = INPUT_SIZE * HIDDEN_SIZE;
			let HH_STRIDE: u32 = HIDDEN_SIZE * HIDDEN_SIZE;

			let input_offset = step * INPUT_SIZE;

			// Compute input gate (i)
			var i_gate: f32 = bias[h_idx];
			for (var j: u32 = 0u; j < INPUT_SIZE; j++) {
				i_gate += input[input_offset + j] * w_ih[h_idx * INPUT_SIZE + j];
			}
			for (var j: u32 = 0u; j < HIDDEN_SIZE; j++) {
				i_gate += hidden[j] * w_hh[h_idx * HIDDEN_SIZE + j];
			}
			i_gate = sigmoid(i_gate);

			// Compute forget gate (f)
			var f_gate: f32 = bias[HIDDEN_SIZE + h_idx];
			for (var j: u32 = 0u; j < INPUT_SIZE; j++) {
				f_gate += input[input_offset + j] * w_ih[IH_STRIDE + h_idx * INPUT_SIZE + j];
			}
			for (var j: u32 = 0u; j < HIDDEN_SIZE; j++) {
				f_gate += hidden[j] * w_hh[HH_STRIDE + h_idx * HIDDEN_SIZE + j];
			}
			f_gate = sigmoid(f_gate);

			// Compute cell gate (g)
			var g_gate: f32 = bias[2u * HIDDEN_SIZE + h_idx];
			for (var j: u32 = 0u; j < INPUT_SIZE; j++) {
				g_gate += input[input_offset + j] * w_ih[2u * IH_STRIDE + h_idx * INPUT_SIZE + j];
			}
			for (var j: u32 = 0u; j < HIDDEN_SIZE; j++) {
				g_gate += hidden[j] * w_hh[2u * HH_STRIDE + h_idx * HIDDEN_SIZE + j];
			}
			g_gate = tanh(g_gate);

			// Compute output gate (o)
			var o_gate: f32 = bias[3u * HIDDEN_SIZE + h_idx];
			for (var j: u32 = 0u; j < INPUT_SIZE; j++) {
				o_gate += input[input_offset + j] * w_ih[3u * IH_STRIDE + h_idx * INPUT_SIZE + j];
			}
			for (var j: u32 = 0u; j < HIDDEN_SIZE; j++) {
				o_gate += hidden[j] * w_hh[3u * HH_STRIDE + h_idx * HIDDEN_SIZE + j];
			}
			o_gate = sigmoid(o_gate);

			// Update cell and hidden states
			let new_c = f_gate * c_val + i_gate * g_gate;
			let new_h = o_gate * tanh(new_c);

			// Store to buffers
			cell[h_idx] = new_c;
			hidden[h_idx] = new_h;
			output[step * HIDDEN_SIZE + h_idx] = new_h;
		}
	`, l.Spec.InputSize, l.Spec.HiddenSize)
}

func (l *LSTMLayer) GenerateBackwardShader() string {
	// Simplified BPTT for LSTM: d_input = sum of gradients through all 4 gate input weights
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

			// For each hidden unit, approximate gradient through all gates
			// Simplified: just use sum of weight contributions
			for (var h: u32 = 0u; h < HIDDEN_SIZE; h++) {
				let h_val = output[t * HIDDEN_SIZE + h];
				// Approximate derivative (simplified)
				let d_h = d_output[t * HIDDEN_SIZE + h];
				
				let ih_step = INPUT_SIZE * HIDDEN_SIZE;
				
				grad += d_h * w_ih[h * INPUT_SIZE + j];           // i
				grad += d_h * w_ih[ih_step + h * INPUT_SIZE + j]; // f
				grad += d_h * w_ih[2u*ih_step + h * INPUT_SIZE + j]; // g
				grad += d_h * w_ih[3u*ih_step + h * INPUT_SIZE + j]; // o
			}

			d_input[idx] = grad * 0.25;  // Average over 4 gates
		}
	`, l.Spec.SeqLen, l.Spec.InputSize, l.Spec.HiddenSize)
}

func (l *LSTMLayer) Compile(ctx *Context, labelPrefix string) error {
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

func (l *LSTMLayer) CompileBackward(ctx *Context, labelPrefix string) error {
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

func (l *LSTMLayer) CreateBindGroup(ctx *Context, labelPrefix string) error {
	// Create one bind group per time step, each with its own step uniform buffer
	l.bindGroups = make([]*wgpu.BindGroup, l.Spec.SeqLen)
	var err error
	for step := 0; step < l.Spec.SeqLen; step++ {
		l.bindGroups[step], err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  fmt.Sprintf("%s_Bind%d", labelPrefix, step),
			Layout: l.pipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
				{Binding: 1, Buffer: l.HiddenBuffer, Size: l.HiddenBuffer.GetSize()},
				{Binding: 2, Buffer: l.CellBuffer, Size: l.CellBuffer.GetSize()},
				{Binding: 3, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
				{Binding: 4, Buffer: l.CombinedWeightsIH, Size: l.CombinedWeightsIH.GetSize()},
				{Binding: 5, Buffer: l.CombinedWeightsHH, Size: l.CombinedWeightsHH.GetSize()},
				{Binding: 6, Buffer: l.CombinedBiases, Size: l.CombinedBiases.GetSize()},
				{Binding: 7, Buffer: l.StepBuffers[step], Size: 4},
			},
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func (l *LSTMLayer) CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error {
	var err error
	l.bwBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_BwdBind",
		Layout: l.bwPipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
			{Binding: 1, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
			{Binding: 2, Buffer: l.CombinedWeightsIH, Size: l.CombinedWeightsIH.GetSize()},
			{Binding: 3, Buffer: l.InputGradientBuffer, Size: l.InputGradientBuffer.GetSize()},
		},
	})
	return err
}

func (l *LSTMLayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	// Dispatch once per time step - each step processes in parallel within hidden units
	// but steps must be sequential due to hidden state dependency
	wg := uint32((l.Spec.HiddenSize + 255) / 256)
	for step := 0; step < l.Spec.SeqLen; step++ {
		pass.SetPipeline(l.pipeline)
		pass.SetBindGroup(0, l.bindGroups[step], nil)
		pass.DispatchWorkgroups(wg, 1, 1)
	}
}

func (l *LSTMLayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(l.bwPipeline)
	pass.SetBindGroup(0, l.bwBindGroup, nil)
	total := l.Spec.SeqLen * l.Spec.InputSize
	pass.DispatchWorkgroups(uint32((total+255)/256), 1, 1)
	pass.End()
}

func (l *LSTMLayer) UploadWeights(ctx *Context) {
	// Already uploaded in AllocateBuffers via NewFloatBuffer with data
	// If re-upload needed:
}

func (l *LSTMLayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	return nil, nil, nil
}

func (l *LSTMLayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	iGrad, err := ReadBuffer(l.InputGradientBuffer, l.Spec.SeqLen*l.Spec.InputSize)
	return nil, nil, iGrad, err
}

func (l *LSTMLayer) Cleanup() {
	bufs := []*wgpu.Buffer{
		l.InputBuffer, l.OutputBuffer, l.StagingBuffer,
		l.HiddenBuffer, l.CellBuffer,
		l.CombinedWeightsIH, l.CombinedWeightsHH, l.CombinedBiases,
		l.InputGradientBuffer,
	}
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
