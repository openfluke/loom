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
type LSTMLayer struct {
	Spec LSTMSpec

	pipeline  *wgpu.ComputePipeline
	bindGroup *wgpu.BindGroup

	InputBuffer   *wgpu.Buffer // [SeqLen * InputSize]
	OutputBuffer  *wgpu.Buffer // [SeqLen * HiddenSize]
	StagingBuffer *wgpu.Buffer
	HiddenBuffer  *wgpu.Buffer // Hidden state [HiddenSize]
	CellBuffer    *wgpu.Buffer // Cell state [HiddenSize]

	// Weight buffers for each gate
	WeightIH_i_Buffer *wgpu.Buffer
	WeightHH_i_Buffer *wgpu.Buffer
	BiasH_i_Buffer    *wgpu.Buffer

	WeightIH_f_Buffer *wgpu.Buffer
	WeightHH_f_Buffer *wgpu.Buffer
	BiasH_f_Buffer    *wgpu.Buffer

	WeightIH_g_Buffer *wgpu.Buffer
	WeightHH_g_Buffer *wgpu.Buffer
	BiasH_g_Buffer    *wgpu.Buffer

	WeightIH_o_Buffer *wgpu.Buffer
	WeightHH_o_Buffer *wgpu.Buffer
	BiasH_o_Buffer    *wgpu.Buffer

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

	inputTotal := l.Spec.SeqLen * l.Spec.InputSize
	outputTotal := l.Spec.SeqLen * l.Spec.HiddenSize

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
		Label: labelPrefix + "_H",
		Size:  uint64(l.Spec.HiddenSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	l.CellBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_C",
		Size:  uint64(l.Spec.HiddenSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	ihSize := l.Spec.HiddenSize * l.Spec.InputSize
	hhSize := l.Spec.HiddenSize * l.Spec.HiddenSize

	allocW := func(data []float32, size int) (*wgpu.Buffer, error) {
		if len(data) > 0 {
			return NewFloatBuffer(data, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
		}
		return NewFloatBuffer(make([]float32, size), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	}

	allocB := func(data []float32) (*wgpu.Buffer, error) {
		if len(data) > 0 {
			return NewFloatBuffer(data, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
		}
		return NewFloatBuffer(make([]float32, l.Spec.HiddenSize), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	}

	// Input gate
	l.WeightIH_i_Buffer, err = allocW(l.Spec.WeightIH_i, ihSize)
	if err != nil {
		return err
	}
	l.WeightHH_i_Buffer, err = allocW(l.Spec.WeightHH_i, hhSize)
	if err != nil {
		return err
	}
	l.BiasH_i_Buffer, err = allocB(l.Spec.BiasH_i)
	if err != nil {
		return err
	}

	// Forget gate
	l.WeightIH_f_Buffer, err = allocW(l.Spec.WeightIH_f, ihSize)
	if err != nil {
		return err
	}
	l.WeightHH_f_Buffer, err = allocW(l.Spec.WeightHH_f, hhSize)
	if err != nil {
		return err
	}
	l.BiasH_f_Buffer, err = allocB(l.Spec.BiasH_f)
	if err != nil {
		return err
	}

	// Cell gate
	l.WeightIH_g_Buffer, err = allocW(l.Spec.WeightIH_g, ihSize)
	if err != nil {
		return err
	}
	l.WeightHH_g_Buffer, err = allocW(l.Spec.WeightHH_g, hhSize)
	if err != nil {
		return err
	}
	l.BiasH_g_Buffer, err = allocB(l.Spec.BiasH_g)
	if err != nil {
		return err
	}

	// Output gate
	l.WeightIH_o_Buffer, err = allocW(l.Spec.WeightIH_o, ihSize)
	if err != nil {
		return err
	}
	l.WeightHH_o_Buffer, err = allocW(l.Spec.WeightHH_o, hhSize)
	if err != nil {
		return err
	}
	l.BiasH_o_Buffer, err = allocB(l.Spec.BiasH_o)
	if err != nil {
		return err
	}

	l.StagingBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Staging",
		Size:  uint64(outputTotal * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	return err
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
	// LSTM step: computes all 4 gates and updates h, c
	// Note: Simplified single-step version
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read_write> hidden : array<f32>;
		@group(0) @binding(2) var<storage, read_write> cell : array<f32>;
		@group(0) @binding(3) var<storage, read_write> output : array<f32>;
		
		// Gate weights (simplified - 4 concatenated weight sets)
		@group(0) @binding(4) var<storage, read> w_ih_i : array<f32>;
		@group(0) @binding(5) var<storage, read> w_hh_i : array<f32>;
		@group(0) @binding(6) var<storage, read> b_i : array<f32>;
		
		@group(0) @binding(7) var<storage, read> w_ih_f : array<f32>;
		@group(0) @binding(8) var<storage, read> w_hh_f : array<f32>;
		@group(0) @binding(9) var<storage, read> b_f : array<f32>;
		
		@group(0) @binding(10) var<storage, read> w_ih_g : array<f32>;
		@group(0) @binding(11) var<storage, read> w_hh_g : array<f32>;
		@group(0) @binding(12) var<storage, read> b_g : array<f32>;
		
		@group(0) @binding(13) var<storage, read> w_ih_o : array<f32>;
		@group(0) @binding(14) var<storage, read> w_hh_o : array<f32>;
		@group(0) @binding(15) var<storage, read> b_o : array<f32>;

		const SEQ_LEN: u32 = %du;
		const INPUT_SIZE: u32 = %du;
		const HIDDEN_SIZE: u32 = %du;

		fn sigmoid(x: f32) -> f32 {
			return 1.0 / (1.0 + exp(-x));
		}

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let h_idx = gid.x;
			if (h_idx >= HIDDEN_SIZE) { return; }

			// Process first time step for simplicity
			let step: u32 = 0u;
			let input_offset = step * INPUT_SIZE;

			// Compute input gate
			var i_gate: f32 = b_i[h_idx];
			for (var j: u32 = 0u; j < INPUT_SIZE; j++) {
				i_gate += input[input_offset + j] * w_ih_i[h_idx * INPUT_SIZE + j];
			}
			for (var j: u32 = 0u; j < HIDDEN_SIZE; j++) {
				i_gate += hidden[j] * w_hh_i[h_idx * HIDDEN_SIZE + j];
			}
			i_gate = sigmoid(i_gate);

			// Compute forget gate
			var f_gate: f32 = b_f[h_idx];
			for (var j: u32 = 0u; j < INPUT_SIZE; j++) {
				f_gate += input[input_offset + j] * w_ih_f[h_idx * INPUT_SIZE + j];
			}
			for (var j: u32 = 0u; j < HIDDEN_SIZE; j++) {
				f_gate += hidden[j] * w_hh_f[h_idx * HIDDEN_SIZE + j];
			}
			f_gate = sigmoid(f_gate);

			// Compute cell gate
			var g_gate: f32 = b_g[h_idx];
			for (var j: u32 = 0u; j < INPUT_SIZE; j++) {
				g_gate += input[input_offset + j] * w_ih_g[h_idx * INPUT_SIZE + j];
			}
			for (var j: u32 = 0u; j < HIDDEN_SIZE; j++) {
				g_gate += hidden[j] * w_hh_g[h_idx * HIDDEN_SIZE + j];
			}
			g_gate = tanh(g_gate);

			// Compute output gate
			var o_gate: f32 = b_o[h_idx];
			for (var j: u32 = 0u; j < INPUT_SIZE; j++) {
				o_gate += input[input_offset + j] * w_ih_o[h_idx * INPUT_SIZE + j];
			}
			for (var j: u32 = 0u; j < HIDDEN_SIZE; j++) {
				o_gate += hidden[j] * w_hh_o[h_idx * HIDDEN_SIZE + j];
			}
			o_gate = sigmoid(o_gate);

			// Update cell and hidden states
			let new_c = f_gate * cell[h_idx] + i_gate * g_gate;
			let new_h = o_gate * tanh(new_c);

			cell[h_idx] = new_c;
			hidden[h_idx] = new_h;
			output[step * HIDDEN_SIZE + h_idx] = new_h;
		}
	`, l.Spec.SeqLen, l.Spec.InputSize, l.Spec.HiddenSize)
}

func (l *LSTMLayer) GenerateBackwardShader() string {
	// Simplified BPTT for LSTM: d_input = sum of gradients through all 4 gate input weights
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> d_output : array<f32>;
		@group(0) @binding(1) var<storage, read> output : array<f32>;
		@group(0) @binding(2) var<storage, read> w_ih_i : array<f32>;
		@group(0) @binding(3) var<storage, read> w_ih_f : array<f32>;
		@group(0) @binding(4) var<storage, read> w_ih_g : array<f32>;
		@group(0) @binding(5) var<storage, read> w_ih_o : array<f32>;
		@group(0) @binding(6) var<storage, read_write> d_input : array<f32>;

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
				
				// Gradient flows through all 4 gates
				grad += d_h * w_ih_i[h * INPUT_SIZE + j];
				grad += d_h * w_ih_f[h * INPUT_SIZE + j];
				grad += d_h * w_ih_g[h * INPUT_SIZE + j];
				grad += d_h * w_ih_o[h * INPUT_SIZE + j];
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
	var err error
	l.bindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_Bind",
		Layout: l.pipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
			{Binding: 1, Buffer: l.HiddenBuffer, Size: l.HiddenBuffer.GetSize()},
			{Binding: 2, Buffer: l.CellBuffer, Size: l.CellBuffer.GetSize()},
			{Binding: 3, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
			{Binding: 4, Buffer: l.WeightIH_i_Buffer, Size: l.WeightIH_i_Buffer.GetSize()},
			{Binding: 5, Buffer: l.WeightHH_i_Buffer, Size: l.WeightHH_i_Buffer.GetSize()},
			{Binding: 6, Buffer: l.BiasH_i_Buffer, Size: l.BiasH_i_Buffer.GetSize()},
			{Binding: 7, Buffer: l.WeightIH_f_Buffer, Size: l.WeightIH_f_Buffer.GetSize()},
			{Binding: 8, Buffer: l.WeightHH_f_Buffer, Size: l.WeightHH_f_Buffer.GetSize()},
			{Binding: 9, Buffer: l.BiasH_f_Buffer, Size: l.BiasH_f_Buffer.GetSize()},
			{Binding: 10, Buffer: l.WeightIH_g_Buffer, Size: l.WeightIH_g_Buffer.GetSize()},
			{Binding: 11, Buffer: l.WeightHH_g_Buffer, Size: l.WeightHH_g_Buffer.GetSize()},
			{Binding: 12, Buffer: l.BiasH_g_Buffer, Size: l.BiasH_g_Buffer.GetSize()},
			{Binding: 13, Buffer: l.WeightIH_o_Buffer, Size: l.WeightIH_o_Buffer.GetSize()},
			{Binding: 14, Buffer: l.WeightHH_o_Buffer, Size: l.WeightHH_o_Buffer.GetSize()},
			{Binding: 15, Buffer: l.BiasH_o_Buffer, Size: l.BiasH_o_Buffer.GetSize()},
		},
	})
	return err
}

func (l *LSTMLayer) CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error {
	var err error
	l.bwBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_BwdBind",
		Layout: l.bwPipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
			{Binding: 1, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
			{Binding: 2, Buffer: l.WeightIH_i_Buffer, Size: l.WeightIH_i_Buffer.GetSize()},
			{Binding: 3, Buffer: l.WeightIH_f_Buffer, Size: l.WeightIH_f_Buffer.GetSize()},
			{Binding: 4, Buffer: l.WeightIH_g_Buffer, Size: l.WeightIH_g_Buffer.GetSize()},
			{Binding: 5, Buffer: l.WeightIH_o_Buffer, Size: l.WeightIH_o_Buffer.GetSize()},
			{Binding: 6, Buffer: l.InputGradientBuffer, Size: l.InputGradientBuffer.GetSize()},
		},
	})
	return err
}

func (l *LSTMLayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	pass.SetPipeline(l.pipeline)
	pass.SetBindGroup(0, l.bindGroup, nil)
	wg := uint32((l.Spec.HiddenSize + 255) / 256)
	pass.DispatchWorkgroups(wg, 1, 1)
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
	// Upload weights for all 4 gates
	upload := func(buf *wgpu.Buffer, data []float32) {
		if len(data) > 0 {
			ctx.Queue.WriteBuffer(buf, 0, wgpu.ToBytes(data))
		}
	}
	upload(l.WeightIH_i_Buffer, l.Spec.WeightIH_i)
	upload(l.WeightHH_i_Buffer, l.Spec.WeightHH_i)
	upload(l.WeightIH_f_Buffer, l.Spec.WeightIH_f)
	upload(l.WeightHH_f_Buffer, l.Spec.WeightHH_f)
	upload(l.WeightIH_g_Buffer, l.Spec.WeightIH_g)
	upload(l.WeightHH_g_Buffer, l.Spec.WeightHH_g)
	upload(l.WeightIH_o_Buffer, l.Spec.WeightIH_o)
	upload(l.WeightHH_o_Buffer, l.Spec.WeightHH_o)
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
		l.WeightIH_i_Buffer, l.WeightHH_i_Buffer, l.BiasH_i_Buffer,
		l.WeightIH_f_Buffer, l.WeightHH_f_Buffer, l.BiasH_f_Buffer,
		l.WeightIH_g_Buffer, l.WeightHH_g_Buffer, l.BiasH_g_Buffer,
		l.WeightIH_o_Buffer, l.WeightHH_o_Buffer, l.BiasH_o_Buffer,
		l.InputGradientBuffer,
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
}
