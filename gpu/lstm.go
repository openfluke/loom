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

	BatchSize int // Number of samples per batch

	pipeline   *wgpu.ComputePipeline
	bindGroups []*wgpu.BindGroup // One bind group per time step

	InputBuffer   *wgpu.Buffer // [SeqLen * InputSize]
	OutputBuffer  *wgpu.Buffer // [SeqLen * HiddenSize]
	StagingBuffer *wgpu.Buffer
	HiddenBuffer  *wgpu.Buffer   // Hidden state [BatchSize * HiddenSize]
	CellBuffer    *wgpu.Buffer   // Cell state [BatchSize * SeqLen * HiddenSize]
	StepBuffers   []*wgpu.Buffer // Uniform buffers for each step

	// Unified Weight Buffer (concatenated [IH, HH, Bias])
	UnifiedWeightsBuffer         *wgpu.Buffer
	UnifiedWeightsGradientBuffer *wgpu.Buffer

	InputGradientBuffer *wgpu.Buffer

	// Gate Gradients Storage [SeqLen * 4 * HiddenSize * BatchSize]
	// Stores calculated delta (dL/dZ) for all 4 gates for all steps
	GateGradientsBuffer *wgpu.Buffer

	// Recurrent Gradient State (for BPTT)
	dHiddenBuffer *wgpu.Buffer // [BatchSize * HiddenSize]
	dCellBuffer   *wgpu.Buffer // [BatchSize * HiddenSize]

	bwGatePipeline   *wgpu.ComputePipeline
	bwGateBindGroups []*wgpu.BindGroup // One per step

	bwPrevPipeline   *wgpu.ComputePipeline
	bwPrevBindGroups []*wgpu.BindGroup // One per step

	bwGradPipeline  *wgpu.ComputePipeline
	bwGradBindGroup *wgpu.BindGroup

	// Gradient application bind groups (cached for training)
	GradCombinedWeightsIHBindGroup *wgpu.BindGroup
	GradCombinedWeightsHHBindGroup *wgpu.BindGroup
	GradCombinedBiasesBindGroup    *wgpu.BindGroup
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

	inputTotal := l.Spec.SeqLen * l.Spec.InputSize * l.BatchSize
	if inputTotal < 1 {
		inputTotal = 1
	}
	outputTotal := l.Spec.SeqLen * l.Spec.HiddenSize * l.BatchSize
	if outputTotal < 1 {
		outputTotal = 1
	}

	hiddenSize := l.Spec.HiddenSize * l.BatchSize
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
		Size:  uint64(outputTotal * 4), // Store cell state for all steps [SeqLen * HiddenSize]
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
	ihData := l.combineWeights(l.Spec.WeightIH_i, l.Spec.WeightIH_f, l.Spec.WeightIH_g, l.Spec.WeightIH_o, ihSize)
	hhData := l.combineWeights(l.Spec.WeightHH_i, l.Spec.WeightHH_f, l.Spec.WeightHH_g, l.Spec.WeightHH_o, hhSize)
	biasData := l.combineWeights(l.Spec.BiasH_i, l.Spec.BiasH_f, l.Spec.BiasH_g, l.Spec.BiasH_o, l.Spec.HiddenSize)

	// Concatenate all weights into UnifiedWeightsBuffer
	totalSize := len(ihData) + len(hhData) + len(biasData)
	unifiedData := make([]float32, totalSize)
	copy(unifiedData[0:], ihData)
	copy(unifiedData[len(ihData):], hhData)
	copy(unifiedData[len(ihData)+len(hhData):], biasData)

	l.UnifiedWeightsBuffer, err = NewFloatBuffer(unifiedData, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
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

// combineWeights combines 4 weight slices into one (i, f, g, o)
func (l *LSTMLayer) combineWeights(w1, w2, w3, w4 []float32, size int) []float32 {
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

func (l *LSTMLayer) AllocateBackwardBuffers(ctx *Context, labelPrefix string) error {
	var err error

	// Input gradients
	l.InputGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_InGrad",
		Size:  uint64(l.Spec.SeqLen * l.Spec.InputSize * l.BatchSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Unified Weight Gradient Buffer
	ihSize := l.Spec.HiddenSize * l.Spec.InputSize
	hhSize := l.Spec.HiddenSize * l.Spec.HiddenSize
	totalSize := 4*ihSize + 4*hhSize + 4*l.Spec.HiddenSize
	l.UnifiedWeightsGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_UnifiedGrad",
		Size:  uint64(totalSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Gate Gradients [SeqLen * 4 * HiddenSize * BatchSize]
	gateGradSize := l.Spec.SeqLen * 4 * l.Spec.HiddenSize * l.BatchSize
	l.GateGradientsBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_GateGrads",
		Size:  uint64(gateGradSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// dHidden State [BatchSize * HiddenSize]
	dHiddenSize := l.BatchSize * l.Spec.HiddenSize
	l.dHiddenBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_dHidden",
		Size:  uint64(dHiddenSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// dCell State [BatchSize * HiddenSize]
	l.dCellBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_dCell",
		Size:  uint64(dHiddenSize * 4),
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
		@group(0) @binding(1) var<storage, read_write> hidden_state : array<f32>;
		@group(0) @binding(2) var<storage, read_write> cell_state : array<f32>;
		@group(0) @binding(3) var<storage, read_write> output : array<f32>;
		
		@group(0) @binding(4) var<storage, read> weights : array<f32>;
		@group(0) @binding(5) var<uniform> step : u32;

		const INPUT_SIZE: u32 = %du;
		const HIDDEN_SIZE: u32 = %du;
		const BATCH_SIZE: u32 = %du;
		const SEQ_LEN: u32 = %du;

		const IH_STRIDE: u32 = INPUT_SIZE * HIDDEN_SIZE; // Size of one gate's IH weights
		const HH_STRIDE: u32 = HIDDEN_SIZE * HIDDEN_SIZE; // Size of one gate's HH weights
		
		// Offsets in Unified Buffer (in f32 elements)
		// Unified: [IH_Total, HH_Total, Bias_Total]
		// IH_Total = 4 * IH_STRIDE
		// HH_Total = 4 * HH_STRIDE
		const OFFSET_HH_START: u32 = 4u * IH_STRIDE;
		const OFFSET_BIAS_START: u32 = 4u * IH_STRIDE + 4u * HH_STRIDE;

		fn sigmoid(x: f32) -> f32 {
			return 1.0 / (1.0 + exp(-x));
		}

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let total = BATCH_SIZE * HIDDEN_SIZE;
			if (idx >= total) { return; }
			
			let batch = idx / HIDDEN_SIZE;
			let h_idx = idx %% HIDDEN_SIZE;

			let seq_base = batch * SEQ_LEN * HIDDEN_SIZE;
			let prev_offset = seq_base + (step - 1u) * HIDDEN_SIZE;

			// Read previous cell state
			var c_prev: f32 = 0.0;
			if (step > 0u) {
				c_prev = cell_state[prev_offset + h_idx];
			}

			let input_offset = batch * SEQ_LEN * INPUT_SIZE + step * INPUT_SIZE;
			
			// --- Helper to read weights ---
			// w_ih[gate] starts at gate * IH_STRIDE
			// w_hh[gate] starts at OFFSET_HH_START + gate * HH_STRIDE
			// bias[gate] starts at OFFSET_BIAS_START + gate * HIDDEN_SIZE

			// Compute input gate (i) - Gate 0
			var i_gate: f32 = weights[OFFSET_BIAS_START + h_idx];
			for (var j: u32 = 0u; j < INPUT_SIZE; j++) {
				i_gate += input[input_offset + j] * weights[h_idx * INPUT_SIZE + j];
			}
			if (step > 0u) {
				for (var j: u32 = 0u; j < HIDDEN_SIZE; j++) {
					let h_prev_j = output[prev_offset + j];
					i_gate += h_prev_j * weights[OFFSET_HH_START + h_idx * HIDDEN_SIZE + j];
				}
			}
			i_gate = sigmoid(i_gate);

			// Compute forget gate (f) - Gate 1
			var f_gate: f32 = weights[OFFSET_BIAS_START + HIDDEN_SIZE + h_idx];
			for (var j: u32 = 0u; j < INPUT_SIZE; j++) {
				f_gate += input[input_offset + j] * weights[IH_STRIDE + h_idx * INPUT_SIZE + j];
			}
			if (step > 0u) {
				for (var j: u32 = 0u; j < HIDDEN_SIZE; j++) {
					let h_prev_j = output[prev_offset + j];
					f_gate += h_prev_j * weights[OFFSET_HH_START + HH_STRIDE + h_idx * HIDDEN_SIZE + j];
				}
			}
			f_gate = sigmoid(f_gate);

			// Compute cell gate (g) - Gate 2
			var g_gate: f32 = weights[OFFSET_BIAS_START + 2u * HIDDEN_SIZE + h_idx];
			for (var j: u32 = 0u; j < INPUT_SIZE; j++) {
				g_gate += input[input_offset + j] * weights[2u * IH_STRIDE + h_idx * INPUT_SIZE + j];
			}
			if (step > 0u) {
				for (var j: u32 = 0u; j < HIDDEN_SIZE; j++) {
					let h_prev_j = output[prev_offset + j];
					g_gate += h_prev_j * weights[OFFSET_HH_START + 2u * HH_STRIDE + h_idx * HIDDEN_SIZE + j];
				}
			}
			g_gate = tanh(g_gate);

			// Compute output gate (o) - Gate 3
			var o_gate: f32 = weights[OFFSET_BIAS_START + 3u * HIDDEN_SIZE + h_idx];
			for (var j: u32 = 0u; j < INPUT_SIZE; j++) {
				o_gate += input[input_offset + j] * weights[3u * IH_STRIDE + h_idx * INPUT_SIZE + j];
			}
			if (step > 0u) {
				for (var j: u32 = 0u; j < HIDDEN_SIZE; j++) {
					let h_prev_j = output[prev_offset + j];
					o_gate += h_prev_j * weights[OFFSET_HH_START + 3u * HH_STRIDE + h_idx * HIDDEN_SIZE + j];
				}
			}
			o_gate = sigmoid(o_gate);

			// Update cell and hidden states
			let new_c = f_gate * c_prev + i_gate * g_gate;
			let new_h = o_gate * tanh(new_c);

			// Store to buffers
			cell_state[seq_base + step * HIDDEN_SIZE + h_idx] = new_c;
			hidden_state[batch * HIDDEN_SIZE + h_idx] = new_h;
			output[seq_base + step * HIDDEN_SIZE + h_idx] = new_h;
		}
	`, l.Spec.InputSize, l.Spec.HiddenSize, l.BatchSize, l.Spec.SeqLen)
}

func (l *LSTMLayer) GenerateBackwardGateShader() string {
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> d_output : array<f32>;
		@group(0) @binding(1) var<storage, read> output : array<f32>;
		@group(0) @binding(2) var<storage, read> cell_state : array<f32>;
		@group(0) @binding(3) var<storage, read> weights : array<f32>;
		@group(0) @binding(4) var<storage, read> d_hidden : array<f32>; 
		@group(0) @binding(5) var<storage, read_write> d_cell : array<f32>;   
		@group(0) @binding(6) var<storage, read_write> gate_grads : array<f32>;
		@group(0) @binding(7) var<storage, read> input : array<f32>;
		@group(0) @binding(8) var<uniform> step : u32;

		const SEQ_LEN: u32 = %du;
		const INPUT_SIZE: u32 = %du;
		const HIDDEN_SIZE: u32 = %du;
		const BATCH_SIZE: u32 = %du;

		const IH_STRIDE: u32 = INPUT_SIZE * HIDDEN_SIZE;
		const HH_STRIDE: u32 = HIDDEN_SIZE * HIDDEN_SIZE;
		const OFFSET_HH_START: u32 = 4u * IH_STRIDE;
		const OFFSET_BIAS_START: u32 = 4u * IH_STRIDE + 4u * HH_STRIDE;

		fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
		fn d_sigmoid(y: f32) -> f32 { return y * (1.0 - y); }
		fn d_tanh(y: f32) -> f32 { return 1.0 - y * y; }

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let total = BATCH_SIZE * HIDDEN_SIZE;
			if (idx >= total) { return; }

			let batch = idx / HIDDEN_SIZE;
			let h_idx = idx %% HIDDEN_SIZE;

			let seq_offset = batch * SEQ_LEN * HIDDEN_SIZE + step * HIDDEN_SIZE + h_idx;
			let batch_offset = batch * HIDDEN_SIZE + h_idx;

			// 1. Gather dH
			let d_h_curr = d_output[seq_offset] + d_hidden[batch_offset];

			// 2. Recompute Forward State
			var h_prev: f32 = 0.0;
			var c_prev: f32 = 0.0;
			if (step > 0u) {
				let prev_offset = batch * SEQ_LEN * HIDDEN_SIZE + (step - 1u) * HIDDEN_SIZE + h_idx;
				h_prev = output[prev_offset];
				c_prev = cell_state[prev_offset];
			}

			// Recompute Pre-activations
			// Bias
			var i_pre: f32 = weights[OFFSET_BIAS_START + h_idx];
			var f_pre: f32 = weights[OFFSET_BIAS_START + HIDDEN_SIZE + h_idx];
			var g_pre: f32 = weights[OFFSET_BIAS_START + 2u * HIDDEN_SIZE + h_idx];
			var o_pre: f32 = weights[OFFSET_BIAS_START + 3u * HIDDEN_SIZE + h_idx];
			
			// W_ih
			let input_base = batch * SEQ_LEN * INPUT_SIZE + step * INPUT_SIZE;
			for (var j: u32 = 0u; j < INPUT_SIZE; j++) {
				let val = input[input_base + j];
				i_pre += val * weights[h_idx * INPUT_SIZE + j];
				f_pre += val * weights[IH_STRIDE + h_idx * INPUT_SIZE + j];
				g_pre += val * weights[2u * IH_STRIDE + h_idx * INPUT_SIZE + j];
				o_pre += val * weights[3u * IH_STRIDE + h_idx * INPUT_SIZE + j];
			}
			
			// W_hh
			if (step > 0u) {
				let prev_base = batch * SEQ_LEN * HIDDEN_SIZE + (step - 1u) * HIDDEN_SIZE;
				for (var j: u32 = 0u; j < HIDDEN_SIZE; j++) {
					let val = output[prev_base + j]; 
					i_pre += val * weights[OFFSET_HH_START + h_idx * HIDDEN_SIZE + j];
					f_pre += val * weights[OFFSET_HH_START + HH_STRIDE + h_idx * HIDDEN_SIZE + j];
					g_pre += val * weights[OFFSET_HH_START + 2u * HH_STRIDE + h_idx * HIDDEN_SIZE + j];
					o_pre += val * weights[OFFSET_HH_START + 3u * HH_STRIDE + h_idx * HIDDEN_SIZE + j];
				}
			}

			let i_val = sigmoid(i_pre);
			let f_val = sigmoid(f_pre);
			let g_val = tanh(g_pre);
			let o_val = sigmoid(o_pre);
			
			let c_curr = f_val * c_prev + i_val * g_val;
			let tanh_c = tanh(c_curr);
			
			// 3. Compute Gradients
			let d_o = d_h_curr * tanh_c * d_sigmoid(o_val);
			let d_c_curr = d_h_curr * o_val * d_tanh(tanh_c) + d_cell[batch_offset];
			let d_i = d_c_curr * g_val * d_sigmoid(i_val);
			let d_g = d_c_curr * i_val * d_tanh(g_val);
			let d_f = d_c_curr * c_prev * d_sigmoid(f_val);
			
			// Store Gate Gradients
			let gate_base = batch * SEQ_LEN * 4u * HIDDEN_SIZE + step * 4u * HIDDEN_SIZE;
			gate_grads[gate_base + h_idx] = d_i;
			gate_grads[gate_base + HIDDEN_SIZE + h_idx] = d_f;
			gate_grads[gate_base + 2u*HIDDEN_SIZE + h_idx] = d_g;
			gate_grads[gate_base + 3u*HIDDEN_SIZE + h_idx] = d_o;
			
			// Update d_cell for t-1
			d_cell[batch_offset] = d_c_curr * f_val;
		}
	`, l.Spec.SeqLen, l.Spec.InputSize, l.Spec.HiddenSize, l.BatchSize)
}

func (l *LSTMLayer) GenerateBackwardPrevShader() string {
	return fmt.Sprintf(`
		@group(0) @binding(3) var<storage, read> weights : array<f32>;
		@group(0) @binding(4) var<storage, read_write> d_hidden : array<f32>; 
		@group(0) @binding(5) var<storage, read> gate_grads : array<f32>;
		@group(0) @binding(6) var<storage, read_write> d_input : array<f32>;
		@group(0) @binding(7) var<uniform> step : u32;

		const SEQ_LEN: u32 = %du;
		const INPUT_SIZE: u32 = %du;
		const HIDDEN_SIZE: u32 = %du;
		const BATCH_SIZE: u32 = %du;

		const IH_STRIDE: u32 = INPUT_SIZE * HIDDEN_SIZE;
		const HH_STRIDE: u32 = HIDDEN_SIZE * HIDDEN_SIZE;
		const OFFSET_HH_START: u32 = 4u * IH_STRIDE;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let per_batch = HIDDEN_SIZE + INPUT_SIZE;
			let total = BATCH_SIZE * per_batch;
			if (idx >= total) { return; }
			
			let batch = idx / per_batch;
			let rem = idx %% per_batch;
			
			let gate_base = batch * SEQ_LEN * 4u * HIDDEN_SIZE + step * 4u * HIDDEN_SIZE;
			
			if (rem < HIDDEN_SIZE) {
				// Compute d_hidden_prev[j] (j = rem)
				let j = rem;
				var sum: f32 = 0.0;
				// d_hidden_prev = sum(gate_grads * w_hh)
				// w_hh access: weights[OFFSET_HH_START + gate*HH_STRIDE + k*HIDDEN + j]
				
				for (var k: u32 = 0u; k < HIDDEN_SIZE; k++) {
					sum += gate_grads[gate_base + k] * weights[OFFSET_HH_START + k * HIDDEN_SIZE + j];
					sum += gate_grads[gate_base + HIDDEN_SIZE + k] * weights[OFFSET_HH_START + HH_STRIDE + k * HIDDEN_SIZE + j];
					sum += gate_grads[gate_base + 2u*HIDDEN_SIZE + k] * weights[OFFSET_HH_START + 2u * HH_STRIDE + k * HIDDEN_SIZE + j];
					sum += gate_grads[gate_base + 3u*HIDDEN_SIZE + k] * weights[OFFSET_HH_START + 3u * HH_STRIDE + k * HIDDEN_SIZE + j];
				}
				d_hidden[batch * HIDDEN_SIZE + j] = sum;
				
			} else {
				// Compute d_input[j] (j = rem - HIDDEN_SIZE)
				let j = rem - HIDDEN_SIZE;
				var sum: f32 = 0.0;
				// w_ih access: weights[gate*IH_STRIDE + k*INPUT + j]
				
				for (var k: u32 = 0u; k < HIDDEN_SIZE; k++) {
					sum += gate_grads[gate_base + k] * weights[k * INPUT_SIZE + j];
					sum += gate_grads[gate_base + HIDDEN_SIZE + k] * weights[IH_STRIDE + k * INPUT_SIZE + j];
					sum += gate_grads[gate_base + 2u*HIDDEN_SIZE + k] * weights[2u * IH_STRIDE + k * INPUT_SIZE + j];
					sum += gate_grads[gate_base + 3u*HIDDEN_SIZE + k] * weights[3u * IH_STRIDE + k * INPUT_SIZE + j];
				}
				
				let input_idx = batch * SEQ_LEN * INPUT_SIZE + step * INPUT_SIZE + j;
				d_input[input_idx] = sum;
			}
		}
	`, l.Spec.SeqLen, l.Spec.InputSize, l.Spec.HiddenSize, l.BatchSize)
}

func (l *LSTMLayer) GenerateBackwardGradsShader() string {
	// LSTM Weight Gradients Calculation
	// Uses 'gate_grads' [Batch, Seq, 4*Hidden] computed in Pass 1.
	// dW = Sum(d_gate * input^T)
	//
	// New Bindings:
	// 0: d_output (Unused now? We use gate_grads)
	// 1: output (h history, used for w_hh)
	// 2: input (used for w_ih)
	// 3: d_w_ih
	// 4: d_w_hh
	// 5: d_bias
	// 6: gate_grads

	// Note: We kept d_output at binding 0 in CreateBackwardBindGroup just to handle the slot,
	// but we don't need it if we have gate_grads.

	return fmt.Sprintf(`
		@group(0) @binding(1) var<storage, read> output : array<f32>; // h history
		@group(0) @binding(2) var<storage, read> input : array<f32>;
		@group(0) @binding(3) var<storage, read_write> d_weights : array<f32>;
		@group(0) @binding(4) var<storage, read> gate_grads : array<f32>;

		const SEQ_LEN: u32 = %du;
		const INPUT_SIZE: u32 = %du;
		const HIDDEN_SIZE: u32 = %du;
		const BATCH_SIZE: u32 = %du;
		
		const IH_SIZE: u32 = %du;   // HIDDEN * INPUT
		const HH_SIZE: u32 = %du;   // HIDDEN * HIDDEN
		const TOTAL_IH: u32 = %du;  // 4 * IH_SIZE
		const TOTAL_HH: u32 = %du;  // 4 * HH_SIZE
		const TOTAL_B: u32 = %du;   // 4 * HIDDEN

		const IH_STRIDE: u32 = INPUT_SIZE * HIDDEN_SIZE;
		const HH_STRIDE: u32 = HIDDEN_SIZE * HIDDEN_SIZE;
		// Offsets in d_weights
		// IH: 0
		// HH: 4 * IH_STRIDE
		// Bias: 4 * IH_STRIDE + 4 * HH_STRIDE
		const OFFSET_HH_START: u32 = 4u * IH_STRIDE;
		const OFFSET_BIAS_START: u32 = 4u * IH_STRIDE + 4u * HH_STRIDE;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			
			// --- Bias Gradients ---
			if (idx < TOTAL_B) {
				let gate = idx / HIDDEN_SIZE; 
				let h = idx %% HIDDEN_SIZE;
				let gate_offset_in_step = gate * HIDDEN_SIZE + h;
				
				var sum: f32 = 0.0;
				for (var b: u32 = 0u; b < BATCH_SIZE; b++) {
					for (var t: u32 = 0u; t < SEQ_LEN; t++) {
						let base = b * SEQ_LEN * 4u * HIDDEN_SIZE + t * 4u * HIDDEN_SIZE;
						sum += gate_grads[base + gate_offset_in_step];
					}
				}
				d_weights[OFFSET_BIAS_START + idx] = sum * 0.1;
			}
			
			// --- W_ih Gradients ---
			if (idx < TOTAL_IH) {
				let gate = idx / IH_SIZE; 
				let rem = idx %% IH_SIZE; 
				let h = rem / INPUT_SIZE;
				let i = rem %% INPUT_SIZE;
				
				let gate_offset_in_step = gate * HIDDEN_SIZE + h;
				
				var sum: f32 = 0.0;
				for (var b: u32 = 0u; b < BATCH_SIZE; b++) {
					for (var t: u32 = 0u; t < SEQ_LEN; t++) {
						let base = b * SEQ_LEN * 4u * HIDDEN_SIZE + t * 4u * HIDDEN_SIZE;
						let d_g = gate_grads[base + gate_offset_in_step];
						
						let input_val = input[b * SEQ_LEN * INPUT_SIZE + t * INPUT_SIZE + i];
						sum += d_g * input_val;
					}
				}
				d_weights[idx] = sum * 0.1;
			}
			
			// --- W_hh Gradients ---
			if (idx < TOTAL_HH) {
				let gate = idx / HH_SIZE;
				let rem = idx %% HH_SIZE;
				let h = rem / HIDDEN_SIZE;       // Target unit
				let h_prev_idx = rem %% HIDDEN_SIZE; // Source unit
				
				let gate_offset_in_step = gate * HIDDEN_SIZE + h;

				var sum: f32 = 0.0;
				for (var b: u32 = 0u; b < BATCH_SIZE; b++) {
					for (var t: u32 = 1u; t < SEQ_LEN; t++) {
						// h_prev is from t-1.
						let base = b * SEQ_LEN * 4u * HIDDEN_SIZE + t * 4u * HIDDEN_SIZE;
						let d_g = gate_grads[base + gate_offset_in_step];
						
						let h_prev_val = output[b * SEQ_LEN * HIDDEN_SIZE + (t - 1u) * HIDDEN_SIZE + h_prev_idx];
						sum += d_g * h_prev_val;
					}
				}
				d_weights[OFFSET_HH_START + idx] = sum * 0.1;
			}
		}
	`, l.Spec.SeqLen, l.Spec.InputSize, l.Spec.HiddenSize, l.BatchSize,
		l.Spec.HiddenSize*l.Spec.InputSize,
		l.Spec.HiddenSize*l.Spec.HiddenSize,
		4*l.Spec.HiddenSize*l.Spec.InputSize,
		4*l.Spec.HiddenSize*l.Spec.HiddenSize,
		4*l.Spec.HiddenSize)
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
	// 1. Gate Shader (Pipeline 1)
	modGate, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_BwdGateShader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateBackwardGateShader()},
	})
	if err != nil {
		return err
	}
	l.bwGatePipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_BwdGatePipe",
		Compute: wgpu.ProgrammableStageDescriptor{Module: modGate, EntryPoint: "main"},
	})
	if err != nil {
		return err
	}

	// 2. Prev Shader (Pipeline 2)
	modPrev, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_BwdPrevShader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateBackwardPrevShader()},
	})
	if err != nil {
		return err
	}
	l.bwPrevPipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_BwdPrevPipe",
		Compute: wgpu.ProgrammableStageDescriptor{Module: modPrev, EntryPoint: "main"},
	})
	if err != nil {
		return err
	}

	// 3. dWeights/dBias Shader (Pipeline 3)
	modGrad, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_BwdGradShader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: l.GenerateBackwardGradsShader()},
	})
	if err != nil {
		return err
	}
	l.bwGradPipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_BwdGradPipe",
		Compute: wgpu.ProgrammableStageDescriptor{Module: modGrad, EntryPoint: "main"},
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
				{Binding: 4, Buffer: l.UnifiedWeightsBuffer, Size: l.UnifiedWeightsBuffer.GetSize()},
				{Binding: 5, Buffer: l.StepBuffers[step], Size: 4},
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

	l.bwGateBindGroups = make([]*wgpu.BindGroup, l.Spec.SeqLen)
	l.bwPrevBindGroups = make([]*wgpu.BindGroup, l.Spec.SeqLen)

	for step := 0; step < l.Spec.SeqLen; step++ {
		// Gate Bind Group
		l.bwGateBindGroups[step], err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  fmt.Sprintf("%s_BwdGateBind_%d", labelPrefix, step),
			Layout: l.bwGatePipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
				{Binding: 1, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
				{Binding: 2, Buffer: l.CellBuffer, Size: l.CellBuffer.GetSize()},
				{Binding: 3, Buffer: l.UnifiedWeightsBuffer, Size: l.UnifiedWeightsBuffer.GetSize()},
				{Binding: 4, Buffer: l.dHiddenBuffer, Size: l.dHiddenBuffer.GetSize()},
				{Binding: 5, Buffer: l.dCellBuffer, Size: l.dCellBuffer.GetSize()},
				{Binding: 6, Buffer: l.GateGradientsBuffer, Size: l.GateGradientsBuffer.GetSize()},
				{Binding: 7, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
				{Binding: 8, Buffer: l.StepBuffers[step], Size: 4}, // Specific step buffer
			},
		})
		if err != nil {
			return err
		}

		// Prev Bind Group
		l.bwPrevBindGroups[step], err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  fmt.Sprintf("%s_BwdPrevBind_%d", labelPrefix, step),
			Layout: l.bwPrevPipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 3, Buffer: l.UnifiedWeightsBuffer, Size: l.UnifiedWeightsBuffer.GetSize()},
				{Binding: 4, Buffer: l.dHiddenBuffer, Size: l.dHiddenBuffer.GetSize()},
				{Binding: 5, Buffer: l.GateGradientsBuffer, Size: l.GateGradientsBuffer.GetSize()},
				{Binding: 6, Buffer: l.InputGradientBuffer, Size: l.InputGradientBuffer.GetSize()},
				{Binding: 7, Buffer: l.StepBuffers[step], Size: 4}, // Same step buffer
			},
		})
		if err != nil {
			return err
		}
	}

	// dWeights/dBias bind group (Pass 3)
	l.bwGradBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_BwdGradBind",
		Layout: l.bwGradPipeline.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 1, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
			{Binding: 2, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
			{Binding: 3, Buffer: l.UnifiedWeightsGradientBuffer, Size: l.UnifiedWeightsGradientBuffer.GetSize()},
			{Binding: 4, Buffer: l.GateGradientsBuffer, Size: l.GateGradientsBuffer.GetSize()},
		},
	})
	return err
}

func (l *LSTMLayer) ZeroGradients(ctx *Context) {
	// Zero unified weight gradients
	ihSize := 4 * l.Spec.HiddenSize * l.Spec.InputSize
	hhSize := 4 * l.Spec.HiddenSize * l.Spec.HiddenSize
	bSize := 4 * l.Spec.HiddenSize
	totalSize := ihSize + hhSize + bSize

	ctx.Queue.WriteBuffer(l.UnifiedWeightsGradientBuffer, 0, wgpu.ToBytes(make([]float32, totalSize)))
}

func (l *LSTMLayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	// Dispatch once per time step - each step processes in parallel within hidden units
	// but steps must be sequential due to hidden state dependency
	total := uint32(l.BatchSize * l.Spec.HiddenSize)
	wg := (total + 255) / 256
	for step := 0; step < l.Spec.SeqLen; step++ {
		pass.SetPipeline(l.pipeline)
		pass.SetBindGroup(0, l.bindGroups[step], nil)
		pass.DispatchWorkgroups(wg, 1, 1)
	}
}

func (l *LSTMLayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	// Sequential BPTT: Iterate t from SeqLen-1 down to 0

	total := uint32(l.BatchSize * l.Spec.HiddenSize)
	wg := (total + 255) / 256

	// Pass 1 & 2 combined logic setup
	for step := l.Spec.SeqLen - 1; step >= 0; step-- {
		// 1. Gate Logic
		passGate := enc.BeginComputePass(nil)
		passGate.SetPipeline(l.bwGatePipeline)
		if step < len(l.bwGateBindGroups) {
			passGate.SetBindGroup(0, l.bwGateBindGroups[step], nil)
			passGate.DispatchWorkgroups(wg, 1, 1)
		}
		passGate.End()

		// 2. Prev Hidden / Input Logic
		// Needs to work over BATCH * (HIDDEN + INPUT)
		perBatch := l.Spec.HiddenSize + l.Spec.InputSize
		totalPrev := uint32(l.BatchSize * perBatch)
		wgPrev := (totalPrev + 255) / 256

		passPrev := enc.BeginComputePass(nil)
		passPrev.SetPipeline(l.bwPrevPipeline)
		if step < len(l.bwPrevBindGroups) {
			passPrev.SetBindGroup(0, l.bwPrevBindGroups[step], nil)
			passPrev.DispatchWorkgroups(wgPrev, 1, 1)
		}
		passPrev.End()
	}

	// 3. dWeights/dBias
	passGrad := enc.BeginComputePass(nil)
	passGrad.SetPipeline(l.bwGradPipeline)
	passGrad.SetBindGroup(0, l.bwGradBindGroup, nil)

	ihTotal := 4 * l.Spec.HiddenSize * l.Spec.InputSize
	hhTotal := 4 * l.Spec.HiddenSize * l.Spec.HiddenSize
	maxTotal := ihTotal
	if hhTotal > maxTotal {
		maxTotal = hhTotal
	}
	passGrad.DispatchWorkgroups(uint32((maxTotal+255)/256), 1, 1)
	passGrad.End()
}

func (l *LSTMLayer) UploadWeights(ctx *Context) {
	ihSize := l.Spec.HiddenSize * l.Spec.InputSize
	if ihSize < 1 {
		ihSize = 1
	}
	hhSize := l.Spec.HiddenSize * l.Spec.HiddenSize
	if hhSize < 1 {
		hhSize = 1
	}

	ihData := l.combineWeights(l.Spec.WeightIH_i, l.Spec.WeightIH_f, l.Spec.WeightIH_g, l.Spec.WeightIH_o, ihSize)
	hhData := l.combineWeights(l.Spec.WeightHH_i, l.Spec.WeightHH_f, l.Spec.WeightHH_g, l.Spec.WeightHH_o, hhSize)
	biasData := l.combineWeights(l.Spec.BiasH_i, l.Spec.BiasH_f, l.Spec.BiasH_g, l.Spec.BiasH_o, l.Spec.HiddenSize)

	// Offsets in bytes
	offsetIH := 0
	offsetHH := len(ihData) * 4
	offsetBias := (len(ihData) + len(hhData)) * 4

	ctx.Queue.WriteBuffer(l.UnifiedWeightsBuffer, uint64(offsetIH), wgpu.ToBytes(ihData))
	ctx.Queue.WriteBuffer(l.UnifiedWeightsBuffer, uint64(offsetHH), wgpu.ToBytes(hhData))
	ctx.Queue.WriteBuffer(l.UnifiedWeightsBuffer, uint64(offsetBias), wgpu.ToBytes(biasData))
}

func (l *LSTMLayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	return nil, nil, nil
}

func (l *LSTMLayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	ihSize := 4 * l.Spec.HiddenSize * l.Spec.InputSize
	hhSize := 4 * l.Spec.HiddenSize * l.Spec.HiddenSize
	bSize := 4 * l.Spec.HiddenSize
	totalSize := ihSize + hhSize + bSize

	unifiedGrad, err := ReadBuffer(l.UnifiedWeightsGradientBuffer, totalSize)
	if err != nil {
		return nil, nil, nil, err
	}

	wihGrad := unifiedGrad[0:ihSize]
	whhGrad := unifiedGrad[ihSize : ihSize+hhSize]
	bGrad := unifiedGrad[ihSize+hhSize:]

	// Input Gradients
	iGrad, err := ReadBuffer(l.InputGradientBuffer, l.Spec.SeqLen*l.Spec.InputSize*l.BatchSize)

	totalKernelSize := len(wihGrad) + len(whhGrad)
	kernelGrad := make([]float32, totalKernelSize)
	copy(kernelGrad[0:], wihGrad)
	copy(kernelGrad[len(wihGrad):], whhGrad)

	return kernelGrad, bGrad, iGrad, err
}

func (l *LSTMLayer) Cleanup() {
	bufs := []*wgpu.Buffer{
		l.InputBuffer, l.OutputBuffer, l.StagingBuffer,
		l.HiddenBuffer, l.CellBuffer,
		l.UnifiedWeightsBuffer, l.UnifiedWeightsGradientBuffer,
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
	if l.bwGatePipeline != nil {
		l.bwGatePipeline.Release()
	}
	for _, bg := range l.bwGateBindGroups {
		if bg != nil {
			bg.Release()
		}
	}
	if l.bwPrevPipeline != nil {
		l.bwPrevPipeline.Release()
	}
	for _, bg := range l.bwPrevBindGroups {
		if bg != nil {
			bg.Release()
		}
	}
	// Also clean arrays of bind groups if we add them
}
