package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// DispatchBackward adds backward pass commands to encoder
func (l *DenseLayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	// Log("Dispatching Backward Dense Layer")
	// Assumes dZ (dOutput) has been computed by generic activation backward shader and placed in dOutputBuffer (bound in CreateBackwardBindGroup).

	// Pipeline 1: Weight Gradients (dW, dB)
	if l.backwardPipelineGrads != nil && l.backwardBindGroupGrads != nil {
		// Workgroups: One for each output neuron (Row)
		// Each thread loops over BatchSize to accumulate gradients.
		gx := (uint32(l.Spec.OutputSize) + 255) / 256
		pass := enc.BeginComputePass(&wgpu.ComputePassDescriptor{
			Label: "Dense_Bwd_Grads",
		})
		pass.SetPipeline(l.backwardPipelineGrads)
		pass.SetBindGroup(0, l.backwardBindGroupGrads, nil)
		pass.DispatchWorkgroups(gx, 1, 1)
		pass.End()
	}

	// Pipeline 2: Input Gradients (dX)
	if l.backwardPipelineInput != nil && l.backwardBindGroupInput != nil {
		// Workgroups: One for each input neuron PER SAMPLE.
		// Total threads = InputSize * BatchSize.
		batch := l.BatchSize
		if batch <= 0 {
			batch = 1
		}

		totalThreads := uint32(l.Spec.InputSize * batch)
		gx := (totalThreads + 255) / 256

		pass := enc.BeginComputePass(&wgpu.ComputePassDescriptor{
			Label: "Dense_Bwd_Input",
		})
		pass.SetPipeline(l.backwardPipelineInput)
		pass.SetBindGroup(0, l.backwardBindGroupInput, nil)
		pass.DispatchWorkgroups(gx, 1, 1)
		pass.End()
	}
}

// CompileBackward creates pipelines for backward pass
// CompileBackward creates pipelines for backward pass with Explicit Layouts
func (l *DenseLayer) CompileBackward(ctx *Context, labelPrefix string) error {
	if Debug {
		Log("Compiling backward dense layer %s", labelPrefix)
	}
	// ------------------------------------------------------------------
	// 1. Gradients Pipeline (dW, dB)
	// ------------------------------------------------------------------
	shaderGrads := l.generateBackwardGradsShader()
	moduleGrads, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_Shader_Bwd_Grads",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shaderGrads},
	})
	if err != nil {
		return fmt.Errorf("create shader grads: %w", err)
	}
	// defer moduleGrads.Release()

	// Explicit Bind Group Layout (Avoids "auto" panic)
	l.backwardBindGroupLayoutGrads, err = ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: labelPrefix + "_BGL_Bwd_Grads",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // dZ
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Input
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // dW
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // dB
		},
	})
	if err != nil {
		return fmt.Errorf("create bgl grads: %w", err)
	}

	plGrads, err := ctx.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            labelPrefix + "_PL_Bwd_Grads",
		BindGroupLayouts: []*wgpu.BindGroupLayout{l.backwardBindGroupLayoutGrads},
	})
	if err != nil {
		return fmt.Errorf("create pl grads: %w", err)
	}

	l.backwardPipelineGrads, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  labelPrefix + "_Pipe_Bwd_Grads",
		Layout: plGrads, // Explicit
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     moduleGrads,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return fmt.Errorf("create pipeline grads: %w", err)
	}
	moduleGrads.Release()

	// ------------------------------------------------------------------
	// 2. Input Gradients Pipeline (dX)
	// ------------------------------------------------------------------
	shaderInput := l.generateBackwardInputShader()
	moduleInput, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_Shader_Bwd_Input",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shaderInput},
	})
	if err != nil {
		return fmt.Errorf("create shader input: %w", err)
	}

	l.backwardBindGroupLayoutInput, err = ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: labelPrefix + "_BGL_Bwd_Input",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // dZ
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Weights
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // dX
		},
	})
	if err != nil {
		return fmt.Errorf("create bgl input: %w", err)
	}

	plInput, err := ctx.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            labelPrefix + "_PL_Bwd_Input",
		BindGroupLayouts: []*wgpu.BindGroupLayout{l.backwardBindGroupLayoutInput},
	})
	if err != nil {
		return fmt.Errorf("create pl input: %w", err)
	}

	l.backwardPipelineInput, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  labelPrefix + "_Pipe_Bwd_Input",
		Layout: plInput, // Explicit
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     moduleInput,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return fmt.Errorf("create pipeline input: %w", err)
	}
	moduleInput.Release()

	return nil
}

// CreateBackwardBindGroup creates bind groups for backward pass
func (l *DenseLayer) CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error {
	if dOutputBuffer == nil {
		return fmt.Errorf("dOutputBuffer cannot be nil")
	}

	// Bind Group for Gradients (dW, dB)
	var err error
	l.backwardBindGroupGrads, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_BG_Bwd_Grads",
		Layout: l.backwardBindGroupLayoutGrads,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
			{Binding: 1, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
			{Binding: 2, Buffer: l.WeightGradientBuffer, Size: l.WeightGradientBuffer.GetSize()},
			{Binding: 3, Buffer: l.BiasGradientBuffer, Size: l.BiasGradientBuffer.GetSize()},
		},
	})
	if err != nil {
		return fmt.Errorf("create bg grads: %w", err)
	}

	// Bind Group for Input Gradients (dX)
	l.backwardBindGroupInput, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_BG_Bwd_Input",
		Layout: l.backwardBindGroupLayoutInput,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
			{Binding: 1, Buffer: l.WeightBuffer, Size: l.WeightBuffer.GetSize()},
			{Binding: 2, Buffer: l.InputGradientBuffer, Size: l.InputGradientBuffer.GetSize()},
		},
	})
	if err != nil {
		return fmt.Errorf("create bg input: %w", err)
	}

	return nil
}

func (l *DenseLayer) generateBackwardGradsShader() string {
	// Calculates dW and dB with Batch Reduction.
	// dZ size = OutputSize * Batch
	// Input size = InputSize * Batch
	// dW size = OutputSize * InputSize
	// dB size = OutputSize
	//
	// Thread ID x corresponds to Output Neuron Index (Row of W).
	// One thread per output neuron.
	// Loop over samples in batch to accumulate.

	batch := l.BatchSize
	if batch <= 0 {
		batch = 1
	}

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> dZ : array<f32>; // [Batch, Output]
		@group(0) @binding(1) var<storage, read> input : array<f32>; // [Batch, Input]
		@group(0) @binding(2) var<storage, read_write> dW : array<f32>; // [Output, Input]
		@group(0) @binding(3) var<storage, read_write> dB : array<f32>; // [Output]

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let row = gid.x; // Output index
			let n_out = %du;
			let n_in = %du;
			let batch_size = %du;

			if (row >= n_out) {
				return;
			}
			
			// Initialize accumulators
			var db_sum: f32 = 0.0;
			// We need to write a whole row of dW.
			// But wait, holding n_in accumulators in registers is too much if n_in is large.
			// Better: This thread computes dB[row] AND dW[row, col] for all col?
			// Inner loop over col? n_in * batch_size ops. Could be heavy.
			// But for small n_in (2) it's fine. For large n_in (1000) it's heavy.
			
			// Optimization: Compute dB in one pass. dW in another?
			// Or just handle dB here for now.
			
			for (var b: u32 = 0u; b < batch_size; b++) {
				let dz_val = dZ[b * n_out + row];
				db_sum += dz_val;
			}
			dB[row] = db_sum; // Or += if accumulating over multiple passes/mini-batches? Assumes overwrite.

			// Compute dW[row, col]
			// We iterate col here?
			for (var col: u32 = 0u; col < n_in; col++) {
				var dw_sum: f32 = 0.0;
				for (var b: u32 = 0u; b < batch_size; b++) {
					let dz_val = dZ[b * n_out + row];
					let in_val = input[b * n_in + col];
					dw_sum += dz_val * in_val;
				}
				// Write to dW[row, col]
				dW[row * n_in + col] = dw_sum;
			}
		}
	`, l.Spec.OutputSize, l.Spec.InputSize, batch)
}

func (l *DenseLayer) generateBackwardInputShader() string {
	// Calculates dX (Input Gradient).
	// dX[b, j] = Sum_i(dZ[b, i] * W[i, j])
	// Parallel over Batch and Input dimension.
	//
	// Thread ID x corresponds to global input index [Batch * Input].

	batch := l.BatchSize
	if batch <= 0 {
		batch = 1
	}

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> dZ : array<f32>; // [Batch, Output]
		@group(0) @binding(1) var<storage, read> weights : array<f32>; // [Output, Input]
		@group(0) @binding(2) var<storage, read_write> dX : array<f32>; // [Batch, Input]

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let n_out = %du;
			let n_in = %du;
			// let batch = %du;

			if (idx >= arrayLength(&dX)) {
				return;
			}
			
			let sample_idx = idx / n_in;
			let col = idx %% n_in; // Input index

			var sum: f32 = 0.0;
			
			// Dot product of dZ[sample] vector and W column col
			// W[row, col]
			// dZ[sample, row]
			
			for (var row: u32 = 0u; row < n_out; row++) {
				// dZ index
				let dz_idx = sample_idx * n_out + row;
				// Weights index
				let w_idx = row * n_in + col;
				
				sum += dZ[dz_idx] * weights[w_idx];
			}

			dX[idx] = sum;
		}
	`, l.Spec.OutputSize, l.Spec.InputSize, batch)
}
