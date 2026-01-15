package nn

import (
	"fmt"

	"github.com/openfluke/loom/gpu"
	"github.com/openfluke/webgpu/wgpu"
)

// applyGradientsGPU applies gradients directly on GPU without CPU transfers
// This is the GPU equivalent of the CPU loop: weight[i] -= lr * gradient[i]
func (n *Network) applyGradientsGPU(learningRate float32) {
	layers, ok := n.gpuLayers.([]gpu.GPULayer)
	if !ok || len(layers) == 0 {
		return
	}

	ctx, ok := n.gpuCtx.(*gpu.Context)
	if !ok || ctx == nil {
		return
	}

	// Ensure pipeline exists (lazy init fallback)
	if err := n.EnsureGradientPipeline(ctx); err != nil {
		fmt.Printf("Error creating gradient pipeline: %v\n", err)
		return
	}

	// Update learning rate in params buffer
	paramsData := []float32{learningRate, 0, 0, 0}
	ctx.Queue.WriteBuffer(n.gpuGradParams, 0, wgpu.ToBytes(paramsData))

	// Create ONE command encoder for ALL gradient applications
	enc, err := ctx.Device.CreateCommandEncoder(nil)
	if err != nil {
		return
	}

	// Apply gradients to each layer's weights and biases (batched into one encoder)
	for _, layer := range layers {
		switch l := layer.(type) {
		case *gpu.DenseLayer:
			applyGradientsToDenseLayerBatched(ctx, n.gpuGradPipeline, n.gpuGradParams, l, enc)
		case *gpu.Conv1DLayer:
			applyGradientsToConv1DLayerBatched(ctx, n.gpuGradPipeline, n.gpuGradParams, l, enc)
		case *gpu.Conv2DLayer:
			applyGradientsToConv2DLayerBatched(ctx, n.gpuGradPipeline, n.gpuGradParams, l, enc)
		case *gpu.RNNLayer:
			applyGradientsToRNNLayerBatched(ctx, n.gpuGradPipeline, n.gpuGradParams, l, enc)
		case *gpu.LSTMLayer:
			applyGradientsToLSTMLayerBatched(ctx, n.gpuGradPipeline, n.gpuGradParams, l, enc)
		}
	}

	// Submit ALL gradient applications in one batch
	cmd, err := enc.Finish(nil)
	if err != nil {
		return
	}
	ctx.Queue.Submit(cmd)

	// No poll - async execution
}
func applyGradientsToDenseLayer(ctx *gpu.Context, pipeline *wgpu.ComputePipeline, paramsBuffer *wgpu.Buffer, layer *gpu.DenseLayer) {
	// This function is no longer used - see applyGradientsGPU for batched implementation
}

// createGradientBindGroups creates and caches gradient application bind groups for a DenseLayer
// Called once during WeightsToGPU initialization
func createGradientBindGroups(ctx *gpu.Context, pipeline *wgpu.ComputePipeline, paramsBuffer *wgpu.Buffer, layer *gpu.DenseLayer) {
	// Create weight bind group
	if layer.WeightBuffer != nil && layer.WeightGradientBuffer != nil {
		var err error
		layer.GradientWeightBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "GradientApplyWeights",
			Layout: pipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: layer.WeightBuffer, Size: layer.WeightBuffer.GetSize()},
				{Binding: 1, Buffer: layer.WeightGradientBuffer, Size: layer.WeightGradientBuffer.GetSize()},
				{Binding: 2, Buffer: paramsBuffer, Size: paramsBuffer.GetSize()},
			},
		})
		if err != nil {
			return
		}
	}

	// Create bias bind group
	if layer.BiasBuffer != nil && layer.BiasGradientBuffer != nil {
		var err error
		layer.GradientBiasBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "GradientApplyBiases",
			Layout: pipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: layer.BiasBuffer, Size: layer.BiasBuffer.GetSize()},
				{Binding: 1, Buffer: layer.BiasGradientBuffer, Size: layer.BiasGradientBuffer.GetSize()},
				{Binding: 2, Buffer: paramsBuffer, Size: paramsBuffer.GetSize()},
			},
		})
		if err != nil {
			return
		}
	}
}

func applyGradientsToDenseLayerBatched(ctx *gpu.Context, pipeline *wgpu.ComputePipeline, paramsBuffer *wgpu.Buffer, layer *gpu.DenseLayer, enc *wgpu.CommandEncoder) {
	// Use cached bind groups from the layer (created once in WeightsToGPU)

	// Apply weight gradients
	if layer.GradientWeightBindGroup != nil {
		workgroups := uint32((layer.Spec.InputSize*layer.Spec.OutputSize + 255) / 256)
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(pipeline)
		pass.SetBindGroup(0, layer.GradientWeightBindGroup, nil)
		pass.DispatchWorkgroups(workgroups, 1, 1)
		pass.End()
	}

	// Apply bias gradients
	if layer.GradientBiasBindGroup != nil {
		workgroups := uint32((layer.Spec.OutputSize + 255) / 256)
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(pipeline)
		pass.SetBindGroup(0, layer.GradientBiasBindGroup, nil)
		pass.DispatchWorkgroups(workgroups, 1, 1)
		pass.End()
	}
}

// applyGradientsToConv1DLayerBatched applies gradients for Conv1D layer
func applyGradientsToConv1DLayerBatched(ctx *gpu.Context, pipeline *wgpu.ComputePipeline, paramsBuffer *wgpu.Buffer, layer *gpu.Conv1DLayer, enc *wgpu.CommandEncoder) {
	// Apply weight gradients
	if layer.WeightBuffer != nil && layer.WeightGradientBuffer != nil {
		weightSize := layer.Spec.OutChannels * layer.Spec.InChannels * layer.Spec.KernelSize
		workgroups := uint32((weightSize + 255) / 256)

		bg, err := ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "Conv1D_GradWeights",
			Layout: pipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: layer.WeightBuffer, Size: layer.WeightBuffer.GetSize()},
				{Binding: 1, Buffer: layer.WeightGradientBuffer, Size: layer.WeightGradientBuffer.GetSize()},
				{Binding: 2, Buffer: paramsBuffer, Size: paramsBuffer.GetSize()},
			},
		})
		if err == nil {
			pass := enc.BeginComputePass(nil)
			pass.SetPipeline(pipeline)
			pass.SetBindGroup(0, bg, nil)
			pass.DispatchWorkgroups(workgroups, 1, 1)
			pass.End()
		}
	}

	// Apply bias gradients
	if layer.BiasBuffer != nil && layer.BiasGradientBuffer != nil {
		workgroups := uint32((layer.Spec.OutChannels + 255) / 256)

		bg, err := ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "Conv1D_GradBias",
			Layout: pipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: layer.BiasBuffer, Size: layer.BiasBuffer.GetSize()},
				{Binding: 1, Buffer: layer.BiasGradientBuffer, Size: layer.BiasGradientBuffer.GetSize()},
				{Binding: 2, Buffer: paramsBuffer, Size: paramsBuffer.GetSize()},
			},
		})
		if err == nil {
			pass := enc.BeginComputePass(nil)
			pass.SetPipeline(pipeline)
			pass.SetBindGroup(0, bg, nil)
			pass.DispatchWorkgroups(workgroups, 1, 1)
			pass.End()
		}
	}
}

// applyGradientsToConv2DLayerBatched applies gradients for Conv2D layer
func applyGradientsToConv2DLayerBatched(ctx *gpu.Context, pipeline *wgpu.ComputePipeline, paramsBuffer *wgpu.Buffer, layer *gpu.Conv2DLayer, enc *wgpu.CommandEncoder) {
	// Apply weight gradients
	if layer.WeightBuffer != nil && layer.WeightGradientBuffer != nil {
		weightSize := layer.Spec.OutChannels * layer.Spec.InChannels * layer.Spec.KernelSize * layer.Spec.KernelSize
		workgroups := uint32((weightSize + 255) / 256)

		bg, err := ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "Conv2D_GradWeights",
			Layout: pipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: layer.WeightBuffer, Size: layer.WeightBuffer.GetSize()},
				{Binding: 1, Buffer: layer.WeightGradientBuffer, Size: layer.WeightGradientBuffer.GetSize()},
				{Binding: 2, Buffer: paramsBuffer, Size: paramsBuffer.GetSize()},
			},
		})
		if err == nil {
			pass := enc.BeginComputePass(nil)
			pass.SetPipeline(pipeline)
			pass.SetBindGroup(0, bg, nil)
			pass.DispatchWorkgroups(workgroups, 1, 1)
			pass.End()
		}
	}

	// Apply bias gradients
	if layer.BiasBuffer != nil && layer.BiasGradientBuffer != nil {
		workgroups := uint32((layer.Spec.OutChannels + 255) / 256)

		bg, err := ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "Conv2D_GradBias",
			Layout: pipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: layer.BiasBuffer, Size: layer.BiasBuffer.GetSize()},
				{Binding: 1, Buffer: layer.BiasGradientBuffer, Size: layer.BiasGradientBuffer.GetSize()},
				{Binding: 2, Buffer: paramsBuffer, Size: paramsBuffer.GetSize()},
			},
		})
		if err == nil {
			pass := enc.BeginComputePass(nil)
			pass.SetPipeline(pipeline)
			pass.SetBindGroup(0, bg, nil)
			pass.DispatchWorkgroups(workgroups, 1, 1)
			pass.End()
		}
	}
}

// applyGradientsToRNNLayerBatched applies gradients for RNN layer
func applyGradientsToRNNLayerBatched(ctx *gpu.Context, pipeline *wgpu.ComputePipeline, paramsBuffer *wgpu.Buffer, layer *gpu.RNNLayer, enc *wgpu.CommandEncoder) {
	// Apply WeightIH gradients
	if layer.WeightIHBuffer != nil && layer.WeightIHGradientBuffer != nil {
		size := layer.Spec.HiddenSize * layer.Spec.InputSize
		workgroups := uint32((size + 255) / 256)

		bg, err := ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "RNN_GradwIH",
			Layout: pipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: layer.WeightIHBuffer, Size: layer.WeightIHBuffer.GetSize()},
				{Binding: 1, Buffer: layer.WeightIHGradientBuffer, Size: layer.WeightIHGradientBuffer.GetSize()},
				{Binding: 2, Buffer: paramsBuffer, Size: paramsBuffer.GetSize()},
			},
		})
		if err == nil {
			pass := enc.BeginComputePass(nil)
			pass.SetPipeline(pipeline)
			pass.SetBindGroup(0, bg, nil)
			pass.DispatchWorkgroups(workgroups, 1, 1)
			pass.End()
		}
	}

	// Apply WeightHH gradients
	if layer.WeightHHBuffer != nil && layer.WeightHHGradientBuffer != nil {
		size := layer.Spec.HiddenSize * layer.Spec.HiddenSize
		workgroups := uint32((size + 255) / 256)

		bg, err := ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "RNN_GradwHH",
			Layout: pipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: layer.WeightHHBuffer, Size: layer.WeightHHBuffer.GetSize()},
				{Binding: 1, Buffer: layer.WeightHHGradientBuffer, Size: layer.WeightHHGradientBuffer.GetSize()},
				{Binding: 2, Buffer: paramsBuffer, Size: paramsBuffer.GetSize()},
			},
		})
		if err == nil {
			pass := enc.BeginComputePass(nil)
			pass.SetPipeline(pipeline)
			pass.SetBindGroup(0, bg, nil)
			pass.DispatchWorkgroups(workgroups, 1, 1)
			pass.End()
		}
	}

	// Apply Bias gradients
	if layer.BiasBuffer != nil && layer.BiasGradientBuffer != nil {
		size := layer.Spec.HiddenSize
		workgroups := uint32((size + 255) / 256)

		bg, err := ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "RNN_GradBias",
			Layout: pipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: layer.BiasBuffer, Size: layer.BiasBuffer.GetSize()},
				{Binding: 1, Buffer: layer.BiasGradientBuffer, Size: layer.BiasGradientBuffer.GetSize()},
				{Binding: 2, Buffer: paramsBuffer, Size: paramsBuffer.GetSize()},
			},
		})
		if err == nil {
			pass := enc.BeginComputePass(nil)
			pass.SetPipeline(pipeline)
			pass.SetBindGroup(0, bg, nil)
			pass.DispatchWorkgroups(workgroups, 1, 1)
			pass.End()
		}
	}
}

// applyGradientsToLSTMLayerBatched applies gradients for LSTM layer
func applyGradientsToLSTMLayerBatched(ctx *gpu.Context, pipeline *wgpu.ComputePipeline, paramsBuffer *wgpu.Buffer, layer *gpu.LSTMLayer, enc *wgpu.CommandEncoder) {
	// Apply Combined Weight IH gradients
	if layer.CombinedWeightsIH != nil && layer.CombinedWeightsIHGradientBuffer != nil {
		ihSize := layer.Spec.HiddenSize * layer.Spec.InputSize
		totalSize := 4 * ihSize
		workgroups := uint32((totalSize + 255) / 256)

		bg, err := ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "LSTM_GradwIH",
			Layout: pipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: layer.CombinedWeightsIH, Size: layer.CombinedWeightsIH.GetSize()},
				{Binding: 1, Buffer: layer.CombinedWeightsIHGradientBuffer, Size: layer.CombinedWeightsIHGradientBuffer.GetSize()},
				{Binding: 2, Buffer: paramsBuffer, Size: paramsBuffer.GetSize()},
			},
		})
		if err == nil {
			pass := enc.BeginComputePass(nil)
			pass.SetPipeline(pipeline)
			pass.SetBindGroup(0, bg, nil)
			pass.DispatchWorkgroups(workgroups, 1, 1)
			pass.End()
		}
	}

	// Apply Combined Weight HH gradients
	if layer.CombinedWeightsHH != nil && layer.CombinedWeightsHHGradientBuffer != nil {
		hhSize := layer.Spec.HiddenSize * layer.Spec.HiddenSize
		totalSize := 4 * hhSize
		workgroups := uint32((totalSize + 255) / 256)

		bg, err := ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "LSTM_GradwHH",
			Layout: pipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: layer.CombinedWeightsHH, Size: layer.CombinedWeightsHH.GetSize()},
				{Binding: 1, Buffer: layer.CombinedWeightsHHGradientBuffer, Size: layer.CombinedWeightsHHGradientBuffer.GetSize()},
				{Binding: 2, Buffer: paramsBuffer, Size: paramsBuffer.GetSize()},
			},
		})
		if err == nil {
			pass := enc.BeginComputePass(nil)
			pass.SetPipeline(pipeline)
			pass.SetBindGroup(0, bg, nil)
			pass.DispatchWorkgroups(workgroups, 1, 1)
			pass.End()
		}
	}

	// Apply Combined Bias gradients
	if layer.CombinedBiases != nil && layer.CombinedBiasesGradientBuffer != nil {
		totalSize := 4 * layer.Spec.HiddenSize
		workgroups := uint32((totalSize + 255) / 256)

		bg, err := ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  "LSTM_GradBias",
			Layout: pipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: layer.CombinedBiases, Size: layer.CombinedBiases.GetSize()},
				{Binding: 1, Buffer: layer.CombinedBiasesGradientBuffer, Size: layer.CombinedBiasesGradientBuffer.GetSize()},
				{Binding: 2, Buffer: paramsBuffer, Size: paramsBuffer.GetSize()},
			},
		})
		if err == nil {
			pass := enc.BeginComputePass(nil)
			pass.SetPipeline(pipeline)
			pass.SetBindGroup(0, bg, nil)
			pass.DispatchWorkgroups(workgroups, 1, 1)
			pass.End()
		}
	}
}

func (n *Network) EnsureGradientPipeline(ctx *gpu.Context) error {
	if n.gpuGradPipeline != nil {
		return nil
	}

	shaderCode := `
		@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
		@group(0) @binding(1) var<storage, read> gradients: array<f32>;
		@group(0) @binding(2) var<uniform> params: vec4<f32>;

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let i = gid.x;
			if (i >= arrayLength(&weights)) { return; }
			weights[i] -= params.x * gradients[i];
		}
	`

	module, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "GradientApplication",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shaderCode},
	})
	if err != nil {
		return fmt.Errorf("create shader module: %w", err)
	}
	defer module.Release()

	n.gpuGradPipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "GradientApplicationPipeline",
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return fmt.Errorf("create compute pipeline: %w", err)
	}

	// Create params buffer once
	paramsData := []float32{0, 0, 0, 0} // Will update learningRate each time
	n.gpuGradParams, err = ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "GradientApplicationParams",
		Contents: wgpu.ToBytes(paramsData),
		Usage:    wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		n.gpuGradPipeline.Release()
		n.gpuGradPipeline = nil
		return fmt.Errorf("create params buffer: %w", err)
	}

	return nil
}
