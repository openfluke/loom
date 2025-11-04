package nn

import (
	"fmt"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// generateConv2DShaderSimple creates WGSL shader for 2D convolution forward pass
func generateConv2DShaderSimple(
	batchSize, inChannels, outChannels, inH, inW, outH, outW, kSize, stride, padding int,
) string {
	return fmt.Sprintf(`
@group(0) @binding(0) var<storage, read> input: array<f32>;      // [batch][inC][inH][inW]
@group(0) @binding(1) var<storage, read> kernel: array<f32>;     // [outC][inC][kH][kW]
@group(0) @binding(2) var<storage, read> bias: array<f32>;       // [outC]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [batch][outC][outH][outW]

const BATCH: u32 = %du;
const IN_C: u32 = %du;
const OUT_C: u32 = %du;
const IN_H: u32 = %du;
const IN_W: u32 = %du;
const OUT_H: u32 = %du;
const OUT_W: u32 = %du;
const K_SIZE: u32 = %du;
const STRIDE: u32 = %du;
const PADDING: i32 = %d;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = BATCH * OUT_C * OUT_H * OUT_W;
    if (idx >= total) { return; }
    
    // Decode indices: [b][oc][oh][ow]
    let b = idx / (OUT_C * OUT_H * OUT_W);
    let remainder1 = idx %% (OUT_C * OUT_H * OUT_W);
    let oc = remainder1 / (OUT_H * OUT_W);
    let remainder2 = remainder1 %% (OUT_H * OUT_W);
    let oh = remainder2 / OUT_W;
    let ow = remainder2 %% OUT_W;
    
    var sum = bias[oc];
    
    // Convolve
    for (var ic: u32 = 0u; ic < IN_C; ic = ic + 1u) {
        for (var kh: u32 = 0u; kh < K_SIZE; kh = kh + 1u) {
            for (var kw: u32 = 0u; kw < K_SIZE; kw = kw + 1u) {
                let ih = i32(oh * STRIDE) + i32(kh) - PADDING;
                let iw = i32(ow * STRIDE) + i32(kw) - PADDING;
                
                if (ih >= 0 && ih < i32(IN_H) && iw >= 0 && iw < i32(IN_W)) {
                    let input_idx = b * IN_C * IN_H * IN_W + 
                                   ic * IN_H * IN_W + 
                                   u32(ih) * IN_W + 
                                   u32(iw);
                    let kernel_idx = oc * IN_C * K_SIZE * K_SIZE + 
                                    ic * K_SIZE * K_SIZE + 
                                    kh * K_SIZE + 
                                    kw;
                    sum = sum + input[input_idx] * kernel[kernel_idx];
                }
            }
        }
    }
    
    output[idx] = sum;
}
`, batchSize, inChannels, outChannels, inH, inW, outH, outW, kSize, stride, padding)
}

// conv2DForwardGPU performs 2D convolution using GPU compute shaders
func conv2DForwardGPU(
	dev *wgpu.Device,
	queue *wgpu.Queue,
	input []float32,
	config *LayerConfig,
	batchSize int,
) ([]float32, error) {
	inH := config.InputHeight
	inW := config.InputWidth
	inC := config.InputChannels
	kSize := config.KernelSize
	stride := config.Stride
	padding := config.Padding
	outC := config.Filters
	outH := config.OutputHeight
	outW := config.OutputWidth

	// Create shader
	shader := generateConv2DShaderSimple(batchSize, inC, outC, inH, inW, outH, outW, kSize, stride, padding)
	module, err := dev.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "conv2d_fwd_shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shader},
	})
	if err != nil {
		return nil, fmt.Errorf("CreateShaderModule: %w", err)
	}
	defer module.Release()

	// Create bind group layout
	bgl, err := dev.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "conv2d_fwd_bgl",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
		},
	})
	if err != nil {
		return nil, err
	}
	defer bgl.Release()

	// Create pipeline
	pl, err := dev.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "conv2d_fwd_pl",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgl},
	})
	if err != nil {
		return nil, err
	}
	defer pl.Release()

	pipeline, err := dev.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  "conv2d_fwd_pipeline",
		Layout: pl,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return nil, err
	}
	defer pipeline.Release()

	// Create buffers
	inputSize := batchSize * inC * inH * inW
	kernelSize := outC * inC * kSize * kSize
	biasSize := outC
	outputSize := batchSize * outC * outH * outW

	// Validate sizes
	if len(input) != inputSize {
		return nil, fmt.Errorf("input size mismatch: got %d, expected %d", len(input), inputSize)
	}
	if len(config.Kernel) != kernelSize {
		return nil, fmt.Errorf("kernel size mismatch: got %d, expected %d", len(config.Kernel), kernelSize)
	}
	if len(config.Bias) != biasSize {
		return nil, fmt.Errorf("bias size mismatch: got %d, expected %d", len(config.Bias), biasSize)
	}

	inputBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "conv2d_input",
		Size:  uint64(inputSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, err
	}
	defer inputBuf.Release()

	kernelBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "conv2d_kernel",
		Size:  uint64(kernelSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, err
	}
	defer kernelBuf.Release()

	biasBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "conv2d_bias",
		Size:  uint64(biasSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, err
	}
	defer biasBuf.Release()

	outputBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "conv2d_output",
		Size:  uint64(outputSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, err
	}
	defer outputBuf.Release()

	readbackBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "conv2d_readback",
		Size:  uint64(outputSize * 4),
		Usage: wgpu.BufferUsageCopyDst | wgpu.BufferUsageMapRead,
	})
	if err != nil {
		return nil, err
	}
	defer readbackBuf.Release()

	// Upload data
	queue.WriteBuffer(inputBuf, 0, unsafe.Slice((*byte)(unsafe.Pointer(&input[0])), inputSize*4))
	queue.WriteBuffer(kernelBuf, 0, unsafe.Slice((*byte)(unsafe.Pointer(&config.Kernel[0])), kernelSize*4))
	queue.WriteBuffer(biasBuf, 0, unsafe.Slice((*byte)(unsafe.Pointer(&config.Bias[0])), biasSize*4))

	// Create bind group
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "conv2d_fwd_bg",
		Layout: bgl,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: inputBuf, Offset: 0, Size: inputBuf.GetSize()},
			{Binding: 1, Buffer: kernelBuf, Offset: 0, Size: kernelBuf.GetSize()},
			{Binding: 2, Buffer: biasBuf, Offset: 0, Size: biasBuf.GetSize()},
			{Binding: 3, Buffer: outputBuf, Offset: 0, Size: outputBuf.GetSize()},
		},
	})
	if err != nil {
		return nil, err
	}
	defer bg.Release()

	// Dispatch
	workgroups := (uint32(outputSize) + 255) / 256
	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(workgroups, 1, 1)
	pass.End()

	// Copy to readback
	enc.CopyBufferToBuffer(outputBuf, 0, readbackBuf, 0, uint64(outputSize*4))
	cb, err := enc.Finish(nil)
	if err != nil {
		enc.Release()
		return nil, err
	}
	enc.Release()
	queue.Submit(cb)
	cb.Release()

	// Read results
	pollDevice(dev, 1000)
	done := false
	size := uint64(outputSize * 4)
	readbackBuf.MapAsync(wgpu.MapModeRead, 0, size, func(wgpu.BufferMapAsyncStatus) { done = true })
	for i := 0; i < 1000 && !done; i++ {
		dev.Poll(true, nil)
	}

	data := readbackBuf.GetMappedRange(0, 0) // 0 means whole buffer
	output := make([]float32, outputSize)
	copy(output, unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), outputSize))
	readbackBuf.Unmap()

	return output, nil
}
