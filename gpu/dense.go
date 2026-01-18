package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// Activation constants matching ActCodeOf in generic sense
const (
	ActNone      = 0
	ActReLU      = 1
	ActLeakyReLU = 2
	ActSigmoid   = 3
	ActTanh      = 4
)

// DenseLayerSpec defines the configuration for a single dense layer
type DenseLayerSpec struct {
	InputSize  int
	OutputSize int
	Activation int       // ActXXX constant
	Weights    []float32 // Flattened [OutputSize * InputSize]
	Biases     []float32 // [OutputSize]
}

// DenseLayer holds resources for a single layer execution
type DenseLayer struct {
	Spec      DenseLayerSpec
	BatchSize int

	pipeline        *wgpu.ComputePipeline
	bindGroupLayout *wgpu.BindGroupLayout // Explicit layout
	bindGroup       *wgpu.BindGroup

	InputBuffer   *wgpu.Buffer
	OutputBuffer  *wgpu.Buffer
	StagingBuffer *wgpu.Buffer // Only needed for last layer usually? Or debug.
	WeightBuffer  *wgpu.Buffer
	BiasBuffer    *wgpu.Buffer

	// Backward Buffers
	WeightGradientBuffer *wgpu.Buffer
	BiasGradientBuffer   *wgpu.Buffer
	InputGradientBuffer  *wgpu.Buffer // Computed gradient w.r.t input (dL/dInput)
	dZBuffer             *wgpu.Buffer // Gradient w.r.t pre-activation (dL/dZ)

	// Context for backward pipeline
	backwardPipelineDZ     *wgpu.ComputePipeline
	backwardBindGroupDZ    *wgpu.BindGroup
	backwardPipelineGrads  *wgpu.ComputePipeline
	backwardBindGroupGrads *wgpu.BindGroup
	backwardPipelineInput  *wgpu.ComputePipeline
	backwardBindGroupInput *wgpu.BindGroup

	// Explicit layouts for backward pass
	backwardBindGroupLayoutGrads *wgpu.BindGroupLayout
	backwardBindGroupLayoutInput *wgpu.BindGroupLayout

	// Gradient application bind groups (cached for training)
	GradientWeightBindGroup *wgpu.BindGroup
	GradientBiasBindGroup   *wgpu.BindGroup

	WorkgroupsX uint32
}

// DenseSequence manages a sequence of dense layers executed on GPU
type DenseSequence struct {
	Layers []*DenseLayer
	Debug  bool
}

// NewDenseSequence creates a new sequence handler
func NewDenseSequence(specs []DenseLayerSpec) *DenseSequence {
	layers := make([]*DenseLayer, len(specs))
	for i, spec := range specs {
		layers[i] = &DenseLayer{Spec: spec}
	}
	return &DenseSequence{
		Layers: layers,
	}
}

// Cleanup releases resources
func (s *DenseSequence) Cleanup() {
	for _, l := range s.Layers {
		l.Cleanup()
	}
}

// Interface Implementation
func (l *DenseLayer) GetInputBuffer() *wgpu.Buffer         { return l.InputBuffer }
func (l *DenseLayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *DenseLayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *DenseLayer) GetInputGradientBuffer() *wgpu.Buffer { return l.InputGradientBuffer }
func (l *DenseLayer) GetDZBuffer() *wgpu.Buffer            { return l.dZBuffer }

func (l *DenseLayer) Cleanup() {
	if l.InputBuffer != nil {
		l.InputBuffer.Destroy()
	}
	if l.OutputBuffer != nil {
		l.OutputBuffer.Destroy()
	}
	if l.StagingBuffer != nil {
		l.StagingBuffer.Destroy()
	}
	if l.WeightBuffer != nil {
		l.WeightBuffer.Destroy()
	}
	if l.BiasBuffer != nil {
		l.BiasBuffer.Destroy()
	}
	if l.WeightGradientBuffer != nil {
		l.WeightGradientBuffer.Destroy()
	}
	if l.BiasGradientBuffer != nil {
		l.BiasGradientBuffer.Destroy()
	}
	if l.InputGradientBuffer != nil {
		l.InputGradientBuffer.Destroy()
	}
	if l.dZBuffer != nil {
		l.dZBuffer.Destroy()
	}
	if l.pipeline != nil {
		l.pipeline.Release()
	}
	if l.bindGroup != nil {
		l.bindGroup.Release()
	}
	if l.bindGroupLayout != nil {
		// l.bindGroupLayout.Release()
	}
	if l.backwardPipelineDZ != nil {
		l.backwardPipelineDZ.Release()
	}
	if l.backwardBindGroupDZ != nil {
		l.backwardBindGroupDZ.Release()
	}
	if l.backwardPipelineGrads != nil {
		l.backwardPipelineGrads.Release()
	}
	if l.backwardBindGroupGrads != nil {
		l.backwardBindGroupGrads.Release()
	}
	if l.backwardPipelineInput != nil {
		l.backwardPipelineInput.Release()
	}
	if l.backwardBindGroupInput != nil {
		l.backwardBindGroupInput.Release()
	}
	if l.backwardBindGroupLayoutGrads != nil {
		// l.backwardBindGroupLayoutGrads.Release()
	}
	if l.backwardBindGroupLayoutInput != nil {
		// l.backwardBindGroupLayoutInput.Release()
	}
}

func (l *DenseLayer) UploadWeights(ctx *Context) {
	if l.WeightBuffer != nil {
		// Weights are [Input, Output] on CPU, but need [Output, Input] on GPU.
		transposed := transposeWeights(l.Spec.Weights, l.Spec.InputSize, l.Spec.OutputSize)
		ctx.Queue.WriteBuffer(l.WeightBuffer, 0, wgpu.ToBytes(transposed))
	}
	if l.BiasBuffer != nil {
		ctx.Queue.WriteBuffer(l.BiasBuffer, 0, wgpu.ToBytes(l.Spec.Biases))
	}
}

func transposeWeights(in []float32, rows, cols int) []float32 {
	out := make([]float32, len(in))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			// In: [Row, Col] -> [r*cols + c]
			// Out: [Col, Row] -> [c*rows + r]
			out[c*rows+r] = in[r*cols+c]
		}
	}
	return out
}

func (l *DenseLayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	// Weights matched [Input, Output] on CPU, but [Output, Input] on GPU.
	// Since verification likely assumes [Input, Output] (original),
	// we should probably untranspose if we want to inspect them?
	// But usually verification only checks Gradients.
	// However, if we do read them, we should be consistent.

	rawWeights, err := ReadBuffer(l.WeightBuffer, len(l.Spec.Weights))
	if err != nil {
		return nil, nil, fmt.Errorf("read weights: %v", err)
	}

	// Untranspose from [Output, Input] (GPU) to [Input, Output] (CPU)
	weights := transposeWeights(rawWeights, l.Spec.OutputSize, l.Spec.InputSize)

	// Biases
	biases, err := ReadBuffer(l.BiasBuffer, len(l.Spec.Biases))
	if err != nil {
		return nil, nil, fmt.Errorf("read biases: %v", err)
	}

	return weights, biases, nil
}

func (l *DenseLayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	// Weights Grads are [Output, Input] on GPU
	rawWGrad, err := ReadBuffer(l.WeightGradientBuffer, len(l.Spec.Weights))
	if err != nil {
		return nil, nil, nil, fmt.Errorf("read wgrad: %v", err)
	}

	// Transpose from [Output, Input] (GPU) -> [Input, Output] (CPU)
	wGrad := transposeWeights(rawWGrad, l.Spec.OutputSize, l.Spec.InputSize)

	// Biases
	bGrad, err := ReadBuffer(l.BiasGradientBuffer, len(l.Spec.Biases))
	if err != nil {
		return nil, nil, nil, fmt.Errorf("read bgrad: %v", err)
	}

	// Input Gradient
	iGrad, err := ReadBuffer(l.InputGradientBuffer, l.Spec.InputSize)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("read igrad: %v", err)
	}

	return wGrad, bGrad, iGrad, nil
}

func (l *DenseLayer) AllocateBackwardBuffers(ctx *Context, labelPrefix string) error {
	var err error

	batch := l.BatchSize
	if batch <= 0 {
		batch = 1
	}

	// Weight Gradient (accumulated over batch)
	l.WeightGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_WGrad",
		Size:  uint64(len(l.Spec.Weights) * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}

	// Bias Gradient (accumulated)
	l.BiasGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_BGrad",
		Size:  uint64(len(l.Spec.Biases) * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}

	// Input Gradients (Result of backward pass for this layer)
	l.InputGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_IGrad",
		Size:  uint64(l.Spec.InputSize * batch * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}

	// dZ Buffer (Intermediate derivative w.r.t pre-activation)
	l.dZBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_dZ",
		Size:  uint64(l.Spec.OutputSize * batch * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}

	// Initialize gradients to 0? Not strictly needed if shader overwrites,
	// but good practice if accumulating. The shader will overwrite in this simple implementation.

	return nil
}

// generateShaderForLayer creates WGSL for a specific layer configuration
func (l *DenseLayer) GenerateShader() string {
	// Map activation Int to code
	actFunc := "return x;"
	switch l.Spec.Activation {
	case ActReLU:
		// Match CPU ScaledReLU (1.1x)
		actFunc = "return select(0.0, 1.1 * x, x > 0.0); "
	case ActLeakyReLU:
		actFunc = "return select(0.01 * x, x, x > 0.0);"
	case ActSigmoid:
		actFunc = "return 1.0 / (1.0 + exp(-x));"
	case ActTanh:
		actFunc = "return tanh(x);"
	}

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read_write> output : array<f32>;
		@group(0) @binding(2) var<storage, read> weights : array<f32>;
		@group(0) @binding(3) var<storage, read> biases : array<f32>;

		fn activate(x: f32) -> f32 {
			%s
		}

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let n_out = %du;
			let n_in = %du;

			if (idx >= arrayLength(&output)) {
				return;
			}
			
			// Handle batch dimension mapping
			// idx = sample_idx * n_out + out_idx
			let sample_idx = idx / n_out;
			let out_idx = idx %% n_out;

			var sum: f32 = biases[out_idx];
			let weight_offset = out_idx * n_in;
			let input_offset = sample_idx * n_in;
			
			for (var i: u32 = 0u; i < n_in; i++) {
				sum += weights[weight_offset + i] * input[input_offset + i];
			}

			output[idx] = activate(sum);
		}
	`, actFunc, l.Spec.OutputSize, l.Spec.InputSize)
}

func (l *DenseLayer) AllocateBuffers(ctx *Context, labelPrefix string) error {
	if Debug {
		Log("Allocating buffers for %s (Batch: %d)", labelPrefix, l.BatchSize)
	}
	var err error

	batch := l.BatchSize
	if batch <= 0 {
		batch = 1
	}

	// Input
	l.InputBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_In",
		Size:  uint64(l.Spec.InputSize * batch * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Output
	l.OutputBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Out",
		Size:  uint64(l.Spec.OutputSize * batch * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Weights
	l.WeightBuffer, err = NewFloatBuffer(l.Spec.Weights, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
	if err != nil {
		return fmt.Errorf("weight buf: %v", err)
	}

	// Biases
	l.BiasBuffer, err = NewFloatBuffer(l.Spec.Biases, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
	if err != nil {
		return fmt.Errorf("bias buf: %v", err)
	}

	// Staging
	l.StagingBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Staging",
		Size:  uint64(l.Spec.OutputSize * batch * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	return nil
}

func (l *DenseLayer) Compile(ctx *Context, labelPrefix string) error {
	if Debug {
		Log("Compiling dense layer %s", labelPrefix)
	}
	shader := l.GenerateShader()
	module, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_Shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shader},
	})
	if err != nil {
		return fmt.Errorf("shader compile: %v", err)
	}

	// Explicit Bind Group Layout to avoid "auto" layout issues in WASM
	l.bindGroupLayout, err = ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: labelPrefix + "_BGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Input
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // Output
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Weights
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // Biases
		},
	})
	if err != nil {
		return fmt.Errorf("create bgl: %v", err)
	}

	// Debug
	pipelineLayout, err := ctx.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            labelPrefix + "_Layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{l.bindGroupLayout},
	})
	if err != nil {
		return fmt.Errorf("create pipeline layout: %v", err)
	}

	l.pipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  labelPrefix + "_Pipe",
		Layout: pipelineLayout, // Explicit Layout!
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return fmt.Errorf("pipeline create: %v", err)
	}
	module.Release()

	batch := l.BatchSize
	if batch <= 0 {
		batch = 1
	}
	totalThreads := uint32(l.Spec.OutputSize * batch)
	l.WorkgroupsX = (totalThreads + 255) / 256
	return nil
}

func (l *DenseLayer) CreateBindGroup(ctx *Context, labelPrefix string) error {
	var err error
	l.bindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_Bind",
		Layout: l.bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
			{Binding: 1, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
			{Binding: 2, Buffer: l.WeightBuffer, Size: l.WeightBuffer.GetSize()},
			{Binding: 3, Buffer: l.BiasBuffer, Size: l.BiasBuffer.GetSize()},
		},
	})
	return err
}

// Build initializes all GPU resources for all layers
func (s *DenseSequence) Build() error {
	ctx, err := GetContext()
	if err != nil {
		return err
	}

	for i, l := range s.Layers {
		label := fmt.Sprintf("L%d", i)
		if err := l.AllocateBuffers(ctx, label); err != nil {
			return err
		}
		if err := l.Compile(ctx, label); err != nil {
			return err
		}
	}
	return nil
}

// Dispatch records the compute pass for this layer
func (l *DenseLayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	if Debug {
		Log("Dispatching dense layer w/ %d workgroups", l.WorkgroupsX)
	}
	pass.SetPipeline(l.pipeline)
	pass.SetBindGroup(0, l.bindGroup, nil)
	pass.DispatchWorkgroups(l.WorkgroupsX, 1, 1)
}

// Forward executes the sequence on GPU
func (s *DenseSequence) Forward(input []float32) ([]float32, error) {
	if len(s.Layers) == 0 {
		return nil, fmt.Errorf("no layers built")
	}

	ctx, err := GetContext()
	if err != nil {
		return nil, err
	}

	enc, err := ctx.Device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}

	// 1. Write Input to Layer 0
	l0 := s.Layers[0]
	if len(input)*4 != int(l0.InputBuffer.GetSize()) {
		// If input size mismatches config, that's an issue.
	}
	ctx.Queue.WriteBuffer(l0.InputBuffer, 0, wgpu.ToBytes(input))

	// 2. Dispatch Sequentially
	for i, l := range s.Layers {
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(l.pipeline)
		pass.SetBindGroup(0, l.bindGroup, nil)
		pass.DispatchWorkgroups(l.WorkgroupsX, 1, 1)
		pass.End()

		// Copy Output to Next Input (if exists)
		if i < len(s.Layers)-1 {
			next := s.Layers[i+1]
			// Copy l.OutputBuffer -> next.InputBuffer
			enc.CopyBufferToBuffer(l.OutputBuffer, 0, next.InputBuffer, 0, l.OutputBuffer.GetSize())
		} else {
			// Last Layer: Copy to Staging for Readback
			enc.CopyBufferToBuffer(l.OutputBuffer, 0, l.StagingBuffer, 0, l.OutputBuffer.GetSize())
		}
	}

	// 3. Submit
	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, err
	}
	ctx.Queue.Submit(cmd)

	// 4. Readback Last Layer
	last := s.Layers[len(s.Layers)-1]
	return readStagingBuffer(ctx, last.StagingBuffer, last.Spec.OutputSize)
}

func readStagingBuffer(c *Context, buf *wgpu.Buffer, size int) ([]float32, error) {
	// Sync Wait
	done := make(chan struct{})
	var mapErr error

	buf.MapAsync(wgpu.MapModeRead, 0, buf.GetSize(), func(status wgpu.BufferMapAsyncStatus) {
		if status != wgpu.BufferMapAsyncStatusSuccess {
			mapErr = fmt.Errorf("map status: %d", status)
		}
		close(done)
	})

	// Poll
Loop:
	for {
		c.Device.Poll(true, nil)
		select {
		case <-done:
			break Loop
		default:
		}
	}

	if mapErr != nil {
		return nil, mapErr
	}

	data := buf.GetMappedRange(0, uint(buf.GetSize()))
	defer buf.Unmap()

	if data == nil {
		return nil, fmt.Errorf("mapped range nil")
	}

	// Copy out
	out := make([]float32, size)
	copy(out, wgpu.FromBytes[float32](data))

	return out, nil
}

// ForwardPipelined executes the sequence using multiple command encoders
func (s *DenseSequence) ForwardPipelined(input []float32) ([]float32, error) {
	// If only 1 layer, fallback to standard Forward
	if len(s.Layers) < 2 {
		return s.Forward(input)
	}

	ctx, err := GetContext()
	if err != nil {
		return nil, err
	}

	// Prepare Input (Layer 0)
	l0 := s.Layers[0]
	ctx.Queue.WriteBuffer(l0.InputBuffer, 0, wgpu.ToBytes(input))

	// Pipeline: Submit Layer 0 locally
	enc0, _ := ctx.Device.CreateCommandEncoder(nil)
	pass0 := enc0.BeginComputePass(nil)
	pass0.SetPipeline(l0.pipeline)
	pass0.SetBindGroup(0, l0.bindGroup, nil)
	pass0.DispatchWorkgroups(l0.WorkgroupsX, 1, 1)
	pass0.End()

	// Copy L0 Out -> L1 In
	l1 := s.Layers[1]
	enc0.CopyBufferToBuffer(l0.OutputBuffer, 0, l1.InputBuffer, 0, l0.OutputBuffer.GetSize())

	cmd0, err := enc0.Finish(nil)
	if err != nil {
		return nil, err
	}
	ctx.Queue.Submit(cmd0)

	// Dispatch remaining layers
	for i := 1; i < len(s.Layers); i++ {
		l := s.Layers[i]
		enc, _ := ctx.Device.CreateCommandEncoder(nil)

		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(l.pipeline)
		pass.SetBindGroup(0, l.bindGroup, nil)
		pass.DispatchWorkgroups(l.WorkgroupsX, 1, 1)
		pass.End()

		if i < len(s.Layers)-1 {
			next := s.Layers[i+1]
			// Copy l.OutputBuffer -> next.InputBuffer
			enc.CopyBufferToBuffer(l.OutputBuffer, 0, next.InputBuffer, 0, l.OutputBuffer.GetSize())
		} else {
			// Last Layer: Copy to Staging
			enc.CopyBufferToBuffer(l.OutputBuffer, 0, l.StagingBuffer, 0, l.OutputBuffer.GetSize())
		}

		cmd, err := enc.Finish(nil)
		if err != nil {
			return nil, err
		}
		ctx.Queue.Submit(cmd)
	}

	// Readback
	last := s.Layers[len(s.Layers)-1]
	return readStagingBuffer(ctx, last.StagingBuffer, last.Spec.OutputSize)
}
