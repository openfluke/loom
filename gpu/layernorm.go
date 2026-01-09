package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

type LayerNormSpec struct {
	NormSize  int
	BatchSize int // Number of vectors to normalize (default 1)
	Epsilon   float32
	Gamma     []float32 // [NormSize]
	Beta      []float32 // [NormSize]
}

type LayerNormLayer struct {
	Spec LayerNormSpec

	pipeline  *wgpu.ComputePipeline
	bindGroup *wgpu.BindGroup

	InputBuffer   *wgpu.Buffer
	OutputBuffer  *wgpu.Buffer
	StagingBuffer *wgpu.Buffer
	GammaBuffer   *wgpu.Buffer
	BetaBuffer    *wgpu.Buffer

	// Backward
	GammaGradientBuffer *wgpu.Buffer
	BetaGradientBuffer  *wgpu.Buffer
	InputGradientBuffer *wgpu.Buffer

	bwPipeline  *wgpu.ComputePipeline
	bwBindGroup *wgpu.BindGroup

	WorkgroupsX uint32
}

// Implement GPULayer Interface
func (l *LayerNormLayer) GetInputBuffer() *wgpu.Buffer         { return l.InputBuffer }
func (l *LayerNormLayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *LayerNormLayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *LayerNormLayer) GetInputGradientBuffer() *wgpu.Buffer { return l.InputGradientBuffer }

func (l *LayerNormLayer) AllocateBuffers(ctx *Context, labelPrefix string) error {
	var err error

	// Ensure batch size is at least 1
	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	totalSize := batch * l.Spec.NormSize

	// Input: batch * normSize elements
	l.InputBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_In",
		Size:  uint64(totalSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Output: batch * normSize elements
	l.OutputBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Out",
		Size:  uint64(totalSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Gamma: shared across batch (just normSize)
	if len(l.Spec.Gamma) > 0 {
		l.GammaBuffer, err = NewFloatBuffer(l.Spec.Gamma, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
		if err != nil {
			return err
		}
	} else {
		// Provide dummy 1-element buffer to satisfy binding layout if missing
		l.GammaBuffer, err = NewFloatBuffer([]float32{1.0}, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
		if err != nil {
			return err
		}
	}

	// Beta: shared across batch (just normSize)
	if len(l.Spec.Beta) > 0 {
		l.BetaBuffer, err = NewFloatBuffer(l.Spec.Beta, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
		if err != nil {
			return err
		}
	} else {
		l.BetaBuffer, err = NewFloatBuffer([]float32{0.0}, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
		if err != nil {
			return err
		}
	}

	// Staging: batch * normSize elements
	l.StagingBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Staging",
		Size:  uint64(totalSize * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	return err
}

func (l *LayerNormLayer) AllocateBackwardBuffers(ctx *Context, labelPrefix string) error {
	var err error

	// Ensure batch size is at least 1
	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	totalSize := batch * l.Spec.NormSize

	// Input Grad: batch * normSize elements
	l.InputGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_InGrad",
		Size:  uint64(totalSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Gamma Grad: shared across batch (just normSize)
	sz := len(l.Spec.Gamma)
	if sz == 0 {
		sz = 1
	}
	l.GammaGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_GammaGrad",
		Size:  uint64(sz * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	// Beta Grad: shared across batch (just normSize)
	sz = len(l.Spec.Beta)
	if sz == 0 {
		sz = 1
	}
	l.BetaGradientBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_BetaGrad",
		Size:  uint64(sz * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}

	return nil
}

func (l *LayerNormLayer) GenerateShader() string {
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read_write> output : array<f32>;
		@group(0) @binding(2) var<storage, read> gamma : array<f32>;
		@group(0) @binding(3) var<storage, read> beta : array<f32>;

		const N: u32 = %du;
		const EPS: f32 = %f;

		@compute @workgroup_size(1)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let b = gid.x;
			let offset = b * N;

			var sum: f32 = 0.0;
			for (var i: u32 = 0u; i < N; i++) {
				sum += input[offset + i];
			}
			let mean = sum / f32(N);

			var sumSq: f32 = 0.0;
			for (var i: u32 = 0u; i < N; i++) {
				let diff = input[offset + i] - mean;
				sumSq += diff * diff;
			}
			let var_val = sumSq / f32(N);
			let stdDev = sqrt(var_val + EPS);

			for (var i: u32 = 0u; i < N; i++) {
				let val = input[offset + i];
				let norm = (val - mean) / stdDev;
				
				// Optional gamma lookup - assumes 0 if OOB? No, we bound buffer.
				// If gamma array is size 1 (dummy), use 1.0. 
				// But real implement expects array of size N.
				// For now assumes correct size matching N.
				var g: f32 = 1.0;
				if (arrayLength(&gamma) >= N) { g = gamma[i]; }
				
				var bt: f32 = 0.0;
				if (arrayLength(&beta) >= N) { bt = beta[i]; }

				output[offset + i] = norm * g + bt;
			}
		}
	`, l.Spec.NormSize, l.Spec.Epsilon)
}

func (l *LayerNormLayer) GenerateBackwardShader() string {
	// Need to calculate:
	// dGamma, dBeta, dInput
	// Inputs: dOutput, Input, Gamma, (Mean/Std recomputed or stored?)
	// Recomputing Mean/Std is standard for memory efficiency.

	// Math:
	// dBeta[i] = sum over batch (dOutput[b, i])
	// dGamma[i] = sum over batch (dOutput[b, i] * x_hat[b, i])
	// dInput[b, i] = ... complex ...

	// For atomic float add, we use u32 atomics with bitcast
	// d_gamma and d_beta are declared as atomic<u32> and we inline CAS loops

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read> d_output : array<f32>;
		@group(0) @binding(2) var<storage, read> gamma : array<f32>;
		
		@group(0) @binding(3) var<storage, read_write> d_input : array<f32>;
		@group(0) @binding(4) var<storage, read_write> d_gamma : array<atomic<u32>>;
		@group(0) @binding(5) var<storage, read_write> d_beta : array<atomic<u32>>;

		const N: u32 = %du;
		const EPS: f32 = %f;

		@compute @workgroup_size(1)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let b = gid.x;
			let offset = b * N;
			
			// 1. Recompute Mean/Std/x_hat
			var sum: f32 = 0.0;
			for (var i: u32 = 0u; i < N; i++) {
				sum += input[offset + i];
			}
			let mean = sum / f32(N);

			var sumSq: f32 = 0.0;
			for (var i: u32 = 0u; i < N; i++) {
				let diff = input[offset + i] - mean;
				sumSq += diff * diff;
			}
			let var_val = sumSq / f32(N);
			let stdDev = sqrt(var_val + EPS);
			let invStd = 1.0 / stdDev;
			
			// 2. Accumulate gradients with inline atomic adds
			var sum_dxhat: f32 = 0.0;
			var sum_dxhat_xhat: f32 = 0.0;
			
			for (var i: u32 = 0u; i < N; i++) {
				let idx = offset + i;
				let val = input[idx];
				let dout = d_output[idx];
				
				let x_hat = (val - mean) * invStd;
				
				// Inline atomic add for d_beta[i]
				{
					var old_val: u32 = atomicLoad(&d_beta[i]);
					loop {
						let old_f32 = bitcast<f32>(old_val);
						let new_f32 = old_f32 + dout;
						let new_val = bitcast<u32>(new_f32);
						let result = atomicCompareExchangeWeak(&d_beta[i], old_val, new_val);
						if (result.exchanged) {
							break;
						}
						old_val = result.old_value;
					}
				}
				
				// Inline atomic add for d_gamma[i]
				{
					let gamma_contrib = dout * x_hat;
					var old_val: u32 = atomicLoad(&d_gamma[i]);
					loop {
						let old_f32 = bitcast<f32>(old_val);
						let new_f32 = old_f32 + gamma_contrib;
						let new_val = bitcast<u32>(new_f32);
						let result = atomicCompareExchangeWeak(&d_gamma[i], old_val, new_val);
						if (result.exchanged) {
							break;
						}
						old_val = result.old_value;
					}
				}
				
				// Backprop to Input
				var g: f32 = 1.0;
				if (arrayLength(&gamma) >= N) { g = gamma[i]; }
				
				let d_xhat = dout * g;
				
				sum_dxhat += d_xhat;
				sum_dxhat_xhat += d_xhat * x_hat;
			}
			
			// 3. Compute dInput (per-batch, no accumulation needed)
			for (var i: u32 = 0u; i < N; i++) {
				let idx = offset + i;
				let val = input[idx];
				let x_hat = (val - mean) * invStd;
				
				var g: f32 = 1.0;
				if (arrayLength(&gamma) >= N) { g = gamma[i]; }
				
				let dout = d_output[idx];
				let d_xhat = dout * g;
				
				let dx = invStd * (d_xhat - (sum_dxhat / f32(N)) - x_hat * (sum_dxhat_xhat / f32(N)));
				d_input[idx] = dx;
			}
		}
	`, l.Spec.NormSize, l.Spec.Epsilon)
}

func (l *LayerNormLayer) Compile(ctx *Context, labelPrefix string) error {
	shader := l.GenerateShader()
	module, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_Shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shader},
	})
	if err != nil {
		return err
	}

	l.pipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_Pipe",
		Compute: wgpu.ProgrammableStageDescriptor{Module: module, EntryPoint: "main"},
	})
	return err
}

func (l *LayerNormLayer) CompileBackward(ctx *Context, labelPrefix string) error {
	shader := l.GenerateBackwardShader()
	module, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_BwdShader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shader},
	})
	if err != nil {
		return err
	}

	l.bwPipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_BwdPipe",
		Compute: wgpu.ProgrammableStageDescriptor{Module: module, EntryPoint: "main"},
	})
	return err
}

func (l *LayerNormLayer) CreateBindGroup(ctx *Context, labelPrefix string) error {
	var err error

	fmt.Printf("[DEBUG] LN CreateBindGroup: In=%d Out=%d Gam=%d Bet=%d\n",
		l.InputBuffer.GetSize(), l.OutputBuffer.GetSize(),
		l.GammaBuffer.GetSize(), l.BetaBuffer.GetSize())

	entries := []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
		{Binding: 1, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
		{Binding: 2, Buffer: l.GammaBuffer, Size: l.GammaBuffer.GetSize()},
		{Binding: 3, Buffer: l.BetaBuffer, Size: l.BetaBuffer.GetSize()},
	}
	l.bindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   labelPrefix + "_Bind",
		Layout:  l.pipeline.GetBindGroupLayout(0),
		Entries: entries,
	})
	return err
}

func (l *LayerNormLayer) CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error {
	var err error
	entries := []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
		{Binding: 1, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
		{Binding: 2, Buffer: l.GammaBuffer, Size: l.GammaBuffer.GetSize()},
		{Binding: 3, Buffer: l.InputGradientBuffer, Size: l.InputGradientBuffer.GetSize()},
		{Binding: 4, Buffer: l.GammaGradientBuffer, Size: l.GammaGradientBuffer.GetSize()},
		{Binding: 5, Buffer: l.BetaGradientBuffer, Size: l.BetaGradientBuffer.GetSize()},
	}
	l.bwBindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   labelPrefix + "_BwdBind",
		Layout:  l.bwPipeline.GetBindGroupLayout(0),
		Entries: entries,
	})
	return err
}

func (l *LayerNormLayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	pass.SetPipeline(l.pipeline)
	pass.SetBindGroup(0, l.bindGroup, nil)
	// Launch one workgroup per batch sample
	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	pass.DispatchWorkgroups(uint32(batch), 1, 1)
}

func (l *LayerNormLayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(l.bwPipeline)
	pass.SetBindGroup(0, l.bwBindGroup, nil)
	// Launch one workgroup per batch sample
	batch := l.Spec.BatchSize
	if batch < 1 {
		batch = 1
	}
	pass.DispatchWorkgroups(uint32(batch), 1, 1)
	pass.End()
}

func (l *LayerNormLayer) UploadWeights(ctx *Context) {
	if len(l.Spec.Gamma) > 0 {
		ctx.Queue.WriteBuffer(l.GammaBuffer, 0, wgpu.ToBytes(l.Spec.Gamma))
	}
	if len(l.Spec.Beta) > 0 {
		ctx.Queue.WriteBuffer(l.BetaBuffer, 0, wgpu.ToBytes(l.Spec.Beta))
	}
}

// ZeroGradients zeroes gradient buffers before backward pass (required for atomic accumulation)
func (l *LayerNormLayer) ZeroGradients(ctx *Context) {
	// Zero gamma gradient buffer
	sz := len(l.Spec.Gamma)
	if sz == 0 {
		sz = 1
	}
	zeros := make([]float32, sz)
	ctx.Queue.WriteBuffer(l.GammaGradientBuffer, 0, wgpu.ToBytes(zeros))

	// Zero beta gradient buffer
	sz = len(l.Spec.Beta)
	if sz == 0 {
		sz = 1
	}
	zeros = make([]float32, sz)
	ctx.Queue.WriteBuffer(l.BetaGradientBuffer, 0, wgpu.ToBytes(zeros))
}

func (l *LayerNormLayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	// Gamma
	sz := len(l.Spec.Gamma)
	if sz == 0 {
		sz = 1
	}
	gamma, err := ReadBuffer(l.GammaBuffer, sz)
	if err != nil {
		return nil, nil, err
	}

	// Beta
	sz = len(l.Spec.Beta)
	if sz == 0 {
		sz = 1
	}
	beta, err := ReadBuffer(l.BetaBuffer, sz)
	if err != nil {
		return nil, nil, err
	}

	return gamma, beta, nil
}

func (l *LayerNormLayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	// Verify signatures match main.go expectation
	// wGrad, bGrad, iGrad

	// Gamma Grad
	sz := len(l.Spec.Gamma)
	if sz == 0 {
		sz = 1
	}
	gGrad, err := ReadBuffer(l.GammaGradientBuffer, sz)
	if err != nil {
		return nil, nil, nil, err
	}

	// Beta Grad
	sz = len(l.Spec.Beta)
	if sz == 0 {
		sz = 1
	}
	bGrad, err := ReadBuffer(l.BetaGradientBuffer, sz)
	if err != nil {
		return nil, nil, nil, err
	}

	// Input Grad
	iGrad, err := ReadBuffer(l.InputGradientBuffer, l.Spec.NormSize) // Batch=1 assumed
	if err != nil {
		return nil, nil, nil, err
	}

	return gGrad, bGrad, iGrad, nil
}

func (l *LayerNormLayer) Cleanup() {
	if l.InputBuffer != nil {
		l.InputBuffer.Destroy()
	}
	if l.OutputBuffer != nil {
		l.OutputBuffer.Destroy()
	}
	if l.StagingBuffer != nil {
		l.StagingBuffer.Destroy()
	}
	if l.GammaBuffer != nil {
		l.GammaBuffer.Destroy()
	}
	if l.BetaBuffer != nil {
		l.BetaBuffer.Destroy()
	}
	if l.InputGradientBuffer != nil {
		l.InputGradientBuffer.Destroy()
	}
	if l.GammaGradientBuffer != nil {
		l.GammaGradientBuffer.Destroy()
	}
	if l.BetaGradientBuffer != nil {
		l.BetaGradientBuffer.Destroy()
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
