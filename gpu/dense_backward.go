package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// BackwardResources holds buffers and pipelines for backprop
type BackwardResources struct {
	errBufs   []*wgpu.Buffer
	targetBuf *wgpu.Buffer
	lrBuf     *wgpu.Buffer
	clipBuf   *wgpu.Buffer
}

// EnsureBackwardInitialized sets up buffers for backward pass
func (s *DenseSequence) EnsureBackwardInitialized() error {
	if s.backward != nil {
		return nil
	}

	c, err := GetContext()
	if err != nil {
		return err
	}

	br := &BackwardResources{}

	// Error buffers (one per layer output)
	for i, l := range s.Layers {
		buf, err := c.Device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: fmt.Sprintf("Error_%d", i),
			Size:  uint64(l.Spec.OutputSize * 4),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
		})
		if err != nil {
			return err
		}
		br.errBufs = append(br.errBufs, buf)
	}

	// Target buffer (sized for last layer)
	lastSize := s.Layers[len(s.Layers)-1].Spec.OutputSize
	br.targetBuf, err = c.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "Targets",
		Size:  uint64(lastSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}

	// LR & Clip
	br.lrBuf, err = NewFloatBuffer([]float32{0.01}, wgpu.BufferUsageUniform|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}
	br.clipBuf, err = NewFloatBuffer([]float32{1.0, -1.0}, wgpu.BufferUsageUniform|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}

	s.backward = br
	return nil
}

// Actually, let's just make two shader functions and two pipelines.
func (l *DenseLayer) GenerateBackwardShaderDZ() string {
	// Derivative act(y) -> f'(z)
	deriv := "return 1.0;"
	switch l.Spec.Activation {
	case ActReLU:
		deriv = "return select(0.0, 1.1, y > 0.0);"
	case ActLeakyReLU:
		deriv = "return select(0.01, 1.0, y > 0.0);" // y>0 implies z>0
	case ActSigmoid:
		deriv = "return y * (1.0 - y);"
	case ActTanh:
		deriv = "return 1.0 - y * y;"
	}

	return fmt.Sprintf(`
		@group(0) @binding(1) var<storage, read> d_output : array<f32>;
		@group(0) @binding(5) var<storage, read_write> d_bias : array<f32>; // stores dZ
		@group(0) @binding(6) var<storage, read> output : array<f32>;

		fn deriv(y: f32) -> f32 {
			%s
		}

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			if (idx >= %du) { return; }
			
			let y = output[idx];
			let dout = d_output[idx];
			d_bias[idx] = dout * deriv(y);
			// d_bias[idx] = 999.0; // DEBUG
		}
	`, deriv, l.Spec.OutputSize)
}

func (l *DenseLayer) GenerateBackwardShaderGrads() string {
	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(2) var<storage, read> weights : array<f32>;
		
		@group(0) @binding(3) var<storage, read_write> d_input : array<f32>;
		@group(0) @binding(4) var<storage, read_write> d_weights : array<f32>;
		@group(0) @binding(5) var<storage, read> d_bias : array<f32>; // dZ (computed previously)

		@compute @workgroup_size(256)
		fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
			let idx = gid.x;
			let n_in = %du;
			let n_out = %du;
			
			// Map global ID to dWeights or dInput tasks?
			// Simplest: 1D dispatch covers max(n_in, n_in*n_out).
			// If idx < n_in * n_out: compute dW[idx]
			// If idx < n_in: compute dInput[idx]
			
			// dW[j, i] = dZ[j] * input[i]
			// weights/d_weights layout: [Output, Input] (Row-Major)
			// idx maps to j * n_in + i
			
			if (idx < n_in * n_out) {
				let j = idx / n_in; // Output index
				let i = idx %% n_in; // Input index
				d_weights[idx] = d_bias[j] * input[i];
			}
			
			// dInput[i] = sum_j (dZ[j] * W[j, i])
			// Only compute if idx < n_in (idx is i)
			if (idx < n_in) {
				var sum: f32 = 0.0;
				for (var j: u32 = 0u; j < n_out; j++) {
					// W[j, i] is at j * n_in + i
					sum += d_bias[j] * weights[j * n_in + idx];
				}
				d_input[idx] = sum;
			}
		}
	`, l.Spec.InputSize, l.Spec.OutputSize)
}

func (l *DenseLayer) CompileBackward(ctx *Context, labelPrefix string) error {
	// Pipeline 1: DZ
	shader1 := l.GenerateBackwardShaderDZ()
	mod1, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_BwdDZ_Shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shader1},
	})
	if err != nil {
		return err
	}

	l.backwardPipelineDZ, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_BwdDZ_Pipe",
		Compute: wgpu.ProgrammableStageDescriptor{Module: mod1, EntryPoint: "main"},
	})
	if err != nil {
		return err
	}

	// Pipeline 2: Grads
	shader2 := l.GenerateBackwardShaderGrads()
	mod2, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          labelPrefix + "_BwdGrads_Shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shader2},
	})
	if err != nil {
		return err
	}

	l.backwardPipelineGrads, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   labelPrefix + "_BwdGrads_Pipe",
		Compute: wgpu.ProgrammableStageDescriptor{Module: mod2, EntryPoint: "main"},
	})
	if err != nil {
		return err
	}

	return nil
}

func (l *DenseLayer) CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error {
	// If dOutputBuffer is nil (last layer), we need to create one?
	// Or pass it in Dispatch? Dispatch argument is better for chaining.
	// But BindGroup creation needs the buffer.
	// We can recreate BindGroup every frame (expensive) or create it once if topology static.
	// Topology IS static. So dOutputBuffer (which is NextLayer.InputGradientBuffer) is known.

	if dOutputBuffer == nil {
		// Create a dummy 1-size buffer or handle external Error buffer (for last layer)
		// For last layer, this is the "Targets" - "Output" diff usually?
		// Or strictly dOutput passed from Loss function.
		// Let's assume the caller provides correct buffer.
		return fmt.Errorf("dOutputBuffer cannot be nil")
	}

	// BG 1: DZ
	// Bindings: 1:d_out, 5:d_bias (dZ), 6:output
	var err error
	l.backwardBindGroupDZ, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_BwdDZ_BG",
		Layout: l.backwardPipelineDZ.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 1, Buffer: dOutputBuffer, Size: dOutputBuffer.GetSize()},
			{Binding: 5, Buffer: l.BiasGradientBuffer, Size: l.BiasGradientBuffer.GetSize()},
			{Binding: 6, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
		},
	})
	if err != nil {
		return err
	}

	// BG 2: Grads
	// Bindings: 0:input, 2:weights, 3:d_input, 4:d_weights, 5:d_bias
	l.backwardBindGroupGrads, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  labelPrefix + "_BwdGrads_BG",
		Layout: l.backwardPipelineGrads.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
			{Binding: 2, Buffer: l.WeightBuffer, Size: l.WeightBuffer.GetSize()},
			{Binding: 3, Buffer: l.InputGradientBuffer, Size: l.InputGradientBuffer.GetSize()},
			{Binding: 4, Buffer: l.WeightGradientBuffer, Size: l.WeightGradientBuffer.GetSize()},
			{Binding: 5, Buffer: l.BiasGradientBuffer, Size: l.BiasGradientBuffer.GetSize()},
		},
	})
	return err
}

func (l *DenseLayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	// Pass 1: DZ
	{
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(l.backwardPipelineDZ)
		pass.SetBindGroup(0, l.backwardBindGroupDZ, nil)
		wgx := (l.Spec.OutputSize + 255) / 256
		pass.DispatchWorkgroups(uint32(wgx), 1, 1) // 1D dispatch over outputs
		pass.End()
	}

	// Pass 2: Grads
	{
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(l.backwardPipelineGrads)
		pass.SetBindGroup(0, l.backwardBindGroupGrads, nil)

		// Work size: max(Input, Input*Output)
		total := l.Spec.InputSize * l.Spec.OutputSize
		if l.Spec.InputSize > total {
			total = l.Spec.InputSize
		}

		wgx2 := (total + 255) / 256
		pass.DispatchWorkgroups(uint32(wgx2), 1, 1)
		pass.End()
	}
}
