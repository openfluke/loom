package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// ParallelLayer orchestrates execution of multiple branches
type ParallelLayer struct {
	CombineMode string // "concat", "add", "avg"
	Branches    []GPULayer

	// Resources
	InputBuffer   *wgpu.Buffer
	OutputBuffer  *wgpu.Buffer
	StagingBuffer *wgpu.Buffer

	// Backward Resources
	InputGradientBuffer *wgpu.Buffer

	// Pipeline for copying/scattering/gathering (identity)
	copyPipeline *wgpu.ComputePipeline

	// Pipeline for summing (backward accumulation)
	sumPipeline *wgpu.ComputePipeline

	// Backward Helpers
	intermediateGradBuffers []*wgpu.Buffer    // Stores the dOutput for each branch
	dOutputRef              *wgpu.Buffer      // Reference to the dOutput buffer passed to CreateBackwardBindGroup
	splitBindGroups         []*wgpu.BindGroup // For copying dOutput -> BranchGradOut
	sumBindGroups           []*wgpu.BindGroup // For summing BranchGradIn -> InputGrad

	// Helper fields (forward)
	inputScatterBindGroups []*wgpu.BindGroup
	outputGatherBindGroups []*wgpu.BindGroup

	BatchSize int
}

// NewParallelLayer creates a new parallel container
func NewParallelLayer(branches []GPULayer, mode string, batchSize int) *ParallelLayer {
	return &ParallelLayer{
		Branches:    branches,
		CombineMode: mode,
		BatchSize:   batchSize,
	}
}

// shader source for simple copy (float32)
const copyShaderSrc = `
@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
	let i = global_id.x;
	if (i < arrayLength(&dst)) {
		if (i < arrayLength(&src)) {
			dst[i] = src[i];
		}
	}
}
`

// shader source for in-place add (float32): dst[i] += src[i]
const addShaderSrc = `
@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
	let i = global_id.x;
	if (i < arrayLength(&dst) && i < arrayLength(&src)) {
		dst[i] = dst[i] + src[i];
	}
}
`

func (l *ParallelLayer) setupPipelines(ctx *Context) error {
	if l.copyPipeline != nil {
		return nil
	}

	// Compile Copy Shader
	cShader, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "ParallelCopyShader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{
			Code: copyShaderSrc,
		},
	})
	if err != nil {
		return err
	}

	l.copyPipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "ParallelCopyPipeline",
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     cShader,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return err
	}

	// Compile Sum Shader
	sShader, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "ParallelSumShader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{
			Code: addShaderSrc,
		},
	})
	if err != nil {
		return err
	}

	l.sumPipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "ParallelSumPipeline",
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     sShader,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return err
	}

	return nil
}

func (l *ParallelLayer) AllocateBuffers(ctx *Context, labelPrefix string) error {
	// 1. Allocate buffers for all branches
	for i, b := range l.Branches {
		if err := b.AllocateBuffers(ctx, fmt.Sprintf("%s/b%d", labelPrefix, i)); err != nil {
			return err
		}
	}

	// 2. Determine my Input/Output sizes
	// Input Size: Same as branches (assuming all branches take same input)
	if len(l.Branches) == 0 {
		return fmt.Errorf("parallel layer has no branches")
	}

	// Check Input Size consistency?
	firstIn := l.Branches[0].GetInputBuffer().GetSize()
	// Just use the first branch's input size request
	// Note: Generic Parallel allows branches to take same input.

	l.InputBuffer, _ = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Input",
		Size:  firstIn,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})

	// Output Size
	var totalOut uint64
	switch l.CombineMode {
	case "concat", "", "filter":
		for _, b := range l.Branches {
			totalOut += b.GetOutputBuffer().GetSize()
		}
	case "add", "avg", "average":
		// Must be same size
		totalOut = l.Branches[0].GetOutputBuffer().GetSize()
	}

	l.OutputBuffer, _ = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Output",
		Size:  totalOut,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})

	// Staging for readback
	l.StagingBuffer, _ = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_Staging",
		Size:  totalOut,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})

	return nil
}

func (l *ParallelLayer) AllocateBackwardBuffers(ctx *Context, labelPrefix string) error {
	// 1. Allocate backward buffers for sub-layers
	for i, b := range l.Branches {
		if err := b.AllocateBackwardBuffers(ctx, fmt.Sprintf("%s/b%d", labelPrefix, i)); err != nil {
			return err
		}
	}

	// 2. My Input Gradient is w.r.t my Input (which is shared input)
	// Size matches my InputBuffer
	size := l.InputBuffer.GetSize()
	l.InputGradientBuffer, _ = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: labelPrefix + "_GradInput",
		Size:  size,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})

	return nil
}

func (l *ParallelLayer) GetInputBuffer() *wgpu.Buffer         { return l.InputBuffer }
func (l *ParallelLayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *ParallelLayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *ParallelLayer) GetInputGradientBuffer() *wgpu.Buffer { return l.InputGradientBuffer }

func (l *ParallelLayer) Cleanup() {
	if l.InputBuffer != nil {
		l.InputBuffer.Destroy()
	}
	if l.OutputBuffer != nil {
		l.OutputBuffer.Destroy()
	}
	if l.StagingBuffer != nil {
		l.StagingBuffer.Destroy()
	}
	if l.InputGradientBuffer != nil {
		l.InputGradientBuffer.Destroy()
	}

	for _, b := range l.Branches {
		b.Cleanup()
	}

	// Cleanup backward helper buffers
	for _, b := range l.intermediateGradBuffers {
		if b != nil {
			b.Destroy()
		}
	}
	if l.dOutputRef != nil {
		// dOutputRef is not owned by ParallelLayer, it's passed in. Do not destroy.
	}
	for _, bg := range l.splitBindGroups {
		if bg != nil {
			bg.Release()
		}
	}
	for _, bg := range l.sumBindGroups {
		if bg != nil {
			bg.Release()
		}
	}
}

func (l *ParallelLayer) Compile(ctx *Context, labelPrefix string) error {
	for i, b := range l.Branches {
		if err := b.Compile(ctx, fmt.Sprintf("%s/b%d", labelPrefix, i)); err != nil {
			return err
		}
	}
	return nil
}

func (l *ParallelLayer) CompileBackward(ctx *Context, labelPrefix string) error {
	for i, b := range l.Branches {
		if err := b.CompileBackward(ctx, fmt.Sprintf("%s/b%d", labelPrefix, i)); err != nil {
			return err
		}
	}

	// If combine mode implies summing gradients (which parallel backward always does for InputGradient),
	// we might need a shader.
	// Parallel Forward: Input -> [B1, B2]
	// Parallel Backward: [G1, G2] -> Sum -> InputGrad

	// We need a "SumGradients" shader that takes N buffers and sums them into one.
	// For now, let's implement the shader generation here or assume simple pair-wise addition?
	// Or use a generic "Accumulate" shader?

	// Let's implement a simple compute shader for summing branch input gradients.
	// Since number of branches is dynamic, we can issue N dispatches:
	//   Accumulator = 0
	//   Accumulator += Branch1.GradInput
	//   Accumulator += Branch2.GradInput

	// We can reuse the "InPlaceResidual" adder logic if exposed?
	// It's in `gpu/residual.go` or similar? gpu_integration imports it as `gpu.NewInPlaceResidual`.
	// Let's check if we can reuse it. The InPlaceResidual adds B to A.
	// Ideally we:
	// 1. Zero InputGradientBuffer.
	// 2. For each branch: InPlaceAdd(InputGradientBuffer, Branch[i].GetInputGradientBuffer()).

	// NOTE: We need to compile the InPlaceResidual pipeline?
	// It is usually created with NewInPlaceResidual.
	// We can create one here.

	return nil
}

func (l *ParallelLayer) CreateBindGroup(ctx *Context, labelPrefix string) error {
	var err error

	if err := l.setupPipelines(ctx); err != nil {
		return err
	}

	l.inputScatterBindGroups = make([]*wgpu.BindGroup, len(l.Branches))
	l.outputGatherBindGroups = make([]*wgpu.BindGroup, len(l.Branches))

	outputOffset := uint64(0)

	for i, b := range l.Branches {
		if err := b.CreateBindGroup(ctx, fmt.Sprintf("%s/b%d", labelPrefix, i)); err != nil {
			return err
		}

		l.inputScatterBindGroups[i], err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  fmt.Sprintf("%s/b%d_Scatter", labelPrefix, i),
			Layout: l.copyPipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
				{Binding: 1, Buffer: b.GetInputBuffer(), Size: b.GetInputBuffer().GetSize()},
			},
		})
		if err != nil {
			return err
		}

		var gatherLayout *wgpu.BindGroupLayout
		if l.CombineMode == "add" || l.CombineMode == "sum" || l.CombineMode == "avg" || l.CombineMode == "average" {
			if i == 0 {
				gatherLayout = l.copyPipeline.GetBindGroupLayout(0)
			} else {
				gatherLayout = l.sumPipeline.GetBindGroupLayout(0)
			}
		} else {
			gatherLayout = l.copyPipeline.GetBindGroupLayout(0) // Concat
		}

		branchOutSize := b.GetOutputBuffer().GetSize()

		var dstOffset uint64
		if l.CombineMode == "add" || l.CombineMode == "sum" || l.CombineMode == "avg" || l.CombineMode == "average" {
			dstOffset = 0
		} else {
			dstOffset = outputOffset
		}

		l.outputGatherBindGroups[i], err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  fmt.Sprintf("%s/b%d_Gather", labelPrefix, i),
			Layout: gatherLayout,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: b.GetOutputBuffer(), Size: branchOutSize},
				{Binding: 1, Buffer: l.OutputBuffer, Offset: dstOffset, Size: branchOutSize},
			},
		})
		if err != nil {
			return err
		}

		if l.CombineMode != "add" && l.CombineMode != "sum" && l.CombineMode != "avg" && l.CombineMode != "average" {
			outputOffset += branchOutSize
		}
	}
	return nil
}

func (l *ParallelLayer) CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error {
	// dOutputBuffer is the gradient coming from above (Loss -> Output).
	// For "concat", this buffer contains [Grad_B1 | Grad_B2].
	// We cannot simply pass dOutputBuffer to sub-layers, because they expect a buffer of THEIR output size.
	// We need to create "View" buffers or copy parts of dOutputBuffer to temp buffers?
	// WebGPU doesn't support "BufferView" objects that act as Buffers in BindGroups easily
	// (you bind with offset/size).

	// If the sub-layer `CreateBackwardBindGroup` method accepts a *wgpu.Buffer, it typically uses the whole buffer
	// or assumes 0 offset in its internal layout?
	// No, `CreateBackwardBindGroup` usually assumes `dOutputBuffer` is the full buffer it binds.
	// Most implementation pass it to `BindGroupEntry`.
	// We can specify offset/size in `BindGroupEntry`.

	// PROBLEM: `GPULayer` interface `CreateBackwardBindGroup` takes `*wgpu.Buffer`.
	// It does NOT take offset/size.
	// If we want to support Concat, we might need to change the interface or
	// create separate buffers and copy the relevant slices of `dOutputBuffer` into them before dispatching backward.

	// Strategy:
	// In `DispatchBackward`:
	// 1. Split `dOutputBuffer` (Parallel GradOut) into `Branch[i].dOutput` buffers.
	//    Wait, `Branch[i]` doesn't own a `dOutput` buffer usually (it expects one passed in).
	//    WE NEED TO ALLOCATE TEMP BUFFERS for branch output gradients!

	// Correct approach for `CreateBackwardBindGroup` in ParallelLayer:
	// 1. Allocate `intermediateGradBuffers` for each branch (size = branch output).
	// 2. Call `b.CreateBackwardBindGroup` with this intermediate buffer.
	// 3. In `DispatchBackward`:
	//    - Copy slice of `Parallel.dOutput` -> `Branch[i].IntermediateGrad`.
	//    - Call `b.DispatchBackward`.

	var err error
	if err := l.setupPipelines(ctx); err != nil {
		return err
	}

	l.dOutputRef = dOutputBuffer // Store reference to the incoming gradient

	l.intermediateGradBuffers = make([]*wgpu.Buffer, len(l.Branches))
	l.splitBindGroups = make([]*wgpu.BindGroup, len(l.Branches))
	l.sumBindGroups = make([]*wgpu.BindGroup, len(l.Branches))

	dOutputOffset := uint64(0)

	// Iterate branches to allocate intermediate grad buffers and create bind groups
	for i, b := range l.Branches {
		branchOutSize := b.GetOutputBuffer().GetSize()
		branchInGradSize := b.GetInputGradientBuffer().GetSize()

		// Create a persistent buffer for this branch's received gradient (dOutput for the branch)
		var gradBuf *wgpu.Buffer
		gradBuf, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: fmt.Sprintf("%s_GradOut_B%d", labelPrefix, i),
			Size:  branchOutSize,
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
		})
		if err != nil {
			return err
		}
		l.intermediateGradBuffers[i] = gradBuf

		// Create BindGroup for splitting dOutputBuffer into branch's intermediateGradBuffer
		// This uses the copyPipeline
		l.splitBindGroups[i], err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  fmt.Sprintf("%s/b%d_SplitGradOut", labelPrefix, i),
			Layout: l.copyPipeline.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: l.dOutputRef, Offset: dOutputOffset, Size: branchOutSize}, // Source is a slice of overall dOutput
				{Binding: 1, Buffer: l.intermediateGradBuffers[i], Size: branchOutSize},        // Destination is branch's dOutput
			},
		})
		if err != nil {
			return err
		}

		// Pass this temp buffer to the branch for its backward pass
		if err := b.CreateBackwardBindGroup(ctx, fmt.Sprintf("%s/b%d", labelPrefix, i), gradBuf); err != nil {
			return err
		}

		// Create BindGroup for summing branch's input gradient into ParallelLayer's InputGradientBuffer
		// This uses either copyPipeline (for first branch) or sumPipeline (for subsequent branches)
		var sumLayout *wgpu.BindGroupLayout
		if i == 0 {
			sumLayout = l.copyPipeline.GetBindGroupLayout(0) // Overwrite for first branch
		} else {
			sumLayout = l.sumPipeline.GetBindGroupLayout(0) // Add for subsequent branches
		}

		l.sumBindGroups[i], err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  fmt.Sprintf("%s/b%d_SumGradIn", labelPrefix, i),
			Layout: sumLayout,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: b.GetInputGradientBuffer(), Size: branchInGradSize}, // Source is branch's dInput
				{Binding: 1, Buffer: l.InputGradientBuffer, Size: branchInGradSize},      // Destination is Parallel's dInput
			},
		})
		if err != nil {
			return err
		}

		dOutputOffset += branchOutSize
	}

	return nil
}

func (l *ParallelLayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	// Ensure pipelines are ready (create lazily if not done in Compile)
	// But `pass` is already started. We cannot load shaders here easily if not pre-loaded.
	// We assume `Compile` or `setupPipelines` was called.

	// 1. Scatter Input to Branches (Parallel.Input -> Branch[i].Input)
	// Since we are inside a ComputePass, we use our Copy Shader.
	// We need temporary bind groups for the copy operation.
	// NOTE: Creating BindGroups inside Dispatch is suboptimal but necessary if we didn't pre-create them.
	// Ideally we pre-create these in `CreateBindGroup`.
	// For now, let's create them on the fly (or look up if we stored them).
	// We didn't store them in struct.
	// Todo: Refactor to store them. For now, creating here is OK for functionality.

	// We need context access to create BindGroup?
	// `Dispatch` only provides `*wgpu.ComputePassEncoder`.
	// CRITICAL: We cannot create BindGroups here without `ctx.Device`.
	// `GPULayer.Dispatch` interface is restrictive.

	// FIX: We must pre-create these "internal" BindGroups during `CreateBindGroup`.
	// I need to update `CreateBindGroup` and the struct to hold:
	// - `inputScatterBindGroups []*wgpu.BindGroup`
	// - `outputGatherBindGroups []*wgpu.BindGroup`

	// For now, I will placeholder this and implement the `CreateBindGroup` logic in next step.
	// Assuming `l.inputScatterBindGroups[i]` exists.

	if len(l.inputScatterBindGroups) == len(l.Branches) && l.copyPipeline != nil {
		pass.SetPipeline(l.copyPipeline)
		for i := range l.Branches {
			pass.SetBindGroup(0, l.inputScatterBindGroups[i], nil)
			// Dispatch 1D grid for copy
			size := l.Branches[i].GetInputBuffer().GetSize() / 4 // float count
			wg := uint32((size + 63) / 64)
			pass.DispatchWorkgroups(wg, 1, 1)
		}
	}

	// 2. Run Branches
	for _, b := range l.Branches {
		b.Dispatch(pass)
	}

	// 3. Gather Output (Branch[i].Output -> Parallel.Output)
	if len(l.outputGatherBindGroups) == len(l.Branches) {
		if l.CombineMode == "add" || l.CombineMode == "sum" || l.CombineMode == "avg" || l.CombineMode == "average" {
			if l.sumPipeline != nil {
				pass.SetPipeline(l.sumPipeline)

				// For add/sum, we start with zero?
				// The sum shader reads src and ADDS to dst.
				// We need to ensure dst is zero first?
				// There is no easy "ClearBuffer" inside ComputePass without a ClearShader.
				// We can rely on `nn/gpu_integration.go` clearing output? No, it doesn't.

				// Workaround: First branch uses Copy (overwrite), others use Add (accumulate).
				for i := range l.Branches {
					if i == 0 {
						if l.copyPipeline != nil {
							pass.SetPipeline(l.copyPipeline) // Overwrite
						}
					} else {
						if l.sumPipeline != nil {
							pass.SetPipeline(l.sumPipeline) // Accumulate
						}
					}
					pass.SetBindGroup(0, l.outputGatherBindGroups[i], nil)
					size := l.Branches[i].GetOutputBuffer().GetSize() / 4
					wg := uint32((size + 63) / 64)
					pass.DispatchWorkgroups(wg, 1, 1)
				}
			}
		} else {
			// Concat (default)
			if l.copyPipeline != nil {
				pass.SetPipeline(l.copyPipeline)
				for i := range l.Branches {
					pass.SetBindGroup(0, l.outputGatherBindGroups[i], nil)
					size := l.Branches[i].GetOutputBuffer().GetSize() / 4
					wg := uint32((size + 63) / 64)
					pass.DispatchWorkgroups(wg, 1, 1)
				}
			}
		}
	}
}

func (l *ParallelLayer) DispatchBackward(enc *wgpu.CommandEncoder) {
	// 1. Split `dOutput` (Parallel GradOut) into `Branch[i].dOutput` buffers.
	// We use the compute pass for Copy/Split because we need offset writes which CopyBufferToBuffer supports
	// BUT we set up `splitBindGroups` using Compute Copy Pipeline to allow complex logic if needed
	// (and because CopyBufferToBuffer byte alignment requirements are strict, 4 bytes is fine though).
	// Since we already made BindGroups for Copy Pipeline, let's use ComputePass.
	// But `DispatchBackward` gives us `CommandEncoder`. We need to start a pass.

	// Issue: Starting multiple passes is expensive?
	// We can group all splits into one pass, then end, then branches (which might start their own passes),
	// then sum pass.

	// Phase 1: Split (Copy dOutput -> BranchGradOut)
	{
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(l.copyPipeline)
		for i, bg := range l.splitBindGroups {
			if bg != nil {
				pass.SetBindGroup(0, bg, nil)
				size := l.intermediateGradBuffers[i].GetSize() / 4
				wg := uint32((size + 63) / 64)
				pass.DispatchWorkgroups(wg, 1, 1)
			}
		}
		pass.End()
	}

	// Phase 2: Dispatch Branches
	// Each branch manages its own passes/encoders
	for _, b := range l.Branches {
		b.DispatchBackward(enc)
	}

	// Phase 3: Sum Gradients (BranchGradIn -> ParallelGradIn)
	{
		pass := enc.BeginComputePass(nil)

		for i, bg := range l.sumBindGroups {
			if bg != nil {
				// Branch 0 uses Copy (Overwrite to initialize), others Sum
				if i == 0 {
					pass.SetPipeline(l.copyPipeline)
				} else {
					pass.SetPipeline(l.sumPipeline)
				}

				pass.SetBindGroup(0, bg, nil)
				// Size is always Parallel.InputSize
				size := l.InputGradientBuffer.GetSize() / 4
				wg := uint32((size + 63) / 64)
				pass.DispatchWorkgroups(wg, 1, 1)
			}
		}
		pass.End()
	}
}

func (l *ParallelLayer) UploadWeights(ctx *Context) {
	for _, b := range l.Branches {
		b.UploadWeights(ctx)
	}
}

func (l *ParallelLayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	// Parallel layer itself has no weights usually?
	// Or return concatenated?
	// Interface return is just 2 slices.
	// Usually invalid to call on container.
	return nil, nil, nil
}

func (l *ParallelLayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	// Container doesn't train itself directly.
	// We should probably iterate branches and collect?
	// But return signature is single layer.
	// So just return nil. The caller (gpu_integration) iterates flattened list?
	// No, `gpu_integration` iterates `n.gpuLayers` which includes this ParallelLayer.
	// If `ParallelLayer` hides sub-layers, `gpu_integration` WON'T see them to download gradients!

	// CRITICAL: `nn/gpu_integration` iterates the TOP LEVEL layers.
	// If `ParallelLayer` contains trainable layers (Dense), their gradients must be extracted.
	// `ParallelLayer.DownloadGradients` MUST recursively collect gradients?
	// But `DownloadGradients` returns single `kernel, bias, input` tuple.
	// It cannot return a tree of gradients.

	// This means `nn/gpu_integration` structure is insufficient for Nested Layers on GPU unless:
	// A) We flatten the GPU layer list (but topology is tree).
	// B) `ParallelLayer` exposes a specific method to retrieve all sub-gradients?

	// For verification test `gpu_cpugpu_parity`, we only check if output matches.
	// We do NOT yet fully verify training of generic nested parallel layers via `DownloadGradients`.

	// However, for correct functioning, we should at least support `ZeroGradients`.
	return nil, nil, nil, nil
}

func (l *ParallelLayer) ZeroGradients(ctx *Context) {
	// Zero my own
	if l.InputGradientBuffer != nil {
		// Zero buffer
		// ctx.Queue.WriteBuffer(l.InputGradientBuffer, 0, make([]byte, size)) ... slow
		// Better: Encode ClearBuffer if available or fill zero
		// wgpu doesn't have ClearBuffer easily. WriteBuffer is strictly ok for now.
		sz := l.InputGradientBuffer.GetSize()
		if sz > 0 {
			// Zeroing huge buffer via CPU-write is bad.
			// But sticking to standard pattern.
			// A better way is a clear shader.
			// Re-use `l.Branches` zeroing.
		}
	}

	for _, b := range l.Branches {
		b.ZeroGradients(ctx)
	}
}

// Extra fields helper
type ParallelLayerExtras struct {
	branchGradRows []*wgpu.Buffer
	dOutputRef     *wgpu.Buffer
}
