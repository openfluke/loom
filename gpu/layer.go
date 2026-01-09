package gpu

import "github.com/openfluke/webgpu/wgpu"

// GPULayer is the common interface for all GPU-accelerated layers
type GPULayer interface {
	// Initialization
	AllocateBuffers(ctx *Context, labelPrefix string) error
	AllocateBackwardBuffers(ctx *Context, labelPrefix string) error // Ensure this exists or make optional?
	Compile(ctx *Context, labelPrefix string) error
	CompileBackward(ctx *Context, labelPrefix string) error
	CreateBindGroup(ctx *Context, labelPrefix string) error
	CreateBackwardBindGroup(ctx *Context, labelPrefix string, dOutputBuffer *wgpu.Buffer) error

	// Execution
	Dispatch(pass *wgpu.ComputePassEncoder)
	DispatchBackward(enc *wgpu.CommandEncoder)

	// Data Transfer
	UploadWeights(ctx *Context)
	DownloadWeights(ctx *Context) ([]float32, []float32, error) // Might need to be generic or interface{}? Or just return raw buffers?
	// For verification, we usually need specific things.
	// Dense: W, B
	// LayerNorm: Gamma, Beta
	// Let's keep it specific or use specialized accessors.
	// Actually for DownloadGradients, we return (grad1, grad2, gradInput).
	// Let's use DownloadGradients() ([]float32, []float32, []float32, error)
	// assuming most layers have at most 2 learnable params + input grad.
	DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error)

	// Resource Access (for chaining)
	GetInputBuffer() *wgpu.Buffer
	GetOutputBuffer() *wgpu.Buffer
	GetStagingBuffer() *wgpu.Buffer
	GetInputGradientBuffer() *wgpu.Buffer // For Backward Chaining
	ZeroGradients(ctx *Context)

	Cleanup()
}
