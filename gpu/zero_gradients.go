package gpu

import "github.com/openfluke/webgpu/wgpu"

// ZeroGradients for DenseLayer (Empty as backward overwrites d_weights and d_bias)
func (l *DenseLayer) ZeroGradients(ctx *Context) {}

// ZeroGradients for Conv1DLayer
func (l *Conv1DLayer) ZeroGradients(ctx *Context) {
	// Zero Weight Gradients
	wSize := l.Spec.OutChannels * l.Spec.InChannels * l.Spec.KernelSize
	// Note: Creating a large slice every frame might be slow.
	// Optimization: Keep a persistent zero buffer or use WriteBuffer with a cleared slice.
	// For now, this is functionally correct.
	ctx.Queue.WriteBuffer(l.WeightGradientBuffer, 0, wgpu.ToBytes(make([]float32, wSize)))

	// Zero Bias Gradients
	ctx.Queue.WriteBuffer(l.BiasGradientBuffer, 0, wgpu.ToBytes(make([]float32, l.Spec.OutChannels)))
}

// ZeroGradients for Conv2DLayer (Empty as it overwrites d_input, no weight grads yet)
func (l *Conv2DLayer) ZeroGradients(ctx *Context) {}

// ZeroGradients for RNNLayer (Empty as it overwrites)
func (l *RNNLayer) ZeroGradients(ctx *Context) {}

// ZeroGradients for MHALayer (Empty as it overwrites)
func (l *MHALayer) ZeroGradients(ctx *Context) {}

// ZeroGradients for SoftmaxLayer (Empty as it overwrites)
func (l *SoftmaxLayer) ZeroGradients(ctx *Context) {}

// ZeroGradients for ResidualLayer (Empty as it overwrites)
func (l *ResidualLayer) ZeroGradients(ctx *Context) {}

// ZeroGradients for SwiGLULayer (Empty as it overwrites/not fully implemented backward)
func (l *SwiGLULayer) ZeroGradients(ctx *Context) {}
