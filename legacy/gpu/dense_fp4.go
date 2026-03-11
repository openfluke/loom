package gpu

import (
	"fmt"
	"math/bits"

	"github.com/openfluke/webgpu/wgpu"
)

// =============================================================================
// FP4DenseLayer — Dense matrix multiply with weights stored as packed E2M1
// nibbles in GPU VRAM.  Weight buffer is half the size of a float32 DenseLayer.
//
// Layout (matches nn.PackedWeights):
//   packed:  uint8 array, 2 nibbles per byte (low=even, high=odd element)
//   scales:  float32 array, one scale per MicroScaleGroup (16) row-elements
//             indexed as scales[col * numRowGroups + rowGroup]
//
// WGSL shader unpacks nibbles → applies scale → accumulates into float32.
// =============================================================================

const fp4MicroScaleGroup = 16 // must match nn.MicroScaleGroup

// FP4DenseSpec is the configuration for an FP4 dense layer.
type FP4DenseSpec struct {
	InputSize    int
	OutputSize   int
	NumRowGroups int // = ceil(InputSize / fp4MicroScaleGroup)
	// PackedData: packed nibbles (from PackedWeights.Data): len = ceil(InputSize*OutputSize / 2)
	PackedData []uint8
	// Per-column per-row-group scales: len = OutputSize * NumRowGroups
	Scales []float32
	// Biases (may be nil)
	Biases []float32
}

// FP4DenseLayer implements gpu.GPULayer with nibble-packed weight storage.
type FP4DenseLayer struct {
	Spec      FP4DenseSpec
	BatchSize int // updated per-dispatch (= seqLen)

	pipeline        *wgpu.ComputePipeline
	bindGroupLayout *wgpu.BindGroupLayout
	bindGroup       *wgpu.BindGroup

	InputBuffer   *wgpu.Buffer
	OutputBuffer  *wgpu.Buffer
	StagingBuffer *wgpu.Buffer
	PackedBuffer  *wgpu.Buffer // uint32 array (nibbles packed 8-per-u32)
	ScaleBuffer   *wgpu.Buffer // float32 array
	BiasBuffer    *wgpu.Buffer

	WorkgroupsX  uint32
	InputAliased bool
}

// ─────────────────────────────────────────────────────────────────────────────
// GPULayer interface — buffer accessors
// ─────────────────────────────────────────────────────────────────────────────

func (l *FP4DenseLayer) GetInputBuffer() *wgpu.Buffer         { return l.InputBuffer }
func (l *FP4DenseLayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *FP4DenseLayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *FP4DenseLayer) GetInputGradientBuffer() *wgpu.Buffer { return nil }
func (l *FP4DenseLayer) GetDZBuffer() *wgpu.Buffer            { return nil }

func (l *FP4DenseLayer) SetInputBuffer(buf *wgpu.Buffer) {
	l.InputBuffer = buf
	l.InputAliased = true
}

// ─────────────────────────────────────────────────────────────────────────────
// Cleanup
// ─────────────────────────────────────────────────────────────────────────────

func (l *FP4DenseLayer) Cleanup() {
	if l.InputBuffer != nil && !l.InputAliased {
		l.InputBuffer.Destroy()
	}
	for _, b := range []*wgpu.Buffer{l.OutputBuffer, l.StagingBuffer, l.PackedBuffer, l.ScaleBuffer, l.BiasBuffer} {
		if b != nil {
			b.Destroy()
		}
	}
	if l.pipeline != nil {
		l.pipeline.Release()
	}
	if l.bindGroup != nil {
		l.bindGroup.Release()
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffer allocation
// ─────────────────────────────────────────────────────────────────────────────

func (l *FP4DenseLayer) AllocateBuffers(ctx *Context, label string) error {
	batch := l.BatchSize
	if batch < 1 {
		batch = 1
	}

	var err error

	// Input buffer (float32)
	if !l.InputAliased {
		l.InputBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
			Label: label + "_FP4In",
			Size:  uint64(batch * l.Spec.InputSize * 4),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
		})

		if err != nil {
			return fmt.Errorf("fp4 input buf: %w", err)
		}
	}

	// Output buffer (float32)
	l.OutputBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: label + "_FP4Out",
		Size:  uint64(l.Spec.OutputSize * batch * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})

	if err != nil {
		return fmt.Errorf("fp4 output buf: %w", err)
	}

	// Packed weight buffer (uint8 packed nibbles → upload as u32 array)
	// We round up to 4-byte alignment for WebGPU (u32 buffer).
	packedByteLen := len(l.Spec.PackedData)
	// Pad to multiple of 4 for u32 buffer
	paddedU32Len := (packedByteLen + 3) / 4
	l.PackedBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: label + "_FP4W",
		Size:  uint64(paddedU32Len * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})

	if err != nil {
		return fmt.Errorf("fp4 packed buf: %w", err)
	}

	// Scale buffer (float32, per-column per-row-group)
	l.ScaleBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: label + "_FP4S",
		Size:  uint64(len(l.Spec.Scales) * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})

	if err != nil {
		return fmt.Errorf("fp4 scale buf: %w", err)
	}

	// Bias buffer — use a 1-element zero buffer if no biases
	biases := l.Spec.Biases
	if len(biases) == 0 {
		biases = make([]float32, l.Spec.OutputSize)
	}
	l.BiasBuffer, err = NewFloatBuffer(biases, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return fmt.Errorf("fp4 bias buf: %w", err)
	}

	// Staging buffer
	l.StagingBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: label + "_FP4Stg",
		Size:  uint64(l.Spec.OutputSize * batch * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})

	if err != nil {
		return fmt.Errorf("fp4 staging buf: %w", err)
	}
	return nil
}

func (l *FP4DenseLayer) AllocateBackwardBuffers(ctx *Context, label string) error {
	return nil // FP4 is inference-only for now
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight upload
// ─────────────────────────────────────────────────────────────────────────────

func (l *FP4DenseLayer) UploadWeights(ctx *Context) {
	if l.PackedBuffer != nil && len(l.Spec.PackedData) > 0 {
		u32data := packBytesToU32(l.Spec.PackedData)
		ctx.Queue.WriteBuffer(l.PackedBuffer, 0, wgpu.ToBytes(u32data))
	}
	if l.ScaleBuffer != nil && len(l.Spec.Scales) > 0 {
		ctx.Queue.WriteBuffer(l.ScaleBuffer, 0, wgpu.ToBytes(l.Spec.Scales))
	}
	if l.BiasBuffer != nil {
		biases := l.Spec.Biases
		if len(biases) == 0 {
			biases = make([]float32, l.Spec.OutputSize)
		}
		ctx.Queue.WriteBuffer(l.BiasBuffer, 0, wgpu.ToBytes(biases))
	}
}

func packBytesToU32(b []uint8) []uint32 {
	n := (len(b) + 3) / 4
	out := make([]uint32, n)
	for i, v := range b {
		out[i/4] |= uint32(v) << (uint(i%4) * 8)
	}
	return out
}

func (l *FP4DenseLayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	return nil, nil, nil // weights are packed, not directly usable as float32
}

func (l *FP4DenseLayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	return nil, nil, nil, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Shader — WGSL FP4 dequant matmul
// ─────────────────────────────────────────────────────────────────────────────

// generateFP4Shader generates the WGSL compute shader that:
//  1. For each (sample, out_col) output element:
//     - iterates over input dimension in groups of MICRO_SCALE_GROUP
//     - unpacks 4-bit nibbles from the u32 weight buffer
//     - decodes E2M1 magnitude and sign
//     - multiplies decoded weight × input element
//     - applies per-group scale wScale (from scales[out_col*numRowGroups+g])
//     - accumulates into float32
//  2. Adds bias and writes output
func (l *FP4DenseLayer) generateFP4Shader() string {
	// E2M1 magnitude table embedded as WGSL array literal.
	// Index = bits[2:0] of nibble → float32 magnitude
	// {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
	return fmt.Sprintf(`
// ── FP4 E2M1 dense matmul shader ────────────────────────────────────────────
// Weights are packed as 4-bit E2M1 nibbles in pairs inside u32 words.
// Two nibbles per byte: byte = (highNibble << 4) | lowNibble
// Four bytes per u32:  u32 = byte3<<24 | byte2<<16 | byte1<<8 | byte0
//
// E2M1 encoding: [S | E1 E0 | M0]  (S=sign, E=2-bit exp biased by 1, M=mantissa)
// Magnitude table: 0→0, 1→0.5, 2→1, 3→1.5, 4→2, 5→3, 6→4, 7→6

const N_OUT : u32 = %du;
const N_IN  : u32 = %du;
const NUM_ROW_GROUPS : u32 = %du;
const MICRO_SCALE_GROUP : u32 = %du;

@group(0) @binding(0) var<storage, read>       input   : array<f32>;
@group(0) @binding(1) var<storage, read_write>  output  : array<f32>;
@group(0) @binding(2) var<storage, read>        packed  : array<u32>;  // nibble pairs
@group(0) @binding(3) var<storage, read>        scales  : array<f32>;  // [out * numRowGroups + g]
@group(0) @binding(4) var<storage, read>        biases  : array<f32>;

// Decode E2M1 nibble → float32 (magnitude only, sign handled separately)
fn fp4_mag(mag3: u32) -> f32 {
    // mag3 = bits[2:0] of nibble (3 bits)
    switch(mag3) {
        case 0u: { return 0.0; }
        case 1u: { return 0.5; }
        case 2u: { return 1.0; }
        case 3u: { return 1.5; }
        case 4u: { return 2.0; }
        case 5u: { return 3.0; }
        case 6u: { return 4.0; }
        default: { return 6.0; } // case 7
    }
}

// Extract nibble at flat index k from the packed u32 buffer.
// k is the element index (0-based).
//   byte_idx = k / 2
//   nibble is low (k even) or high (k odd)
//   u32 word index = byte_idx / 4, byte within u32 = byte_idx %% 4
fn get_nibble(k: u32) -> u32 {
    let byte_idx = k >> 1u;
    let word_idx = byte_idx >> 2u;
    let byte_in_word = byte_idx & 3u;
    let word = packed[word_idx];
    let byte_val = (word >> (byte_in_word * 8u)) & 0xFFu;
    if (k & 1u) == 0u {
        return byte_val & 0x0Fu;        // low nibble
    } else {
        return (byte_val >> 4u) & 0x0Fu; // high nibble
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = arrayLength(&output);
    if (idx >= total) { return; }

    // Map linear index to (sample, output_column)
    let sample  = idx / N_OUT;
    let out_col = idx %% N_OUT;

    var acc : f32 = biases[out_col];

    var g : u32 = 0u;
    loop {
        if g >= NUM_ROW_GROUPS { break; }

        let row_start = g * MICRO_SCALE_GROUP;
        var row_end = row_start + MICRO_SCALE_GROUP;
        if row_end > N_IN { row_end = N_IN; }

        // Weight scale for this (output_column, row_group)
        let w_scale = scales[out_col * NUM_ROW_GROUPS + g];

        var group_acc : f32 = 0.0;

        var row : u32 = row_start;
        loop {
            if row >= row_end { break; }

            // Weight flat index: weight[row][out_col] = row * N_OUT + out_col
            let w_flat = row * N_OUT + out_col;
            let nibble  = get_nibble(w_flat);

            // Decode E2M1 nibble
            let w_sign = (nibble >> 3u) & 1u;
            let w_mag  = fp4_mag(nibble & 0x7u);
            let w_val  = select(w_mag, -w_mag, w_sign == 1u);

            // Input element for this sample and row dimension
            let a_val = input[sample * N_IN + row];

            group_acc += w_val * a_val;

            row += 1u;
            continuing { }
        }

        // Apply weight-group scale — activation is full float32 so no aScale
        acc += group_acc * w_scale;

        g += 1u;
        continuing { }
    }

    output[idx] = acc;
}
`, l.Spec.OutputSize, l.Spec.InputSize, l.Spec.NumRowGroups, fp4MicroScaleGroup)
}

// ─────────────────────────────────────────────────────────────────────────────
// Compile
// ─────────────────────────────────────────────────────────────────────────────

func (l *FP4DenseLayer) Compile(ctx *Context, label string) error {
	shader := l.generateFP4Shader()
	mod, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          label + "_FP4Shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shader},
	})
	if err != nil {
		return fmt.Errorf("fp4 shader compile: %w", err)
	}
	defer mod.Release()

	// 5 bindings: input, output, packed, scales, biases
	l.bindGroupLayout, err = ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: label + "_FP4BGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 4, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
		},
	})
	if err != nil {
		return fmt.Errorf("fp4 bgl: %w", err)
	}

	pl, err := ctx.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            label + "_FP4Layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{l.bindGroupLayout},
	})
	if err != nil {
		return fmt.Errorf("fp4 pipeline layout: %w", err)
	}

	l.pipeline, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  label + "_FP4Pipe",
		Layout: pl,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     mod,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return fmt.Errorf("fp4 pipeline: %w", err)
	}

	batch := l.BatchSize
	if batch < 1 {
		batch = 1
	}
	total := uint32(l.Spec.OutputSize * batch)
	l.WorkgroupsX = (total + 255) / 256
	return nil
}

func (l *FP4DenseLayer) CompileBackward(ctx *Context, label string) error { return nil }

func (l *FP4DenseLayer) CreateBindGroup(ctx *Context, label string) error {
	var err error
	l.bindGroup, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  label + "_FP4Bind",
		Layout: l.bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: l.InputBuffer, Size: l.InputBuffer.GetSize()},
			{Binding: 1, Buffer: l.OutputBuffer, Size: l.OutputBuffer.GetSize()},
			{Binding: 2, Buffer: l.PackedBuffer, Size: l.PackedBuffer.GetSize()},
			{Binding: 3, Buffer: l.ScaleBuffer, Size: l.ScaleBuffer.GetSize()},
			{Binding: 4, Buffer: l.BiasBuffer, Size: l.BiasBuffer.GetSize()},
		},
	})
	return err
}

func (l *FP4DenseLayer) CreateBackwardBindGroup(ctx *Context, label string, dOut *wgpu.Buffer) error {
	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Runtime
// ─────────────────────────────────────────────────────────────────────────────

func (l *FP4DenseLayer) UpdateParams(ctx *Context, seqTokens int, cachePos int) {
	if seqTokens > 0 {
		l.BatchSize = seqTokens
		total := uint32(l.Spec.OutputSize * l.BatchSize)
		l.WorkgroupsX = (total + 255) / 256
	}
}

func (l *FP4DenseLayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	pass.SetPipeline(l.pipeline)
	pass.SetBindGroup(0, l.bindGroup, nil)
	pass.DispatchWorkgroups(l.WorkgroupsX, 1, 1)
}

func (l *FP4DenseLayer) DispatchBackward(enc *wgpu.CommandEncoder) {}

func (l *FP4DenseLayer) ZeroGradients(ctx *Context) {}

// ─────────────────────────────────────────────────────────────────────────────
// VRAM size helpers
// ─────────────────────────────────────────────────────────────────────────────

// WeightBytesGPU returns the number of bytes the packed weights occupy in VRAM.
func (l *FP4DenseLayer) WeightBytesGPU() int {
	paddedU32Len := (len(l.Spec.PackedData) + 3) / 4
	scaleBytes := len(l.Spec.Scales) * 4
	return paddedU32Len*4 + scaleBytes
}

// WeightBytesFloat32 returns what a float32 equivalent would cost.
func (l *FP4DenseLayer) WeightBytesFloat32() int {
	return l.Spec.InputSize * l.Spec.OutputSize * 4
}

// leading-zeros helper needed by packBytesToU32 (uses bits package)
var _ = bits.LeadingZeros32
