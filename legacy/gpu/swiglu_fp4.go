package gpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// =============================================================================
// FP4SwiGLULayer — SwiGLU with packed E2M1 nibble weights in VRAM.
//
// Implements the same 3-stage pipeline as SwiGLULayer but weight buffers are
// nibble-packed u32 arrays instead of float32.
//
// Stage 1: GateUp  — two simultaneous FP4 matmuls (gate, up) into intermediate
// Stage 2: Activate — SiLU(gate) * up
// Stage 3: Down     — FP4 matmul from intermediate back to hidden size
// =============================================================================

type FP4SwiGLUSpec struct {
	InputSize        int
	IntermediateSize int
	// Gate and Up projections: input→intermediate
	NumRowGroupsGUp int
	GateData        []uint8
	GateScales      []float32
	UpData          []uint8
	UpScales        []float32
	// Down projection: intermediate→input
	NumRowGroupsDown int
	DownData         []uint8
	DownScales       []float32
}

type FP4SwiGLULayer struct {
	Spec      FP4SwiGLUSpec
	BatchSize int

	// Stage 1: gate + up projections (two separate FP4 dense dispatches)
	pipelineGate *wgpu.ComputePipeline
	pipelineUp   *wgpu.ComputePipeline
	bglGate      *wgpu.BindGroupLayout
	bglUp        *wgpu.BindGroupLayout
	bgGate       *wgpu.BindGroup
	bgUp         *wgpu.BindGroup

	// Stage 2: activate (SiLU + multiply)
	pipelineActivate *wgpu.ComputePipeline
	bgActivate       *wgpu.BindGroup

	// Stage 3: down projection
	pipelineDown *wgpu.ComputePipeline
	bglDown      *wgpu.BindGroupLayout
	bgDown       *wgpu.BindGroup

	// Buffers
	InputBuffer   *wgpu.Buffer
	OutputBuffer  *wgpu.Buffer
	StagingBuffer *wgpu.Buffer

	GatePackedBuf *wgpu.Buffer
	GateScaleBuf  *wgpu.Buffer
	UpPackedBuf   *wgpu.Buffer
	UpScaleBuf    *wgpu.Buffer
	DownPackedBuf *wgpu.Buffer
	DownScaleBuf  *wgpu.Buffer

	GateOutBuf *wgpu.Buffer // intermediate [batch * intermediateSize]
	UpOutBuf   *wgpu.Buffer // intermediate [batch * intermediateSize]
	InterBuf   *wgpu.Buffer // after activation [batch * intermediateSize]

	InputAliased bool
}

// ─────────────────────────────────────────────────────────────────────────────
// GPULayer interface
// ─────────────────────────────────────────────────────────────────────────────

func (l *FP4SwiGLULayer) GetInputBuffer() *wgpu.Buffer         { return l.InputBuffer }
func (l *FP4SwiGLULayer) GetOutputBuffer() *wgpu.Buffer        { return l.OutputBuffer }
func (l *FP4SwiGLULayer) GetStagingBuffer() *wgpu.Buffer       { return l.StagingBuffer }
func (l *FP4SwiGLULayer) GetInputGradientBuffer() *wgpu.Buffer { return nil }

func (l *FP4SwiGLULayer) SetInputBuffer(buf *wgpu.Buffer) {
	l.InputBuffer = buf
	l.InputAliased = true
}

func (l *FP4SwiGLULayer) Cleanup() {
	if l.InputBuffer != nil && !l.InputAliased {
		l.InputBuffer.Destroy()
	}
	for _, b := range []*wgpu.Buffer{
		l.OutputBuffer, l.StagingBuffer,
		l.GatePackedBuf, l.GateScaleBuf,
		l.UpPackedBuf, l.UpScaleBuf,
		l.DownPackedBuf, l.DownScaleBuf,
		l.GateOutBuf, l.UpOutBuf, l.InterBuf,
	} {
		if b != nil {
			b.Destroy()
		}
	}
	for _, p := range []*wgpu.ComputePipeline{l.pipelineGate, l.pipelineUp, l.pipelineActivate, l.pipelineDown} {
		if p != nil {
			p.Release()
		}
	}
	for _, bg := range []*wgpu.BindGroup{l.bgGate, l.bgUp, l.bgActivate, l.bgDown} {
		if bg != nil {
			bg.Release()
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// AllocateBuffers
// ─────────────────────────────────────────────────────────────────────────────

func (l *FP4SwiGLULayer) AllocateBuffers(ctx *Context, label string) error {
	batch := l.BatchSize
	if batch < 1 {
		batch = 1
	}
	var err error

	// Input / Output
	if !l.InputAliased {
		l.InputBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
			Label: label + "_FP4SIn",
			Size:  uint64(batch * l.Spec.InputSize * 4),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
		})

		if err != nil {
			return err
		}
	}
	l.OutputBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: label + "_FP4SOut",
		Size:  uint64(batch * l.Spec.InputSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})

	if err != nil {
		return err
	}
	l.StagingBuffer, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
		Label: label + "_FP4SStg",
		Size:  uint64(batch * l.Spec.InputSize * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})

	if err != nil {
		return err
	}

	// Intermediate buffers
	interSize := uint64(batch * l.Spec.IntermediateSize * 4)
	for i, ptr := range []*(*wgpu.Buffer){&l.GateOutBuf, &l.UpOutBuf, &l.InterBuf} {
		names := []string{"_GateOut", "_UpOut", "_Inter"}
		*ptr, err = ctx.CreateBuffer(&wgpu.BufferDescriptor{
			Label: label + names[i],
			Size:  interSize,
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
		})

		if err != nil {
			return err
		}
	}

	// Packed weight buffers — helper
	allocPacked := func(packed []uint8, lbl string) (*wgpu.Buffer, error) {
		u32 := packBytesToU32(packed)
		buf, err := ctx.CreateBuffer(&wgpu.BufferDescriptor{
			Label: label + lbl,
			Size:  uint64(len(u32) * 4),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})

		return buf, err
	}
	allocScale := func(scales []float32, lbl string) (*wgpu.Buffer, error) {
		return NewFloatBuffer(scales, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	}

	if l.GatePackedBuf, err = allocPacked(l.Spec.GateData, "_GP"); err != nil {
		return err
	}
	if l.GateScaleBuf, err = allocScale(l.Spec.GateScales, "_GS"); err != nil {
		return err
	}
	if l.UpPackedBuf, err = allocPacked(l.Spec.UpData, "_UP"); err != nil {
		return err
	}
	if l.UpScaleBuf, err = allocScale(l.Spec.UpScales, "_US"); err != nil {
		return err
	}
	if l.DownPackedBuf, err = allocPacked(l.Spec.DownData, "_DP"); err != nil {
		return err
	}
	if l.DownScaleBuf, err = allocScale(l.Spec.DownScales, "_DS"); err != nil {
		return err
	}
	return nil
}

func (l *FP4SwiGLULayer) AllocateBackwardBuffers(ctx *Context, label string) error { return nil }

// ─────────────────────────────────────────────────────────────────────────────
// Weight upload
// ─────────────────────────────────────────────────────────────────────────────

func (l *FP4SwiGLULayer) UploadWeights(ctx *Context) {
	upload := func(data []uint8, buf *wgpu.Buffer) {
		if buf == nil || len(data) == 0 {
			return
		}
		ctx.Queue.WriteBuffer(buf, 0, wgpu.ToBytes(packBytesToU32(data)))
	}
	uploadF := func(scales []float32, buf *wgpu.Buffer) {
		if buf == nil || len(scales) == 0 {
			return
		}
		ctx.Queue.WriteBuffer(buf, 0, wgpu.ToBytes(scales))
	}
	upload(l.Spec.GateData, l.GatePackedBuf)
	uploadF(l.Spec.GateScales, l.GateScaleBuf)
	upload(l.Spec.UpData, l.UpPackedBuf)
	uploadF(l.Spec.UpScales, l.UpScaleBuf)
	upload(l.Spec.DownData, l.DownPackedBuf)
	uploadF(l.Spec.DownScales, l.DownScaleBuf)
}

func (l *FP4SwiGLULayer) DownloadWeights(ctx *Context) ([]float32, []float32, error) {
	return nil, nil, nil
}
func (l *FP4SwiGLULayer) DownloadGradients(ctx *Context) ([]float32, []float32, []float32, error) {
	return nil, nil, nil, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Shaders
// ─────────────────────────────────────────────────────────────────────────────

// fp4ProjShader generates an FP4 projection shader (same as FP4DenseLayer's
// shader but with customised constant names).
func fp4ProjShader(nOut, nIn, numRowGroups int) string {
	return fmt.Sprintf(`
const N_OUT : u32 = %du;
const N_IN  : u32 = %du;
const NUM_ROW_GROUPS : u32 = %du;
const MICRO_SCALE_GROUP : u32 = 16u;

@group(0) @binding(0) var<storage, read>      input  : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<storage, read>       packed : array<u32>;
@group(0) @binding(3) var<storage, read>       scales : array<f32>;

fn fp4_mag(m: u32) -> f32 {
    switch(m) {
        case 0u: { return 0.0; }
        case 1u: { return 0.5; }
        case 2u: { return 1.0; }
        case 3u: { return 1.5; }
        case 4u: { return 2.0; }
        case 5u: { return 3.0; }
        case 6u: { return 4.0; }
        default: { return 6.0; }
    }
}

fn get_nibble(k: u32) -> u32 {
    let byte_idx = k >> 1u;
    let word = packed[byte_idx >> 2u];
    let byte_val = (word >> ((byte_idx & 3u) * 8u)) & 0xFFu;
    if (k & 1u) == 0u { return byte_val & 0x0Fu; }
    return (byte_val >> 4u) & 0x0Fu;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&output)) { return; }
    let sample  = idx / N_OUT;
    let out_col = idx %% N_OUT;
    var acc : f32 = 0.0;
    var g : u32 = 0u;
    loop {
        if g >= NUM_ROW_GROUPS { break; }
        let row_start = g * MICRO_SCALE_GROUP;
        var row_end = row_start + MICRO_SCALE_GROUP;
        if row_end > N_IN { row_end = N_IN; }
        let w_scale = scales[out_col * NUM_ROW_GROUPS + g];
        var grp_acc : f32 = 0.0;
        var row : u32 = row_start;
        loop {
            if row >= row_end { break; }
            let nibble = get_nibble(row * N_OUT + out_col);
            let w_sign = (nibble >> 3u) & 1u;
            let w_val  = select(fp4_mag(nibble & 7u), -fp4_mag(nibble & 7u), w_sign == 1u);
            grp_acc += w_val * input[sample * N_IN + row];
            row += 1u;
            continuing { }
        }
        acc += grp_acc * w_scale;
        g += 1u;
        continuing { }
    }
    output[idx] = acc;
}
`, nOut, nIn, numRowGroups)
}

func activateShader(interSize int) string {
	return fmt.Sprintf(`
const INTER_SIZE : u32 = %du;
@group(0) @binding(0) var<storage, read>      gate_out : array<f32>;
@group(0) @binding(1) var<storage, read>      up_out   : array<f32>;
@group(0) @binding(2) var<storage, read_write> inter   : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&inter)) { return; }
    let x = gate_out[idx];
    let silu = x / (1.0 + exp(-x));
    inter[idx] = silu * up_out[idx];
}
`, interSize)
}

// ─────────────────────────────────────────────────────────────────────────────
// Compile
// ─────────────────────────────────────────────────────────────────────────────

func compileFP4Proj(ctx *Context, label string, nOut, nIn, groups int) (*wgpu.ComputePipeline, *wgpu.BindGroupLayout, error) {
	code := fp4ProjShader(nOut, nIn, groups)
	mod, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          label + "_Shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: code},
	})
	if err != nil {
		return nil, nil, fmt.Errorf("%s shader: %w", label, err)
	}
	defer mod.Release()

	bgl, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: label + "_BGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
		},
	})
	if err != nil {
		return nil, nil, fmt.Errorf("%s bgl: %w", label, err)
	}
	pl, err := ctx.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            label + "_PL",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgl},
	})
	if err != nil {
		return nil, nil, fmt.Errorf("%s pl: %w", label, err)
	}
	pipe, err := ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  label,
		Layout: pl,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     mod,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return nil, nil, fmt.Errorf("%s pipeline: %w", label, err)
	}
	return pipe, bgl, nil
}

func (l *FP4SwiGLULayer) Compile(ctx *Context, label string) error {
	var err error

	// Gate projection: input(N_IN) → gateOut(N_OUT=intermediate)
	l.pipelineGate, l.bglGate, err = compileFP4Proj(ctx, label+"_Gate",
		l.Spec.IntermediateSize, l.Spec.InputSize, l.Spec.NumRowGroupsGUp)
	if err != nil {
		return err
	}

	// Up projection: same dimensions
	l.pipelineUp, l.bglUp, err = compileFP4Proj(ctx, label+"_Up",
		l.Spec.IntermediateSize, l.Spec.InputSize, l.Spec.NumRowGroupsGUp)
	if err != nil {
		return err
	}

	// Activate: SiLU(gate) * up → inter
	actCode := activateShader(l.Spec.IntermediateSize)
	actMod, err := ctx.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          label + "_ActShader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: actCode},
	})
	if err != nil {
		return fmt.Errorf("activate shader: %w", err)
	}
	defer actMod.Release()
	actBGL, err := ctx.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: label + "_ActBGL",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
		},
	})
	if err != nil {
		return fmt.Errorf("activate bgl: %w", err)
	}
	actPL, err := ctx.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            label + "_ActPL",
		BindGroupLayouts: []*wgpu.BindGroupLayout{actBGL},
	})
	if err != nil {
		return fmt.Errorf("activate pl: %w", err)
	}
	l.pipelineActivate, err = ctx.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  label + "_ActPipe",
		Layout: actPL,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     actMod,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return fmt.Errorf("activate pipeline: %w", err)
	}

	// Down projection: intermediate → output(input size)
	l.pipelineDown, l.bglDown, err = compileFP4Proj(ctx, label+"_Down",
		l.Spec.InputSize, l.Spec.IntermediateSize, l.Spec.NumRowGroupsDown)
	if err != nil {
		return err
	}

	return nil
}

func (l *FP4SwiGLULayer) CompileBackward(ctx *Context, label string) error { return nil }

// ─────────────────────────────────────────────────────────────────────────────
// Bind groups
// ─────────────────────────────────────────────────────────────────────────────

func (l *FP4SwiGLULayer) CreateBindGroup(ctx *Context, label string) error {
	var err error

	makeBG := func(lbl string, bgl *wgpu.BindGroupLayout, in, out, packed, scale *wgpu.Buffer) (*wgpu.BindGroup, error) {
		return ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  lbl,
			Layout: bgl,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: in, Size: in.GetSize()},
				{Binding: 1, Buffer: out, Size: out.GetSize()},
				{Binding: 2, Buffer: packed, Size: packed.GetSize()},
				{Binding: 3, Buffer: scale, Size: scale.GetSize()},
			},
		})
	}

	l.bgGate, err = makeBG(label+"_GBG", l.bglGate, l.InputBuffer, l.GateOutBuf, l.GatePackedBuf, l.GateScaleBuf)
	if err != nil {
		return fmt.Errorf("gate bg: %w", err)
	}
	l.bgUp, err = makeBG(label+"_UBG", l.bglUp, l.InputBuffer, l.UpOutBuf, l.UpPackedBuf, l.UpScaleBuf)
	if err != nil {
		return fmt.Errorf("up bg: %w", err)
	}

	// Activate bind group
	l.bgActivate, err = ctx.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  label + "_ABG",
		Layout: l.pipelineActivate.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: l.GateOutBuf, Size: l.GateOutBuf.GetSize()},
			{Binding: 1, Buffer: l.UpOutBuf, Size: l.UpOutBuf.GetSize()},
			{Binding: 2, Buffer: l.InterBuf, Size: l.InterBuf.GetSize()},
		},
	})
	if err != nil {
		return fmt.Errorf("activate bg: %w", err)
	}

	// Down bind group
	l.bgDown, err = makeBG(label+"_DBG", l.bglDown, l.InterBuf, l.OutputBuffer, l.DownPackedBuf, l.DownScaleBuf)
	if err != nil {
		return fmt.Errorf("down bg: %w", err)
	}
	return nil
}

func (l *FP4SwiGLULayer) CreateBackwardBindGroup(ctx *Context, label string, dOut *wgpu.Buffer) error {
	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch
// ─────────────────────────────────────────────────────────────────────────────

func (l *FP4SwiGLULayer) Dispatch(pass *wgpu.ComputePassEncoder) {
	batch := l.BatchSize
	if batch < 1 {
		batch = 1
	}
	interTotal := uint32(batch * l.Spec.IntermediateSize)
	inputTotal := uint32(batch * l.Spec.InputSize)
	wg := func(n uint32) uint32 { return (n + 255) / 256 }

	// Gate projection
	pass.SetPipeline(l.pipelineGate)
	pass.SetBindGroup(0, l.bgGate, nil)
	pass.DispatchWorkgroups(wg(interTotal), 1, 1)

	// Up projection
	pass.SetPipeline(l.pipelineUp)
	pass.SetBindGroup(0, l.bgUp, nil)
	pass.DispatchWorkgroups(wg(interTotal), 1, 1)

	// Activate
	pass.SetPipeline(l.pipelineActivate)
	pass.SetBindGroup(0, l.bgActivate, nil)
	pass.DispatchWorkgroups(wg(interTotal), 1, 1)

	// Down
	pass.SetPipeline(l.pipelineDown)
	pass.SetBindGroup(0, l.bgDown, nil)
	pass.DispatchWorkgroups(wg(inputTotal), 1, 1)
}

func (l *FP4SwiGLULayer) DispatchBackward(enc *wgpu.CommandEncoder) {}

func (l *FP4SwiGLULayer) UpdateParams(ctx *Context, seqTokens int, cachePos int) {
	if seqTokens > 0 {
		l.BatchSize = seqTokens
	}
}

func (l *FP4SwiGLULayer) ZeroGradients(ctx *Context) {}
