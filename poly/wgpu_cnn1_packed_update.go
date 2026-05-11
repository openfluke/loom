package poly

import (
	"fmt"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

type WGPUCNN1PackedUpdateParams struct {
	Batch    uint32
	InC      uint32
	InL      uint32
	Filters  uint32
	LR       float32
	ClipVal  float32
	OutL     uint32
	Scale    float32
	Pad1     uint32
	Pad2     uint32
	Pad3     uint32
	Pad4     uint32
	Pad5     uint32
	Pad6     uint32
	Pad7     uint32
	Pad8     uint32
}

func shaderCNN1PackedApplyGradients(dtype DType) (string, error) {
	var weightsPerWord uint32 = 1
	var shiftExpr string
	var maskExpr string
	var decodeFn string
	var encodeFn string

	switch dtype {
	case DTypeFloat32:
		weightsPerWord = 1
		shiftExpr = "0u"
		maskExpr = "0xFFFFFFFFu"
		decodeFn = "fn decodeWeight(code: u32) -> f32 { return bitcast<f32>(code); }"
		encodeFn = "fn encodeWeight(v: f32) -> u32 { return bitcast<u32>(v); }"

	case DTypeInt16, DTypeUint16, DTypeFloat16, DTypeBFloat16:
		weightsPerWord = 2
		shiftExpr = "lane * 16u"
		maskExpr = "0xFFFFu"
		if dtype == DTypeInt16 {
			decodeFn = "fn decodeWeight(code: u32) -> f32 { var q = i32(code & 0xFFFFu); if (q > 32767) { q -= 65536; } return f32(q); }"
			encodeFn = "fn encodeWeight(v: f32) -> u32 { return bitcast<u32>(i32(clamp(round(v), -32768.0, 32767.0))) & 0xFFFFu; }"
		} else if dtype == DTypeUint16 {
			decodeFn = "fn decodeWeight(code: u32) -> f32 { return f32(code & 0xFFFFu); }"
			encodeFn = "fn encodeWeight(v: f32) -> u32 { return u32(clamp(round(v), 0.0, 65535.0)) & 0xFFFFu; }"
		} else if dtype == DTypeFloat16 {
			decodeFn = `
fn decodeWeight(code: u32) -> f32 {
    let sign = (code >> 15u) & 0x1u; let exp = (code >> 10u) & 0x1Fu; let mant = code & 0x3FFu;
    if (exp == 0u && mant == 0u) { return 0.0; }
    if (exp == 31u) { return 0.0; }
    let resExp = exp + 127u - 15u;
    return bitcast<f32>((sign << 31u) | (resExp << 23u) | (mant << 13u));
}
`
			encodeFn = `
fn encodeWeight(v: f32) -> u32 {
    let f = bitcast<u32>(v); let sign = (f >> 31u) & 0x1u; var exp = i32((f >> 23u) & 0xFFu) - 127 + 15;
    if (v == 0.0) { return sign << 15u; }
    if (exp <= 0) { return sign << 15u; }
    if (exp >= 31) { return (sign << 15u) | (0x1Fu << 10u); }
    return (sign << 15u) | (u32(exp) << 10u) | ((f >> 13u) & 0x3FFu);
}
`
		} else { // BF16
			decodeFn = "fn decodeWeight(code: u32) -> f32 { return bitcast<f32>((code & 0xFFFFu) << 16u); }"
			encodeFn = "fn encodeWeight(v: f32) -> u32 { return bitcast<u32>(v) >> 16u; }"
		}

	case DTypeInt8, DTypeUint8, DTypeFP8E4M3, DTypeFP8E5M2:
		weightsPerWord = 4
		shiftExpr = "lane * 8u"
		maskExpr = "0xFFu"
		if dtype == DTypeUint8 || dtype == DTypeInt8 {
			decodeFn = `
fn decodeWeight(code: u32) -> f32 {
    var q = i32(code & 0xFFu);
    if (q > 127) { q -= 256; }
    return f32(q);
}
`
			if dtype == DTypeUint8 {
				decodeFn = "fn decodeWeight(code: u32) -> f32 { return f32(code & 0xFFu); }"
				encodeFn = "fn encodeWeight(v: f32) -> u32 { return u32(clamp(round(v), 0.0, 255.0)) & 0xFFu; }"
			} else {
				encodeFn = "fn encodeWeight(v: f32) -> u32 { return bitcast<u32>(i32(clamp(round(v), -128.0, 127.0))) & 0xFFu; }"
			}
		} else if dtype == DTypeFP8E4M3 {
			decodeFn = `
fn decodeWeight(code: u32) -> f32 {
    let sign = (code >> 7u) & 0x1u; let exp = (code >> 3u) & 0xFu; let mant = code & 0x7u;
    if (exp == 0u && mant == 0u) { return 0.0; }
    let resExp = exp + 127u - 7u;
    return bitcast<f32>((sign << 31u) | (resExp << 23u) | (mant << 20u));
}
`
			encodeFn = `
fn encodeWeight(v: f32) -> u32 {
    let f = bitcast<u32>(v); let sign = (f >> 31u) & 0x1u; var exp = i32((f >> 23u) & 0xFFu) - 127 + 7;
    if (v == 0.0) { return sign << 7u; }
    if (exp <= 0) { return sign << 7u; }
    if (exp >= 15) { return (sign << 7u) | 0x7Fu; }
    return (sign << 7u) | (u32(exp) << 3u) | ((f >> 20u) & 0x7u);
}
`
		} else { // E5M2
			decodeFn = `
fn decodeWeight(code: u32) -> f32 {
    let sign = (code >> 7u) & 0x1u; let exp = (code >> 2u) & 0x1Fu; let mant = code & 0x3u;
    if (exp == 0u && mant == 0u) { return 0.0; }
    if (exp == 31u) { return 0.0; }
    let resExp = exp + 127u - 15u;
    return bitcast<f32>((sign << 31u) | (resExp << 23u) | (mant << 21u));
}
`
			encodeFn = `
fn encodeWeight(v: f32) -> u32 {
    let f = bitcast<u32>(v); let sign = (f >> 31u) & 0x1u; var exp = i32((f >> 23u) & 0xFFu) - 127 + 15;
    if (v == 0.0) { return sign << 7u; }
    if (exp <= 0) { return sign << 7u; }
    if (exp >= 31) { return (sign << 7u) | (0x1Fu << 2u); }
    return (sign << 7u) | (u32(exp) << 2u) | ((f >> 21u) & 0x3u);
}
`
		}

	case DTypeInt4, DTypeUint4, DTypeFP4:
		weightsPerWord = 8
		shiftExpr = "lane * 4u"
		maskExpr = "0xFu"
		if dtype == DTypeUint4 {
			decodeFn = "fn decodeWeight(code: u32) -> f32 { return f32(code & 0xFu); }"
			encodeFn = "fn encodeWeight(v: f32) -> u32 { return u32(clamp(round(v), 0.0, 15.0)) & 0xFu; }"
		} else if dtype == DTypeInt4 {
			decodeFn = "fn decodeWeight(code: u32) -> f32 { var q = i32(code & 0xFu); if (q > 7) { q -= 16; } return f32(q); }"
			encodeFn = "fn encodeWeight(v: f32) -> u32 { return bitcast<u32>(i32(clamp(round(v), -8.0, 7.0))) & 0xFu; }"
		} else { // FP4
			decodeFn = `
fn decodeWeight(code: u32) -> f32 {
    let sign = (code >> 3u) & 1u; let exp = (code >> 1u) & 3u; let mant = code & 1u;
    if (exp == 0u && mant == 0u) { return 0.0; }
    if (exp == 0u) { return (1.0 - 2.0*f32(sign)) * 0.5 * f32(mant); }
    let resExp = exp + 127u - 1u;
    return bitcast<f32>((sign << 31u) | (resExp << 23u) | (mant << 22u));
}
`
			encodeFn = `
fn encodeWeight(v: f32) -> u32 {
    let f = bitcast<u32>(v); let sign = (f >> 31u) & 1u; let exp = i32((f >> 23u) & 0xFFu) - 127 + 1;
    if (v == 0.0) { return sign << 3u; }
    if (exp <= 0) { return (sign << 3u) | u32(clamp(round(abs(v)*2.0), 0.0, 1.0)); }
    if (exp >= 3) { return (sign << 3u) | (3u << 1u) | ((f >> 22u) & 1u); }
    return (sign << 3u) | (u32(exp) << 1u) | ((f >> 22u) & 1u);
}
`
		}

	case DTypeInt2, DTypeUint2, DTypeTernary:
		weightsPerWord = 16
		shiftExpr = "lane * 2u"
		maskExpr = "0x3u"
		if dtype == DTypeUint2 {
			decodeFn = "fn decodeWeight(code: u32) -> f32 { return f32(code & 0x3u); }"
			encodeFn = "fn encodeWeight(v: f32) -> u32 { return u32(clamp(round(v), 0.0, 3.0)) & 0x3u; }"
		} else if dtype == DTypeInt2 {
			decodeFn = "fn decodeWeight(code: u32) -> f32 { var q = i32(code & 0x3u); if (q > 1) { q -= 4; } return f32(q); }"
			encodeFn = "fn encodeWeight(v: f32) -> u32 { return bitcast<u32>(i32(clamp(round(v), -2.0, 1.0))) & 0x3u; }"
		} else { // Ternary
			decodeFn = `
fn decodeWeight(code: u32) -> f32 {
    if (code == 0u) { return -1.0; }
    if (code == 1u) { return 0.0; }
    if (code == 2u) { return 1.0; }
    return 0.0;
}
`
			encodeFn = `
fn encodeWeight(v: f32) -> u32 {
    var q = i32(round(v));
    if (q < -1) { q = -1; }
    if (q > 1) { q = 1; }
    return u32(q + 1);
}
`
		}

	case DTypeBinary:
		weightsPerWord = 32
		shiftExpr = "lane"
		maskExpr = "0x1u"
		decodeFn = "fn decodeWeight(code: u32) -> f32 { if ((code & 1u) == 0u) { return -1.0; } return 1.0; }"
		encodeFn = "fn encodeWeight(v: f32) -> u32 { if (v >= 0.0) { return 1u; } return 0u; }"

	default:
		return "", fmt.Errorf("packed update not implemented for %s", dtype)
	}

	// High-precision float types do NOT use the Quantization Scale
	useScale := true
	if dtype == DTypeFloat32 || dtype == DTypeFloat16 || dtype == DTypeBFloat16 {
		useScale = false
	}

	updateExpr := "updatedVal"
	if useScale {
		updateExpr = "updatedVal / max(params.scale, 1e-8)"
	}

	return fmt.Sprintf(`
struct WGPUCNN1PackedUpdateParams {
    batch:   u32,
    inC:     u32,
    inL:     u32,
    filters: u32,
    lr:      f32,
    clipVal: f32,
    outL:    u32,
    scale:   f32,
    _p1:     u32,
    _p2:     u32,
    _p3:     u32,
    _p4:     u32,
    _p5:     u32,
    _p6:     u32,
    _p7:     u32,
    _p8:     u32,
}
@group(0) @binding(0) var<uniform> params: WGPUCNN1PackedUpdateParams;
@group(0) @binding(1) var<storage, read_write> packedWeights: array<u32>;
@group(0) @binding(2) var<storage, read> gradWeights: array<f32>;
@group(0) @binding(3) var<storage, read_write> masterWeights: array<f32>;

%s
%s

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&masterWeights)) { return; }

    let g = gradWeights[idx];
    var clippedG = g;
    if (params.clipVal > 0.0) {
        clippedG = clamp(g, -params.clipVal, params.clipVal);
    }

    let updatedVal = masterWeights[idx] - params.lr * clippedG;
    masterWeights[idx] = updatedVal;

    let wordIdx = idx / %du;
    let lane    = idx %% %du;
    let newCode = encodeWeight(%s);

    var currentWord = packedWeights[wordIdx];
    let shift = %s;
    currentWord = (currentWord & ~(%s << shift)) | ((newCode & %s) << shift);
    packedWeights[wordIdx] = currentWord;
}
`, decodeFn, encodeFn, weightsPerWord, weightsPerWord, updateExpr, shiftExpr, maskExpr, maskExpr), nil
}

func (c *WGPUContext) DispatchCNN1PackedApplyGradients(
	dtype DType,
	size int,
	lr float32,
	clipVal float32,
	scale float32,
	packedBuf *wgpu.Buffer,
	gradBuf *wgpu.Buffer,
	masterBuf *wgpu.Buffer,
) error {
	shaderSrc, err := shaderCNN1PackedApplyGradients(dtype)
	if err != nil {
		return err
	}
	pipeline, err := c.CreateComputePipeline(shaderSrc)
	if err != nil {
		return err
	}

	p := WGPUCNN1PackedUpdateParams{
		LR:      lr,
		ClipVal: clipVal,
		Scale:   scale,
	}
	pBuf := c.GetUniformBuffer(uint64(unsafe.Sizeof(p)))
	c.Queue.WriteBuffer(pBuf, 0, wgpu.ToBytes([]WGPUCNN1PackedUpdateParams{p}))

	bindGroup, err := c.GetBindGroup(pipeline, pBuf, packedBuf, gradBuf, masterBuf)
	if err != nil {
		return err
	}

	enc, owned, err := ctxEncoder(c)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)

	var weightsPerWord uint32 = 1
	switch dtype {
	case DTypeFloat32:
		weightsPerWord = 1
	case DTypeInt16, DTypeUint16, DTypeFloat16, DTypeBFloat16:
		weightsPerWord = 2
	case DTypeInt8, DTypeUint8, DTypeFP8E4M3, DTypeFP8E5M2:
		weightsPerWord = 4
	case DTypeInt4, DTypeUint4, DTypeFP4:
		weightsPerWord = 8
	case DTypeInt2, DTypeUint2, DTypeTernary:
		weightsPerWord = 16
	case DTypeBinary:
		weightsPerWord = 32
	}
	wordCount := (uint32(size) + weightsPerWord - 1) / weightsPerWord
	pass.DispatchWorkgroups((wordCount+63)/64, 1, 1)
	pass.End()
	ctxSubmit(c, enc, owned)
	return nil
}
