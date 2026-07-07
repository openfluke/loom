package poly

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/openfluke/loom/poly/accel"
)

// DiscoverAccel loads vendor plugins (Intel OpenVINO CABI).
func DiscoverAccel(cfg accel.AccelConfig) (*accel.Registry, error) {
	return accel.Discover(cfg)
}

// ReleaseAccel frees compiled graphs and closes plugins.
func (n *VolumetricNetwork) ReleaseAccel() {
	if n == nil {
		return
	}
	for i := range n.Layers {
		n.Layers[i].AccelBinding = nil
	}
	if n.Accel != nil {
		n.Accel.Close()
		n.Accel = nil
	}
}

// SyncToAccel compiles offloaded layers once and uploads weights from WeightStore.
func (n *VolumetricNetwork) SyncToAccel(sizeLabel string) error {
	if n == nil || n.Accel == nil {
		return nil
	}
	for i := range n.Layers {
		if err := syncLayerToAccel(&n.Layers[i], n.Accel, sizeLabel); err != nil {
			return err
		}
	}
	return nil
}

func syncLayerToAccel(l *VolumetricLayer, reg *accel.Registry, sizeLabel string) error {
	if l == nil || !l.ExecTarget.UseAccel() {
		return nil
	}
	if l.AccelBinding != nil {
		l.AccelBinding.Release()
		l.AccelBinding = nil
	}
	desc, ok := intelLayerDesc(l, sizeLabel)
	if !ok {
		return fmt.Errorf("layer z=%d y=%d x=%d type=%v: no Intel mapping", l.Z, l.Y, l.X, l.Type)
	}
	plug := reg.PluginFor(l.ExecTarget)
	if plug == nil {
		return fmt.Errorf("no plugin for %s", l.ExecTarget)
	}
	compiled, err := plug.CompileLayer(desc, LayerWeightBytesForAccel(l))
	if err != nil {
		return err
	}
	l.AccelBinding = &accel.LayerBinding{
		Target:       l.ExecTarget,
		Desc:         desc,
		Compiled:     compiled.Layer,
		CompileMs:    compiled.CompileMs,
		FirstInferMs: compiled.FirstInferMs,
	}
	return nil
}

func intelLayerDesc(l *VolumetricLayer, sizeLabel string) (accel.LayerDesc, bool) {
	dtype, ok := intelDTypeLabel(l.DType)
	if !ok {
		return accel.LayerDesc{}, false
	}
	name, ok := intelBenchLayerName(l)
	if !ok {
		return accel.LayerDesc{}, false
	}
	return accel.LayerDesc{LayerName: name, DType: dtype, SizeLabel: sizeLabel}, true
}

func intelDTypeLabel(dt DType) (string, bool) {
	switch dt {
	case DTypeFloat32:
		return "FP32", true
	case DTypeFloat16:
		return "FP16", true
	case DTypeInt16:
		return "INT16", true
	case DTypeInt8:
		return "INT8", true
	case DTypeInt4:
		return "INT4", true
	default:
		return "", false
	}
}

func intelBenchLayerName(l *VolumetricLayer) (string, bool) {
	switch l.Type {
	case LayerDense:
		switch l.Activation {
		case ActivationReLU:
			return "ReLU", true
		case ActivationGELU:
			return "GELU", true
		case ActivationSigmoid:
			return "Sigmoid", true
		case ActivationLinear:
			return "MatMul", true
		default:
			return "MatMul", true
		}
	case LayerCNN1:
		return "Conv1D", true
	case LayerCNN2:
		return "Conv2D", true
	case LayerSoftmax:
		return "Softmax", true
	case LayerLayerNorm:
		return "LayerNorm", true
	case LayerRMSNorm:
		return "RMSNorm", true
	case LayerMultiHeadAttention:
		return "MHA-MatMul", true
	default:
		return "", false
	}
}

func LayerWeightBytesForAccel(l *VolumetricLayer) []byte {
	if l == nil || l.WeightStore == nil {
		return nil
	}
	dt := l.DType
	if dt != DTypeFloat32 {
		l.WeightStore.Morph(dt)
	}
	switch dt {
	case DTypeFloat32:
		w := l.WeightStore.GetActive(DTypeFloat32)
		f32, ok := w.([]float32)
		if !ok || len(f32) == 0 {
			return nil
		}
		out := make([]byte, len(f32)*4)
		for i, v := range f32 {
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
		}
		return out
	case DTypeFloat16:
		if native := l.WeightStore.GetNative(DTypeFloat16); native != nil {
			if u16, ok := native.([]uint16); ok && len(u16) > 0 {
				out := make([]byte, len(u16)*2)
				for i, v := range u16 {
					binary.LittleEndian.PutUint16(out[i*2:], v)
				}
				return out
			}
		}
		w := l.WeightStore.GetActive(DTypeFloat16)
		f32 := CastWeights[float32](w)
		if len(f32) == 0 {
			return nil
		}
		out := make([]byte, len(f32)*2)
		for i, v := range f32 {
			binary.LittleEndian.PutUint16(out[i*2:], float32ToFloat16Bits(v))
		}
		return out
	case DTypeInt8, DTypeInt16, DTypeInt4:
		// Quantized modes upload FP32 weight values; the accelerator requantizes to
		// the target fixed-point precision (INT8/INT16=8-bit weights, INT4=4-bit).
		w := l.WeightStore.GetActive(dt)
		f32 := CastWeights[float32](w)
		if len(f32) == 0 {
			return nil
		}
		out := make([]byte, len(f32)*4)
		for i, v := range f32 {
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
		}
		return out
	default:
		return nil
	}
}

// DispatchAccelForward runs a layer through a compiled Intel binding.
func DispatchAccelForward[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (*Tensor[T], *Tensor[T], bool) {
	if layer == nil || input == nil || layer.AccelBinding == nil || !layer.ExecTarget.UseAccel() {
		return nil, nil, false
	}
	inBytes, err := tensorToAccelBytes(input, layer.AccelBinding.Desc.DType)
	if err != nil {
		return nil, nil, false
	}
	if uintptr(len(inBytes)) != layer.AccelBinding.InBytes() {
		return nil, nil, false
	}
	outBytes := make([]byte, layer.AccelBinding.OutBytes())
	if _, err := layer.AccelBinding.Infer(inBytes, outBytes); err != nil {
		return nil, nil, false
	}
	out, err := accelBytesToTensor[T](outBytes, layer.AccelBinding.Desc.DType, input.Shape)
	if err != nil {
		return nil, nil, false
	}
	return out, out, true
}

func tensorToAccelBytes[T Numeric](t *Tensor[T], dtypeLabel string) ([]byte, error) {
	switch dtypeLabel {
	case "INT8":
		return int8TensorToAccelFP32Bytes(t)
	case "FP16":
		return floatTensorToFP16Bytes(t)
	default:
		// FP32 + quantized activations (INT16/INT4) all hand over FP32 values;
		// the accelerator quantizes activations to its target precision.
		return floatTensorToFP32Bytes(t)
	}
}

func floatTensorToFP32Bytes[T Numeric](t *Tensor[T]) ([]byte, error) {
	data := ConvertSlice[T, float32](t.Data)
	out := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out, nil
}

func int8TensorToAccelFP32Bytes[T Numeric](t *Tensor[T]) ([]byte, error) {
	data := ConvertSlice[T, int8](t.Data)
	out := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(float32(v)))
	}
	return out, nil
}

func floatTensorToFP16Bytes[T Numeric](t *Tensor[T]) ([]byte, error) {
	data := ConvertSlice[T, float32](t.Data)
	out := make([]byte, len(data)*2)
	for i, v := range data {
		binary.LittleEndian.PutUint16(out[i*2:], float32ToFloat16Bits(v))
	}
	return out, nil
}

func accelBytesToTensor[T Numeric](b []byte, dtypeLabel string, shape []int) (*Tensor[T], error) {
	switch dtypeLabel {
	case "INT8":
		n := len(b) / 4
		f32 := make([]float32, n)
		for i := 0; i < n; i++ {
			f32[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		}
		i8 := make([]int8, n)
		for i, v := range f32 {
			i8[i] = int8(v)
		}
		return NewTensorFromSlice(ConvertSlice[int8, T](i8), fitShape(shape, n)...), nil
	case "FP16":
		n := len(b) / 2
		f32 := make([]float32, n)
		for i := 0; i < n; i++ {
			f32[i] = float16BitsToFloat32(binary.LittleEndian.Uint16(b[i*2:]))
		}
		return NewTensorFromSlice(ConvertSlice[float32, T](f32), fitShape(shape, n)...), nil
	default:
		n := len(b) / 4
		f32 := make([]float32, n)
		for i := 0; i < n; i++ {
			f32[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		}
		return NewTensorFromSlice(ConvertSlice[float32, T](f32), fitShape(shape, n)...), nil
	}
}

func float32ToFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits >> 23) & 0xff)
	frac := bits & 0x7fffff
	switch exp {
	case 0:
		return sign
	case 0xff:
		return sign | 0x7c00 | uint16(frac>>13)
	default:
		newExp := exp - 127 + 15
		if newExp >= 0x1f {
			return sign | 0x7c00
		}
		if newExp <= 0 {
			return sign
		}
		return sign | uint16(newExp<<10) | uint16(frac>>13)
	}
}

func float16BitsToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := int((h >> 10) & 0x1f)
	frac := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		return math.Float32frombits(sign) // subnormal — good enough for parity tests
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | (frac << 13))
	default:
		return math.Float32frombits(sign | uint32((exp-15+127)<<23) | (frac << 13))
	}
}

func fitShape(shape []int, n int) []int {
	if len(shape) == 0 {
		return []int{n}
	}
	prod := 1
	for _, d := range shape {
		prod *= d
	}
	if prod == n {
		return shape
	}
	return []int{n}
}
