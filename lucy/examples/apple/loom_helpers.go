package apple

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/openfluke/loom/poly"
	"github.com/openfluke/loom/poly/accel"
)

func parseDType(label string) (poly.DType, bool) {
	switch label {
	case "FP32":
		return poly.DTypeFloat32, true
	case "FP16":
		return poly.DTypeFloat16, true
	case "BF16":
		return poly.DTypeBFloat16, true
	case "INT16":
		return poly.DTypeInt16, true
	case "INT8":
		return poly.DTypeInt8, true
	case "INT4":
		return poly.DTypeInt4, true
	default:
		return 0, false
	}
}

func inputKind(layerName string) string {
	switch layerName {
	case "Conv1D":
		return "conv1d"
	case "Conv2D":
		return "conv2d"
	default:
		return "dense"
	}
}

func buildLoomSpec(layer ManifestLayer, p SizeProfile, dt poly.DType) ([]byte, error) {
	if layer.Loom == nil {
		return nil, fmt.Errorf("no loom mapping")
	}
	loomID := *layer.Loom

	switch loomID {
	case "matmul", "mha_matmul", "relu", "gelu", "sigmoid":
		act := "LINEAR"
		switch loomID {
		case "relu":
			act = "RELU"
		case "gelu":
			act = "GELU"
		case "sigmoid":
			act = "SIGMOID"
		}
		return marshalNet(dt, map[string]any{
			"type": "DENSE", "activation": act,
			"input_height": p.Dense.Dim, "output_height": p.Dense.Dim,
		})
	case "softmax":
		return marshalNet(dt, map[string]any{
			"type": "SOFTMAX", "activation": "RELU",
			"input_height": p.Dense.Dim, "output_height": p.Dense.Dim,
			"softmax_type": 1, "softmax_rows": p.Dense.Batch, "softmax_cols": p.Dense.Dim,
		})
	case "layernorm":
		return marshalNet(dt, map[string]any{
			"type": "LAYERNORM", "activation": "RELU",
			"input_height": p.Dense.Dim, "output_height": p.Dense.Dim,
		})
	case "rmsnorm":
		return marshalNet(dt, map[string]any{
			"type": "RMSNORM", "activation": "RELU",
			"input_height": p.Dense.Dim, "output_height": p.Dense.Dim,
		})
	case "conv1d":
		return marshalNet(dt, map[string]any{
			"type": "CNN1", "activation": "RELU",
			"input_channels": p.Conv1D.InC, "filters": p.Conv1D.Filters,
			"input_height": p.Conv1D.Length, "output_height": p.Conv1D.Length,
			"kernel_size": 3, "stride": 1, "padding": 1,
		})
	case "conv2d":
		return marshalNet(dt, map[string]any{
			"type": "CNN2", "activation": "RELU",
			"input_channels": p.Conv2D.InC, "filters": p.Conv2D.Filters,
			"input_height": p.Conv2D.H, "input_width": p.Conv2D.W,
			"output_height": p.Conv2D.H, "output_width": p.Conv2D.W,
			"kernel_size": 3, "stride": 1, "padding": 1,
		})
	default:
		return nil, fmt.Errorf("unsupported loom id %q", loomID)
	}
}

func marshalNet(dt poly.DType, layer map[string]any) ([]byte, error) {
	layer["z"], layer["y"], layer["x"], layer["l"] = 0, 0, 0, 0
	layer["dtype"] = dt.String()
	spec := map[string]any{
		"id": "apple-bridge", "depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": []any{layer},
	}
	return json.Marshal(spec)
}

func forwardOK(net *poly.VolumetricNetwork, inKind string, p SizeProfile, dt poly.DType) bool {
	in := makeLayerInput(inKind, p, dt)
	switch dt {
	case poly.DTypeInt8:
		out, _, _ := poly.ForwardPolymorphic(net, in.(*poly.Tensor[int8]))
		return out != nil && len(out.Data) > 0
	default:
		out, _, _ := poly.ForwardPolymorphic(net, in.(*poly.Tensor[float32]))
		return out != nil && len(out.Data) > 0
	}
}

func makeLayerInput(inKind string, p SizeProfile, dt poly.DType) any {
	switch inKind {
	case "conv1d":
		return newTensor(dt, p.Conv1D.Batch, p.Conv1D.InC, p.Conv1D.Length)
	case "conv2d":
		return newTensor(dt, p.Conv2D.Batch, p.Conv2D.InC, p.Conv2D.H, p.Conv2D.W)
	default:
		return newTensor(dt, p.Dense.Batch, p.Dense.Dim)
	}
}

func newTensor(dt poly.DType, shape ...int) any {
	size := 1
	for _, d := range shape {
		size *= d
	}
	switch dt {
	case poly.DTypeInt8:
		data := make([]int8, size)
		for i := range data {
			data[i] = int8((i % 7) + 1)
		}
		return poly.NewTensorFromSlice(data, shape...)
	default:
		data := make([]float32, size)
		for i := range data {
			data[i] = 0.01 * float32(i%11)
		}
		return poly.NewTensorFromSlice(data, shape...)
	}
}

func resetNet(net *poly.VolumetricNetwork) {
	for i := range net.Layers {
		net.Layers[i].ResetState()
	}
}

func tensorToBytes(in any, dtypeLabel string) ([]byte, error) {
	switch dtypeLabel {
	case "INT8":
		// The Apple plugin computes in fp32; host uploads dequantized FP32 values,
		// matching poly/accel_intel.go's handover contract.
		t := in.(*poly.Tensor[int8])
		out := make([]byte, len(t.Data)*4)
		for i, v := range t.Data {
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(float32(v)))
		}
		return out, nil
	case "FP16":
		t := in.(*poly.Tensor[float32])
		out := make([]byte, len(t.Data)*2)
		for i, v := range t.Data {
			binary.LittleEndian.PutUint16(out[i*2:], float32ToFloat16Bits(v))
		}
		return out, nil
	case "BF16":
		t := in.(*poly.Tensor[float32])
		out := make([]byte, len(t.Data)*2)
		for i, v := range t.Data {
			binary.LittleEndian.PutUint16(out[i*2:], float32ToBFloat16Bits(v))
		}
		return out, nil
	default:
		t := in.(*poly.Tensor[float32])
		out := make([]byte, len(t.Data)*4)
		for i, v := range t.Data {
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
		}
		return out, nil
	}
}

// RunMultiHopDemo: Loom Dense → Apple MatMul (GPU) → Loom ReLU — compile once, many forwards.
func RunMultiHopDemo() {
	fmt.Println("\n--- Multi-hop: Loom Dense → Apple MatMul → Loom ReLU (medium, FP32) ---")

	m, err := LoadManifest()
	if err != nil {
		fmt.Println("manifest:", err)
		return
	}
	profile := m.Sizes["medium"]

	plug, err := openPlugin("GPU")
	if err != nil {
		plug, err = openPlugin("CPU")
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Println("GPU unavailable — using CPU reference backend for middle hop")
	} else {
		fmt.Println("Middle hop on Metal GPU (MPSGraph)")
	}
	defer plug.Close()

	dt := poly.DTypeFloat32
	specPre, _ := marshalNet(dt, map[string]any{
		"type": "DENSE", "activation": "LINEAR",
		"input_height": profile.Dense.Dim, "output_height": profile.Dense.Dim,
	})
	specPost, _ := marshalNet(dt, map[string]any{
		"type": "DENSE", "activation": "RELU",
		"input_height": profile.Dense.Dim, "output_height": profile.Dense.Dim,
	})

	netPre, _ := poly.BuildNetworkFromJSON(specPre)
	netPost, _ := poly.BuildNetworkFromJSON(specPost)
	_ = poly.ConfigureNetworkForMode(netPre, poly.TrainingModeCPUMC)
	_ = poly.ConfigureNetworkForMode(netPost, poly.TrainingModeCPUMC)

	var weights []byte
	if len(netPre.Layers) > 0 {
		weights = poly.LayerWeightBytesForAccel(&netPre.Layers[0])
	}
	desc := accel.LayerDesc{LayerName: "MatMul", DType: "FP32", SizeLabel: "medium"}
	compiled, err := plug.CompileLayer(desc, weights)
	if err != nil {
		fmt.Println("compile:", err)
		return
	}
	defer compiled.Layer.Release()

	fmt.Printf("Init tax (once): compile=%.2f ms  first_infer=%.2f ms\n", compiled.CompileMs, compiled.FirstInferMs)

	in := newTensor(dt, profile.Dense.Batch, profile.Dense.Dim).(*poly.Tensor[float32])
	midBuf, _ := tensorToBytes(in, "FP32")
	outBuf := make([]byte, compiled.OutBytes)
	postIn := poly.NewTensor[float32](profile.Dense.Batch, profile.Dense.Dim)

	const hops = 30
	var loomPreMs, cabiMs, loomPostMs, copyMs float64
	for i := 0; i < hops; i++ {
		resetNet(netPre)
		resetNet(netPost)

		t0 := time.Now()
		preOut, _, _ := poly.ForwardPolymorphic(netPre, in)
		loomPreMs += float64(time.Since(t0).Microseconds()) / 1000.0

		t1 := time.Now()
		copy(midBuf, tensorBytesFP32(preOut))
		copyMs += float64(time.Since(t1).Microseconds()) / 1000.0

		res, err := compiled.Layer.Infer(midBuf, outBuf)
		if err != nil {
			fmt.Println("infer:", err)
			return
		}
		cabiMs += res.InferMs

		for j := range postIn.Data {
			postIn.Data[j] = bytesToFP32(outBuf, j)
		}
		t3 := time.Now()
		_, _, _ = poly.ForwardPolymorphic(netPost, postIn)
		loomPostMs += float64(time.Since(t3).Microseconds()) / 1000.0
	}

	fmt.Printf("Steady per-hop (median of %d): Loom-pre=%.3f ms  copy=%.3f ms  Apple=%.3f ms  Loom-post=%.3f ms\n",
		hops, loomPreMs/float64(hops), copyMs/float64(hops), cabiMs/float64(hops), loomPostMs/float64(hops))
	fmt.Println("No re-compile between hops — init tax paid once.")
}

func tensorBytesFP32(t *poly.Tensor[float32]) []byte {
	b := make([]byte, len(t.Data)*4)
	for i, v := range t.Data {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

func syntheticInput(n int, dtypeLabel string) []byte {
	b := make([]byte, n)
	switch dtypeLabel {
	case "INT8":
		for i := 0; i < n/4; i++ {
			v := float32((i % 7) + 1)
			binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
		}
	case "FP16":
		for i := 0; i < n/2; i++ {
			v := float32(0.01 * float64(i%11))
			binary.LittleEndian.PutUint16(b[i*2:], float32ToFloat16Bits(v))
		}
	case "BF16":
		for i := 0; i < n/2; i++ {
			v := float32(0.01 * float64(i%11))
			binary.LittleEndian.PutUint16(b[i*2:], float32ToBFloat16Bits(v))
		}
	default:
		for i := 0; i < n/4; i++ {
			v := float32(0.01 * float64(i%11))
			binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
		}
	}
	return b
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

func float32ToBFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	if bits&0x7fffffff > 0x7f800000 { // NaN → quiet NaN
		return uint16(bits>>16) | 0x0040
	}
	lsb := (bits >> 16) & 1
	bits += 0x7fff + lsb
	return uint16(bits >> 16)
}

func bytesToFP32(b []byte, idx int) float32 {
	off := idx * 4
	if off+4 > len(b) {
		return 0
	}
	return math.Float32frombits(binary.LittleEndian.Uint32(b[off:]))
}
