package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"

	"github.com/openfluke/loom/poly"
)

//export LoomDenseForward
func LoomDenseForward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64:
		pre, post = poly.DenseForwardPolymorphic(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		pre, post = poly.DenseForwardPolymorphic(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		pre, post = poly.DenseForwardPolymorphic(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		pre, post = poly.DenseForwardPolymorphic(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		pre, post = poly.DenseForwardPolymorphic(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		pre, post = poly.DenseForwardPolymorphic(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		pre, post = poly.DenseForwardPolymorphic(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		pre, post = poly.DenseForwardPolymorphic(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		pre, post = poly.DenseForwardPolymorphic(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		pre, post = poly.DenseForwardPolymorphic(l, c.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for dense")
	}

	res := map[string]interface{}{
		"pre":  pre,
		"post": post,
	}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomRMSNormForward
func LoomRMSNormForward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64:
		pre, post = poly.RMSNormForwardPolymorphic(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		pre, post = poly.RMSNormForwardPolymorphic(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		pre, post = poly.RMSNormForwardPolymorphic(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		pre, post = poly.RMSNormForwardPolymorphic(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		pre, post = poly.RMSNormForwardPolymorphic(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		pre, post = poly.RMSNormForwardPolymorphic(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		pre, post = poly.RMSNormForwardPolymorphic(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		pre, post = poly.RMSNormForwardPolymorphic(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		pre, post = poly.RMSNormForwardPolymorphic(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		pre, post = poly.RMSNormForwardPolymorphic(l, c.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for rmsnorm")
	}

	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomLayerNormForward
func LoomLayerNormForward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64:
		pre, post = poly.LayerNormForwardPolymorphic(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		pre, post = poly.LayerNormForwardPolymorphic(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		pre, post = poly.LayerNormForwardPolymorphic(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		pre, post = poly.LayerNormForwardPolymorphic(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		pre, post = poly.LayerNormForwardPolymorphic(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		pre, post = poly.LayerNormForwardPolymorphic(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		pre, post = poly.LayerNormForwardPolymorphic(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		pre, post = poly.LayerNormForwardPolymorphic(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		pre, post = poly.LayerNormForwardPolymorphic(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		pre, post = poly.LayerNormForwardPolymorphic(l, c.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for layernorm")
	}

	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomMHAForward
func LoomMHAForward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64:
		pre, post = poly.MHAForwardPolymorphic(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		pre, post = poly.MHAForwardPolymorphic(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		pre, post = poly.MHAForwardPolymorphic(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		pre, post = poly.MHAForwardPolymorphic(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		pre, post = poly.MHAForwardPolymorphic(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		pre, post = poly.MHAForwardPolymorphic(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		pre, post = poly.MHAForwardPolymorphic(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		pre, post = poly.MHAForwardPolymorphic(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		pre, post = poly.MHAForwardPolymorphic(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		pre, post = poly.MHAForwardPolymorphic(l, c.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for mha")
	}

	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomSoftmaxForward
func LoomSoftmaxForward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var out interface{}
	switch c.DType {
	case poly.DTypeFloat64:
		_, post := poly.SoftmaxForwardPolymorphic(l, c.State.(*poly.Tensor[float64]))
		out = post
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		_, post := poly.SoftmaxForwardPolymorphic(l, c.State.(*poly.Tensor[float32]))
		out = post
	case poly.DTypeInt64:
		_, post := poly.SoftmaxForwardPolymorphic(l, c.State.(*poly.Tensor[int64]))
		out = post
	case poly.DTypeInt32:
		_, post := poly.SoftmaxForwardPolymorphic(l, c.State.(*poly.Tensor[int32]))
		out = post
	case poly.DTypeInt16:
		_, post := poly.SoftmaxForwardPolymorphic(l, c.State.(*poly.Tensor[int16]))
		out = post
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		_, post := poly.SoftmaxForwardPolymorphic(l, c.State.(*poly.Tensor[int8]))
		out = post
	case poly.DTypeUint64:
		_, post := poly.SoftmaxForwardPolymorphic(l, c.State.(*poly.Tensor[uint64]))
		out = post
	case poly.DTypeUint32:
		_, post := poly.SoftmaxForwardPolymorphic(l, c.State.(*poly.Tensor[uint32]))
		out = post
	case poly.DTypeUint16:
		_, post := poly.SoftmaxForwardPolymorphic(l, c.State.(*poly.Tensor[uint16]))
		out = post
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		_, post := poly.SoftmaxForwardPolymorphic(l, c.State.(*poly.Tensor[uint8]))
		out = post
	default:
		return errJSON("unsupported dtype for softmax")
	}

	data, _ := json.Marshal(out)
	return C.CString(string(data))
}

//export LoomSwiGLUForward
func LoomSwiGLUForward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64:
		pre, post = poly.SwiGLUForwardPolymorphic(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		pre, post = poly.SwiGLUForwardPolymorphic(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		pre, post = poly.SwiGLUForwardPolymorphic(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		pre, post = poly.SwiGLUForwardPolymorphic(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		pre, post = poly.SwiGLUForwardPolymorphic(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		pre, post = poly.SwiGLUForwardPolymorphic(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		pre, post = poly.SwiGLUForwardPolymorphic(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		pre, post = poly.SwiGLUForwardPolymorphic(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		pre, post = poly.SwiGLUForwardPolymorphic(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		pre, post = poly.SwiGLUForwardPolymorphic(l, c.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for swiglu")
	}

	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomEmbeddingForward
func LoomEmbeddingForward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64:
		pre, post = poly.EmbeddingForwardPolymorphic(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		pre, post = poly.EmbeddingForwardPolymorphic(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		pre, post = poly.EmbeddingForwardPolymorphic(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		pre, post = poly.EmbeddingForwardPolymorphic(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		pre, post = poly.EmbeddingForwardPolymorphic(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		pre, post = poly.EmbeddingForwardPolymorphic(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		pre, post = poly.EmbeddingForwardPolymorphic(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		pre, post = poly.EmbeddingForwardPolymorphic(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		pre, post = poly.EmbeddingForwardPolymorphic(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		pre, post = poly.EmbeddingForwardPolymorphic(l, c.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for embedding")
	}

	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomResidualForward
func LoomResidualForward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong, skipHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	s, ok := getSystolicContainer(int64(skipHandle))
	if !ok { return errJSON("invalid skip handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64:
		pre, post = poly.ResidualForwardPolymorphic(l, c.State.(*poly.Tensor[float64]), s.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		pre, post = poly.ResidualForwardPolymorphic(l, c.State.(*poly.Tensor[float32]), s.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		pre, post = poly.ResidualForwardPolymorphic(l, c.State.(*poly.Tensor[int64]), s.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		pre, post = poly.ResidualForwardPolymorphic(l, c.State.(*poly.Tensor[int32]), s.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		pre, post = poly.ResidualForwardPolymorphic(l, c.State.(*poly.Tensor[int16]), s.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		pre, post = poly.ResidualForwardPolymorphic(l, c.State.(*poly.Tensor[int8]), s.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		pre, post = poly.ResidualForwardPolymorphic(l, c.State.(*poly.Tensor[uint64]), s.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		pre, post = poly.ResidualForwardPolymorphic(l, c.State.(*poly.Tensor[uint32]), s.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		pre, post = poly.ResidualForwardPolymorphic(l, c.State.(*poly.Tensor[uint16]), s.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		pre, post = poly.ResidualForwardPolymorphic(l, c.State.(*poly.Tensor[uint8]), s.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for residual")
	}

	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomKMeansForward
func LoomKMeansForward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64:
		pre, post = poly.KMeansForwardPolymorphic(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		pre, post = poly.KMeansForwardPolymorphic(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		pre, post = poly.KMeansForwardPolymorphic(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		pre, post = poly.KMeansForwardPolymorphic(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		pre, post = poly.KMeansForwardPolymorphic(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		pre, post = poly.KMeansForwardPolymorphic(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		pre, post = poly.KMeansForwardPolymorphic(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		pre, post = poly.KMeansForwardPolymorphic(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		pre, post = poly.KMeansForwardPolymorphic(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		pre, post = poly.KMeansForwardPolymorphic(l, c.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for kmeans")
	}

	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}
//export LoomRNNForward
func LoomRNNForward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64:
		pre, post = poly.RNNForwardPolymorphic(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		pre, post = poly.RNNForwardPolymorphic(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		pre, post = poly.RNNForwardPolymorphic(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		pre, post = poly.RNNForwardPolymorphic(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		pre, post = poly.RNNForwardPolymorphic(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		pre, post = poly.RNNForwardPolymorphic(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		pre, post = poly.RNNForwardPolymorphic(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		pre, post = poly.RNNForwardPolymorphic(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		pre, post = poly.RNNForwardPolymorphic(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		pre, post = poly.RNNForwardPolymorphic(l, c.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for rnn")
	}

	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomLSTMForward
func LoomLSTMForward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64:
		pre, post = poly.LSTMForwardPolymorphic(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		pre, post = poly.LSTMForwardPolymorphic(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		pre, post = poly.LSTMForwardPolymorphic(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		pre, post = poly.LSTMForwardPolymorphic(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		pre, post = poly.LSTMForwardPolymorphic(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		pre, post = poly.LSTMForwardPolymorphic(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		pre, post = poly.LSTMForwardPolymorphic(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		pre, post = poly.LSTMForwardPolymorphic(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		pre, post = poly.LSTMForwardPolymorphic(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		pre, post = poly.LSTMForwardPolymorphic(l, c.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for lstm")
	}

	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomCNN1Forward
func LoomCNN1Forward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64:
		pre, post = poly.CNN1ForwardPolymorphic(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		pre, post = poly.CNN1ForwardPolymorphic(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		pre, post = poly.CNN1ForwardPolymorphic(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		pre, post = poly.CNN1ForwardPolymorphic(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		pre, post = poly.CNN1ForwardPolymorphic(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		pre, post = poly.CNN1ForwardPolymorphic(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		pre, post = poly.CNN1ForwardPolymorphic(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		pre, post = poly.CNN1ForwardPolymorphic(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		pre, post = poly.CNN1ForwardPolymorphic(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		pre, post = poly.CNN1ForwardPolymorphic(l, c.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for cnn1")
	}

	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomCNN2Forward
func LoomCNN2Forward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64:
		pre, post = poly.CNN2ForwardPolymorphic(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		pre, post = poly.CNN2ForwardPolymorphic(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		pre, post = poly.CNN2ForwardPolymorphic(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		pre, post = poly.CNN2ForwardPolymorphic(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		pre, post = poly.CNN2ForwardPolymorphic(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		pre, post = poly.CNN2ForwardPolymorphic(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		pre, post = poly.CNN2ForwardPolymorphic(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		pre, post = poly.CNN2ForwardPolymorphic(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		pre, post = poly.CNN2ForwardPolymorphic(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		pre, post = poly.CNN2ForwardPolymorphic(l, c.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for cnn2")
	}

	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomCNN3Forward
func LoomCNN3Forward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64:
		pre, post = poly.CNN3ForwardPolymorphic(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		pre, post = poly.CNN3ForwardPolymorphic(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		pre, post = poly.CNN3ForwardPolymorphic(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		pre, post = poly.CNN3ForwardPolymorphic(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		pre, post = poly.CNN3ForwardPolymorphic(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		pre, post = poly.CNN3ForwardPolymorphic(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		pre, post = poly.CNN3ForwardPolymorphic(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		pre, post = poly.CNN3ForwardPolymorphic(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		pre, post = poly.CNN3ForwardPolymorphic(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		pre, post = poly.CNN3ForwardPolymorphic(l, c.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for cnn3")
	}

	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomConvTransposed1DForward
func LoomConvTransposed1DForward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64:
		pre, post = poly.ConvTransposed1DForwardPolymorphic(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		pre, post = poly.ConvTransposed1DForwardPolymorphic(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		pre, post = poly.ConvTransposed1DForwardPolymorphic(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		pre, post = poly.ConvTransposed1DForwardPolymorphic(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		pre, post = poly.ConvTransposed1DForwardPolymorphic(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		pre, post = poly.ConvTransposed1DForwardPolymorphic(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		pre, post = poly.ConvTransposed1DForwardPolymorphic(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		pre, post = poly.ConvTransposed1DForwardPolymorphic(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		pre, post = poly.ConvTransposed1DForwardPolymorphic(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		pre, post = poly.ConvTransposed1DForwardPolymorphic(l, c.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for convtransposed1d")
	}

	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomConvTransposed2DForward
func LoomConvTransposed2DForward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64:
		pre, post = poly.ConvTransposed2DForwardPolymorphic(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		pre, post = poly.ConvTransposed2DForwardPolymorphic(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		pre, post = poly.ConvTransposed2DForwardPolymorphic(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		pre, post = poly.ConvTransposed2DForwardPolymorphic(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		pre, post = poly.ConvTransposed2DForwardPolymorphic(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		pre, post = poly.ConvTransposed2DForwardPolymorphic(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		pre, post = poly.ConvTransposed2DForwardPolymorphic(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		pre, post = poly.ConvTransposed2DForwardPolymorphic(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		pre, post = poly.ConvTransposed2DForwardPolymorphic(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		pre, post = poly.ConvTransposed2DForwardPolymorphic(l, c.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for convtransposed2d")
	}

	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomConvTransposed3DForward
func LoomConvTransposed3DForward(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64:
		pre, post = poly.ConvTransposed3DForwardPolymorphic(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		pre, post = poly.ConvTransposed3DForwardPolymorphic(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64:
		pre, post = poly.ConvTransposed3DForwardPolymorphic(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32:
		pre, post = poly.ConvTransposed3DForwardPolymorphic(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16:
		pre, post = poly.ConvTransposed3DForwardPolymorphic(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		pre, post = poly.ConvTransposed3DForwardPolymorphic(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64:
		pre, post = poly.ConvTransposed3DForwardPolymorphic(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32:
		pre, post = poly.ConvTransposed3DForwardPolymorphic(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16:
		pre, post = poly.ConvTransposed3DForwardPolymorphic(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		pre, post = poly.ConvTransposed3DForwardPolymorphic(l, c.State.(*poly.Tensor[uint8]))
	default:
		return errJSON("unsupported dtype for convtransposed3d")
	}

	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}
