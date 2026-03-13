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

//export LoomDenseForwardTiled
func LoomDenseForwardTiled(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64: pre, post = poly.DenseForwardTiled(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: pre, post = poly.DenseForwardTiled(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: pre, post = poly.DenseForwardTiled(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: pre, post = poly.DenseForwardTiled(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: pre, post = poly.DenseForwardTiled(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: pre, post = poly.DenseForwardTiled(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: pre, post = poly.DenseForwardTiled(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: pre, post = poly.DenseForwardTiled(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: pre, post = poly.DenseForwardTiled(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: pre, post = poly.DenseForwardTiled(l, c.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for dense tiled")
	}
	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomCNN1ForwardTiled
func LoomCNN1ForwardTiled(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64: pre, post = poly.CNN1ForwardTiled(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: pre, post = poly.CNN1ForwardTiled(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: pre, post = poly.CNN1ForwardTiled(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: pre, post = poly.CNN1ForwardTiled(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: pre, post = poly.CNN1ForwardTiled(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: pre, post = poly.CNN1ForwardTiled(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: pre, post = poly.CNN1ForwardTiled(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: pre, post = poly.CNN1ForwardTiled(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: pre, post = poly.CNN1ForwardTiled(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: pre, post = poly.CNN1ForwardTiled(l, c.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for cnn1 tiled")
	}
	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomCNN2ForwardTiled
func LoomCNN2ForwardTiled(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64: pre, post = poly.CNN2ForwardTiled(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: pre, post = poly.CNN2ForwardTiled(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: pre, post = poly.CNN2ForwardTiled(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: pre, post = poly.CNN2ForwardTiled(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: pre, post = poly.CNN2ForwardTiled(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: pre, post = poly.CNN2ForwardTiled(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: pre, post = poly.CNN2ForwardTiled(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: pre, post = poly.CNN2ForwardTiled(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: pre, post = poly.CNN2ForwardTiled(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: pre, post = poly.CNN2ForwardTiled(l, c.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for cnn2 tiled")
	}
	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomCNN3ForwardTiled
func LoomCNN3ForwardTiled(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64: pre, post = poly.CNN3ForwardTiled(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: pre, post = poly.CNN3ForwardTiled(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: pre, post = poly.CNN3ForwardTiled(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: pre, post = poly.CNN3ForwardTiled(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: pre, post = poly.CNN3ForwardTiled(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: pre, post = poly.CNN3ForwardTiled(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: pre, post = poly.CNN3ForwardTiled(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: pre, post = poly.CNN3ForwardTiled(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: pre, post = poly.CNN3ForwardTiled(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: pre, post = poly.CNN3ForwardTiled(l, c.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for cnn3 tiled")
	}
	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomRNNForwardTiled
func LoomRNNForwardTiled(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64: pre, post = poly.RNNForwardTiled(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: pre, post = poly.RNNForwardTiled(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: pre, post = poly.RNNForwardTiled(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: pre, post = poly.RNNForwardTiled(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: pre, post = poly.RNNForwardTiled(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: pre, post = poly.RNNForwardTiled(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: pre, post = poly.RNNForwardTiled(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: pre, post = poly.RNNForwardTiled(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: pre, post = poly.RNNForwardTiled(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: pre, post = poly.RNNForwardTiled(l, c.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for rnn tiled")
	}
	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomLSTMForwardTiled
func LoomLSTMForwardTiled(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64: pre, post = poly.LSTMForwardTiled(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: pre, post = poly.LSTMForwardTiled(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: pre, post = poly.LSTMForwardTiled(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: pre, post = poly.LSTMForwardTiled(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: pre, post = poly.LSTMForwardTiled(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: pre, post = poly.LSTMForwardTiled(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: pre, post = poly.LSTMForwardTiled(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: pre, post = poly.LSTMForwardTiled(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: pre, post = poly.LSTMForwardTiled(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: pre, post = poly.LSTMForwardTiled(l, c.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for lstm tiled")
	}
	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomMHAForwardTiled
func LoomMHAForwardTiled(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64: pre, post = poly.MHAForwardTiled(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: pre, post = poly.MHAForwardTiled(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: pre, post = poly.MHAForwardTiled(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: pre, post = poly.MHAForwardTiled(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: pre, post = poly.MHAForwardTiled(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: pre, post = poly.MHAForwardTiled(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: pre, post = poly.MHAForwardTiled(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: pre, post = poly.MHAForwardTiled(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: pre, post = poly.MHAForwardTiled(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: pre, post = poly.MHAForwardTiled(l, c.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for mha tiled")
	}
	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomSwiGLUForwardTiled
func LoomSwiGLUForwardTiled(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64: pre, post = poly.SwiGLUForwardTiled(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: pre, post = poly.SwiGLUForwardTiled(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: pre, post = poly.SwiGLUForwardTiled(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: pre, post = poly.SwiGLUForwardTiled(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: pre, post = poly.SwiGLUForwardTiled(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: pre, post = poly.SwiGLUForwardTiled(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: pre, post = poly.SwiGLUForwardTiled(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: pre, post = poly.SwiGLUForwardTiled(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: pre, post = poly.SwiGLUForwardTiled(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: pre, post = poly.SwiGLUForwardTiled(l, c.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for swiglu tiled")
	}
	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomEmbeddingForwardTiled
func LoomEmbeddingForwardTiled(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64: pre, post = poly.EmbeddingForwardTiled(l, c.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: pre, post = poly.EmbeddingForwardTiled(l, c.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: pre, post = poly.EmbeddingForwardTiled(l, c.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: pre, post = poly.EmbeddingForwardTiled(l, c.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: pre, post = poly.EmbeddingForwardTiled(l, c.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: pre, post = poly.EmbeddingForwardTiled(l, c.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: pre, post = poly.EmbeddingForwardTiled(l, c.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: pre, post = poly.EmbeddingForwardTiled(l, c.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: pre, post = poly.EmbeddingForwardTiled(l, c.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: pre, post = poly.EmbeddingForwardTiled(l, c.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for embedding tiled")
	}
	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomResidualForwardTiled
func LoomResidualForwardTiled(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong, skipHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	c, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	s, ok := getSystolicContainer(int64(skipHandle))
	if !ok { return errJSON("invalid skip handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var pre, post interface{}
	switch c.DType {
	case poly.DTypeFloat64: pre, post = poly.ResidualForwardTiled(l, c.State.(*poly.Tensor[float64]), s.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: pre, post = poly.ResidualForwardTiled(l, c.State.(*poly.Tensor[float32]), s.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: pre, post = poly.ResidualForwardTiled(l, c.State.(*poly.Tensor[int64]), s.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: pre, post = poly.ResidualForwardTiled(l, c.State.(*poly.Tensor[int32]), s.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: pre, post = poly.ResidualForwardTiled(l, c.State.(*poly.Tensor[int16]), s.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: pre, post = poly.ResidualForwardTiled(l, c.State.(*poly.Tensor[int8]), s.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: pre, post = poly.ResidualForwardTiled(l, c.State.(*poly.Tensor[uint64]), s.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: pre, post = poly.ResidualForwardTiled(l, c.State.(*poly.Tensor[uint32]), s.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: pre, post = poly.ResidualForwardTiled(l, c.State.(*poly.Tensor[uint16]), s.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: pre, post = poly.ResidualForwardTiled(l, c.State.(*poly.Tensor[uint8]), s.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for residual tiled")
	}
	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomCNN1BackwardTiled
func LoomCNN1BackwardTiled(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	go_cont, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	in, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	pa, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var gi, gw interface{}
	switch in.DType {
	case poly.DTypeFloat64: gi, gw = poly.CNN1BackwardTiled(l, go_cont.State.(*poly.Tensor[float64]), in.State.(*poly.Tensor[float64]), pa.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: gi, gw = poly.CNN1BackwardTiled(l, go_cont.State.(*poly.Tensor[float32]), in.State.(*poly.Tensor[float32]), pa.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: gi, gw = poly.CNN1BackwardTiled(l, go_cont.State.(*poly.Tensor[int64]), in.State.(*poly.Tensor[int64]), pa.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: gi, gw = poly.CNN1BackwardTiled(l, go_cont.State.(*poly.Tensor[int32]), in.State.(*poly.Tensor[int32]), pa.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: gi, gw = poly.CNN1BackwardTiled(l, go_cont.State.(*poly.Tensor[int16]), in.State.(*poly.Tensor[int16]), pa.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: gi, gw = poly.CNN1BackwardTiled(l, go_cont.State.(*poly.Tensor[int8]), in.State.(*poly.Tensor[int8]), pa.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: gi, gw = poly.CNN1BackwardTiled(l, go_cont.State.(*poly.Tensor[uint64]), in.State.(*poly.Tensor[uint64]), pa.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: gi, gw = poly.CNN1BackwardTiled(l, go_cont.State.(*poly.Tensor[uint32]), in.State.(*poly.Tensor[uint32]), pa.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: gi, gw = poly.CNN1BackwardTiled(l, go_cont.State.(*poly.Tensor[uint16]), in.State.(*poly.Tensor[uint16]), pa.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: gi, gw = poly.CNN1BackwardTiled(l, go_cont.State.(*poly.Tensor[uint8]), in.State.(*poly.Tensor[uint8]), pa.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for cnn1 backward tiled")
	}
	res := map[string]interface{}{"gradInput": gi, "gradWeights": gw}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomCNN2BackwardTiled
func LoomCNN2BackwardTiled(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	go_cont, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	in, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	pa, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var gi, gw interface{}
	switch in.DType {
	case poly.DTypeFloat64: gi, gw = poly.CNN2BackwardTiled(l, go_cont.State.(*poly.Tensor[float64]), in.State.(*poly.Tensor[float64]), pa.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: gi, gw = poly.CNN2BackwardTiled(l, go_cont.State.(*poly.Tensor[float32]), in.State.(*poly.Tensor[float32]), pa.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: gi, gw = poly.CNN2BackwardTiled(l, go_cont.State.(*poly.Tensor[int64]), in.State.(*poly.Tensor[int64]), pa.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: gi, gw = poly.CNN2BackwardTiled(l, go_cont.State.(*poly.Tensor[int32]), in.State.(*poly.Tensor[int32]), pa.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: gi, gw = poly.CNN2BackwardTiled(l, go_cont.State.(*poly.Tensor[int16]), in.State.(*poly.Tensor[int16]), pa.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: gi, gw = poly.CNN2BackwardTiled(l, go_cont.State.(*poly.Tensor[int8]), in.State.(*poly.Tensor[int8]), pa.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: gi, gw = poly.CNN2BackwardTiled(l, go_cont.State.(*poly.Tensor[uint64]), in.State.(*poly.Tensor[uint64]), pa.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: gi, gw = poly.CNN2BackwardTiled(l, go_cont.State.(*poly.Tensor[uint32]), in.State.(*poly.Tensor[uint32]), pa.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: gi, gw = poly.CNN2BackwardTiled(l, go_cont.State.(*poly.Tensor[uint16]), in.State.(*poly.Tensor[uint16]), pa.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: gi, gw = poly.CNN2BackwardTiled(l, go_cont.State.(*poly.Tensor[uint8]), in.State.(*poly.Tensor[uint8]), pa.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for cnn2 backward tiled")
	}
	res := map[string]interface{}{"gradInput": gi, "gradWeights": gw}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomCNN3BackwardTiled
func LoomCNN3BackwardTiled(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	go_cont, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	in, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	pa, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var gi, gw interface{}
	switch in.DType {
	case poly.DTypeFloat64: gi, gw = poly.CNN3BackwardTiled(l, go_cont.State.(*poly.Tensor[float64]), in.State.(*poly.Tensor[float64]), pa.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: gi, gw = poly.CNN3BackwardTiled(l, go_cont.State.(*poly.Tensor[float32]), in.State.(*poly.Tensor[float32]), pa.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: gi, gw = poly.CNN3BackwardTiled(l, go_cont.State.(*poly.Tensor[int64]), in.State.(*poly.Tensor[int64]), pa.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: gi, gw = poly.CNN3BackwardTiled(l, go_cont.State.(*poly.Tensor[int32]), in.State.(*poly.Tensor[int32]), pa.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: gi, gw = poly.CNN3BackwardTiled(l, go_cont.State.(*poly.Tensor[int16]), in.State.(*poly.Tensor[int16]), pa.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: gi, gw = poly.CNN3BackwardTiled(l, go_cont.State.(*poly.Tensor[int8]), in.State.(*poly.Tensor[int8]), pa.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: gi, gw = poly.CNN3BackwardTiled(l, go_cont.State.(*poly.Tensor[uint64]), in.State.(*poly.Tensor[uint64]), pa.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: gi, gw = poly.CNN3BackwardTiled(l, go_cont.State.(*poly.Tensor[uint32]), in.State.(*poly.Tensor[uint32]), pa.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: gi, gw = poly.CNN3BackwardTiled(l, go_cont.State.(*poly.Tensor[uint16]), in.State.(*poly.Tensor[uint16]), pa.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: gi, gw = poly.CNN3BackwardTiled(l, go_cont.State.(*poly.Tensor[uint8]), in.State.(*poly.Tensor[uint8]), pa.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for cnn3 backward tiled")
	}
	res := map[string]interface{}{"gradInput": gi, "gradWeights": gw}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomRNNBackwardTiled
func LoomRNNBackwardTiled(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	go_cont, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	in, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	pa, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var gi, gw interface{}
	switch in.DType {
	case poly.DTypeFloat64: gi, gw = poly.RNNBackwardTiled(l, go_cont.State.(*poly.Tensor[float64]), in.State.(*poly.Tensor[float64]), pa.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: gi, gw = poly.RNNBackwardTiled(l, go_cont.State.(*poly.Tensor[float32]), in.State.(*poly.Tensor[float32]), pa.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: gi, gw = poly.RNNBackwardTiled(l, go_cont.State.(*poly.Tensor[int64]), in.State.(*poly.Tensor[int64]), pa.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: gi, gw = poly.RNNBackwardTiled(l, go_cont.State.(*poly.Tensor[int32]), in.State.(*poly.Tensor[int32]), pa.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: gi, gw = poly.RNNBackwardTiled(l, go_cont.State.(*poly.Tensor[int16]), in.State.(*poly.Tensor[int16]), pa.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: gi, gw = poly.RNNBackwardTiled(l, go_cont.State.(*poly.Tensor[int8]), in.State.(*poly.Tensor[int8]), pa.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: gi, gw = poly.RNNBackwardTiled(l, go_cont.State.(*poly.Tensor[uint64]), in.State.(*poly.Tensor[uint64]), pa.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: gi, gw = poly.RNNBackwardTiled(l, go_cont.State.(*poly.Tensor[uint32]), in.State.(*poly.Tensor[uint32]), pa.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: gi, gw = poly.RNNBackwardTiled(l, go_cont.State.(*poly.Tensor[uint16]), in.State.(*poly.Tensor[uint16]), pa.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: gi, gw = poly.RNNBackwardTiled(l, go_cont.State.(*poly.Tensor[uint8]), in.State.(*poly.Tensor[uint8]), pa.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for rnn backward tiled")
	}
	res := map[string]interface{}{"gradInput": gi, "gradWeights": gw}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomLSTMBackwardTiled
func LoomLSTMBackwardTiled(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	go_cont, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	in, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	pa, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var gi, gw interface{}
	switch in.DType {
	case poly.DTypeFloat64: gi, gw = poly.LSTMBackwardTiled(l, go_cont.State.(*poly.Tensor[float64]), in.State.(*poly.Tensor[float64]), pa.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: gi, gw = poly.LSTMBackwardTiled(l, go_cont.State.(*poly.Tensor[float32]), in.State.(*poly.Tensor[float32]), pa.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: gi, gw = poly.LSTMBackwardTiled(l, go_cont.State.(*poly.Tensor[int64]), in.State.(*poly.Tensor[int64]), pa.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: gi, gw = poly.LSTMBackwardTiled(l, go_cont.State.(*poly.Tensor[int32]), in.State.(*poly.Tensor[int32]), pa.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: gi, gw = poly.LSTMBackwardTiled(l, go_cont.State.(*poly.Tensor[int16]), in.State.(*poly.Tensor[int16]), pa.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: gi, gw = poly.LSTMBackwardTiled(l, go_cont.State.(*poly.Tensor[int8]), in.State.(*poly.Tensor[int8]), pa.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: gi, gw = poly.LSTMBackwardTiled(l, go_cont.State.(*poly.Tensor[uint64]), in.State.(*poly.Tensor[uint64]), pa.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: gi, gw = poly.LSTMBackwardTiled(l, go_cont.State.(*poly.Tensor[uint32]), in.State.(*poly.Tensor[uint32]), pa.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: gi, gw = poly.LSTMBackwardTiled(l, go_cont.State.(*poly.Tensor[uint16]), in.State.(*poly.Tensor[uint16]), pa.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: gi, gw = poly.LSTMBackwardTiled(l, go_cont.State.(*poly.Tensor[uint8]), in.State.(*poly.Tensor[uint8]), pa.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for lstm backward tiled")
	}
	res := map[string]interface{}{"gradInput": gi, "gradWeights": gw}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomParallelBackward
func LoomParallelBackward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	go_cont, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	in, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	pa, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var gi, gw interface{}
	switch in.DType {
	case poly.DTypeFloat64: gi, gw = poly.ParallelBackwardPolymorphic(l, go_cont.State.(*poly.Tensor[float64]), in.State.(*poly.Tensor[float64]), pa.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: gi, gw = poly.ParallelBackwardPolymorphic(l, go_cont.State.(*poly.Tensor[float32]), in.State.(*poly.Tensor[float32]), pa.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: gi, gw = poly.ParallelBackwardPolymorphic(l, go_cont.State.(*poly.Tensor[int64]), in.State.(*poly.Tensor[int64]), pa.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: gi, gw = poly.ParallelBackwardPolymorphic(l, go_cont.State.(*poly.Tensor[int32]), in.State.(*poly.Tensor[int32]), pa.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: gi, gw = poly.ParallelBackwardPolymorphic(l, go_cont.State.(*poly.Tensor[int16]), in.State.(*poly.Tensor[int16]), pa.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: gi, gw = poly.ParallelBackwardPolymorphic(l, go_cont.State.(*poly.Tensor[int8]), in.State.(*poly.Tensor[int8]), pa.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: gi, gw = poly.ParallelBackwardPolymorphic(l, go_cont.State.(*poly.Tensor[uint64]), in.State.(*poly.Tensor[uint64]), pa.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: gi, gw = poly.ParallelBackwardPolymorphic(l, go_cont.State.(*poly.Tensor[uint32]), in.State.(*poly.Tensor[uint32]), pa.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: gi, gw = poly.ParallelBackwardPolymorphic(l, go_cont.State.(*poly.Tensor[uint16]), in.State.(*poly.Tensor[uint16]), pa.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: gi, gw = poly.ParallelBackwardPolymorphic(l, go_cont.State.(*poly.Tensor[uint8]), in.State.(*poly.Tensor[uint8]), pa.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for parallel backward")
	}
	res := map[string]interface{}{"gradInput": gi, "gradWeights": gw}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomSwiGLUBackwardTiled
func LoomSwiGLUBackwardTiled(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	go_cont, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	in, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	pa, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var gi, gw interface{}
	switch in.DType {
	case poly.DTypeFloat64: gi, gw = poly.SwiGLUBackwardTiled(l, go_cont.State.(*poly.Tensor[float64]), in.State.(*poly.Tensor[float64]), pa.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: gi, gw = poly.SwiGLUBackwardTiled(l, go_cont.State.(*poly.Tensor[float32]), in.State.(*poly.Tensor[float32]), pa.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: gi, gw = poly.SwiGLUBackwardTiled(l, go_cont.State.(*poly.Tensor[int64]), in.State.(*poly.Tensor[int64]), pa.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: gi, gw = poly.SwiGLUBackwardTiled(l, go_cont.State.(*poly.Tensor[int32]), in.State.(*poly.Tensor[int32]), pa.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: gi, gw = poly.SwiGLUBackwardTiled(l, go_cont.State.(*poly.Tensor[int16]), in.State.(*poly.Tensor[int16]), pa.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: gi, gw = poly.SwiGLUBackwardTiled(l, go_cont.State.(*poly.Tensor[int8]), in.State.(*poly.Tensor[int8]), pa.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: gi, gw = poly.SwiGLUBackwardTiled(l, go_cont.State.(*poly.Tensor[uint64]), in.State.(*poly.Tensor[uint64]), pa.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: gi, gw = poly.SwiGLUBackwardTiled(l, go_cont.State.(*poly.Tensor[uint32]), in.State.(*poly.Tensor[uint32]), pa.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: gi, gw = poly.SwiGLUBackwardTiled(l, go_cont.State.(*poly.Tensor[uint16]), in.State.(*poly.Tensor[uint16]), pa.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: gi, gw = poly.SwiGLUBackwardTiled(l, go_cont.State.(*poly.Tensor[uint8]), in.State.(*poly.Tensor[uint8]), pa.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for swiglu backward tiled")
	}
	res := map[string]interface{}{"gradInput": gi, "gradWeights": gw}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomEmbeddingBackwardTiled
func LoomEmbeddingBackwardTiled(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	go_cont, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	in, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	pa, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var gi, gw interface{}
	switch in.DType {
	case poly.DTypeFloat64: gi, gw = poly.EmbeddingBackwardTiled(l, go_cont.State.(*poly.Tensor[float64]), in.State.(*poly.Tensor[float64]), pa.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: gi, gw = poly.EmbeddingBackwardTiled(l, go_cont.State.(*poly.Tensor[float32]), in.State.(*poly.Tensor[float32]), pa.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: gi, gw = poly.EmbeddingBackwardTiled(l, go_cont.State.(*poly.Tensor[int64]), in.State.(*poly.Tensor[int64]), pa.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: gi, gw = poly.EmbeddingBackwardTiled(l, go_cont.State.(*poly.Tensor[int32]), in.State.(*poly.Tensor[int32]), pa.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: gi, gw = poly.EmbeddingBackwardTiled(l, go_cont.State.(*poly.Tensor[int16]), in.State.(*poly.Tensor[int16]), pa.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: gi, gw = poly.EmbeddingBackwardTiled(l, go_cont.State.(*poly.Tensor[int8]), in.State.(*poly.Tensor[int8]), pa.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: gi, gw = poly.EmbeddingBackwardTiled(l, go_cont.State.(*poly.Tensor[uint64]), in.State.(*poly.Tensor[uint64]), pa.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: gi, gw = poly.EmbeddingBackwardTiled(l, go_cont.State.(*poly.Tensor[uint32]), in.State.(*poly.Tensor[uint32]), pa.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: gi, gw = poly.EmbeddingBackwardTiled(l, go_cont.State.(*poly.Tensor[uint16]), in.State.(*poly.Tensor[uint16]), pa.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: gi, gw = poly.EmbeddingBackwardTiled(l, go_cont.State.(*poly.Tensor[uint8]), in.State.(*poly.Tensor[uint8]), pa.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for embedding backward tiled")
	}
	res := map[string]interface{}{"gradInput": gi, "gradWeights": gw}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomResidualBackwardTiled
func LoomResidualBackwardTiled(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	go_cont, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	in, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	pa, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }
	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	var gi, gw interface{}
	switch in.DType {
	case poly.DTypeFloat64: gi, gw = poly.ResidualBackwardTiled(l, go_cont.State.(*poly.Tensor[float64]), in.State.(*poly.Tensor[float64]), pa.State.(*poly.Tensor[float64]))
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2: gi, gw = poly.ResidualBackwardTiled(l, go_cont.State.(*poly.Tensor[float32]), in.State.(*poly.Tensor[float32]), pa.State.(*poly.Tensor[float32]))
	case poly.DTypeInt64: gi, gw = poly.ResidualBackwardTiled(l, go_cont.State.(*poly.Tensor[int64]), in.State.(*poly.Tensor[int64]), pa.State.(*poly.Tensor[int64]))
	case poly.DTypeInt32: gi, gw = poly.ResidualBackwardTiled(l, go_cont.State.(*poly.Tensor[int32]), in.State.(*poly.Tensor[int32]), pa.State.(*poly.Tensor[int32]))
	case poly.DTypeInt16: gi, gw = poly.ResidualBackwardTiled(l, go_cont.State.(*poly.Tensor[int16]), in.State.(*poly.Tensor[int16]), pa.State.(*poly.Tensor[int16]))
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary: gi, gw = poly.ResidualBackwardTiled(l, go_cont.State.(*poly.Tensor[int8]), in.State.(*poly.Tensor[int8]), pa.State.(*poly.Tensor[int8]))
	case poly.DTypeUint64: gi, gw = poly.ResidualBackwardTiled(l, go_cont.State.(*poly.Tensor[uint64]), in.State.(*poly.Tensor[uint64]), pa.State.(*poly.Tensor[uint64]))
	case poly.DTypeUint32: gi, gw = poly.ResidualBackwardTiled(l, go_cont.State.(*poly.Tensor[uint32]), in.State.(*poly.Tensor[uint32]), pa.State.(*poly.Tensor[uint32]))
	case poly.DTypeUint16: gi, gw = poly.ResidualBackwardTiled(l, go_cont.State.(*poly.Tensor[uint16]), in.State.(*poly.Tensor[uint16]), pa.State.(*poly.Tensor[uint16]))
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2: gi, gw = poly.ResidualBackwardTiled(l, go_cont.State.(*poly.Tensor[uint8]), in.State.(*poly.Tensor[uint8]), pa.State.(*poly.Tensor[uint8]))
	default: return errJSON("unsupported dtype for residual backward tiled")
	}
	res := map[string]interface{}{"gradInput": gi, "gradWeights": gw}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomInitLayerNormCell
func LoomInitLayerNormCell(networkHandle C.longlong, z C.int, y C.int, x C.int, l C.int, size C.int, dtype C.int) {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return }
	n.InitLayerNormCell(int(z), int(y), int(x), int(l), int(size), poly.DType(dtype))
}

//export LoomDispatchLayer
func LoomDispatchLayer(networkHandle C.longlong, layerIdx C.int, inputHandle C.longlong, skipHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	in, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	
	var skip *poly.Tensor[float32]
	if skipHandle != 0 {
		if s, ok := getSystolicContainer(int64(skipHandle)); ok {
			skip = s.State.(*poly.Tensor[float32])
		}
	}

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	// Simplified: assuming float32 for high-level dispatch if not specified
	pre, post := poly.DispatchLayer(l, in.State.(*poly.Tensor[float32]), skip)
	
	res := map[string]interface{}{"pre": pre, "post": post}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomDispatchLayerBackward
func LoomDispatchLayerBackward(networkHandle C.longlong, layerIdx C.int, gradOutputHandle C.longlong, inputHandle C.longlong, skipHandle C.longlong, preActHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	go_cont, ok := getSystolicContainer(int64(gradOutputHandle))
	if !ok { return errJSON("invalid gradOutput handle") }
	in, ok := getSystolicContainer(int64(inputHandle))
	if !ok { return errJSON("invalid input handle") }
	pa, ok := getSystolicContainer(int64(preActHandle))
	if !ok { return errJSON("invalid preAct handle") }

	var skip *poly.Tensor[float32]
	if skipHandle != 0 {
		if s, ok := getSystolicContainer(int64(skipHandle)); ok {
			skip = s.State.(*poly.Tensor[float32])
		}
	}

	if int(layerIdx) < 0 || int(layerIdx) >= len(n.Layers) { return errJSON("layer index out of range") }
	l := &n.Layers[int(layerIdx)]

	gi, gw := poly.DispatchLayerBackward(l, go_cont.State.(*poly.Tensor[float32]), in.State.(*poly.Tensor[float32]), skip, pa.State.(*poly.Tensor[float32]))
	
	res := map[string]interface{}{"gradInput": gi, "gradWeights": gw}
	data, _ := json.Marshal(res)
	return C.CString(string(data))
}

//export LoomNewVolumetricNetwork
func LoomNewVolumetricNetwork(depth C.int, rows C.int, cols C.int, layersPerCell C.int) C.longlong {
	n := poly.NewVolumetricNetwork(int(depth), int(rows), int(cols), int(layersPerCell))
	
	networkMu.Lock()
	id := networkNextID
	networkNextID++
	networks[id] = n
	networkMu.Unlock()
	
	return C.longlong(id)
}

//export LoomComputeLayerStats
func LoomComputeLayerStats(tensorHandle C.longlong) *C.char {
	c, ok := getSystolicContainer(int64(tensorHandle))
	if !ok { return errJSON("invalid tensor handle") }
	
	var stats poly.LayerStats
	if t, ok := c.State.(*poly.Tensor[float32]); ok {
		stats = poly.ComputeLayerStats(t)
	}
	
	data, _ := json.Marshal(stats)
	return C.CString(string(data))
}
