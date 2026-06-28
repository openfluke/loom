package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/base64"
	"encoding/json"
	"unsafe"

	"github.com/openfluke/loom/poly"
)

func entityBytes(buf *C.char, length C.int) []byte {
	if length < 0 {
		return []byte(C.GoString(buf))
	}
	return C.GoBytes(unsafe.Pointer(buf), length)
}

func entityLoadOptionsFromJSON(jsonStr string) *poly.EntityLoadOptions {
	if jsonStr == "" || jsonStr == "null" {
		return nil
	}
	var indices []int
	if err := json.Unmarshal([]byte(jsonStr), &indices); err != nil || len(indices) == 0 {
		return nil
	}
	return &poly.EntityLoadOptions{LayerIndices: indices}
}

func buildTransformerFromEntity(et *poly.EntityTransformer, dt poly.DType, template poly.Template) (interface{}, poly.DType) {
	poly.PrepareEntityTransformerInference(et)
	if dt == 0 {
		dt = poly.DTypeFloat32
	}
	switch dt {
	case poly.DTypeFloat64:
		return poly.BuildTransformerFromEntity[float64](et, template), dt
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		return poly.BuildTransformerFromEntity[float32](et, template), poly.DTypeFloat32
	case poly.DTypeInt64:
		return poly.BuildTransformerFromEntity[int64](et, template), dt
	case poly.DTypeInt32:
		return poly.BuildTransformerFromEntity[int32](et, template), dt
	case poly.DTypeInt16:
		return poly.BuildTransformerFromEntity[int16](et, template), dt
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		return poly.BuildTransformerFromEntity[int8](et, template), poly.DTypeInt8
	case poly.DTypeUint64:
		return poly.BuildTransformerFromEntity[uint64](et, template), dt
	case poly.DTypeUint32:
		return poly.BuildTransformerFromEntity[uint32](et, template), dt
	case poly.DTypeUint16:
		return poly.BuildTransformerFromEntity[uint16](et, template), dt
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		return poly.BuildTransformerFromEntity[uint8](et, template), poly.DTypeUint8
	default:
		return poly.BuildTransformerFromEntity[float32](et, template), poly.DTypeFloat32
	}
}

//export LoomLoadEntity
func LoomLoadEntity(path *C.char) C.longlong {
	n, err := poly.LoadEntity(C.GoString(path))
	if err != nil {
		return -1
	}
	networkMu.Lock()
	id := networkNextID
	networkNextID++
	networks[id] = n
	networkMu.Unlock()
	return C.longlong(id)
}

//export LoomLoadEntityWithOptions
func LoomLoadEntityWithOptions(path *C.char, layerIndicesJSON *C.char) C.longlong {
	n, err := poly.LoadEntityWithOptions(C.GoString(path), entityLoadOptionsFromJSON(C.GoString(layerIndicesJSON)))
	if err != nil {
		return -1
	}
	networkMu.Lock()
	id := networkNextID
	networkNextID++
	networks[id] = n
	networkMu.Unlock()
	return C.longlong(id)
}

func registerEntityNetwork(n *poly.VolumetricNetwork) C.longlong {
	networkMu.Lock()
	id := networkNextID
	networkNextID++
	networks[id] = n
	networkMu.Unlock()
	return C.longlong(id)
}

//export LoomSerializeEntity
func LoomSerializeEntity(networkHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	data, err := poly.SerializeEntity(n)
	if err != nil {
		return errJSON(err.Error())
	}
	out, _ := json.Marshal(map[string]string{
		"b64": base64.StdEncoding.EncodeToString(data),
	})
	return C.CString(string(out))
}

//export LoomDeserializeEntity
func LoomDeserializeEntity(dataBuf *C.char, length C.int) C.longlong {
	n, err := poly.DeserializeEntity(entityBytes(dataBuf, length))
	if err != nil {
		return -1
	}
	return registerEntityNetwork(n)
}

//export LoomDeserializeEntityLayer
func LoomDeserializeEntityLayer(dataBuf *C.char, length C.int, layerIndex C.int) C.longlong {
	n, err := poly.DeserializeEntityLayer(entityBytes(dataBuf, length), int(layerIndex))
	if err != nil {
		return -1
	}
	return registerEntityNetwork(n)
}

//export LoomLayerPersistenceFromEntity
func LoomLayerPersistenceFromEntity(dataBuf *C.char, length C.int, layerIndex C.int) *C.char {
	raw, scale, native, err := poly.LayerPersistenceFromEntity(entityBytes(dataBuf, length), int(layerIndex))
	if err != nil {
		return errJSON(err.Error())
	}
	out, _ := json.Marshal(map[string]interface{}{
		"weights": base64.StdEncoding.EncodeToString(raw),
		"scale":   scale,
		"native":  native,
	})
	return C.CString(string(out))
}

//export LoomEntityGPUWeightDType
func LoomEntityGPUWeightDType(storedDType C.int, useGPU C.int) C.int {
	return C.int(poly.EntityGPUWeightDType(poly.DType(storedDType), int(useGPU) != 0))
}

//export LoomPackQ4_0GPU
func LoomPackQ4_0GPU(weightsJSON *C.char) *C.char {
	var weights []float32
	if err := json.Unmarshal([]byte(C.GoString(weightsJSON)), &weights); err != nil {
		return errJSON("invalid weights JSON")
	}
	scales, packed := poly.PackQ4_0GPU(weights)
	out, _ := json.Marshal(map[string]interface{}{"scales": scales, "packed": packed})
	return C.CString(string(out))
}

//export LoomLoadEntityTransformer
func LoomLoadEntityTransformer(path *C.char) C.longlong {
	et, err := poly.LoadEntityTransformer(C.GoString(path))
	if err != nil {
		return -1
	}
	networkMu.Lock()
	id := entityTransformerNextID
	entityTransformerNextID++
	entityTransformers[id] = et
	networkMu.Unlock()
	return C.longlong(id)
}

//export LoomBuildTransformerFromEntity
func LoomBuildTransformerFromEntity(entityHandle C.longlong, dtype C.int) C.longlong {
	et, ok := getEntityTransformer(int64(entityHandle))
	if !ok {
		return -1
	}
	tr, dt := buildTransformerFromEntity(et, poly.DType(dtype), poly.Template{})
	return C.longlong(registerTransformer(tr, dt))
}

//export LoomFreeEntityTransformer
func LoomFreeEntityTransformer(entityHandle C.longlong) {
	networkMu.Lock()
	delete(entityTransformers, int64(entityHandle))
	networkMu.Unlock()
}

//export LoomLoadEntityTransformerAs
func LoomLoadEntityTransformerAs(path *C.char, dtype C.int) C.longlong {
	et, err := poly.LoadEntityTransformer(C.GoString(path))
	if err != nil {
		return -1
	}
	tr, dt := buildTransformerFromEntity(et, poly.DType(dtype), poly.Template{})
	return C.longlong(registerTransformer(tr, dt))
}

func registerEntityFile(ef *poly.EntityFile) C.longlong {
	networkMu.Lock()
	id := entityFileNextID
	entityFileNextID++
	entityFiles[id] = ef
	networkMu.Unlock()
	return C.longlong(id)
}

func registerEntityTransformerHandle(et *poly.EntityTransformer) C.longlong {
	networkMu.Lock()
	id := entityTransformerNextID
	entityTransformerNextID++
	entityTransformers[id] = et
	networkMu.Unlock()
	return C.longlong(id)
}

//export LoomOpenEntityFile
func LoomOpenEntityFile(path *C.char) C.longlong {
	ef, err := poly.OpenEntityFile(C.GoString(path))
	if err != nil {
		return -1
	}
	return registerEntityFile(ef)
}

//export LoomOpenEntityFileAt
func LoomOpenEntityFileAt(path *C.char, baseOffset C.longlong, maxLoomEnd C.longlong) C.longlong {
	ef, err := poly.OpenEntityFileAt(C.GoString(path), int64(baseOffset), int64(maxLoomEnd))
	if err != nil {
		return -1
	}
	return registerEntityFile(ef)
}

//export LoomCloseEntityFile
func LoomCloseEntityFile(entityFileHandle C.longlong) {
	networkMu.Lock()
	if ef, ok := entityFiles[int64(entityFileHandle)]; ok {
		_ = ef.Close()
		delete(entityFiles, int64(entityFileHandle))
	}
	networkMu.Unlock()
}

//export LoomLoadEntityTransformerFromFile
func LoomLoadEntityTransformerFromFile(path *C.char) C.longlong {
	et, err := poly.LoadEntityTransformerFromFile(C.GoString(path))
	if err != nil {
		return -1
	}
	return registerEntityTransformerHandle(et)
}

//export LoomLoadEntityTransformerFromFileAt
func LoomLoadEntityTransformerFromFileAt(path *C.char, baseOffset C.longlong, maxLoomEnd C.longlong) C.longlong {
	et, err := poly.LoadEntityTransformerFromFileAt(C.GoString(path), int64(baseOffset), int64(maxLoomEnd))
	if err != nil {
		return -1
	}
	return registerEntityTransformerHandle(et)
}

//export LoomLoadEntityTransformerTopology
func LoomLoadEntityTransformerTopology(entityFileHandle C.longlong) C.longlong {
	ef, ok := getEntityFile(int64(entityFileHandle))
	if !ok {
		return -1
	}
	et, err := ef.LoadEntityTransformerTopology()
	if err != nil {
		return -1
	}
	return registerEntityTransformerHandle(et)
}

//export LoomLoadNetworkLayerWeights
func LoomLoadNetworkLayerWeights(entityFileHandle C.longlong, networkHandle C.longlong, layerIndicesJSON *C.char) *C.char {
	ef, ok := getEntityFile(int64(entityFileHandle))
	if !ok {
		return errJSON("invalid entity file handle")
	}
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	var indices []int
	if err := json.Unmarshal([]byte(C.GoString(layerIndicesJSON)), &indices); err != nil {
		return errJSON("invalid layer indices JSON")
	}
	if err := ef.LoadNetworkLayerWeights(n, indices); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status":"ok"}`)
}

//export LoomPrepareEntityTransformerLayerIndices
func LoomPrepareEntityTransformerLayerIndices(entityHandle C.longlong, layerIndicesJSON *C.char) *C.char {
	et, ok := getEntityTransformer(int64(entityHandle))
	if !ok {
		return errJSON("invalid entity transformer handle")
	}
	var indices []int
	if err := json.Unmarshal([]byte(C.GoString(layerIndicesJSON)), &indices); err != nil {
		return errJSON("invalid layer indices JSON")
	}
	poly.PrepareEntityTransformerLayerIndices(et, indices)
	return C.CString(`{"status":"ok"}`)
}

//export LoomDequantizeQ4_0GPUPacked
func LoomDequantizeQ4_0GPUPacked(scalesPackedJSON *C.char) *C.char {
	var in struct {
		Scales []float32 `json:"scales"`
		Packed []uint32  `json:"packed"`
	}
	if err := json.Unmarshal([]byte(C.GoString(scalesPackedJSON)), &in); err != nil {
		return errJSON("invalid scales/packed JSON")
	}
	weights := poly.DequantizeQ4_0GPUPacked(in.Scales, in.Packed)
	out, _ := json.Marshal(weights)
	return C.CString(string(out))
}
