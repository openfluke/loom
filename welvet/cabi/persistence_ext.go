package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"
	"unsafe"

	"github.com/openfluke/loom/poly"
)

//export LoomSerializeNetwork
func LoomSerializeNetwork(networkHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	data, err := poly.SerializeNetwork(n)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(data))
}

//export LoomDeserializeNetwork
func LoomDeserializeNetwork(jsonBuf *C.char, length C.int) C.longlong {
	var slice []byte
	if length < 0 {
		slice = []byte(C.GoString(jsonBuf))
	} else {
		slice = C.GoBytes(unsafe.Pointer(jsonBuf), length)
	}
	n, err := poly.DeserializeNetwork(slice)
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

//export LoomLayerPersistenceFromJSON
func LoomLayerPersistenceFromJSON(jsonBuf *C.char, length C.int, layerIndex C.int) *C.char {
	var slice []byte
	if length < 0 {
		slice = []byte(C.GoString(jsonBuf))
	} else {
		slice = C.GoBytes(unsafe.Pointer(jsonBuf), length)
	}
	weights, scale, native, err := poly.LayerPersistenceFromJSON(slice, int(layerIndex))
	if err != nil {
		return errJSON(err.Error())
	}
	out, _ := json.Marshal(map[string]interface{}{
		"weights": weights,
		"scale":   scale,
		"native":  native,
	})
	return C.CString(string(out))
}

//export LoomLayerNativePersistenceSnapshot
func LoomLayerNativePersistenceSnapshot(networkHandle C.longlong, layerIndex C.int, dtype C.int) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}
	if int(layerIndex) < 0 || int(layerIndex) >= len(n.Layers) {
		return errJSON("layer index out of range")
	}
	l := &n.Layers[int(layerIndex)]
	if l.WeightStore == nil {
		return errJSON("layer has no WeightStore")
	}
	weights, scale, okSnap := poly.LayerNativePersistenceSnapshot(l.WeightStore, poly.DType(dtype))
	if !okSnap {
		return errJSON("native persistence snapshot failed")
	}
	out, _ := json.Marshal(map[string]interface{}{
		"weights": weights,
		"scale":   scale,
		"native":  true,
		"dtype":   poly.DType(dtype).String(),
	})
	return C.CString(string(out))
}
