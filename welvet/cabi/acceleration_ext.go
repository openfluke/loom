package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
)

//export LoomInitWGPU
func LoomInitWGPU(networkHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	
	if err := n.InitWGPU(); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomSyncToGPU
func LoomSyncToGPU(networkHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return errJSON("invalid network handle") }
	
	if err := n.SyncAllToGPU(); err != nil {
		return errJSON(err.Error())
	}
	return C.CString(`{"status": "ok"}`)
}

//export LoomSyncToCPU
func LoomSyncToCPU(networkHandle C.longlong) {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return }
	
	for i := range n.Layers {
		n.Layers[i].SyncToCPU()
	}
	n.UseGPU = false
}
