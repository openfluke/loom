package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import "github.com/openfluke/loom/poly"

//export LoomNetworkTanhiConfigure
func LoomNetworkTanhiConfigure(handle C.longlong, enabled C.int, hostC *C.char, port C.int, sendShape C.int) {
	networkMu.Lock()
	defer networkMu.Unlock()
	n, ok := networks[int64(handle)]
	if !ok {
		return
	}
	host := "127.0.0.1"
	if hostC != nil {
		s := C.GoString(hostC)
		if s != "" {
			host = s
		}
	}
	p := int(port)
	if p < 0 {
		p = 0
	}
	if p == 0 {
		p = poly.DefaultTanhiUDPPort
	}
	n.Tanhi = &poly.TanhiUDPConfig{
		Enabled:   enabled != 0,
		Host:      host,
		Port:      p,
		SendShape: sendShape != 0,
	}
}

//export LoomNetworkTanhiDisable
func LoomNetworkTanhiDisable(handle C.longlong) {
	networkMu.Lock()
	defer networkMu.Unlock()
	n, ok := networks[int64(handle)]
	if !ok {
		return
	}
	n.Tanhi = nil
}

//export LoomTanhiDefaultPort
func LoomTanhiDefaultPort() C.int {
	return C.int(poly.DefaultTanhiUDPPort)
}
