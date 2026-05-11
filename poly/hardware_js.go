//go:build js && wasm

package poly

import (
	"fmt"
	"syscall/js"
)

func getUptime() string {
	// Performance.now() gives time since page load
	perf := js.Global().Get("performance")
	if !perf.IsUndefined() {
		ms := perf.Call("now").Float()
		return fmt.Sprintf("%.2fms (page uptime)", ms)
	}
	return "Unknown (Wasm)"
}

func getCPUModel() string {
	nav := js.Global().Get("navigator")
	if !nav.IsUndefined() {
		cores := nav.Get("hardwareConcurrency")
		if !cores.IsUndefined() {
			return fmt.Sprintf("Generic Wasm CPU (%d cores)", cores.Int())
		}
	}
	return "Wasm Virtual CPU"
}

func getSystemMemory() (total, free uint64) {
	nav := js.Global().Get("navigator")
	if !nav.IsUndefined() {
		mem := nav.Get("deviceMemory") // in GiB
		if !mem.IsUndefined() {
			total = uint64(mem.Float() * 1024 * 1024 * 1024)
			free = total // Browser doesn't expose free RAM easily
		}
	}
	return
}

func getGPUInfo() GPUInfo {
	// WebGPU could be queried here, but requires async requestAdapter
	return GPUInfo{Model: "Browser GPU (WebGPU/WebGL)"}
}

func getDiskUsage() []DiskInfo {
	return []DiskInfo{{Path: "Browser Sandbox (IDB/LocalStorage)"}}
}
