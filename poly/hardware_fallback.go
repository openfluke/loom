//go:build !windows && !linux && !android && !darwin && !ios && !js

package poly

func getUptime() string { return "Unknown" }
func getCPUModel() string { return "Unknown CPU" }
func getSystemMemory() (total, free uint64) { return 0, 0 }
func getGPUInfo() GPUInfo { return GPUInfo{Model: "Unknown GPU"} }
func getDiskUsage() []DiskInfo { return nil }
