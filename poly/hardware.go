package poly

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"runtime"
	"runtime/debug"
)

// SystemAudit contains comprehensive hardware and OS information.
type SystemAudit struct {
	OS        OSInfo        `json:"os"`
	CPU       CPUInfo       `json:"cpu"`
	RAM       RAMInfo       `json:"ram"`
	GPU       GPUInfo       `json:"gpu"`
	Disk      []DiskInfo    `json:"disk"`
	Network   []NetworkInfo `json:"network"`
	GoRuntime GoRuntimeInfo `json:"go_runtime"`
}

type OSInfo struct {
	Hostname string `json:"hostname"`
	Platform string `json:"platform"` // GOOS
	Arch     string `json:"arch"`     // GOARCH
	Uptime   string `json:"uptime"`
}

type CPUInfo struct {
	Model      string `json:"model"`
	Logical    int    `json:"logical_cores"`
	GOMAXPROCS int    `json:"gomaxprocs"`
}

type RAMInfo struct {
	Total uint64 `json:"total_bytes"`
	Free  uint64 `json:"free_bytes"`
	Used  uint64 `json:"used_bytes"`
}

type GPUInfo struct {
	Model  string `json:"model"`
	VRAM   uint64 `json:"vram_bytes"`
	Vendor string `json:"vendor,omitempty"`
}

type DiskInfo struct {
	Path  string `json:"path"`
	Total uint64 `json:"total_bytes"`
	Free  uint64 `json:"free_bytes"`
}

type NetworkInfo struct {
	Name  string   `json:"name"`
	MAC   string   `json:"mac"`
	IPs   []string `json:"ips"`
	Flags string   `json:"flags"`
	MTU   int      `json:"mtu"`
}

type GoRuntimeInfo struct {
	Version   string           `json:"version"`
	NumGorout int              `json:"num_goroutines"`
	MemAlloc  uint64           `json:"mem_alloc_bytes"`
	BuildInfo *debug.BuildInfo `json:"build_info,omitempty"`
}

// AuditSystem extracts as much information as possible natively without using external commands.
func AuditSystem(netCtx *VolumetricNetwork) *SystemAudit {
	audit := &SystemAudit{}

	// 1. Common OS Info
	audit.OS.Hostname, _ = os.Hostname()
	audit.OS.Platform = runtime.GOOS
	audit.OS.Arch = runtime.GOARCH
	audit.OS.Uptime = getUptime()

	// 2. CPU Info
	audit.CPU.Model = getCPUModel()
	audit.CPU.Logical = runtime.NumCPU()
	audit.CPU.GOMAXPROCS = runtime.GOMAXPROCS(0)

	// 3. RAM Info
	audit.RAM.Total, audit.RAM.Free = getSystemMemory()
	audit.RAM.Used = audit.RAM.Total - audit.RAM.Free

	// 4. GPU Info
	audit.GPU = getGPUInfo()
	// Fallback to WebGPU context if provided and native detection was sparse
	if audit.GPU.Model == "" || audit.GPU.Model == "Unknown GPU" {
		if netCtx != nil && netCtx.GPUContext != nil && netCtx.GPUContext.Adapter != nil {
			info := netCtx.GPUContext.Adapter.GetInfo()
			audit.GPU.Model = fmt.Sprintf("%s (%v)", info.Name, info.BackendType)
		}
	}

	// 5. Disk Info
	audit.Disk = getDiskUsage()

	// 6. Network Info
	audit.Network = getNetworkInfo()

	// 7. Go Runtime Info
	audit.GoRuntime.Version = runtime.Version()
	audit.GoRuntime.NumGorout = runtime.NumGoroutine()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	audit.GoRuntime.MemAlloc = ms.Alloc
	audit.GoRuntime.BuildInfo, _ = debug.ReadBuildInfo()

	return audit
}

// ToJSON returns the audit formatted as a JSON string.
func (a *SystemAudit) ToJSON() string {
	b, err := json.MarshalIndent(a, "", "  ")
	if err != nil {
		return "{ \"error\": \"failed to marshal audit\" }"
	}
	return string(b)
}

// GetDeviceDescription returns a human-readable string summary.
func GetDeviceDescription(netCtx *VolumetricNetwork) string {
	a := AuditSystem(netCtx)
	ramGB := float64(a.RAM.Total) / (1024 * 1024 * 1024)
	return fmt.Sprintf("OS: %s | CPU: %s (%d) | RAM: %.2f GB | GPU: %s (%.1f GB)",
		a.OS.Platform, a.CPU.Model, a.CPU.Logical, ramGB, a.GPU.Model, float64(a.GPU.VRAM)/(1024*1024*1024))
}

func getNetworkInfo() []NetworkInfo {
	var infos []NetworkInfo
	ifaces, err := net.Interfaces()
	if err == nil {
		for _, iface := range ifaces {
			ni := NetworkInfo{
				Name:  iface.Name,
				MAC:   iface.HardwareAddr.String(),
				Flags: iface.Flags.String(),
				MTU:   iface.MTU,
			}
			addrs, _ := iface.Addrs()
			for _, addr := range addrs {
				ni.IPs = append(ni.IPs, addr.String())
			}
			infos = append(infos, ni)
		}
	}
	return infos
}
