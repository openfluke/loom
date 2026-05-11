//go:build windows

package poly

import (
	"strconv"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

var (
	kernel32 = syscall.NewLazyDLL("kernel32.dll")
	advapi32 = syscall.NewLazyDLL("advapi32.dll")
	dxgi     = syscall.NewLazyDLL("dxgi.dll")
	ole32    = syscall.NewLazyDLL("ole32.dll")

	procGetTickCount64          = kernel32.NewProc("GetTickCount64")
	procGlobalMemoryStatusEx    = kernel32.NewProc("GlobalMemoryStatusEx")
	procGetDiskFreeSpaceExW     = kernel32.NewProc("GetDiskFreeSpaceExW")
	procGetLogicalDriveStringsW = kernel32.NewProc("GetLogicalDriveStringsW")

	procRegOpenKeyExW    = advapi32.NewProc("RegOpenKeyExW")
	procRegQueryValueExW = advapi32.NewProc("RegQueryValueExW")
	procRegCloseKey      = advapi32.NewProc("RegCloseKey")

	procCreateDXGIFactory1 = dxgi.NewProc("CreateDXGIFactory1") // Use DXGI 1.1+
	procIIDFromString      = ole32.NewProc("IIDFromString")
)

type memoryStatusEx struct {
	cbSize                  uint32
	dwMemoryLoad            uint32
	ullTotalPhys            uint64
	ullAvailPhys            uint64
	ullTotalPageFile        uint64
	ullAvailPageFile        uint64
	ullTotalVirtual         uint64
	ullAvailVirtual         uint64
	ullAvailExtendedVirtual uint64
}

type dxgiAdapterDesc struct {
	Description           [128]uint16
	VendorId              uint32
	DeviceId              uint32
	SubSysId              uint32
	Revision              uint32
	DedicatedVideoMemory  uintptr
	DedicatedSystemMemory uintptr
	SharedSystemMemory    uintptr
	AdapterLuid           int64
}

type guid struct {
	Data1 uint32
	Data2 uint16
	Data3 uint16
	Data4 [8]byte
}

func getUptime() string {
	r, _, _ := procGetTickCount64.Call()
	if r != 0 {
		return time.Duration(int64(r) * int64(time.Millisecond)).String()
	}
	return "Unknown"
}

func getCPUModel() string {
	var hKey syscall.Handle
	keyPath, _ := syscall.UTF16PtrFromString(`HARDWARE\DESCRIPTION\System\CentralProcessor\0`)
	r, _, _ := procRegOpenKeyExW.Call(
		uintptr(0x80000002), // HKEY_LOCAL_MACHINE
		uintptr(unsafe.Pointer(keyPath)),
		0,
		1, // KEY_QUERY_VALUE
		uintptr(unsafe.Pointer(&hKey)),
	)
	if r == 0 {
		defer procRegCloseKey.Call(uintptr(hKey))
		valName, _ := syscall.UTF16PtrFromString("ProcessorNameString")
		var buf [256]uint16
		var bufLen uint32 = uint32(len(buf) * 2)
		r, _, _ = procRegQueryValueExW.Call(
			uintptr(hKey),
			uintptr(unsafe.Pointer(valName)),
			0,
			0,
			uintptr(unsafe.Pointer(&buf[0])),
			uintptr(unsafe.Pointer(&bufLen)),
		)
		if r == 0 {
			return syscall.UTF16ToString(buf[:])
		}
	}
	return "Unknown CPU"
}

func getSystemMemory() (total, free uint64) {
	var ms memoryStatusEx
	ms.cbSize = uint32(unsafe.Sizeof(ms))
	r, _, _ := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&ms)))
	if r != 0 {
		total = ms.ullTotalPhys
		free = ms.ullAvailPhys
	}
	return
}

func getGPUInfo() GPUInfo {
	var info GPUInfo
	info.Model = "Unknown GPU"

	var factory uintptr
	// IID_IDXGIFactory1: {7703019d-b830-48ba-9923-0504100c5c64}
	var iidFactory guid
	iidStr, _ := syscall.UTF16PtrFromString("{7703019d-b830-48ba-9923-0504100c5c64}")
	r, _, _ := procIIDFromString.Call(uintptr(unsafe.Pointer(iidStr)), uintptr(unsafe.Pointer(&iidFactory)))
	if r != 0 {
		return info
	}

	r, _, _ = procCreateDXGIFactory1.Call(uintptr(unsafe.Pointer(&iidFactory)), uintptr(unsafe.Pointer(&factory)))
	if r == 0 && factory != 0 {
		var adapter uintptr
		// EnumAdapters(0, &adapter) - method index 7
		if comCall(factory, 7, 0, uintptr(unsafe.Pointer(&adapter))) == 0 {
			var desc dxgiAdapterDesc
			// GetDesc(&desc) - method index 8
			if comCall(adapter, 8, uintptr(unsafe.Pointer(&desc))) == 0 {
				info.Model = syscall.UTF16ToString(desc.Description[:])
				info.VRAM = uint64(desc.DedicatedVideoMemory)
				info.Vendor = strconv.FormatUint(uint64(desc.VendorId), 16)
			}
			// Release() - method index 2
			comCall(adapter, 2)
		}
		// Release() - method index 2
		comCall(factory, 2)
	}
	return info
}

func comCall(obj uintptr, index int, args ...uintptr) uintptr {
	vtable := *(*uintptr)(unsafe.Pointer(obj))
	method := *(*uintptr)(unsafe.Pointer(vtable + uintptr(index)*unsafe.Sizeof(uintptr(0))))

	switch len(args) {
	case 0:
		r1, _, _ := syscall.Syscall(method, 1, obj, 0, 0)
		return r1
	case 1:
		r1, _, _ := syscall.Syscall(method, 2, obj, args[0], 0)
		return r1
	case 2:
		r1, _, _ := syscall.Syscall(method, 3, obj, args[0], args[1])
		return r1
	default:
		return 1 // Not implemented for more args
	}
}

func getDiskUsage() []DiskInfo {
	var disks []DiskInfo
	var buf [256]uint16
	r, _, _ := procGetLogicalDriveStringsW.Call(uintptr(len(buf)), uintptr(unsafe.Pointer(&buf[0])))
	if r != 0 {
		s := syscall.UTF16ToString(buf[:r])
		for _, drive := range strings.Split(s, "\x00") {
			if drive == "" {
				continue
			}
			drivePtr, _ := syscall.UTF16PtrFromString(drive)
			var free, total, totalFree uint64
			r, _, _ := procGetDiskFreeSpaceExW.Call(
				uintptr(unsafe.Pointer(drivePtr)),
				uintptr(unsafe.Pointer(&free)),
				uintptr(unsafe.Pointer(&total)),
				uintptr(unsafe.Pointer(&totalFree)),
			)
			if r != 0 {
				disks = append(disks, DiskInfo{Path: drive, Total: total, Free: free})
			}
		}
	}
	return disks
}
