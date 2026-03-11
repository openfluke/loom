package gpu

import (
	"fmt"
	"sort"
	"sync"
)

var (
	vramMu     sync.Mutex
	vramUsage  = make(map[string]uint64)
	totalLayer int
	fp4Layer   int
)

// RecordLayerCounts stores the layer counts for the VRAM report.
func RecordLayerCounts(total, fp4 int) {
	vramMu.Lock()
	defer vramMu.Unlock()
	totalLayer = total
	fp4Layer = fp4
}

// TrackVRAM tracks the amount of VRAM allocated for a specific component.
func TrackVRAM(name string, sizeBytes uint64) {
	vramMu.Lock()
	defer vramMu.Unlock()
	vramUsage[name] += sizeBytes
}

// UntrackVRAM removes a specific amount of VRAM from tracking for a component.
func UntrackVRAM(name string, sizeBytes uint64) {
	vramMu.Lock()
	defer vramMu.Unlock()
	if vramUsage[name] >= sizeBytes {
		vramUsage[name] -= sizeBytes
	} else {
		vramUsage[name] = 0
	}
}

// PrintVRAMUsage profiles the top consumers of VRAM.
func PrintVRAMUsage() {
	vramMu.Lock()
	defer vramMu.Unlock()

	type usage struct {
		name string
		size uint64
	}
	var usages []usage
	var total uint64
	for name, size := range vramUsage {
		if size > 0 {
			usages = append(usages, usage{name, size})
			total += size
		}
	}

	sort.Slice(usages, func(i, j int) bool {
		return usages[i].size > usages[j].size
	})

	fmt.Println("\n--- VRAM Usage Profile ---")
	if totalLayer > 0 {
		fmt.Printf("Layers: %d total (%d FP4)\n", totalLayer, fp4Layer)
		fmt.Println("--------------------------")
	}
	for i, u := range usages {
		if i >= 15 {
			break // Only show top 15
		}
		mb := float64(u.size) / 1024 / 1024
		fmt.Printf("%-30s: %8.2f MB\n", u.name, mb)
	}
	fmt.Printf("Total Tracked VRAM: %.2f MB\n", float64(total)/1024/1024)
	fmt.Println("--------------------------")
}
