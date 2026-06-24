package poly

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"
)

// MemorySample is one timed observation during model load or inference setup.
type MemorySample struct {
	ElapsedSec    float64 `json:"elapsed_sec"`
	Label         string  `json:"label"`
	HostWeightsMB float64 `json:"host_weights_mb"`
	GPUWeightsMB  float64 `json:"gpu_weights_mb"`
	GPUKVMB       float64 `json:"gpu_kv_mb"`
	VRAMTotalMB   float64 `json:"vram_total_mb"`
	HeapAllocMB   float64 `json:"heap_alloc_mb"`
	HeapSysMB     float64 `json:"heap_sys_mb"`
	ProcessRSSMB  float64 `json:"process_rss_mb"`
}

// MemoryHistory records memory samples over time for load-path diagnostics.
type MemoryHistory struct {
	mu       sync.Mutex
	session  string
	started  time.Time
	samples  []MemorySample
	finished bool
}

// GlobalMemoryHistory is the process-wide recorder used by Lucy/Loom load paths.
var GlobalMemoryHistory = &MemoryHistory{}

var memoryHistoryRecording *bool

// SetMemoryHistoryRecording toggles load-path sampling for this process.
// Lucy sets this from the interactive GPU load prompt; env LOOM_MEMORY_HISTORY=1 also enables.
func SetMemoryHistoryRecording(enabled bool) {
	memoryHistoryRecording = &enabled
}

// ResetMemoryHistoryRecording clears the runtime override so only env controls recording.
func ResetMemoryHistoryRecording() {
	memoryHistoryRecording = nil
}

// MemoryHistoryEnabled reports whether load-path recording is active.
func MemoryHistoryEnabled() bool {
	if memoryHistoryRecording != nil {
		return *memoryHistoryRecording
	}
	v := strings.TrimSpace(os.Getenv("LOOM_MEMORY_HISTORY"))
	switch strings.ToLower(v) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// BeginSession starts a new recording session. Previous samples are discarded.
func (h *MemoryHistory) BeginSession(name string) {
	if h == nil || !MemoryHistoryEnabled() {
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	h.session = strings.TrimSpace(name)
	if h.session == "" {
		h.session = "load"
	}
	h.started = time.Now()
	h.samples = h.samples[:0]
	h.finished = false
}

func (h *MemoryHistory) elapsedSecLocked() float64 {
	if h.started.IsZero() {
		return 0
	}
	return time.Since(h.started).Seconds()
}

func readRuntimeMemoryMB() (heapAllocMB, heapSysMB, processRSSMB float64) {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	heapAllocMB = float64(ms.HeapAlloc) / (1024 * 1024)
	heapSysMB = float64(ms.Sys) / (1024 * 1024)
	if rss := processRSSBytes(); rss > 0 {
		processRSSMB = float64(rss) / (1024 * 1024)
	}
	return heapAllocMB, heapSysMB, processRSSMB
}

// Record captures a sample from an explicit footprint and runtime stats.
func (h *MemoryHistory) Record(label string, fp MemoryFootprint, vramTotalBytes int64) {
	if h == nil || !MemoryHistoryEnabled() {
		return
	}
	heapAllocMB, heapSysMB, processRSSMB := readRuntimeMemoryMB()
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.started.IsZero() {
		h.session = "load"
		h.started = time.Now()
	}
	h.samples = append(h.samples, MemorySample{
		ElapsedSec:    h.elapsedSecLocked(),
		Label:         label,
		HostWeightsMB: fp.HostWeightsMB,
		GPUWeightsMB:  fp.GPUWeightsMB,
		GPUKVMB:       fp.GPUKVMB,
		VRAMTotalMB:   float64(vramTotalBytes) / (1024 * 1024),
		HeapAllocMB:   heapAllocMB,
		HeapSysMB:     heapSysMB,
		ProcessRSSMB:  processRSSMB,
	})
}

// RecordFromTransformer captures Poly-accounted weights plus runtime memory.
func RecordFromTransformer[T Numeric](h *MemoryHistory, tr *Transformer[T], label string) {
	if h == nil || tr == nil {
		return
	}
	fp := NewMemoryFootprintFromTransformer(tr)
	var vram int64
	if tr.Network != nil {
		vram = tr.Network.GetVRAMUsage()
	}
	h.Record(label, fp, vram)
}

// Samples returns a copy of recorded samples.
func (h *MemoryHistory) Samples() []MemorySample {
	if h == nil {
		return nil
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]MemorySample, len(h.samples))
	copy(out, h.samples)
	return out
}

// PrintTerminalChart draws the in-terminal memory timeline graph.
func (h *MemoryHistory) PrintTerminalChart() {
	h.WriteTerminalChart(os.Stdout)
}

// FinishSession prints the terminal chart, sample table, and optional JSON export.
func (h *MemoryHistory) FinishSession() error {
	if h == nil || !MemoryHistoryEnabled() {
		return nil
	}
	h.mu.Lock()
	if len(h.samples) == 0 {
		h.mu.Unlock()
		return nil
	}
	h.finished = true
	h.mu.Unlock()

	h.PrintTerminalChart()
	h.PrintTerminalSummary()
	h.printMemoryLoadDiagnosis()

	if path := strings.TrimSpace(os.Getenv("LOOM_MEMORY_HISTORY_JSON")); path != "" {
		return h.WriteJSON(path)
	}
	return nil
}

// WriteJSON writes samples as JSON for external tooling.
func (h *MemoryHistory) WriteJSON(path string) error {
	samples := h.Samples()
	payload := struct {
		Session string         `json:"session"`
		Samples []MemorySample `json:"samples"`
	}{
		Session: h.sessionName(),
		Samples: samples,
	}
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func (h *MemoryHistory) sessionName() string {
	if h == nil {
		return ""
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.session
}

// PrintTerminalSummary prints the sample table and peak overlap hint.
func (h *MemoryHistory) PrintTerminalSummary() {
	samples := h.Samples()
	if len(samples) == 0 {
		return
	}
	fmt.Println("\n   sample log")
	fmt.Println("   t(s)   host MB  gpu MB  vram MB   rss MB   label")
	for _, s := range samples {
		fmt.Printf("   %5.2f %8.2f %8.2f %8.2f %8.2f  %s\n",
			s.ElapsedSec, s.HostWeightsMB, s.GPUWeightsMB, s.VRAMTotalMB, s.ProcessRSSMB, s.Label)
	}
	if peak := h.peakHostGPUOverlapMB(samples); peak > 0 {
		fmt.Printf("\n   ⚠ peak host+gpu Poly weights overlap: %.2f MB\n", peak)
	}
}

func findMemorySample(samples []MemorySample, label string) (MemorySample, bool) {
	for _, s := range samples {
		if s.Label == label {
			return s, true
		}
	}
	return MemorySample{}, false
}

func (h *MemoryHistory) printMemoryLoadDiagnosis() {
	samples := h.Samples()
	if len(samples) < 2 {
		return
	}

	fmt.Println("\n   diagnosis")

	// Per-block decoder release: compare first block after_sync vs after_release.
	if sync, ok1 := findMemorySample(samples, "block_01_after_sync"); ok1 {
		if rel, ok2 := findMemorySample(samples, "block_01_after_release"); ok2 {
			hostDrop := sync.HostWeightsMB - rel.HostWeightsMB
			if hostDrop > 0.5 {
				fmt.Printf("   ✓ per-block decoder release works (~%.1f MB host freed after block 1 sync)\n", hostDrop)
			} else {
				fmt.Println("   ✗ block_01: host weights not freed after GPU sync (decoder release broken)")
			}
		}
	}

	lastBlockRelease, hasLastBlock := findMemorySample(samples, "block_28_after_release")
	if !hasLastBlock {
		for i := len(samples) - 1; i >= 0; i-- {
			if strings.Contains(samples[i].Label, "_after_release") && strings.HasPrefix(samples[i].Label, "block_") {
				lastBlockRelease = samples[i]
				hasLastBlock = true
				break
			}
		}
	}

	embSync, hasEmbSync := findMemorySample(samples, "embeddings_after_sync")
	embRelease, hasEmbRelease := findMemorySample(samples, "embeddings_after_release")
	// Legacy label from older builds.
	embeddings, hasEmbLegacy := findMemorySample(samples, "embeddings_on_gpu")
	released, hasRel := findMemorySample(samples, "host_weights_released")
	globalsDone, hasGlobalsDone := findMemorySample(samples, "final_norm_after_release")

	if hasLastBlock {
		fmt.Printf("   after last decoder block: host %.0f MB | gpu %.0f MB\n",
			lastBlockRelease.HostWeightsMB, lastBlockRelease.GPUWeightsMB)
	}

	if hasEmbSync && hasEmbRelease {
		overlap := embSync.HostWeightsMB + embSync.GPUWeightsMB
		fmt.Printf("   embeddings sync: host %.0f MB + gpu %.0f MB = %.0f MB peak overlap\n",
			embSync.HostWeightsMB, embSync.GPUWeightsMB, overlap)
		hostDrop := embSync.HostWeightsMB - embRelease.HostWeightsMB
		if hostDrop > 50 {
			fmt.Printf("   ✓ embeddings CPU released after GPU upload (−%.0f MB host)\n", hostDrop)
		} else if embSync.HostWeightsMB > 100 {
			fmt.Println("   ✗ embeddings CPU not released after GPU upload")
		}
	} else if hasLastBlock && hasEmbLegacy {
		fmt.Printf("   at embeddings_on_gpu: host %.0f MB + gpu %.0f MB = %.0f MB Poly overlap\n",
			embeddings.HostWeightsMB, embeddings.GPUWeightsMB, embeddings.HostWeightsMB+embeddings.GPUWeightsMB)
		if embeddings.HostWeightsMB > 100 && embeddings.GPUWeightsMB > lastBlockRelease.GPUWeightsMB+100 {
			fmt.Println("   ⚠ DOUBLING: global GPU upload ran while CPU weights still resident")
			fmt.Printf("      RSS jumped %.0f → %.0f MB at this step\n", lastBlockRelease.ProcessRSSMB, embeddings.ProcessRSSMB)
		}
	}

	if hasGlobalsDone {
		fmt.Printf("   after global sequential upload: host %.0f MB | gpu %.0f MB\n",
			globalsDone.HostWeightsMB, globalsDone.GPUWeightsMB)
	}

	if hasRel {
		fmt.Printf("   after final release: host %.0f MB | gpu %.0f MB (steady state)\n",
			released.HostWeightsMB, released.GPUWeightsMB)
	}
	if hasGlobalsDone && hasRel && hasLastBlock {
		rssDelta := released.ProcessRSSMB - lastBlockRelease.ProcessRSSMB
		if rssDelta > 50 {
			fmt.Printf("   process RSS +%.0f MB vs end of block upload (Go/OS may retain pages until pressure)\n", rssDelta)
		}
	}
}

func (h *MemoryHistory) peakHostGPUOverlapMB(samples []MemorySample) float64 {
	var peak float64
	for _, s := range samples {
		sum := s.HostWeightsMB + s.GPUWeightsMB
		if sum > peak {
			peak = sum
		}
	}
	if len(samples) < 2 {
		return 0
	}
	first := samples[0].HostWeightsMB + samples[0].GPUWeightsMB
	if peak <= first*1.05 {
		return 0
	}
	return peak
}

// FormatMemoryDelta returns a signed delta string in MB for logs.
func FormatMemoryDelta(beforeMB, afterMB float64) string {
	delta := afterMB - beforeMB
	sign := "+"
	if delta < 0 {
		sign = ""
	}
	if math.Abs(delta) < 0.005 {
		return "±0.00 MB"
	}
	return fmt.Sprintf("%s%.2f MB", sign, delta)
}
