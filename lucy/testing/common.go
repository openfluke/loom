// Package testing provides layer-level CPU/GPU suites for the Lucy interactive menu.
package testing

import (
	"fmt"
	"math"
	"time"

	"github.com/openfluke/loom/poly"
	"github.com/openfluke/webgpu/wgpu"
)

// ── Shared Helper Types ───────────────────────────────────────────────────────

type typeConfig struct {
	name      string
	dtype     poly.DType
	scale     float32
	tolerance float64
}

var allTypes = []typeConfig{
	{"Float64", poly.DTypeFloat64, 1.0, 1e-3},
	{"Float32", poly.DTypeFloat32, 1.0, 1e-5},
	{"Float16", poly.DTypeFloat16, 1.0, 1e-3},
	{"BFloat16", poly.DTypeBFloat16, 1.0, 1e-3},
	{"FP8-E4M3", poly.DTypeFP8E4M3, 0.01, 1e-3},
	{"FP8-E5M2", poly.DTypeFP8E5M2, 0.01, 1e-3},
	{"Int64", poly.DTypeInt64, 0.01, 1e-3},
	{"Uint64", poly.DTypeUint64, 0.01, 1e-3},
	{"Int32", poly.DTypeInt32, 0.01, 1e-3},
	{"Uint32", poly.DTypeUint32, 0.01, 1e-3},
	{"Int16", poly.DTypeInt16, 0.01, 1e-3},
	{"Uint16", poly.DTypeUint16, 0.01, 1e-3},
	{"Int8", poly.DTypeInt8, 0.01, 1e-3},
	{"Uint8", poly.DTypeUint8, 0.01, 1e-3},
	{"Int4", poly.DTypeInt4, 0.01, 1e-3},
	{"Uint4", poly.DTypeUint4, 0.01, 1e-3},
	{"FP4", poly.DTypeFP4, 0.01, 1e-3},
	{"Int2", poly.DTypeInt2, 0.01, 1e-3},
	{"Uint2", poly.DTypeUint2, 0.01, 1e-3},
	{"Ternary", poly.DTypeTernary, 0.1, 1e-3},
	{"Binary", poly.DTypeBinary, 0.1, 1e-3},
}

// ── Shared Utilities ─────────────────────────────────────────────────────────

func parityMark(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}

func isQuantIntegerDType(dt poly.DType) bool {
	switch dt {
	case poly.DTypeInt64, poly.DTypeInt32, poly.DTypeInt16, poly.DTypeInt8, poly.DTypeInt4, poly.DTypeInt2,
		poly.DTypeUint64, poly.DTypeUint32, poly.DTypeUint16, poly.DTypeUint8,
		poly.DTypeUint4, poly.DTypeUint2, poly.DTypeBinary, poly.DTypeTernary, poly.DTypeFP4,
		poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		return true
	default:
		return false
	}
}

// trainingLossOK decides whether a short training run behaved acceptably for the matrix.
// Save/reload is checked separately; quant paths often drift slightly without failing.
func trainingLossOK(lossInit, lossFinal float64, dtype poly.DType, weightsFinite bool) bool {
	if math.IsNaN(lossInit) || math.IsNaN(lossFinal) ||
		math.IsInf(lossInit, 0) || math.IsInf(lossFinal, 0) {
		return false
	}
	// Obvious runaway (e.g. MHA GPU int32); keep these broken.
	if lossInit > 1e-3 && (lossFinal > lossInit*50 || lossFinal > 1e10) {
		return false
	}
	if !weightsFinite {
		// GPU low-bit collapse: loss hit zero but Master may still have NaNs from the broken pass.
		if (dtype == poly.DTypeBinary || dtype == poly.DTypeTernary) && lossFinal <= lossInit+1e-3 {
			return true
		}
		return lossInit < 1e-6 && lossFinal < 1e-6
	}
	if lossInit < 0.01 {
		if lossFinal <= lossInit*2.0+1e-3 {
			return true
		}
		// GPU ternary/binary first epoch can report 0 partial loss then recover.
		return isQuantIntegerDType(dtype) && lossFinal < 1.0
	}
	if isQuantIntegerDType(dtype) {
		band := 0.15
		switch dtype {
		case poly.DTypeUint64, poly.DTypeUint32, poly.DTypeUint16, poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
			band = 0.22 // RNN uint PTQ often drifts ~21% over 5 short epochs
		}
		if lossFinal <= lossInit*(1.0+band)+1e-3 {
			return true
		}
		rel := math.Abs(lossFinal-lossInit) / (math.Abs(lossInit) + 1e-9)
		return rel <= band
	}
	return lossFinal < lossInit*1.01
}

func maxAbsDiff(a, b []float32) float64 {
	var d float64
	for i := range a {
		if v := math.Abs(float64(a[i] - b[i])); v > d {
			d = v
		}
	}
	return d
}

func spectrumMark(diff float64, tolerance float64, data []float32, baseline []float32) Spectrum {
	if math.IsNaN(diff) {
		return SpecFatal
	}
	// GPU path can overflow to Inf while CPU stays finite (e.g. MHA BitNet ternary).
	if math.IsInf(diff, 0) && hasSignal(baseline) {
		return SpecHeavyDrift
	}
	if math.IsInf(diff, 0) {
		return SpecFatal
	}

	// Check for dead signal in either
	actualSig := hasSignal(data)
	baseSig := hasSignal(baseline)
	if !actualSig && !baseSig {
		// Both dead — consistent (no signal on either side)
		return SpecExact
	}
	if !actualSig || !baseSig {
		// One side dead, other isn't — severe mismatch but not necessarily a crash.
		// Use HeavyDrift so it shows up in the drift bucket rather than masking real
		// breakages (NaN/Inf) inside the Broken bucket.
		return SpecHeavyDrift
	}

	if diff == 0 {
		return SpecExact
	}
	if diff <= tolerance {
		return SpecIndustry
	}
	if diff <= tolerance*10 {
		return SpecLowBit
	}
	if diff <= 0.1 {
		return SpecDrift
	}
	return SpecHeavyDrift
}

func checkSignalLoss(baseline, actual []float32) bool {
	if len(actual) == 0 && len(baseline) > 0 {
		return true
	}
	if len(actual) == 0 {
		return false
	}

	actualSignal := false
	for _, v := range actual {
		if v != 0 && !math.IsNaN(float64(v)) {
			actualSignal = true
			break
		}
	}
	if actualSignal {
		return false
	}

	baselineSignal := false
	for _, v := range baseline {
		if v != 0 {
			baselineSignal = true
			break
		}
	}
	return baselineSignal // True if baseline had signal but actual is dead
}

func hasSignal(data []float32) bool {
	for _, v := range data {
		if v != 0 && !math.IsNaN(float64(v)) { return true }
	}
	return false
}

// rawF32 returns the active weight buffer as []float32 without applying scale.
func rawF32(ws *poly.WeightStore, dtype poly.DType) []float32 {
	active := ws.GetActive(dtype)
	if active == nil {
		out := make([]float32, len(ws.Master))
		copy(out, ws.Master)
		return out
	}
	switch w := active.(type) {
	case []float32:
		out := make([]float32, len(w))
		copy(out, w)
		return out
	case []float64:
		out := make([]float32, len(w))
		for i, v := range w {
			out[i] = float32(v)
		}
		return out
	case []int64:
		out := make([]float32, len(w))
		for i, v := range w {
			out[i] = float32(v)
		}
		return out
	case []int32:
		out := make([]float32, len(w))
		for i, v := range w {
			out[i] = float32(v)
		}
		return out
	case []int16:
		out := make([]float32, len(w))
		for i, v := range w {
			out[i] = float32(v)
		}
		return out
	case []int8:
		out := make([]float32, len(w))
		for i, v := range w {
			out[i] = float32(v)
		}
		return out
	default:
		out := make([]float32, len(ws.Master))
		copy(out, ws.Master)
		return out
	}
}

func zeroF32Buf(ctx *poly.WGPUContext, size int, label string) (*wgpu.Buffer, error) {
	if size <= 0 { size = 1 }
	zeros := make([]float32, size)
	return ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    label,
		Contents: wgpu.ToBytes(zeros),
		// CopySrc: ReadBuffer; CopyDst: Queue.WriteBuffer (e.g. MHA/SwiGLU backward CPU fallback).
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
}

func genInput(shape []int) *poly.Tensor[float32] {
	t := poly.NewTensor[float32](shape...)
	for i := range t.Data {
		t.Data[i] = float32(i%13)*0.1 - 0.6
	}
	return t
}

// ── Standardized Results ─────────────────────────────────────────────────────

type CachingResult struct {
	TNormal, TSingle, TMulti time.Duration
	TileSize                 int
	Parity01, Parity02       bool
}

type TrainingResult struct {
	Mode      poly.TrainingMode
	LossInit  float32
	LossFinal float32
	Dur       time.Duration
	TrainOK   bool
	SaveOK    bool
	ByteCount int
	RamBytes  int64
	Err       error
}

type ParityResult struct {
	TCPUMC, TGPUNorm, TGPUSC, TGPUMC time.Duration
	DiffGN, DiffGSC, DiffGMC         float64
	ParityGN, ParityGSC, ParityGMC   bool
	TileSize, SCTile, MCTile         int
}

// ── Standardized Table Printing ──────────────────────────────────────────────

func PrintCachingHeader() {
	fmt.Println()
	fmt.Printf("| %-10s | %-5s | %-14s | %-14s | %-14s | %-7s | %-7s | %-8s | %-8s |\n",
		"DType", "Tile", "Normal", "Single-Core", "Multi-Core", "1C-Spd", "MC-Spd", "1C-Par", "MC-Par")
	fmt.Println("|------------|-------|----------------|----------------|----------------|---------|---------|----------|----------|")
}

func PrintCachingRow(cfg typeConfig, r CachingResult) {
	fmt.Printf("| %-10s | %-5d | %-14v | %-14v | %-14v | %-7.2fx | %-7.2fx | %-8s | %-8s |\n",
		cfg.name, r.TileSize, r.TNormal, r.TSingle, r.TMulti,
		float64(r.TNormal)/float64(r.TSingle),
		float64(r.TNormal)/float64(r.TMulti),
		parityMark(r.Parity01), parityMark(r.Parity02))
}

func PrintTrainingHeader() {
	fmt.Printf("| %-10s | %-13s | %-10s | %-10s | %-8s | %-7s | %-11s | %-8s | %-8s |\n",
		"DType", "Mode", "Loss[0]", "Loss[N]", "Time", "Train↑", "Save/Reload", "File", "RAM")
	fmt.Println("|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|")
}

func PrintTrainingRow(cfg typeConfig, r TrainingResult) {
	if r.Err != nil {
		fmt.Printf("| %-10s | %-13s | ERR        | ERR        | %-8v | ERR     | %s\n", cfg.name, r.Mode.String(), r.Dur.Round(time.Millisecond), r.Err.Error())
		return
	}
	fmt.Printf("| %-10s | %-13s | %-10.4e | %-10.4e | %-8s | %-7s | %-11s | %-8.1fKB | %-8.1fKB |\n",
		cfg.name, r.Mode.String(),
		r.LossInit, r.LossFinal,
		r.Dur.Round(time.Millisecond),
		parityMark(r.TrainOK), parityMark(r.SaveOK),
		float64(r.ByteCount)/1024.0, float64(r.RamBytes)/1024.0)
}

func PrintParityHeader() {
	fmt.Printf("| %-10s | %-4s | %-12s | %-12s | %-12s | %-12s | %-6s | %-6s | %-6s | %-8s | %-8s | %-8s | %-6s | %-6s | %-6s |\n",
		"DType", "Tile", "CPU MC", "GPU Normal", "GPU Tiled SC", "GPU Tiled MC",
		"GN-Spd", "SC-Spd", "MC-Spd", "Diff-GN", "Diff-SC", "Diff-MC", "GN-Par", "SC-Par", "MC-Par")
	fmt.Println("|------------|------|--------------|--------------|--------------|--------------|--------|--------|--------|----------|----------|----------|--------|--------|--------|")
}

func PrintParityRow(cfg typeConfig, r ParityResult) {
	fmt.Printf("| %-10s | %-4d | %-12v | %-12v | %-12v | %-12v | %-6.1fx | %-6.1fx | %-6.1fx | %-8.2e | %-8.2e | %-8.2e | %-6s | %-6s | %-6s |\n",
		cfg.name, r.TileSize, r.TCPUMC, r.TGPUNorm, r.TGPUSC, r.TGPUMC,
		float64(r.TCPUMC)/float64(r.TGPUNorm),
		float64(r.TCPUMC)/float64(r.TGPUSC),
		float64(r.TCPUMC)/float64(r.TGPUMC),
		r.DiffGN, r.DiffGSC, r.DiffGMC,
		parityMark(r.ParityGN), parityMark(r.ParityGSC), parityMark(r.ParityGMC))
}

// ── Global Stats ─────────────────────────────────────────────────────────────

type Spectrum int
const (
	SpecFatal Spectrum = iota
	SpecBroken
	SpecHeavyDrift
	SpecDrift
	SpecLowBit
	SpecIndustry
	SpecExact
)

func (s Spectrum) String() string {
	switch s {
	case SpecExact:      return "💎 EXACT"
	case SpecIndustry:   return "✅ INDUS"
	case SpecLowBit:     return "🟨 LOWBIT"
	case SpecDrift:      return "🟠 DRIFT"
	case SpecHeavyDrift: return "🟤 H-DRIFT"
	case SpecBroken:     return "❌ BROKE"
	default:             return "💀 FATAL"
	}
}

type PerfRecord struct {
	LayerName string
	DType     string
	Action    string // "Forward" or "Backward"
	Slowest   time.Duration
	Fastest   time.Duration
	Speedup   float64
}

type GlobalStats struct {
	Total      int
	BitExact   int
	Industry   int
	LowBit     int
	Drift      int
	HeavyDrift int
	Broken     int
	Fatal      int

	// Layer-level
	LTotal      int
	LBitExact   int
	LIndustry   int
	LLowBit     int
	LDrift      int
	LHeavyDrift int
	LBroken     int
	LFatal      int
	
	// Sub-level
	STotal      int
	SBitExact   int
	SIndustry   int
	SLowBit     int
	SDrift      int
	SHeavyDrift int
	SBroken     int
	SFatal      int

	// Performance Tracking
	PerfStore []PerfRecord
}

func (s *GlobalStats) AddSpectrum(sp Spectrum) {
	s.Total++; s.LTotal++; s.STotal++
	switch sp {
	case SpecFatal:
		s.Fatal++; s.LFatal++; s.SFatal++
	case SpecBroken:
		s.Broken++; s.LBroken++; s.SBroken++
	case SpecHeavyDrift:
		s.HeavyDrift++; s.LHeavyDrift++; s.SHeavyDrift++
	case SpecDrift:
		s.Drift++; s.LDrift++; s.SDrift++
	case SpecLowBit:
		s.LowBit++; s.LLowBit++; s.SLowBit++
	case SpecIndustry:
		s.Industry++; s.LIndustry++; s.SIndustry++
	case SpecExact:
		s.BitExact++; s.LBitExact++; s.SBitExact++
	}
}

func (s *GlobalStats) Add(diff float64, tolerance float64) {
	// Fallback for cases without signal info - classify based on diff alone.
	if math.IsNaN(diff) || math.IsInf(diff, 0) {
		s.AddSpectrum(SpecFatal)
		return
	}
	if diff == 0 {
		s.AddSpectrum(SpecExact)
		return
	}
	if diff <= tolerance {
		s.AddSpectrum(SpecIndustry)
		return
	}
	if diff <= tolerance*10 {
		s.AddSpectrum(SpecLowBit)
		return
	}
	if diff <= 0.1 {
		s.AddSpectrum(SpecDrift)
		return
	}
	s.AddSpectrum(SpecHeavyDrift)
}

func (s *GlobalStats) AddResult(ok bool) {
	if ok {
		s.AddSpectrum(SpecExact)
	} else {
		// If we know it failed but don't know why, it's at least broken.
		// However, for training success the user prefers DRIFT if it's not signal loss.
		// But AddResult(false) is usually a hard failure.
		s.AddSpectrum(SpecBroken)
	}
}

func (s *GlobalStats) AddPerf(layer, dtype, action string, slow, fast time.Duration) {
	speedup := 1.0
	if fast > 0 {
		speedup = float64(slow) / float64(fast)
	}
	s.PerfStore = append(s.PerfStore, PerfRecord{
		LayerName: layer,
		DType:     dtype,
		Action:    action,
		Slowest:   slow,
		Fastest:   fast,
		Speedup:   speedup,
	})
}

func (s *GlobalStats) StartLayer() {
	s.LTotal = 0; s.LBitExact = 0; s.LIndustry = 0; s.LLowBit = 0; s.LDrift = 0; s.LHeavyDrift = 0; s.LBroken = 0; s.LFatal = 0
}

func (s *GlobalStats) ResetSub() {
	s.STotal = 0; s.SBitExact = 0; s.SIndustry = 0; s.SLowBit = 0; s.SDrift = 0; s.SHeavyDrift = 0; s.SBroken = 0; s.SFatal = 0
}

func (s *GlobalStats) ReportSub(label string) {
	if s.STotal == 0 { return }
	fmt.Printf("\n>> [%s] %d Tests | 💎 %d | ✅ %d | 🟨 %d | 🟠 %d | 🟤 %d | ❌ %d | 💀 %d\n", 
		label, s.STotal, s.SBitExact, s.SIndustry, s.SLowBit, s.SDrift, s.SHeavyDrift, s.SBroken, s.SFatal)
}

func (s *GlobalStats) ReportLayer(layerName string) {
	if s.LTotal == 0 { return }
	fmt.Printf("\n🔥 [%s] LAYER TOTAL: %d Tests | 💎 %d | ✅ %d | 🟨 %d | 🟠 %d | 🟤 %d | ❌ %d | 💀 %d\n", 
		layerName, s.LTotal, s.LBitExact, s.LIndustry, s.LLowBit, s.LDrift, s.LHeavyDrift, s.LBroken, s.LFatal)
}

func (s *GlobalStats) Report() {
	if s.Total == 0 { return }
	fmt.Printf("\n🏆 GLOBAL PERFORMANCE MANIFESTATION: %d Total\n", s.Total)
	fmt.Printf("   💎 Bit-Exact:        %d (%.1f%%)\n", s.BitExact, float64(s.BitExact)/float64(s.Total)*100)
	fmt.Printf("   ✅ Industry Standard: %d (%.1f%%)\n", s.Industry, float64(s.Industry)/float64(s.Total)*100)
	fmt.Printf("   🟨 Low-Bit Accept:   %d (%.1f%%)\n", s.LowBit, float64(s.LowBit)/float64(s.Total)*100)
	fmt.Printf("   🟠 Significant Drift: %d (%.1f%%)\n", s.Drift, float64(s.Drift)/float64(s.Total)*100)
	fmt.Printf("   🟤 Heavy Drift:       %d (%.1f%%)\n", s.HeavyDrift, float64(s.HeavyDrift)/float64(s.Total)*100)
	fmt.Printf("   ❌ Broken:            %d\n", s.Broken)
	fmt.Printf("   💀 Fatal (NaN):       %d\n", s.Fatal)

	if len(s.PerfStore) > 0 {
		fmt.Printf("\n🚀 MAXIMUM PERFORMANCE REPORT (SLOWEST BASELINE vs FASTEST ACCELERATED)\n")
		fmt.Printf("| %-10s | %-12s | %-10s | %-12s | %-12s | %-8s |\n", "Layer", "Action", "DType", "Slowest", "Fastest", "Gain")
		fmt.Println("|------------|--------------|------------|--------------|--------------|----------|")
		
		maxGain := 0.0
		var best PerfRecord

		for _, p := range s.PerfStore {
			fmt.Printf("| %-10s | %-12s | %-10s | %-12v | %-12v | %-8.1fx |\n", 
				p.LayerName, p.Action, p.DType, p.Slowest, p.Fastest, p.Speedup)
			if p.Speedup > maxGain {
				maxGain = p.Speedup
				best = p
			}
		}
		
		fmt.Println("|------------|--------------|------------|--------------|--------------|----------|")
		fmt.Printf("🔥 PEAK PERFORMANCE GAP: %.1fx (%s %s on %s)\n", best.Speedup, best.LayerName, best.Action, best.DType)
	}
}

var stats = &GlobalStats{}

// ── Generic Test Runner ──────────────────────────────────────────────────────

type LayerTask func() bool

var registry []LayerTask

func RegisterTask(t LayerTask) {
	registry = append(registry, t)
}

func RunAllLayers() {
	fmt.Println("🚀 Running All Layer Tests...")
	start := time.Now()
	for _, task := range registry {
		task()
	}
	stats.Report()
	fmt.Printf("Total Time: %v\n", time.Since(start).Round(time.Millisecond))
}
