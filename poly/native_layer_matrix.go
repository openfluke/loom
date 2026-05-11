package poly

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/openfluke/webgpu/wgpu"
)

const (
	defaultEpochs       = 8
	defaultBenchIters   = 10
	defaultBenchProfile = "medium"
	defaultBenchTarget  = 500.0
	defaultTrainSize    = 16
	defaultEvalSize     = 10
	defaultLR           = 0.05
)

type NativeMatrixCase struct {
	Name      string
	DType     DType
	Tolerance float64
}

var NativeMatrixAllCases = []NativeMatrixCase{
	{Name: "Float64", DType: DTypeFloat64, Tolerance: 1e-3},
	{Name: "Float32", DType: DTypeFloat32, Tolerance: 1e-5},
	{Name: "Float16", DType: DTypeFloat16, Tolerance: 1e-3},
	{Name: "BFloat16", DType: DTypeBFloat16, Tolerance: 1e-3},
	{Name: "FP8-E4M3", DType: DTypeFP8E4M3, Tolerance: 2e-2},
	{Name: "FP8-E5M2", DType: DTypeFP8E5M2, Tolerance: 2e-2},
	{Name: "Int64", DType: DTypeInt64, Tolerance: 2e-2},
	{Name: "Int32", DType: DTypeInt32, Tolerance: 2e-2},
	{Name: "Int16", DType: DTypeInt16, Tolerance: 2e-2},
	{Name: "Int8", DType: DTypeInt8, Tolerance: 2e-2},
	{Name: "Uint64", DType: DTypeUint64, Tolerance: 2e-2},
	{Name: "Uint32", DType: DTypeUint32, Tolerance: 2e-2},
	{Name: "Uint16", DType: DTypeUint16, Tolerance: 2e-2},
	{Name: "Uint8", DType: DTypeUint8, Tolerance: 2e-2},
	{Name: "Int4", DType: DTypeInt4, Tolerance: 5e-2},
	{Name: "Uint4", DType: DTypeUint4, Tolerance: 5e-2},
	{Name: "FP4", DType: DTypeFP4, Tolerance: 5e-2},
	{Name: "Int2", DType: DTypeInt2, Tolerance: 7e-2},
	{Name: "Uint2", DType: DTypeUint2, Tolerance: 7e-2},
	{Name: "Ternary", DType: DTypeTernary, Tolerance: 1e-1},
	{Name: "Binary", DType: DTypeBinary, Tolerance: 1.5e-1},
}

type NativeMatrixConfig struct {
	Title            string
	Epochs           int
	BenchIters       int
	BenchProfile     string
	BenchTargetMS    float64
	GPUStress        bool
	GPUBenchTargetMS float64
	TrainPerClass    int
	EvalPerClass     int
	LearningRate     float64
	SkipGPU          bool
	Cases            []NativeMatrixCase
}

type NativeMatrixHooks struct {
	BuildProof          func() (NativeMatrixBuildProof, error)
	BuildSamples        func(trainPerClass, evalPerClass int) NativeMatrixSamples
	ResolveBenchProfile func(spec string) (string, any, string, error)
	GrowBenchShape      func(shape any, factor float64, gpuStress bool) (any, string, error)
	BuildBenchmarkInput func(shape any) (*Tensor[float32], error)
	BuildDefaultNetwork func(dtype DType) (*VolumetricNetwork, error)
	BuildNetworkForShape func(dtype DType, shape any) (*VolumetricNetwork, error)
	SupportsNativeCPUExec  func(dtype DType) bool
	SupportsNativeCPUTrain func(dtype DType) bool
	SupportsNativeGPUExec  func(dtype DType) bool
	SupportsNativeGPUTrain func(dtype DType) bool
	UsesPackedGPU          func(dtype DType) bool
	CapabilityNotes        func(tc NativeMatrixCase) string
}

type NativeMatrixBenchmarkPlan struct {
	ProfileName string
	Shape       any
	ShapeSummary string
	Input       *Tensor[float32]
	Note        string
	ProbeMode   TrainingMode
}

type NativeMatrixBuildProof struct {
	Layers         int
	LayersPerCell  int
	SequentialHits int
}

type NativeMatrixSamples struct {
	Train      TrainingBatch[float32]
	TrainCount int
	ParityIn   *Tensor[float32]
	ParityTgt  *Tensor[float32]
	EvalInputs []*Tensor[float32]
	EvalTgts   []*Tensor[float32]
	EvalLabels []int
	EvalCount  int
}

type passSnapshot struct {
	Output     []float32
	InputGrad  []float32
	WeightGrad []float32
}

type cpuParityRow struct {
	TypeName  string
	FwdSC     float64
	FwdMC     float64
	BwdSC     float64
	BwdMC     float64
	Pass      bool
	Skipped   bool
	Reason    string
	Err       error
	SCTiles   string
	MCTiles   string
	NormalDur time.Duration
	SCDur     time.Duration
	MCDur     time.Duration
}

type NativeGPUParityRow struct {
	TypeName string
	FwdGN    float64
	FwdSC    float64
	FwdMC    float64
	BwdGN    float64
	BwdSC    float64
	BwdMC    float64
	Pass     bool
	Skipped  bool
	Reason   string
	Err      error
	GNTiles  string
	GSCTiles string
	GMCTiles string
	GNDur    time.Duration
	GSCDur   time.Duration
	GMCDur   time.Duration
}

type NativeTrainingRow struct {
	TypeName   string
	Mode       TrainingMode
	BeforeAcc  float64
	AfterAcc   float64
	BeforeLoss float64
	AfterLoss  float64
	Duration   time.Duration
	Learned    bool
	Err        error
	Skipped    bool
	Reason     string
	TilePlan   string
}

type NativeCapabilityRow struct {
	TypeName string
	CPUExec  string
	CPUTrain string
	GPUExec  string
	GPUTrain string
	Notes    string
}

type NativePerfRow struct {
	TypeName      string
	Mode          TrainingMode
	Forward       time.Duration
	Backward      time.Duration
	Total         time.Duration
	SamplesPerSec float64
	HostRAMBytes  int64
	VRAMBytes     int64
	TilePlan      string
	Skipped       bool
	Reason        string
	Err           error
}

type spectrum int

const (
	specFatal spectrum = iota
	specBroken
	specHeavyDrift
	specDrift
	specLowBit
	specIndustry
	specExact
)

type spectrumSummary struct {
	Total      int
	Exact      int
	Industry   int
	LowBit     int
	Drift      int
	HeavyDrift int
	Broken     int
	Fatal      int
}

type speedupGroup struct {
	normal *NativePerfRow
	sc     *NativePerfRow
	mc     *NativePerfRow
}

type cpuForwardCapture struct {
	Output  *Tensor[float32]
	HistIn  []*Tensor[float32]
	HistPre []*Tensor[float32]
}

type gpuForwardState struct {
	Ctx       *WGPUContext
	Output    *Tensor[float32]
	HistIn    []*wgpu.Buffer
	HistPre   []*wgpu.Buffer
	WeightBuf []*wgpu.Buffer
	Owned     []*wgpu.Buffer
}
var activeNativeMatrixHooks NativeMatrixHooks
var activeNativeMatrixConfig NativeMatrixConfig

func DefaultNativeMatrixConfig() NativeMatrixConfig {
	return NativeMatrixConfig{
		Epochs:           defaultEpochs,
		BenchIters:       defaultBenchIters,
		BenchProfile:     defaultBenchProfile,
		BenchTargetMS:    defaultBenchTarget,
		GPUBenchTargetMS: defaultBenchTarget,
		TrainPerClass:    defaultTrainSize,
		EvalPerClass:     defaultEvalSize,
		LearningRate:     defaultLR,
		Cases:            append([]NativeMatrixCase(nil), NativeMatrixAllCases...),
	}
}

func RunNativeLayerMatrix(cfg NativeMatrixConfig, hooks NativeMatrixHooks) error {
	if hooks.BuildDefaultNetwork == nil || hooks.BuildNetworkForShape == nil || hooks.BuildSamples == nil ||
		hooks.ResolveBenchProfile == nil || hooks.GrowBenchShape == nil || hooks.BuildBenchmarkInput == nil {
		return fmt.Errorf("native layer matrix hooks are incomplete")
	}
	activeNativeMatrixHooks = hooks
	activeNativeMatrixConfig = cfg

	cases := cfg.Cases
	if len(cases) == 0 {
		cases = append([]NativeMatrixCase(nil), NativeMatrixAllCases...)
	}

	baseBenchProfileName, baseBenchShape, baseBenchSummary, err := resolveBenchProfile(cfg.BenchProfile)
	if err != nil {
		return err
	}

	samples := hooks.BuildSamples(cfg.TrainPerClass, cfg.EvalPerClass)
	proof := NativeMatrixBuildProof{}
	if hooks.BuildProof != nil {
		proof, err = hooks.BuildProof()
		if err != nil {
			return err
		}
	}

	gpuAvailable := false
	gpuReason := "disabled by flag"
	if !cfg.SkipGPU {
		gpuAvailable, gpuReason = probeGPU()
	}

	probeCase := NativeMatrixCase{Name: "Float32", DType: DTypeFloat32, Tolerance: 1e-5}
	cpuBenchPlan, err := tuneBenchmarkPlan(baseBenchProfileName, baseBenchShape, baseBenchSummary, cfg.BenchIters, cfg.BenchTargetMS, probeCase, TrainingModeCPUNormal)
	if err != nil {
		return err
	}
	gpuBenchPlan := cpuBenchPlan
	if gpuAvailable && cfg.GPUStress {
		gpuBenchPlan.Note = fmt.Sprintf("pending GPU-stress retune from shared %s baseline", TrainingModeGPUNormal.String())
	}

	hw := GetHardwareInfo()
	title := cfg.Title
	if title == "" {
		title = "Native Layer Matrix"
	}

	fmt.Printf("=== %s ===\n", title)
	fmt.Printf("System: %s\n", GetDeviceDescription(nil))
	fmt.Printf("CPU Detection: %d cores, L1=%s, L2=%s, L3=%s\n",
		hw.NumCPU, humanBytes(int64(hw.L1DataCacheSize)), humanBytes(int64(hw.L2CacheSize)), humanBytes(int64(hw.L3CacheSize)))
	fmt.Printf("Build proof: layers=%d layers_per_cell=%d sequential_layers=%d\n",
		proof.Layers, proof.LayersPerCell, proof.SequentialHits)
	fmt.Printf("Dataset: train=%d eval=%d epochs=%d lr=%.4f\n",
		samples.TrainCount, samples.EvalCount, cfg.Epochs, cfg.LearningRate)
	fmt.Printf("Benchmark CPU: profile=%s %s bench_iters=%d\n",
		cpuBenchPlan.ProfileName, cpuBenchPlan.ShapeSummary, cfg.BenchIters)
	if cpuBenchPlan.Note != "" {
		fmt.Printf("Benchmark CPU Target: %s\n", cpuBenchPlan.Note)
	}
	if gpuAvailable && cfg.GPUStress {
		fmt.Printf("Benchmark GPU-Stress: enabled, target=%s aggregate on %s baseline; final GPU plan is reported in the performance section\n",
			compactDur(time.Duration(cfg.GPUBenchTargetMS*float64(time.Millisecond))),
			TrainingModeGPUNormal.String(),
		)
	}
	if gpuAvailable {
		fmt.Println("GPU probe: available")
	} else {
		fmt.Printf("GPU probe: skipped (%s)\n", gpuReason)
	}
	fmt.Println()

	cpuRows := make([]cpuParityRow, 0, len(cases))
	gpuRows := make([]NativeGPUParityRow, 0, len(cases))
	trainRows := make([]NativeTrainingRow, 0, len(cases)*6)
	capRows := make([]NativeCapabilityRow, 0, len(cases))
	NativePerfRows := make([]NativePerfRow, 0, len(cases)*6)

	for _, tc := range cases {
		capRows = append(capRows, capabilityForCase(tc))
		cpuRows = append(cpuRows, runCPUParity(tc, samples.ParityIn, samples.ParityTgt))
		if gpuAvailable {
			gpuRows = append(gpuRows, runGPUParity(tc, samples.ParityIn, samples.ParityTgt))
		}
		for _, mode := range trainingModes(gpuAvailable) {
			trainRows = append(trainRows, runTraining(tc, mode, cfg, samples))
			plan := cpuBenchPlan
			if mode.IsGPU() && gpuAvailable && cfg.GPUStress {
				plan = gpuBenchPlan
			}
			NativePerfRows = append(NativePerfRows, runPerformance(tc, mode, cfg.BenchIters, plan.Shape, plan.Input))
		}
	}

	if gpuAvailable && cfg.GPUStress {
		gpuBenchPlan, NativePerfRows, err = retuneGPUStressPlan(cases, NativePerfRows, cpuBenchPlan, cfg)
		if err != nil {
			return err
		}
	}

	printCapabilityAudit(capRows)
	printCPUParity(cpuRows)
	printSpectrumSummary("CPU Parity Spectrum", summarizeCPUParity(cpuRows))
	if gpuAvailable {
		printGPUParity(gpuRows)
		printSpectrumSummary("GPU Parity Spectrum", summarizeGPUParity(gpuRows))
	}
	printTrainingMatrix(trainRows)
	printSpectrumSummary("Learning Spectrum", summarizeTraining(trainRows))
	printPerformanceMatrix(NativePerfRows, cfg.BenchIters, cpuBenchPlan, gpuBenchPlan, gpuAvailable && cfg.GPUStress)
	printTilingSpeedupSummary(NativePerfRows, gpuAvailable && cfg.GPUStress)
	return nil
}

func trainingModes(includeGPU bool) []TrainingMode {
	modes := []TrainingMode{TrainingModeCPUNormal, TrainingModeCPUSC, TrainingModeCPUMC}
	if includeGPU {
		modes = append(modes, TrainingModeGPUNormal, TrainingModeGPUSC, TrainingModeGPUMC)
	}
	return modes
}

func supportsNativeCPUExec(dtype DType) bool {
	if activeNativeMatrixHooks.SupportsNativeCPUExec == nil {
		return false
	}
	return activeNativeMatrixHooks.SupportsNativeCPUExec(dtype)
}

func supportsNativeCPUTrain(dtype DType) bool {
	if activeNativeMatrixHooks.SupportsNativeCPUTrain == nil {
		return false
	}
	return activeNativeMatrixHooks.SupportsNativeCPUTrain(dtype)
}

func supportsNativeGPUExec(dtype DType) bool {
	if activeNativeMatrixHooks.SupportsNativeGPUExec == nil {
		return false
	}
	return activeNativeMatrixHooks.SupportsNativeGPUExec(dtype)
}

func supportsNativeGPUTrain(dtype DType) bool {
	if activeNativeMatrixHooks.SupportsNativeGPUTrain == nil {
		return false
	}
	return activeNativeMatrixHooks.SupportsNativeGPUTrain(dtype)
}

func usesPackedGPUNative(dtype DType) bool {
	if activeNativeMatrixHooks.UsesPackedGPU == nil {
		return false
	}
	return activeNativeMatrixHooks.UsesPackedGPU(dtype)
}

func capabilityForCase(tc NativeMatrixCase) NativeCapabilityRow {
	row := NativeCapabilityRow{TypeName: tc.Name}
	if supportsNativeCPUExec(tc.DType) {
		row.CPUExec = "native"
	} else {
		row.CPUExec = "sim"
	}
	if supportsNativeCPUTrain(tc.DType) {
		row.CPUTrain = "native"
	} else {
		row.CPUTrain = "sim"
	}
	if supportsNativeGPUExec(tc.DType) {
		row.GPUExec = "native"
	} else {
		row.GPUExec = "sim/none"
	}
	if tc.DType == DTypeFloat32 {
		row.GPUTrain = "native"
	} else if supportsNativeGPUTrain(tc.DType) {
		row.GPUTrain = "hybrid"
	} else {
		row.GPUTrain = "sim/none"
	}
	if activeNativeMatrixHooks.CapabilityNotes != nil {
		row.Notes = activeNativeMatrixHooks.CapabilityNotes(tc)
	}
	return row
}

func SelectNativeMatrixCases(spec string, all []NativeMatrixCase) ([]NativeMatrixCase, error) {
	if len(all) == 0 {
		all = NativeMatrixAllCases
	}
	switch strings.ToLower(strings.TrimSpace(spec)) {
	case "", "all":
		out := make([]NativeMatrixCase, len(all))
		copy(out, all)
		return out, nil
	case "fast":
		return SelectNativeMatrixCases("float64,float32,float16,int8,int4,binary", all)
	}

	index := make(map[string]NativeMatrixCase, len(all))
	for _, tc := range all {
		index[strings.ToLower(tc.Name)] = tc
		index[strings.ToLower(tc.DType.String())] = tc
	}

	parts := strings.Split(spec, ",")
	out := make([]NativeMatrixCase, 0, len(parts))
	for _, part := range parts {
		key := strings.ToLower(strings.TrimSpace(part))
		tc, ok := index[key]
		if !ok {
			return nil, fmt.Errorf("unknown dtype %q", part)
		}
		out = append(out, tc)
	}
	return out, nil
}

func probeGPU() (bool, string) {
	net, err := buildDefaultNetwork(DTypeFloat32)
	if err != nil {
		return false, err.Error()
	}
	defer func() {
		if net.GPUContext != nil {
			net.DestroyWGPU()
		}
	}()
	if err := ConfigureNetworkForMode(net, TrainingModeGPUNormal); err != nil {
		return false, err.Error()
	}
	if net.GPUContext != nil {
		limits := net.GPUContext.Limits
		fmt.Printf("GPU Probe: adapter=%s [vulkan]\n", "vulkan")
		fmt.Printf("  MaxComputeWorkgroupStorageSize=%s\n", humanBytes(int64(limits.MaxComputeWorkgroupStorageSize)))
		fmt.Printf("  MaxComputeInvocationsPerWorkgroup=%d\n", limits.MaxComputeInvocationsPerWorkgroup)
		fmt.Printf("  Auto-Detected GPUTileSize=%d\n", net.GPUContext.GPUTileSize)
	}
	return true, ""
}

func buildDefaultNetwork(dtype DType) (*VolumetricNetwork, error) {
	return activeNativeMatrixHooks.BuildDefaultNetwork(dtype)
}

func buildNetworkForShape(dtype DType, shape any) (*VolumetricNetwork, error) {
	return activeNativeMatrixHooks.BuildNetworkForShape(dtype, shape)
}

func prepareNetworkForDType(net *VolumetricNetwork, dtype DType) {
	net.UseTiling = true
	net.EnableMultiCoreTiling = true
	for i := range net.Layers {
		layer := &net.Layers[i]
		layer.DType = dtype
		layer.UseTiling = true
		layer.EnableMultiCoreTiling = true
		if layer.WeightStore == nil {
			continue
		}
		layer.WeightStore.Scale = 1.0
		layer.WeightStore.InvalidateVersions()
		if dtype != DTypeFloat32 {
			layer.WeightStore.Morph(dtype)
		}
	}
	net.SyncToCPU()
	if net.GPUContext != nil {
		net.SyncToGPU()
	}
}

func resolveBenchProfile(spec string) (string, any, string, error) {
	return activeNativeMatrixHooks.ResolveBenchProfile(spec)
}

func tuneBenchmarkPlan(profileName string, base any, baseSummary string, benchIters int, targetMS float64, probeCase NativeMatrixCase, probeMode TrainingMode) (NativeMatrixBenchmarkPlan, error) {
	plan := NativeMatrixBenchmarkPlan{ProfileName: profileName, Shape: base, ShapeSummary: baseSummary, ProbeMode: probeMode}
	if targetMS <= 0 || benchIters <= 0 {
		input, err := buildBenchmarkInput(base)
		if err != nil {
			return NativeMatrixBenchmarkPlan{}, err
		}
		plan.Input = input
		plan.Note = "auto-scale disabled"
		return plan, nil
	}

	targetWindow := time.Duration(targetMS * float64(time.Millisecond))
	shape := base
	summary := baseSummary
	lastWindow := time.Duration(0)
	for step := 0; step < 6; step++ {
		input, err := buildBenchmarkInput(shape)
		if err != nil {
			return NativeMatrixBenchmarkPlan{}, err
		}
		row := runPerformanceMeasured(probeCase, probeMode, 1, shape, input, false)
		if row.Err != nil {
			return NativeMatrixBenchmarkPlan{}, row.Err
		}
		if row.Skipped {
			return NativeMatrixBenchmarkPlan{}, fmt.Errorf("benchmark probe skipped for %s", probeMode.String())
		}

		lastWindow = row.Total * time.Duration(benchIters)
		if lastWindow >= targetWindow {
			plan.Shape, plan.ShapeSummary, plan.Input = shape, summary, input
			plan.Note = fmt.Sprintf("target=%s aggregate on %s baseline via wall-clock probe, achieved=%s", compactDur(targetWindow), probeMode.String(), compactDur(lastWindow))
			if summary != baseSummary {
				plan.ProfileName += "+auto"
			}
			return plan, nil
		}

		next, nextSummary, err := growBenchShape(shape, 1.5, false)
		if err != nil {
			return NativeMatrixBenchmarkPlan{}, err
		}
		if nextSummary == summary {
			break
		}
		shape, summary = next, nextSummary
	}

	input, err := buildBenchmarkInput(shape)
	if err != nil {
		return NativeMatrixBenchmarkPlan{}, err
	}
	plan.Shape, plan.ShapeSummary, plan.Input = shape, summary, input
	plan.Note = fmt.Sprintf("target=%s aggregate on %s baseline via wall-clock probe, capped at %s", compactDur(targetWindow), probeMode.String(), compactDur(lastWindow))
	if summary != baseSummary {
		plan.ProfileName += "+auto"
	}
	return plan, nil
}

func retuneGPUStressPlan(cases []NativeMatrixCase, NativePerfRows []NativePerfRow, cpuPlan NativeMatrixBenchmarkPlan, cfg NativeMatrixConfig) (NativeMatrixBenchmarkPlan, []NativePerfRow, error) {
	plan := cpuPlan
	plan.ProbeMode = TrainingModeGPUNormal
	if cfg.GPUBenchTargetMS <= 0 {
		plan.Note = "gpu-stress auto-scale disabled; shared CPU benchmark plan reused"
		return plan, NativePerfRows, nil
	}

	seed := findPerfRow(NativePerfRows, "Float32", TrainingModeGPUNormal)
	if seed == nil || seed.Err != nil || seed.Skipped || seed.Total <= 0 {
		plan.Note = "gpu-stress retune skipped; shared GPU baseline unavailable"
		return plan, NativePerfRows, nil
	}

	targetWindow := time.Duration(cfg.GPUBenchTargetMS * float64(time.Millisecond))
	sharedWindow := seed.Total * time.Duration(cfg.BenchIters)
	if sharedWindow >= targetWindow {
		plan.Note = fmt.Sprintf("shared %s benchmark already met target=%s, achieved=%s", TrainingModeGPUNormal.String(), compactDur(targetWindow), compactDur(sharedWindow))
		return plan, NativePerfRows, nil
	}

	shape, summary, note := chooseGPUStressShape(cpuPlan.Shape, cpuPlan.ShapeSummary, sharedWindow, targetWindow, cfg.BenchIters)
	input, err := buildBenchmarkInput(shape)
	if err != nil {
		return NativeMatrixBenchmarkPlan{}, nil, err
	}
	plan.Shape, plan.ShapeSummary, plan.Input = shape, summary, input
	plan.ProfileName = cpuPlan.ProfileName + "+gpu"
	plan.Note = note

	rows := append([]NativePerfRow(nil), NativePerfRows...)
	for _, tc := range cases {
		for _, mode := range []TrainingMode{TrainingModeGPUNormal, TrainingModeGPUSC, TrainingModeGPUMC} {
			row := safeRunPerformance(tc, mode, cfg.BenchIters, plan.Shape, plan.Input)
			replacePerfRow(rows, row)
		}
	}

	if achieved := findPerfRow(rows, "Float32", TrainingModeGPUNormal); achieved != nil && achieved.Err == nil && !achieved.Skipped && achieved.Total > 0 {
		plan.Note = fmt.Sprintf("target=%s aggregate on %s baseline via shared-shape heuristic, achieved=%s", compactDur(targetWindow), TrainingModeGPUNormal.String(), compactDur(achieved.Total*time.Duration(cfg.BenchIters)))
	}
	return plan, rows, nil
}

func chooseGPUStressShape(base any, baseSummary string, currentWindow, targetWindow time.Duration, benchIters int) (any, string, string) {
	if currentWindow <= 0 || targetWindow <= currentWindow {
		return base, baseSummary, fmt.Sprintf("shared %s benchmark already satisfied the GPU target window", TrainingModeGPUNormal.String())
	}

	type candidate struct {
		shape any
		summary string
	}
	candidates := []candidate{{shape: base, summary: baseSummary}}
	for _, factor := range []float64{2.0, 1.6, 1.35, 1.2} {
		next, nextSummary, err := growBenchShape(base, factor, true)
		if err == nil {
			candidates = append(candidates, candidate{shape: next, summary: nextSummary})
		}
	}
	candidates = append(candidates, candidate{shape: base, summary: baseSummary})

	probeCase := NativeMatrixCase{Name: "Float32", DType: DTypeFloat32, Tolerance: 1e-5}
	seen := map[string]bool{}
	for _, candidate := range candidates {
		if seen[candidate.summary] {
			continue
		}
		seen[candidate.summary] = true
		input, err := buildBenchmarkInput(candidate.shape)
		if err != nil {
			continue
		}
		row := safeRunPerformance(probeCase, TrainingModeGPUNormal, 1, candidate.shape, input)
		if row.Err != nil || row.Skipped || row.Total <= 0 {
			continue
		}
		aggregate := row.Total * time.Duration(max(1, benchIters))
		if candidate.summary == baseSummary {
			return candidate.shape, candidate.summary, fmt.Sprintf("gpu-stress fallback kept shared plan after larger shapes failed; shared baseline=%s", compactDur(aggregate))
		}
		return candidate.shape, candidate.summary, fmt.Sprintf("gpu-stress canary selected a stable shape from shared baseline=%s toward target=%s", compactDur(currentWindow), compactDur(targetWindow))
	}
	return base, baseSummary, "gpu-stress fallback kept the shared benchmark plan after all larger canaries failed"
}

func safeRunPerformance(tc NativeMatrixCase, mode TrainingMode, benchIters int, shape any, input *Tensor[float32]) (row NativePerfRow) {
	defer func() {
		if r := recover(); r != nil {
			row = NativePerfRow{TypeName: tc.Name, Mode: mode, Skipped: true, Reason: "panic", Err: fmt.Errorf("panic during performance run: %v", r)}
		}
	}()
	return runPerformance(tc, mode, benchIters, shape, input)
}

func findPerfRow(rows []NativePerfRow, typeName string, mode TrainingMode) *NativePerfRow {
	for i := range rows {
		if rows[i].TypeName == typeName && rows[i].Mode == mode {
			return &rows[i]
		}
	}
	return nil
}

func replacePerfRow(rows []NativePerfRow, next NativePerfRow) {
	for i := range rows {
		if rows[i].TypeName == next.TypeName && rows[i].Mode == next.Mode {
			rows[i] = next
			return
		}
	}
}

func growBenchShape(shape any, factor float64, gpuStress bool) (any, string, error) {
	return activeNativeMatrixHooks.GrowBenchShape(shape, factor, gpuStress)
}

func buildBenchmarkInput(shape any) (*Tensor[float32], error) {
	return activeNativeMatrixHooks.BuildBenchmarkInput(shape)
}

func growInt(v int, factor float64, multiple int) int {
	if v <= 0 {
		v = 1
	}
	next := int(math.Ceil(float64(v) * factor))
	if next <= v {
		next = v + 1
	}
	if multiple > 1 {
		next = ((next + multiple - 1) / multiple) * multiple
	}
	return next
}

func runCPUParity(tc NativeMatrixCase, input, target *Tensor[float32]) cpuParityRow {
	row := cpuParityRow{TypeName: tc.Name}
	if !supportsNativeCPUExec(tc.DType) {
		row.Skipped = true
		row.Reason = "simulated"
		return row
	}

	t0 := time.Now()
	normal, err := snapshotCPU(tc.DType, TrainingModeCPUNormal, input, target)
	row.NormalDur = time.Since(t0)
	if err != nil {
		row.Err = err
		return row
	}

	t1 := time.Now()
	sc, err := snapshotCPU(tc.DType, TrainingModeCPUSC, input, target)
	row.SCDur = time.Since(t1)
	if err != nil {
		row.Err = err
		return row
	}

	t2 := time.Now()
	mc, err := snapshotCPU(tc.DType, TrainingModeCPUMC, input, target)
	row.MCDur = time.Since(t2)
	if err != nil {
		row.Err = err
		return row
	}

	row.SCTiles = tilePlanForMode(nil, TrainingModeCPUSC, tc.DType) // Simplified, buildDefaultNetwork handles layer-specifics
	row.MCTiles = tilePlanForMode(nil, TrainingModeCPUMC, tc.DType)

	row.FwdSC = maxAbsDiff(normal.Output, sc.Output)
	row.FwdMC = maxAbsDiff(normal.Output, mc.Output)
	row.BwdSC = max3(
		maxAbsDiff(normal.InputGrad, sc.InputGrad),
		maxAbsDiff(normal.WeightGrad, sc.WeightGrad),
		0,
	)
	row.BwdMC = max3(
		maxAbsDiff(normal.InputGrad, mc.InputGrad),
		maxAbsDiff(normal.WeightGrad, mc.WeightGrad),
		0,
	)
	row.Pass = row.FwdSC <= tc.Tolerance &&
		row.FwdMC <= tc.Tolerance &&
		row.BwdSC <= tc.Tolerance &&
		row.BwdMC <= tc.Tolerance
	return row
}

func runGPUParity(tc NativeMatrixCase, input, target *Tensor[float32]) NativeGPUParityRow {
	row := NativeGPUParityRow{TypeName: tc.Name}
	if !supportsNativeGPUExec(tc.DType) {
		row.Skipped = true
		row.Reason = "simulated-or-missing"
		return row
	}

	t0 := time.Now()
	cpuBase, err := snapshotCPU(tc.DType, TrainingModeCPUNormal, input, target)
	if err != nil {
		row.Err = err
		return row
	}

	tGN := time.Now()
	gn, err := snapshotGPU(tc.DType, TrainingModeGPUNormal, input, target)
	row.GNDur = time.Since(tGN)
	if err != nil {
		row.Err = err
		return row
	}

	tGSC := time.Now()
	gsc, err := snapshotGPU(tc.DType, TrainingModeGPUSC, input, target)
	row.GSCDur = time.Since(tGSC)
	if err != nil {
		row.Err = err
		return row
	}

	tGMC := time.Now()
	gmc, err := snapshotGPU(tc.DType, TrainingModeGPUMC, input, target)
	row.GMCDur = time.Since(tGMC)
	if err != nil {
		row.Err = err
		return row
	}

	row.GNTiles = "-"
	row.GSCTiles = tilePlanForMode(nil, TrainingModeGPUSC, tc.DType)
	row.GMCTiles = tilePlanForMode(nil, TrainingModeGPUMC, tc.DType)
	_ = t0 // Silence unused

	row.FwdGN = maxAbsDiff(cpuBase.Output, gn.Output)
	row.FwdSC = maxAbsDiff(cpuBase.Output, gsc.Output)
	row.FwdMC = maxAbsDiff(cpuBase.Output, gmc.Output)
	row.BwdGN = max3(maxAbsDiff(cpuBase.InputGrad, gn.InputGrad), maxAbsDiff(cpuBase.WeightGrad, gn.WeightGrad), 0)
	row.BwdSC = max3(maxAbsDiff(cpuBase.InputGrad, gsc.InputGrad), maxAbsDiff(cpuBase.WeightGrad, gsc.WeightGrad), 0)
	row.BwdMC = max3(maxAbsDiff(cpuBase.InputGrad, gmc.InputGrad), maxAbsDiff(cpuBase.WeightGrad, gmc.WeightGrad), 0)

	limit := tc.Tolerance * 4
	if limit < 1e-4 {
		limit = 1e-4
	}
	row.Pass = row.FwdGN <= limit &&
		row.FwdSC <= limit &&
		row.FwdMC <= limit &&
		row.BwdGN <= limit &&
		row.BwdSC <= limit &&
		row.BwdMC <= limit
	return row
}

func runTraining(tc NativeMatrixCase, mode TrainingMode, cfg NativeMatrixConfig, samples NativeMatrixSamples) NativeTrainingRow {
	row := NativeTrainingRow{
		TypeName: tc.Name,
		Mode:     mode,
	}
	if mode.IsGPU() && !supportsNativeGPUTrain(tc.DType) {
		row.Skipped = true
		row.Reason = "simulated-or-missing"
		return row
	}
	if !mode.IsGPU() && !supportsNativeCPUTrain(tc.DType) {
		row.Skipped = true
		row.Reason = "simulated"
		return row
	}

	net, err := buildDefaultNetwork(tc.DType)
	if err != nil {
		row.Err = err
		return row
	}
	defer func() {
		if net.GPUContext != nil {
			net.DestroyWGPU()
		}
	}()

	row.TilePlan = tilePlanForMode(net, mode, tc.DType)

	row.BeforeAcc, row.BeforeLoss, err = scoreNetwork(net, samples.EvalInputs, samples.EvalTgts, samples.EvalLabels)
	if err != nil {
		row.Err = err
		return row
	}

	start := time.Now()
	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		_, err = Train(net, []TrainingBatch[float32]{samples.Train}, &TrainingConfig{
			Epochs:       1,
			LearningRate: float32(cfg.LearningRate),
			LossType:     "mse",
			Mode:         mode,
			Verbose:      false,
		})
		if err != nil {
			row.Err = err
			return row
		}
		if mode.IsGPU() {
			if err := SyncWeightsFromGPU(net); err != nil {
				row.Err = err
				return row
			}
		}
	}
	row.Duration = time.Since(start)

	prepareNetworkForDType(net, tc.DType)
	row.AfterAcc, row.AfterLoss, err = scoreNetwork(net, samples.EvalInputs, samples.EvalTgts, samples.EvalLabels)
	if err != nil {
		row.Err = err
		return row
	}

	improvedLoss := row.AfterLoss < row.BeforeLoss
	improvedAccuracy := row.AfterAcc > row.BeforeAcc
	saturatedAccuracy := row.BeforeAcc >= 99.9 && row.AfterAcc >= row.BeforeAcc
	row.Learned = improvedLoss && (improvedAccuracy || saturatedAccuracy)
	return row
}

func runPerformance(tc NativeMatrixCase, mode TrainingMode, benchIters int, shape any, input *Tensor[float32]) NativePerfRow {
	return runPerformanceMeasured(tc, mode, benchIters, shape, input, true)
}

func runPerformanceMeasured(tc NativeMatrixCase, mode TrainingMode, benchIters int, shape any, input *Tensor[float32], allowTimestampTiming bool) NativePerfRow {
	row := NativePerfRow{
		TypeName: tc.Name,
		Mode:     mode,
	}
	if benchIters <= 0 {
		row.Skipped = true
		row.Reason = "disabled"
		return row
	}
	if mode.IsGPU() && !supportsNativeGPUExec(tc.DType) {
		row.Skipped = true
		row.Reason = "simulated-or-missing"
		return row
	}
	if !mode.IsGPU() && !supportsNativeCPUExec(tc.DType) {
		row.Skipped = true
		row.Reason = "simulated"
		return row
	}

	net, err := buildNetworkForShape(tc.DType, shape)
	if err != nil {
		row.Err = err
		return row
	}
	defer func() {
		if net.GPUContext != nil {
			net.DestroyWGPU()
		}
	}()

	if err := ConfigureNetworkForMode(net, mode); err != nil {
		row.Err = err
		return row
	}
	row.TilePlan = tilePlanForMode(net, mode, tc.DType)

	outClasses := 1
	if len(net.Layers) > 0 {
		outClasses = max(1, net.Layers[len(net.Layers)-1].Filters)
	}
	gradOut := benchmarkGradForBatch(input.Shape[0], outClasses)
	if mode.IsGPU() {
		state, err := forwardGPUTrace(net, input, mode)
		if err != nil {
			row.Err = err
			return row
		}
		err = backwardGPUTrace(net, state, gradOut, mode)
		state.Destroy()
		if err != nil {
			row.Err = err
			return row
		}
		row.VRAMBytes = net.GetVRAMUsage()
	} else {
		capture := forwardCPU(net, input)
		_, _, _ = BackwardPolymorphic(net, gradOut, capture.HistIn, capture.HistPre)
	}
	row.HostRAMBytes = int64(net.CalculateTotalMemory())
	useTimestampTiming := allowTimestampTiming && mode.IsGPU() && net.GPUContext != nil && net.GPUContext.HasTimestampQuery

	var fwdTotal time.Duration
	for i := 0; i < benchIters; i++ {
		if mode.IsGPU() {
			if useTimestampTiming {
				var state *gpuForwardState
				dur, err := net.GPUContext.TimeCommands("cnn1_suite_perf_fwd", func() error {
					var callErr error
					state, callErr = forwardGPUTrace(net, input, mode)
					return callErr
				})
				if state != nil {
					state.Destroy()
				}
				if err != nil {
					row.Err = err
					return row
				}
				fwdTotal += dur
				continue
			}
			start := time.Now()
			state, err := forwardGPUTrace(net, input, mode)
			if err != nil {
				row.Err = err
				return row
			}
			fwdTotal += time.Since(start)
			state.Destroy()
			continue
		}
		start := time.Now()
		_ = forwardCPU(net, input)
		fwdTotal += time.Since(start)
	}

	var bwdTotal time.Duration
	for i := 0; i < benchIters; i++ {
		if mode.IsGPU() {
			state, err := forwardGPUTrace(net, input, mode)
			if err != nil {
				row.Err = err
				return row
			}
			if useTimestampTiming {
				dur, err := net.GPUContext.TimeCommands("cnn1_suite_perf_bwd", func() error {
					return backwardGPUTrace(net, state, gradOut, mode)
				})
				state.Destroy()
				if err != nil {
					row.Err = err
					return row
				}
				bwdTotal += dur
				continue
			}
			start := time.Now()
			err = backwardGPUTrace(net, state, gradOut, mode)
			bwdTotal += time.Since(start)
			state.Destroy()
			if err != nil {
				row.Err = err
				return row
			}
			continue
		}
		capture := forwardCPU(net, input)
		start := time.Now()
		_, _, _ = BackwardPolymorphic(net, gradOut, capture.HistIn, capture.HistPre)
		bwdTotal += time.Since(start)
	}

	var totalTotal time.Duration
	for i := 0; i < benchIters; i++ {
		if mode.IsGPU() {
			if useTimestampTiming {
				var state *gpuForwardState
				dur, err := net.GPUContext.TimeCommands("cnn1_suite_perf_total", func() error {
					var callErr error
					state, callErr = forwardGPUTrace(net, input, mode)
					if callErr != nil {
						return callErr
					}
					return backwardGPUTrace(net, state, gradOut, mode)
				})
				if state != nil {
					state.Destroy()
				}
				if err != nil {
					row.Err = err
					return row
				}
				totalTotal += dur
				continue
			}
			start := time.Now()
			state, err := forwardGPUTrace(net, input, mode)
			if err != nil {
				row.Err = err
				return row
			}
			err = backwardGPUTrace(net, state, gradOut, mode)
			state.Destroy()
			if err != nil {
				row.Err = err
				return row
			}
			totalTotal += time.Since(start)
			continue
		}
		start := time.Now()
		capture := forwardCPU(net, input)
		_, _, _ = BackwardPolymorphic(net, gradOut, capture.HistIn, capture.HistPre)
		totalTotal += time.Since(start)
	}

	row.Forward = fwdTotal / time.Duration(benchIters)
	row.Backward = bwdTotal / time.Duration(benchIters)
	row.Total = totalTotal / time.Duration(benchIters)
	if row.Total > 0 {
		row.SamplesPerSec = float64(input.Shape[0]) / row.Total.Seconds()
	}
	if mode.IsGPU() {
		if vram := net.GetVRAMUsage(); vram > row.VRAMBytes {
			row.VRAMBytes = vram
		}
	}
	return row
}

func benchmarkGradForBatch(batchSize, outputClasses int) *Tensor[float32] {
	grad := NewTensor[float32](batchSize, outputClasses, 1)
	for i := range grad.Data {
		col := i % outputClasses
		grad.Data[i] = -0.12
		if col == (i/outputClasses)%outputClasses {
			grad.Data[i] = 0.45
		}
	}
	return grad
}

func addPulseSized(data []float32, seqLen, channel, center int, amp float32) {
	kernel := []float32{0.15, 0.6, 1.0, 0.6, 0.15}
	base := channel * seqLen
	for i := range kernel {
		pos := center + i - 2
		if pos < 0 || pos >= seqLen {
			continue
		}
		data[base+pos] += amp * kernel[i]
	}
}

func scoreNetwork(
	net *VolumetricNetwork,
	inputs []*Tensor[float32],
	targets []*Tensor[float32],
	labels []int,
) (float64, float64, error) {
	if err := ConfigureNetworkForMode(net, TrainingModeCPUNormal); err != nil {
		return 0, 0, err
	}

	correct := 0
	totalLoss := 0.0
	for i := range inputs {
		output, _, _ := ForwardPolymorphic(net, inputs[i])
		if argmax(output.Data) == labels[i] {
			correct++
		}
		totalLoss += CalculateLoss(output, targets[i], "mse")
	}

	accuracy := 100 * float64(correct) / float64(len(inputs))
	avgLoss := totalLoss / float64(len(inputs))
	return accuracy, avgLoss, nil
}

func snapshotCPU(dtype DType, mode TrainingMode, input, target *Tensor[float32]) (passSnapshot, error) {
	net, err := buildDefaultNetwork(dtype)
	if err != nil {
		return passSnapshot{}, err
	}
	defer func() {
		if net.GPUContext != nil {
			net.DestroyWGPU()
		}
	}()

	if err := ConfigureNetworkForMode(net, mode); err != nil {
		return passSnapshot{}, err
	}

	capture := forwardCPU(net, input)
	gradOut := ComputeLossGradient(capture.Output, target, "mse")
	inputGrad, layerGrads, _ := BackwardPolymorphic(net, gradOut, capture.HistIn, capture.HistPre)

	return passSnapshot{
		Output:     cloneSlice(capture.Output.Data),
		InputGrad:  cloneSlice(inputGrad.Data),
		WeightGrad: flattenCPUWeightGrads(layerGrads),
	}, nil
}

func forwardCPU(net *VolumetricNetwork, input *Tensor[float32]) cpuForwardCapture {
	histIn := make([]*Tensor[float32], len(net.Layers))
	histPre := make([]*Tensor[float32], len(net.Layers))
	current := input

	for i := range net.Layers {
		layer := &net.Layers[i]
		histIn[i] = current
		pre, post := DispatchLayer(layer, current, nil)
		histPre[i] = pre
		current = post
	}

	return cpuForwardCapture{
		Output:  current,
		HistIn:  histIn,
		HistPre: histPre,
	}
}

func snapshotGPU(dtype DType, mode TrainingMode, input, target *Tensor[float32]) (passSnapshot, error) {
	net, err := buildDefaultNetwork(dtype)
	if err != nil {
		return passSnapshot{}, err
	}
	defer func() {
		if net.GPUContext != nil {
			net.DestroyWGPU()
		}
	}()

	if err := ConfigureNetworkForMode(net, mode); err != nil {
		return passSnapshot{}, err
	}

	state, err := forwardGPU(net, input, mode)
	if err != nil {
		return passSnapshot{}, err
	}
	defer state.Destroy()

	gradOut := ComputeLossGradient(state.Output, target, "mse")
	inputGrad, weightGrad, err := backwardGPU(net, state, gradOut, mode)
	if err != nil {
		return passSnapshot{}, err
	}

	return passSnapshot{
		Output:     cloneSlice(state.Output.Data),
		InputGrad:  inputGrad,
		WeightGrad: weightGrad,
	}, nil
}

func forwardGPU(net *VolumetricNetwork, input *Tensor[float32], mode TrainingMode) (*gpuForwardState, error) {
	return forwardGPUInternal(net, input, mode, true)
}

func forwardGPUTrace(net *VolumetricNetwork, input *Tensor[float32], mode TrainingMode) (*gpuForwardState, error) {
	return forwardGPUInternal(net, input, mode, false)
}

func forwardGPUInternal(net *VolumetricNetwork, input *Tensor[float32], mode TrainingMode, readOutput bool) (*gpuForwardState, error) {
	ctx := net.GPUContext
	if ctx == nil {
		return nil, fmt.Errorf("gpu context not initialized")
	}

	inputBuf, err := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "cnn1_suite_input",
		Contents: wgpu.ToBytes(input.Data),
		Usage:    wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, err
	}

	state := &gpuForwardState{
		Ctx:       ctx,
		HistIn:    make([]*wgpu.Buffer, len(net.Layers)),
		HistPre:   make([]*wgpu.Buffer, len(net.Layers)),
		WeightBuf: make([]*wgpu.Buffer, len(net.Layers)),
		Owned:     []*wgpu.Buffer{inputBuf},
	}

	current := inputBuf
	batchSize := input.Shape[0]
	for i := range net.Layers {
		layer := &net.Layers[i]
		outElems := batchSize * layer.Filters * layer.OutputHeight
		preBuf := ctx.GetActivationBuffer(
			fmt.Sprintf("cnn1_suite_pre_%d_%d", i, mode),
			uint64(outElems*4),
			wgpu.BufferUsageStorage,
		)
		if preBuf == nil {
			return nil, fmt.Errorf("failed to allocate pre buffer for layer %d", i)
		}

		wBuf := GetGPUWeightBuffer(layer)
		if wBuf == nil {
			return nil, fmt.Errorf("missing gpu weight buffer for layer %d", i)
		}
		state.WeightBuf[i] = wBuf
		state.HistIn[i] = current
		state.HistPre[i] = preBuf

		if mode == TrainingModeGPUNormal {
			if usesPackedGPUNative(layer.DType) {
				err = ctx.DispatchCNN1Packed(
					layer.DType,
					batchSize,
					layer.InputChannels,
					layer.InputHeight,
					layer.Filters,
					layer.OutputHeight,
					layer.KernelSize,
					layer.Stride,
					layer.Padding,
					cnn1SuitePackedScale(layer),
					current,
					wBuf,
					preBuf,
				)
			} else {
				err = ctx.DispatchCNN1(
					batchSize,
					layer.InputChannels,
					layer.InputHeight,
					layer.Filters,
					layer.OutputHeight,
					layer.KernelSize,
					layer.Stride,
					layer.Padding,
					current,
					wBuf,
					preBuf,
				)
			}
		} else {
			tileSize := layer.GetGPUSCTileSize(layer.DType)
			if mode == TrainingModeGPUMC {
				tileSize = layer.GetGPUMCTileSize(layer.DType)
			}
			if usesPackedGPUNative(layer.DType) {
				err = ctx.DispatchCNN1PackedTiled(
					layer.DType,
					tileSize,
					layer.InputChannels*layer.KernelSize,
					batchSize,
					layer.InputChannels,
					layer.InputHeight,
					layer.Filters,
					layer.OutputHeight,
					layer.KernelSize,
					layer.Stride,
					layer.Padding,
					cnn1SuitePackedScale(layer),
					current,
					wBuf,
					preBuf,
				)
			} else {
				err = ctx.DispatchCNN1Tiled(
					tileSize,
					layer.InputChannels*layer.KernelSize,
					batchSize,
					layer.InputChannels,
					layer.InputHeight,
					layer.Filters,
					layer.OutputHeight,
					layer.KernelSize,
					layer.Stride,
					layer.Padding,
					1.0,
					current,
					wBuf,
					preBuf,
				)
			}
		}
		if err != nil {
			return nil, err
		}

		if layer.Activation == ActivationLinear {
			current = preBuf
			continue
		}

		postBuf := ctx.GetActivationBuffer(
			fmt.Sprintf("cnn1_suite_post_%d_%d", i, mode),
			uint64(outElems*4),
			wgpu.BufferUsageStorage,
		)
		if postBuf == nil {
			return nil, fmt.Errorf("failed to allocate post buffer for layer %d", i)
		}
		if err := ctx.DispatchActivation(outElems, layer.Activation, preBuf, postBuf); err != nil {
			return nil, err
		}
		current = postBuf
	}

	if readOutput {
		outputData, err := ctx.ReadBuffer(current)
		if err != nil {
			return nil, err
		}
		last := &net.Layers[len(net.Layers)-1]
		state.Output = NewTensor[float32](batchSize, last.Filters, last.OutputHeight)
		copy(state.Output.Data, outputData)
	}
	return state, nil
}

func backwardGPU(net *VolumetricNetwork, state *gpuForwardState, gradOut *Tensor[float32], mode TrainingMode) ([]float32, []float32, error) {
	return backwardGPUInternal(net, state, gradOut, mode, true)
}

func backwardGPUTrace(net *VolumetricNetwork, state *gpuForwardState, gradOut *Tensor[float32], mode TrainingMode) error {
	_, _, err := backwardGPUInternal(net, state, gradOut, mode, false)
	return err
}

func backwardGPUInternal(net *VolumetricNetwork, state *gpuForwardState, gradOut *Tensor[float32], mode TrainingMode, readBack bool) ([]float32, []float32, error) {
	ctx := state.Ctx
	gradBuf, err := ctx.CreatePersistentBuffer(gradOut.Data, "cnn1_suite_grad")
	if err != nil {
		return nil, nil, err
	}
	state.Owned = append(state.Owned, gradBuf)

	currentGrad := gradBuf
	layerGradients := make([][]float32, len(net.Layers))
	var inputGrad []float32
	batchSize := gradOut.Shape[0]

	for i := len(net.Layers) - 1; i >= 0; i-- {
		layer := &net.Layers[i]
		dxBuf, err := zeroF32Buf(ctx, batchSize*layer.InputChannels*layer.InputHeight, fmt.Sprintf("cnn1_suite_dx_%d", i))
		if err != nil {
			return nil, nil, err
		}
		dwBuf, err := zeroF32Buf(ctx, len(layer.WeightStore.Master), fmt.Sprintf("cnn1_suite_dw_%d", i))
		if err != nil {
			return nil, nil, err
		}
		state.Owned = append(state.Owned, dxBuf, dwBuf)

		if mode == TrainingModeGPUNormal {
			if usesPackedGPUNative(layer.DType) {
				if err := ctx.DispatchCNN1PackedBackwardDX(
					layer.DType,
					batchSize,
					layer.InputChannels,
					layer.InputHeight,
					layer.Filters,
					layer.OutputHeight,
					layer.KernelSize,
					layer.Stride,
					layer.Padding,
					layer.Activation,
					cnn1SuitePackedScale(layer),
					currentGrad,
					state.WeightBuf[i],
					state.HistPre[i],
					dxBuf,
				); err != nil {
					return nil, nil, err
				}
			} else {
				if err := ctx.DispatchCNN1BackwardDX(
					batchSize,
					layer.InputChannels,
					layer.InputHeight,
					layer.Filters,
					layer.OutputHeight,
					layer.KernelSize,
					layer.Stride,
					layer.Padding,
					layer.Activation,
					currentGrad,
					state.WeightBuf[i],
					state.HistPre[i],
					dxBuf,
				); err != nil {
					return nil, nil, err
				}
			}
			if err := ctx.DispatchCNN1BackwardDW(
				batchSize,
				layer.InputChannels,
				layer.InputHeight,
				layer.Filters,
				layer.OutputHeight,
				layer.KernelSize,
				layer.Stride,
				layer.Padding,
				layer.Activation,
				currentGrad,
				state.HistIn[i],
				state.HistPre[i],
				dwBuf,
			); err != nil {
				return nil, nil, err
			}
		} else {
			tileSize := layer.GetGPUSCTileSize(layer.DType)
			if mode == TrainingModeGPUMC {
				tileSize = layer.GetGPUMCTileSize(layer.DType)
			}
			kernelVol := layer.InputChannels * layer.KernelSize
			if usesPackedGPUNative(layer.DType) {
				if err := ctx.DispatchCNN1PackedBackwardDXTiled(
					layer.DType,
					tileSize,
					kernelVol,
					batchSize,
					layer.InputChannels,
					layer.InputHeight,
					layer.Filters,
					layer.OutputHeight,
					layer.KernelSize,
					layer.Stride,
					layer.Padding,
					layer.Activation,
					cnn1SuitePackedScale(layer),
					currentGrad,
					state.WeightBuf[i],
					state.HistPre[i],
					dxBuf,
				); err != nil {
					return nil, nil, err
				}
			} else {
				if err := ctx.DispatchCNN1TiledBackwardDX(
					tileSize,
					kernelVol,
					batchSize,
					layer.InputChannels,
					layer.InputHeight,
					layer.Filters,
					layer.OutputHeight,
					layer.KernelSize,
					layer.Stride,
					layer.Padding,
					layer.Activation,
					currentGrad,
					state.WeightBuf[i],
					state.HistPre[i],
					dxBuf,
				); err != nil {
					return nil, nil, err
				}
			}
			if err := ctx.DispatchCNN1TiledBackwardDW(
				tileSize,
				batchSize,
				layer.InputChannels,
				layer.InputHeight,
				layer.Filters,
				layer.OutputHeight,
				layer.KernelSize,
				layer.Stride,
				layer.Padding,
				layer.Activation,
				currentGrad,
				state.HistIn[i],
				state.HistPre[i],
				dwBuf,
			); err != nil {
				return nil, nil, err
			}
		}

		if readBack {
			layerGradients[i], err = ctx.ReadBuffer(dwBuf)
			if err != nil {
				return nil, nil, err
			}
			if i == 0 {
				inputGrad, err = ctx.ReadBuffer(dxBuf)
				if err != nil {
					return nil, nil, err
				}
			}
		}
		currentGrad = dxBuf
	}

	if !readBack {
		return nil, nil, nil
	}
	return inputGrad, flattenNested(layerGradients), nil
}

func (s *gpuForwardState) Destroy() {
	for _, buf := range s.Owned {
		if buf != nil {
			buf.Destroy()
		}
	}
	if s.Ctx != nil {
		// These suite helpers allocate short-lived buffers per pass; clearing bind
		// groups avoids reusing cached bindings that still point at destroyed buffers.
		s.Ctx.ResetCache()
	}
}

func dequantizedWeights(layer *VolumetricLayer) []float32 {
	if layer.WeightStore == nil {
		return nil
	}
	active := layer.WeightStore.GetActive(layer.DType)
	if active == nil {
		return cloneSlice(layer.WeightStore.Master)
	}
	return cloneSlice(CastWeights[float32](active))
}

func cnn1SuitePackedScale(layer *VolumetricLayer) float32 {
	if layer == nil || layer.WeightStore == nil {
		return 1.0
	}
	
	if layer.WeightStore.Scale != 0 {
		return layer.WeightStore.Scale
	}
	return 1.0
}

func zeroF32Buf(ctx *WGPUContext, size int, label string) (*wgpu.Buffer, error) {
	if size <= 0 {
		size = 1
	}
	zeros := make([]float32, size)
	return ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    label,
		Contents: wgpu.ToBytes(zeros),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
}

func flattenCPUWeightGrads(grads [][2]*Tensor[float32]) []float32 {
	total := 0
	for i := range grads {
		if grads[i][1] != nil {
			total += len(grads[i][1].Data)
		}
	}

	out := make([]float32, 0, total)
	for i := range grads {
		if grads[i][1] != nil {
			out = append(out, grads[i][1].Data...)
		}
	}
	return out
}

func flattenNested(parts [][]float32) []float32 {
	total := 0
	for _, part := range parts {
		total += len(part)
	}
	out := make([]float32, 0, total)
	for _, part := range parts {
		out = append(out, part...)
	}
	return out
}

func cloneSlice(in []float32) []float32 {
	out := make([]float32, len(in))
	copy(out, in)
	return out
}

func argmax(data []float32) int {
	bestIdx := 0
	bestVal := data[0]
	for i := 1; i < len(data); i++ {
		if data[i] > bestVal {
			bestVal = data[i]
			bestIdx = i
		}
	}
	return bestIdx
}

func maxAbsDiff(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	maxDiff := 0.0
	for i := range a {
		diff := math.Abs(float64(a[i] - b[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	return maxDiff
}

func max3(a, b, c float64) float64 {
	if a < b {
		a = b
	}
	if a < c {
		a = c
	}
	return a
}

func mark(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}

func classifySpectrum(diff, tolerance float64) spectrum {
	if math.IsNaN(diff) || math.IsInf(diff, 0) {
		return specFatal
	}
	if diff < 0 {
		return specBroken
	}
	if diff == 0 {
		return specExact
	}
	if diff <= tolerance {
		return specIndustry
	}
	if diff <= tolerance*10 {
		return specLowBit
	}
	if diff <= 0.1 {
		return specDrift
	}
	return specHeavyDrift
}

func (s *spectrumSummary) add(sp spectrum) {
	s.Total++
	switch sp {
	case specExact:
		s.Exact++
	case specIndustry:
		s.Industry++
	case specLowBit:
		s.LowBit++
	case specDrift:
		s.Drift++
	case specHeavyDrift:
		s.HeavyDrift++
	case specBroken:
		s.Broken++
	case specFatal:
		s.Fatal++
	}
}

func printSpectrumSummary(label string, summary spectrumSummary) {
	if summary.Total == 0 {
		return
	}
	fmt.Printf(">> [%s] %d Tests | \U0001F48E %d | \u2705 %d | \U0001F7E8 %d | \U0001F7E0 %d | \U0001F7E4 %d | \u274C %d | \U0001F480 %d\n",
		label,
		summary.Total,
		summary.Exact,
		summary.Industry,
		summary.LowBit,
		summary.Drift,
		summary.HeavyDrift,
		summary.Broken,
		summary.Fatal,
	)
	fmt.Println()
}

func summarizeCPUParity(rows []cpuParityRow) spectrumSummary {
	tol := toleranceByType()
	var out spectrumSummary
	for _, row := range rows {
		if row.Skipped || row.Err != nil {
			continue
		}
		out.add(classifySpectrum(row.FwdSC, tol[row.TypeName]))
		out.add(classifySpectrum(row.BwdSC, tol[row.TypeName]))
		out.add(classifySpectrum(row.FwdMC, tol[row.TypeName]))
		out.add(classifySpectrum(row.BwdMC, tol[row.TypeName]))
	}
	return out
}

func summarizeGPUParity(rows []NativeGPUParityRow) spectrumSummary {
	tol := toleranceByType()
	var out spectrumSummary
	for _, row := range rows {
		if row.Skipped || row.Err != nil {
			continue
		}
		out.add(classifySpectrum(row.FwdGN, tol[row.TypeName]))
		out.add(classifySpectrum(row.BwdGN, tol[row.TypeName]))
		out.add(classifySpectrum(row.FwdSC, tol[row.TypeName]))
		out.add(classifySpectrum(row.BwdSC, tol[row.TypeName]))
		out.add(classifySpectrum(row.FwdMC, tol[row.TypeName]))
		out.add(classifySpectrum(row.BwdMC, tol[row.TypeName]))
	}
	return out
}

func summarizeTraining(rows []NativeTrainingRow) spectrumSummary {
	var out spectrumSummary
	for _, row := range rows {
		if row.Skipped {
			continue
		}
		if row.Err != nil {
			out.add(specFatal)
			continue
		}
		if row.Learned {
			out.add(specExact)
		} else {
			out.add(specBroken)
		}
	}
	return out
}

func toleranceByType() map[string]float64 {
	out := make(map[string]float64, len(NativeMatrixAllCases))
	for _, tc := range NativeMatrixAllCases {
		out[tc.Name] = tc.Tolerance
	}
	return out
}

func printCapabilityAudit(rows []NativeCapabilityRow) {
	fmt.Println("Native capability audit (CNN1 exact path, not PTQ/QAT simulation)")
	fmt.Printf("| %-10s | %-8s | %-8s | %-8s | %-8s | %-36s |\n",
		"DType", "CPU-Exec", "CPU-Train", "GPU-Exec", "GPU-Train", "Notes")
	fmt.Println("|------------|----------|----------|----------|----------|--------------------------------------|")
	for _, row := range rows {
		fmt.Printf("| %-10s | %-8s | %-8s | %-8s | %-8s | %-36s |\n",
			row.TypeName, row.CPUExec, row.CPUTrain, row.GPUExec, row.GPUTrain, row.Notes)
	}
	fmt.Println()
}

func printCPUParity(rows []cpuParityRow) {
	fmt.Println("CPU native forward/backward parity (normal vs tiled variants)")
	fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s | %-20s | %-20s | %-20s | %-6s |\n",
		"DType", "Fwd-SC", "Fwd-MC", "Bwd-SC", "Bwd-MC", "SC-Tiles", "MC-Tiles", "Timing (N/SC/MC)", "Pass")
	fmt.Println("|------------|------------|------------|------------|------------|----------------------|----------------------|----------------------|--------|")
	for _, row := range rows {
		if row.Err != nil {
			fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s | %-20s | %-20s | %-20s | %-6s |\n",
				row.TypeName, "ERR", "ERR", "ERR", "ERR", "ERR", "ERR", "ERR", "ERR")
			continue
		}
		if row.Skipped {
			fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s | %-20s | %-20s | %-20s | %-6s |\n",
				row.TypeName, "SKIP", "SKIP", "SKIP", "SKIP", "SKIP", "SKIP", "SKIP", row.Reason)
			continue
		}
		timing := fmt.Sprintf("%-6s/%-6s/%-6s",
			compactDur(row.NormalDur),
			compactDelta(row.NormalDur, row.SCDur),
			compactDelta(row.NormalDur, row.MCDur))

		fmt.Printf("| %-10s | %-10.2e | %-10.2e | %-10.2e | %-10.2e | %-20s | %-20s | %-20s | %-6s |\n",
			row.TypeName, row.FwdSC, row.FwdMC, row.BwdSC, row.BwdMC,
			row.SCTiles, row.MCTiles, timing, mark(row.Pass))
	}
	fmt.Println()
}

func printGPUParity(rows []NativeGPUParityRow) {
	fmt.Println("GPU native forward/backward parity (vs CPU normal)")
	fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-24s | %-24s | %-6s |\n",
		"DType", "Fwd-GN", "Fwd-GSC", "Fwd-GMC", "Bwd-GN", "Bwd-GSC", "Bwd-GMC", "Tiles (GSC/GMC)", "Timing (GN/GSC/GMC)", "Pass")
	fmt.Println("|------------|------------|------------|------------|------------|------------|------------|--------------------------|--------------------------|--------|")
	for _, row := range rows {
		if row.Err != nil {
			fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-24s | %-24s | %-6s |\n",
				row.TypeName, "ERR", "ERR", "ERR", "ERR", "ERR", "ERR", "ERR", "ERR", "ERR")
			continue
		}
		if row.Skipped {
			fmt.Printf("| %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-24s | %-24s | %-6s |\n",
				row.TypeName, "SKIP", "SKIP", "SKIP", "SKIP", "SKIP", "SKIP", "SKIP", "SKIP", row.Reason)
			continue
		}
		tiles := fmt.Sprintf("%-11s/%-11s", row.GSCTiles, row.GMCTiles)
		timing := fmt.Sprintf("%-6s/%-6s/%-6s",
			compactDur(row.GNDur),
			compactDelta(row.GNDur, row.GSCDur),
			compactDelta(row.GNDur, row.GMCDur))

		fmt.Printf("| %-10s | %-10.2e | %-10.2e | %-10.2e | %-10.2e | %-10.2e | %-10.2e | %-24s | %-24s | %-6s |\n",
			row.TypeName, row.FwdGN, row.FwdSC, row.FwdMC, row.BwdGN, row.BwdSC, row.BwdMC,
			tiles, timing, mark(row.Pass))
	}
	fmt.Println()
}

func printTrainingMatrix(rows []NativeTrainingRow) {
	fmt.Println("Native learning matrix (one-batch epochs, simulation skipped)")
	fmt.Printf("| %-10s | %-13s | %-8s | %-8s | %-10s | %-10s | %-8s | %-12s | %-6s |\n",
		"DType", "Mode", "Acc[0]", "Acc[N]", "Loss[0]", "Loss[N]", "Time", "Tiles", "Learn")
	fmt.Println("|------------|---------------|----------|----------|------------|------------|----------|--------------|--------|")
	for _, row := range rows {
		if row.Err != nil {
			fmt.Printf("| %-10s | %-13s | %-8s | %-8s | %-10s | %-10s | %-8s | %-12s | %-6s |\n",
				row.TypeName, row.Mode.String(), "ERR", "ERR", "ERR", "ERR", "ERR", "ERR", row.Err.Error())
			continue
		}
		if row.Skipped {
			fmt.Printf("| %-10s | %-13s | %-8s | %-8s | %-10s | %-10s | %-8s | %-12s | %-6s |\n",
				row.TypeName, row.Mode.String(), "SKIP", "SKIP", "SKIP", "SKIP", "SKIP", "SKIP", row.Reason)
			continue
		}
		fmt.Printf("| %-10s | %-13s | %7.1f%% | %7.1f%% | %-10.4e | %-10.4e | %-8s | %-12s | %-6s |\n",
			row.TypeName, row.Mode.String(), row.BeforeAcc, row.AfterAcc,
			row.BeforeLoss, row.AfterLoss, row.Duration.Round(time.Millisecond), row.TilePlan, mark(row.Learned))
	}
	fmt.Println()
}

func compactDur(d time.Duration) string {
	if d == 0 {
		return "0s"
	}
	if d < time.Millisecond {
		return fmt.Sprintf("%dµs", d.Microseconds())
	}
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	return fmt.Sprintf("%.1fs", d.Seconds())
}

func compactDelta(normal, tiled time.Duration) string {
	delta := tiled - normal
	if delta == 0 {
		return "0s"
	}
	sign := "+"
	if delta < 0 {
		sign = "-"
		delta = -delta
	}
	return sign + compactDur(delta)
}

func printPerformanceMatrix(rows []NativePerfRow, benchIters int, cpuPlan, gpuPlan NativeMatrixBenchmarkPlan, gpuStress bool) {
	fmt.Printf("Native performance matrix (avg per iter over %d runs; HostRAM=CPU weight footprint, VRAM=live GPU buffers)\n", benchIters)
	fmt.Printf("CPU plan: profile=%s %s\n", cpuPlan.ProfileName, cpuPlan.ShapeSummary)
	if cpuPlan.Note != "" {
		fmt.Printf("CPU benchmark target note: %s\n", cpuPlan.Note)
	}
	if gpuStress {
		fmt.Printf("GPU plan: profile=%s %s\n", gpuPlan.ProfileName, gpuPlan.ShapeSummary)
		if gpuPlan.Note != "" {
			fmt.Printf("GPU benchmark target note: %s\n", gpuPlan.Note)
		}
		fmt.Println("CPU and GPU rows below use separate benchmark plans; compare gains within the same device.")
	} else {
		fmt.Println("CPU and GPU rows below share the same benchmark plan.")
	}
	fmt.Println("GPU timings use timestamp-query measurements when the adapter exposes them; otherwise they fall back to wall time.")
	fmt.Printf("| %-10s | %-13s | %-8s | %-8s | %-8s | %-9s | %-9s | %-9s | %-12s |\n",
		"DType", "Mode", "Forward", "Backward", "Total", "Samples/s", "HostRAM", "VRAM", "Tiles")
	fmt.Println("|------------|---------------|----------|----------|----------|-----------|-----------|-----------|--------------|")
	for _, row := range rows {
		if row.Err != nil {
			fmt.Printf("| %-10s | %-13s | %-8s | %-8s | %-8s | %-9s | %-9s | %-9s | %-12s |\n",
				row.TypeName, row.Mode.String(), "ERR", "ERR", "ERR", "ERR", "ERR", row.Err.Error(), "ERR")
			continue
		}
		if row.Skipped {
			fmt.Printf("| %-10s | %-13s | %-8s | %-8s | %-8s | %-9s | %-9s | %-9s | %-12s |\n",
				row.TypeName, row.Mode.String(), "SKIP", "SKIP", "SKIP", "SKIP", "SKIP", row.Reason, "SKIP")
			continue
		}
		fmt.Printf("| %-10s | %-13s | %-8s | %-8s | %-8s | %-9.1f | %-9s | %-9s | %-12s |\n",
			row.TypeName,
			row.Mode.String(),
			row.Forward.Round(time.Microsecond),
			row.Backward.Round(time.Microsecond),
			row.Total.Round(time.Microsecond),
			row.SamplesPerSec,
			humanBytes(row.HostRAMBytes),
			humanBytes(row.VRAMBytes),
			row.TilePlan,
		)
	}
	fmt.Println()
}

func printTilingSpeedupSummary(rows []NativePerfRow, gpuStress bool) {
	groups := groupPerfRows(rows)
	if gpuStress {
		fmt.Println("Tiling performance race (CPU and GPU use separate benchmark plans here; compare winners within the same device)")
	} else {
		fmt.Println("Tiling performance race (actual totals, x gains vs same-device normal, and memory footprint)")
	}
	fmt.Printf("| %-10s | %-6s | %-10s | %-10s | %-10s | %-6s | %-6s | %-22s | %-22s | %-8s |\n",
		"DType", "Device", "Normal T", "SC T", "MC T", "SCx", "MCx", "RAM N/SC/MC", "VRAM N/SC/MC", "Winner")
	fmt.Println("|------------|--------|------------|------------|------------|--------|--------|------------------------|------------------------|----------|")
	for _, tc := range NativeMatrixAllCases {
		for _, device := range []string{"CPU", "GPU"} {
			key := tc.Name + "|" + device
			group := groups[key]
			if group == nil || group.normal == nil || group.normal.Skipped || group.normal.Err != nil {
				continue
			}

			scTime, mcTime := "SKIP", "SKIP"
			scMul, mcMul := "SKIP", "SKIP"
			winner := "Normal"
			best := group.normal.Total
			if group.sc != nil && !group.sc.Skipped && group.sc.Err == nil {
				scTime = compactDur(group.sc.Total)
				scMul = fmt.Sprintf("%.2fx", speedup(group.normal.Total, group.sc.Total))
				if group.sc.Total < best {
					best = group.sc.Total
					winner = "SC"
				}
			}
			if group.mc != nil && !group.mc.Skipped && group.mc.Err == nil {
				mcTime = compactDur(group.mc.Total)
				mcMul = fmt.Sprintf("%.2fx", speedup(group.normal.Total, group.mc.Total))
				if group.mc.Total < best {
					best = group.mc.Total
					winner = "MC"
				}
			}

			fmt.Printf("| %-10s | %-6s | %-10s | %-10s | %-10s | %-6s | %-6s | %-22s | %-22s | %-8s |\n",
				tc.Name,
				device,
				compactDur(group.normal.Total),
				scTime,
				mcTime,
				scMul,
				mcMul,
				memoryTriplet(group.normal, group.sc, group.mc, false),
				memoryTriplet(group.normal, group.sc, group.mc, true),
				winner,
			)
		}
	}
	fmt.Println()

	fmt.Println("Tiling timing delta summary (vs same-device normal mode; negative deltas are faster)")
	fmt.Printf("| %-10s | %-6s | %-20s | %-26s | %-26s | %-6s | %-6s | %-8s |\n",
		"DType", "Device", "Normal F/B/T", "SC ΔF/ΔB/ΔT", "MC ΔF/ΔB/ΔT", "SCx", "MCx", "Winner")
	fmt.Println("|------------|--------|----------------------|----------------------------|----------------------------|--------|--------|----------|")
	for _, tc := range NativeMatrixAllCases {
		for _, device := range []string{"CPU", "GPU"} {
			key := tc.Name + "|" + device
			group := groups[key]
			if group == nil || group.normal == nil || group.normal.Skipped || group.normal.Err != nil {
				continue
			}
			scDelta, mcDelta := "SKIP", "SKIP"
			scMul, mcMul := "SKIP", "SKIP"
			winner := "Normal"
			best := group.normal.Total
			if group.sc != nil && !group.sc.Skipped && group.sc.Err == nil {
				scDelta = deltaTriplet(group.normal, group.sc)
				scMul = fmt.Sprintf("%.2fx", speedup(group.normal.Total, group.sc.Total))
				if group.sc.Total < best {
					best = group.sc.Total
					winner = "SC"
				}
			}
			if group.mc != nil && !group.mc.Skipped && group.mc.Err == nil {
				mcDelta = deltaTriplet(group.normal, group.mc)
				mcMul = fmt.Sprintf("%.2fx", speedup(group.normal.Total, group.mc.Total))
				if group.mc.Total < best {
					best = group.mc.Total
					winner = "MC"
				}
			}
			fmt.Printf("| %-10s | %-6s | %-20s | %-26s | %-26s | %-6s | %-6s | %-8s |\n",
				tc.Name,
				device,
				timingTriplet(group.normal),
				scDelta,
				mcDelta,
				scMul,
				mcMul,
				winner,
			)
		}
	}
	fmt.Println()
}

func groupPerfRows(rows []NativePerfRow) map[string]*speedupGroup {
	groups := make(map[string]*speedupGroup)
	for i := range rows {
		row := &rows[i]
		device := perfDevice(row.Mode)
		key := row.TypeName + "|" + device
		if _, ok := groups[key]; !ok {
			groups[key] = &speedupGroup{}
		}
		switch row.Mode {
		case TrainingModeCPUNormal, TrainingModeGPUNormal:
			groups[key].normal = row
		case TrainingModeCPUSC, TrainingModeGPUSC:
			groups[key].sc = row
		case TrainingModeCPUMC, TrainingModeGPUMC:
			groups[key].mc = row
		}
	}
	return groups
}

func memoryTriplet(normal, sc, mc *NativePerfRow, vram bool) string {
	return fmt.Sprintf("%s/%s/%s",
		rowMemory(normal, vram),
		rowMemory(sc, vram),
		rowMemory(mc, vram),
	)
}

func rowMemory(row *NativePerfRow, vram bool) string {
	if row == nil || row.Skipped || row.Err != nil {
		return "SKIP"
	}
	if vram {
		return humanBytes(row.VRAMBytes)
	}
	return humanBytes(row.HostRAMBytes)
}

func timingTriplet(row *NativePerfRow) string {
	if row == nil {
		return "SKIP"
	}
	return fmt.Sprintf("%s/%s/%s",
		row.Forward.Round(time.Microsecond),
		row.Backward.Round(time.Microsecond),
		row.Total.Round(time.Microsecond),
	)
}

func deltaTriplet(normal, tiled *NativePerfRow) string {
	if normal == nil || tiled == nil {
		return "SKIP"
	}
	return fmt.Sprintf("%s/%s/%s",
		signedDuration(tiled.Forward-normal.Forward),
		signedDuration(tiled.Backward-normal.Backward),
		signedDuration(tiled.Total-normal.Total),
	)
}

func signedDuration(delta time.Duration) string {
	if delta == 0 {
		return "0s"
	}
	if delta > 0 {
		return "+" + delta.Round(time.Microsecond).String()
	}
	return "-" + (-delta).Round(time.Microsecond).String()
}

func speedup(normal, tiled time.Duration) float64 {
	if normal <= 0 || tiled <= 0 {
		return 0
	}
	return float64(normal) / float64(tiled)
}

func perfDevice(mode TrainingMode) string {
	if mode.IsGPU() {
		return "GPU"
	}
	return "CPU"
}

func tilePlanForMode(net *VolumetricNetwork, mode TrainingMode, dtype DType) string {
	if mode == TrainingModeCPUNormal || mode == TrainingModeGPUNormal {
		return "-"
	}
	if net == nil {
		tmp, err := buildDefaultNetwork(dtype)
		if err != nil {
			return "ERR-BUILD"
		}
		net = tmp
		defer func() {
			if net.GPUContext != nil {
				net.DestroyWGPU()
			}
		}()
	}
	parts := make([]string, 0, len(net.Layers))
	for i := range net.Layers {
		layer := &net.Layers[i]
		tile := 0
		switch mode {
		case TrainingModeCPUSC, TrainingModeCPUMC:
			tile = layer.GetCPUTileSize(dtype)
		case TrainingModeGPUSC:
			tile = layer.GetGPUSCTileSize(dtype)
		case TrainingModeGPUMC:
			tile = layer.GetGPUMCTileSize(dtype)
		}
		parts = append(parts, fmt.Sprintf("L%d:%d", i, tile))
	}
	return strings.Join(parts, " ")
}

func humanBytes(bytes int64) string {
	if bytes <= 0 {
		return "0 B"
	}
	units := []string{"B", "KiB", "MiB", "GiB"}
	value := float64(bytes)
	unit := 0
	for value >= 1024 && unit < len(units)-1 {
		value /= 1024
		unit++
	}
	if unit == 0 {
		return fmt.Sprintf("%d %s", bytes, units[unit])
	}
	return fmt.Sprintf("%.1f %s", value, units[unit])
}
