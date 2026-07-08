package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/openfluke/loom/poly"
)

const lucyEntitiesDir = "lucy_entities"

type entityCatalogEntry struct {
	ModelID            string
	SnapshotDir        string
	Supported          bool
	Reason             string
	EntityPath         string
	EntityExists       bool
	EntityBytes        int64
	HFSafetensorsBytes int64
	StoredDType        poly.DType
	Converted          bool
}

func lucyEntityPath(modelID string) string {
	name := strings.ReplaceAll(modelID, "/", "--") + ".entity"
	return filepath.Join(lucyEntitiesDir, name)
}

func entityDTypeLabel(dt poly.DType) string {
	switch dt {
	case poly.DTypeInt4:
		return "Q4 (INT4)"
	case poly.DTypeInt8:
		return "INT8"
	case poly.DTypeTernary:
		return "BitNet (TERNARY)"
	case poly.DTypeFloat32:
		return "FP32"
	default:
		if dt == 0 {
			return "FP32"
		}
		return dt.String()
	}
}

func isBitNetEntityModel(modelID string) bool {
	name := strings.ToLower(modelID)
	return strings.Contains(name, "bitnet") || strings.Contains(name, "1bit")
}

func parseConvertDTypeInput(s string) poly.DType {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "32", "fp32", "f32":
		return poly.DTypeFloat32
	case "8", "int8":
		return poly.DTypeInt8
	case "bitnet", "ternary", "t", "b1.58", "158":
		return poly.DTypeTernary
	default:
		return poly.DTypeInt4
	}
}

func formatBytes(n int64) string {
	if n <= 0 {
		return "—"
	}
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for v := n / unit; v >= unit; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGTPE"[exp])
}

func snapshotSafetensorsBytes(snapshotDir string) int64 {
	if snapshotDir == "" {
		return 0
	}
	files, _ := filepath.Glob(filepath.Join(snapshotDir, "*.safetensors"))
	var total int64
	for _, f := range files {
		if st, err := os.Stat(f); err == nil {
			total += st.Size()
		}
	}
	return total
}

func formatHFToEntitySize(hfBytes, entityBytes int64) string {
	if entityBytes <= 0 {
		if hfBytes > 0 {
			return formatBytes(hfBytes) + " → —"
		}
		return "—"
	}
	if hfBytes <= 0 {
		return formatBytes(entityBytes)
	}
	ratio := float64(entityBytes) / float64(hfBytes)
	if ratio <= 1.0 {
		saved := (1.0 - ratio) * 100
		return fmt.Sprintf("%s → %s (%.1f%% smaller)", formatBytes(hfBytes), formatBytes(entityBytes), saved)
	}
	larger := (ratio - 1.0) * 100
	return fmt.Sprintf("%s → %s (%.1f%% larger)", formatBytes(hfBytes), formatBytes(entityBytes), larger)
}

func classifyModelForEntity(modelID, snapshotDir string) (supported bool, reason string) {
	if snapshotDir == "" {
		return false, "not in HF cache"
	}
	configPath := filepath.Join(snapshotDir, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return false, "missing config.json"
	}
	var config map[string]interface{}
	if err := json.Unmarshal(configData, &config); err != nil {
		return false, "bad config.json"
	}
	return classifyModelForEntityConfig(modelID, config, snapshotDir)
}

func classifyModelForEntityConfig(modelID string, config map[string]interface{}, snapshotDir string) (supported bool, reason string) {
	if poly.IsHFBitNetCheckpoint(modelID, config) {
		return true, "bitnet-style"
	}
	if poly.DetectHFArchitecture(config) == poly.HFArchUnknown {
		return false, "unsupported arch"
	}
	safetensorFiles, _ := filepath.Glob(filepath.Join(snapshotDir, "*.safetensors"))
	if len(safetensorFiles) == 0 {
		return false, "no .safetensors"
	}
	if _, err := poly.ParseHFDecoderDims(config, safetensorFiles); err != nil {
		return false, "bad dims"
	}
	return true, "llama-style"
}

func refreshEntityFileStats(e *entityCatalogEntry) {
	if st, err := os.Stat(e.EntityPath); err == nil {
		e.EntityExists = true
		e.EntityBytes = st.Size()
		dt, err := poly.EntityTransformerWeightDType(e.EntityPath)
		if err == nil {
			e.StoredDType = dt
		}
	} else {
		e.EntityExists = false
		e.EntityBytes = 0
		e.StoredDType = 0
	}
}

func buildEntityCatalog(hubDir string, models []string) []entityCatalogEntry {
	if err := os.MkdirAll(lucyEntitiesDir, 0o755); err != nil {
		log.Fatalf("Could not create %s: %v", lucyEntitiesDir, err)
	}
	out := make([]entityCatalogEntry, 0, len(models))
	for _, modelID := range models {
		entry := entityCatalogEntry{
			ModelID:    modelID,
			EntityPath: lucyEntityPath(modelID),
		}
		snap, err := poly.HFResolveSnapshotDirPreferManual(hubDir, modelID)
		if err != nil {
			entry.Reason = "not in HF cache"
		} else {
			entry.SnapshotDir = snap
			entry.HFSafetensorsBytes = snapshotSafetensorsBytes(snap)
			entry.Supported, entry.Reason = classifyModelForEntity(modelID, snap)
		}
		refreshEntityFileStats(&entry)
		out = append(out, entry)
	}
	return out
}

func printEntityCatalog(title string, entries []entityCatalogEntry) {
	fmt.Println(title)
	fmt.Printf("  Folder: %s/  (decoder quant is baked into each .entity at convert time)\n\n", lucyEntitiesDir)
	fmt.Printf("  %-4s  %-34s  %-18s  %-8s  %s\n", "#", "Model", "Support", ".entity", "HF → .entity")
	fmt.Println("  " + strings.Repeat("─", 108))
	for i, e := range entries {
		entityCol := "missing"
		if e.EntityExists {
			entityCol = entityDTypeLabel(e.StoredDType)
		}
		if e.Converted {
			entityCol = "new " + entityCol
		}
		sizeCol := formatHFToEntitySize(e.HFSafetensorsBytes, e.EntityBytes)
		if !e.EntityExists && e.HFSafetensorsBytes > 0 {
			sizeCol = formatBytes(e.HFSafetensorsBytes) + " → —"
		}
		fmt.Printf("  [%2d]  %-34s  %-18s  %-8s  %s\n",
			i+1, truncateStr(e.ModelID, 34), truncateStr(e.Reason, 18), entityCol, sizeCol)
	}
	fmt.Println()
}

func truncateStr(s string, max int) string {
	if len(s) <= max {
		return s
	}
	if max <= 3 {
		return s[:max]
	}
	return s[:max-3] + "..."
}

func deleteAllEntityFiles() int {
	files, _ := filepath.Glob(filepath.Join(lucyEntitiesDir, "*.entity"))
	n := 0
	for _, f := range files {
		if err := os.Remove(f); err == nil {
			n++
		}
	}
	return n
}

func convertEntityEntry(e *entityCatalogEntry, dtype poly.DType, force bool) error {
	if !e.Supported || e.SnapshotDir == "" {
		return fmt.Errorf("not convertible")
	}
	if e.EntityExists && !force {
		return fmt.Errorf("already exists (use force reconvert)")
	}
	if e.EntityExists && force {
		_ = os.Remove(e.EntityPath)
		e.EntityExists = false
	}
	fmt.Printf("⏳ Converting %s → %s [%s] …\n", e.ModelID, e.EntityPath, entityDTypeLabel(dtype))
	var (
		res *poly.HFImportResult
		err error
	)
	if isBitNetEntityModel(e.ModelID) {
		res, err = poly.ImportHFBitNetCheckpointDir(e.SnapshotDir, e.ModelID)
		dtype = poly.DTypeTernary
	} else {
		res, err = poly.ImportHFCheckpointDir(e.SnapshotDir, poly.HFImportOptions{WeightDType: poly.DTypeFloat32})
	}
	if err != nil {
		return err
	}
	et := poly.NewEntityTransformer(
		res.Network,
		res.Architecture,
		res.Dims,
		res.Embeddings,
		res.LMHead,
		res.FinalNorm,
		res.HasFinalNorm,
	)
	et.WeightDType = dtype
	if err := poly.SaveEntityTransformer(e.EntityPath, et); err != nil {
		return err
	}
	refreshEntityFileStats(e)
	e.Converted = true
	fmt.Printf("   ✅ %s  %s\n", filepath.Base(e.EntityPath), formatHFToEntitySize(e.HFSafetensorsBytes, e.EntityBytes))
	return nil
}

func convertibleEntries(entries []entityCatalogEntry) []entityCatalogEntry {
	out := make([]entityCatalogEntry, 0)
	for _, e := range entries {
		if e.Supported && e.SnapshotDir != "" {
			out = append(out, e)
		}
	}
	return out
}

func promptEntityConversion(reader *bufio.Reader, entries []entityCatalogEntry) {
	convertible := convertibleEntries(entries)
	if len(convertible) == 0 {
		fmt.Println("ℹ️  No convertible models in HF cache (need llama-style + safetensors).")
		return
	}

	if readInput(reader, "🗑️  Delete ALL existing .entity files first? (1=yes / 0=no) [0]: ", "0") == "1" {
		n := deleteAllEntityFiles()
		fmt.Printf("   Removed %d file(s) from %s/\n", n, lucyEntitiesDir)
		for i := range entries {
			refreshEntityFileStats(&entries[i])
		}
	}

	fmt.Println("\n🔧 Convert HF → .entity (pick models, quant baked in at save):")
	fmt.Println("  [0] Skip — chat with existing .entity only")
	fmt.Println("  [a] Convert all missing (supported, in cache)")
	for i, e := range convertible {
		status := "missing"
		if e.EntityExists {
			status = entityDTypeLabel(e.StoredDType) + " on disk"
		}
		fmt.Printf("  [%d] %s  (%s)\n", i+1, e.ModelID, status)
	}
	fmt.Println("  Or comma-separated numbers (e.g. 1,3,5)")

	choice := strings.TrimSpace(readInput(reader, "Convert choice [0]: ", "0"))
	if choice == "0" || choice == "" {
		return
	}

	quantInput := readInput(reader, "💎 Save quant in .entity? (4=Q4 recommended / 32=FP32 / 8=INT8 native / bitnet=BitNet ternary) [4]: ", "4")
	convertDType := parseConvertDTypeInput(quantInput)
	force := readInput(reader, "♻️  Force reconvert (delete existing .entity for selected)? (1=yes / 0=no) [0]: ", "0") == "1"

	var picks []*entityCatalogEntry
	if strings.EqualFold(choice, "a") {
		for i := range entries {
			if entries[i].Supported && entries[i].SnapshotDir != "" {
				if !entries[i].EntityExists || force {
					picks = append(picks, &entries[i])
				}
			}
		}
	} else {
		for _, part := range strings.Split(choice, ",") {
			part = strings.TrimSpace(part)
			idx, err := strconv.Atoi(part)
			if err != nil || idx < 1 || idx > len(convertible) {
				fmt.Printf("   ⚠️  skipping invalid selection %q\n", part)
				continue
			}
			modelID := convertible[idx-1].ModelID
			for i := range entries {
				if entries[i].ModelID == modelID {
					picks = append(picks, &entries[i])
					break
				}
			}
		}
	}
	if len(picks) == 0 {
		fmt.Println("   Nothing selected.")
		return
	}
	for _, e := range picks {
		if err := convertEntityEntry(e, convertDType, force); err != nil {
			fmt.Printf("   ❌ %s: %v\n", e.ModelID, err)
		}
	}
}

func readEntityTalkLaunchOptions(reader *bufio.Reader, modelID string, storedDType poly.DType) polyTalkLaunch {
	poly.ResetMemoryHistoryRecording()
	cfg := polyTalkLaunch{
		weightDType: storedDType,
		useTiling:   true,
		tileSize:    -1,
	}

	detInput := readInput(reader, "🎯 Deterministic mode? (1=yes / 0=no) [0]: ", "0")
	cfg.deterministic = detInput == "1"
	deterministic = cfg.deterministic

	fmt.Printf("💎 Checkpoint quant: %s (read from .entity — GPU uses baked weights when available)\n", entityDTypeLabel(storedDType))

	fmt.Print("🎮 Enable GPU Acceleration? (1=yes / 0=no) [0]: ")
	input, _ := reader.ReadString('\n')
	cfg.useGPU = strings.TrimSpace(input) == "1"

	if !cfg.useGPU {
		if poly.Plan9SimdEnabled() {
			cfg.useSIMD = readInput(reader, "⚡ CPU SIMD forward? (Plan 9 AVX2/NEON — Dense/SwiGLU/MHA/CNN/RNN/LSTM) (1=on / 0=off) [0]: ", "0") == "1"
		} else {
			fmt.Println("⚡ CPU SIMD forward: not linked for this build/arch — using scalar tiled forward.")
		}
	}

	fmt.Println("\n🚀 Tiling mode:")
	fmt.Println("  [1] Tiled — GPU: single-workgroup; CPU: multi-core tiled")
	fmt.Println("  [2] Tiled — GPU: multi-workgroup; CPU: multi-core tiled")
	execModeInput := readInput(reader, "Choice [2]: ", "2")
	cfg.tilingMode, cfg.tileSize = parseLLMExecutionMode(execModeInput)

	if cfg.useGPU {
		cfg.sequentialGPULoad = readInput(reader, "📥 Block-by-block GPU upload? (1=yes / 0=no) [1]: ", "1") == "1"
	}
	cfg.measureMemoryLoad = promptMeasureMemoryDuringLoad(reader)

	applyModelSpecificLaunchOptions(reader, modelID, &cfg, storedDType)
	deterministic = cfg.deterministic
	weightDType = storedDType
	return cfg
}

func loadTokenizerFromSnapshot(snapshotDir string) error {
	tokenizerPath := filepath.Join(snapshotDir, "tokenizer.json")
	var err error
	tk, err = poly.LoadTokenizer(tokenizerPath)
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}
	configPath := filepath.Join(snapshotDir, "config.json")
	eosTokens = poly.LoadEOSTokenIDsFromConfigPath(configPath)
	eosTokens = mergeIntSets(eosTokens, loadEOSTokensFromJSON(filepath.Join(snapshotDir, "generation_config.json")))
	return nil
}

func runEntityTalkMode(reader *bufio.Reader) {
	hubDir, models, err := poly.HFInventoryMergedModels()
	if err != nil {
		log.Fatalf("Could not scan HuggingFace cache: %v", err)
	}
	if len(models) == 0 {
		log.Fatalf("No models found in HuggingFace cache at: %s", hubDir)
	}

	fmt.Println("\n📦 ENTITY Talk — HF cache → quant .entity → chat")
	fmt.Printf("   Hub: %s\n", hubDir)

	entries := buildEntityCatalog(hubDir, models)
	printEntityCatalog("\n📋 HF cache (before convert):", entries)

	promptEntityConversion(reader, entries)
	printEntityCatalog("\n📋 After convert:", entries)

	runnable := make([]entityCatalogEntry, 0)
	for _, e := range entries {
		if e.Supported && e.EntityExists {
			runnable = append(runnable, e)
		}
	}
	if len(runnable) == 0 {
		log.Fatalf("No .entity checkpoints. Pick models in the convert step (default quant Q4).")
	}

	fmt.Println("🎯 Runnable (.entity on disk — quant is already baked in):")
	for i, e := range runnable {
		fmt.Printf("  [%d] %s  [%s]\n", i+1, e.ModelID, entityDTypeLabel(e.StoredDType))
		fmt.Printf("      %s\n", formatHFToEntitySize(e.HFSafetensorsBytes, e.EntityBytes))
	}

	modelInput := readInput(reader, "\nSelect model to run: ", "1")
	selectedIdx := 0
	fmt.Sscanf(modelInput, "%d", &selectedIdx)
	if selectedIdx < 1 || selectedIdx > len(runnable) {
		log.Fatalf("Invalid model selection: %d", selectedIdx)
	}
	pick := runnable[selectedIdx-1]

	storedDType := pick.StoredDType
	if storedDType == 0 {
		storedDType = poly.DTypeFloat32
	}
	launch := readEntityTalkLaunchOptions(reader, pick.ModelID, storedDType)

	fmt.Printf("\n⏳ Loading %s from %s …\n", pick.ModelID, pick.EntityPath)
	fmt.Printf("   Weights: .entity (%s). Tokenizer: HF snapshot only.\n", entityDTypeLabel(storedDType))
	if err := loadTokenizerFromSnapshot(pick.SnapshotDir); err != nil {
		log.Fatalf("❌ %v", err)
	}

	if launch.measureMemoryLoad {
		session := "entity_cpu_load"
		if launch.useGPU {
			session = "entity_gpu_load"
		}
		beginMemoryHistorySession(session)
		fmt.Println("📈 Memory timeline recording — terminal chart prints when load finishes.")
		recordMemoryHistoryRuntime("before_entity_load")
	}

	ef, err := poly.OpenEntityFile(pick.EntityPath)
	if err != nil {
		log.Fatalf("❌ OpenEntityFile: %v", err)
	}
	defer ef.Close()

	et, err := ef.LoadEntityTransformerTopology()
	if err != nil {
		log.Fatalf("❌ LoadEntityTransformerTopology: %v", err)
	}
	if launch.measureMemoryLoad {
		recordMemoryHistoryRuntime("entity_topology_loaded")
	}
	if et.WeightDType != 0 {
		storedDType = et.WeightDType
	}
	poly.PrepareEntityTransformerInference(et)
	template := templateForModel(pick.ModelID)
	numLayers := et.Dims.NumLayers
	if numLayers <= 0 && et.Network != nil {
		numLayers = len(et.Network.Layers) / 4
	}
	hiddenSize := et.HiddenSize
	rmsNormEps := et.Dims.RMSNormEps
	vocabSize := et.VocabSize
	tr = poly.BuildTransformerFromEntity[float32](et, template)
	if launch.measureMemoryLoad {
		recordMemoryHistory("entity_cpu_weights_loaded")
	}

	if numLayers <= 0 {
		numLayers = len(tr.Network.Layers) / 4
	}
	isQwen := strings.Contains(strings.ToLower(pick.ModelID), "qwen")

	gpuWeightDType := poly.EntityGPUWeightDType(storedDType, launch.useGPU)
	if launch.useGPU && gpuWeightDType != storedDType {
		fmt.Printf("🎮 GPU upload: %s (checkpoint on disk: %s)\n",
			entityDTypeLabel(gpuWeightDType), entityDTypeLabel(storedDType))
	}

	infCfg := inferenceConfig{
		useGPU:            launch.useGPU,
		useTiling:         launch.useTiling,
		tilingMode:        launch.tilingMode,
		tileSize:          launch.tileSize,
		weightDType:       gpuWeightDType,
		sequentialGPULoad: launch.sequentialGPULoad,
		numLayers:         numLayers,
		hiddenSize:        hiddenSize,
		isQwen:            isQwen,
		rmsNormEps:        rmsNormEps,
		fromEntity:        true,
		entityFile:        ef,
		entityBundle:      et,
	}
	useGPU := setupTransformerForInference(tr, infCfg)
	et = nil
	poly.ReleaseInferenceTransientMemory()

	if !useGPU && tr.Network != nil {
		tr.Network.SetSimdForwardRecursive(launch.useSIMD)
		if launch.useSIMD {
			fmt.Println("⚡ CPU SIMD forward: ON (Plan 9 AVX2/NEON)")
		} else {
			fmt.Println("⚡ CPU SIMD forward: OFF (scalar tiled)")
		}
	}

	applyGlitchTanhiIfRequested(reader, tr.Network)

	fmt.Printf("\n✅ ENTITY loaded! (%d layers, vocab=%d, quant=%s)\n", numLayers, vocabSize, entityDTypeLabel(storedDType))
	fmt.Printf("   %s\n", formatHFToEntitySize(pick.HFSafetensorsBytes, pick.EntityBytes))
	printPostLoadMemorySnapshot(tr)
	if len(poly.TokenizerBannedSpecialExceptEOS(tk, eosTokens)) > 0 {
		fmt.Printf("🧯 Special-token mask active\n")
	}

	runPolyTalkChatSession(reader, pick.ModelID, numLayers, launch, useGPU)
}
