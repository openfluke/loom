package contextsuite

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/openfluke/loom/poly"
)

const (
	entitiesDir = "lucy_entities"
	maxSeqLen   = 512
)

// ModelSpec is one target checkpoint for the context suite.
type ModelSpec struct {
	RepoID    string
	ShortName string
}

// TargetModels are the three models requested for long-context / multi-prompt testing.
var TargetModels = []ModelSpec{
	{RepoID: "microsoft/bitnet-b1.58-2B-4T", ShortName: "bitnet"},
	{RepoID: "Qwen/Qwen3-0.6B", ShortName: "qwen3-0.6b"},
	{RepoID: "HuggingFaceTB/SmolLM2-135M-Instruct", ShortName: "smol2-135m"},
}

func entityPath(modelID string) string {
	name := strings.ReplaceAll(modelID, "/", "--") + ".entity"
	return filepath.Join(entitiesDir, name)
}

// ExecProfile is one CPU/GPU × single-core/multi-core execution mode.
type ExecProfile struct {
	Name      string
	UseGPU    bool
	MultiCore bool
}

// ExecProfiles is the full execution matrix (GPU cells skip gracefully when unavailable).
var ExecProfiles = []ExecProfile{
	{Name: "cpu_sc", UseGPU: false, MultiCore: false},
	{Name: "cpu_mc", UseGPU: false, MultiCore: true},
	{Name: "gpu_sc", UseGPU: true, MultiCore: false},
	{Name: "gpu_mc", UseGPU: true, MultiCore: true},
}

func defaultSystemPrompt(modelID string) string {
	name := strings.ToLower(modelID)
	if strings.Contains(name, "bitnet") || strings.Contains(name, "1bit") {
		return ""
	}
	if strings.Contains(name, "qwen") {
		return "You are a helpful assistant. Respond directly with the final answer only. Do not expose internal reasoning or chain-of-thought."
	}
	if strings.Contains(name, "instruct") || strings.Contains(name, "smollm") {
		return "You are a helpful assistant. Be concise, coherent, and avoid repetition."
	}
	return ""
}

func isBitNetModel(modelID string) bool {
	name := strings.ToLower(modelID)
	return strings.Contains(name, "bitnet") || strings.Contains(name, "1bit")
}

func resolveSnapshotDir(modelID string) (string, error) {
	hubDir, models, err := poly.HFInventoryMergedModels()
	if err != nil {
		return "", err
	}
	found := false
	for _, m := range models {
		if m == modelID {
			found = true
			break
		}
	}
	if !found {
		return "", fmt.Errorf("model %q not in HF cache at %s", modelID, hubDir)
	}
	snap, err := poly.HFResolveSnapshotDirPreferManual(hubDir, modelID)
	if err != nil {
		return "", fmt.Errorf("resolve snapshot: %w", err)
	}
	return snap, nil
}

func entityExists(modelID string) bool {
	st, err := os.Stat(entityPath(modelID))
	return err == nil && st.Size() > 0
}

func availableModels() []ModelSpec {
	out := make([]ModelSpec, 0, len(TargetModels))
	for _, m := range TargetModels {
		if entityExists(m.RepoID) {
			out = append(out, m)
		}
	}
	return out
}

func maxGenTokens(modelID string) int {
	if isBitNetModel(modelID) {
		return 64
	}
	return 48
}
