package poly

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// HFManualSnapshotDirName is preferred under snapshots/ when present (SoulGlitch manual downloads).
const HFManualSnapshotDirName = "manual-download"

// HFHubCandidateDirs returns Hugging Face hub roots to scan on desktop / SoulGlitch layouts.
func HFHubCandidateDirs() ([]string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	return []string{
		filepath.Join(homeDir, ".cache", "huggingface", "hub"),
		filepath.Join(homeDir, "Documents", "huggingface", "hub"),
		filepath.Join(homeDir, "Library", "Containers", "com.openfluke.soulglitch", "Data", ".cache", "huggingface", "hub"),
	}, nil
}

// HFInventoryMergedModels scans candidate hub dirs, merges unique model IDs, and picks a
// preferred hub root (first non-empty listing wins over the default cache path), matching
// loom/lucy and loom/glitch behavior.
func HFInventoryMergedModels() (preferredHub string, models []string, err error) {
	candidateHubDirs, err := HFHubCandidateDirs()
	if err != nil {
		return "", nil, err
	}
	defaultHubDir := candidateHubDirs[0]

	entriesByHub := map[string][]os.DirEntry{}
	for _, dir := range candidateHubDirs {
		entries, readErr := os.ReadDir(dir)
		if readErr == nil {
			entriesByHub[dir] = entries
		}
	}

	if len(entriesByHub) == 0 {
		if mkErr := os.MkdirAll(defaultHubDir, 0o755); mkErr != nil {
			return "", nil, mkErr
		}
		entriesByHub[defaultHubDir] = []os.DirEntry{}
	}

	preferredHub = defaultHubDir
	seenModels := map[string]struct{}{}
	for _, dir := range candidateHubDirs {
		entries, ok := entriesByHub[dir]
		if !ok {
			continue
		}
		localCount := 0
		for _, entry := range entries {
			if entry.IsDir() && strings.HasPrefix(entry.Name(), "models--") {
				modelName := strings.TrimPrefix(entry.Name(), "models--")
				modelName = strings.Replace(modelName, "--", "/", 1)
				if _, seen := seenModels[modelName]; !seen {
					seenModels[modelName] = struct{}{}
					models = append(models, modelName)
					localCount++
				}
			}
		}
		if localCount > 0 && preferredHub == defaultHubDir {
			preferredHub = dir
		}
	}
	return preferredHub, models, nil
}

// HFResolveSnapshotDir finds a snapshot folder for modelID, starting at preferredHubRoot
// then falling back across HFHubCandidateDirs. Uses first directory entry from ReadDir
// under snapshots/ for parity with legacy CLI tools (undefined order).
func HFResolveSnapshotDir(preferredHubRoot, modelID string) (string, error) {
	modelDir := filepath.Join(preferredHubRoot, "models--"+strings.ReplaceAll(modelID, "/", "--"), "snapshots")
	snaps, err := os.ReadDir(modelDir)
	if err == nil && len(snaps) > 0 {
		return filepath.Join(modelDir, snaps[0].Name()), nil
	}

	candidates, err := HFHubCandidateDirs()
	if err != nil {
		return "", fmt.Errorf("no snapshots found for model %s", modelID)
	}
	for _, candidate := range candidates {
		modelDir = filepath.Join(candidate, "models--"+strings.ReplaceAll(modelID, "/", "--"), "snapshots")
		snaps, err = os.ReadDir(modelDir)
		if err == nil && len(snaps) > 0 {
			return filepath.Join(modelDir, snaps[0].Name()), nil
		}
	}
	return "", fmt.Errorf("no snapshots found for model %s", modelID)
}

// HFResolveSnapshotDirPreferManual picks manual-download when present, otherwise the
// lexicographically first snapshot subfolder (SoulGlitch listInstalledModels parity).
func HFResolveSnapshotDirPreferManual(hubRoot, modelID string) (string, error) {
	snapshotsRoot := filepath.Join(hubRoot, "models--"+strings.ReplaceAll(modelID, "/", "--"), "snapshots")
	manual := filepath.Join(snapshotsRoot, HFManualSnapshotDirName)
	if st, err := os.Stat(manual); err == nil && st.IsDir() {
		return manual, nil
	}
	entries, err := os.ReadDir(snapshotsRoot)
	if err != nil {
		return "", err
	}
	var dirs []string
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dirs = append(dirs, e.Name())
	}
	if len(dirs) == 0 {
		return "", fmt.Errorf("no snapshots under %s", snapshotsRoot)
	}
	sort.Strings(dirs)
	return filepath.Join(snapshotsRoot, dirs[0]), nil
}

// HFInstalledModel pairs an HF model id with an absolute snapshot directory.
type HFInstalledModel struct {
	ModelID     string `json:"id"`
	SnapshotDir string `json:"snapshot_dir"`
}

// HFListInstalledModels lists models under a single hub root with resolved snapshot paths.
func HFListInstalledModels(hubRoot string) ([]HFInstalledModel, error) {
	entries, err := os.ReadDir(hubRoot)
	if err != nil {
		return nil, err
	}
	var out []HFInstalledModel
	for _, e := range entries {
		if !e.IsDir() || !strings.HasPrefix(e.Name(), "models--") {
			continue
		}
		modelName := strings.TrimPrefix(e.Name(), "models--")
		modelName = strings.Replace(modelName, "--", "/", 1)
		snap, err := HFResolveSnapshotDirPreferManual(hubRoot, modelName)
		if err != nil {
			continue
		}
		out = append(out, HFInstalledModel{ModelID: modelName, SnapshotDir: snap})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ModelID < out[j].ModelID })
	return out, nil
}
