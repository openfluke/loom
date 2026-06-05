package main

import (
	"bufio"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/openfluke/loom/poly"
)

// lucyHFModelSpec is one SoulGlitch-style approved model (manual file list + hub layout).
type lucyHFModelSpec struct {
	Repo  string
	Files []string
}

// lucyApprovedHFSpecs mirrors SoulGlitch `model_download_screen.dart` _approvedModels:
// same repo IDs and requiredFiles lists; same URL pattern and hub layout (snapshots/manual-download + refs/main).
var lucyApprovedHFSpecs = []lucyHFModelSpec{
	{
		Repo: "microsoft/bitnet-b1.58-2B-4T",
		Files: []string{
			"config.json",
			"generation_config.json",
			"model.safetensors",
			"special_tokens_map.json",
			"tokenizer.json",
			"tokenizer_config.json",
		},
	},
	{
		Repo: "Qwen/Qwen3-0.6B",
		Files: []string{
			"config.json",
			"generation_config.json",
			"merges.txt",
			"model.safetensors",
			"tokenizer.json",
			"tokenizer_config.json",
			"vocab.json",
		},
	},
	{
		Repo: "Qwen/Qwen3-1.7B",
		Files: []string{
			"config.json",
			"generation_config.json",
			"merges.txt",
			"model-00001-of-00002.safetensors",
			"model-00002-of-00002.safetensors",
			"model.safetensors.index.json",
			"tokenizer.json",
			"tokenizer_config.json",
			"vocab.json",
		},
	},
	{
		Repo: "Qwen/Qwen3-4B",
		Files: []string{
			"config.json",
			"generation_config.json",
			"merges.txt",
			"model-00001-of-00003.safetensors",
			"model-00002-of-00003.safetensors",
			"model-00003-of-00003.safetensors",
			"model.safetensors.index.json",
			"tokenizer.json",
			"tokenizer_config.json",
			"vocab.json",
		},
	},
	{
		Repo: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
		Files: []string{
			"config.json",
			"generation_config.json",
			"model.safetensors",
			"special_tokens_map.json",
			"tokenizer.json",
			"tokenizer_config.json",
		},
	},
	{
		Repo: "HuggingFaceTB/SmolLM2-135M-Instruct",
		Files: []string{
			"config.json",
			"generation_config.json",
			"model.safetensors",
			"special_tokens_map.json",
			"tokenizer.json",
			"tokenizer_config.json",
		},
	},
	{
		Repo: "HuggingFaceTB/SmolLM2-360M-Instruct",
		Files: []string{
			"config.json",
			"generation_config.json",
			"model.safetensors",
			"special_tokens_map.json",
			"tokenizer.json",
			"tokenizer_config.json",
		},
	},
}

func hfResolveURL(repoID, fileName string) string {
	return fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repoID, fileName)
}

func hfHubRepoDir(hubRoot, repoID string) string {
	return filepath.Join(hubRoot, "models--"+strings.ReplaceAll(repoID, "/", "--"))
}

func hfManualSnapshotDir(hubRoot, repoID string) string {
	return filepath.Join(hfHubRepoDir(hubRoot, repoID), "snapshots", poly.HFManualSnapshotDirName)
}

func hfDefaultHubRoot() (string, error) {
	cands, err := poly.HFHubCandidateDirs()
	if err != nil {
		return "", err
	}
	if len(cands) == 0 {
		return "", fmt.Errorf("no Hugging Face hub candidate dirs")
	}
	root := cands[0]
	if err := os.MkdirAll(root, 0o755); err != nil {
		return "", err
	}
	return root, nil
}

func downloadOneFile(client *http.Client, url, destPath string) error {
	if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
		return err
	}
	part := destPath + ".part"
	_ = os.Remove(part)

	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	// Match SoulGlitch HttpClient userAgent for CDN / mirror behavior.
	req.Header.Set("User-Agent", "SoulGlitch/1.0")
	if tok := strings.TrimSpace(os.Getenv("HUGGING_FACE_HUB_TOKEN")); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d %s", resp.StatusCode, url)
	}

	out, err := os.Create(part)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(out, resp.Body)
	closeErr := out.Close()
	if copyErr != nil {
		_ = os.Remove(part)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(part)
		return closeErr
	}
	fi, statErr := os.Stat(part)
	if statErr != nil || fi.Size() == 0 {
		_ = os.Remove(part)
		return fmt.Errorf("empty or missing download: %s", url)
	}
	_ = os.Remove(destPath)
	if err := os.Rename(part, destPath); err != nil {
		return err
	}
	return nil
}

func writeRefsMainLikeSoulGlitch(hubRoot, repoID string) error {
	refsDir := filepath.Join(hfHubRepoDir(hubRoot, repoID), "refs")
	if err := os.MkdirAll(refsDir, 0o755); err != nil {
		return err
	}
	// SoulGlitch: refs/main contains the snapshot folder name (manual-download).
	return os.WriteFile(filepath.Join(refsDir, "main"), []byte(poly.HFManualSnapshotDirName), 0o644)
}

func downloadApprovedModelManual(client *http.Client, hubRoot string, spec lucyHFModelSpec) error {
	snapDir := hfManualSnapshotDir(hubRoot, spec.Repo)
	if err := os.MkdirAll(snapDir, 0o755); err != nil {
		return err
	}

	n := len(spec.Files)
	for i, name := range spec.Files {
		dest := filepath.Join(snapDir, name)
		if st, err := os.Stat(dest); err == nil && st.Size() > 0 {
			fmt.Printf("   [%d/%d] skip (already have) %s\n", i+1, n, name)
			continue
		}
		url := hfResolveURL(spec.Repo, name)
		fmt.Printf("   [%d/%d] downloading %s …\n", i+1, n, name)
		t0 := time.Now()
		if err := downloadOneFile(client, url, dest); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		fmt.Printf("        done %s (%v)\n", name, time.Since(t0).Round(time.Millisecond))
	}

	if err := writeRefsMainLikeSoulGlitch(hubRoot, spec.Repo); err != nil {
		return err
	}
	return nil
}

// runApprovedHFModelsDownload mirrors SoulGlitch manual downloads: HTTPS resolve/main → hub/.../snapshots/manual-download.
func runApprovedHFModelsDownload(reader *bufio.Reader) {
	hubRoot, err := hfDefaultHubRoot()
	if err != nil {
		fmt.Printf("❌ %v\n", err)
		return
	}

	client := &http.Client{Timeout: 0} // large safetensors; no whole-response deadline
	if os.Getenv("HUGGING_FACE_HUB_TOKEN") == "" {
		fmt.Println("   (Set HUGGING_FACE_HUB_TOKEN if a repo returns 401.)")
	}

	fmt.Println("\n📥 Download approved Hugging Face models (SoulGlitch-style manual HTTP)")
	fmt.Printf("   Hub root: %s\n", hubRoot)
	fmt.Printf("   Snapshot: …/snapshots/%s/  (same as poly.HFResolveSnapshotDirPreferManual)\n", poly.HFManualSnapshotDirName)

	fmt.Println("\n  [0] Download all (sequential)")
	for i, s := range lucyApprovedHFSpecs {
		fmt.Printf("  [%d] %s  (%d files)\n", i+1, s.Repo, len(s.Files))
	}
	choice := strings.TrimSpace(readInput(reader, "Choice [0]: ", "0"))

	var pick []lucyHFModelSpec
	switch choice {
	case "0":
		pick = append(pick, lucyApprovedHFSpecs...)
	default:
		idx := 0
		if _, err := fmt.Sscanf(choice, "%d", &idx); err != nil || idx < 1 || idx > len(lucyApprovedHFSpecs) {
			fmt.Println("Invalid selection.")
			return
		}
		pick = append(pick, lucyApprovedHFSpecs[idx-1])
	}

	for _, spec := range pick {
		fmt.Printf("\n── %s ──\n", spec.Repo)
		if err := downloadApprovedModelManual(client, hubRoot, spec); err != nil {
			fmt.Printf("❌ %v\n", err)
			continue
		}
		fmt.Printf("✅ Finished → %s\n", hfManualSnapshotDir(hubRoot, spec.Repo))
	}
	fmt.Println("\nDone. Run [1] Poly Talk or [8] ENTITY Talk — models should appear under the scanned hub root.")
}
