package ninelayer

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// BenchManifest mirrors chaosglue/npu/intel/example/bench_manifest.json.
type BenchManifest struct {
	DTypes       []string               `json:"dtypes"`
	SizeOrder    []string               `json:"size_order"`
	WarmupBySize map[string]int         `json:"warmup_by_size"`
	ItersBySize  map[string]int         `json:"iters_by_size"`
	Sizes        map[string]SizeProfile `json:"sizes"`
	Layers       []ManifestLayer        `json:"layers"`
}

type SizeProfile struct {
	Note    string      `json:"note"`
	Dense   DenseShape  `json:"dense"`
	Conv1D  Conv1DShape `json:"conv1d"`
	Conv2D  Conv2DShape `json:"conv2d"`
	Spatial SpatialShape `json:"spatial"`
}

type DenseShape struct {
	Batch int `json:"batch"`
	Dim   int `json:"dim"`
}

type Conv1DShape struct {
	Batch   int `json:"batch"`
	InC     int `json:"in_c"`
	Length  int `json:"length"`
	Filters int `json:"filters"`
	Kernel  int `json:"kernel"`
	Pad     int `json:"pad"`
}

type Conv2DShape struct {
	Batch   int `json:"batch"`
	InC     int `json:"in_c"`
	H       int `json:"h"`
	W       int `json:"w"`
	Filters int `json:"filters"`
	Kernel  int `json:"kernel"`
	Pad     int `json:"pad"`
}

type SpatialShape struct {
	Batch      int `json:"batch"`
	Channels   int `json:"channels"`
	H          int `json:"h"`
	W          int `json:"w"`
	Kernel     int `json:"kernel"`
	Pad        int `json:"pad"`
	PoolKS     int `json:"pool_ks"`
	PoolStride int `json:"pool_stride"`
}

type ManifestLayer struct {
	Name     string  `json:"name"`
	LoomType string  `json:"loom_type"`
	Loom     *string `json:"loom"`
}

func manifestPath() string {
	if v := os.Getenv("LOOM_BENCH_MANIFEST"); v != "" {
		return v
	}
	if root := os.Getenv("CHAOSGLUE_ROOT"); root != "" {
		return filepath.Join(root, "npu/intel/example/bench_manifest.json")
	}
	for _, rel := range []string{
		filepath.Join("..", "..", "npu", "intel", "example", "bench_manifest.json"),
		filepath.Join("..", "npu", "intel", "example", "bench_manifest.json"),
	} {
		if abs, err := filepath.Abs(rel); err == nil {
			if _, err := os.Stat(abs); err == nil {
				return abs
			}
		}
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "git/chaosglue/npu/intel/example/bench_manifest.json")
}

func LoadManifest() (BenchManifest, error) {
	b, err := os.ReadFile(manifestPath())
	if err != nil {
		return BenchManifest{}, err
	}
	var m BenchManifest
	if err := json.Unmarshal(b, &m); err != nil {
		return BenchManifest{}, err
	}
	return m, nil
}

func DefaultPluginPath() string {
	return defaultPluginPath()
}
