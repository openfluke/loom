package tfsimdbench

import (
	"math"
	"os"
	"testing"

	"github.com/openfluke/loom/poly"
)

func chdirLucyRoot(t *testing.T) {
	if _, err := os.Stat(entitiesDir); err != nil {
		if err := os.Chdir("../.."); err != nil {
			t.Skipf("cannot locate lucy root: %v", err)
		}
	}
}

// diagModel loads a model, reports MHA/SwiGLU DType + whether the packed-ternary
// CPU path (which bypasses SIMD) is active, then runs a tiny SC-vs-SIMD compare.
func diagModel(t *testing.T, repoID, short string, genTok int) {
	spec := ModelSpec{RepoID: repoID, ShortName: short}
	if !entityExists(spec.RepoID) {
		t.Skipf("entity not found: %s", entityPath(spec.RepoID))
	}
	if _, err := resolveSnapshotDir(spec.RepoID); err != nil {
		t.Skipf("HF snapshot unavailable: %v", err)
	}

	m, err := loadModelCPU(spec)
	if err != nil {
		t.Fatalf("loadModelCPU: %v", err)
	}
	defer m.close()

	mha := &m.tr.Network.Layers[1]
	mlp := &m.tr.Network.Layers[3]
	t.Logf("%s: layers=%d hidden=%d storedDType=%v ternary=%v", short, m.numLayers, m.hiddenSize, m.dtype, m.ternary)
	t.Logf("%s: MHA.DType=%v SwiGLU.DType=%v UseExactDType=%v", short, mha.DType, mlp.DType, m.tr.Network.UseExactDType)

	sys := defaultSystemPrompt(spec.RepoID)
	opts := poly.GenOptions{
		MaxTokens: genTok, MinTokens: genTok, Temperature: 0, TopK: 1,
		Deterministic: true, EOSTokens: m.eosTokens,
		BannedTokens: poly.TokenizerBannedSpecialExceptEOS(m.tk, m.eosTokens), Silent: true,
	}

	// scalar single-core
	applyProfile(m.tr, BenchProfile{Name: "cpu_sc", MultiCore: false, Simd: false})
	scReply, scM := m.tr.Generate(m.encode, m.decode, nil, sys, fixedPrompt, opts)
	if math.IsNaN(float64(scM.FirstLogit)) {
		t.Errorf("%s cpu_sc: NaN first logit", short)
	}
	t.Logf("%s cpu_sc:      decode=%.2f tok/s reply=%q", short, scM.DecodeTokPerSec, truncate(scReply, 70))

	// SIMD single-core
	applyProfile(m.tr, BenchProfile{Name: "cpu_simd_sc", MultiCore: false, Simd: true})
	sdReply, sdM := m.tr.Generate(m.encode, m.decode, nil, sys, fixedPrompt, opts)
	if math.IsNaN(float64(sdM.FirstLogit)) {
		t.Errorf("%s cpu_simd_sc: NaN first logit", short)
	}
	t.Logf("%s cpu_simd_sc: decode=%.2f tok/s reply=%q", short, sdM.DecodeTokPerSec, truncate(sdReply, 70))

	speedup := 0.0
	if scM.DecodeTokPerSec > 0 {
		speedup = sdM.DecodeTokPerSec / scM.DecodeTokPerSec
	}
	parity := "match"
	if scReply != sdReply {
		parity = "DIVERGE"
	}
	t.Logf("%s: SIMD speedup=%.2fx parity=%s", short, speedup, parity)
}

func TestDiagQwen3(t *testing.T) {
	chdirLucyRoot(t)
	diagModel(t, "Qwen/Qwen3-0.6B", "qwen3-0.6b", 12)
}

func TestDiagBitnet(t *testing.T) {
	chdirLucyRoot(t)
	diagModel(t, "microsoft/bitnet-b1.58-2B-4T", "bitnet", 8)
}
