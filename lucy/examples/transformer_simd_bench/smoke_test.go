package tfsimdbench

import (
	"os"
	"testing"
)

// TestSmokeSmol2 loads smol2-135m from .entity and runs the full CPU profile
// matrix. Skips cleanly when the checkpoint or HF tokenizer snapshot is absent.
func TestSmokeSmol2(t *testing.T) {
	// Entity paths are relative to the lucy module root (where lucy_entities/ lives),
	// but `go test` runs with CWD at the package dir; hop up to the lucy root.
	if _, err := os.Stat(entitiesDir); err != nil {
		if err := os.Chdir("../.."); err != nil {
			t.Skipf("cannot locate lucy root: %v", err)
		}
	}

	spec := ModelSpec{RepoID: "HuggingFaceTB/SmolLM2-135M-Instruct", ShortName: "smol2-135m"}
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

	if m.numLayers <= 0 || m.hiddenSize <= 0 {
		t.Fatalf("bad topology: layers=%d hidden=%d", m.numLayers, m.hiddenSize)
	}

	var baseReply string
	for i, prof := range BenchProfiles {
		res := runProfile(m, prof)
		if res.Status != "OK" || res.GenTokens == 0 {
			t.Fatalf("profile %s: status=%s gen=%d err=%s", prof.Name, res.Status, res.GenTokens, res.Error)
		}
		if res.DecodeTokS <= 0 {
			t.Fatalf("profile %s: non-positive decode tok/s %.3f", prof.Name, res.DecodeTokS)
		}
		t.Logf("%-12s decode=%.2f tok/s prefill=%.2f tok/s reply=%q",
			prof.Name, res.DecodeTokS, res.PrefillTokS, truncate(res.Reply, 80))
		if i == 0 {
			baseReply = res.Reply
			continue
		}
		// cpu_mc must be bit-identical to cpu_sc (same math, just parallelized).
		if prof.Name == "cpu_mc" && res.Reply != baseReply {
			t.Errorf("cpu_mc diverged from cpu_sc:\n sc=%q\n mc=%q", baseReply, res.Reply)
		}
	}
}
