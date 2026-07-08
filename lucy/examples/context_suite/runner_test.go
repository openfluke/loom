package contextsuite_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	contextsuite "github.com/openfluke/loom/lucy/examples/context_suite"
)

func TestFixtures(t *testing.T) {
	if len(contextsuite.TargetModels) != 3 {
		t.Fatalf("expected 3 target models, got %d", len(contextsuite.TargetModels))
	}
	if len(contextsuite.AllScenarios) < 5 {
		t.Fatalf("expected at least 5 scenarios, got %d", len(contextsuite.AllScenarios))
	}
	if len(contextsuite.ExecProfiles) != 4 {
		t.Fatalf("expected 4 exec profiles, got %d", len(contextsuite.ExecProfiles))
	}
}

func entityPath(modelID string) string {
	return filepath.Join("lucy_entities", strings.ReplaceAll(modelID, "/", "--")+".entity")
}

func TestRunFullMatrixIfEntitiesPresent(t *testing.T) {
	lucyRoot := filepath.Join("..", "..")
	if _, err := os.Stat(filepath.Join(lucyRoot, "main.go")); err != nil {
		t.Skip("not running from loom/lucy tree")
	}
	if err := os.Chdir(lucyRoot); err != nil {
		t.Fatalf("chdir: %v", err)
	}

	found := false
	for _, m := range contextsuite.TargetModels {
		if _, err := os.Stat(entityPath(m.RepoID)); err == nil {
			found = true
			break
		}
	}
	if !found {
		t.Skip("no target .entity files (convert via Lucy [8])")
	}

	contextsuite.ResetSummary()
	contextsuite.RunFullMatrix()
}
