package tfsimdbench

import (
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/loom/poly"
)

// fixedPrompt is used for every model × profile so decode throughput is comparable.
const fixedPrompt = "Explain in one short paragraph why the sky appears blue during the day."

// benchTokens is the fixed decode length (min==max so every cell generates the
// same number of tokens regardless of EOS, keeping tok/s comparable + parity easy).
const benchTokens = 32

// BenchResult is one model × profile measurement.
type BenchResult struct {
	Model        string  `json:"model"`
	RepoID       string  `json:"repo_id"`
	Profile      string  `json:"profile"`
	MultiCore    bool    `json:"multi_core"`
	Simd         bool    `json:"simd"`
	SimdApplied  bool    `json:"simd_applied"` // false for packed-ternary (bitnet) — SIMD is a no-op
	Status       string  `json:"status"`
	Error        string  `json:"error,omitempty"`
	PromptTokens int     `json:"prompt_tokens"`
	GenTokens    int     `json:"generated_tokens"`
	PrefillTokS  float64 `json:"prefill_tok_s"`
	DecodeTokS   float64 `json:"decode_tok_s"`
	SpeedupVsSC  float64 `json:"speedup_vs_sc"` // decode tok/s relative to cpu_sc
	ParityVsSC   string  `json:"parity_vs_sc"`  // match / diverge / baseline / n-a
	Reply        string  `json:"reply"`
	DurationMS   int64   `json:"duration_ms"`
}

func genOptions(m *loadedModel) poly.GenOptions {
	return poly.GenOptions{
		MaxTokens:     benchTokens,
		MinTokens:     benchTokens, // force full length for comparable timing/parity
		Temperature:   0,
		TopK:          1,
		Deterministic: true, // greedy → identical output across SC/MC (SIMD may drift)
		EOSTokens:     m.eosTokens,
		BannedTokens:  poly.TokenizerBannedSpecialExceptEOS(m.tk, m.eosTokens),
		Silent:        true,
	}
}

func runProfile(m *loadedModel, prof BenchProfile) BenchResult {
	start := time.Now()
	res := BenchResult{
		Model:       m.spec.ShortName,
		RepoID:      m.spec.RepoID,
		Profile:     prof.Name,
		MultiCore:   prof.MultiCore,
		Simd:        prof.Simd,
		SimdApplied: prof.Simd,
	}

	applyProfile(m.tr, prof)

	sys := defaultSystemPrompt(m.spec.RepoID)
	opts := genOptions(m)

	// Warmup (untimed) to stabilize tile-size caches + first-touch allocations.
	warm := opts
	warm.MaxTokens = 4
	warm.MinTokens = 4
	_, _ = m.tr.Generate(m.encode, m.decode, nil, sys, fixedPrompt, warm)
	m.tr.Reset()

	reply, metrics := m.tr.Generate(m.encode, m.decode, nil, sys, fixedPrompt, opts)

	res.PromptTokens = metrics.PrefillTokens
	res.GenTokens = metrics.GeneratedTokens
	res.PrefillTokS = metrics.PrefillTokPerSec
	res.DecodeTokS = metrics.DecodeTokPerSec
	res.Reply = strings.TrimSpace(reply)
	res.DurationMS = time.Since(start).Milliseconds()
	res.Status = "OK"
	if res.GenTokens == 0 {
		res.Status = "EMPTY"
	}
	return res
}

func runModel(spec ModelSpec) []BenchResult {
	fmt.Printf("\n▶ %s (%s) — loading .entity …\n", spec.ShortName, spec.RepoID)
	m, err := loadModelCPU(spec)
	if err != nil {
		fmt.Printf("   ❌ load failed: %v\n", err)
		out := make([]BenchResult, 0, len(BenchProfiles))
		for _, prof := range BenchProfiles {
			r := BenchResult{
				Model: spec.ShortName, RepoID: spec.RepoID, Profile: prof.Name,
				MultiCore: prof.MultiCore, Simd: prof.Simd,
				Status: "LOAD_FAIL", Error: err.Error(),
			}
			recordCell(r)
			out = append(out, r)
		}
		return out
	}
	defer m.close()

	fmt.Printf("   ✅ loaded: %d layers · hidden=%d · dtype=%v · ternary=%v\n",
		m.numLayers, m.hiddenSize, m.dtype, m.ternary)
	if m.ternary {
		fmt.Println("   ℹ️  packed-ternary BitNet: AVX2 integer matvec when SIMD profile is on.")
	}

	results := make([]BenchResult, 0, len(BenchProfiles))
	var baseTokS float64
	var baseReply string
	haveBase := false

	for _, prof := range BenchProfiles {
		fmt.Printf("\n   ── profile: %s (mc=%v simd=%v)\n", prof.Name, prof.MultiCore, prof.Simd)
		res := runProfile(m, prof)

		if prof.Name == "cpu_sc" {
			baseTokS = res.DecodeTokS
			baseReply = res.Reply
			haveBase = true
			res.SpeedupVsSC = 1.0
			res.ParityVsSC = "baseline"
		} else {
			if haveBase && baseTokS > 0 {
				res.SpeedupVsSC = res.DecodeTokS / baseTokS
			}
			switch {
			case !haveBase:
				res.ParityVsSC = "n-a"
			case res.Reply == baseReply:
				res.ParityVsSC = "match"
			default:
				res.ParityVsSC = "diverge"
			}
		}

		body := formatBody(spec, prof, res)
		_ = saveGeneration(spec.ShortName, prof.Name, body)
		recordCell(res)
		results = append(results, res)

		fmt.Printf("      decode=%.2f tok/s  prefill=%.2f tok/s  speedup=%.2fx  parity=%s\n",
			res.DecodeTokS, res.PrefillTokS, res.SpeedupVsSC, res.ParityVsSC)
		fmt.Printf("      reply: %q\n", truncate(res.Reply, 100))
	}

	printModelTable(spec, results)
	return results
}

func formatBody(spec ModelSpec, prof BenchProfile, res BenchResult) string {
	var b strings.Builder
	fmt.Fprintf(&b, "model: %s (%s)\n", spec.ShortName, spec.RepoID)
	fmt.Fprintf(&b, "profile: %s (mc=%v simd=%v simd_applied=%v)\n",
		prof.Name, prof.MultiCore, prof.Simd, res.SimdApplied)
	fmt.Fprintf(&b, "prompt_tokens: %d\n", res.PromptTokens)
	fmt.Fprintf(&b, "generated_tokens: %d\n", res.GenTokens)
	fmt.Fprintf(&b, "prefill_tok_s: %.2f\n", res.PrefillTokS)
	fmt.Fprintf(&b, "decode_tok_s: %.2f\n", res.DecodeTokS)
	fmt.Fprintf(&b, "speedup_vs_sc: %.3f\n", res.SpeedupVsSC)
	fmt.Fprintf(&b, "parity_vs_sc: %s\n", res.ParityVsSC)
	fmt.Fprintf(&b, "\n--- prompt ---\n%s\n", fixedPrompt)
	fmt.Fprintf(&b, "\n--- reply ---\n%s\n", res.Reply)
	return b.String()
}

// RunFullMatrix benchmarks every available model across the full CPU profile matrix.
func RunFullMatrix() {
	models := availableModels()
	if len(models) == 0 {
		printNoModels()
		return
	}
	runSpecs(models)
}

// RunSingleModel benchmarks one target model across the full CPU profile matrix.
func RunSingleModel(spec ModelSpec) {
	if !entityExists(spec.RepoID) {
		fmt.Printf("❌ No .entity checkpoint for %s.\n", spec.ShortName)
		fmt.Printf("   Expected: %s\n", entityPath(spec.RepoID))
		fmt.Println("   Convert via Lucy [8] ENTITY Talk first.")
		return
	}
	runSpecs([]ModelSpec{spec})
}

func runSpecs(models []ModelSpec) {
	host := currentHostInfo()
	fmt.Printf("Host: %s · %d CPUs · SIMD=%s\n", host.Arch, host.NumCPU, host.SimdKind)
	fmt.Printf("Models on disk: %d/%d · decode length: %d tokens\n",
		len(models), len(TargetModels), benchTokens)

	start := time.Now()
	var all []BenchResult
	for _, spec := range models {
		all = append(all, runModel(spec)...)
	}
	printGlobalTable(all)
	fmt.Printf("\n⏱  Total wall time: %v\n", time.Since(start).Round(time.Millisecond))
}

func printNoModels() {
	fmt.Println("❌ No .entity checkpoints found for target models.")
	fmt.Println("   Expected under lucy_entities/:")
	for _, m := range TargetModels {
		fmt.Printf("     • %s\n", entityPath(m.RepoID))
	}
	fmt.Println("   Convert via Lucy [8] ENTITY Talk first.")
}

func truncate(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= n {
		return s
	}
	return s[:n-3] + "..."
}
