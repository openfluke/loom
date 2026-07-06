package contextsuite

import (
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/loom/poly"
)

// CellResult is one model × exec × scenario test outcome.
type CellResult struct {
	Model         string  `json:"model"`
	ExecProfile   string  `json:"exec_profile"`
	Scenario      string  `json:"scenario"`
	Status        string  `json:"status"`
	Error         string  `json:"error,omitempty"`
	PromptTokens  int     `json:"prompt_tokens"`
	MaxSeqLen     int     `json:"max_seq_len"`
	OverMaxSeqLen bool    `json:"over_max_seq_len"`
	GeneratedToks int     `json:"generated_tokens"`
	PrefillTokS   float64 `json:"prefill_tok_s"`
	DecodeTokS    float64 `json:"decode_tok_s"`
	RecallHit     bool    `json:"recall_hit"`
	ExpectRecall  string  `json:"expect_recall,omitempty"`
	FinalReply    string  `json:"final_reply"`
	OutputFile    string  `json:"output_file"`
	DurationMS    int64   `json:"duration_ms"`
}

func runScenario(m *loadedModel, sc Scenario) CellResult {
	start := time.Now()
	res := CellResult{
		Model:       m.spec.ShortName,
		Scenario:    sc.ID,
		MaxSeqLen:   maxSeqLen,
		ExpectRecall: sc.RecallNeedle,
	}

	systemPrompt := defaultSystemPrompt(m.spec.RepoID)
	maxTok := sc.MaxTokens
	if maxTok <= 0 {
		maxTok = maxGenTokens(m.spec.RepoID)
	}

	opts := poly.GenOptions{
		MaxTokens:             maxTok,
		MinTokens:             4,
		Temperature:           0.7,
		TopK:                  40,
		Deterministic:         false,
		EOSTokens:             m.eosTokens,
		BannedTokens:          poly.TokenizerBannedSpecialExceptEOS(m.tk, m.eosTokens),
		RepetitionPenalty:     1.1,
		RepetitionWindow:      64,
		MaxConsecutiveRepeats: 3,
		NoRepeatNGram:         3,
		Silent:                true,
	}

	var chatTurns []poly.Turn
	var lastReply string
	var lastMetrics poly.GenMetrics
	var lastPromptTokens int

	for turnIdx, userMsg := range sc.Turns {
		prompt := m.tr.Template.BuildPrompt(chatTurns, systemPrompt, userMsg)
		promptTokens := len(m.encode(prompt))
		lastPromptTokens = promptTokens

		reply, metrics := m.tr.Generate(m.encode, m.decode, chatTurns, systemPrompt, userMsg, opts)
		lastReply = strings.TrimSpace(reply)
		lastMetrics = metrics

		fmt.Printf("      turn %d/%d: prompt=%d tok (maxSeqLen=%d) gen=%d reply=%q\n",
			turnIdx+1, len(sc.Turns), promptTokens, maxSeqLen, metrics.GeneratedTokens, truncate(lastReply, 80))

		chatTurns = append(chatTurns, poly.Turn{User: userMsg, Assistant: lastReply})
	}

	res.PromptTokens = lastPromptTokens
	res.OverMaxSeqLen = lastPromptTokens > maxSeqLen
	res.GeneratedToks = lastMetrics.GeneratedTokens
	res.PrefillTokS = lastMetrics.PrefillTokPerSec
	res.DecodeTokS = lastMetrics.DecodeTokPerSec
	res.FinalReply = lastReply
	res.DurationMS = time.Since(start).Milliseconds()

	if sc.RecallNeedle != "" {
		res.RecallHit = strings.Contains(strings.ToLower(lastReply), strings.ToLower(sc.RecallNeedle))
	} else {
		res.RecallHit = lastReply != ""
	}

	res.Status = classifyStatus(sc, res)
	return res
}

func classifyStatus(sc Scenario, res CellResult) string {
	if res.FinalReply == "" && res.GeneratedToks == 0 {
		return "EMPTY"
	}
	if sc.ExpectOverflow {
		if res.OverMaxSeqLen {
			return "OVERFLOW_OK"
		}
		if sc.RecallNeedle != "" && !res.RecallHit {
			return "OVERFLOW_MISS"
		}
		return "OVERFLOW_OK"
	}
	if sc.RecallNeedle != "" && !res.RecallHit {
		return "RECALL_MISS"
	}
	if res.OverMaxSeqLen {
		return "OVER_MAX"
	}
	return "PASS"
}

func truncate(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= n {
		return s
	}
	return s[:n-3] + "..."
}

func runModelProfile(spec ModelSpec, prof ExecProfile, scenarios []Scenario) {
	fmt.Printf("\n▶ %s / %s — loading .entity …\n", spec.ShortName, prof.Name)
	m, err := loadModel(spec, prof)
	if err != nil {
		fmt.Printf("   ❌ load failed: %v\n", err)
		for _, sc := range scenarios {
			recordCell(CellResult{
				Model:       spec.ShortName,
				ExecProfile: prof.Name,
				Scenario:    sc.ID,
				Status:      "LOAD_FAIL",
				Error:       err.Error(),
			})
		}
		return
	}
	defer m.close()

	fmt.Printf("   ✅ loaded (%d layers, hidden=%d, gpu=%v)\n", m.numLayers, m.hiddenSize, m.useGPU)

	for _, sc := range scenarios {
		fmt.Printf("\n   ── scenario: %s — %s\n", sc.ID, sc.Description)
		res := runScenario(m, sc)
		res.ExecProfile = prof.Name

		body := formatOutputBody(spec, prof, sc, res)
		res.OutputFile = SaveGeneration(spec.ShortName, prof.Name, sc.ID, body)
		recordCell(res)

		fmt.Printf("   → status=%s recall=%v prompt=%d tok output=%s\n",
			res.Status, res.RecallHit, res.PromptTokens, res.OutputFile)
	}
}

func formatOutputBody(spec ModelSpec, prof ExecProfile, sc Scenario, res CellResult) string {
	var b strings.Builder
	fmt.Fprintf(&b, "model: %s (%s)\n", spec.ShortName, spec.RepoID)
	fmt.Fprintf(&b, "exec:  %s (gpu=%v mc=%v)\n", prof.Name, prof.UseGPU, prof.MultiCore)
	fmt.Fprintf(&b, "scenario: %s\n", sc.ID)
	fmt.Fprintf(&b, "description: %s\n", sc.Description)
	fmt.Fprintf(&b, "max_seq_len: %d\n", maxSeqLen)
	fmt.Fprintf(&b, "prompt_tokens_last_turn: %d\n", res.PromptTokens)
	fmt.Fprintf(&b, "over_max_seq_len: %v\n", res.OverMaxSeqLen)
	fmt.Fprintf(&b, "generated_tokens: %d\n", res.GeneratedToks)
	fmt.Fprintf(&b, "prefill_tok_s: %.2f\n", res.PrefillTokS)
	fmt.Fprintf(&b, "decode_tok_s: %.2f\n", res.DecodeTokS)
	fmt.Fprintf(&b, "status: %s\n", res.Status)
	if sc.RecallNeedle != "" {
		fmt.Fprintf(&b, "recall_needle: %q hit=%v\n", sc.RecallNeedle, res.RecallHit)
	}
	fmt.Fprintf(&b, "\n--- turns ---\n")
	for i, msg := range sc.Turns {
		fmt.Fprintf(&b, "user[%d]: %s\n", i+1, msg)
	}
	fmt.Fprintf(&b, "\n--- final reply ---\n%s\n", res.FinalReply)
	return b.String()
}

// RunFullMatrix runs all available models × exec profiles × scenarios.
func RunFullMatrix() {
	models := availableModels()
	if len(models) == 0 {
		fmt.Println("❌ No .entity checkpoints found for target models.")
		fmt.Println("   Expected under lucy_entities/:")
		for _, m := range TargetModels {
			fmt.Printf("     • %s\n", entityPath(m.RepoID))
		}
		fmt.Println("   Convert via Lucy [8] ENTITY Talk or download + convert first.")
		return
	}

	fmt.Printf("Models on disk: %d/%d\n", len(models), len(TargetModels))
	for _, m := range models {
		fmt.Printf("  • %s → %s\n", m.ShortName, entityPath(m.RepoID))
	}
	fmt.Printf("\nScenarios: %d | Exec profiles: %d | MaxSeqLen: %d\n",
		len(AllScenarios), len(ExecProfiles), maxSeqLen)
	fmt.Println("  Context limit: Lucy sets MaxSeqLen=512 on all MHA layers (KV cache size).")
	fmt.Println("  Long multi-turn chats accumulate tokens in the prompt; beyond ~512 the model")
	fmt.Println("  cannot attend to earlier context (no sliding window yet).")

	total := len(models) * len(ExecProfiles) * len(AllScenarios)
	fmt.Printf("\n🚀 Running %d cells (models × exec × scenarios) …\n", total)

	start := time.Now()
	for _, spec := range models {
		for _, prof := range ExecProfiles {
			runModelProfile(spec, prof, AllScenarios)
		}
	}
	fmt.Printf("\n⏱  Total wall time: %v\n", time.Since(start).Round(time.Millisecond))
}
