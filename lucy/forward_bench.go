package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/loom/poly"
)

const bitNetBenchModelNeedle = "bitnet-b1.58-2b-4t"

func findBitNetBenchModel(models []string) (int, string, bool) {
	for i, m := range models {
		if strings.Contains(strings.ToLower(m), bitNetBenchModelNeedle) {
			return i + 1, m, true
		}
	}
	return 0, "", false
}

func runForwardScheduleComparison(
	tr *poly.Transformer[float32],
	encode func(string) []uint32,
	systemPrompt string,
	numBlocks int,
) {
	if tr == nil {
		return
	}
	stepsPerTok := numBlocks*6 + 1

	longUser := strings.TrimSpace(strings.Repeat(
		"Explain wavefront decoder scheduling, KV cache striping, and why multiple prompt tokens can sit in different transformer blocks at the same clock tick. ", 6))
	shortUser := "What is 2+2? Answer briefly."

	fmt.Println("\n══════════════════════════════════════════════════════════════")
	fmt.Println("  Token schedule: Normal (batched) vs Pipeline (per-token wavefront)")
	fmt.Println("  Model: microsoft/bitnet-b1.58-2b-4t  |  CPU  |  deterministic")
	fmt.Println("══════════════════════════════════════════════════════════════")
	fmt.Printf("  ~%d sub-layer steps per token through the decoder stack\n\n", stepsPerTok)

	// --- Long prefill: the case where tokens differ in time ---
	runTokenScheduleScenario(tr, encode, systemPrompt, "Long prefill", longUser, numBlocks)

	fmt.Println()
	// --- Short prefill for decode context ---
	runTokenScheduleScenario(tr, encode, systemPrompt, "Short prefill", shortUser, numBlocks)

	fmt.Println("══════════════════════════════════════════════════════════════")
}

func runTokenScheduleScenario(
	tr *poly.Transformer[float32],
	encode func(string) []uint32,
	systemPrompt, title, userMsg string,
	numBlocks int,
) {
	fmt.Printf("── %s ──\n", title)
	inputIDs := encode(tr.Template.BuildPrompt(nil, systemPrompt, userMsg))
	nTok := len(inputIDs)
	embeds := tr.TokensToTensor(inputIDs)
	fmt.Printf("   prompt tokens: %d\n\n", nTok)

	// Normal fused prefill
	normal := tr.BenchmarkCPUPrefill(embeds.Clone(), poly.TransformerForwardNormal)

	// Pipeline prefill + hidden parity vs normal (same prompt, two forwards)
	pipeBench, maxDiff := tr.ComparePrefillToNormal(embeds.Clone())
	pipe := pipeBench.Pipeline
	tl := pipe.SummarizeTokenTimeline()

	fmt.Println("  ┌─────────────────────┬──────────────────────────────────────────┐")
	fmt.Println("  │                     │  Normal [1]          Pipeline [4]         │")
	fmt.Println("  ├─────────────────────┼──────────────────────────────────────────┤")
	fmt.Printf("  │ Wall time           │  %8s            %8s           │\n",
		normal.WallTime.Round(time.Millisecond), pipeBench.WallTime.Round(time.Millisecond))
	fmt.Printf("  │ Prompt tok/s        │  %8.2f            %8.2f           │\n",
		normal.TokPerSec, pipeBench.TokPerSec)
	fmt.Println("  ├─────────────────────┼──────────────────────────────────────────┤")
	fmt.Printf("  │ How tokens move     │  all %d batched      one job per token      │\n", nTok)
	fmt.Println("  │                     │  per block           (~181 clocks each)   │")
	fmt.Println("  ├─────────────────────┼──────────────────────────────────────────┤")
	fmt.Printf("  │ Peak tokens in flight│  %8d            %8d           │\n", 1, pipe.MaxActiveJobs)
	fmt.Printf("  │ Peak block spread   │  %8d            %8d           │\n", 0, pipe.MaxBlockSpread)
	fmt.Printf("  │ Pipeline ticks      │  %8s            %8d           │\n", "—", pipe.PipelineTicks)
	fmt.Printf("  │ Sub-layer ops       │  %8s            %8d           │\n", "—", pipe.SubLayerOps)
	fmt.Printf("  │ Hidden vs normal     │  %8s            max|Δ|=%.4g       │\n", "ref", maxDiff)
	fmt.Println("  └─────────────────────┴──────────────────────────────────────────┘")
	fmt.Println()

	fmt.Print(tl.FormatComparison(normal.WallTime.Seconds(), pipe.PipelineTicks))

	if nTok > 1 && len(pipe.TokenDoneTick) >= nTok {
		fmt.Println("  Pipeline: where each prompt token *finishes* the stack (done tick)")
		printTokenDoneASCII(pipe.TokenDoneTick, 60)
		fmt.Println()
	}

	if pipe.MaxActiveJobs > 1 {
		fmt.Printf("  While prefill runs: up to %d tokens at different blocks (spread %d) in the same clock tick.\n",
			pipe.MaxActiveJobs, pipe.MaxBlockSpread)
		fmt.Println("  Normal never has that — all positions share each block together.")
	}
}

// printTokenDoneASCII prints a compact ASCII timeline of completion ticks.
func printTokenDoneASCII(done []int, width int) {
	if len(done) == 0 {
		return
	}
	maxTick := 0
	for _, t := range done {
		if t > maxTick {
			maxTick = t
		}
	}
	if maxTick <= 0 {
		return
	}
	if width > len(done) {
		width = len(done)
	}
	step := len(done) / width
	if step < 1 {
		step = 1
	}
	fmt.Print("    ")
	for i := 0; i < len(done); i += step {
		tick := done[i]
		if tick < 0 {
			fmt.Print("_")
			continue
		}
		// Map tick to shade 0-9
		ch := '0' + byte(9*tick/maxTick)
		if ch < '0' {
			ch = '0'
		}
		if ch > '9' {
			ch = '9'
		}
		fmt.Printf("%c", ch)
	}
	fmt.Printf("  (0=early … 9=late, %d tokens sampled)\n", (len(done)+step-1)/step)
}
