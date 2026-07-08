# Context suite — long context / multi-prompt tests

Lucy menu **[10]** — automated inference tests for `.entity` checkpoints.

## Models

| Short name   | HF repo                              | Entity file |
|--------------|--------------------------------------|-------------|
| bitnet       | microsoft/bitnet-b1.58-2B-4T         | `lucy_entities/microsoft--bitnet-b1.58-2B-4T.entity` |
| qwen3-0.6b   | Qwen/Qwen3-0.6B                      | `lucy_entities/Qwen--Qwen3-0.6B.entity` |
| smol2-135m   | HuggingFaceTB/SmolLM2-135M-Instruct  | `lucy_entities/HuggingFaceTB--SmolLM2-135M-Instruct.entity` |

Convert missing checkpoints via Lucy **[8] ENTITY Talk**.

## Execution matrix

Each model runs on four profiles:

- `cpu_sc` — CPU, single-core (no tiling)
- `cpu_mc` — CPU, multi-core tiled
- `gpu_sc` — GPU, single workgroup
- `gpu_mc` — GPU, multi-workgroup tiled

GPU cells skip gracefully when WebGPU is unavailable.

## Scenarios

1. **short_baseline** — sanity check (`2+2`)
2. **long_prefill** — ~380-token single turn near `MaxSeqLen=512`
3. **multi_turn_4** — four turns, recall name "Alice"
4. **multi_turn_recall** — needle `BANANA-42` after filler turns
5. **overflow_probe** — eight turns designed to exceed the 512-token KV window

## Context limit

Lucy sets `MaxSeqLen=512` on all MHA layers (see `support.go`). The transformer poly default is 2048, but Lucy overrides it. Multi-turn prompts accumulate tokens; beyond ~512 the model cannot attend to earlier context (no sliding window yet). The suite records `prompt_tokens`, `over_max_seq_len`, and saved replies so you can see exactly when recall fails.

## Running

```bash
cd loom/lucy
./lucy          # choose [10], then [0] for full matrix
```

Non-interactive:

```bash
LOOM_CONTEXT_SUITE=1 ./lucy              # full matrix
LOOM_CONTEXT_SUITE=1 LOOM_CONTEXT_SUITE_SMOKE=1 ./lucy   # quick smoke (smol2, 2 scenarios)
```

Go test (runs full matrix when entities exist):

```bash
go test ./examples/context_suite/ -run TestRunFullMatrixIfEntitiesPresent -timeout 2h
```

## Outputs (gitignored)

All under `lucy_testing_output/context_suite/`:

- `context_suite.txt` — full tee log
- `results.json` — machine-readable summary
- `outputs/<model>__<exec>__<scenario>.txt` — per-cell prompt + reply
