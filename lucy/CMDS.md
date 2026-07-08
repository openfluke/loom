# Lucy — background test commands

Run everything from the **`lucy/`** directory:

```bash
cd loom/lucy          # Linux / Mac
cd loom\lucy           # Windows
```

Build once (optional, faster than `go run` every time):

```bash
go build -o lucy .     # Linux / Mac
go build -o lucy.exe . # Windows
```

All test logs go under **`lucy_testing_output/`** (gitignored).

---

## Quick reference

| Menu | Suite | Log / output | Auto env var |
|------|-------|--------------|--------------|
| **[10]** | Context suite (long context / multi-prompt) | `lucy_testing_output/context_suite/` | `LOOM_CONTEXT_SUITE=1` |
| **[7]** | Seven-layer CPU (210 dtype matrix) | `lucy_testing_output/seven_layer.txt` | pipe stdin (below) |
| **[6]** | Five-layer examples | `lucy_testing_output/five_layer.txt` | pipe stdin |
| **[3]** | Layer testing CPU/GPU | `lucy_testing_output/log.txt` | pipe stdin |
| **[9]** | Intel NPU bridge | `lucy_testing_output/nine_layer.txt` | `LOOM_NINE_LAYER=1` + pipe |
| **[2]** | Dense adaptation benchmark | terminal only | pipe stdin |
| **[5]** | BitNet forward benchmark | terminal only | pipe stdin |

**Go unit tests** (no menu):

| Package | Command |
|---------|---------|
| Layer matrix | `go test ./testing/ -timeout 30m` |
| Seven-layer | `go test ./examples/seven_layer/ -timeout 2h` |
| Context suite | `go test ./examples/context_suite/ -run TestRunFullMatrixIfEntitiesPresent -timeout 3h` |

---

## Watching / stopping a background job

**Tail log** (Ctrl+C only stops `tail`, not the job):

```bash
tail -f context_suite_run.log
tail -f lucy_testing_output/context_suite/context_suite.txt
```

**Check still running:**

```bash
# Linux / Mac
pgrep -fl "lucy|go run"

# Windows PowerShell
Get-Process | Where-Object { $_.ProcessName -match "lucy|go" }
```

**Stop:**

```bash
kill <PID>           # Linux / Mac
taskkill /PID <PID>  # Windows cmd
Stop-Process -Id <PID>  # Windows PowerShell
```

---

## Linux & Mac

### Context suite [10] — full matrix (~1–2 h)

```bash
cd loom/lucy
LOOM_CONTEXT_SUITE=1 nohup go run . > context_suite_run.log 2>&1 &
echo $!    # save PID
```

With built binary:

```bash
LOOM_CONTEXT_SUITE=1 nohup ./lucy > context_suite_run.log 2>&1 &
```

**Quick smoke** (smol2, 2 scenarios, ~few min):

```bash
LOOM_CONTEXT_SUITE=1 LOOM_CONTEXT_SUITE_SMOKE=1 nohup go run . > context_suite_smoke.log 2>&1 &
```

Survives **SSH disconnect** when started with `nohup`.

**Safer for long runs — tmux:**

```bash
tmux new -s context
cd loom/lucy
LOOM_CONTEXT_SUITE=1 go run .
# detach: Ctrl+B then D
# reattach: tmux attach -t context
```

---

### Seven-layer suite [7] — all 10 layers (~long)

Pipe menu choices: main menu **7**, sub-menu **0** (all layers).

```bash
cd loom/lucy
printf '7\n0\n' | nohup go run . > seven_layer_run.log 2>&1 &
```

Or:

```bash
printf '7\n0\n' | nohup ./lucy > seven_layer_run.log 2>&1 &
```

Watch: `tail -f seven_layer_run.log` or `tail -f lucy_testing_output/seven_layer.txt`

---

### Five-layer examples [6] — all layers

```bash
printf '6\n0\n' | nohup go run . > five_layer_run.log 2>&1 &
```

Log: `lucy_testing_output/five_layer.txt`

---

### Layer testing [3] — all layers, save log

Pipe: **3** → layer **0** (all) → confirm **1** → save log **1**

```bash
printf '3\n0\n1\n1\n' | nohup go run . > layer_testing_run.log 2>&1 &
```

Log: `lucy_testing_output/log.txt`

Single layer (e.g. Dense = **6**), all sub-tests (**0**), save log:

```bash
printf '3\n6\n0\n1\n' | nohup go run . > layer_dense_run.log 2>&1 &
```

---

### Intel NPU bridge [9]

Skips main menu; still picks sub-suite inside [9].

```bash
# DispatchLayer medium (default-ish)
LOOM_NINE_LAYER=1 printf '4\n' | nohup go run . > nine_layer_run.log 2>&1 &

# Full dispatch matrix
LOOM_NINE_LAYER=1 printf '5\n' | nohup go run . > nine_layer_run.log 2>&1 &
```

Log: `lucy_testing_output/nine_layer.txt`

---

### Dense adaptation benchmark [2]

```bash
printf '2\n' | nohup go run . > dense_adapt_run.log 2>&1 &
```

(Interactive prompts may still appear depending on benchmark options.)

---

### BitNet forward benchmark [5]

Needs BitNet in HF cache. Pipes main menu only; model/GPU prompts may still ask.

```bash
printf '5\n' | nohup go run . > forward_bench_run.log 2>&1 &
```

---

### Go unit tests in background

```bash
# Layer testing package
nohup go test ./testing/ -v -timeout 30m > testing_unit.log 2>&1 &

# Seven-layer regressions
nohup go test ./examples/seven_layer/ -v -timeout 2h > seven_layer_unit.log 2>&1 &

# Context suite (needs .entity files in lucy_entities/)
nohup go test ./examples/context_suite/ -run TestRunFullMatrixIfEntitiesPresent -v -timeout 3h > context_unit.log 2>&1 &
```

---

## Windows

Use **PowerShell** from `loom\lucy`. Set env vars for the session:

```powershell
$env:LOOM_CONTEXT_SUITE = "1"
```

### Context suite [10] — full matrix

**PowerShell — background job:**

```powershell
cd loom\lucy
$env:LOOM_CONTEXT_SUITE = "1"
Start-Job -Name ContextSuite -ScriptBlock {
  Set-Location $using:PWD
  go run . *> context_suite_run.log
}
# watch: Receive-Job -Name ContextSuite -Keep
# or:   Get-Content context_suite_run.log -Wait
```

**PowerShell — Start-Process (survives closing window if parent detached):**

```powershell
$env:LOOM_CONTEXT_SUITE = "1"
Start-Process -FilePath "go" -ArgumentList "run","." `
  -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput "context_suite_run.log" `
  -RedirectStandardError "context_suite_err.log" `
  -WindowStyle Hidden
```

**Quick smoke:**

```powershell
$env:LOOM_CONTEXT_SUITE = "1"
$env:LOOM_CONTEXT_SUITE_SMOKE = "1"
Start-Process -FilePath "go" -ArgumentList "run","." `
  -RedirectStandardOutput "context_suite_smoke.log" `
  -RedirectStandardError "context_suite_smoke_err.log" `
  -WindowStyle Hidden
```

**cmd.exe — simple background:**

```cmd
cd loom\lucy
set LOOM_CONTEXT_SUITE=1
start /B go run . > context_suite_run.log 2>&1
```

---

### Seven-layer [7] — all layers

**PowerShell:**

```powershell
"7`n0`n" | go run . *> seven_layer_run.log
# background:
Start-Job { Set-Location loom\lucy; "7`n0`n" | go run . *> seven_layer_run.log }
```

**cmd.exe:**

```cmd
(echo 7& echo 0) | go run . > seven_layer_run.log 2>&1
```

---

### Five-layer [6] — all layers

```powershell
"6`n0`n" | go run . *> five_layer_run.log
```

```cmd
(echo 6& echo 0) | go run . > five_layer_run.log 2>&1
```

---

### Layer testing [3] — all layers + save log

```powershell
"3`n0`n1`n1`n" | go run . *> layer_testing_run.log
```

```cmd
(echo 3& echo 0& echo 1& echo 1) | go run . > layer_testing_run.log 2>&1
```

---

### Intel NPU bridge [9]

```powershell
$env:LOOM_NINE_LAYER = "1"
"4`n" | go run . *> nine_layer_run.log
```

---

### Go unit tests (Windows)

```powershell
Start-Job { Set-Location loom\lucy; go test ./testing/ -v -timeout 30m *> testing_unit.log }
Start-Job { Set-Location loom\lucy; go test ./examples/seven_layer/ -v -timeout 2h *> seven_layer_unit.log }
Start-Job { Set-Location loom\lucy; go test ./examples/context_suite/ -run TestRunFullMatrixIfEntitiesPresent -v -timeout 3h *> context_unit.log }
```

---

## Context suite details

See also [`examples/context_suite/README.md`](examples/context_suite/README.md).

| Variable | Effect |
|----------|--------|
| `LOOM_CONTEXT_SUITE=1` | Skip main menu → run full 60-cell matrix |
| `LOOM_CONTEXT_SUITE_SMOKE=1` | With above → smol2 only, 2 scenarios |

**Outputs:**

- `lucy_testing_output/context_suite/context_suite.txt` — full log
- `lucy_testing_output/context_suite/results.json` — summary JSON
- `lucy_testing_output/context_suite/outputs/*.txt` — per-run generations

**Models tested:** bitnet · qwen3-0.6b · smol2-135m (need `.entity` in `lucy_entities/`)

**Exec matrix:** cpu_sc · cpu_mc · gpu_sc · gpu_mc

---

## Notes

- **`nohup`** (Linux/Mac): job keeps running after SSH exit. **`tail -f`** and **Ctrl+C** do not stop the job.
- **Windows** has no `nohup`; use `Start-Job`, `Start-Process`, or **tmux** (WSL/Git Bash).
- **Menu [1], [4], [8]** are interactive chat/download flows — not suited for unattended background unless you script every prompt.
- **GPU on Linux (NVIDIA):** you may need `VK_ICD_FILENAMES` / `WGPU_ADAPTER_NAME` in the same shell before launching (see [`README.md`](README.md)).
