# TANHI — UDP layer telemetry

**TANHI** streams **sparse, non-blocking JSON-line events** over **UDP** so external tools (notably the **SoulGlitch → TANHI** HUD) can visualize **per-layer forward/backward** activity, timing, dtypes, and **routing links** (parallel branches / sequential substeps).

Implementation: **`poly/tanhi.go`**. Integration hooks live in **`poly/forward.go`**, **`poly/backward.go`**, and **`poly/wgpu_forward.go`** (GPU transformer path). Optional **Welvet C-ABI** exports: **`welvet/cabi/tanhi_ext.go`**.

---

## Defaults

| Constant / env | Default | Meaning |
| :--- | :--- | :--- |
| **`poly.DefaultTanhiUDPPort`** | **17481** | UDP destination port (IANA unassigned range). |
| Host | `127.0.0.1` | When `TanhiUDPConfig.Host` is empty. |
| Disabled | `nil` / off | `VolumetricNetwork.Tanhi == nil` or `Enabled == false` → no UDP. |

---

## Configuration (`TanhiUDPConfig`)

Set on **`VolumetricNetwork.Tanhi`**:

- **`Enabled`** — master switch.
- **`Host`**, **`Port`** — UDP listener address (engine **sends** to this address).
- **`SendShape`** — include approximate tensor **`shape`** in each event (CPU path uses activations when available; GPU path uses **`TanhiGPULayerShapeHint`** — no readback).

Telemetry is **best-effort**: a buffered queue (**1024** packets); **overflow drops** silently so training/inference never blocks on HUD lag.

---

## Wire format

Each datagram payload is **one JSON object per line** (newline-terminated). Schema version **`v`: `"tanhi1"`**.

Typical fields:

| Field | Meaning |
| :--- | :--- |
| `seq` | Monotonic sequence number |
| `phase` | `"fwd"` or `"bwd"` |
| `idx` | Layer index in traversal (-1 or special indices possible on GPU fused paths) |
| `z`, `y`, `x`, `l` | Volumetric coordinate |
| `layer` | Layer type string |
| `dtype` | Integer dtype code |
| `connections` | Fan-out hint from weight masters (or override for GPU LM head) |
| `t0_ns`, `t1_ns` | Wall-clock nanoseconds around the layer |
| `shape` | Optional shape slice when `SendShape` is true |
| `links` | Optional routing targets for **LayerParallel** / **LayerSequential** (capped, for arc drawing) |

---

## SoulGlitch / Glitch CLI

- **`GLITCH_TANHI=1`** — enable when running **`loom/glitch`** interactively (or answer the prompt).
- **`TANHI_HOST`**, **`TANHI_PORT`**, **`TANHI_SHAPE=1`** — override host, port, and shape inclusion (same conventions in **`glitch/measure/*`** harnesses).

Open SoulGlitch **first**; set the listener **port** to match **`TANHI_PORT`** / **17481**.

---

## C-ABI (Welvet)

- **`LoomNetworkTanhiConfigure`** — enable/disable, host C string, port (0 → default **17481**), send-shape flag.
- **`LoomNetworkTanhiDisable`** — clear `Tanhi` on the network handle.
- **`LoomTanhiDefaultPort`** — returns **`DefaultTanhiUDPPort`**.

---

## Security note

UDP telemetry is **localhost-oriented** by default. Pointing **`Host`** at a remote machine sends layer metadata and timing to that address — use only on **trusted networks** when not using **`127.0.0.1`**.
