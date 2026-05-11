# Donate compute (TCP)

The **`donate_compute_*.go`** files in `poly/` implement an optional **TCP protocol** so a **donor** machine can accept inference-style work from clients on the same network (or loopback). Work is exchanged as **length-prefixed JSON** frames over a single connection — there is no HTTP server inside `poly` for this path.

**Status:** The server’s inference and prompt paths are **stubs** (`stubInfer` / `stubPrompt`) until wired to real model loading, `poly` execution, or subprocess hooks.

---

## Why it exists

- **LAN-friendly**: bind to `0.0.0.0` (or a specific interface) and let another host submit jobs without bundling a separate HTTP stack in `poly`.
- **Two modes** (see below): push weights + token **`infer`**, or **`prompt`**-only against a local LM path advertised in the hello.

---

## Wire format (`donate_compute_framing.go`)

Each message is:

1. **`uint32` length**, little-endian (4 bytes).
2. **UTF-8 JSON object** of exactly that length.

Constants:

| Constant | Value | Meaning |
| :--- | :--- | :--- |
| `DonateComputeDefaultPort` | **17001** | Default listen/dial port (adjacent to construct TCP dev on **17000**). |
| `MaxDonateFrameBytes` | 64 MiB | Maximum single-frame payload; large models use **many** weight chunks, not one giant frame. |

Helpers: **`WriteDonateFrame`**, **`ReadDonateFrame`**.

---

## Message types (`donate_compute_types.go`)

Version **v1** uses a `"type"` string discriminator. Constants include:

- **`hello`** — first frame from server after connect; client may echo hello.
- **`model_begin`**, **`weights_chunk`**, **`model_commit`**, **`model_status`** — **model_push** upload lifecycle.
- **`infer`**, **`infer_result`** — token-ID jobs against a **mounted** pushed model.
- **`prompt`**, **`prompt_result`** — text jobs for **local_lm** nodes.
- **`queue_status`**, **`error`** — optional / error paths.

Structs (`DonateHello`, `DonateModelBegin`, `DonateWeightsChunk`, `DonateInfer`, `DonatePrompt`, …) mirror the JSON fields.

---

## Server (`donate_compute_server.go`)

**`ServeDonateComputeTCP(opts DonateComputeServerOptions)`** returns a **`net.Listener`**. It:

- Sends a **`DonateHello`** immediately after each accept (mode, role `server`, optional `LocalLmPath`, queue capacity hint).
- Parses frames in a loop per connection.
- Enqueues **`infer`** / **`prompt`** jobs on a **global FIFO channel**; **one worker** drains the queue (serial execution — not N parallel model mounts).

**`DonateComputeServerMode`:**

| Mode | Behavior |
| :--- | :--- |
| **`model_push`** | Client sends **`model_begin`** (config JSON + expected weight length), **`weights_chunk`** (base64 slices), **`model_commit`**. Server acknowledges with **`model_status`**. Then client may send **`infer`** with `input_ids` / `max_tokens`. |
| **`local_lm`** | **`infer`** is rejected; client uses **`prompt`** with full text. Server may advertise **`LocalLmPath`** in hello (informational). |

**`CloseDonateListener`** closes the listener.

---

## Client (`donate_compute_client.go`)

- **`DialDonateCompute(addr)`** — TCP dial (default `127.0.0.1:17001` if empty), read server hello, send client hello; returns **`DonateClient`** + **`DonateHello`**.
- **`PutModel(configJSON, weights)`** — stream model for **model_push** nodes.
- **`EnqueueInfer`**, **`EnqueuePrompt`** — send one job and wait for the matching result frame.

---

## Tests

**`donate_compute_test.go`** covers framing and client/server interaction.

---

## Security

**v1 has no TLS and no authentication.** It is intended for **trusted networks** (e.g. same Wi‑Fi lab). Do not expose the raw port to the public Internet without a VPN, SSH tunnel, or application-layer gateway.

---

## File map

| File | Role |
| :--- | :--- |
| `donate_compute_types.go` | v1 constants and JSON structs |
| `donate_compute_framing.go` | Frame encode/decode, default port, size limits |
| `donate_compute_server.go` | TCP server, modes, queue, stubs |
| `donate_compute_client.go` | Dial, `PutModel`, `EnqueueInfer`, `EnqueuePrompt` |
| `donate_compute_test.go` | Tests |
