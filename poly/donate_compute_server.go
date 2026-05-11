package poly

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
)

// DonateComputeServerMode selects how a donor node accepts work.
type DonateComputeServerMode string

const (
	DonateServerModelPush DonateComputeServerMode = "model_push"
	DonateServerLocalLM   DonateComputeServerMode = "local_lm"
)

// DonateComputeServerOptions configures [ServeDonateComputeTCP].
type DonateComputeServerOptions struct {
	Addr          string
	Mode          DonateComputeServerMode
	LocalLmPath   string
	// QueueCapacity is the buffered channel size (max pending jobs).
	QueueCapacity int
	// WorkerCount controls prompt/infer workers processing jobs in parallel.
	WorkerCount int
}

type donateServerJob struct {
	conn net.Conn
	raw  map[string]json.RawMessage
}

// ServeDonateComputeTCP listens on addr and runs one global FIFO queue for inference/prompt jobs.
// Stub inference: infer echoes tokens; prompt echoes text. Replace with real nn / subprocess hooks later.
func ServeDonateComputeTCP(opts DonateComputeServerOptions) (net.Listener, error) {
	if opts.Addr == "" {
		opts.Addr = fmt.Sprintf("0.0.0.0:%d", DonateComputeDefaultPort)
	}
	if opts.Mode == "" {
		opts.Mode = DonateServerModelPush
	}
	if opts.QueueCapacity <= 0 {
		opts.QueueCapacity = 64
	}
	if opts.WorkerCount <= 0 {
		opts.WorkerCount = 1
	}
	ln, err := net.Listen("tcp", opts.Addr)
	if err != nil {
		return nil, err
	}

	jobs := make(chan donateServerJob, opts.QueueCapacity)

	for i := 0; i < opts.WorkerCount; i++ {
		go func() {
			for j := range jobs {
				_ = processDonateServerJob(j, &opts, jobs)
			}
		}()
	}

	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go handleDonateClient(c, opts, jobs)
		}
	}()

	return ln, nil
}

func handleDonateClient(
	conn net.Conn,
	opts DonateComputeServerOptions,
	jobs chan<- donateServerJob,
) {
	defer conn.Close()

	if err := WriteDonateFrame(conn, DonateHello{
		V:             1,
		Type:          DonateMsgHello,
		Mode:          string(opts.Mode),
		Role:          "server",
		LocalLmPath:   opts.LocalLmPath,
		QueueCapacity: cap(jobs),
	}); err != nil {
		return
	}

	var (
		configBuf   []byte
		weightsBuf  []byte
		expectW     int
		gotW        int
		modelReady  bool
		partialHash = sha256.New()
	)

	for {
		var raw map[string]json.RawMessage
		if err := ReadDonateFrame(conn, &raw); err != nil {
			if errors.Is(err, io.EOF) {
				return
			}
			return
		}
		t := stringField(raw, "type")
		switch t {
		case DonateMsgHello:
			// Optional client hello — ignore after server hello.
			continue

		case DonateMsgModelBegin:
			if opts.Mode != DonateServerModelPush {
				_ = WriteDonateFrame(conn, map[string]any{
					"v": 1, "type": DonateMsgError,
					"error": "model_push not enabled on this node",
				})
				continue
			}
			configBuf = nil
			weightsBuf = nil
			gotW = 0
			partialHash.Reset()
			var mb DonateModelBegin
			if bb, err := json.Marshal(raw); err == nil {
				_ = json.Unmarshal(bb, &mb)
			}
			configBuf = append([]byte(nil), []byte(mb.ConfigJSON)...)
			expectW = mb.WeightsLen
			// Use int64 so the cap cannot overflow int on 32-bit toolchains.
			const maxW = int64(MaxDonateFrameBytes) * 100
			if expectW < 0 || int64(expectW) > maxW {
				_ = WriteDonateFrame(conn, map[string]any{"v": 1, "type": DonateMsgError, "error": "bad weights_len"})
				continue
			}
			if expectW > 0 {
				weightsBuf = make([]byte, 0, expectW)
			}
			modelReady = false

		case DonateMsgWeightsChunk:
			if opts.Mode != DonateServerModelPush {
				continue
			}
			var chunk DonateWeightsChunk
			b, _ := json.Marshal(raw)
			_ = json.Unmarshal(b, &chunk)
			if chunk.DataB64 == "" {
				continue
			}
			part, err := base64.StdEncoding.DecodeString(chunk.DataB64)
			if err != nil {
				_ = WriteDonateFrame(conn, map[string]any{"v": 1, "type": DonateMsgError, "error": "weights b64 decode"})
				continue
			}
			gotW += len(part)
			if expectW > 0 && gotW > expectW {
				_ = WriteDonateFrame(conn, map[string]any{"v": 1, "type": DonateMsgError, "error": "weights overflow"})
				continue
			}
			weightsBuf = append(weightsBuf, part...)
			_, _ = partialHash.Write(part)

		case DonateMsgModelCommit:
			if opts.Mode != DonateServerModelPush {
				continue
			}
			if expectW > 0 && gotW != expectW {
				_ = WriteDonateFrame(conn, DonateModelStatus{
					V: 1, Type: DonateMsgModelStatus, OK: false,
					Message: fmt.Sprintf("weights size mismatch: got %d expect %d", gotW, expectW),
				})
				continue
			}
			if len(configBuf) == 0 {
				_ = WriteDonateFrame(conn, DonateModelStatus{V: 1, Type: DonateMsgModelStatus, OK: false, Message: "no config"})
				continue
			}
			// Mount = retained buffers for a real loader later.
			_ = partialHash.Sum(nil)
			modelReady = true
			_ = WriteDonateFrame(conn, DonateModelStatus{V: 1, Type: DonateMsgModelStatus, OK: true, Message: "mounted"})

		case DonateMsgInfer:
			if opts.Mode == DonateServerLocalLM {
				_ = WriteDonateFrame(conn, map[string]any{
					"v": 1, "type": DonateMsgError, "error": "this node is local_lm; send prompt, not infer",
				})
				continue
			}
			if !modelReady {
				_ = WriteDonateFrame(conn, map[string]any{"v": 1, "type": DonateMsgError, "error": "model not mounted"})
				continue
			}
			select {
			case jobs <- donateServerJob{conn: conn, raw: cloneRawMap(raw)}:
			default:
				_ = WriteDonateFrame(conn, map[string]any{"v": 1, "type": DonateMsgError, "error": "queue full"})
			}

		case DonateMsgPrompt:
			if opts.Mode == DonateServerModelPush {
				_ = WriteDonateFrame(conn, map[string]any{
					"v": 1, "type": DonateMsgError, "error": "this node is model_push; mount model and send infer",
				})
				continue
			}
			select {
			case jobs <- donateServerJob{conn: conn, raw: cloneRawMap(raw)}:
			default:
				_ = WriteDonateFrame(conn, map[string]any{"v": 1, "type": DonateMsgError, "error": "queue full"})
			}

		default:
			_ = WriteDonateFrame(conn, map[string]any{
				"v": 1, "type": DonateMsgError, "error": "unknown type: " + t,
			})
		}
	}
}

func cloneRawMap(m map[string]json.RawMessage) map[string]json.RawMessage {
	out := make(map[string]json.RawMessage, len(m))
	for k, v := range m {
		b := make([]byte, len(v))
		copy(b, v)
		out[k] = json.RawMessage(b)
	}
	return out
}

func stringField(m map[string]json.RawMessage, k string) string {
	j, ok := m[k]
	if !ok {
		return ""
	}
	var s string
	_ = json.Unmarshal(j, &s)
	return s
}

func processDonateServerJob(j donateServerJob, opts *DonateComputeServerOptions, jobCh chan donateServerJob) error {
	conn := j.conn
	t := stringField(j.raw, "type")
	depth := len(jobCh)

	switch t {
	case DonateMsgInfer:
		var req DonateInfer
		b, _ := json.Marshal(j.raw)
		_ = json.Unmarshal(b, &req)
		out := stubInfer(req.InputIDs, req.MaxTokens)
		return WriteDonateFrame(conn, DonateInferResult{
			V: 1, Type: DonateMsgInferResult, JobID: req.JobID, OK: true,
			OutputIDs: out, QueueDepth: depth,
		})

	case DonateMsgPrompt:
		var req DonatePrompt
		b, _ := json.Marshal(j.raw)
		_ = json.Unmarshal(b, &req)
		text := stubPrompt(req.Prompt, opts.LocalLmPath)
		return WriteDonateFrame(conn, DonatePromptResult{
			V: 1, Type: DonateMsgPromptResult, JobID: req.JobID, OK: true,
			Text: text, QueueDepth: depth,
		})
	}
	return nil
}

func stubInfer(input []int32, maxTok int) []int32 {
	if maxTok <= 0 {
		maxTok = 8
	}
	out := append([]int32(nil), input...)
	for i := 0; i < maxTok && i < 8; i++ {
		out = append(out, int32(1+i%7))
	}
	return out
}

func stubPrompt(prompt, localPath string) string {
	p := strings.TrimSpace(prompt)
	l := strings.ToLower(p)
	if strings.Contains(l, "reply with one word only: yes, no, or maybe") {
		return "maybe"
	}
	if idx := strings.LastIndex(l, "user question:"); idx != -1 {
		p = strings.TrimSpace(p[idx+len("user question:"):])
	}
	reply := buildStubReplyFromPrompt(p)
	if localPath != "" {
		return fmt.Sprintf("[local_lm stub @ %s] %s", localPath, reply)
	}
	return fmt.Sprintf("[prompt stub] %s", reply)
}

func buildStubReplyFromPrompt(userText string) string {
	s := strings.TrimSpace(userText)
	if s == "" {
		return "donate node online"
	}
	if strings.HasSuffix(s, "?") {
		return "stub says: maybe"
	}
	parts := strings.Fields(s)
	if len(parts) > 8 {
		parts = parts[:8]
	}
	return "stub says: " + strings.Join(parts, " ")
}

// CloseDonateListener closes the listener (shuts down accept loop).
func CloseDonateListener(ln net.Listener) {
	if ln != nil {
		_ = ln.Close()
	}
}
