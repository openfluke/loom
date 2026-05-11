package poly

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net"
	"time"
)

// DonateClient is a thin v1 client for the donate-compute TCP protocol.
type DonateClient struct {
	Conn net.Conn
}

// DialDonateCompute connects to a donate-compute node and reads the server [DonateMsgHello] frame.
func DialDonateCompute(addr string) (*DonateClient, *DonateHello, error) {
	if addr == "" {
		addr = fmt.Sprintf("127.0.0.1:%d", DonateComputeDefaultPort)
	}
	c, err := net.DialTimeout("tcp", addr, 8*time.Second)
	if err != nil {
		return nil, nil, err
	}
	var raw map[string]json.RawMessage
	if err := ReadDonateFrame(c, &raw); err != nil {
		_ = c.Close()
		return nil, nil, err
	}
	if stringField(raw, "type") != DonateMsgHello {
		_ = c.Close()
		return nil, nil, fmt.Errorf("donate_compute: expected hello, got %q", stringField(raw, "type"))
	}
	b, _ := json.Marshal(raw)
	var hi DonateHello
	_ = json.Unmarshal(b, &hi)
	if err := WriteDonateFrame(c, DonateHello{
		V: 1, Type: DonateMsgHello, Mode: hi.Mode, Role: "client",
	}); err != nil {
		_ = c.Close()
		return nil, nil, err
	}
	return &DonateClient{Conn: c}, &hi, nil
}

// Close releases the connection.
func (c *DonateClient) Close() error {
	if c == nil || c.Conn == nil {
		return nil
	}
	return c.Conn.Close()
}

// PutModel streams a config JSON string and raw weights to a model_push node.
func (c *DonateClient) PutModel(configJSON string, weights []byte) error {
	if c == nil || c.Conn == nil {
		return fmt.Errorf("donate_compute: no connection")
	}
	if err := WriteDonateFrame(c.Conn, DonateModelBegin{
		V: 1, Type: DonateMsgModelBegin, ConfigJSON: configJSON, WeightsLen: len(weights),
	}); err != nil {
		return err
	}
	const chunk = 1 << 20
	for i, off := 0, 0; off < len(weights); i++ {
		n := chunk
		if off+n > len(weights) {
			n = len(weights) - off
		}
		end := off + n
		slice := weights[off:end]
		b64 := base64.StdEncoding.EncodeToString(slice)
		last := end >= len(weights)
		if err := WriteDonateFrame(c.Conn, DonateWeightsChunk{
			V: 1, Type: DonateMsgWeightsChunk, Index: i, Last: last, DataB64: b64,
		}); err != nil {
			return err
		}
		off = end
	}
	if err := WriteDonateFrame(c.Conn, DonateModelCommit{V: 1, Type: DonateMsgModelCommit}); err != nil {
		return err
	}
	var st map[string]json.RawMessage
	if err := ReadDonateFrame(c.Conn, &st); err != nil {
		return err
	}
	if stringField(st, "type") == DonateMsgModelStatus {
		var ms DonateModelStatus
		b, _ := json.Marshal(st)
		_ = json.Unmarshal(b, &ms)
		if !ms.OK {
			return fmt.Errorf("model status: %s", ms.Message)
		}
		return nil
	}
	if stringField(st, "type") == DonateMsgError {
		return fmt.Errorf("server: %s", stringField(st, "error"))
	}
	return fmt.Errorf("unexpected frame: %v", st)
}

// EnqueueInfer sends an infer job and waits for the result.
func (c *DonateClient) EnqueueInfer(jobID string, inputIDs []int32, maxTok int) (*DonateInferResult, error) {
	if err := WriteDonateFrame(c.Conn, DonateInfer{
		V: 1, Type: DonateMsgInfer, JobID: jobID, InputIDs: inputIDs, MaxTokens: maxTok,
	}); err != nil {
		return nil, err
	}
	var raw map[string]json.RawMessage
	if err := ReadDonateFrame(c.Conn, &raw); err != nil {
		return nil, err
	}
	if stringField(raw, "type") == DonateMsgError {
		return nil, fmt.Errorf("server: %s", stringField(raw, "error"))
	}
	var res DonateInferResult
	bb, _ := json.Marshal(raw)
	_ = json.Unmarshal(bb, &res)
	return &res, nil
}

// EnqueuePrompt sends a prompt job (local_lm nodes).
func (c *DonateClient) EnqueuePrompt(jobID, prompt string, maxOut int) (*DonatePromptResult, error) {
	if err := WriteDonateFrame(c.Conn, DonatePrompt{
		V: 1, Type: DonateMsgPrompt, JobID: jobID, Prompt: prompt, MaxOut: maxOut,
	}); err != nil {
		return nil, err
	}
	var raw map[string]json.RawMessage
	if err := ReadDonateFrame(c.Conn, &raw); err != nil {
		return nil, err
	}
	if stringField(raw, "type") == DonateMsgError {
		return nil, fmt.Errorf("server: %s", stringField(raw, "error"))
	}
	var res DonatePromptResult
	bb, _ := json.Marshal(raw)
	_ = json.Unmarshal(bb, &res)
	return &res, nil
}
