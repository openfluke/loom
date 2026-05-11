package poly

// Decoded one-of payloads for Type field (v1).
const (
	DonateMsgHello        = "hello"
	DonateMsgModelBegin   = "model_begin"
	DonateMsgWeightsChunk = "weights_chunk"
	DonateMsgModelCommit  = "model_commit"
	DonateMsgModelStatus  = "model_status"
	DonateMsgInfer        = "infer"
	DonateMsgInferResult  = "infer_result"
	DonateMsgPrompt       = "prompt"
	DonateMsgPromptResult = "prompt_result"
	DonateMsgQueueStatus  = "queue_status"
	DonateMsgError        = "error"
)

// DonateHello is the first frame after TCP connect.
type DonateHello struct {
	V    int    `json:"v"`
	Type string `json:"type"`
	// Mode at server: model_push (remote uploads weights) or local_lm (node runs local path).
	Mode string `json:"mode"`
	Role string `json:"role"` // "server" or "client"
	// LocalLmPath is set when server offers local_lm (path on donor machine; informational).
	LocalLmPath string `json:"local_lm_path,omitempty"`
	// QueueCapacity hints max buffered jobs on the node.
	QueueCapacity int `json:"queue_capacity,omitempty"`
}

// DonateModelBegin starts an uploaded model (model_push mode).
type DonateModelBegin struct {
	V             int    `json:"v"`
	Type          string `json:"type"`
	ConfigJSON    string `json:"config_json"`
	WeightsLen    int    `json:"weights_len"` // total bytes expected across chunks
	WeightsSHA256 string `json:"weights_sha256,omitempty"`
}

// DonateWeightsChunk carries part of safetensors / raw weights.
type DonateWeightsChunk struct {
	V       int    `json:"v"`
	Type    string `json:"type"`
	Index   int    `json:"index"`
	Last    bool   `json:"last"`
	DataB64 string `json:"data_b64"`
}

// DonateModelCommit finalizes uploaded buffers on the host.
type DonateModelCommit struct {
	V    int    `json:"v"`
	Type string `json:"type"`
}

// DonateModelStatus acknowledges mount state.
type DonateModelStatus struct {
	V       int    `json:"v"`
	Type    string `json:"type"`
	OK      bool   `json:"ok"`
	Message string `json:"message,omitempty"`
}

// DonateInfer queues tensor/token work against a mounted pushed model.
type DonateInfer struct {
	V         int     `json:"v"`
	Type      string  `json:"type"`
	JobID     string  `json:"job_id"`
	InputIDs  []int32 `json:"input_ids,omitempty"`
	MaxTokens int     `json:"max_tokens,omitempty"`
}

// DonateInferResult returns after the job leaves the queue.
type DonateInferResult struct {
	V          int     `json:"v"`
	Type       string  `json:"type"`
	JobID      string  `json:"job_id"`
	OK         bool    `json:"ok"`
	OutputIDs  []int32 `json:"output_ids,omitempty"`
	Error      string  `json:"error,omitempty"`
	QueueDepth int     `json:"queue_depth"`
}

// DonatePrompt enqueues full-text work (local_lm mode or relay).
type DonatePrompt struct {
	V       int    `json:"v"`
	Type    string `json:"type"`
	JobID   string `json:"job_id"`
	Prompt  string `json:"prompt"`
	MaxOut  int    `json:"max_out,omitempty"`
}

// DonatePromptResult is the completed prompt job.
type DonatePromptResult struct {
	V          int    `json:"v"`
	Type       string `json:"type"`
	JobID      string `json:"job_id"`
	OK         bool   `json:"ok"`
	Text       string `json:"text,omitempty"`
	Error      string `json:"error,omitempty"`
	QueueDepth int    `json:"queue_depth"`
}

// DonateQueueStatus is informational (optional push from server).
type DonateQueueStatus struct {
	V          int    `json:"v"`
	Type       string `json:"type"`
	Depth      int    `json:"depth"`
	Capacity   int    `json:"capacity"`
	InProgress bool   `json:"in_progress"`
}
