package nn

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// computeLayerStats calculates summary statistics for an activation slice
func computeLayerStats(data []float32, layerType string, threshold float32) LayerStats {
	if len(data) == 0 {
		return LayerStats{LayerType: layerType}
	}

	var sum, max, min float32
	max = data[0]
	min = data[0]
	activeCount := 0

	for _, v := range data {
		sum += v
		if v > max {
			max = v
		}
		if v < min {
			min = v
		}
		if v > threshold {
			activeCount++
		}
	}

	return LayerStats{
		AvgActivation: sum / float32(len(data)),
		MaxActivation: max,
		MinActivation: min,
		ActiveNeurons: activeCount,
		TotalNeurons:  len(data),
		LayerType:     layerType,
	}
}

// notifyObserver sends an event to the layer's observer if one exists
// This is a helper to reduce boilerplate in layer implementations
func notifyObserver(config *LayerConfig, eventType string, layerIdx int, input, output []float32, stepCount uint64) {
	if config.Observer == nil {
		return
	}

	stats := computeLayerStats(output, layerTypeString(config.Type), 0.0)

	event := LayerEvent{
		Type:      eventType,
		LayerIdx:  layerIdx,
		LayerType: config.Type,
		Stats:     stats,
		Input:     input,
		Output:    output,
		StepCount: stepCount,
		// Grid position and model ID from the config
		GridRow:   config.GridRow,
		GridCol:   config.GridCol,
		CellLayer: config.CellLayer,
		ModelID:   config.ModelID,
	}

	if eventType == "forward" {
		config.Observer.OnForward(event)
	} else {
		config.Observer.OnBackward(event)
	}
}

// layerTypeString converts LayerType to string for stats (internal observer use)
func layerTypeString(lt LayerType) string {
	switch lt {
	case LayerDense:
		return "dense"
	case LayerConv2D:
		return "conv2d"
	case LayerMultiHeadAttention:
		return "attention"
	case LayerRNN:
		return "rnn"
	case LayerLSTM:
		return "lstm"
	case LayerSoftmax:
		return "softmax"
	case LayerNorm:
		return "layernorm"
	case LayerRMSNorm:
		return "rmsnorm"
	case LayerSwiGLU:
		return "swiglu"
	case LayerParallel:
		return "parallel"
	case LayerResidual:
		return "residual"
	default:
		return "unknown"
	}
}

// =============================================================================
// Example Observer Implementations
// =============================================================================

// ConsoleObserver prints layer events to stdout
type ConsoleObserver struct {
	Verbose bool // If true, print input/output data (can be large!)
}

func (o *ConsoleObserver) OnForward(event LayerEvent) {
	modelPart := ""
	if event.ModelID != "" {
		modelPart = fmt.Sprintf("[%s] ", event.ModelID)
	}
	fmt.Printf("%s[FWD] Grid[%d,%d,%d] (%s): avg=%.4f max=%.4f active=%d/%d\n",
		modelPart,
		event.GridRow, event.GridCol, event.CellLayer,
		event.Stats.LayerType,
		event.Stats.AvgActivation, event.Stats.MaxActivation,
		event.Stats.ActiveNeurons, event.Stats.TotalNeurons)

	if o.Verbose && event.Output != nil && len(event.Output) <= 20 {
		fmt.Printf("       Output: %v\n", event.Output)
	}
}

func (o *ConsoleObserver) OnBackward(event LayerEvent) {
	modelPart := ""
	if event.ModelID != "" {
		modelPart = fmt.Sprintf("[%s] ", event.ModelID)
	}
	fmt.Printf("%s[BWD] Grid[%d,%d,%d] (%s): grad_avg=%.4f grad_max=%.4f\n",
		modelPart,
		event.GridRow, event.GridCol, event.CellLayer,
		event.Stats.LayerType,
		event.Stats.AvgActivation, event.Stats.MaxActivation)
}

// HTTPObserver sends layer events to an HTTP endpoint (for visualization)
type HTTPObserver struct {
	URL     string
	Timeout time.Duration
	client  *http.Client
}

func NewHTTPObserver(url string) *HTTPObserver {
	return &HTTPObserver{
		URL:     url,
		Timeout: 100 * time.Millisecond, // Fast timeout to not block training
		client: &http.Client{
			Timeout: 100 * time.Millisecond,
		},
	}
}

func (o *HTTPObserver) OnForward(event LayerEvent) {
	o.sendEvent(event)
}

func (o *HTTPObserver) OnBackward(event LayerEvent) {
	o.sendEvent(event)
}

func (o *HTTPObserver) sendEvent(event LayerEvent) {
	// Don't include raw data in HTTP to keep payloads small
	eventCopy := event
	eventCopy.Input = nil
	eventCopy.Output = nil

	data, err := json.Marshal(eventCopy)
	if err != nil {
		return
	}

	// Fire and forget (non-blocking)
	go func() {
		resp, err := o.client.Post(o.URL, "application/json", bytes.NewReader(data))
		if err == nil && resp != nil {
			resp.Body.Close()
		}
	}()
}

// ChannelObserver sends events to a Go channel (for internal processing)
type ChannelObserver struct {
	Events chan LayerEvent
}

func NewChannelObserver(bufferSize int) *ChannelObserver {
	return &ChannelObserver{
		Events: make(chan LayerEvent, bufferSize),
	}
}

func (o *ChannelObserver) OnForward(event LayerEvent) {
	select {
	case o.Events <- event:
	default:
		// Channel full, drop event to avoid blocking
	}
}

func (o *ChannelObserver) OnBackward(event LayerEvent) {
	select {
	case o.Events <- event:
	default:
		// Channel full, drop event to avoid blocking
	}
}

// RecordingObserver collects all layer events for saving to a file
type RecordingObserver struct {
	Events    []LayerEvent
	ModelID   string
	StartTime time.Time
}

// RecordedActivity represents the full recording of neural activity
type RecordedActivity struct {
	ModelID     string       `json:"model_id"`
	RecordedAt  string       `json:"recorded_at"`
	Duration    float64      `json:"duration_seconds"`
	TotalEvents int          `json:"total_events"`
	Events      []LayerEvent `json:"events"`
}

func NewRecordingObserver(modelID string) *RecordingObserver {
	return &RecordingObserver{
		Events:    make([]LayerEvent, 0, 1000),
		ModelID:   modelID,
		StartTime: time.Now(),
	}
}

func (o *RecordingObserver) OnForward(event LayerEvent) {
	// Clone event without full data arrays to save space
	eventCopy := event
	// Keep only summary stats, not full input/output
	if len(eventCopy.Input) > 20 {
		eventCopy.Input = eventCopy.Input[:20] // Sample
	}
	if len(eventCopy.Output) > 20 {
		eventCopy.Output = eventCopy.Output[:20] // Sample
	}
	o.Events = append(o.Events, eventCopy)
}

func (o *RecordingObserver) OnBackward(event LayerEvent) {
	eventCopy := event
	if len(eventCopy.Input) > 20 {
		eventCopy.Input = eventCopy.Input[:20]
	}
	if len(eventCopy.Output) > 20 {
		eventCopy.Output = eventCopy.Output[:20]
	}
	o.Events = append(o.Events, eventCopy)
}

// GetRecording returns the full recorded activity
func (o *RecordingObserver) GetRecording() RecordedActivity {
	return RecordedActivity{
		ModelID:     o.ModelID,
		RecordedAt:  o.StartTime.Format(time.RFC3339),
		Duration:    time.Since(o.StartTime).Seconds(),
		TotalEvents: len(o.Events),
		Events:      o.Events,
	}
}

// Reset clears the recorded events
func (o *RecordingObserver) Reset() {
	o.Events = o.Events[:0]
	o.StartTime = time.Now()
}
