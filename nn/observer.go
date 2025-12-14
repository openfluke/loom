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
	fmt.Printf("[FWD] Layer %d (%s): avg=%.4f max=%.4f active=%d/%d\n",
		event.LayerIdx, event.Stats.LayerType,
		event.Stats.AvgActivation, event.Stats.MaxActivation,
		event.Stats.ActiveNeurons, event.Stats.TotalNeurons)

	if o.Verbose && event.Output != nil && len(event.Output) <= 20 {
		fmt.Printf("       Output: %v\n", event.Output)
	}
}

func (o *ConsoleObserver) OnBackward(event LayerEvent) {
	fmt.Printf("[BWD] Layer %d (%s): grad_avg=%.4f grad_max=%.4f\n",
		event.LayerIdx, event.Stats.LayerType,
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
