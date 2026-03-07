package poly

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sync"
	"time"
)

// PolyObserver defines the interface for tracking neural activity in polymorphic layers.
type PolyObserver interface {
	OnForward(event PolyLayerEvent)
	OnBackward(event PolyLayerEvent)
}

// PolyLayerEvent captures state during a forward or backward pass.
type PolyLayerEvent struct {
	Mode      string      `json:"mode"`
	Type      string      `json:"type"` // "forward" or "backward"
	Z int `json:"z"`
	Y int `json:"y"`
	X int `json:"x"`
	L int `json:"l"`
	LayerType LayerType   `json:"layer_type"`
	Stats     LayerStats  `json:"stats"`
	StepCount uint64      `json:"step_count"`
	ModelID   string      `json:"model_id"`
}

// LayerStats provides summary statistics for a tensor's activations or gradients.
type LayerStats struct {
	Avg float32 `json:"avg"`
	Max float32 `json:"max"`
	Min float32 `json:"min"`
	Active int  `json:"active"`
	Total  int  `json:"total"`
}

// ComputeLayerStats calculates summary statistics for a tensor.
func ComputeLayerStats[T Numeric](t *Tensor[T]) LayerStats {
	if t == nil || len(t.Data) == 0 { return LayerStats{} }
	
	var sum, max, min float32
	max = float32(t.Data[0])
	min = float32(t.Data[0])
	active := 0

	for _, v := range t.Data {
		f := float32(v)
		sum += f
		if f > max { max = f }
		if f < min { min = f }
		if math.Abs(float64(f)) > 1e-6 { active++ }
	}

	return LayerStats{
		Avg:    sum / float32(len(t.Data)),
		Max:    max,
		Min:    min,
		Active: active,
		Total:  len(t.Data),
	}
}

// =============================================================================
// Observer Implementations
// =============================================================================

// ConsoleObserver prints events to stdout.
type ConsoleObserver struct{}

func (o *ConsoleObserver) OnForward(e PolyLayerEvent) {
	fmt.Printf("[FWD] (%d,%d,%d,%d) %v: avg=%.4f max=%.4f active=%d/%d\n",
		e.Z, e.Y, e.X, e.L, e.LayerType, e.Stats.Avg, e.Stats.Max, e.Stats.Active, e.Stats.Total)
}

func (o *ConsoleObserver) OnBackward(e PolyLayerEvent) {
	fmt.Printf("[BWD] (%d,%d,%d,%d) %v: grad_avg=%.4f grad_max=%.4f\n",
		e.Z, e.Y, e.X, e.L, e.LayerType, e.Stats.Avg, e.Stats.Max)
}

// HTTPObserver sends events to an HTTP endpoint.
type HTTPObserver struct {
	URL    string
	client *http.Client
}

func NewHTTPObserver(url string) *HTTPObserver {
	return &HTTPObserver{
		URL: url,
		client: &http.Client{Timeout: 100 * time.Millisecond},
	}
}

func (o *HTTPObserver) OnForward(e PolyLayerEvent) { o.send(e) }
func (o *HTTPObserver) OnBackward(e PolyLayerEvent) { o.send(e) }

func (o *HTTPObserver) send(e PolyLayerEvent) {
	data, _ := json.Marshal(e)
	go func() {
		resp, err := o.client.Post(o.URL, "application/json", bytes.NewReader(data))
		if err == nil && resp != nil { resp.Body.Close() }
	}()
}
// AggregatingObserver collects statistics over time windows.
type AggregatingObserver struct {
	WindowSize int
	History    []LayerStats
	Events     []PolyLayerEvent
	mu         sync.Mutex
}

func NewAggregatingObserver(windowSize int) *AggregatingObserver {
	return &AggregatingObserver{
		WindowSize: windowSize,
		History:    make([]LayerStats, 0),
		Events:     make([]PolyLayerEvent, 0),
	}
}

func (o *AggregatingObserver) OnForward(e PolyLayerEvent) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.Events = append(o.Events, e)
	if len(o.Events) >= o.WindowSize {
		// Aggregate
		var avg, max, min float32
		active := 0
		for _, ev := range o.Events {
			avg += ev.Stats.Avg
			if ev.Stats.Max > max { max = ev.Stats.Max }
			if ev.Stats.Min < min { min = ev.Stats.Min }
			active += ev.Stats.Active
		}
		n := float32(len(o.Events))
		o.History = append(o.History, LayerStats{
			Avg:    avg / n,
			Max:    max,
			Min:    min,
			Active: active / len(o.Events),
			Total:  e.Stats.Total,
		})
		o.Events = nil
	}
}

func (o *AggregatingObserver) OnBackward(e PolyLayerEvent) {}

// PolyGradientObserver tracks gradient flow through layers.
type PolyGradientObserver interface {
	OnGradient(event PolyLayerEvent)
}
