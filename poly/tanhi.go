package poly

import (
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"time"
)

// DefaultTanhiUDPPort is the default client destination (non-blocking send). IANA: unassigned range.
const DefaultTanhiUDPPort = 17481

// TanhiUDPConfig controls JSON line telemetry to a UDP listener (e.g. SoulGlitch TANHI HUD).
// Host defaults to 127.0.0.1, Port to DefaultTanhiUDPPort when zero.
type TanhiUDPConfig struct {
	Enabled   bool
	Host      string
	Port      int
	SendShape bool
}

// tanhiCoord is a single volumetric address (Z,Y,X,L) for routing visualization.
type tanhiCoord struct {
	Z int `json:"z"`
	Y int `json:"y"`
	X int `json:"x"`
	L int `json:"l"`
}

type tanhiWire struct {
	V           string      `json:"v"`
	Seq         uint64      `json:"seq"`
	Phase       string      `json:"phase"`
	Idx         int         `json:"idx"`
	Z           int         `json:"z"`
	Y           int         `json:"y"`
	X           int         `json:"x"`
	L           int         `json:"l"`
	Layer       string      `json:"layer"`
	DType       int         `json:"dtype"`
	Connections int         `json:"connections"`
	T0Ns        int64       `json:"t0_ns"`
	T1Ns        int64       `json:"t1_ns"`
	Shape       []int       `json:"shape,omitempty"`
	Links       []tanhiCoord `json:"links,omitempty"` // parallel / sequential routing targets (SoulGlitch arcs)
}

var (
	tanhiSeq      atomic.Uint64
	tanhiQueue    = make(chan tanhiPacket, 1024)
	tanhiInitOnce sync.Once
	tanhiAddrMu   sync.RWMutex
	tanhiAddr     = make(map[string]*net.UDPAddr)
)

type tanhiPacket struct {
	addr *net.UDPAddr
	data []byte
}

func tanhiWriterLoop() {
	var pc net.PacketConn
	for pkt := range tanhiQueue {
		if pc == nil {
			c, err := net.ListenPacket("udp", ":0")
			if err != nil {
				continue
			}
			pc = c
		}
		_, _ = pc.WriteTo(pkt.data, pkt.addr)
	}
}

func tanhiEnsureWriter() {
	tanhiInitOnce.Do(func() {
		go tanhiWriterLoop()
	})
}

func tanhiResolveUDPAddr(cfg *TanhiUDPConfig) *net.UDPAddr {
	if cfg == nil {
		return nil
	}
	host := cfg.Host
	if host == "" {
		host = "127.0.0.1"
	}
	port := cfg.Port
	if port == 0 {
		port = DefaultTanhiUDPPort
	}
	key := fmt.Sprintf("%s:%d", host, port)
	tanhiAddrMu.RLock()
	a, ok := tanhiAddr[key]
	tanhiAddrMu.RUnlock()
	if ok {
		return a
	}
	addr, err := net.ResolveUDPAddr("udp", key)
	if err != nil {
		return nil
	}
	tanhiAddrMu.Lock()
	tanhiAddr[key] = addr
	tanhiAddrMu.Unlock()
	return addr
}

func tanhiConnectionCount(layer *VolumetricLayer) int {
	if layer == nil || layer.WeightStore == nil || layer.WeightStore.Master == nil {
		return 0
	}
	return len(layer.WeightStore.Master)
}

func tanhiCoordFromRoutingTarget(ref *VolumetricLayer) tanhiCoord {
	if ref == nil {
		return tanhiCoord{}
	}
	if ref.IsRemoteLink {
		return tanhiCoord{Z: ref.TargetZ, Y: ref.TargetY, X: ref.TargetX, L: ref.TargetL}
	}
	return tanhiCoord{Z: ref.Z, Y: ref.Y, X: ref.X, L: ref.L}
}

// tanhiRoutingLinks lists distinct volumetric endpoints for parallel branches and sequential substeps
// (remote links use TargetZ,Y,X,L). Capped for UDP size.
func tanhiRoutingLinks(layer *VolumetricLayer) []tanhiCoord {
	if layer == nil {
		return nil
	}
	seen := make(map[tanhiCoord]struct{})
	var out []tanhiCoord
	add := func(c tanhiCoord) {
		if _, ok := seen[c]; ok {
			return
		}
		seen[c] = struct{}{}
		out = append(out, c)
	}
	switch layer.Type {
	case LayerParallel:
		for i := range layer.ParallelBranches {
			add(tanhiCoordFromRoutingTarget(&layer.ParallelBranches[i]))
		}
	case LayerSequential:
		for i := range layer.SequentialLayers {
			add(tanhiCoordFromRoutingTarget(&layer.SequentialLayers[i]))
		}
	}
	const maxLinks = 48
	if len(out) > maxLinks {
		out = out[:maxLinks]
	}
	return out
}

// TanhiGPULayerShapeHint returns an approximate tensor shape for telemetry (no GPU readback).
func TanhiGPULayerShapeHint(layer *VolumetricLayer, numTokens int) []int {
	if layer == nil || numTokens <= 0 {
		return nil
	}
	switch layer.Type {
	case LayerRMSNorm:
		return []int{numTokens, layer.InputHeight}
	case LayerMultiHeadAttention:
		return []int{numTokens, layer.DModel}
	case LayerSwiGLU:
		return []int{numTokens, layer.InputHeight}
	case LayerDense:
		if layer.OutputHeight > 0 && layer.InputHeight > 0 {
			return []int{numTokens, layer.OutputHeight}
		}
		return []int{numTokens, layer.InputHeight}
	case LayerEmbedding:
		if layer.EmbeddingDim > 0 {
			return []int{numTokens, layer.EmbeddingDim}
		}
		return []int{numTokens, layer.InputHeight}
	default:
		if layer.OutputHeight > 0 {
			return []int{numTokens, layer.OutputHeight}
		}
		return []int{numTokens, layer.InputHeight}
	}
}

// tanhiEmit records one layer boundary event (non-blocking; drops if queue is full).
func tanhiEmit(n *VolumetricNetwork, phase string, idx int, layer *VolumetricLayer, t0, t1 time.Time, shape []int) {
	tanhiEmitWithConn(n, phase, idx, layer, t0, t1, shape, -1)
}

// connOverride >= 0 replaces WeightStore-based connection count (e.g. GPU LM head / tied weights).
func tanhiEmitWithConn(n *VolumetricNetwork, phase string, idx int, layer *VolumetricLayer, t0, t1 time.Time, shape []int, connOverride int) {
	if n == nil || n.Tanhi == nil || !n.Tanhi.Enabled || layer == nil {
		return
	}
	addr := tanhiResolveUDPAddr(n.Tanhi)
	if addr == nil {
		return
	}
	conn := tanhiConnectionCount(layer)
	if connOverride >= 0 {
		conn = connOverride
	}
	w := tanhiWire{
		V:           "tanhi1",
		Seq:         tanhiSeq.Add(1),
		Phase:       phase,
		Idx:         idx,
		Z:           layer.Z,
		Y:           layer.Y,
		X:           layer.X,
		L:           layer.L,
		Layer:       layer.Type.String(),
		DType:       int(layer.DType),
		Connections: conn,
		T0Ns:        t0.UnixNano(),
		T1Ns:        t1.UnixNano(),
	}
	if n.Tanhi.SendShape && len(shape) > 0 {
		w.Shape = shape
	}
	if links := tanhiRoutingLinks(layer); len(links) > 0 {
		w.Links = links
	}
	line, err := json.Marshal(w)
	if err != nil {
		return
	}
	line = append(line, '\n')
	tanhiEnsureWriter()
	select {
	case tanhiQueue <- tanhiPacket{addr: addr, data: line}:
	default:
	}
}
