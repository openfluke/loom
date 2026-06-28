package poly

import (
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

type memoryChartSeries struct {
	label string
	glyph rune
	value func(MemorySample) float64
}

func memoryChartSeriesDefs() []memoryChartSeries {
	return []memoryChartSeries{
		{label: "host weights (Poly)", glyph: 'H', value: func(s MemorySample) float64 { return s.HostWeightsMB }},
		{label: "GPU weights (Poly)", glyph: 'G', value: func(s MemorySample) float64 { return s.GPUWeightsMB }},
		{label: "process RSS", glyph: 'R', value: func(s MemorySample) float64 { return s.ProcessRSSMB }},
		{label: "VRAM total (Poly)", glyph: 'V', value: func(s MemorySample) float64 { return s.VRAMTotalMB }},
	}
}

func terminalColumns() int {
	if c := strings.TrimSpace(os.Getenv("COLUMNS")); c != "" {
		if n, err := strconv.Atoi(c); err == nil && n >= 60 {
			if n > 120 {
				return 120
			}
			return n
		}
	}
	return 80
}

func seriesValues(samples []MemorySample, pick func(MemorySample) float64) []float64 {
	out := make([]float64, len(samples))
	for i, s := range samples {
		out[i] = pick(s)
	}
	return out
}

func chartMaxY(samples []MemorySample, defs []memoryChartSeries) float64 {
	maxY := 1.0
	for _, s := range samples {
		for _, def := range defs {
			maxY = math.Max(maxY, def.value(s))
		}
	}
	return maxY * 1.05
}

func renderBlockSparkline(values []float64, width int, maxY float64) string {
	if width < 2 || len(values) == 0 {
		return ""
	}
	// ASCII ramp — renders on all terminals (Unicode ▁▇ blocks show as mojibake in many shells).
	const blocks = " .:-=+*#"
	if maxY <= 0 {
		maxY = 1
	}
	out := make([]byte, width)
	for col := 0; col < width; col++ {
		idx := 0
		if width > 1 && len(values) > 1 {
			idx = int(float64(col) / float64(width-1) * float64(len(values)-1))
		}
		v := values[idx]
		if v < 0 {
			v = 0
		}
		bi := int((v / maxY) * 7)
		if bi > 7 {
			bi = 7
		}
		out[col] = blocks[bi]
	}
	return string(out)
}

func renderBrailleChart(samples []MemorySample, defs []memoryChartSeries, widthChars, heightChars int, maxY float64) string {
	if len(samples) < 2 || widthChars < 2 || heightChars < 2 {
		return ""
	}
	if maxY <= 0 {
		maxY = 1
	}

	pixelW := widthChars * 2
	pixelH := heightChars * 4
	grid := make([][]uint8, heightChars)
	for y := 0; y < heightChars; y++ {
		grid[y] = make([]uint8, widthChars)
	}

	setPixel := func(px, py int) {
		if px < 0 || py < 0 || px >= pixelW || py >= pixelH {
			return
		}
		cx := px / 2
		cy := py / 4
		dx := px % 2
		dy := py % 4
		dot := brailleDotMask[dy*2+dx]
		grid[cy][cx] |= dot
	}

	for _, def := range defs {
		for i, sample := range samples {
			px := 0
			if len(samples) > 1 {
				px = int(float64(i) / float64(len(samples)-1) * float64(pixelW-1))
			}
			v := def.value(sample)
			if v < 0 {
				v = 0
			}
			py := pixelH - 1 - int((v/maxY)*float64(pixelH-1))
			setPixel(px, py)
			if i > 0 {
				prev := samples[i-1]
				prevPx := int(float64(i-1) / float64(len(samples)-1) * float64(pixelW-1))
				prevV := def.value(prev)
				if prevV < 0 {
					prevV = 0
				}
				prevPy := pixelH - 1 - int((prevV/maxY)*float64(pixelH-1))
				drawBrailleLine(setPixel, prevPx, prevPy, px, py)
			}
		}
	}

	var b strings.Builder
	for y := 0; y < heightChars; y++ {
		for x := 0; x < widthChars; x++ {
			b.WriteRune(brailleBase + rune(grid[y][x]))
		}
		if y+1 < heightChars {
			b.WriteByte('\n')
		}
	}
	return b.String()
}

var brailleBase = rune(0x2800)

var brailleDotMask = [8]uint8{0x01, 0x08, 0x02, 0x10, 0x04, 0x20, 0x40, 0x80}

func drawBrailleLine(set func(px, py int), x0, y0, x1, y1 int) {
	dx := absInt(x1 - x0)
	sx := -1
	if x0 < x1 {
		sx = 1
	}
	dy := -absInt(y1 - y0)
	sy := -1
	if y0 < y1 {
		sy = 1
	}
	err := dx + dy
	for {
		set(x0, y0)
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func formatMB(v float64) string {
	switch {
	case v >= 1024:
		return fmt.Sprintf("%.2f GB", v/1024)
	default:
		return fmt.Sprintf("%.0f MB", v)
	}
}

// WriteTerminalChart renders a multi-series memory timeline for the terminal.
func (h *MemoryHistory) WriteTerminalChart(w io.Writer) {
	samples := h.Samples()
	if len(samples) == 0 {
		return
	}

	defs := memoryChartSeriesDefs()
	maxY := chartMaxY(samples, defs)
	cols := terminalColumns()
	sparkW := cols - 18
	if sparkW < 24 {
		sparkW = 24
	}
	brailleW := sparkW
	brailleH := 10

	fmt.Fprintf(w, "\n📈 Memory history (%s)\n", h.sessionName())
	fmt.Fprintf(w, "   %d samples · %.1fs · shared scale 0 → %s\n\n", len(samples), samples[len(samples)-1].ElapsedSec, formatMB(maxY))

	if len(samples) >= 2 {
		fmt.Fprintln(w, renderBrailleChart(samples, defs, brailleW, brailleH, maxY))
		fmt.Fprintf(w, "   %5.1fs%*s%5.1fs\n\n", samples[0].ElapsedSec, brailleW-10, "", samples[len(samples)-1].ElapsedSec)
	}

	for _, def := range defs {
		values := seriesValues(samples, def.value)
		peak := 0.0
		for _, v := range values {
			peak = math.Max(peak, v)
		}
		line := renderBlockSparkline(values, sparkW, maxY)
		fmt.Fprintf(w, " %c %s\n   %s  peak %s\n", def.glyph, def.label, line, formatMB(peak))
	}

	minT := samples[0].ElapsedSec
	maxT := samples[len(samples)-1].ElapsedSec
	fmt.Fprintf(w, "   %5.1fs%*s%5.1fs\n", minT, sparkW-10, "", maxT)
}
