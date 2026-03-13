package pods

import "math"

type STFTIn struct {
	Audio   AudioBuffer
	WinSize int
	Hop     int
	Window  string // "hann" (default), …
}
type STFTOut struct {
	Spectrogram [][]complex64 // [frame][bin]
}

type STFTPod struct{}

func (STFTPod) Name() string { return "audio/stft" }

func (STFTPod) Run(_ *ExecContext, in any) (any, error) {
	a := in.(STFTIn)
	if a.WinSize == 0 {
		a.WinSize = 1024
	}
	if a.Hop == 0 {
		a.Hop = a.WinSize / 4
	}
	win := make([]float64, a.WinSize)
	for i := range win {
		switch a.Window {
		default: // hann
			win[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(a.WinSize))
		}
	}
	mono := a.Audio.Samples[0]
	var frames [][]complex64
	for off := 0; off+a.WinSize <= len(mono); off += a.Hop {
		frame := make([]complex64, a.WinSize)
		for i := 0; i < a.WinSize; i++ {
			frame[i] = complex64(complex(mono[off+i]*float32(win[i]), 0))
		}
		// Naive DFT (O(N^2)) placeholder; swap for FFT later.
		out := make([]complex64, a.WinSize)
		for k := 0; k < a.WinSize; k++ {
			var sum complex128
			for n := 0; n < a.WinSize; n++ {
				angle := -2 * math.Pi * float64(k*n) / float64(a.WinSize)
				sum += complex(float64(real(frame[n])), 0) * complex(math.Cos(angle), math.Sin(angle))
			}
			out[k] = complex64(sum)
		}
		frames = append(frames, out)
	}
	return STFTOut{Spectrogram: frames}, nil
}
