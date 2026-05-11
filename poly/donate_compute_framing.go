package poly

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
)

const (
	// DonateComputeDefaultPort sits beside construct TCP dev (17000).
	DonateComputeDefaultPort = 17001
	// MaxDonateFrameBytes caps one framed JSON payload (model chunks use multiple frames).
	MaxDonateFrameBytes = 64 << 20
)

// WriteDonateFrame writes u32 little-endian length + JSON bytes to w.
func WriteDonateFrame(w io.Writer, v any) error {
	body, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if len(body) > MaxDonateFrameBytes {
		return errors.New("donate_compute: frame exceeds MaxDonateFrameBytes")
	}
	var hdr [4]byte
	binary.LittleEndian.PutUint32(hdr[:], uint32(len(body)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err = w.Write(body)
	return err
}

// ReadDonateFrame reads one length-prefixed JSON object into dest (must be pointer).
func ReadDonateFrame(r io.Reader, dest any) error {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return err
	}
	n := binary.LittleEndian.Uint32(hdr[:])
	if n == 0 || n > MaxDonateFrameBytes {
		return errors.New("donate_compute: invalid frame length")
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return err
	}
	return json.Unmarshal(buf, dest)
}
