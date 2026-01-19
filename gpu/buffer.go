package gpu

import (
	"fmt"
	"time"

	"github.com/openfluke/webgpu/wgpu"
)

// EnsureGPU ensures the GPU context is initialized
func EnsureGPU() error {
	_, err := GetContext()
	return err
}

// NewFloatBuffer creates a buffer with the given float32 data
func NewFloatBuffer(data []float32, usage wgpu.BufferUsage) (*wgpu.Buffer, error) {
	c, err := GetContext()
	if err != nil {
		return nil, err
	}

	buf, err := c.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Contents: wgpu.ToBytes(data),
		Usage:    usage,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer: %v", err)
	}
	return buf, nil
}

// ReadBuffer safely reads an entire buffer
func ReadBuffer(buffer *wgpu.Buffer, size int) ([]float32, error) {
	c, err := GetContext()
	if err != nil {
		return nil, err
	}

	// Create staging buffer
	sizeBytes := uint64(size * 4)
	stagingBuf, err := c.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "ReadStaging",
		Size:  sizeBytes,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create staging buffer: %v", err)
	}
	defer stagingBuf.Destroy()

	// Copy to staging
	encoder, err := c.Device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create command encoder: %v", err)
	}
	encoder.CopyBufferToBuffer(buffer, 0, stagingBuf, 0, sizeBytes)
	cmd, err := encoder.Finish(nil)
	if err != nil {
		return nil, fmt.Errorf("failed to finish command: %v", err)
	}
	c.Queue.Submit(cmd)

	// Map and read
	done := make(chan struct{})
	var mapErr error

	err = stagingBuf.MapAsync(wgpu.MapModeRead, 0, sizeBytes, func(status wgpu.BufferMapAsyncStatus) {
		if status != wgpu.BufferMapAsyncStatusSuccess {
			mapErr = fmt.Errorf("map failed: %v", status)
		}
		close(done)
	})
	if err != nil {
		return nil, fmt.Errorf("MapAsync failed: %v", err)
	}

	// Poll until done
	// Poll until done
	timeout := time.After(2 * time.Second)
Loop:
	for {
		c.Device.Poll(false, nil) // use Poll(false) to just check without blocking indefinitely? Or better use true but only once per iteration.
		// Native Poll(true) waits for completion. If it hangs, we can't interrupt it easily unless we do Poll(false).
		// Let's use Poll(false) to be safe and sleep.

		select {
		case <-done:
			break Loop
		case <-timeout:
			return nil, fmt.Errorf("ReadBuffer timed out after 2s")
		default:
			time.Sleep(time.Millisecond) // Don't busy wait too hot
		}
	}

	if mapErr != nil {
		return nil, mapErr
	}

	data := stagingBuf.GetMappedRange(0, uint(sizeBytes))
	if data == nil {
		return nil, fmt.Errorf("failed to get mapped range")
	}

	// Copy data out
	result := make([]float32, size)
	copy(result, wgpu.FromBytes[float32](data))
	stagingBuf.Unmap()

	return result, nil
}
