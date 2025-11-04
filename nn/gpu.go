package nn

import (
	"fmt"

	"github.com/openfluke/loom/detector"
	"github.com/openfluke/webgpu/wgpu"
)

// InitGPU initializes GPU resources for the network
func (n *Network) InitGPU() error {
	rep, err := detector.Detect()
	if err != nil {
		return fmt.Errorf("detector: %w", err)
	}

	inst := wgpu.CreateInstance(nil)
	if inst == nil {
		return fmt.Errorf("CreateInstance nil")
	}

	pp := wgpu.PowerPreferenceHighPerformance
	if rep.AdapterType == "integrated-gpu" {
		pp = wgpu.PowerPreferenceLowPower
	}

	ad, err := inst.RequestAdapter(&wgpu.RequestAdapterOptions{PowerPreference: pp})
	if err != nil || ad == nil {
		inst.Release()
		return fmt.Errorf("RequestAdapter failed")
	}

	dev, err := ad.RequestDevice(&wgpu.DeviceDescriptor{})
	if err != nil || dev == nil {
		ad.Release()
		inst.Release()
		return fmt.Errorf("RequestDevice failed")
	}

	q := dev.GetQueue()

	wgx := rep.Recommended.WorkgroupX
	if wgx == 0 {
		wgx = 64
	}
	if wgx > rep.Limits.MaxComputeWorkgroupSizeX {
		wgx = rep.Limits.MaxComputeWorkgroupSizeX
	}

	n.deviceInfo = &GPUDeviceInfo{
		Device:     dev,
		Queue:      q,
		WorkgroupX: wgx,
		release: func() {
			dev.Release()
			ad.Release()
			inst.Release()
		},
	}

	return nil
}

// generateForwardShader generates WGSL shader code for forward pass activation
func generateForwardShader(wgx uint32, activation int, N int) string {
	var activationCode string

	switch activation {
	case 0: // ActivationScaledReLU
		activationCode = `
fn activate(v: f32) -> f32 {
    let scaled = v * 1.1;
    return max(0.0, scaled);
}`
	case 1: // ActivationSigmoid
		activationCode = `
fn activate(v: f32) -> f32 {
    return 1.0 / (1.0 + exp(-v));
}`
	case 2: // ActivationTanh
		activationCode = `
fn activate(v: f32) -> f32 {
    let e2x = exp(2.0 * v);
    return (e2x - 1.0) / (e2x + 1.0);
}`
	case 3: // ActivationSoftplus
		activationCode = `
fn activate(v: f32) -> f32 {
    return log(1.0 + exp(v));
}`
	case 4: // ActivationLeakyReLU
		activationCode = `
fn activate(v: f32) -> f32 {
    if (v < 0.0) {
        return v * 0.1;
    }
    return v;
}`
	}

	return fmt.Sprintf(`
@group(0) @binding(0) var<storage, read>        src : array<f32>;
@group(0) @binding(1) var<storage, read_write>  dst : array<f32>;

const N: u32 = %du;

%s

@compute @workgroup_size(%d, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N) { return; }
    
    let v = src[i];
    dst[i] = activate(v);
}
`, N, activationCode, wgx)
}

// generateBackwardShader generates WGSL shader code for backward pass (gradient computation)
func generateBackwardShader(wgx uint32, activation int, N int) string {
	var derivativeCode string

	switch activation {
	case 0: // ActivationScaledReLU
		derivativeCode = `
fn activate_derivative(pre_v: f32) -> f32 {
    // d/dv (max(0, 1.1*v)) = 1.1 if v > 0, else 0
    if (pre_v > 0.0) {
        return 1.1;
    }
    return 0.0;
}`
	case 1: // ActivationSigmoid
		derivativeCode = `
fn activate_derivative(pre_v: f32) -> f32 {
    // d/dv (1/(1+e^-v)) = sigmoid(v) * (1 - sigmoid(v))
    let sig = 1.0 / (1.0 + exp(-pre_v));
    return sig * (1.0 - sig);
}`
	case 2: // ActivationTanh
		derivativeCode = `
fn activate_derivative(pre_v: f32) -> f32 {
    // d/dv tanh(v) = 1 - tanh^2(v)
    let e2x = exp(2.0 * pre_v);
    let t = (e2x - 1.0) / (e2x + 1.0);
    return 1.0 - t * t;
}`
	case 3: // ActivationSoftplus
		derivativeCode = `
fn activate_derivative(pre_v: f32) -> f32 {
    // d/dv log(1 + e^v) = e^v / (1 + e^v) = sigmoid(v)
    return 1.0 / (1.0 + exp(-pre_v));
}`
	case 4: // ActivationLeakyReLU
		derivativeCode = `
fn activate_derivative(pre_v: f32) -> f32 {
    // d/dv (v if v >= 0, else 0.1*v) = 1 if v >= 0, else 0.1
    if (pre_v >= 0.0) {
        return 1.0;
    }
    return 0.1;
}`
	}

	return fmt.Sprintf(`
@group(0) @binding(0) var<storage, read>        grad_in       : array<f32>;
@group(0) @binding(1) var<storage, read>        pre_activation: array<f32>;
@group(0) @binding(2) var<storage, read_write>  grad_out      : array<f32>;

const N: u32 = %du;

%s

@compute @workgroup_size(%d, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N) { return; }
    
    let grad = grad_in[i];
    let pre_v = pre_activation[i];
    let derivative = activate_derivative(pre_v);
    
    grad_out[i] = grad * derivative;
}
`, N, derivativeCode, wgx)
}
