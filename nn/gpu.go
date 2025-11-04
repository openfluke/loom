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

// generateConv2DForwardShader generates WGSL shader code for Conv2D forward pass
func generateConv2DForwardShader(wgx uint32, activation int, batchSize, inC, inH, inW, outH, outW, filters, kSize, stride, padding int) string {
	var activationCode string

	switch activation {
	case 0: // ActivationScaledReLU
		activationCode = `
fn activate(v: f32) -> f32 {
    return max(0.0, 1.1 * v);
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
    if (v >= 0.0) { return v; }
    return 0.1 * v;
}`
	}

	return fmt.Sprintf(`
@group(0) @binding(0) var<storage, read>  input  : array<f32>;
@group(0) @binding(1) var<storage, read>  kernel : array<f32>;
@group(0) @binding(2) var<storage, read>  bias   : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;

const batchSize: u32 = %du;
const inC: u32 = %du;
const inH: u32 = %du;
const inW: u32 = %du;
const outH: u32 = %du;
const outW: u32 = %du;
const filters: u32 = %du;
const kSize: u32 = %du;
const stride: u32 = %du;
const padding: i32 = %d;

%s

@compute @workgroup_size(%d, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let totalOutputs = batchSize * filters * outH * outW;
    if (idx >= totalOutputs) { return; }
    
    // Decode indices: batch, filter, oh, ow
    let b = idx / (filters * outH * outW);
    let rem1 = idx %% (filters * outH * outW);
    let f = rem1 / (outH * outW);
    let rem2 = rem1 %% (outH * outW);
    let oh = rem2 / outW;
    let ow = rem2 %% outW;
    
    var sum: f32 = bias[f];
    
    for (var ic: u32 = 0u; ic < inC; ic = ic + 1u) {
        for (var kh: u32 = 0u; kh < kSize; kh = kh + 1u) {
            for (var kw: u32 = 0u; kw < kSize; kw = kw + 1u) {
                let ih = i32(oh * stride + kh) - padding;
                let iw = i32(ow * stride + kw) - padding;
                
                if (ih >= 0 && ih < i32(inH) && iw >= 0 && iw < i32(inW)) {
                    let inputIdx = b * inC * inH * inW + ic * inH * inW + u32(ih) * inW + u32(iw);
                    let kernelIdx = f * inC * kSize * kSize + ic * kSize * kSize + kh * kSize + kw;
                    sum = sum + input[inputIdx] * kernel[kernelIdx];
                }
            }
        }
    }
    
    output[idx] = activate(sum);
}
`, batchSize, inC, inH, inW, outH, outW, filters, kSize, stride, padding, activationCode, wgx)
}

// generateConv2DBackwardShader generates WGSL shader code for Conv2D backward pass
func generateConv2DBackwardShader(wgx uint32, activation int, batchSize, inC, inH, inW, outH, outW, filters, kSize, stride, padding int) string {
	var derivativeCode string

	switch activation {
	case 0: // ActivationScaledReLU
		derivativeCode = `
fn activate_derivative(pre_v: f32) -> f32 {
    if (pre_v > 0.0) { return 1.1; }
    return 0.0;
}`
	case 1: // ActivationSigmoid
		derivativeCode = `
fn activate_derivative(pre_v: f32) -> f32 {
    let sig = 1.0 / (1.0 + exp(-pre_v));
    return sig * (1.0 - sig);
}`
	case 2: // ActivationTanh
		derivativeCode = `
fn activate_derivative(pre_v: f32) -> f32 {
    let e2x = exp(2.0 * pre_v);
    let t = (e2x - 1.0) / (e2x + 1.0);
    return 1.0 - t * t;
}`
	case 3: // ActivationSoftplus
		derivativeCode = `
fn activate_derivative(pre_v: f32) -> f32 {
    return 1.0 / (1.0 + exp(-pre_v));
}`
	case 4: // ActivationLeakyReLU
		derivativeCode = `
fn activate_derivative(pre_v: f32) -> f32 {
    if (pre_v >= 0.0) { return 1.0; }
    return 0.1;
}`
	}

	return fmt.Sprintf(`
@group(0) @binding(0) var<storage, read>  grad_output    : array<f32>;
@group(0) @binding(1) var<storage, read>  input          : array<f32>;
@group(0) @binding(2) var<storage, read>  kernel         : array<f32>;
@group(0) @binding(3) var<storage, read>  pre_activation : array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_input     : array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_kernel    : array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_bias      : array<f32>;

const batchSize: u32 = %du;
const inC: u32 = %du;
const inH: u32 = %du;
const inW: u32 = %du;
const outH: u32 = %du;
const outW: u32 = %du;
const filters: u32 = %du;
const kSize: u32 = %du;
const stride: u32 = %du;
const padding: i32 = %d;

%s

@compute @workgroup_size(%d, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let totalInputs = batchSize * inC * inH * inW;
    if (idx >= totalInputs) { return; }
    
    // Decode input indices: batch, channel, ih, iw
    let b = idx / (inC * inH * inW);
    let rem1 = idx %% (inC * inH * inW);
    let ic = rem1 / (inH * inW);
    let rem2 = rem1 %% (inH * inW);
    let ih = rem2 / inW;
    let iw = rem2 %% inW;
    
    var grad_sum: f32 = 0.0;
    
    // Accumulate gradients from all output positions that used this input
    for (var f: u32 = 0u; f < filters; f = f + 1u) {
        for (var kh: u32 = 0u; kh < kSize; kh = kh + 1u) {
            for (var kw: u32 = 0u; kw < kSize; kw = kw + 1u) {
                // Which output position uses this input?
                // ih = oh * stride + kh - padding
                // => oh = (ih + padding - kh) / stride
                let numerator_h = i32(ih) + padding - i32(kh);
                let numerator_w = i32(iw) + padding - i32(kw);
                
                if (numerator_h >= 0 && numerator_w >= 0 &&
                    numerator_h %% i32(stride) == 0 && numerator_w %% i32(stride) == 0) {
                    let oh = u32(numerator_h) / stride;
                    let ow = u32(numerator_w) / stride;
                    
                    if (oh < outH && ow < outW) {
                        let outputIdx = b * filters * outH * outW + f * outH * outW + oh * outW + ow;
                        let kernelIdx = f * inC * kSize * kSize + ic * kSize * kSize + kh * kSize + kw;
                        
                        let grad_out = grad_output[outputIdx];
                        let pre_act = pre_activation[outputIdx];
                        let derivative = activate_derivative(pre_act);
                        
                        grad_sum = grad_sum + grad_out * derivative * kernel[kernelIdx];
                    }
                }
            }
        }
    }
    
    grad_input[idx] = grad_sum;
}
`, batchSize, inC, inH, inW, outH, outW, filters, kSize, stride, padding, derivativeCode, wgx)
}
