#include "cpu_reference.hpp"
#include "half.hpp"

#include <cmath>
#include <cstring>

namespace loom_apple {
namespace {

float decode_f32_le(const uint8_t* p) {
    uint32_t bits = uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t decode_u16_le(const uint8_t* p) {
    return uint16_t(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
}

// Constant fill values mirror accel/intel/src/layer_models.cpp const_f() defaults.
float constant_fill(const std::string& name) {
    if (name == "MHA-MatMul") {
        return 0.01f;
    }
    return 0.02f;  // MatMul, Conv1D, Conv2D, DepthwiseConv
}

}  // namespace

std::vector<float> resolve_weights(
    const std::string& name,
    const ShapeSpec& s,
    const void* weight_bytes,
    size_t weight_byte_len,
    const std::string& dtype_label) {
    const size_t count = weight_float_count(name, s);
    if (count == 0) {
        return {};  // op uses inline constants (activations, norms, add, mul, pool)
    }

    std::vector<float> w(count, constant_fill(name));

    if (weight_bytes == nullptr || weight_byte_len == 0) {
        return w;  // Loom uploaded nothing → constant fill (matches intel default graph)
    }

    const auto* p = static_cast<const uint8_t*>(weight_bytes);
    if (dtype_label == "FP16") {
        if (weight_byte_len >= count * 2) {
            for (size_t i = 0; i < count; ++i) {
                w[i] = half_to_float(decode_u16_le(p + i * 2));
            }
        }
    } else if (dtype_label == "BF16") {
        if (weight_byte_len >= count * 2) {
            for (size_t i = 0; i < count; ++i) {
                w[i] = bfloat16_to_float(decode_u16_le(p + i * 2));
            }
        }
    } else {  // FP32 / INT8 / INT16 / INT4 all upload FP32 values (see poly/accel_intel.go)
        if (weight_byte_len >= count * 4) {
            for (size_t i = 0; i < count; ++i) {
                w[i] = decode_f32_le(p + i * 4);
            }
        }
    }
    return w;
}

bool cpu_forward(const Prepared& p, const float* in, size_t in_n, float* out, size_t out_n, std::string* err) {
    const ShapeSpec& s = p.spec;
    const std::string& name = p.name;

    auto fail = [&](const char* m) {
        if (err) {
            *err = m;
        }
        return false;
    };

    if (in == nullptr || out == nullptr) {
        return fail("null in/out");
    }
    if (in_n != p.in_elems || out_n != p.out_elems) {
        return fail("element count mismatch");
    }

    // ---- MatMul / MHA-MatMul: out[b,j] = sum_k in[b,k] * W[k,j] ----
    if (name == "MatMul" || name == "MHA-MatMul") {
        const int B = s.dense_batch, D = s.dim;
        const float* W = p.weights.data();
        for (int b = 0; b < B; ++b) {
            for (int j = 0; j < D; ++j) {
                float acc = 0.0f;
                for (int k = 0; k < D; ++k) {
                    acc += in[b * D + k] * W[k * D + j];
                }
                out[b * D + j] = acc;
            }
        }
        return true;
    }

    // ---- Conv1D (cross-correlation, pad=1, k=3, stride=1) → flat [B, F*L] ----
    if (name == "Conv1D") {
        const int B = s.c1_batch, IC = s.c1_in_c, L = s.c1_len;
        const int F = s.c1_filters, K = s.c1_kernel, P = s.c1_pad;
        const float* W = p.weights.data();
        for (int b = 0; b < B; ++b) {
            for (int f = 0; f < F; ++f) {
                for (int t = 0; t < L; ++t) {
                    float acc = 0.0f;
                    for (int ic = 0; ic < IC; ++ic) {
                        for (int kk = 0; kk < K; ++kk) {
                            const int src = t + kk - P;
                            if (src < 0 || src >= L) continue;
                            acc += in[(b * IC + ic) * L + src] * W[(f * IC + ic) * K + kk];
                        }
                    }
                    out[(b * F + f) * L + t] = acc;
                }
            }
        }
        return true;
    }

    // ---- Conv2D (cross-correlation, pad=1, k=3, stride=1) → flat [B, F*H*W] ----
    if (name == "Conv2D") {
        const int B = s.c2_batch, IC = s.c2_in_c, H = s.c2_h, Wd = s.c2_w;
        const int F = s.c2_filters, K = s.c2_kernel, P = s.c2_pad;
        const float* W = p.weights.data();
        for (int b = 0; b < B; ++b) {
            for (int f = 0; f < F; ++f) {
                for (int oh = 0; oh < H; ++oh) {
                    for (int ow = 0; ow < Wd; ++ow) {
                        float acc = 0.0f;
                        for (int ic = 0; ic < IC; ++ic) {
                            for (int ky = 0; ky < K; ++ky) {
                                const int sy = oh + ky - P;
                                if (sy < 0 || sy >= H) continue;
                                for (int kx = 0; kx < K; ++kx) {
                                    const int sx = ow + kx - P;
                                    if (sx < 0 || sx >= Wd) continue;
                                    const float iv = in[((b * IC + ic) * H + sy) * Wd + sx];
                                    const float wv = W[((f * IC + ic) * K + ky) * K + kx];
                                    acc += iv * wv;
                                }
                            }
                        }
                        out[((b * F + f) * H + oh) * Wd + ow] = acc;
                    }
                }
            }
        }
        return true;
    }

    // ---- DepthwiseConv (per-channel k=3 fill 0.02, pad=1, stride=1) ----
    if (name == "DepthwiseConv") {
        const int B = s.sp_batch, C = s.sp_c, H = s.sp_h, Wd = s.sp_w;
        const int K = s.sp_kernel, P = s.sp_pad;
        const float wv = 0.02f;
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                for (int oh = 0; oh < H; ++oh) {
                    for (int ow = 0; ow < Wd; ++ow) {
                        float acc = 0.0f;
                        for (int ky = 0; ky < K; ++ky) {
                            const int sy = oh + ky - P;
                            if (sy < 0 || sy >= H) continue;
                            for (int kx = 0; kx < K; ++kx) {
                                const int sx = ow + kx - P;
                                if (sx < 0 || sx >= Wd) continue;
                                acc += in[((b * C + c) * H + sy) * Wd + sx] * wv;
                            }
                        }
                        out[((b * C + c) * H + oh) * Wd + ow] = acc;
                    }
                }
            }
        }
        return true;
    }

    // ---- AvgPool / MaxPool (ks=2, stride=2, pad=0) ----
    if (name == "AvgPool" || name == "MaxPool") {
        const bool is_avg = (name == "AvgPool");
        const int B = s.sp_batch, C = s.sp_c, H = s.sp_h, Wd = s.sp_w;
        const int KS = s.sp_pool_ks, ST = s.sp_pool_stride;
        const int OH = pool_out(H, KS, ST), OW = pool_out(Wd, KS, ST);
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        float acc = is_avg ? 0.0f : -INFINITY;
                        for (int ky = 0; ky < KS; ++ky) {
                            for (int kx = 0; kx < KS; ++kx) {
                                const int sy = oh * ST + ky;
                                const int sx = ow * ST + kx;
                                const float v = in[((b * C + c) * H + sy) * Wd + sx];
                                if (is_avg) {
                                    acc += v;
                                } else if (v > acc) {
                                    acc = v;
                                }
                            }
                        }
                        if (is_avg) acc /= float(KS * KS);
                        out[((b * C + c) * OH + oh) * OW + ow] = acc;
                    }
                }
            }
        }
        return true;
    }

    // ---- Elementwise activations & gates on [B, D] ----
    if (name == "ReLU") {
        for (size_t i = 0; i < in_n; ++i) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
        return true;
    }
    if (name == "GELU") {
        const float inv_sqrt2 = 0.7071067811865476f;
        for (size_t i = 0; i < in_n; ++i) {
            out[i] = 0.5f * in[i] * (1.0f + std::erf(in[i] * inv_sqrt2));
        }
        return true;
    }
    if (name == "Sigmoid") {
        for (size_t i = 0; i < in_n; ++i) out[i] = 1.0f / (1.0f + std::exp(-in[i]));
        return true;
    }
    if (name == "Add") {  // residual add of const 0.01 (intel add_residual)
        for (size_t i = 0; i < in_n; ++i) out[i] = in[i] + 0.01f;
        return true;
    }
    if (name == "Multiply") {  // gate by const 0.5 (intel mul_gate)
        for (size_t i = 0; i < in_n; ++i) out[i] = in[i] * 0.5f;
        return true;
    }

    // ---- Softmax over the feature axis (per row of dim D) ----
    if (name == "Softmax") {
        const int B = s.dense_batch, D = s.dim;
        for (int b = 0; b < B; ++b) {
            const float* row = in + b * D;
            float* orow = out + b * D;
            float mx = row[0];
            for (int j = 1; j < D; ++j) mx = row[j] > mx ? row[j] : mx;
            float sum = 0.0f;
            for (int j = 0; j < D; ++j) {
                orow[j] = std::exp(row[j] - mx);
                sum += orow[j];
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int j = 0; j < D; ++j) orow[j] *= inv;
        }
        return true;
    }

    // ---- LayerNorm (MVN normalize_variance, eps inside sqrt; gamma=1,beta=0) ----
    if (name == "LayerNorm") {
        const int B = s.dense_batch, D = s.dim;
        const float eps = 1e-5f;
        for (int b = 0; b < B; ++b) {
            const float* row = in + b * D;
            float* orow = out + b * D;
            float mean = 0.0f;
            for (int j = 0; j < D; ++j) mean += row[j];
            mean /= float(D);
            float var = 0.0f;
            for (int j = 0; j < D; ++j) {
                const float d = row[j] - mean;
                var += d * d;
            }
            var /= float(D);
            const float inv = 1.0f / std::sqrt(var + eps);
            for (int j = 0; j < D; ++j) orow[j] = (row[j] - mean) * inv;
        }
        return true;
    }

    // ---- RMSNorm (x / sqrt(mean(x^2)+eps) * scale=1) ----
    if (name == "RMSNorm") {
        const int B = s.dense_batch, D = s.dim;
        const float eps = 1e-5f;
        for (int b = 0; b < B; ++b) {
            const float* row = in + b * D;
            float* orow = out + b * D;
            float ms = 0.0f;
            for (int j = 0; j < D; ++j) ms += row[j] * row[j];
            ms /= float(D);
            const float inv = 1.0f / std::sqrt(ms + eps);
            for (int j = 0; j < D; ++j) orow[j] = row[j] * inv;
        }
        return true;
    }

    return fail("unsupported layer");
}

}  // namespace loom_apple
