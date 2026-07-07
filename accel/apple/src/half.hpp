#pragma once
//
// half.hpp — IEEE-754 half <-> float conversion (matches poly/accel_intel.go's
// float32ToFloat16Bits / float16BitsToFloat32 so byte layouts agree across the C ABI).
//
#include <cstdint>
#include <cstring>

namespace loom_apple {

inline uint16_t float_to_half(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint16_t sign = uint16_t((bits >> 16) & 0x8000);
    const int exp = int((bits >> 23) & 0xff);
    const uint32_t frac = bits & 0x7fffff;
    if (exp == 0) {
        return sign;
    }
    if (exp == 0xff) {
        return uint16_t(sign | 0x7c00 | uint16_t(frac >> 13));
    }
    int new_exp = exp - 127 + 15;
    if (new_exp >= 0x1f) {
        return uint16_t(sign | 0x7c00);
    }
    if (new_exp <= 0) {
        return sign;
    }
    return uint16_t(sign | uint16_t(new_exp << 10) | uint16_t(frac >> 13));
}

// bfloat16 = top 16 bits of fp32 (keeps fp32 exponent range). Native on Apple
// silicon / AMX. Round-to-nearest-even, matching poly/accel_intel.go.
inline uint16_t float_to_bfloat16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    if ((bits & 0x7fffffff) > 0x7f800000) {  // NaN → quiet NaN
        return uint16_t((bits >> 16) | 0x0040);
    }
    const uint32_t lsb = (bits >> 16) & 1;
    bits += 0x7fff + lsb;
    return uint16_t(bits >> 16);
}

inline float bfloat16_to_float(uint16_t b) {
    const uint32_t bits = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline float half_to_float(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    const int exp = int((h >> 10) & 0x1f);
    const uint32_t frac = uint32_t(h & 0x3ff);
    uint32_t out;
    if (exp == 0) {
        out = sign;  // zero / subnormal (good enough for parity tests)
    } else if (exp == 0x1f) {
        out = sign | 0x7f800000 | (frac << 13);
    } else {
        out = sign | uint32_t((exp - 15 + 127) << 23) | (frac << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

}  // namespace loom_apple
