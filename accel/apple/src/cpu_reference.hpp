#pragma once
//
// cpu_reference.hpp — portable CPU forward for every bench layer. This is the
// parity anchor (like Qualcomm's QnnCpu backend): deterministic, always available,
// and the fallback for any op the Metal/MPSGraph path does not yet accelerate.
//
#include "shapes.hpp"

#include <string>
#include <vector>

namespace loom_apple {

// A compiled layer with weights already resolved (uploaded bytes decoded, or the
// fixed constants that accel/intel's layer_models.cpp bakes when Loom uploads none).
struct Prepared {
    std::string name;
    std::string dtype_label;
    ShapeSpec spec;
    std::vector<float> weights;  // empty for ops that use inline constants
    size_t in_elems = 0;
    size_t out_elems = 0;
};

// Decode Loom-uploaded weight bytes to float (FP32 4-byte, FP16 2-byte half), or
// synthesize the constant fill for layers Loom does not upload weights for.
std::vector<float> resolve_weights(
    const std::string& name,
    const ShapeSpec& s,
    const void* weight_bytes,
    size_t weight_byte_len,
    const std::string& dtype_label);

// Deterministic CPU forward. Returns true on success, false + msg on error.
bool cpu_forward(const Prepared& p, const float* in, size_t in_n, float* out, size_t out_n, std::string* err);

}  // namespace loom_apple
