#pragma once
//
// mps_backend.hpp — Metal / MPSGraph GPU forward for the subset of layers that map
// cleanly to MPSGraph today (MatMul, MHA-MatMul, ReLU, Sigmoid, Softmax, Add, Multiply).
// Everything else returns a null build handle so the C ABI falls back to the CPU
// reference — the same "accelerate a subset, stay honest about the rest" posture as
// the Intel (OpenVINO) and Qualcomm (QNN) plugins.
//
#include "cpu_reference.hpp"

#include <string>

namespace loom_apple {

// True if a Metal system default device exists.
bool mps_device_available();

// Opaque compiled MPSGraph layer.
struct MpsLayer;

// Build a GPU graph for supported ops; nullptr (no err set) means "unsupported —
// use CPU reference". nullptr WITH err set means a real Metal error.
MpsLayer* mps_build(const Prepared& p, std::string* err);

void mps_release(MpsLayer* layer);

// Run one forward on the GPU. Input/output are packed fp32 host buffers.
bool mps_run(MpsLayer* layer, const float* in, size_t in_n, float* out, size_t out_n, std::string* err);

}  // namespace loom_apple
