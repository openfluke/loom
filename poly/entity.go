package poly

// ENTITY — Every Numerical Type In Native TopologY
//
// Native Loom checkpoint format (.entity). Spec: docs/entity.md
//
// Why this exists:
//
// HuggingFace .safetensors is the right *import* lane for PyTorch/HF checkpoints, but
// it cannot express what Loom actually trains and saves:
//
//   - All 21 DTypes with native bit-packing (Int4 nibbles, Binary 8:1, Ternary, …)
//   - Per-layer WeightStore.Scale (quant mapping the checkpoint was trained with)
//   - Volumetric placement (Z,Y,X,L) — not just flat tensor names
//   - Layer topology: parallel branches, sequential stacks, remote links
//   - One bit-perfect reload of a full VolumetricNetwork (see persistence.go idempotency)
//
// Today we have:
//
//   - safetensors.go  → read HF weights (decode mostly to FP32); SaveSafetensors is F32-only
//   - persistence.go  → full fidelity via JSON + Base64 (works, but huge and slow)
//
// ENTITY is the binary middle path: safe length-prefixed header + contiguous blobs, same
// native packing rules as persistence.go (encodeNativeWeights / decodeNativeWeights), without
// pretending to be .safetensors.
//
// Import:  model.safetensors  (HF)
// Native:  fluffy.entity       (ENTITY)
//
// No implementation here yet — this file is the placeholder until we write the spec.
