#ifndef CABI_TEST_H
#define CABI_TEST_H

#include <stdint.h>
#include <stdlib.h>

#ifdef _WIN32
#define CABI_API __declspec(dllimport)
#else
#define CABI_API
#endif

// --- DTypes ---
typedef enum {
    DTypeFloat64  = 0,
    DTypeFloat32  = 1,
    DTypeFloat16  = 2,
    DTypeBFloat16 = 3,
    DTypeFP8E4M3  = 4,
    DTypeFP8E5M2  = 5,
    DTypeInt64    = 6,
    DTypeInt32    = 7,
    DTypeInt16    = 8,
    DTypeInt8     = 9,
    DTypeUint64   = 10,
    DTypeUint32   = 11,
    DTypeUint16   = 12,
    DTypeUint8    = 13,
    DTypeInt4     = 14,
    DTypeUint4    = 15,
    DTypeFP4      = 16,
    DTypeInt2     = 17,
    DTypeUint2    = 18,
    DTypeTernary  = 19,
    DTypeBinary   = 20
} LoomDType;

// --- Structs from structs_ext.go ---

typedef struct {
    int Z, Y, X, L;
    int Type;
    int Activation;
    int DType;
    int InputHeight, InputWidth, InputDepth;
    int OutputHeight, OutputWidth, OutputDepth;
    int InputChannels, Filters, KernelSize, Stride, Padding;
    int NumHeads, NumKVHeads, DModel, SeqLength;
    int VocabSize, EmbeddingDim;
    int NumClusters;
    int UseTiling, TileSize;
} LoomLayerSpec;

typedef struct {
    float Avg;
    float Max;
    float Min;
    int Active;
    int Total;
} LoomLayerStats;

typedef struct {
    int DType;
    int Rank;
    int64_t Shape[8];
} LoomTensorMeta;

typedef struct {
    const char* Name;
    int DType;
    int Rank;
    int64_t Shape[8];
    uint64_t DataOffset;
    uint64_t DataLength;
} LoomTensorInfo;

typedef struct {
    int NumTensors;
    LoomTensorInfo* Tensors;
} LoomSafetensorsHeader;

typedef struct {
    const char* ID;
    const char* Name;
    int GridDepth;
    int GridRows;
    int GridCols;
} LoomNetworkBlueprint;

// --- Structs from acceleration_ext.go ---

typedef struct {
    uint32_t BatchSize;
    uint32_t InputSize;
    uint32_t OutputSize;
    uint32_t TileSize;
} WGPUDenseParams;

typedef struct {
    uint32_t NumHeads;
    uint32_t NumKVHeads;
    uint32_t HeadDim;
    uint32_t SeqLen;
    uint32_t KVOffset;
    uint32_t MaxSeqLen;
    uint32_t TileSize;
    uint32_t Padding;
} WGPUMHAParams;

typedef struct {
    uint32_t Size;
    float Epsilon;
} WGPURMSNormParams;

typedef struct {
    uint32_t Offset;
    uint32_t HeadDim;
    uint32_t MaxSeqLen;
    uint32_t NumKVHeads;
    uint32_t NumTokens;
} WGPUKVParams;

typedef struct {
    uint32_t SeqLen;
    uint32_t HeadDim;
    uint32_t NumHeads;
    uint32_t Offset;
    float Theta;
} WGPURoPEParams;

typedef struct {
    uint32_t VocabSize;
    uint32_t HiddenSize;
    uint32_t NumTokens;
    uint32_t Padding;
} WGPUEmbeddingParams;

// --- Function Prototypes ---
// (We only include a few core ones for functional verification, 
// the rest will be dynamically loaded/verified in cabi_verify.c)

CABI_API void FreeLoomString(char* ptr);
CABI_API long long LoomBuildNetworkFromJSON(const char* jsonConfig);
CABI_API void LoomFreeNetwork(long long handle);
CABI_API char* LoomGetNetworkInfo(long long handle);
CABI_API LoomLayerSpec LoomGetLayerSpec(long long networkHandle, int layerIdx);

#endif // CABI_TEST_H
