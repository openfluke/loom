#include "mps_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace loom_apple {

// Held as a C++ struct with strong ObjC members (compiled under -fobjc-arc).
struct MpsLayer {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    MPSGraph* graph = nil;
    MPSGraphTensor* input = nil;
    MPSGraphTensor* output = nil;
    NSArray<NSNumber*>* in_shape = nil;
    size_t in_elems = 0;
    size_t out_elems = 0;
};

namespace {

NSArray<NSNumber*>* shape2d(int a, int b) {
    return @[ @(a), @(b) ];
}

// Weight constant [D,D] (row-major) for the matmul family.
MPSGraphTensor* weight_const(MPSGraph* g, const std::vector<float>& w, int rows, int cols) {
    NSData* data = [NSData dataWithBytes:w.data() length:w.size() * sizeof(float)];
    return [g constantWithData:data shape:shape2d(rows, cols) dataType:MPSDataTypeFloat32];
}

}  // namespace

bool mps_device_available() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        return dev != nil;
    }
}

MpsLayer* mps_build(const Prepared& p, std::string* err) {
    const std::string& name = p.name;

    // Curated set of ops accelerated on the GPU today. Anything else → CPU fallback.
    const bool supported =
        name == "MatMul" || name == "MHA-MatMul" || name == "ReLU" ||
        name == "Sigmoid" || name == "Softmax" || name == "Add" || name == "Multiply";
    if (!supported) {
        return nullptr;  // no err → caller uses CPU reference
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            if (err) *err = "no Metal device";
            return nullptr;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            if (err) *err = "failed to create Metal command queue";
            return nullptr;
        }

        MPSGraph* graph = [[MPSGraph alloc] init];
        const int B = p.spec.dense_batch;
        const int D = p.spec.dim;

        MPSGraphTensor* ph = [graph placeholderWithShape:shape2d(B, D)
                                                dataType:MPSDataTypeFloat32
                                                    name:@"input"];
        MPSGraphTensor* out = nil;

        if (name == "MatMul" || name == "MHA-MatMul") {
            MPSGraphTensor* w = weight_const(graph, p.weights, D, D);
            out = [graph matrixMultiplicationWithPrimaryTensor:ph secondaryTensor:w name:@"matmul"];
        } else if (name == "ReLU") {
            out = [graph reLUWithTensor:ph name:@"relu"];
        } else if (name == "Sigmoid") {
            out = [graph sigmoidWithTensor:ph name:@"sigmoid"];
        } else if (name == "Softmax") {
            out = [graph softMaxWithTensor:ph axis:1 name:@"softmax"];
        } else if (name == "Add") {
            MPSGraphTensor* c = [graph constantWithScalar:0.01 dataType:MPSDataTypeFloat32];
            out = [graph additionWithPrimaryTensor:ph secondaryTensor:c name:@"add"];
        } else if (name == "Multiply") {
            MPSGraphTensor* c = [graph constantWithScalar:0.5 dataType:MPSDataTypeFloat32];
            out = [graph multiplicationWithPrimaryTensor:ph secondaryTensor:c name:@"mul"];
        }

        if (out == nil) {
            if (err) *err = "failed to build MPSGraph op";
            return nullptr;
        }

        auto* layer = new MpsLayer();
        layer->device = device;
        layer->queue = queue;
        layer->graph = graph;
        layer->input = ph;
        layer->output = out;
        layer->in_shape = shape2d(B, D);
        layer->in_elems = p.in_elems;
        layer->out_elems = p.out_elems;
        return layer;
    }
}

void mps_release(MpsLayer* layer) {
    delete layer;  // ARC releases the strong ObjC members
}

bool mps_run(MpsLayer* layer, const float* in, size_t in_n, float* out, size_t out_n, std::string* err) {
    if (layer == nullptr) {
        if (err) *err = "null mps layer";
        return false;
    }
    if (in_n != layer->in_elems || out_n != layer->out_elems) {
        if (err) *err = "mps element count mismatch";
        return false;
    }

    @autoreleasepool {
        id<MTLBuffer> inBuf = [layer->device newBufferWithBytes:in
                                                         length:in_n * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        if (inBuf == nil) {
            if (err) *err = "failed to allocate Metal input buffer";
            return false;
        }

        MPSGraphTensorData* inData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf
                                                    shape:layer->in_shape
                                                 dataType:MPSDataTypeFloat32];

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
            [layer->graph runWithMTLCommandQueue:layer->queue
                                           feeds:@{layer->input : inData}
                                   targetTensors:@[ layer->output ]
                                targetOperations:nil];

        MPSGraphTensorData* outData = results[layer->output];
        if (outData == nil) {
            if (err) *err = "MPSGraph produced no output";
            return false;
        }

        [[outData mpsndarray] readBytes:out strideBytes:nil];
        return true;
    }
}

}  // namespace loom_apple
