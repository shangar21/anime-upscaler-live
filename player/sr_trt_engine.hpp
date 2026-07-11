#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>

// These are populated at load() time by querying the engine's own binding
// shapes, so the plugin automatically adapts to 2x, 4x, or any other scale
// without recompiling -- just swap the engine file and rebuild the TRT engine.
//
// Confirmed engine contract (apisr_fp16.trt, 4x):
//   input  "input"  fp32 NCHW (batch, 3, 64, 64)
//   output "output" fp32 NCHW (batch, 3, 256, 256)
//   profile: batch [1..64], spatial [32..128]
//
// For a 2x model trained with --sr_rate 2, the contract would be:
//   input  "input"  fp32 NCHW (batch, 3, 64, 64)
//   output "output" fp32 NCHW (batch, 3, 128, 128)
// and kScale would auto-derive as 2, kTileHR as 128.

class SrTrtEngine {
public:
    // Loads the engine and populates kTileLR, kTileHR, kScale, kMaxBatch.
    // Returns false (and logs) if the engine can't be loaded or the binding
    // contract doesn't look like an SR model (3-channel in, 3-channel out,
    // spatial ratio is a whole number >= 1).
    bool load(const std::string &enginePath);

    bool isLoaded() const { return impl_ != nullptr && d_input_ != nullptr; }

    // Runs inference for `batchSize` tiles (<= kMaxBatch) already packed
    // into the persistent input buffer, writing into the persistent output
    // buffer. Caller is responsible for filling/reading those.
    void infer(int batchSize, cudaStream_t stream);

    float *inputDevicePtr()  const { return d_input_; }
    float *outputDevicePtr() const { return d_output_; }

    // Runtime-derived geometry (valid after a successful load()).
    int tileLR()   const { return kTileLR_; }
    int tileHR()   const { return kTileHR_; }
    int scale()    const { return kScale_; }
    int maxBatch() const { return kMaxBatch_; }

    ~SrTrtEngine();

private:
    struct Impl;
    Impl *impl_ = nullptr;
    float *d_input_  = nullptr;
    float *d_output_ = nullptr;

    int kTileLR_   = 64;
    int kTileHR_   = 256;
    int kScale_    = 4;
    int kMaxBatch_ = 32;
};
