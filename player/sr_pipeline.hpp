#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <npp.h>
#include "sr_trt_engine.hpp"

class SrPipeline {
public:
    bool init(int gpuId, cudaStream_t stream);

    void processFrame(
        const uint8_t *inRgba, int inPitch, int width, int height,
        uint8_t *outRgba, int outPitch,
        int overlapHr, SrTrtEngine &engine, cudaStream_t stream);

    ~SrPipeline();

private:
    void ensureFrameBuffers(int width, int height, int scale);
    void ensureBlendMask(int tileHR, int overlapHr);
    void ensureTileOriginBuffers(int maxTiles);

    NppStreamContext nppCtx_{};
    int curW_ = 0, curH_ = 0, curScale_ = 0;
    int curMaxTiles_ = 0;

    // Input frame buffers
    Npp8u  *inPlaneR_ = nullptr, *inPlaneG_ = nullptr,
           *inPlaneB_ = nullptr, *inPlaneA_ = nullptr;
    Npp32f *scratchF32Plane_ = nullptr;
    Npp32f *frame_planar_    = nullptr;   // 3 x W x H planar fp32 [0,1]

    // Output accumulators
    Npp32f *accum_  = nullptr;   // 3 x outW x outH
    Npp32f *weight_ = nullptr;   // outW x outH

    // Output planes for the pack kernel
    Npp8u *outPlaneR_ = nullptr, *outPlaneG_ = nullptr, *outPlaneB_ = nullptr;

    // Blend mask and scratch
    Npp32f *blendMask_ = nullptr;
    int builtMaskTileHR_  = -1;
    int builtMaskOverlap_ = -1;

    // Device-side tile origin arrays (LR and HR space)
    // Uploaded once per frame when the tile grid is built.
    int *d_tileOrigX_lr_ = nullptr;  // LR tile origins X
    int *d_tileOrigY_lr_ = nullptr;  // LR tile origins Y
    int *d_tileOrigX_hr_ = nullptr;  // HR tile origins X
    int *d_tileOrigY_hr_ = nullptr;  // HR tile origins Y
};
