// SrPipeline — rewritten to use batch tile kernels instead of per-tile NPP
// dispatch. The old implementation issued ~700 individual NPP kernel launches
// per frame (7 per tile × 96 tiles), each tiny enough that launch overhead
// dominated. This version does:
//   - 1 NPP call to split RGBA into planes
//   - 3 NPP calls to convert+normalize per channel  (3 total)
//   - 1 launchExtractTiles  (all tiles in parallel, one kernel)
//   - 1 engine.infer        (TRT, batched)
//   - 1 launchStitchTiles   (all tiles blended in parallel, one kernel)
//   - 1 NPP threshold + 3 NPP divide  (normalize accum)
//   - 3 NPP mulC + convert  (fp32->uint8 per channel)
//   - 1 launchPackRgbFromPlanes
// Total: ~12 kernel launches per frame regardless of tile count.
#include "sr_pipeline.hpp"
#include "tile_kernels.cuh"
#include "pack_rgba_kernel.cuh"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace {
constexpr float kWeightEpsilon = 1e-6f;

#define NPPCHECK(expr) do { \
    NppStatus _st = (expr); \
    if (_st != NPP_SUCCESS) \
        std::cerr << "NPP error " << _st << " at " << __FILE__ << ":" \
                  << __LINE__ << "\n"; \
} while (0)
}

bool SrPipeline::init(int gpuId, cudaStream_t stream) {
    cudaSetDevice(gpuId);
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, device);
    int ccMajor = 0, ccMinor = 0;
    cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, device);
    unsigned int streamFlags = 0;
    cudaStreamGetFlags(stream, &streamFlags);

    nppCtx_.hStream                            = stream;
    nppCtx_.nCudaDeviceId                      = device;
    nppCtx_.nMultiProcessorCount               = props.multiProcessorCount;
    nppCtx_.nMaxThreadsPerMultiProcessor        = props.maxThreadsPerMultiProcessor;
    nppCtx_.nMaxThreadsPerBlock                 = props.maxThreadsPerBlock;
    nppCtx_.nSharedMemPerBlock                  = props.sharedMemPerBlock;
    nppCtx_.nCudaDevAttrComputeCapabilityMajor  = ccMajor;
    nppCtx_.nCudaDevAttrComputeCapabilityMinor  = ccMinor;
    nppCtx_.nStreamFlags                       = streamFlags;
    return true;
}

void SrPipeline::ensureBlendMask(int tileHR, int overlapHr) {
    if (tileHR == builtMaskTileHR_ && overlapHr == builtMaskOverlap_) return;

    if (blendMask_ && tileHR != builtMaskTileHR_) {
        cudaFree(blendMask_);
        blendMask_ = nullptr;
    }
    if (!blendMask_)
        cudaMalloc(&blendMask_, size_t(tileHR) * tileHR * sizeof(float));

    std::vector<float> mask(size_t(tileHR) * tileHR, 1.0f);
    for (int i = 0; i < overlapHr; ++i) {
        float w = 0.5f * (1.0f - std::cos(float(M_PI) * i / overlapHr));
        for (int x = 0; x < tileHR; ++x) {
            mask[i * tileHR + x]                  = std::min(mask[i * tileHR + x], w);
            mask[(tileHR-1-i) * tileHR + x]       = std::min(mask[(tileHR-1-i) * tileHR + x], w);
        }
        for (int y = 0; y < tileHR; ++y) {
            mask[y * tileHR + i]                  = std::min(mask[y * tileHR + i], w);
            mask[y * tileHR + (tileHR-1-i)]       = std::min(mask[y * tileHR + (tileHR-1-i)], w);
        }
    }
    cudaMemcpy(blendMask_, mask.data(), mask.size() * sizeof(float), cudaMemcpyHostToDevice);
    builtMaskTileHR_  = tileHR;
    builtMaskOverlap_ = overlapHr;
}

void SrPipeline::ensureTileOriginBuffers(int maxTiles) {
    if (maxTiles <= curMaxTiles_) return;
    auto freeIf = [](void *p) { if (p) cudaFree(p); };
    freeIf(d_tileOrigX_lr_); freeIf(d_tileOrigY_lr_);
    freeIf(d_tileOrigX_hr_); freeIf(d_tileOrigY_hr_);
    cudaMalloc(&d_tileOrigX_lr_, maxTiles * sizeof(int));
    cudaMalloc(&d_tileOrigY_lr_, maxTiles * sizeof(int));
    cudaMalloc(&d_tileOrigX_hr_, maxTiles * sizeof(int));
    cudaMalloc(&d_tileOrigY_hr_, maxTiles * sizeof(int));
    curMaxTiles_ = maxTiles;
}

void SrPipeline::ensureFrameBuffers(int width, int height, int scale) {
    if (width == curW_ && height == curH_ && scale == curScale_) return;
    auto freeIf = [](void *p) { if (p) cudaFree(p); };
    freeIf(inPlaneR_); freeIf(inPlaneG_); freeIf(inPlaneB_); freeIf(inPlaneA_);
    freeIf(scratchF32Plane_); freeIf(frame_planar_);
    freeIf(accum_); freeIf(weight_);
    freeIf(outPlaneR_); freeIf(outPlaneG_); freeIf(outPlaneB_);

    int outW = width * scale, outH = height * scale;
    size_t inPx = size_t(width) * height, outPx = size_t(outW) * outH;
    cudaMalloc(&inPlaneR_,       inPx  * sizeof(Npp8u));
    cudaMalloc(&inPlaneG_,       inPx  * sizeof(Npp8u));
    cudaMalloc(&inPlaneB_,       inPx  * sizeof(Npp8u));
    cudaMalloc(&inPlaneA_,       inPx  * sizeof(Npp8u));
    cudaMalloc(&scratchF32Plane_,inPx  * sizeof(Npp32f));
    cudaMalloc(&frame_planar_,   inPx  * 3 * sizeof(Npp32f));
    cudaMalloc(&accum_,          outPx * 3 * sizeof(Npp32f));
    cudaMalloc(&weight_,         outPx * sizeof(Npp32f));
    cudaMalloc(&outPlaneR_,      outPx * sizeof(Npp8u));
    cudaMalloc(&outPlaneG_,      outPx * sizeof(Npp8u));
    cudaMalloc(&outPlaneB_,      outPx * sizeof(Npp8u));
    curW_ = width; curH_ = height; curScale_ = scale;
    std::cerr << "SrPipeline: buffers for " << width << "x" << height
              << " -> " << outW << "x" << outH << " (scale " << scale << "x)\n";
}

void SrPipeline::processFrame(
    const uint8_t *inRgba, int inPitch, int width, int height,
    uint8_t *outRgba, int outPitch,
    int overlapHr, SrTrtEngine &engine, cudaStream_t stream)
{
    const int kScale    = engine.scale();
    const int kTileLR   = engine.tileLR();
    const int kTileHR   = engine.tileHR();
    const int kMaxBatch = engine.maxBatch();

    ensureFrameBuffers(width, height, kScale);
    ensureBlendMask(kTileHR, overlapHr);
    nppCtx_.hStream = stream;

    const int outW = width * kScale, outH = height * kScale;
    NppiSize roiFull{ width, height };
    NppiSize roiOut { outW, outH };

    // --- 1. RGBA -> 4 planes -> normalized planar fp32 ---
    {
        Npp8u *planes4[4] = { inPlaneR_, inPlaneG_, inPlaneB_, inPlaneA_ };
        NPPCHECK(nppiCopy_8u_C4P4R_Ctx(
            (const Npp8u *)inRgba, inPitch, planes4, width, roiFull, nppCtx_));
    }
    const Npp8u *srcPlanes[3] = { inPlaneR_, inPlaneG_, inPlaneB_ };
		//const Npp8u *srcPlanes[3] = { inPlaneB_, inPlaneG_, inPlaneR_ };
    for (int c = 0; c < 3; ++c) {
        Npp32f *dst = frame_planar_ + size_t(c) * width * height;
        NPPCHECK(nppiConvert_8u32f_C1R_Ctx(
            srcPlanes[c], width, scratchF32Plane_,
            width * sizeof(Npp32f), roiFull, nppCtx_));
        NPPCHECK(nppiMulC_32f_C1R_Ctx(
            scratchF32Plane_, width * sizeof(Npp32f), 1.0f / 255.0f,
            dst, width * sizeof(Npp32f), roiFull, nppCtx_));
    }

    // --- 2. Build tile grid ---
    const int lrOverlap = std::max(1, overlapHr / kScale);
    const int lrStep    = std::max(1, kTileLR - 2 * lrOverlap);
    const int nw = (width  > kTileLR) ? (width  - kTileLR + lrStep - 1) / lrStep + 1 : 1;
    const int nh = (height > kTileLR) ? (height - kTileLR + lrStep - 1) / lrStep + 1 : 1;

    std::vector<int> hx_lr, hy_lr, hx_hr, hy_hr;
    hx_lr.reserve(nw * nh); hy_lr.reserve(nw * nh);
    hx_hr.reserve(nw * nh); hy_hr.reserve(nw * nh);
    for (int row = 0; row < nh; ++row) {
        int ty = std::min(row * lrStep, std::max(0, height - kTileLR));
        for (int col = 0; col < nw; ++col) {
            int tx = std::min(col * lrStep, std::max(0, width - kTileLR));
            hx_lr.push_back(tx);
            hy_lr.push_back(ty);
            hx_hr.push_back(tx * kScale);
            hy_hr.push_back(ty * kScale);
        }
    }
    int numTiles = int(hx_lr.size());

    static bool loggedTiles = false;
    if (!loggedTiles) {
        std::cerr << "SrPipeline: " << numTiles << " tiles/frame "
                  << "(" << nw << "x" << nh << " grid, lrStep=" << lrStep
                  << ", lrOverlap=" << lrOverlap << ", scale=" << kScale << "x)\n";
        for (int s = 0; s < numTiles; s += kMaxBatch) {
            int n = std::min(numTiles - s, kMaxBatch);
            std::cerr << "  batch " << (s/kMaxBatch + 1) << ": " << n << " tiles\n";
        }
        loggedTiles = true;
    }

    // Upload tile origins to device once per frame
    ensureTileOriginBuffers(numTiles);
    cudaMemcpyAsync(d_tileOrigX_lr_, hx_lr.data(), numTiles * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_tileOrigY_lr_, hy_lr.data(), numTiles * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_tileOrigX_hr_, hx_hr.data(), numTiles * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_tileOrigY_hr_, hy_hr.data(), numTiles * sizeof(int), cudaMemcpyHostToDevice, stream);

    // --- 3. Zero accumulators ---
    cudaMemsetAsync(accum_,  0, size_t(outW) * outH * 3 * sizeof(float), stream);
    cudaMemsetAsync(weight_, 0, size_t(outW) * outH * sizeof(float), stream);

    // --- 4. Process in batches ---
    for (int start = 0; start < numTiles; start += kMaxBatch) {
        int batchN = std::min(numTiles - start, kMaxBatch);

        // Extract all tiles in this batch in one parallel kernel
        launchExtractTiles(
            frame_planar_, width, height,
            engine.inputDevicePtr(),
            d_tileOrigX_lr_ + start, d_tileOrigY_lr_ + start,
            batchN, kTileLR, kTileLR, stream);

        // TRT inference
        engine.infer(batchN, stream);

        // Stitch all output tiles back in one parallel kernel
        launchStitchTiles(
            engine.outputDevicePtr(),
            blendMask_,
            accum_, weight_,
            d_tileOrigX_hr_ + start, d_tileOrigY_hr_ + start,
            batchN, kTileHR, kTileHR,
            outW, outH, stream);
    }

    // --- 5. Normalize ---
    NPPCHECK(nppiThreshold_LTVal_32f_C1IR_Ctx(
        weight_, outW * sizeof(Npp32f), roiOut,
        kWeightEpsilon, kWeightEpsilon, nppCtx_));
    Npp32f *accumPlanes[3] = {
        accum_, accum_ + size_t(outW)*outH, accum_ + 2*size_t(outW)*outH
    };
    for (int c = 0; c < 3; ++c) {
        NPPCHECK(nppiDiv_32f_C1IR_Ctx(
            weight_, outW * sizeof(Npp32f),
            accumPlanes[c], outW * sizeof(Npp32f), roiOut, nppCtx_));
    }

    // --- 6. fp32 -> uint8 planes -> RGBA ---
    Npp8u *dstPlanes[3] = { outPlaneR_, outPlaneG_, outPlaneB_ };
    for (int c = 0; c < 3; ++c) {
        NPPCHECK(nppiMulC_32f_C1R_Ctx(
            accumPlanes[c], outW * sizeof(Npp32f), 255.0f,
            accumPlanes[c], outW * sizeof(Npp32f), roiOut, nppCtx_));
        NPPCHECK(nppiConvert_32f8u_C1R_Ctx(
            accumPlanes[c], outW * sizeof(Npp32f),
            dstPlanes[c], outW, roiOut, NPP_RND_NEAR, nppCtx_));
    }
		// --- DEBUG: Draw a solid Red stripe across the top 100 rows ---
    cudaMemsetAsync(outPlaneR_, 255, 100 * outW * sizeof(Npp8u), stream);
    cudaMemsetAsync(outPlaneG_, 0,   100 * outW * sizeof(Npp8u), stream);
    cudaMemsetAsync(outPlaneB_, 0,   100 * outW * sizeof(Npp8u), stream);

    launchPackRgbFromPlanes(outPlaneR_, outPlaneG_, outPlaneB_,
                             outRgba, outPitch, outW, outH, stream);
}

SrPipeline::~SrPipeline() {
    auto freeIf = [](void *p) { if (p) cudaFree(p); };
    freeIf(inPlaneR_); freeIf(inPlaneG_); freeIf(inPlaneB_); freeIf(inPlaneA_);
    freeIf(scratchF32Plane_); freeIf(frame_planar_);
    freeIf(accum_); freeIf(weight_);
    freeIf(outPlaneR_); freeIf(outPlaneG_); freeIf(outPlaneB_);
    freeIf(blendMask_);
    freeIf(d_tileOrigX_lr_); freeIf(d_tileOrigY_lr_);
    freeIf(d_tileOrigX_hr_); freeIf(d_tileOrigY_hr_);
}
