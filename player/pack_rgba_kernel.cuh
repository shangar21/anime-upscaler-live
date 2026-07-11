#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Interleaves 3 separate uint8 planes (contiguous, row pitch == width) into
// an RGBA destination with the given byte pitch, leaving each pixel's 4th
// (alpha) byte untouched.
//
// This is the one hand-written CUDA kernel in this project. It's here
// because we checked the real npp.h (via grep against the installed
// headers) and confirmed no stock NPP function performs a
// "3-planes-in, skip-the-4th-channel-out" interleave -- the closest
// matches (nppiCopy_8u_C4P4R and friends) only go in the other direction
// (packed -> planar) or don't skip a channel. Everything else in this
// pipeline is NPP calls.
void launchPackRgbFromPlanes(
    const uint8_t* rPlane, const uint8_t* gPlane, const uint8_t* bPlane,
    uint8_t* outRgba, int outPitch, int width, int height, cudaStream_t stream);

