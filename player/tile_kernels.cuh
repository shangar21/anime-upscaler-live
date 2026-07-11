#pragma once
#include <cuda_runtime.h>

// Extracts all tiles from a planar fp32 frame into a packed TRT input batch
// in one parallel kernel launch, replacing the per-tile nppiCopy_32f_C1R loop.
//
// frame:     (3, frameH, frameW) planar fp32 [0,1]
// trtInput:  (numTiles, 3, tileH, tileW) packed fp32 -- TRT's input buffer
// tileOrigX/Y: arrays of tile top-left corner coords in the frame (LR space)
void launchExtractTiles(
    const float *frame, int frameW, int frameH,
    float *trtInput,
    const int *tileOrigX, const int *tileOrigY,
    int numTiles, int tileW, int tileH,
    cudaStream_t stream);

// Accumulates all output tiles from a packed TRT output batch into the
// weighted accumulator buffers in one parallel kernel launch, replacing the
// per-tile nppiMul + nppiAdd loop.
//
// trtOutput:   (numTiles, 3, tileHR, tileWR) packed fp32
// blendMask:   (tileHR, tileWR) cosine weight mask
// accum:       (3, outH, outW) accumulator -- written atomically
// weight:      (outH, outW) weight accumulator -- written atomically
// tileOrigX/Y: tile origins in HR (output) space
void launchStitchTiles(
    const float *trtOutput,
    const float *blendMask,
    float *accum, float *weight,
    const int *tileOrigX, const int *tileOrigY,
    int numTiles, int tileWR, int tileHR,
    int outW, int outH,
    cudaStream_t stream);
