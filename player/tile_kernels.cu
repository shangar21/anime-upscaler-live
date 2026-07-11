#include "tile_kernels.cuh"

// ---------------------------------------------------------------------------
// Tile extraction
// One thread per (tile, channel, y, x) -- all tiles processed in parallel.
// Grid: (numTiles, 3, tileH) x (tileW) threads.
// ---------------------------------------------------------------------------
__global__ void extractTilesKernel(
    const float *frame, int frameW, int frameH,
    float *trtInput,
    const int *tileOrigX, const int *tileOrigY,
    int numTiles, int tileW, int tileH)
{
    int tile = blockIdx.x;
    int ch   = blockIdx.y;
    int y    = blockIdx.z;
    int x    = threadIdx.x;

    if (tile >= numTiles || x >= tileW || y >= tileH) return;

    int fx = tileOrigX[tile] + x;
    int fy = tileOrigY[tile] + y;
    // Clamp to frame bounds (handles edge tiles that were clamped in the grid)
    fx = min(fx, frameW - 1);
    fy = min(fy, frameH - 1);

    float val = frame[(size_t)ch * frameH * frameW + (size_t)fy * frameW + fx];
    trtInput[(size_t)tile * 3 * tileH * tileW + (size_t)ch * tileH * tileW + y * tileW + x] = val;
}

void launchExtractTiles(
    const float *frame, int frameW, int frameH,
    float *trtInput,
    const int *tileOrigX, const int *tileOrigY,
    int numTiles, int tileW, int tileH,
    cudaStream_t stream)
{
    // Grid: tile x channel x row, block: column
    // tileW is 64 -- fits in one warp-aligned block of threads
    dim3 grid(numTiles, 3, tileH);
    dim3 block(tileW);
    extractTilesKernel<<<grid, block, 0, stream>>>(
        frame, frameW, frameH, trtInput,
        tileOrigX, tileOrigY, numTiles, tileW, tileH);
}

// ---------------------------------------------------------------------------
// Tile stitching with cosine blend
// One thread per (tile, channel, y, x) -- all tiles blended in parallel.
// Uses atomicAdd since tiles can overlap and multiple threads may write to
// the same output pixel from adjacent tiles.
// ---------------------------------------------------------------------------
__global__ void stitchTilesKernel(
    const float *trtOutput,
    const float *blendMask,
    float *accum, float *weight,
    const int *tileOrigX, const int *tileOrigY,
    int numTiles, int tileW, int tileH,
    int outW, int outH)
{
    int tile = blockIdx.x;
    int ch   = blockIdx.y;
    int y    = blockIdx.z;
    int x    = threadIdx.x;

    if (tile >= numTiles || x >= tileW || y >= tileH) return;

    int ox = tileOrigX[tile] + x;
    int oy = tileOrigY[tile] + y;
    if (ox >= outW || oy >= outH) return;

    float maskVal = blendMask[y * tileW + x];
    float outVal  = trtOutput[(size_t)tile * 3 * tileH * tileW + (size_t)ch * tileH * tileW + y * tileW + x];
    float weighted = outVal * maskVal;

    atomicAdd(&accum [(size_t)ch * outH * outW + (size_t)oy * outW + ox], weighted);
    if (ch == 0)
        atomicAdd(&weight[(size_t)oy * outW + ox], maskVal);
}

void launchStitchTiles(
    const float *trtOutput,
    const float *blendMask,
    float *accum, float *weight,
    const int *tileOrigX, const int *tileOrigY,
    int numTiles, int tileW, int tileH,
    int outW, int outH,
    cudaStream_t stream)
{
    dim3 grid(numTiles, 3, tileH);
    dim3 block(tileW);
    stitchTilesKernel<<<grid, block, 0, stream>>>(
        trtOutput, blendMask, accum, weight,
        tileOrigX, tileOrigY, numTiles, tileW, tileH, outW, outH);
}
