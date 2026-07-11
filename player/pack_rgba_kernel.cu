#include "pack_rgba_kernel.cuh"

__global__ void packRgbFromPlanesKernel(
    const uint8_t* r, const uint8_t* g, const uint8_t* b,
    uint8_t* out, int outPitch, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uint8_t* px = out + size_t(y) * outPitch + size_t(x) * 4;
    px[0] = r[idx];
    px[1] = g[idx];
    px[2] = b[idx];
    // px[3] (alpha) intentionally left untouched.
}

void launchPackRgbFromPlanes(
    const uint8_t* rPlane, const uint8_t* gPlane, const uint8_t* bPlane,
    uint8_t* outRgba, int outPitch, int width, int height, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    packRgbFromPlanesKernel<<<grid, block, 0, stream>>>(
        rPlane, gPlane, bPlane, outRgba, outPitch, width, height);
}

