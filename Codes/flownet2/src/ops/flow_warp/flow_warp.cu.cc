#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <iostream>

#include "flow_warp.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#define RA_TILE 32
#define RA_ROWS 8

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

__global__ void FlowWarpKernel(
  const float *image,
  const float *flow,
  float       *warped,
  const int    batch_size,
  const int    channels,
  const int    cblocks,
  const int    width,
  const int    wblocks,
  const int    height,
  const int    width_height) {
  int y = blockIdx.y;
  int n = blockIdx.z;

  __shared__ float x2_buf[FW_TILE_X], y2_buf[FW_TILE_X];
  __shared__ float buffer[FW_TILE_C][FW_TILE_X + 1];

  int x;
  int c;

  x = blockIdx.x * FW_TILE_X + threadIdx.x;

  if ((threadIdx.y == 0) && (x < width)) {
    const int idx = ((n * height + y) * width + x) * 2;
    x2_buf[threadIdx.x] = float(x) + flow[idx];
    y2_buf[threadIdx.x] = float(y) + flow[idx + 1];
  }

  __syncthreads();

  float x2 = x2_buf[threadIdx.y];
  float y2 = y2_buf[threadIdx.y];

  int ix2_L = int(x2);
  int iy2_T = int(y2);
  int ix2_R = min(ix2_L + 1, width - 1);
  int iy2_B = min(iy2_T + 1, height - 1);

  int off_TL = ((n * height + iy2_T) * width + ix2_L) * channels;
  int off_TR = ((n * height + iy2_T) * width + ix2_R) * channels;
  int off_BL = ((n * height + iy2_B) * width + ix2_L) * channels;
  int off_BR = ((n * height + iy2_B) * width + ix2_R) * channels;

  float alpha   = x2 - ix2_L;
  float beta    = y2 - iy2_T;
  float coeffTL = (1 - alpha) * (1 - beta);
  float coeffTR = alpha * (1 - beta);
  float coeffBL = (1 - alpha) * beta;
  float coeffBR = alpha * beta;

  for (int cb = 0; cb < cblocks; cb++) {
    __syncthreads();

    buffer[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    c = cb * FW_TILE_C + threadIdx.x;

    if ((x2 >= 0) && (y2 >= 0) && (x2 < width) && (y2 < height) && (c < channels)) {
      buffer[threadIdx.y][threadIdx.x] = // buffer [x][c]
                                         coeffTL * image[off_TL + c] +
                                         coeffTR * image[off_TR + c] +
                                         coeffBL * image[off_BL + c] +
                                         coeffBR * image[off_BR + c];
    }

    __syncthreads();

    c = cb * FW_TILE_C + threadIdx.y;
    x = blockIdx.x * FW_TILE_X + threadIdx.x;

    if ((c < channels) && (x < width)) {
      warped[((n * height + y) * width + x) * channels + c] = buffer[threadIdx.x][threadIdx.y];
    }
  }
}

void FlowWarp(const GPUDevice& device,
              typename TTypes<float, 4>::ConstTensor input,
              typename TTypes<float, 4>::ConstTensor flow,
              typename TTypes<float, 4>::Tensor output) {
  const int batch_size = input.dimension(0);
  const int height     = input.dimension(1);
  const int width      = input.dimension(2);
  const int channels   = input.dimension(3);

  const int width_height = width * height;
  int  wblocks           = ((width - 1) / FW_TILE_X + 1);
  int  cblocks           = ((channels - 1) / FW_TILE_C + 1);
  dim3 warpThreads(FW_TILE_X, FW_TILE_C);
  dim3 warpBlocks(wblocks, height, batch_size);

  cudaMemset(output.data(), 0, batch_size * height * width * 2 * sizeof(float));

  FlowWarpKernel << < warpBlocks, warpThreads, 0, device.stream() >> > (
    input.data(),
    flow.data(),
    output.data(),
    batch_size,
    channels,
    cblocks,
    width,
    wblocks,
    height,
    width_height);
}
} // end namespace tensorflow

#endif  // GOOGLE_CUDA
