#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <iostream>

#include "pad.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

__global__ void PadData(
  const float *in,
  int          in_widthheight,
  int          in_width,
  int          in_height,
  int          out_width,
  int          out_height,
  int          channels,
  int          padding,
  float       *out) {
  int xy = blockIdx.x * blockDim.x + threadIdx.x;

  int x  = xy % in_width;
  int y  = xy / in_width;
  int ch = blockIdx.y;
  int n  = blockIdx.z;

  if (xy >= in_widthheight) {
    out[((n * out_height + y) * out_width + x) * channels + ch] = 0.0;
    return;
  }

  float value = in[((n * in_height + y) * in_width + x) * channels + ch];

  __syncthreads();

  int xpad = x + padding;
  int ypad = y + padding;

  out[((n * out_height + ypad) * out_width + xpad) * channels + ch] = value;
}

void Pad(const GPUDevice& device,
         const float     *input,
         int              batch_size,
         int              input_height,
         int              input_width,
         int              input_channels,
         int              output_height,
         int              output_width,
         float           *output) {
  int  in_widthheight    = input_width * input_height;
  int  threads_per_block = 16;
  dim3 totalBlocks((in_widthheight - 1) / threads_per_block + 1, input_channels, batch_size);

  cudaMemset(output, 0, batch_size * output_height * output_width * input_channels * sizeof(float));

  int padding = (output_height - input_height) / 2;

  // LAUNCH KERNEL
  PadData << < totalBlocks, threads_per_block, 0, device.stream() >> > (
    input,
    in_widthheight,
    input_width,
    input_height,
    output_width,
    output_height,
    input_channels,
    padding,
    output);
}
}
#endif // if GOOGLE_CUDA
