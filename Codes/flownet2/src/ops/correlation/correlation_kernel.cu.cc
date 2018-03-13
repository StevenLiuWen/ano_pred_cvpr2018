#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32

#include <stdio.h>
#include <iostream>

#include "correlation_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

__global__ void CorrelateData(int          batch_size,
                              int          out_width,
                              int          out_height,
                              int          out_channels,
                              int          out_count,
                              int          max_displacement,
                              int          neighborhood_grid_radius,
                              int          neighborhood_grid_width,
                              int          kernel_radius,
                              int          kernel_size,
                              int          stride_1,
                              int          stride_2,
                              int          in_width_padded,
                              int          in_height_padded,
                              int          in_channels,
                              const float *input_a,
                              const float *input_b,
                              float       *output) {
  extern __shared__ char patch_data_char[];

  float *patch_data = (float *)patch_data_char;

  // First (upper left) position of kernel upper-left corner in current center
  // position of neighborhood in image 1
  int x1     = blockIdx.x * stride_1 + max_displacement;
  int y1     = blockIdx.y * stride_1 + max_displacement;
  int item   = blockIdx.z;
  int ch_off = threadIdx.x;

  // Load 3D patch into shared shared memory
  // HEIGHT
  for (int j = 0; j < kernel_size; j++) {
    // WIDTH
    for (int i = 0; i < kernel_size; i++) {
      int ji_off = ((j * kernel_size) + i) * in_channels;

      // CHANNELS
      for (int ch = ch_off; ch < in_channels; ch += (WARPS_PER_BLOCK * THREADS_PER_WARP)) {
        int idx1 = ((item * in_height_padded + y1 + j) * in_width_padded + x1 + i) *
                   in_channels + ch;
        int idxPatchData = ji_off + ch;
        patch_data[idxPatchData] = input_a[idx1];
      }
    }
  }

  __syncthreads();

  __shared__ float sum[WARPS_PER_BLOCK * THREADS_PER_WARP];

  // Compute correlation
  for (int out_channel = 0; out_channel < out_channels; out_channel++) {
    sum[ch_off] = 0;

    int s2o = (out_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride_2;
    int s2p = (out_channel / neighborhood_grid_width - neighborhood_grid_radius) * stride_2;
    int x2  = x1 + s2o;
    int y2  = y1 + s2p;

    // HEIGHT
    for (int j = 0; j < kernel_size; j++) {
      // WIDTH
      for (int i = 0; i < kernel_size; i++) {
        int ji_off = ((j * kernel_size) + i) * in_channels;

        // CHANNELS
        for (int ch = ch_off; ch < in_channels; ch += (WARPS_PER_BLOCK * THREADS_PER_WARP)) {
          int idxPatchData = ji_off + ch;
          int idx2         = ((item * in_height_padded + y2 + j) * in_width_padded + x2 + i) *
                             in_channels + ch;

          sum[ch_off] += patch_data[idxPatchData] * input_b[idx2];
        }
      }
    }

    __syncthreads();

    if (ch_off == 0) {
      float total_sum = 0;

      for (int idx = 0; idx < WARPS_PER_BLOCK * THREADS_PER_WARP; idx++) {
        total_sum += sum[idx];
      }
      const int sumelems = kernel_size * kernel_size * in_channels;
      const int index    = (blockIdx.y * out_width + blockIdx.x) * out_channels + out_channel;

      /* from Caffe:   const int index    = ((out_channel * out_height +
         blockIdx.y) * out_width) + blockIdx.x; */
      output[index + item * out_count] = total_sum / (float)sumelems;

      // Caffe, NKHW: ((n * K + k) * H + h) * W + w at point (n, k, h, w)
      // TF, NHWK: ((n * H + h) * W + w) * K + k at point (n, h, w, k)
      // n = 0
      // caffe: ((k * H + h) * W + w)  +   n * K * H * W
      // tf: (h * W + w) * K + k       +   n * H * W * K
    }
  }
}

void Correlation(const GPUDevice& device,
                 const float     *input_a,
                 const float     *input_b,
                 const int        batch_size,
                 const int        out_height,
                 const int        out_width,
                 const int        out_channels,
                 const int        out_count,
                 const int        in_height_padded,
                 const int        in_width_padded,
                 const int        in_channels,
                 int              max_displacement,
                 int              neighborhood_grid_radius,
                 int              neighborhood_grid_width,
                 int              kernel_radius,
                 int              kernel_size,
                 int              stride_1,
                 int              stride_2,
                 float           *output) {
  dim3 totalBlocksCorr(out_width, out_height, batch_size);
  dim3 threadsPerBlock(THREADS_PER_WARP *WARPS_PER_BLOCK);
  const int shared_memory_per_block = (kernel_size * kernel_size) * in_channels;

  CorrelateData << < totalBlocksCorr, threadsPerBlock, shared_memory_per_block * sizeof(float),
    device.stream() >> > (
    batch_size, out_width, out_height, out_channels, out_count,
    max_displacement, neighborhood_grid_radius, neighborhood_grid_width, kernel_radius,
    kernel_size, stride_1, stride_2, in_width_padded, in_height_padded, in_channels,
    input_a, input_b, output);
}
} // end namespace tensorflow

#endif  // GOOGLE_CUDA
