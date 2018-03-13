#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "flow_warp.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

__global__ void FlowWarpGradKernel(
  const float *image,
  float       *image_grad,
  const float *flow,
  float       *flow_grad,
  const float *gradient,
  int          batch_size,
  int          channels,
  int          cblocks,
  int          width,
  int          wblocks,
  int          height,
  int          widthheight) {
  int x = blockIdx.x * FW_TILE_X + threadIdx.x;

  if (x >= width) return;

  int y = blockIdx.y;
  int n = blockIdx.z;

  const int flow_idx = ((n * height + y) * width + x) * 2;
  float     x2       = float(x) + flow[flow_idx];
  float     y2       = float(y) + flow[flow_idx + 1];

  if ((x2 >= 0.f) && (y2 >= 0.f) && (x2 < width) && (y2 < height)) {
    int ix2_L = int(x2);
    int iy2_T = int(y2);
    int ix2_R = min(ix2_L + 1, width - 1);
    int iy2_B = min(iy2_T + 1, height - 1);

    float alpha = x2 - ix2_L;
    float beta  = y2 - iy2_T;

    for (int c = 0; c < channels; c++) {
      float warped_diff_value = gradient[((n * height + y) * width + x) * channels + c];
      atomicAdd(&image_grad[((n * height + iy2_T) * width + ix2_L) * channels + c],
                warped_diff_value * (1 - alpha) * (1 - beta));
      atomicAdd(&image_grad[((n * height + iy2_T) * width + ix2_R) * channels + c],
                warped_diff_value * alpha * (1 - beta));
      atomicAdd(&image_grad[((n * height + iy2_B) * width + ix2_L) * channels + c],
                warped_diff_value * (1 - alpha) * beta);
      atomicAdd(&image_grad[((n * height + iy2_B) * width + ix2_R) * channels + c],
                warped_diff_value * alpha * beta);
    }

    float gamma    = iy2_B - y2;
    float bot_diff = 0;

    for (int c = 0; c < channels; c++) {
      int   ch_off = (n * channels + c) * height;
      float temp   = 0;
      temp += gamma *
              (image[((n * height + iy2_T) * width + ix2_R) * channels + c] -
               image[((n * height + iy2_T) * width + ix2_L) * channels + c]);
      temp += (1 - gamma) *
              (image[((n * height + iy2_B) * width + ix2_R) * channels + c] -
               image[((n * height + iy2_B) * width + ix2_L) * channels + c]);

      bot_diff += gradient[((n * height + y) * width + x) * channels + c] * temp;
    }
    flow_grad[((n * height + y) * width + x) * 2] = bot_diff;

    gamma    = ix2_R - x2;
    bot_diff = 0;

    for (int c = 0; c < channels; c++) {
      float temp = 0;
      temp += gamma *
              (image[((n * height + iy2_B) * width + ix2_L) * channels + c] -
               image[((n * height + iy2_T) * width + ix2_L) * channels + c]);
      temp += (1 - gamma) *
              (image[((n * height + iy2_B) * width + ix2_R) * channels + c] -
               image[((n * height + iy2_T) * width + ix2_R) * channels + c]);

      bot_diff += gradient[((n * height + y) * width + x) * channels + c] * temp;
    }
    flow_grad[((n * height + y) * width + x) * 2 + 1] = bot_diff;
  }
}

void FlowWarpGrad(const GPUDevice& device,
                  typename TTypes<float, 4>::ConstTensor image,
                  typename TTypes<float, 4>::ConstTensor flow,
                  typename TTypes<float, 4>::ConstTensor gradient,
                  typename TTypes<float, 4>::Tensor image_grad,
                  typename TTypes<float, 4>::Tensor flow_grad) {
  const int batch_size   = image.dimension(0);
  const int height       = image.dimension(1);
  const int width        = image.dimension(2);
  const int channels     = image.dimension(3);
  const int width_height = width * height;

  int  wblocks = ((width - 1) / FW_TILE_X + 1);
  int  cblocks = ((channels - 1) / FW_TILE_C + 1);
  dim3 warpThreads(FW_TILE_X, 1);
  dim3 warpBlocks(wblocks, height, batch_size);

  cudaMemset(image_grad.data(), 0, batch_size * height * width * channels * sizeof(float));
  cudaMemset(flow_grad.data(),  0, batch_size * height * width * 2 * sizeof(float));

  FlowWarpGradKernel << < warpBlocks, warpThreads, 0, device.stream() >> > (
    image.data(),
    image_grad.data(),
    flow.data(),
    flow_grad.data(),
    gradient.data(),
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
