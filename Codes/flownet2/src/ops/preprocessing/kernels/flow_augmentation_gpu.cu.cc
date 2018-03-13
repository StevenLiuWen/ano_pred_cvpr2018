#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <iostream>

#include "flow_augmentation.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

inline __device__ __host__ int clamp(int f, int a, int b) {
  return max(a, min(f, b));
}

__global__ void FillFlowAugmentationKernel(
  const int32 nthreads,
  const float *flow_ptr,
  const float *transforms_from_a,
  const float *inv_transforms_from_b,
  const int src_total_count, const int src_height, const int src_width,
  const int batch_size, const int out_height,
  const int out_width, float *output_ptr) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const float x = (float)(index % out_width);
    const float y = (float)((index / out_width) % out_height);
    const int   n = (index / out_width / out_height);

    const int transformIdx = n * 6;

    // Apply transformation matrix applied to second image
    const float xpos1 = x * transforms_from_a[transformIdx + 0]
                        + y * transforms_from_a[transformIdx + 1]
                        + transforms_from_a[transformIdx + 2];
    const float ypos1 = x * transforms_from_a[transformIdx + 3]
                        + y * transforms_from_a[transformIdx + 4]
                        + transforms_from_a[transformIdx + 5];

    // Caffe, NKHW: ((n * K + k) * H + h) * W + w at point (n, k, h, w)
    // TF, NHWK: ((n * H + h) * W + w) * K + k at point (n, h, w, k)
    const int srcXIdx =
      ((n * src_height + (int)(ypos1 + 0.5)) * src_width + (int)(xpos1 + 0.5)) *
      2 + 0;
    const int srcYIdx = srcXIdx + 1;

    const float xpos2 = xpos1 + flow_ptr[clamp(srcXIdx, 0, src_total_count - 1)];
    const float ypos2 = ypos1 + flow_ptr[clamp(srcYIdx, 0, src_total_count - 1)];

    // Apply inverse of the transformation matrix applied to first image
    const float xpos3 = xpos2 * inv_transforms_from_b[transformIdx + 0]
                        + ypos2 * inv_transforms_from_b[transformIdx + 1]
                        + inv_transforms_from_b[transformIdx + 2];
    const float ypos3 = xpos2 * inv_transforms_from_b[transformIdx + 3]
                        + ypos2 * inv_transforms_from_b[transformIdx + 4]
                        + inv_transforms_from_b[transformIdx + 5];

    output_ptr[((n * out_height + (int)y) * out_width + (int)x) * 2 + 0] = xpos3 -
                                                                           x;
    output_ptr[((n * out_height + (int)y) * out_width + (int)x) * 2 + 1] = ypos3 -
                                                                           y;
  }
}

template<>
void FillFlowAugmentation(const GPUDevice& device,
                          typename TTypes<float, 4>::Tensor output,
                          typename TTypes<float, 4>::ConstTensor flows,
                          typename TTypes<const float, 2>::ConstTensor transforms_from_a,
                          typename TTypes<const float, 2>::ConstTensor transforms_from_b) {
  const int batch_size      = output.dimension(0);
  const int out_height      = output.dimension(1);
  const int out_width       = output.dimension(2);
  const int depth           = 2;
  const int total_count     = batch_size * out_height * out_width * depth;
  const int src_total_count = flows.dimension(0) * flows.dimension(1) *
                              flows.dimension(2) * flows.dimension(3);

  CudaLaunchConfig config = GetCudaLaunchConfig(total_count / 2, device);

  FillFlowAugmentationKernel << < config.block_count, config.thread_per_block, 0,
    device.stream() >> > (
    total_count / 2, flows.data(), transforms_from_a.data(),
    transforms_from_b.data(),
    src_total_count, flows.dimension(1), flows.dimension(2), batch_size,
    out_height, out_width, output.data());
}
} // end namespace tensorflow

#endif  // GOOGLE_CUDA
