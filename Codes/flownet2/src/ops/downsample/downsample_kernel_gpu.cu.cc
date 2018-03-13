#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <iostream>

#include "downsample_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#define CUDART_NAN_F            __int_as_float(0x7fffffff)

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

__global__ void DownsampleKernel(
    const int32 nthreads,
    const float* input_ptr,
    float* output_ptr,
    const int in_width,
    const int in_height,
    const int out_width,
    const int out_height,
    const int channels,
    const float width_scale,
    const float height_scale,
    const int wradius,
    const int hradius) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            const int c = index % channels;
            const int destx = (index / channels) % out_width;
            const int desty = (index / channels / out_width) % out_height;
            const int n = (index / channels / out_width) / out_height;

            const float srcx = ((float)destx / (float)(out_width - 1)) * (float)(in_width - 1);
            const float srcy = ((float)desty / (float)(out_height - 1)) * (float)(in_height - 1);

            const int isrcx = round(srcx);
            const int isrcy = round(srcy);

            float accum_value = 0;
            float accum_weight = 0;
            float accum_nan = 0;

            for (int dy = -hradius; dy <= hradius; dy++) {
                int yoff = isrcy + dy;
                //
                for (int dx = -wradius; dx <= wradius; dx++) {
                    int xoff = isrcx + dx;

                    if (xoff >= 0 && yoff >= 0 && xoff < in_width && yoff < in_height) {
                        int idx = ((n * in_height + yoff) * in_width + xoff) * channels + c;
                        float sample = input_ptr[idx];
                        float weight = fmaxf(0.0f, 1.0f - (fabsf((float)xoff - srcx) / width_scale))
                                       * fmaxf(0.0f, 1.0f - (fabsf((float)yoff - srcy) / height_scale));
                        if (sample != sample) { // isnan
                            accum_nan += weight;
                            sample = 0;
                            weight = 0;
                        }
                        accum_value += sample * weight;
                        accum_weight += weight;
                    }
                }
            }

            if (accum_nan / accum_weight > 0.5) {
                output_ptr[index] = CUDART_NAN_F;
            } else {
                output_ptr[index] = accum_value / accum_weight;
            }
        }
}

bool Downsample(const GPUDevice& device,
                typename TTypes<float, 4>::ConstTensor input,
                typename TTypes<float, 4>::Tensor output) {
    const int batch_size = output.dimension(0);
    const int out_height = output.dimension(1);
    const int out_width = output.dimension(2);
    const int out_channels = output.dimension(3);
    const int total_count = batch_size * out_height * out_width * out_channels;

    const int in_height = input.dimension(1);
    const int in_width = input.dimension(2);

    const float width_scale = (float)(in_width - 1) / (float)(out_width - 1);
    const float height_scale = (float)(in_height - 1) / (float)(out_height - 1);

    const int wradius = ceil(width_scale);
    const int hradius = ceil(height_scale);

    CudaLaunchConfig config = GetCudaLaunchConfig(total_count, device);
    DownsampleKernel<<<config.block_count, config.thread_per_block, 0,
                        device.stream()>>>(total_count, input.data(), output.data(),
                        in_width, in_height, out_width, out_height, out_channels,
                        width_scale, height_scale, wradius, hradius);
    return device.ok();
}

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
