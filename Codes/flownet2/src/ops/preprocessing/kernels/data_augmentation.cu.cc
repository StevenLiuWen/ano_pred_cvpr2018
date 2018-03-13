#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "augmentation_base.h"
#include "data_augmentation.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
inline __device__ __host__ float clamp(float f, float a, float b) {
  return fmaxf(a, fminf(f, b));
}

__global__ void SpatialAugmentation(
  const int32  nthreads,
  const int    src_width,
  const int    src_height,
  const int    channels,
  const int    src_count,
  const int    out_width,
  const int    out_height,
  const float *src_data,
  float       *out_data,
  const float *transMats) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // Caffe, NKHW: ((n * K + k) * H + h) * W + w at point (n, k, h, w)
    // TF, NHWK: ((n * H + h) * W + w) * K + k at point (n, h, w, k)
    int c = index % channels;
    int x = (index / channels) % out_width;
    int y = (index / channels / out_width) % out_height;
    int n = index / channels / out_width / out_height;

    const float *transMat = transMats + n * 6;
    float xpos            = x * transMat[0] + y * transMat[1] + transMat[2];
    float ypos            = x * transMat[3] + y * transMat[4] + transMat[5];

    xpos = clamp(xpos, 0.0f, (float)(src_width) - 1.05f);
    ypos = clamp(ypos, 0.0f, (float)(src_height) - 1.05f);

    float tlx = floor(xpos);
    float tly = floor(ypos);

    // Bilinear interpolation
    int srcTLIdx = ((n * src_height + tly) * src_width + tlx) * channels + c;
    int srcTRIdx = min((int)(((n * src_height + tly) * src_width + (tlx + 1)) * channels + c),
                       src_count);
    int srcBLIdx = min((int)(((n * src_height + (tly + 1)) * src_width + tlx) * channels + c),
                       src_count);
    int srcBRIdx = min((int)(((n * src_height + (tly + 1)) * src_width + (tlx + 1)) * channels + c),
                       src_count);

    float xdist = xpos - tlx;
    float ydist = ypos - tly;

    float dest = (1 - xdist) * (1 - ydist) * src_data[srcTLIdx]
                 + (xdist) * (ydist) * src_data[srcBRIdx]
                 + (1 - xdist) * (ydist) * src_data[srcBLIdx]
                 + (xdist) * (1 - ydist) * src_data[srcTRIdx];

    out_data[index] = dest;
  }
}

typedef Eigen::GpuDevice GPUDevice;

template<>
void Augment(OpKernelContext *context,
             const GPUDevice& d,
             const int        batch_size,
             const int        channels,
             const int        src_width,
             const int        src_height,
             const int        src_count,
             const int        out_width,
             const int        out_height,
             const float     *src_data,
             float           *out_data,
             const float     *transMats,
             float           *chromatic_coeffs) {
  const int out_count     = batch_size * out_height * out_width * channels;
  CudaLaunchConfig config = GetCudaLaunchConfig(out_count, d);

  printf("Chromatic transform not yet implemented on GPU, ignoring.");

  SpatialAugmentation << < config.block_count, config.thread_per_block, 0, d.stream() >> > (
    config.virtual_thread_count, src_width, src_height, channels, src_count,
    out_width, out_height,
    src_data, out_data, transMats);
}

//
// template<typename Device>
// class DataAugmentation : public OpKernel {
//   public:
//     explicit DataAugmentation(OpKernelConstruction *ctx) : OpKernel(ctx) {
//       // Get the crop [height, width] tensor and verify its dimensions
//       OP_REQUIRES_OK(ctx, ctx->GetAttr("crop", &crop_));
//       OP_REQUIRES(ctx, crop_.size() == 2,
//                   errors::InvalidArgument("crop must be 2 dimensions"));
//
//       // TODO: Verify params are all the same length
//
//       // Get the tensors for params_a and verify their dimensions
//       OP_REQUIRES_OK(ctx, ctx->GetAttr("params_a_name", &params_a_name_));
//       OP_REQUIRES_OK(ctx,
//                      ctx->GetAttr("params_a_rand_type",
// &params_a_rand_type_));
//       OP_REQUIRES_OK(ctx, ctx->GetAttr("params_a_exp", &params_a_exp_));
//       OP_REQUIRES_OK(ctx, ctx->GetAttr("params_a_mean", &params_a_mean_));
//       OP_REQUIRES_OK(ctx, ctx->GetAttr("params_a_spread",
// &params_a_spread_));
//       OP_REQUIRES_OK(ctx, ctx->GetAttr("params_a_prob", &params_a_prob_));
//
//       // Get the tensors for params_b and verify their dimensions
//       OP_REQUIRES_OK(ctx, ctx->GetAttr("params_b_name", &params_b_name_));
//       OP_REQUIRES_OK(ctx,
//                      ctx->GetAttr("params_b_rand_type",
// &params_b_rand_type_));
//       OP_REQUIRES_OK(ctx, ctx->GetAttr("params_b_exp", &params_b_exp_));
//       OP_REQUIRES_OK(ctx, ctx->GetAttr("params_b_mean", &params_b_mean_));
//       OP_REQUIRES_OK(ctx, ctx->GetAttr("params_b_spread",
// &params_b_spread_));
//       OP_REQUIRES_OK(ctx, ctx->GetAttr("params_b_prob", &params_b_prob_));
//     }
//
//     void Compute(OpKernelContext *ctx) override {
//       const GPUDevice& device = ctx->eigen_gpu_device();
//
//       // Get the input images
//       const Tensor& input_a_t = ctx->input(0);
//       const Tensor& input_b_t = ctx->input(1);
//
//       // Dimension constants
//       const int batch_size = input_a_t.dim_size(0);
//       const int src_height = input_a_t.dim_size(1);
//       const int src_width  = input_a_t.dim_size(2);
//       const int channels   = input_a_t.dim_size(3);
//       const int src_count  = batch_size * src_height * src_width * channels;
//       const int out_height = crop_[0];
//       const int out_width  = crop_[1];
//       const int out_count  = batch_size * out_height * out_width * channels;
//
//       // Allocate the memory for the output images
//       Tensor *output_a_t;
//       Tensor *output_b_t;
//
//       OP_REQUIRES_OK(ctx,
//                      ctx->allocate_output(0, TensorShape({ batch_size,
// crop_[0], crop_[1],
//                                                            channels }),
// &output_a_t));
//       OP_REQUIRES_OK(ctx,
//                      ctx->allocate_output(1, TensorShape({ batch_size,
// crop_[0], crop_[1],
//                                                            channels }),
// &output_b_t));
//
//       // Allocate the memory for the output spatial transforms
//       Tensor *spat_transform_a_t;
//       Tensor *spat_transform_b_t;
//
//       OP_REQUIRES_OK(ctx,
//                      ctx->allocate_output(2, TensorShape({ batch_size, 6 }),
//  &spat_transform_a_t));
//       OP_REQUIRES_OK(ctx,
//                      ctx->allocate_output(3, TensorShape({ batch_size, 6 }),
// &spat_transform_b_t));
//
//       // Allocate temporary pinned memory for the spatial transforms to be
// used
//       // on the GPU
//       tensorflow::AllocatorAttributes pinned_allocator;
//       pinned_allocator.set_on_host(true);
//       pinned_allocator.set_gpu_compatible(true);
//
//       Tensor spat_transform_a_pinned_t;
//       Tensor spat_transform_b_pinned_t;
//       OP_REQUIRES_OK(ctx,
//                      ctx->allocate_temp(DataTypeToEnum<float>::value,
//                                         TensorShape({ batch_size, 6 }),
//                                         &spat_transform_a_pinned_t,
// pinned_allocator));
//       OP_REQUIRES_OK(ctx,
//                      ctx->allocate_temp(DataTypeToEnum<float>::value,
//                                         TensorShape({ batch_size, 6 }),
//                                         &spat_transform_b_pinned_t,
// pinned_allocator));
//       auto spat_transform_a_pinned = spat_transform_a_pinned_t.tensor<float,
// 2>();
//       auto spat_transform_b_pinned = spat_transform_b_pinned_t.tensor<float,
// 2>();
//
//       /*** BEGIN AUGMENTATION TO IMAGE A ***/
//       auto input_a  = input_a_t.tensor<float, 4>();
//       auto output_a = output_a_t->tensor<float, 4>();
//
//       // Load augmentation parameters for image A
//       AugmentationParams aug_a = AugmentationParams(out_height, out_width,
//                                                     params_a_name_,
//                                                     params_a_rand_type_,
//                                                     params_a_exp_,
//                                                     params_a_mean_,
//                                                     params_a_spread_,
//                                                     params_a_prob_);
//
//       std::vector<AugmentationCoeff> coeffs_a;
//
//       bool gen_spatial_transform = aug_a.should_do_spatial_transform();
//
//       for (int n = 0; n < batch_size; n++) {
//         AugmentationCoeff coeff;
//
//         if (gen_spatial_transform) {
//           AugmentationLayerBase::generate_valid_spatial_coeffs(aug_a, coeff,
//                                                                src_width,
// src_height,
//                                                                out_width,
// out_height);
//         }
//
//         coeffs_a.push_back(coeff);
//       }
//
//       // Copy spatial coefficients A to the output Tensor on the CPU (output
// for
//       // FlowAugmentation)
//       auto spat_transform_a = spat_transform_a_t->tensor<float, 2>();
//       AugmentationLayerBase::copy_spatial_coeffs_to_tensor(coeffs_a,
//                                                            out_width,
// out_height,
//                                                            src_width,
// src_height,
//                                                            spat_transform_a);
//
//       // ...as well as a Tensor going to the GPU
//       AugmentationLayerBase::copy_spatial_coeffs_to_tensor(coeffs_a,
//                                                            out_width,
//                                                            out_height,
//                                                            src_width,
//                                                            src_height,
//
//
//
//                                                    spat_transform_a_pinned);
//
//       CudaLaunchConfig config = GetCudaLaunchConfig(out_count, device);
//       SpatialAugmentation << < config.block_count, config.thread_per_block,
// 0,
//         device.stream() >> > (
//         config.virtual_thread_count, src_width, src_height, channels,
// src_count,
//         out_width, out_height,
//         input_a.data(), output_a.data(), spat_transform_a_pinned.data());
//
//       /*** END AUGMENTATION TO IMAGE A ***/
//
//       /*** BEGIN GENERATE NEW COEFFICIENTS FOR IMAGE B ***/
//       AugmentationParams aug_b = AugmentationParams(out_height, out_width,
//                                                     params_b_name_,
//                                                     params_b_rand_type_,
//                                                     params_b_exp_,
//                                                     params_b_mean_,
//                                                     params_b_spread_,
//                                                     params_b_prob_);
//
//       std::vector<AugmentationCoeff> coeffs_b;
//
//       gen_spatial_transform = aug_b.should_do_spatial_transform();
//
//       for (int n = 0; n < batch_size; n++) {
//         AugmentationCoeff coeff;
//
//         if (gen_spatial_transform) {
//           AugmentationLayerBase::generate_valid_spatial_coeffs(aug_b, coeff,
//                                                                src_width,
// src_height,
//                                                                out_width,
// out_height);
//         }
//
//         coeffs_b.push_back(coeff);
//       }
//
//       /*** END GENERATE NEW COEFFICIENTS FOR IMAGE B ***/
//
//       /*** BEGIN AUGMENTATION TO IMAGE B ***/
//       auto input_b  = input_b_t.tensor<float, 4>();
//       auto output_b = output_b_t->tensor<float, 4>();
//
//       // Copy spatial coefficients B to the output Tensor on the CPU
//       auto spat_transform_b = spat_transform_b_t->tensor<float, 2>();
//       AugmentationLayerBase::copy_spatial_coeffs_to_tensor(coeffs_b,
//                                                            out_width,
// out_height,
//                                                            src_width,
// src_height,
//                                                            spat_transform_b,
//                                                            true);
//       AugmentationLayerBase::copy_spatial_coeffs_to_tensor(coeffs_b,
//                                                            out_width,
// out_height,
//                                                            src_width,
// src_height,
//
//
//
//                                                    spat_transform_b_pinned);
//
//       SpatialAugmentation << < config.block_count, config.thread_per_block,
// 0,
//         device.stream() >> > (
//         config.virtual_thread_count, src_width, src_height, channels,
// src_count,
//         out_width, out_height,
//         input_b.data(), output_b.data(), spat_transform_b_pinned.data());
//
//       /*** END AUGMENTATION TO IMAGE B ***/
//     }
//
//   private:
//     std::vector<int32>crop_;
//
//     // Params A
//     std::vector<string>params_a_name_;
//     std::vector<string>params_a_rand_type_;
//     std::vector<bool>params_a_exp_;
//     std::vector<float>params_a_mean_;
//     std::vector<float>params_a_spread_;
//     std::vector<float>params_a_prob_;
//
//     // Params B
//     std::vector<string>params_b_name_;
//     std::vector<string>params_b_rand_type_;
//     std::vector<bool>params_b_exp_;
//     std::vector<float>params_b_mean_;
//     std::vector<float>params_b_spread_;
//     std::vector<float>params_b_prob_;
// };
} // namespace tensorflow
#endif // GOOGLE_CUDA
