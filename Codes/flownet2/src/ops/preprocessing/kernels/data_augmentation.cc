#define EIGEN_USE_THREADS

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "augmentation_base.h"
#include "data_augmentation.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice        GPUDevice;

inline float clamp(float f, float a, float b) {
  return fmaxf(a, fminf(f, b));
}

template<>
void Augment(OpKernelContext *context,
             const CPUDevice& d,
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
  const int64 channel_count                          = batch_size * out_height * out_width;
  const int   kCostPerChannel                        = 10;
  const DeviceBase::CpuWorkerThreads& worker_threads =
    *context->device()->tensorflow_cpu_worker_threads();

  Shard(worker_threads.num_threads,
        worker_threads.workers,
        channel_count,
        kCostPerChannel,
        [batch_size, channels, src_width,
         src_height, src_count, out_width, out_height, src_data,
         out_data, transMats, chromatic_coeffs](
          int64 start_channel, int64 end_channel) {
      // TF, NHWK: ((n * H + h) * W + w) * K + k at point (n, h, w, k)
      for (int index = start_channel; index < end_channel; index++) {
        int x = index % out_width;
        int y = (index / out_width) % out_height;
        int n = index / out_width / out_height;

        const float *transMat = transMats + n * 6;

        float gamma, brightness, contrast;

        if (chromatic_coeffs) {
          gamma      = chromatic_coeffs[n * 6 + 0];
          brightness = chromatic_coeffs[n * 6 + 1];
          contrast   = chromatic_coeffs[n * 6 + 2];
        }

        float xpos = x * transMat[0] + y * transMat[1] + transMat[2];
        float ypos = x * transMat[3] + y * transMat[4] + transMat[5];

        xpos = clamp(xpos, 0.0f, (float)(src_width) - 1.05f);
        ypos = clamp(ypos, 0.0f, (float)(src_height) - 1.05f);

        float tlx = floor(xpos);
        float tly = floor(ypos);

        float xdist = xpos - tlx;
        float ydist = ypos - tly;

        int srcTLIdxOffset = ((n * src_height + (int)tly) * src_width + (int)tlx) * channels;

        // ((n * src_height + tly) * src_width + (tlx + 1)) * channels
        int srcTRIdxOffset = srcTLIdxOffset + channels;

        // ((n * src_height + (tly + 1)) * src_width + tlx) * channels
        int srcBLIdxOffset = srcTLIdxOffset + channels * src_width;

        // ((n * src_height + (tly + 1)) * src_width + (tlx + 1)) * channels
        int srcBRIdxOffset = srcTLIdxOffset + channels + channels * src_width;

        // Variables for chromatic transform
        int   data_index[3];
        float rgb[3];
        float mean_in  = 0;
        float mean_out = 0;

        for (int c = 0; c < channels; c++) {
          // Bilinear interpolation
          int srcTLIdx = srcTLIdxOffset + c;
          int srcTRIdx = std::min(srcTRIdxOffset + c, src_count);
          int srcBLIdx = std::min(srcBLIdxOffset + c, src_count);
          int srcBRIdx = std::min(srcBRIdxOffset + c, src_count);

          float dest = (1 - xdist) * (1 - ydist) * src_data[srcTLIdx]
                       + (xdist) * (ydist) * src_data[srcBRIdx]
                       + (1 - xdist) * (ydist) * src_data[srcBLIdx]
                       + (xdist) * (1 - ydist) * src_data[srcTRIdx];

          if (chromatic_coeffs) {
            // Gather data for chromatic transform
            data_index[c] = index * channels + c;
            rgb[c]        = dest;
            mean_in      += rgb[c];

            // Note: coeff[3] == color1, coeff[4] == color2, ...
            rgb[c] *= chromatic_coeffs[n * 6 + (3 + c)];

            mean_out += rgb[c];
          } else {
            out_data[index * channels + c] = dest;
          }
        }

        float brightness_coeff = mean_in / (mean_out + 0.01f);

        if (chromatic_coeffs) {
          // Chromatic transformation
          for (int c = 0; c < channels; c++) {
            // compensate brightness
            rgb[c] = clamp(rgb[c] * brightness_coeff, 0.0f, 1.0f);

            // gamma change
            rgb[c] = pow(rgb[c], gamma);

            // brightness change
            rgb[c] = rgb[c] + brightness;

            // contrast change
            rgb[c] = 0.5f + (rgb[c] - 0.5f) * contrast;

            out_data[data_index[c]] = clamp(rgb[c], 0.0f, 1.0f);
          }
        }
      }
    });
}

template<typename Device>
class DataAugmentation : public OpKernel {
  public:
    explicit DataAugmentation(OpKernelConstruction *ctx) : OpKernel(ctx) {
      // Get the crop [height, width] tensor and verify its dimensions
      OP_REQUIRES_OK(ctx, ctx->GetAttr("crop", &crop_));
      OP_REQUIRES(ctx, crop_.size() == 2,
                  errors::InvalidArgument("crop must be 2 dimensions"));

      // TODO: Verify params are all the same length

      // Get the tensors for params_a and verify their dimensions
      OP_REQUIRES_OK(ctx, ctx->GetAttr("params_a_name", &params_a_name_));
      OP_REQUIRES_OK(ctx,
                     ctx->GetAttr("params_a_rand_type", &params_a_rand_type_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("params_a_exp", &params_a_exp_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("params_a_mean", &params_a_mean_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("params_a_spread", &params_a_spread_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("params_a_prob", &params_a_prob_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("params_a_coeff_schedule", &params_a_coeff_schedule_));

      // Get the tensors for params_b and verify their dimensions
      OP_REQUIRES_OK(ctx, ctx->GetAttr("params_b_name", &params_b_name_));
      OP_REQUIRES_OK(ctx,
                     ctx->GetAttr("params_b_rand_type", &params_b_rand_type_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("params_b_exp", &params_b_exp_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("params_b_mean", &params_b_mean_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("params_b_spread", &params_b_spread_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("params_b_prob", &params_b_prob_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("params_b_coeff_schedule", &params_b_coeff_schedule_));
    }

    void Compute(OpKernelContext *ctx) override {
      // Get the input images
      const Tensor& input_a_t = ctx->input(0);
      const Tensor& input_b_t = ctx->input(1);

      // Get the global step value
      const Tensor& global_step_t = ctx->input(2);
      auto global_step_eigen      = global_step_t.tensor<int64, 0>();
      const int64 global_step     = global_step_eigen.data()[0];

      // Dimension constants
      const int batch_size = input_a_t.dim_size(0);
      const int src_height = input_a_t.dim_size(1);
      const int src_width  = input_a_t.dim_size(2);
      const int channels   = input_a_t.dim_size(3);
      const int src_count  = batch_size * src_height * src_width * channels;
      const int out_height = crop_[0];
      const int out_width  = crop_[1];
      const int out_count  = batch_size * out_height * out_width * channels;

      // All tensors for this op
      Tensor chromatic_coeffs_a_t;
      Tensor chromatic_coeffs_b_t;

      // Allocate the memory for the output images
      Tensor *output_a_t;
      Tensor *output_b_t;

      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(0, TensorShape({ batch_size, crop_[0], crop_[1],
                                                           channels }), &output_a_t));
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(1, TensorShape({ batch_size, crop_[0], crop_[1],
                                                           channels }), &output_b_t));

      // Allocate the memory for the output spatial transforms
      Tensor *spat_transform_a_t;
      Tensor *spat_transform_b_t;

      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(2, TensorShape({ batch_size, 6 }),
                                          &spat_transform_a_t));
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(3, TensorShape({ batch_size, 6 }),
                                          &spat_transform_b_t));

      // Compute discount for coefficients if using a schedule
      float discount_coeff_a = 1.0;
      float discount_coeff_b = 1.0;

      if (params_a_coeff_schedule_.size() == 3) {
        float half_life     = params_a_coeff_schedule_[0];
        float initial_coeff = params_a_coeff_schedule_[1];
        float final_coeff   = params_a_coeff_schedule_[2];
        discount_coeff_a = initial_coeff + (final_coeff - initial_coeff) *
                           (2.0 / (1.0 + exp(-1.0986 * global_step / half_life)) - 1.0);
      }

      if (params_b_coeff_schedule_.size() == 3) {
        if (params_a_coeff_schedule_.size() == 3) {
          discount_coeff_b = discount_coeff_a;
        } else {
          float half_life     = params_b_coeff_schedule_[0];
          float initial_coeff = params_b_coeff_schedule_[1];
          float final_coeff   = params_b_coeff_schedule_[2];
          discount_coeff_b = initial_coeff + (final_coeff - initial_coeff) *
                             (2.0 / (1.0 + exp(-1.0986 * global_step / half_life)) - 1.0);
        }
      }

      /*** BEGIN AUGMENTATION TO IMAGE A ***/
      auto input_a  = input_a_t.tensor<float, 4>();
      auto output_a = output_a_t->tensor<float, 4>();

      // Load augmentation parameters for image A
      AugmentationParams aug_a = AugmentationParams(out_height, out_width,
                                                    params_a_name_,
                                                    params_a_rand_type_,
                                                    params_a_exp_,
                                                    params_a_mean_,
                                                    params_a_spread_,
                                                    params_a_prob_);

      std::vector<AugmentationCoeff> coeffs_a;


      bool gen_spatial_transform   = aug_a.should_do_spatial_transform();
      bool gen_chromatic_transform = aug_a.should_do_chromatic_transform();

      for (int n = 0; n < batch_size; n++) {
        AugmentationCoeff coeff;

        if (gen_spatial_transform) {
          AugmentationLayerBase::generate_valid_spatial_coeffs(discount_coeff_a, aug_a, coeff,
                                                               src_width, src_height,
                                                               out_width, out_height);
        }

        if (gen_chromatic_transform) {
          AugmentationLayerBase::generate_chromatic_coeffs(discount_coeff_a, aug_a, coeff);
        }

        coeffs_a.push_back(coeff);
      }

      // Copy spatial coefficients A to the output Tensor on the CPU
      // (output for FlowAugmentation)
      auto spat_transform_a = spat_transform_a_t->tensor<float, 2>();
      AugmentationLayerBase::copy_spatial_coeffs_to_tensor(coeffs_a,
                                                           out_width, out_height,
                                                           src_width, src_height,
                                                           spat_transform_a);

      float *chromatic_coeffs_a_data = NULL;

      if (gen_chromatic_transform) {
        // Allocate a temporary tensor to hold the chromatic coefficients
        OP_REQUIRES_OK(ctx,
                       ctx->allocate_temp(DataTypeToEnum<float>::value,
                                          TensorShape({ batch_size, 6 }),
                                          &chromatic_coeffs_a_t));

        // Copy the chromatic coefficients A to a temporary Tensor on the CPU
        auto chromatic_coeffs_a = chromatic_coeffs_a_t.tensor<float, 2>();
        AugmentationLayerBase::copy_chromatic_coeffs_to_tensor(coeffs_a, chromatic_coeffs_a);
        chromatic_coeffs_a_data = chromatic_coeffs_a.data();
      }

      // Perform augmentation either on CPU or GPU
      Augment<Device>(
        ctx,
        ctx->eigen_device<Device>(),
        batch_size,
        channels,
        src_width,
        src_height,
        src_count,
        out_width,
        out_height,
        input_a.data(),
        output_a.data(),
        spat_transform_a.data(),
        chromatic_coeffs_a_data);

      /*** END AUGMENTATION TO IMAGE A ***/

      /*** BEGIN GENERATE NEW COEFFICIENTS FOR IMAGE B ***/
      AugmentationParams aug_b = AugmentationParams(out_height, out_width,
                                                    params_b_name_,
                                                    params_b_rand_type_,
                                                    params_b_exp_,
                                                    params_b_mean_,
                                                    params_b_spread_,
                                                    params_b_prob_);

      std::vector<AugmentationCoeff> coeffs_b;

      bool gen_spatial_transform_b   = aug_b.should_do_spatial_transform();
      bool gen_chromatic_transform_b = aug_b.should_do_chromatic_transform();

      for (int n = 0; n < batch_size; n++) {
        AugmentationCoeff coeff(coeffs_a[n]);

        // If we did a spatial transform on image A, we need to do the same one
        // (+ possibly more) on image B
        if (gen_spatial_transform_b) {
          AugmentationLayerBase::generate_valid_spatial_coeffs(discount_coeff_b, aug_b, coeff,
                                                               src_width, src_height,
                                                               out_width, out_height);
        }

        if (gen_chromatic_transform_b) {
          AugmentationLayerBase::generate_chromatic_coeffs(discount_coeff_b, aug_b, coeff);
        }

        coeffs_b.push_back(coeff);
      }

      /*** END GENERATE NEW COEFFICIENTS FOR IMAGE B ***/

      /*** BEGIN AUGMENTATION TO IMAGE B ***/
      auto input_b  = input_b_t.tensor<float, 4>();
      auto output_b = output_b_t->tensor<float, 4>();

      // Copy spatial coefficients B to the output Tensor on the CPU
      auto spat_transform_b = spat_transform_b_t->tensor<float, 2>();
      AugmentationLayerBase::copy_spatial_coeffs_to_tensor(coeffs_b,
                                                           out_width, out_height,
                                                           src_width, src_height,
                                                           spat_transform_b);

      float *chromatic_coeffs_b_data = NULL;

      if (gen_chromatic_transform || gen_chromatic_transform_b) {
        // Allocate a temporary tensor to hold the chromatic coefficients
        tensorflow::AllocatorAttributes pinned_allocator;
        pinned_allocator.set_on_host(true);
        pinned_allocator.set_gpu_compatible(true);
        OP_REQUIRES_OK(ctx,
                       ctx->allocate_temp(DataTypeToEnum<float>::value,
                                          TensorShape({ batch_size, 6 }),
                                          &chromatic_coeffs_b_t, pinned_allocator));

        // Copy the chromatic coefficients A to a temporary Tensor on the CPU
        auto chromatic_coeffs_b = chromatic_coeffs_b_t.tensor<float, 2>();
        AugmentationLayerBase::copy_chromatic_coeffs_to_tensor(coeffs_b, chromatic_coeffs_b);
        chromatic_coeffs_b_data = chromatic_coeffs_b.data();
      }

      // Perform augmentation either on CPU or GPU
      Augment<Device>(
        ctx,
        ctx->eigen_device<Device>(),
        batch_size,
        channels,
        src_width,
        src_height,
        src_count,
        out_width,
        out_height,
        input_b.data(),
        output_b.data(),
        spat_transform_b.data(),
        chromatic_coeffs_b_data);

      // FlowAugmentation needs the inverse
      // TODO: To avoid rewriting, can we invert when we read on the
      // FlowAugmentation side?
      AugmentationLayerBase::copy_spatial_coeffs_to_tensor(coeffs_b,
                                                           out_width, out_height,
                                                           src_width, src_height,
                                                           spat_transform_b,
                                                           true);

      /*** END AUGMENTATION TO IMAGE B ***/
    }

  private:
    std::vector<int32>crop_;

    // Params A
    std::vector<string>params_a_name_;
    std::vector<string>params_a_rand_type_;
    std::vector<bool>params_a_exp_;
    std::vector<float>params_a_mean_;
    std::vector<float>params_a_spread_;
    std::vector<float>params_a_prob_;
    std::vector<float>params_a_coeff_schedule_;

    // Params B
    std::vector<string>params_b_name_;
    std::vector<string>params_b_rand_type_;
    std::vector<bool>params_b_exp_;
    std::vector<float>params_b_mean_;
    std::vector<float>params_b_spread_;
    std::vector<float>params_b_prob_;
    std::vector<float>params_b_coeff_schedule_;
};


REGISTER_KERNEL_BUILDER(Name("DataAugmentation")
                        .Device(DEVICE_CPU)
                        .HostMemory("global_step")
                        .HostMemory("transforms_from_a")
                        .HostMemory("transforms_from_b"),
                        DataAugmentation<CPUDevice>)

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("DataAugmentation")
                        .Device(DEVICE_GPU)
                        .HostMemory("global_step")
                        .HostMemory("transforms_from_a")
                        .HostMemory("transforms_from_b"),
                        DataAugmentation<GPUDevice>)
#endif // GOOGLE_CUDA
} // namespace tensorflow
