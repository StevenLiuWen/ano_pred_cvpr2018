#define EIGEN_USE_THREADS

#include <utility>

#include "correlation_kernel.h"
#include "pad.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
template<typename Device>
class CorrelationKernel : public OpKernel {
  public:
    explicit CorrelationKernel(OpKernelConstruction *ctx) : OpKernel(ctx) {
      // Get the attributes
      OP_REQUIRES_OK(ctx, ctx->GetAttr("kernel_size", &kernel_size));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("max_displacement", &max_displacement));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("stride_1", &stride_1));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("stride_2", &stride_2));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("pad", &pad));

      OP_REQUIRES(ctx, kernel_size % 2 != 0, errors::InvalidArgument("kernel_size must be odd"));
    }

    void Compute(OpKernelContext *ctx) override {
      // Get the input images and transforms and verify their dimensions
      const Tensor& input_a_t = ctx->input(0);
      const Tensor& input_b_t = ctx->input(1);

      OP_REQUIRES(ctx, input_a_t.dims() == 4, errors::InvalidArgument("input_a must have rank 4"));
      OP_REQUIRES(ctx, input_b_t.dims() == 4, errors::InvalidArgument("input_b must have rank 4"));

      // Get dimensions of input (already padded)
      int batch_size     = input_a_t.dim_size(0);
      int input_height   = input_a_t.dim_size(1);
      int input_width    = input_a_t.dim_size(2);
      int input_channels = input_a_t.dim_size(3);
      int padded_height  = input_height + 2 * pad;
      int padded_width   = input_width + 2 * pad;

      // The size of unreachable border region on each side
      int kernel_radius = (kernel_size - 1) / 2;
      int border_size   = max_displacement + kernel_radius;

      // Calculate the output dimensions
      int output_height = ceil((float)(padded_height - border_size * 2) / (float)stride_1);
      int output_width  = ceil((float)(padded_width - border_size * 2) / (float)stride_1);

      OP_REQUIRES(ctx, output_height >= 1,
                  errors::InvalidArgument("Neighborhood and kernel don't fit in input height."));
      OP_REQUIRES(ctx, output_width >= 1,
                  errors::InvalidArgument("Neighborhood and kernel don't fit in input width."));

      int neighborhood_grid_radius = max_displacement / stride_2;
      int neighborhood_grid_width  = neighborhood_grid_radius * 2 + 1;
      int output_channels          = neighborhood_grid_width * neighborhood_grid_width;

      // Allocate the memory for the output
      Tensor *output_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(
                       0,
                       TensorShape({ batch_size, output_height, output_width, output_channels }),
                       &output_t));

      // Get the tensors
      auto input_a = input_a_t.tensor<float, 4>();
      auto input_b = input_b_t.tensor<float, 4>();
      auto output  = output_t->tensor<float, 4>();

      // Create temporary tensors for padded inputs
      Tensor padded_input_a_t, padded_input_b_t;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DataTypeToEnum<float>::value,
                                        TensorShape({ batch_size, padded_height, padded_width, input_channels }),
                                        &padded_input_a_t));
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DataTypeToEnum<float>::value,
                                        TensorShape({ batch_size, padded_height, padded_width, input_channels }),
                                        &padded_input_b_t));
      auto padded_input_a = padded_input_a_t.tensor<float, 4>();
      auto padded_input_b = padded_input_b_t.tensor<float, 4>();

      // Pad the inputs
      Pad(ctx->eigen_device<Device>(),
          input_a.data(),
          batch_size,
          input_height,
          input_width,
          input_channels,
          padded_height,
          padded_width,
          padded_input_a.data());
      Pad(ctx->eigen_device<Device>(),
          input_b.data(),
          batch_size,
          input_height,
          input_width,
          input_channels,
          padded_height,
          padded_width,
          padded_input_b.data());

      // Perform cross correlation
      Correlation(ctx->eigen_device<Device>(),
                  padded_input_a.data(),
                  padded_input_b.data(),
                  batch_size,
                  output_height,
                  output_width,
                  output_channels,
                  output_height * output_width * output_channels,
                  padded_height,
                  padded_width,
                  input_channels,
                  max_displacement,
                  neighborhood_grid_radius,
                  neighborhood_grid_width,
                  kernel_radius,
                  kernel_size,
                  stride_1,
                  stride_2,
                  output.data());
    }

  private:
    int kernel_size;
    int max_displacement;
    int stride_1;
    int stride_2;
    int pad;
};

REGISTER_KERNEL_BUILDER(Name("Correlation")
                        .Device(DEVICE_GPU),
                        CorrelationKernel<GPUDevice>)
} // end namespace tensorflow
