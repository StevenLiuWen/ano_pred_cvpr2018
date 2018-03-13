#define EIGEN_USE_THREADS

#include "flow_warp.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

template<typename Device>
class FlowWarpKernel : public OpKernel {
  public:
    explicit FlowWarpKernel(OpKernelConstruction *ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext *ctx) override {
      // Get the input image and flow and verify dimensions
      const Tensor& input_t = ctx->input(0);
      const Tensor& flow_t  = ctx->input(1);

      OP_REQUIRES(ctx, input_t.dims() == 4,
                  errors::InvalidArgument("Input image must have rank 4"));
      OP_REQUIRES(ctx, flow_t.dims() == 4,
                  errors::InvalidArgument("Input flow must have rank 4"));
      OP_REQUIRES(ctx,
                  input_t.dim_size(0) == flow_t.dim_size(0) && input_t.dim_size(
                    1) == flow_t.dim_size(1) && input_t.dim_size(2) == flow_t.dim_size(2),
                  errors::InvalidArgument(
                    "Input image and flow must have same N x H x W dimensions"));

      // Allocate the memory for the output
      Tensor *output_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_t.shape(), &output_t));

      // Perform flow augmentation
      auto input  = input_t.tensor<float, 4>();
      auto flow   = flow_t.tensor<float, 4>();
      auto output = output_t->tensor<float, 4>();

      FlowWarp(ctx->eigen_gpu_device(), input, flow, output);
    }
};

REGISTER_KERNEL_BUILDER(Name("FlowWarp")
                        .Device(DEVICE_GPU),
                        FlowWarpKernel<GPUDevice>)
} // end namespace tensorflow
