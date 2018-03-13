#define EIGEN_USE_THREADS

#include "flow_warp.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

template<typename Device>
class FlowWarpGradKernel : public OpKernel {
  public:
    explicit FlowWarpGradKernel(OpKernelConstruction *ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext *ctx) override {
      // Get the input image and flow and verify dimensions
      const Tensor& image_t = ctx->input(0);
      const Tensor& flow_t  = ctx->input(1);
      const Tensor& grad_t  = ctx->input(2);

      OP_REQUIRES(ctx, image_t.dims() == 4,
                  errors::InvalidArgument("Input image must have rank 4"));
      OP_REQUIRES(ctx, flow_t.dims() == 4,
                  errors::InvalidArgument("Input flow must have rank 4"));
      OP_REQUIRES(ctx,
                  image_t.dim_size(0) == flow_t.dim_size(0) && image_t.dim_size(
                    1) == flow_t.dim_size(1) && image_t.dim_size(2) == flow_t.dim_size(2),
                  errors::InvalidArgument(
                    "Input image and flow must have same N x H x W dimensions"));

      // Allocate the memory for the output
      Tensor *image_grad_t;
      Tensor *flow_grad_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, image_t.shape(), &image_grad_t));
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, flow_t.shape(), &flow_grad_t));

      auto image      = image_t.tensor<float, 4>();
      auto flow       = flow_t.tensor<float, 4>();
      auto gradient   = grad_t.tensor<float, 4>();
      auto image_grad = image_grad_t->tensor<float, 4>();
      auto flow_grad  = flow_grad_t->tensor<float, 4>();

      FlowWarpGrad(ctx->eigen_gpu_device(),
                   image,
                   flow,
                   gradient,
                   image_grad,
                   flow_grad);
    }
};

REGISTER_KERNEL_BUILDER(Name("FlowWarpGrad")
                        .Device(DEVICE_GPU),
                        FlowWarpGradKernel<GPUDevice>)
} // end namespace tensorflow
