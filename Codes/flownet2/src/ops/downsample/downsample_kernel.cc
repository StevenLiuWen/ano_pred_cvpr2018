#define EIGEN_USE_THREADS

#include "downsample_kernel.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device>
class DownsampleKernel : public OpKernel {
 public:
  explicit DownsampleKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // Get the size [height, width] tensor and verify its dimensions
    OP_REQUIRES_OK(ctx, ctx->GetAttr("size", &size_));
    OP_REQUIRES(ctx, size_.size() == 2, errors::InvalidArgument("size must be 2 dimensions"));
  }

  void Compute(OpKernelContext* ctx) override {
    // Get the input images and transforms and verify their dimensions
    const Tensor& input_t = ctx->input(0);
    OP_REQUIRES(ctx, input_t.dims() == 4,
                errors::InvalidArgument("Input images must have rank 4"));

    // Allocate the memory for the output
    Tensor* output_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        0, TensorShape({input_t.dim_size(0), size_[0], size_[1], input_t.dim_size(3)}), &output_t));

    // Perform flow augmentation
    auto input = input_t.tensor<float, 4>();
    auto output = output_t->tensor<float, 4>();

    Downsample(ctx->eigen_gpu_device(), input, output);
  }

  private:
    std::vector<int32> size_;
};

REGISTER_KERNEL_BUILDER(Name("Downsample")
                          .Device(DEVICE_GPU),
                      DownsampleKernel<GPUDevice>)
}  // end namespace tensorflow
