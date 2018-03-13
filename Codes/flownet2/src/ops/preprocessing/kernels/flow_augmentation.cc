#define EIGEN_USE_THREADS

#include "flow_augmentation.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice        GPUDevice;

inline int clamp(int f, int a, int b) {
  return std::max(a, std::min(f, b));
}

template<>
void FillFlowAugmentation(const CPUDevice& device,
                          typename TTypes<float, 4>::Tensor output,
                          typename TTypes<float, 4>::ConstTensor flows,
                          typename TTypes<float, 2>::ConstTensor transforms_from_a,
                          typename TTypes<float, 2>::ConstTensor transforms_from_b) {
  const int batch_size      = output.dimension(0);
  const int out_height      = output.dimension(1);
  const int out_width       = output.dimension(2);
  const int src_height      = flows.dimension(1);
  const int src_width       = flows.dimension(2);
  const int src_total_count = flows.dimension(0) * flows.dimension(1) *
                              flows.dimension(2) * flows.dimension(3);
  float *output_ptr     = output.data();
  const float *flow_ptr = flows.data();

  for (int n = 0; n < batch_size; n++) {
    const float *transMatA = transforms_from_a.data() + n * 6;
    const float *transMatB = transforms_from_b.data() + n * 6;

    for (int y = 0; y < out_height; y++) {
      int outputIdxOffset = (n * out_height + y) * out_width;

      for (int x = 0; x < out_width; x++) {
        // Apply transformation matrix applied to first image
        const float xpos1 = x * transMatA[0] + y * transMatA[1] + transMatA[2];
        const float ypos1 = x * transMatA[3] + y * transMatA[4] + transMatA[5];

        const int srcXIdx =
          ((n * src_height + (int)(ypos1 + 0.5)) * src_width + (int)(xpos1 + 0.5)) * 2 + 0;
        const int srcYIdx = srcXIdx + 1;

        const float xpos2 = xpos1 + flow_ptr[clamp(srcXIdx, 0, src_total_count - 1)];
        const float ypos2 = ypos1 + flow_ptr[clamp(srcYIdx, 0, src_total_count - 1)];

        // Apply inverse of the transformation matrix applied to second image
        const float xpos3 = xpos2 * transMatB[0] + ypos2 * transMatB[1] + transMatB[2];
        const float ypos3 = xpos2 * transMatB[3] + ypos2 * transMatB[4] + transMatB[5];

        output_ptr[(outputIdxOffset + x) * 2 + 0] = xpos3 - (float)x;
        output_ptr[(outputIdxOffset + x) * 2 + 1] = ypos3 - (float)y;
      }
    }
  }
}

template<typename Device>
class FlowAugmentation : public OpKernel {
  public:
    explicit FlowAugmentation(OpKernelConstruction *ctx) : OpKernel(ctx) {
      // Get the crop [height, width] tensor and verify its dimensions
      OP_REQUIRES_OK(ctx, ctx->GetAttr("crop", &crop_));
      OP_REQUIRES(ctx, crop_.size() == 2,
                  errors::InvalidArgument("crop must be 2 dimensions"));
    }

    void Compute(OpKernelContext *ctx) override {
      // Get the input images and transforms and verify their dimensions
      const Tensor& flows_t             = ctx->input(0);
      const Tensor& transforms_from_a_t = ctx->input(1);
      const Tensor& transforms_from_b_t = ctx->input(2);

      OP_REQUIRES(ctx, flows_t.dims() == 4,
                  errors::InvalidArgument("Input images must have rank 4"));
      OP_REQUIRES(ctx,
                  (TensorShapeUtils::IsMatrix(transforms_from_a_t.shape()) &&
                   transforms_from_a_t.dim_size(0) ==
                   flows_t.dim_size(0) &&
                   transforms_from_a_t.dim_size(1) == 6),
                  errors::InvalidArgument(
                    "Input transforms_from_a should be num_images x 6"));
      OP_REQUIRES(ctx,
                  (TensorShapeUtils::IsMatrix(transforms_from_b_t.shape()) &&
                   transforms_from_b_t.dim_size(0) ==
                   flows_t.dim_size(0) &&
                   transforms_from_b_t.dim_size(1) == 6),
                  errors::InvalidArgument(
                    "Input transforms_from_b should be num_images x 6"));

      // Allocate the memory for the output
      Tensor *output_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(
                       0,
                       TensorShape({ flows_t.dim_size(0), crop_[0], crop_[1],
                                     flows_t.dim_size(3) }), &output_t));

      // Perform flow augmentation
      auto flows             = flows_t.tensor<float, 4>();
      auto transforms_from_a = transforms_from_a_t.tensor<float, 2>();
      auto transforms_from_b = transforms_from_b_t.tensor<float, 2>();
      auto output            = output_t->tensor<float, 4>();

      FillFlowAugmentation(ctx->eigen_device<Device>(),
                           output,
                           flows,
                           transforms_from_a,
                           transforms_from_b);
    }

  private:
    std::vector<int32>crop_;
};

REGISTER_KERNEL_BUILDER(Name("FlowAugmentation")
                        .Device(DEVICE_CPU),
                        FlowAugmentation<CPUDevice>)

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("FlowAugmentation")
                        .Device(DEVICE_GPU),
                        FlowAugmentation<GPUDevice>)
#endif // GOOGLE_CUDA
} // end namespace tensorflow
