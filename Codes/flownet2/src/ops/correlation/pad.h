#ifndef FLOWNET_PAD_H_
#define FLOWNET_PAD_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

void Pad(const GPUDevice& device,
         const float     *input,
         int              batch_size,
         int              input_height,
         int              input_width,
         int              input_channels,
         int              output_height,
         int              output_width,
         float           *output);
} // end namespace tensorflow

#endif // ifndef FLOWNET_PAD_H_
