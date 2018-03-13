#ifndef FLOWNET_CORRELATION_H_
#define FLOWNET_CORRELATION_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

void Correlation(const GPUDevice& device,
                 const float     *input_a,
                 const float     *input_b,
                 const int        batch_size,
                 const int        out_height,
                 const int        out_width,
                 const int        out_channels,
                 const int        out_count,
                 const int        in_height_padded,
                 const int        in_width_padded,
                 const int        in_channels,
                 int              max_displacement,
                 int              neighborhood_grid_radius,
                 int              neighborhood_grid_width,
                 int              kernel_radius,
                 int              kernel_size,
                 int              stride_1,
                 int              stride_2,
                 float           *output);


void CorrelationGradA(const GPUDevice& device,
                      const int        batch_size,
                      const int        out_width,
                      const int        out_height,
                      const int        out_channels,
                      const int        max_displacement,
                      const int        neighborhood_grid_radius,
                      const int        neighborhood_grid_width,
                      const int        kernel_radius,
                      const int        stride_1,
                      const int        stride_2,
                      const int        in_width,
                      const int        in_height,
                      const int        padded_in_width,
                      const int        padded_in_height,
                      const int        in_channels,
                      const int        in_count_per_sample,
                      const int        pad,
                      const float     *input_b,
                      const float     *gradient,
                      float           *output_a_gradient);

void CorrelationGradB(const GPUDevice& device,
                      const int        batch_size,
                      const int        out_width,
                      const int        out_height,
                      const int        out_channels,
                      const int        max_displacement,
                      const int        neighborhood_grid_radius,
                      const int        neighborhood_grid_width,
                      const int        kernel_radius,
                      const int        stride_1,
                      const int        stride_2,
                      const int        in_width,
                      const int        in_height,
                      const int        padded_in_width,
                      const int        padded_in_height,
                      const int        in_channels,
                      const int        in_count_per_sample,
                      const int        pad,
                      const float     *input_a,
                      const float     *gradient,
                      float           *output_b_gradient);
} // end namespace tensorflow

#endif  // FLOWNET_CORRELATION_H_
