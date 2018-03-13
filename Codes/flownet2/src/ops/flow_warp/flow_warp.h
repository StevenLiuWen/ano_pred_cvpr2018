#ifndef FLOWNET_FLOWWARP_H_
#define FLOWNET_FLOWWARP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

#define FW_THREADS 32
#define FW_TILE_X FW_THREADS
#define FW_TILE_C FW_THREADS

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

void FlowWarp(const GPUDevice& device,
              typename TTypes<float, 4>::ConstTensor input,
              typename TTypes<float, 4>::ConstTensor flow,
              typename TTypes<float, 4>::Tensor output);

void FlowWarpGrad(const GPUDevice& device,
                  typename TTypes<float, 4>::ConstTensor image,
                  typename TTypes<float, 4>::ConstTensor flow,
                  typename TTypes<float, 4>::ConstTensor gradient,
                  typename TTypes<float, 4>::Tensor image_grad,
                  typename TTypes<float, 4>::Tensor flow_grad);
} // end namespace tensorflow

#endif  // FLOWNET_FLOWWARP_H_
