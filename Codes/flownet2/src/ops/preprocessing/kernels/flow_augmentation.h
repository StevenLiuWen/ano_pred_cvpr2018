#ifndef FLOWNET_FLOW_AUG_H_
#define FLOWNET_FLOW_AUG_H_

// See docs in ../ops/image_ops.cc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
template<class Device>
void FillFlowAugmentation(const Device& device,
                          typename TTypes<float, 4>::Tensor output,
                          typename TTypes<float, 4>::ConstTensor flows,
                          typename TTypes<float, 2>::ConstTensor transforms_from_a,
                          typename TTypes<float, 2>::ConstTensor transforms_from_b);
} // end namespace tensorflow

#endif  // FLOWNET_FLOW_AUG_H_
