#ifndef FLOWNET_DATA_AUGMENTATION_H_
#define FLOWNET_DATA_AUGMENTATION_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
template<class Device>
void Augment(OpKernelContext *context,
             const Device   & d,
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
             float           *chromatic_coeffs);
} // namespace tensorflow
#endif // FLOWNET_DATA_AUGMENTATION_H_
