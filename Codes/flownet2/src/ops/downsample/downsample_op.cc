#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

Status SetOutputToSizedImage(InferenceContext* c) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
  DimensionHandle batch = c->Dim(input, 0);
  DimensionHandle depth = c->Dim(input, 3);
  std::vector<int32> size_;
  c->GetAttr("size", &size_);
  DimensionHandle height = c->MakeDim(size_[0]);
  DimensionHandle width  = c->MakeDim(size_[1]);
  c->set_output(0, c->MakeShape({batch, height, width, depth}));
  return Status::OK();
}

REGISTER_OP("Downsample")
    .Input("input: float32")
    .Attr("size: list(int) >= 2")
    .Output("output: float32")
    .SetShapeFn(SetOutputToSizedImage);

}  // namespace tensorflow
