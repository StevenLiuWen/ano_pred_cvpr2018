#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("FlowWarp")
.Input("image: float32")
.Input("flow: float32")
.Output("output: float32")
.SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

REGISTER_OP("FlowWarpGrad")
.Input("image: float32")
.Input("flow: float32")
.Input("gradient: float32")
.Output("image_grad: float32")
.Output("flow_grad: float32")
.SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    c->set_output(1, c->input(1));
    return Status::OK();
  });
} // namespace tensorflow
