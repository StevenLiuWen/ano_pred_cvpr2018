#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

Status SetOutput(InferenceContext *c) {
  ShapeHandle input_a, input_b, input;

  // Get shapes of both inputs and verify they are rank 4
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_a));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &input_b));

  // Verify inputs are same dimensions
  TF_RETURN_IF_ERROR(c->Merge(input_a, input_b, &input));

  // Get the attributes
  int kernel_size, max_displacement, stride_1, stride_2, pad;
  TF_RETURN_IF_ERROR(c->GetAttr("kernel_size", &kernel_size));
  TF_RETURN_IF_ERROR(c->GetAttr("max_displacement", &max_displacement));
  TF_RETURN_IF_ERROR(c->GetAttr("stride_1", &stride_1));
  TF_RETURN_IF_ERROR(c->GetAttr("stride_2", &stride_2));
  TF_RETURN_IF_ERROR(c->GetAttr("pad", &pad));

  // Get dimensions of input (already padded)
  int64 batch         = c->Value(c->Dim(input, 0));
  int64 input_height  = c->Value(c->Dim(input, 1));
  int64 input_width   = c->Value(c->Dim(input, 2));
  int64 padded_height = input_height + 2 * pad;
  int64 padded_width  = input_width + 2 * pad;

  // The size of unreachable border region on each side
  int kernel_radius = (kernel_size - 1) / 2;
  int border_size   = max_displacement + kernel_radius;

  // Calculate the output dimensions
  int64 output_height = (int64)ceil((float)(padded_height - border_size * 2) / (float)stride_1);
  int64 output_width  = (int64)ceil((float)(padded_width - border_size * 2) / (float)stride_1);

  // TODO: Verify output size >= 1

  int   neighborhood_grid_radius = max_displacement / stride_2;
  int   neighborhood_grid_width  = neighborhood_grid_radius * 2 + 1;
  int64 output_channels          = neighborhood_grid_width * neighborhood_grid_width;

  // Set output shape
  c->set_output(0, c->MakeShape({ batch, output_height, output_width, output_channels }));
  return Status::OK();
}

REGISTER_OP("Correlation")
.Input("input_a: float32")
.Input("input_b: float32")
.Attr("kernel_size: int")
.Attr("max_displacement: int")
.Attr("stride_1: int")
.Attr("stride_2: int")
.Attr("pad: int")
.Output("output: float32")
.SetShapeFn(SetOutput);

REGISTER_OP("CorrelationGrad")
.Input("gradients: float32")
.Input("input_a: float32")
.Input("input_b: float32")
.Attr("kernel_size: int")
.Attr("max_displacement: int")
.Attr("stride_1: int")
.Attr("stride_2: int")
.Attr("pad: int")
.Output("backprops_a: float32")
.Output("backprops_b: float32")
.SetShapeFn([](InferenceContext *c) {
    // Output gradients should be the same dimensions as the inputs
    ShapeHandle out;
    TF_RETURN_IF_ERROR(c->Merge(c->input(1), c->input(2), &out));
    c->set_output(0, out);
    c->set_output(1, out);
    return Status::OK();
  });
} // namespace tensorflow
