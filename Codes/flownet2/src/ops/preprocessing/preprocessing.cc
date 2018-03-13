#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

Status SetOutputToSizedImage(InferenceContext *c) {
  ShapeHandle input;

  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
  DimensionHandle batch    = c->Dim(input, 0);
  DimensionHandle    depth = c->Dim(input, 3);
  std::vector<int32> crop_;
  c->GetAttr("crop", &crop_);
  DimensionHandle height = c->MakeDim(crop_[0]);
  DimensionHandle width  = c->MakeDim(crop_[1]);
  c->set_output(0, c->MakeShape({ batch, height, width, depth }));
  return Status::OK();
}

REGISTER_OP("DataAugmentation")
.Input("image_a: float32")
.Input("image_b: float32")
.Input("global_step: int64")
.Attr("crop: list(int) >= 2")
.Attr("params_a_name: list(string)")
.Attr("params_a_rand_type: list(string)")
.Attr("params_a_exp: list(bool)")
.Attr("params_a_mean: list(float)")
.Attr("params_a_spread: list(float)")
.Attr("params_a_prob: list(float)")
.Attr("params_a_coeff_schedule: list(float)")
.Attr("params_b_name: list(string)")
.Attr("params_b_rand_type: list(string)")
.Attr("params_b_exp: list(bool)")
.Attr("params_b_mean: list(float)")
.Attr("params_b_spread: list(float)")
.Attr("params_b_prob: list(float)")
.Attr("params_b_coeff_schedule: list(float)")
.Output("aug_image_a: float32")
.Output("aug_image_b: float32")
.Output("transforms_from_a: float32")
.Output("transforms_from_b: float32")
.SetShapeFn([](InferenceContext *c) {
    // Verify input A and input B both have 4 dimensions
    ShapeHandle input_shape_a, input_shape_b;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape_a));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &input_shape_b));

    // TODO: Verify params vectors all have the same length

    // TODO: Move this out of here and into Compute
    // Verify input A and input B are the same shape
    DimensionHandle batch_size, unused;
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(input_shape_a, 0),
                                    c->Value(c->Dim(input_shape_b, 0)),
                                    &batch_size));
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(input_shape_a, 1),
                                    c->Value(c->Dim(input_shape_b, 1)), &unused));
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(input_shape_a, 2),
                                    c->Value(c->Dim(input_shape_b, 2)), &unused));
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(input_shape_a, 3),
                                    c->Value(c->Dim(input_shape_b, 3)), &unused));

    // Get cropping dimensions
    std::vector<int32>crop_;
    TF_RETURN_IF_ERROR(c->GetAttr("crop", &crop_));

    // Reshape input shape to cropped shape
    TF_RETURN_IF_ERROR(c->ReplaceDim(input_shape_a, 1, c->MakeDim(crop_[0]),
                                     &input_shape_a));
    TF_RETURN_IF_ERROR(c->ReplaceDim(input_shape_a, 2, c->MakeDim(crop_[1]),
                                     &input_shape_a));

    // Set output images shapes
    c->set_output(0, input_shape_a);
    c->set_output(1, input_shape_a);

    // Set output spatial transforms shapes
    c->set_output(2, c->MakeShape({ batch_size, 6 }));
    c->set_output(3, c->MakeShape({ batch_size, 6 }));

    return Status::OK();
  });

REGISTER_OP("FlowAugmentation")
.Input("flows: float32")
.Input("transforms_from_a: float32")
.Input("transforms_from_b: float32")
.Attr("crop: list(int) >= 2")
.Output("transformed_flows: float32")
.SetShapeFn(SetOutputToSizedImage);
} // namespace tensorflow
