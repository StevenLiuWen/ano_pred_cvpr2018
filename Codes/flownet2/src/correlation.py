import tensorflow as tf

_correlation_ops = tf.load_op_library(
    tf.resource_loader.get_path_to_datafile("./ops/build/correlation.so"))


def correlation(input_a, input_b, kernel_size, max_displacement, stride_1, stride_2, padding):
    return _correlation_ops.correlation(input_a,
                                        input_b,
                                        kernel_size,
                                        max_displacement,
                                        stride_1,
                                        stride_2,
                                        padding)


@tf.RegisterGradient("Correlation")
def _correlation_grad(corr_op, gradients):
    kernel_size = corr_op.get_attr("kernel_size")
    max_displacement = corr_op.get_attr("max_displacement")
    stride_1 = corr_op.get_attr("stride_1")
    stride_2 = corr_op.get_attr("stride_2")
    pad = corr_op.get_attr("pad")

    corr_grads = _correlation_ops.correlation_grad(gradients,
                                                   corr_op.inputs[0],
                                                   corr_op.inputs[1],
                                                   kernel_size,
                                                   max_displacement,
                                                   stride_1,
                                                   stride_2,
                                                   pad)

    # Return the gradients with respect to input_a and input_b
    return corr_grads.backprops_a, corr_grads.backprops_b
