import tensorflow as tf

_flow_warp_ops = tf.load_op_library(
    tf.resource_loader.get_path_to_datafile("./ops/build/flow_warp.so"))


def flow_warp(image, flow):
    return _flow_warp_ops.flow_warp(image, flow)


@tf.RegisterGradient("FlowWarp")
def _flow_warp_grad(flow_warp_op, gradients):
    return _flow_warp_ops.flow_warp_grad(flow_warp_op.inputs[0],
                                         flow_warp_op.inputs[1],
                                         gradients)
