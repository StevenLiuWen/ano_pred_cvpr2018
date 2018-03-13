import tensorflow as tf

_downsample = tf.load_op_library(
    tf.resource_loader.get_path_to_datafile("./ops/build/downsample.so"))


def downsample(tensor, size):
    return _downsample.downsample(tensor, size)
