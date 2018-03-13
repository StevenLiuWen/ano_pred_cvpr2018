from ..net import Net, Mode
from ..flownet_cs.flownet_cs import FlowNetCS
from ..flownet_s.flownet_s import FlowNetS
from ..flow_warp import flow_warp
import tensorflow as tf


class FlowNetCSS(Net):

    def __init__(self, mode=Mode.TRAIN, debug=False):
        self.net_cs = FlowNetCS(mode, debug)
        self.net_s = FlowNetS(mode, debug)
        super(FlowNetCSS, self).__init__(mode=mode, debug=debug)

    def model(self, inputs, training_schedule, trainable=True):
        with tf.variable_scope('FlowNetCSS'):
            # Forward pass through FlowNetCS with weights frozen
            net_cs_predictions = self.net_cs.model(inputs, training_schedule, trainable=True)

            # Perform flow warping (to move image B closer to image A based on flow prediction)
            warped = flow_warp(inputs['input_b'], net_cs_predictions['flow'])

            # Compute brightness error: sqrt(sum (input_a - warped)^2 over channels)
            brightness_error = inputs['input_a'] - warped
            brightness_error = tf.square(brightness_error)
            brightness_error = tf.reduce_sum(brightness_error, keep_dims=True, axis=3)
            brightness_error = tf.sqrt(brightness_error)

            # Gather all inputs to FlowNetS
            inputs_to_s = {
                'input_a': inputs['input_a'],
                'input_b': inputs['input_b'],
                'warped': warped,
                'flow': net_cs_predictions['flow'] * 0.05,
                'brightness_error': brightness_error,
            }

            return self.net_s.model(inputs_to_s, training_schedule, trainable=trainable)

    def loss(self, flow, predictions):
        return self.net_s.loss(flow, predictions)
