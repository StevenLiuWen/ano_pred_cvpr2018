from ..net import Net, Mode
from ..flownet_css.flownet_css import FlowNetCSS
from ..flownet_sd.flownet_sd import FlowNetSD
from ..flow_warp import flow_warp
from ..utils import LeakyReLU, average_endpoint_error, pad, antipad
from ..downsample import downsample
import tensorflow as tf
slim = tf.contrib.slim


class FlowNet2(Net):

    def __init__(self, mode=Mode.TRAIN, debug=False):
        self.net_css = FlowNetCSS(mode, debug)
        self.net_sd = FlowNetSD(mode, debug)
        super(FlowNet2, self).__init__(mode=mode, debug=debug)

    def model(self, inputs, training_schedule, trainable=True):
        _, height, width, _ = inputs['input_a'].shape.as_list()
        with tf.variable_scope('FlowNet2'):
            # Forward pass through FlowNetCSS and FlowNetSD with weights frozen
            net_css_predictions = self.net_css.model(inputs, training_schedule, trainable=True)
            net_sd_predictions = self.net_sd.model(inputs, training_schedule, trainable=True)

            def ChannelNorm(tensor):
                sq = tf.square(tensor)
                r_sum = tf.reduce_sum(sq, keep_dims=True, axis=3)
                return tf.sqrt(r_sum)

            sd_flow_norm = ChannelNorm(net_sd_predictions['flow'])
            css_flow_norm = ChannelNorm(net_css_predictions['flow'])

            flow_warp_sd = flow_warp(inputs['input_b'], net_sd_predictions['flow'])
            img_diff_sd = inputs['input_a'] - flow_warp_sd
            img_diff_sd_norm = ChannelNorm(img_diff_sd)

            flow_warp_css = flow_warp(inputs['input_b'], net_css_predictions['flow'])
            img_diff_css = inputs['input_a'] - flow_warp_css
            img_diff_css_norm = ChannelNorm(img_diff_css)

            input_to_fusion = tf.concat([inputs['input_a'],
                                         net_sd_predictions['flow'],
                                         net_css_predictions['flow'],
                                         sd_flow_norm,
                                         css_flow_norm,
                                         img_diff_sd_norm,
                                         img_diff_css_norm], axis=3)

            # Fusion Network
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                # Only backprop this network if trainable
                                trainable=trainable,
                                # He (aka MSRA) weight initialization
                                weights_initializer=slim.variance_scaling_initializer(),
                                activation_fn=LeakyReLU,
                                # We will do our own padding to match the original Caffe code
                                padding='VALID'):

                weights_regularizer = slim.l2_regularizer(training_schedule['weight_decay'])
                with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
                    fuse_conv0 = slim.conv2d(pad(input_to_fusion), 64, 3, scope='fuse_conv0')
                    fuse_conv1 = slim.conv2d(pad(fuse_conv0), 64, 3, stride=2, scope='fuse_conv1')
                    fuse_conv1_1 = slim.conv2d(pad(fuse_conv1), 128, 3, scope='fuse_conv1_1')
                    fuse_conv2 = slim.conv2d(pad(fuse_conv1_1), 128, 3,
                                             stride=2, scope='fuse_conv2')
                    fuse_conv2_1 = slim.conv2d(pad(fuse_conv2), 128, 3, scope='fuse_conv2_1')

                    predict_flow2 = slim.conv2d(pad(fuse_conv2_1), 2, 3,
                                                scope='predict_flow2',
                                                activation_fn=None)
                    fuse_deconv1 = antipad(slim.conv2d_transpose(fuse_conv2_1, 32, 4,
                                                                 stride=2,
                                                                 scope='fuse_deconv1'))
                    fuse_upsample_flow2to1 = antipad(slim.conv2d_transpose(predict_flow2, 2, 4,
                                                                           stride=2,
                                                                           scope='fuse_upsample_flow2to1',
                                                                           activation_fn=None))
                    concat1 = tf.concat([fuse_conv1_1, fuse_deconv1,
                                         fuse_upsample_flow2to1], axis=3)
                    fuse_interconv1 = slim.conv2d(pad(concat1), 32, 3,
                                                  activation_fn=None, scope='fuse_interconv1')

                    predict_flow1 = slim.conv2d(pad(fuse_interconv1), 2, 3,
                                                scope='predict_flow1',
                                                activation_fn=None)
                    fuse_deconv0 = antipad(slim.conv2d_transpose(concat1, 16, 4,
                                                                 stride=2,
                                                                 scope='fuse_deconv0'))
                    fuse_upsample_flow1to0 = antipad(slim.conv2d_transpose(predict_flow1, 2, 4,
                                                                           stride=2,
                                                                           scope='fuse_upsample_flow1to0',
                                                                           activation_fn=None))
                    concat0 = tf.concat([fuse_conv0, fuse_deconv0, fuse_upsample_flow1to0], axis=3)
                    fuse_interconv0 = slim.conv2d(pad(concat0), 16, 3,
                                                  activation_fn=None, scope='fuse_interconv0')

                    predict_flow0 = slim.conv2d(pad(fuse_interconv0), 2,
                                                3, activation_fn=None, scope='predict_flow0')

                    flow = tf.image.resize_bilinear(
                        predict_flow0, tf.stack([height, width]), align_corners=True)
                    print(predict_flow0)
                    print(flow)
                    return {
                        'predict_flow0': predict_flow0,
                        'flow': flow,
                    }

    def loss(self, flow, predictions):
        # L2 loss between predict_flow0, true flow (weighted w/ 0.005)
        predict_flow0 = predictions['predict_flow0']
        size = [predict_flow0.shape[1], predict_flow0.shape[2]]
        downsampled_flow0 = downsample(flow, size)
        loss = average_endpoint_error(downsampled_flow0, predict_flow0)
        tf.losses.add_loss(loss)

        # Return the 'total' loss: loss fns + regularization terms defined in the model
        return tf.losses.get_total_loss()
