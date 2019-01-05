import tensorflow as tf
import tensorflow.contrib.slim as slim

class NetHandler(object):
    def __init__(
            self,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weight_decay=0.0001,
            padding='SAME'):
        self.padding = padding
        self.weights_initializer = weights_initializer
        self.weight_decay = weight_decay

    def vgg16_net(self, inputs, depth_suf=''):
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'pool5'
        )
        kernel_size = 3
        num_outputs = 64
        net = {}
        current = inputs
        for i, name in enumerate(layers):
            if depth_suf == '_d' and i == 0:
                current = slim.conv2d(current, 64, [3, 3],
                                      weights_initializer=self.weights_initializer,
                                      padding=self.padding,
                                      stride=1,
                                      activation_fn=None)
                net[name] = current
                continue

            kind = name[:4]
            if kind == 'conv':
                if name[:5] == 'conv1':
                    num_outputs = 64
                elif name[:5] == 'conv2':
                    num_outputs = 128
                elif name[:5] == 'conv3':
                    num_outputs = 256
                elif name[:5] == 'conv4':
                    num_outputs = 512
                elif name[:5] == 'conv5':
                    num_outputs = 512
                _, _, _, c = current.get_shape()
                kernels = tf.get_variable(name=name + "_w" + depth_suf, shape=[kernel_size, kernel_size, c, num_outputs],
                                               initializer=self.weights_initializer,
                                               regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                                               trainable=True)
                _, _, _, bias_size = kernels.get_shape()
                bias = tf.get_variable(name=name + "_b" + depth_suf,
                                       shape=[bias_size],
                                       initializer=tf.zeros_initializer(),
                                       trainable=True)
                conv = tf.nn.conv2d(current, kernels, strides=[1, 1, 1, 1], padding="SAME")
                current = tf.nn.bias_add(conv, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current, name=name)
            elif kind == 'pool':
                current = tf.nn.max_pool(current, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            net[name] = current
        return net

    def RGBD_SW_net(self, image, depth):
        image_net = self.vgg16_net(image)
        depth_net = self.vgg16_net(depth, depth_suf='_d')
        conv_5 = image_net["relu5_3"]
        conv_4 = image_net["relu4_3"]
        conv_3 = image_net["relu3_3"]
        conv_2 = image_net["relu2_2"]
        conv_1 = image_net["relu1_2"]

        depth_5 = depth_net["relu5_3"]
        depth_4 = depth_net["relu4_3"]
        depth_3 = depth_net["relu3_3"]
        depth_2 = depth_net["relu2_2"]
        depth_1 = depth_net["relu1_2"]

        with slim.arg_scope([slim.conv2d],
                            weights_initializer=self.weights_initializer,
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            padding=self.padding,
                            stride=1,
                            activation_fn=tf.nn.relu):
            conv5 = slim.repeat(conv_5, 2, slim.conv2d, 64, [3, 3], scope='conv5')
            conv4 = slim.repeat(conv_4, 2, slim.conv2d, 64, [3, 3], scope='conv4')
            conv3 = slim.repeat(conv_3, 2, slim.conv2d, 64, [3, 3], scope='conv3')
            conv2 = slim.repeat(conv_2, 2, slim.conv2d, 64, [3, 3], scope='conv2')
            conv1 = slim.repeat(conv_1, 2, slim.conv2d, 64, [3, 3], scope='conv1')

            depth5 = slim.repeat(depth_5, 2, slim.conv2d, 64, [3, 3], scope='depth5')
            depth4 = slim.repeat(depth_4, 2, slim.conv2d, 64, [3, 3], scope='depth4')
            depth3 = slim.repeat(depth_3, 2, slim.conv2d, 64, [3, 3], scope='depth3')
            depth2 = slim.repeat(depth_2, 2, slim.conv2d, 64, [3, 3], scope='depth2')
            depth1 = slim.repeat(depth_1, 2, slim.conv2d, 64, [3, 3], scope='depth1')

            conv5_up = tf.image.resize_images(conv5, [224, 224])
            conv4_up = tf.image.resize_images(conv4, [224, 224])
            conv3_up = tf.image.resize_images(conv3, [224, 224])
            conv2_up = tf.image.resize_images(conv2, [224, 224])

            depth5_up = tf.image.resize_images(depth5, [224, 224])
            depth4_up = tf.image.resize_images(depth4, [224, 224])
            depth3_up = tf.image.resize_images(depth3, [224, 224])
            depth2_up = tf.image.resize_images(depth2, [224, 224])

            concat4_im = tf.concat([conv4_up, conv5_up], 3)
            feat4_im = slim.conv2d(concat4_im, 64, [3, 3], scope='feat4_im')
            concat3_im = tf.concat([conv3_up, feat4_im], 3)
            feat3_im = slim.conv2d(concat3_im, 64, [3, 3], scope='feat3_im')
            concat2_im = tf.concat([conv2_up, feat3_im], 3)
            feat2_im = slim.conv2d(concat2_im, 64, [3, 3], scope='feat2_im')
            concat1_im = tf.concat([conv1, feat2_im], 3)
            feat1_im = slim.conv2d(concat1_im, 64, [3, 3], scope='feat1_im')

            concat4_d = tf.concat([depth4_up, depth5_up], 3)
            feat4_d = slim.conv2d(concat4_d, 64, [3, 3], scope='feat4_d')
            concat3_d = tf.concat([depth3_up, feat4_d], 3)
            feat3_d = slim.conv2d(concat3_d, 64, [3, 3], scope='feat3_d')
            concat2_d = tf.concat([depth2_up, feat3_d], 3)
            feat2_d = slim.conv2d(concat2_d, 64, [3, 3], scope='feat2_d')
            concat1_d = tf.concat([depth1, feat2_d], 3)
            feat1_d = slim.conv2d(concat1_d, 64, [3, 3], scope='feat1_d')

            conv1_im_logits = slim.conv2d(feat1_im, 1, [1, 1], activation_fn=None, scope='conv1_im_logits')
            conv1_d_logits = slim.conv2d(feat1_d, 1, [1, 1], activation_fn=None, scope='conv1_d_logits')

            feat1 = slim.conv2d(tf.concat([feat1_im, feat1_d], 3), 64, [3, 3], scope='feat1')
            SW_map = tf.nn.sigmoid(slim.conv2d(feat1, 1, [1, 1], activation_fn=None, scope='feat1_attn'))

            conv1_fused_logits = SW_map * conv1_im_logits + (1.0 - SW_map) * conv1_d_logits

            return conv1_fused_logits

