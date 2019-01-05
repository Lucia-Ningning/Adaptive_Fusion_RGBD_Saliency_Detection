import tensorflow as tf

def sigmoid_CEloss(logits, gt):
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=tf.cast(gt, tf.float32)))
    return loss

def SW_loss(im_logits, SW_map, gt):
    label=tf.cast(gt, tf.float32)
    sigmoid_im = tf.nn.sigmoid(im_logits)
    SW_gt = label*sigmoid_im + (1.0-label)*(1.0-sigmoid_im)
    cost = - SW_gt * tf.log(tf.clip_by_value(SW_map, 1e-8, 1.0)) \
           - (1.0 - SW_gt) * tf.log(tf.clip_by_value((1.0-SW_map), 1e-8, 1.0))
    return tf.reduce_mean(cost)


def edge_loss(logits, gt):
    gt = tf.cast(gt, tf.float32)
    sigmoid_p = tf.nn.sigmoid(logits)
    x_weight = tf.reshape(tf.constant([-1, 0, +1], tf.float32), [1, 3, 1, 1])
    y_weight = tf.reshape(x_weight, [3, 1, 1, 1])

    xgrad_gt = tf.nn.conv2d(gt, x_weight, [1, 1, 1, 1], 'SAME')
    ygrad_gt = tf.nn.conv2d(gt, y_weight, [1, 1, 1, 1], 'SAME')

    xgrad_sal = tf.nn.conv2d(sigmoid_p, x_weight, [1, 1, 1, 1], 'SAME')
    ygrad_sal = tf.nn.conv2d(sigmoid_p, y_weight, [1, 1, 1, 1], 'SAME')
    loss = tf.losses.mean_squared_error(xgrad_gt, xgrad_sal) + tf.losses.mean_squared_error(ygrad_gt, ygrad_sal)
    return loss


