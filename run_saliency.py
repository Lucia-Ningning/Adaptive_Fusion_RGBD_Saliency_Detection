from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
from net import *

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "./data/", "path to dataset")
tf.flags.DEFINE_string("ckpt_file", "./model/AF-Net", "checkpoint file")
tf.flags.DEFINE_string("save_dir", "./results", "path to prediction directory")

IMAGE_SIZE = 224

def _transform(filename, _channels=True):
    image = misc.imread(filename)
    if _channels and len(image.shape) < 3:
        image = np.array([image for _ in range(3)])

    resize_image = misc.imresize(image, [IMAGE_SIZE, IMAGE_SIZE], interp='nearest')
    return resize_image

def main(argv=None):
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    depth = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="input_depth")
    processed_image = image - [123.68, 116.779, 103.939]

    net_handler = NetHandler()
    logits = net_handler.RGBD_SW_net(processed_image, depth)
    pred_annotation = tf.sigmoid(logits)

    files = os.listdir(os.path.join(FLAGS.data_dir + '/RGB/'))
    test_num = len(files)
    test_RGB = np.array([_transform(os.path.join(FLAGS.data_dir + '/RGB/' + filename), _channels=True)
                         for filename in files])
    test_depth = np.array(
        [np.expand_dims(_transform(os.path.join(FLAGS.data_dir + '/depth/' + filename), _channels=False), axis=3)
         for filename in files])


    sess = tf.Session()
    print('Reading params from {}'.format(FLAGS.ckpt_file))
    saver = tf.train.Saver(None)
    saver.restore(sess, FLAGS.ckpt_file)

    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)


    for k in range(test_num):
        test_prediction = sess.run(pred_annotation, feed_dict={image: test_RGB[k:k + 1],
                                                    depth: test_depth[k:k+1]})
        test_origin_RGB = misc.imread(os.path.join(FLAGS.data_dir + '/RGB/' + files[k].split('.')[0] +'.jpg'))
        image_shape = test_origin_RGB.shape

        test_pred = misc.imresize(test_prediction[0, :, :, 0], image_shape, interp='bilinear')
        misc.imsave('{}/{}'.format(FLAGS.save_dir, files[k].split('.')[0] + '.png'), test_pred.astype(np.uint8))

    print("Save results in to %s" % (FLAGS.save_dir))

if __name__ == "__main__":
    tf.app.run()
