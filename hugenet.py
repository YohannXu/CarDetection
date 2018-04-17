import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import print_activations, cacl_iou
from car_detection_input import car_input

tf.app.flags.DEFINE_integer('lambda_noobj', 1, 'the coefficient of no object')
tf.app.flags.DEFINE_integer('lambda_coord', 5, 'the coefficient of coordinate')
tf.app.flags.DEFINE_integer('num_epoches',100, 'The number of epoches.')
FLAGS = tf.app.flags.FLAGS

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

basename = ' '

def restore_vars():
    basename = os.path.dirname(__file__)
    d = {}
    with tf.Session(config=config) as sess:
        saver=tf.train.import_meta_graph(os.path.join(basename,
'checkpoint/model1.ckpt.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(basename, 'checkpoint/')))
        all_vars = tf.trainable_variables()
        for var in all_vars:
            d[var.name] = sess.run(var)
    return d
class hugenet():
    def __init__(self, variables):
        self.pretrain_variables = variables
        self.lambda_noobj = FLAGS.lambda_noobj
        self.lambda_coord = FLAGS.lambda_coord
        self.label = tf.placeholder(tf.float32, shape=[None, 2, 2, 7])
        self.input = tf.placeholder(tf.float32, [None, 100, 100, 3])
    def inference(self):
        with tf.variable_scope('conv1') as scope:
            kernel = tf.get_variable(name='weight',
                                     shape=[3, 3, 3, 32],
                                     dtype=tf.float32, initializer=tf.constant_initializer(self.pretrain_variables['layer1_conv/weights:0']))
            biases = tf.get_variable(name='bias',
                                     shape=[32],
                                     dtype=tf.float32, initializer=tf.constant_initializer(self.pretrain_variables['layer1_conv/biases:0']))
            net = tf.nn.conv2d(self.input, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(net, bias=biases)
            net = tf.nn.relu(net)
            print_activations(net)
        with tf.variable_scope('pool1') as scope:
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            print_activations(net)
        with tf.variable_scope('conv2') as scope:
            kernel = tf.get_variable(name='weight',
                                     shape=[3, 3, 32, 64],
                                     dtype=tf.float32, initializer=tf.constant_initializer(self.pretrain_variables['layer2_conv/weights:0']))
            biases = tf.get_variable(name='bias',
                                     shape=[64],
                                     dtype=tf.float32, initializer=tf.constant_initializer(self.pretrain_variables['layer2_conv/biases:0']))
            net = tf.nn.conv2d(net, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(net, bias=biases)
            net = tf.nn.relu(net)
            print_activations(net)
        with tf.variable_scope('pool2') as scope:
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            print_activations(net)
        with tf.variable_scope('conv3') as scope:
            kernel = tf.get_variable(name='weight',
                                     shape=[3, 3, 64, 128],
                                     dtype=tf.float32, initializer=tf.constant_initializer(self.pretrain_variables['layer3_conv/weights:0']))
            biases = tf.get_variable(name='bias',
                                     shape=[128],
                                     dtype=tf.float32, initializer=tf.constant_initializer(self.pretrain_variables['layer3_conv/biases:0']))
            net = tf.nn.conv2d(net, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(net, bias=biases)
            net = tf.nn.relu(net)
            print_activations(net)

        with tf.variable_scope('pool3') as scope:
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            print_activations(net)
        with tf.variable_scope('conv4') as scope:
            kernel = tf.get_variable(name='weight',
                                     shape=[3, 3, 128, 256],
                                     dtype=tf.float32, initializer=tf.constant_initializer(self.pretrain_variables['layer4_conv/weights:0']))
            biases = tf.get_variable(name='bias',
                                     shape=[256],
                                     dtype=tf.float32, initializer=tf.constant_initializer(self.pretrain_variables['layer4_conv/biases:0']))
            net = tf.nn.conv2d(net, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(net, bias=biases)
            net = tf.nn.relu(net)
            print_activations(net)
        with tf.variable_scope('pool4') as scope:
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            print_activations(net)
        with tf.variable_scope('conv5') as scope:
            kernel = tf.get_variable(name='weight',
                                     shape=[3, 3, 256, 256],
                                     dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
            biases = tf.get_variable(name='bias',
                                     shape=[256],
                                     dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
            net = tf.nn.conv2d(net, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(net, bias=biases)
            net = tf.nn.relu(net)
            print_activations(net)
        with tf.variable_scope('pool5') as scope:
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            print_activations(net)
        with tf.variable_scope('fc1') as scope:
            net = tf.reshape(net, [-1, 3 * 3 * 256])
            weight = tf.get_variable(name='weight',
                                     shape=[3 * 3 * 256, 512],
                                     dtype=tf.float32, initializer=tf.constant_initializer(self.pretrain_variables['layer5_fc/weights:0']))
            biases = tf.get_variable(name='bias',
                                     shape=[512],
                                     dtype=tf.float32, initializer=tf.constant_initializer(self.pretrain_variables['layer5_fc/biases:0']))
            net = tf.matmul(net, weight) + biases
            net = tf.nn.relu(net)
            print_activations(net)
        with tf.variable_scope('fc2') as scope:
            weight = tf.get_variable(name='weight',
                                     shape=[512, 256],
                                     dtype=tf.float32, initializer=tf.constant_initializer(self.pretrain_variables['layer6_fc/weights:0']))
            biases = tf.get_variable(name='bias',
                                     shape=[256],
                                     dtype=tf.float32, initializer=tf.constant_initializer(self.pretrain_variables['layer6_fc/biases:0']))
            net = tf.matmul(net, weight) + biases
            net = tf.nn.relu(net)
            print_activations(net)
        with tf.variable_scope('fc3') as scope:
            weight = tf.get_variable(name='weight',
                                     shape=[256, 28],
                                     dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
            biases = tf.get_variable(name='bias',
                                     shape=[28],
                                     dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
            net = tf.matmul(net, weight) + biases
            net = tf.nn.relu(net)
            print_activations(net)
        net = tf.reshape(net, [-1, 2, 2, 7])
        print_activations(net)
        self.predict = net
        return net
    def train(self, net):
        pobj_pred = net[:, :, :, 0:1]
        pobj_truth = self.label[:, :, :, 0:1]
        print_activations(pobj_pred)
        print_activations(pobj_truth)
        box_pred = net[:, :, :, 1:5]
        box_truth = self.label[:, :, :, 1:5]
        print_activations(box_pred)
        print_activations(box_truth)
        pclass_pred = net[:, :, :, 5:]
        pclass_truth = self.label[:, :, :, 5:]
        print_activations(pclass_pred)
        print_activations(pclass_truth)
        IOU = cacl_iou(box1=box_pred, box2=box_truth)
        print_activations(IOU)
        object_mask = tf.cast(IOU >= 0.5, tf.float32) * pobj_truth
        print_activations(object_mask)
        noonject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask
        print_activations(noonject_mask)
        loss_obj = tf.reduce_sum(tf.square(pobj_truth - pobj_pred) * object_mask)
        print_activations(loss_obj)
        loss_noobj = tf.reduce_sum(tf.square(pobj_truth - pobj_pred) * noonject_mask)
        print_activations(loss_noobj)
        loss_coord = tf.reduce_sum(tf.reduce_sum(tf.square(box_pred[:, :, :,0:2]
 			- box_truth[:, :, :, 0:2]), axis=3, keep_dims=True) * object_mask)
 			+ tf.reduce_sum(tf.reduce_sum(tf.square(tf.sqrt(box_pred[:, :, :, 2:4])
 			- tf.sqrt(box_truth[:, :, :, 2:4])), axis=3, keep_dims=True) * object_mask)
        print_activations(loss_coord)
        print_activations(box_pred[:, :, :, 2:])
        print_activations(box_pred[:, :, :, 0:2])
        loss_class = tf.reduce_sum(tf.square(pclass_pred - pclass_truth) * pobj_truth)
        print_activations(loss_class)
        total_loss = loss_obj + self.lambda_noobj * loss_noobj + self.lambda_coord * loss_coord + loss_class
        train_step = tf.train.AdamOptimizer(1e-2).minimize(total_loss / FLAGS.train_batch_size)
        image_batch_raw, label_batch_raw, filename_batch_raw = car_input(is_random=True, is_train=True)
        image_batch_test, label_batch_test, filename_batch_test = car_input(is_random=False, is_train=False)
        train_IOU = tf.reduce_mean(IOU)
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            for i in range(int(FLAGS.num_epoches * FLAGS.train_set_size / FLAGS.train_batch_size)):
                image_batch, label_batch = sess.run([image_batch_raw, label_batch_raw])
                sess.run([train_step], feed_dict={self.input: image_batch, 
                    self.label: label_batch})
                if not i % 10:
                    predict, loss, train_iou, iou = sess.run([net, total_loss, train_IOU, IOU], 
                            feed_dict={
                                self.input: image_batch, 
                                self.label: label_batch})
                    print('step %d, loss: %.4f, train_IOU: %.4f ' % (i, loss, train_iou))
            image_test, label_test = sess.run([image_batch_test, label_batch_test])
            loss = sess.run(total_loss, feed_dict={self.input: image_test, self.label: label_test})
            print(loss / FLAGS.test_batch_size)
            coord.request_stop()
            coord.join(threads)
    def test(self):
        image_batch_test, label_batch_test, filename_batch_test = car_input(is_random=False, is_train=False)
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            image_test, label_test = sess.run([image_batch_test, label_batch_test])
            predict = sess.run(self.predict, feed_dict={self.input: image_test})
            for i in range(FLAGS.test_batch_size):
                print(predict[i], label_test[i])
                print('--------------------------')

def main(_):
    variables = restore_vars()
    print(variables.keys())
    net = hugenet(variables)
    feature = net.inference()
    net.train(feature)
    net.test()

if __name__ == '__main__':
    tf.app.run()
