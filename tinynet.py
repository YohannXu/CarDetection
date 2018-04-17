import tensorflow as tf
import os
from tensorflow.contrib import slim
from utils import print_activations, cacl_iou
from car_input import car_input
import matplotlib.pyplot as plt

flags = tf.app.flags
flags.DEFINE_integer('num_channel', 3, 'the number of input channels')
flags.DEFINE_integer('image_size', (50, 50), 'the size of input image')
flags.DEFINE_integer('num_class', 2, 'the number of class')
flags.DEFINE_integer('batch_size', 16, 'The size of batch')
flags.DEFINE_integer('num_epoches', 100, 'The number of epoches.')
FLAGS = flags.FLAGS

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class TinyNet(object):
    def __init__(self):
        self.batch_size = FLAGS.batch_size
        self.num_channel = FLAGS.num_channel
        self.image_size = FLAGS.image_size
        self.num_class = FLAGS.num_class
        self.basename = os.path.dirname(__file__)
        self.writer = tf.summary.FileWriter(os.path.join(self.basename,'log-pretrain'))
        self.input_image = self.input = tf.placeholder(tf.float32, shape=[None, self.image_size[0], self.image_size[1], self.num_channel])
        self.label = tf.placeholder(tf.float32, shape=[None, self.num_class])
    def inference(self):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=3,
                            reuse=None,
                            trainable=True):
            with slim.arg_scope([slim.max_pool2d],
                                kernel_size=2,
                                stride=2,
                                padding='VALID'):
                net = slim.conv2d(self.input_image, 32, scope='layer1_conv')
                print_activations(net)
                net = slim.max_pool2d(net, scope='pool1')
                print_activations(net)
                net = slim.conv2d(net, 64, scope='layer2_conv')
                print_activations(net)
                net = slim.max_pool2d(net, scope='pool2')
                print_activations(net)
                net = slim.conv2d(net, 128, scope='layer3_conv')
                print_activations(net)
                net = slim.max_pool2d(net, scope='pool3')
                print_activations(net)
                net = slim.conv2d(net, 256, scope='layer4_conv')
                print_activations(net)
                net = slim.max_pool2d(net, scope='pool4')
                print_activations(net)
                net = tf.reshape(net, [-1, 3 * 3 * 256])
                net = slim.fully_connected(net, 512, scope='layer5_fc')
                print_activations(net)
                net = slim.fully_connected(net, 256, scope='layer6_fc')
                print_activations(net)
                net=slim.fully_connected(net,self.num_class,activation_fn=None, scope='layer7_fc')
                print_activations(net)
                net = slim.softmax(net)
                print_activations(net)
                return net

    def train(self, net):
        label_offset_pretrain = -tf.ones([FLAGS.pretrain_batch_size], dtype=tf.int64)
        label_offset_pretest = -tf.ones([FLAGS.pretest_set_size], dtype=tf.int64)
        image_batch_raw, label_batch_raw, filename_batch_raw = car_input(is_random=True, is_pretrain=True)
        label_batch_one_hot=tf.one_hot(tf.add(label_batch_raw, label_offset_pretrain), depth=2, on_value=1.0, off_value=0.0)
        test_image_batch_raw,test_label_batch_raw,_=car_input(is_random=True, is_pretrain=False)
        test_label_batch_one_hot = tf.one_hot(tf.add(test_label_batch_raw, label_offset_pretest), depth=2, on_value=1.0, off_value=0.0)
        total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=net))
        total_loss_summary = tf.summary.scalar('total_loss', total_loss)
        train_op = tf.train.AdamOptimizer(1e-5).minimize(total_loss)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net, 1), tf.argmax(self.label, 1)), tf.float32))
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            self.writer.add_graph(sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            for i in range(int(FLAGS.num_epoches * FLAGS.pretrain_set_size / FLAGS.batch_size)):
                image_batch, label_batch, filename_batch = sess.run([image_batch_raw, label_batch_one_hot, filename_batch_raw])
                print(type(image_batch))
                sess.run([train_op],feed_dict={self.input:image_batch, self.label:label_batch})
                if i % 10 == 0:             			                                                            	            
                    acc,loss,loss_summary=sess.run([accuracy,total_loss,total_loss_summary],feed_dict={
                        self.in put:image_batch, 
                        self.label:label_batch})
                    self.writer.add_summary(loss_summary, i)
                    print('step %d accuracy: %.2f, loss: %.4f' % (i, acc, loss))
            saver.save(sess, os.path.join(self.basename, 'checkpoint/model1.ckpt'))
            test_image_batch, test_label_batch = sess.run([test_image_batch_raw, test_label_batch_one_hot])
            acc = sess.run(accuracy, feed_dict={
                self.input:test_image_batch, 
                self.label:test_label_batch})
            print('test accuracy : %.2f' % acc)
            for var in tf.trainable_variables():
                print(var.name)
            coord.request_stop()
            coord.join(threads)

def main(_):
    net = TinyNet()
    feature = net.inference()
    net.train(feature)

if __name__ == '__main__':
    tf.app.run()
