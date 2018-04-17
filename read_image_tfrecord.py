import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image
from find_tfrecord import tfrecord_auto_traversal
flags = tf.app.flags
flags.DEFINE_integer('num_pretrain_images', 320, 'Number of pretrain images in your tfrecord.')
flags.DEFINE_integer('num_pretest_images', 64, 'Number of pretest images in your tfrcord.')
flags.DEFINE_string('pretrain_tfrecord_path', 'tfrecord/pretrain', 'The path of pretrain tfrecord files.')
flags.DEFINE_string('pretest_tfrecord_path', 'tfrecord/pretest', 'The path of pretest tfrecord files.')
flags.DEFINE_string('pretrain_save_path', 'resized_image/pretrain', 'The path to save resized pretrain images.')
flags.DEFINE_string('pretest_save_path', 'resized_image/pretest', 'The path to save resized pretest images.')
flags.DEFINE_integer('class_number', 2, 'Number of class in your tfrecord.')
flags.DEFINE_integer('image_height', 50, 'Height of the output image after crop and resize.')
flags.DEFINE_integer('image_width', 50, 'Width of the output image after crop and resize')
FLAGS = flags.FLAGS

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    if isinstance(value, str):
        value = value.encode()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class image_object():
    def __init__(self):
        self.image = tf.Variable([], dtype=tf.string)
        self.height = tf.Variable([], dtype=tf.int64)
        self.width = tf.Variable([], dtype=tf.int64)
        self.filename = tf.Variable([], dtype=tf.string)
        self.label = tf.Variable([], dtype=tf.int32)

def read_and_decode(filename_quene):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_quene)
    features = tf.parse_single_example(serialized_example, features={
        'image/encoded':tf.FixedLenFeature([], tf.string),
        'image/height':tf.FixedLenFeature([], tf.int64),
        'image/width':tf.FixedLenFeature([], tf.int64),
        'image/filename':tf.FixedLenFeature([], tf.string),
        'image/label':tf.FixedLenFeature([], tf.int64)
    })
    image_encoded = features['image/encoded']
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3)
    current_image_object = image_object()
    resized_image=tf.image.resize_images(image_raw,[FLAGS.image_height,FLAGS.image_width])
    current_image_object.image = tf.cast(resized_image, dtype=tf.uint8)
    current_image_object.height = features['image/height']
    current_image_object.width = features['image/width']
    current_image_object.filename = features['image/filename']
    current_image_object.label = tf.cast(features['image/label'], tf.int32)
    return current_image_object

def read_image_from_tfrecord(dataset_name):
    if ('pretrain' == dataset_name):
        num_images = FLAGS.num_pretrain_images
        tfrecord_filename = FLAGS.pretrain_tfrecord_path
        save_path = FLAGS.pretrain_save_path
    elif ('pretest' == dataset_name):
        num_images = FLAGS.num_pretest_images
        tfrecord_filename = FLAGS.pretest_tfrecord_path
        save_path = FLAGS.pretest_save_path
    else:
        print('Please type right tfrecord dataset!')
        sys.exit(0)
    basename = os.path.dirname(__file__)
    print(basename)
    file_path = os.path.join(basename, tfrecord_filename)
    print(file_path)
    save_path = os.path.join(basename, save_path)
    filename_quene = tf.train.string_input_producer(tfrecord_auto_traversal(file_path))
    current_image_object = read_and_decode(filename_quene)
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print("Write resized images to the fold" + save_path)
        for i in range(num_images):
            image, label = sess.run([current_image_object.image, current_image_object.label])
            img = Image.fromarray(image, 'RGB')
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            img.save(os.path.join(save_path, 'class_' + str(label) + '_Index_' + str(i) + '.jpeg'))
            if not i % 10:
                print('%d images in %d has been saved!' % (i, num_images))
        print('Complete!')
        coord.request_stop()
        coord.join(threads)
    print('cd to current directory, the folder "resized_image" should contains %d images with %dx%d size.' % (num_images, FLAGS.image_height, FLAGS.image_width))

if __name__ == '__main__':
    read_image_from_tfrecord('pretrain')
    read_image_from_tfrecord('pretest')
