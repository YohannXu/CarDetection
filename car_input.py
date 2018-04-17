import tensorflow as tf
import os
from find_tfrecord import tfrecord_auto_traversal
import matplotlib.pyplot as plt
tf.app.flags.DEFINE_integer('pretrain_set_size', 320, 'the size of training image set')
tf.app.flags.DEFINE_integer('pretest_set_size', 64, 'the size of training image set')
tf.app.flags.DEFINE_integer('cropped_image_size', 50, 'the size of the output image after crop and resize')
tf.app.flags.DEFINE_string('pretrain_tfrecord_path', 'tfrecord/pretrain', 'The path of pretrain tfrecord files.')
tf.app.flags.DEFINE_string('pretest_tfrecord_path', 'tfrecord/pretest/', 'The path of pretest tfrecord files.')
tf.app.flags.DEFINE_integer('pretrain_batch_size', 16, 'The size of pretrain batch')
tf.app.flags.DEFINE_integer('pretest_batch_size', 64, 'The size of pretest batch')
FLAGS = tf.app.flags.FLAGS

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    if isinstance(value, str):
        value = value.encode()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class ImageObject():
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
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/label': tf.FixedLenFeature([], tf.int64)
    })
    image_encoded = features['image/encoded']
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3)
    image_object = ImageObject()
    image = tf.image.resize_images(image_raw, [FLAGS.cropped_image_size, FLAGS.cropped_image_size])
    image = tf.cast(image, dtype=tf.uint8)
    image = tf.image.per_image_standardization(image)
    image_object.image = image
    image_object.height = features['image/height']
    image_object.width = features['image/width']
    image_object.filename = features['image/filename']
    image_object.label = tf.cast(features['image/label'], tf.int64)
    return image_object

def car_input(is_random=True, is_pretrain=True):
    if(is_pretrain):
        batch_size = FLAGS.pretrain_batch_size
        tfrecord_filename = FLAGS.pretrain_tfrecord_path
    else:
        batch_size = FLAGS.pretest_batch_size
        tfrecord_filename = FLAGS.pretest_tfrecord_path
    basename = os.path.dirname(__file__)
    tfrecord_filename = os.path.join(basename, tfrecord_filename)
    filenames = tfrecord_auto_traversal(tfrecord_filename)
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    filename_quene = tf.train.string_input_producer(filenames, shuffle=False)
    image_object = read_and_decode(filename_quene)
    image = image_object.image
    label = image_object.label
    filename = image_object.filename
    if(is_random):
        min_fraction_of_examples_in_quene = 2
        min_quene_examples = int(
            FLAGS.pretrain_set_size * min_fraction_of_examples_in_quene)
        print('Filling quene with %d images before starting to train. This will take a few minutes' % min_quene_examples)
        num_preprocess_threads = 1
        image_batch, label_batch, filename_batch = tf.train.shuffle_batch(
            [image, label, filename],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_quene_examples + 3 * batch_size,
            min_after_dequeue=min_quene_examples
        )
        return image_batch, label_batch, filename_batch
    else:
        image_batch, label_batch, filename_batch = tf.train.batch(
            [image, label, filename],
            batch_size=batch_size,
            num_threads=1,
            allow_smaller_final_batch=True,
            capacity=4 * batch_size
        )
        return image_batch, label_batch, filename_batch
def main(_):
    image_batch, label_batch, filename_batch = car_input(is_random=False, is_pretrain=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        try:
            while not coord.should_stop():
                images,labels,filenames=sess.run([image_batch, label_batch, filename_batch])
                print(labels, filenames)
                plt.imshow(images[0])
                plt.show()
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
