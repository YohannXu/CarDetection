import os
import random
import sys
import threading
import pandas as pd
from datetime import datetime
import re
import numpy as np
import tensorflow as tf
from PIL import Image

tf.app.flags.DEFINE_string('train_directory',
                           '/home/xyh/PycharmProjects/CarCount/dataset/images/train', 'Training data directory')
tf.app.flags.DEFINE_string('test_directory',
                           '/home/xyh/PycharmProjects/CarCount/dataset/images/test', 'test data directory')
tf.app.flags.DEFINE_string('train_tfrecord_directory',
                           '/home/xyh/PycharmProjects/CarCount/dataset/tfrecord/train/', 'train tfrecord directory')
tf.app.flags.DEFINE_string('test_tfrecord_directory',
                           '/home/xyh/PycharmProjects/CarCount/dataset/tfrecord/test/', 'test tfrecord directory')
tf.app.flags.DEFINE_integer(
    'train_shards', 4, 'Number of shards in training TFrecord files.')
tf.app.flags.DEFINE_integer(
    'test_shards', 4, 'Number of shards in test TFrecord files.')
tf.app.flags.DEFINE_integer(
    'num_threads', 4, 'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_string(
    'train_labels_file', '/home/xyh/PycharmProjects/CarCount/dataset/labels/train_label.txt', 'Train labels file')
tf.app.flags.DEFINE_string(
    'test_labels_file', '/home/xyh/PycharmProjects/CarCount/dataset/labels/test_label.txt', 'Test labels file')
FLAGS = tf.app.flags.FLAGS

i = 0
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

def _convert_to_example(filename, image_buffer, label, height, width):
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/label': _int64_feature(label),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)
    }))
    return example

class ImageCoder(object):
    def __init__(self):
        self._sess = tf.Session(config=config)
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(
            image, format='rgb', quality=100)
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)
    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})
    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg, feed_dict={
                               self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _is_png(filename):
    return '.png' in filename

def _process_image(filename, coder):
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)
    image = coder.decode_jpeg(image_data)
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width

def _process_image_files_batch(coder, thread_index, ranges, name, filenames, labels, num_shards, tfrecord_directory):
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)
    shards_ranges = np.linspace(ranges[thread_index][0],
                                ranges[thread_index][1],
                                num_shards_per_batch + 1)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
    counter = 0
    for s in range(num_shards_per_batch):
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.2d-of-%.2d.tfrecord' % (
            name, shard, num_shards)
        output_file = os.path.join(tfrecord_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        shard_counter = 0
        files_in_shard = np.arange(
            shards_ranges[s], shards_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            image_buffer, height, width = _process_image(filename, coder)
            example = _convert_to_example(
                filename, image_buffer, label, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1
            print(counter)
            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()

def _process_image_files(name, filenames, labels, num_shards, tfrecord_directory):
    assert len(filenames) == len(labels)
    spacing = np.linspace(
        0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    print('Launching %d threads for spacings: %s' %
          (FLAGS.num_threads, ranges))
    sys.stdout.flush()
    coord = tf.train.Coordinator()
    coder = ImageCoder()
    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                labels, num_shards, tfrecord_directory)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()

def _find_image_files(data_dir, labels_file):
    label_list = []
    labels = pd.read_csv(labels_file, sep=' ', header=None)
    for i in range(labels.shape[0]):
        label_list.append(list(labels.ix[i]))
    labels = []
    filenames = []
    texts = []
    jpeg_file_path = '%s/*' % data_dir
    matching_files = tf.gfile.Glob(jpeg_file_path)
    print(len(matching_files))
    filenames.extend(matching_files)
    for name in filenames:
        print(name)
        expression = \
         '/home/xyh/PycharmProjects/CarCount/dataset/images/(train|test)/Index_([0-9]*).jpeg'
        index = int(re.search(expression, name).group(2))
        labels.append(label_list[index])
    return filenames, labels
def _process_dataset(name, directory, num_shards, labels_file, tfrecord_directory):
    filenames, labels = _find_image_files(directory, labels_file)
    _process_image_files(name, filenames, labels,
                         num_shards, tfrecord_directory)

def main(_):
    assert not FLAGS.train_shards % FLAGS.num_threads, \
        ('Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    print('Saving results to %s !' % FLAGS.train_tfrecord_directory
    _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards,
                     FLAGS.train_labels_file, FLAGS.train_tfrecord_directory)
    _process_dataset('test', FLAGS.test_directory, FLAGS.test_shards,
                     FLAGS.test_labels_file, FLAGS.test_tfrecord_directory)
if __name__ == '__main__':
    tf.app.run()
