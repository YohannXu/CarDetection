import os
import tensorflow as tf

tf.app.flags.DEFINE_string('train_tfrecord_directory', '/home/xyh/PycharmProjects/CarCount/dataset/tfrecord/train', 'train tfrecord directory')
tf.app.flags.DEFINE_string('test_tfrecord_directory', './tfrecord/test/', 'test tfrecord directory')
FLAGS = tf.app.flags.FLAGS

def list_tfrecord_file(file_list, target_path):
    tfrecord_list = []
    for i in range(len(file_list)):
        current_file_abs_path = os.path.abspath(os.path.join(target_path, file_list[i]))
        if current_file_abs_path.endswith('.tfrecord'):
            tfrecord_list.append(current_file_abs_path)
            print('Found %s successfully!' % file_list[i])
        else:
            pass
    return tfrecord_list

def tfrecord_auto_traversal(target_dir):
    print(target_dir)
    current_folder_filename_list = os.listdir(target_dir)
    tfrecord_list = []
    if current_folder_filename_list != None:
        print('%s files were found under current folder.' % len(current_folder_filename_list))
        print("Please be noted that only files end with '*.tfrecord' will be loaded!")
        tfrecord_list = list_tfrecord_file(current_folder_filename_list, target_dir)
        if len(tfrecord_list) != 0 :
            for list_index in range(len(tfrecord_list)):
                print(tfrecord_list[list_index])
        else:
            print('Cannot find any tfrecord files, please check the path.')
    return tfrecord_list

def main():
    tfrecord_list = tfrecord_auto_traversal(FLAGS.train_tfrecord_directory)
    tfrecord_list = tfrecord_auto_traversal(FLAGS.test_tfrecord_directory)

if __name__ == '__main__':
    main()
