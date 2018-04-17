import tensorflow as tf

def print_activations(t):
    print(t.op.name, t.get_shape().as_list())

def cacl_iou(box1, box2):
    box1 = tf.stack([box1[:, :, :, 0] - box1[:, :, :, 2] / 2.0,
                     box1[:, :, :, 1] - box1[:, :, :, 3] / 2.0,
                     box1[:, :, :, 0] + box1[:, :, :, 2] / 2.0,
                     box1[:, :, :, 1] + box1[:, :, :, 3] / 2.0], axis=3)
    box2 = tf.stack([box2[:, :, :, 0] - box2[:, :, :, 2] / 2.0,
                     box2[:, :, :, 1] - box2[:, :, :, 3] / 2.0,
                     box2[:, :, :, 0] + box2[:, :, :, 2] / 2.0,
                     box2[:, :, :, 1] + box2[:, :, :, 3] / 2.0], axis=3)
    print_activations(box1)
    print_activations(box2)
    left_up = tf.maximum(box1[:, :, :, :2], box2[:, :, :, :2])
    right_down = tf.minimum(box1[:, :, :, 2:], box2[:, :, :, 2:])
    intersection = tf.maximum(0.0, right_down - left_up)
    inter_square = intersection[:, :, :, 0:1] * intersection[:, :, :, 1:]
    print_activations(inter_square)
    square1 = (box1[:, :, :, 2] - box1[:, :, :, 0]) * (box1[:, :, :, 3] - box1[:, :, :, 1])
    square2 = (box2[:, :, :, 2] - box2[:, :, :, 0]) * (box2[:, :, :, 3] - box2[:, :, :, 1])
    square1 = tf.expand_dims(square1, axis=3)
    square2 = tf.expand_dims(square2, axis=3)
    print_activations(square2)
    union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)
    print_activations(union_square)
    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)
