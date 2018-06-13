# 图片二进制读写

writer = tf.python_io.TFRecordWriter(filename)

image = cv2.imread(image_name)
image_raw = image.toString()

example = tf.train.Example(features=tf.train.Features(features={
    'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
}))

writer.write(example.SerializeToString())
writer.close()



filename_queue = tf.train.string_input_producer([tfrecord_filename],)
reader = tf.TFRecordReader()
\_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example, features={
    'image':tf.FixedLenFeature([], tf.int64)
})
image_encoded = features['image']
image_raw = tf.image.decode_jpeg(image_encoded, channels=3)


# 网络构建
with tf.variable_scope('conv1') as scope:
    kernel = tf.get_variable(name='weight',
        shape=[3, 3, 3, 32],
        dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
    biases = tf.get_variable(name='bias',
        shape=[32],
        dtype=tf.float32, initializer=tf.constant_initializer(0))
    net = tf.nn.conv2d(self.input, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.bias_add(net, bias=biases)
    net = tf.nn.relu(net)
with tf.variable_scope('pool1') as scope:
    net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
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

# 参数解析

reader = tf.train.NewCheckpointReader(filename)
for tensor_name in tensor_names:
    tensor = reader.get_tensor(tensor_name)
    tf.assign(tf.get_tensor(tensor_name), tensor)

# 执行检测

image = image_reader()
image_resized = tf.image.resize(image, (new_height, new_width))
with tf.Session() as sess:
    predict = sess.run(detection, feed_dict={image_input:image_resized})

# 非极大值抑制

x1 = dets[:, 0]
y1 = dets[:, 1]
x2 = dets[:, 2]
y2 = dets[:, 3]
scores = dets[:, 4]

areas = (x2 - x1 + 1) * (y2 - y1 + 1)
order = scores.argsort()[::-1]

keep = []
while order.size > 0:
    i = order[0]
    keep.append(i)
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (areas[i] + areas[order[1:]] - inter)

    inds = np.where(ovr <= thresh)[0]
    order = order[inds + 1]

# 检测结果持久化

with open(filename) as f:
    for box in boxes:
        f.write(json.dumps(box))
        f.write(':')
    f.write('\n')

# 检测结果读取

with open(filename) as f:
    boxes = f.readline().split(':')
    for box in boxes:
        box = json.loads(box)

# 新车校验

x_center = (box['topleft']['x'] + box['bottomright']['x'])/2
y_center = (box['topleft']['y'] + box['bottomright']['y'])/2

if abs(y_center - y_standard) 小于　s:
    if not frozen:
        new = True
    else:
        new = False
else:
    new = False

# 道路流量统计

for box in boxes:
    if new(box):
        count += 1
        frozen = -4
    elif frozen:
        frozen -= 1

# 统计结果持久化

with open(filename) as f:
    while True:
        if not i % skip_frame:
            f.write(str(count))
            f.write('\n')
        i += 1

# 检测结果查询

filename = os.path.join(prefix, key_words)
with open(filename) as f:
    line = f.readline()
    while line:
        count = 1
        image_name = image_ + str(count) + '.jpg'
        image = cv2.imread(image_name)
        boxes = line.split(':')
        for box in boxes:
        topleft = (det['topleft']['x'], det['topleft']['y'])
        bottomright = (det['bottomright']['x'], det['bottomright']['y'])
        cv2.rectangle(image, topleft, bottomright, (255,0,0), 5)
        line = f.readline()

videoWriter = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 25, (1280, 720))

for image in images:
    videoWriter.write(image)

# 统计结果查询

同上

lines = f.readlines()
length = len(lines)

add = int(length / divide)

x = [i for i in range(0, length, add)]
y = [int(lines[i]) for i in x]

y2 = []
for i in range(divide+1):
    print(i)
    if i == 0:
        y2.append(y[i])
    else:
        y2.append(y[i]-y[i-1])

plt.figure(1)
plt.subplot(211)
plt.plot(x, y)
plt.subplot(212)
plt.plot(x, y2)
plt.show()

