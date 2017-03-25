### Define your architecture here.
### Feel free to use as many code cells as needed.

import tensorflow as tf
from load_data import image_shape, classes, X_train, y_train, X_test, y_test,\
    X_valid, y_valid

def max_pool_2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')

def conv_2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding='SAME')

# input
x = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1]])
y = tf.placeholder(tf.float32, shape=[None, len(classes)])

# convolutional layer #1
conv1_depth = 32
w_conv1 = tf.Variable(tf.truncated_normal([5, 5, image_shape[2], conv1_depth],
                                          stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

x = tf.reshape(x, shape=[-1, image_shape[0], image_shape[1], image_shape[2]])

# Relu
h_conv1_relu = tf.nn.relu(conv_2d(x, w_conv1) + b_conv1)
h_conv1_pool = max_pool_2d(h_conv1_relu)

num_hidden = 128

# connected layer
reduce_size = image_shape[0] // 4 * image_shape[1] // 4 * conv1_depth
w_fc1 = tf.Variable(tf.truncated_normal([reduce_size, num_hidden], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[num_hidden]))

shape = h_conv1_pool.get_shape().as_list()
h_conv1_pool_flat = tf.reshape(h_conv1_pool,
                               [-1, shape[1] * shape[2] * shape[3]])
h_fc1_relu = tf.nn.relu(tf.matmul(h_conv1_pool_flat, w_fc1) + b_fc1)

# dropout
# keep probability
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1_relu, keep_prob)

# out layer
w_fc2 = tf.Variable(tf.truncated_normal([num_hidden, len(classes)], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[len(classes)]))

y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

#prediction = tf.nn.softmax(y_conv)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,
                                                              labels=y))
is_correct = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

train_feed_dict = {x: X_train, y: y_train, keep_prob: 1.0}
test_feed_dict = {x: X_test, y: y_test, keep_prob: 1.0}
valid_feed_dict = {x: X_valid, y: y_valid, keep_prob: 1.0}
