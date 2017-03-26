### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

import tensorflow as tf
import math
from model import cost
from load_data import image_shape, classes, X_train, y_train, X_test, y_test, \
    X_valid, y_valid
from model import x, y, keep_prob, train_feed_dict, test_feed_dict, \
    valid_feed_dict, cost, accuracy

epochs = 10
batch_size = 60
learn_rate = 0.05

# Gradient Descent
opt = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

init_var = tf.initialize_all_variables()
valid_accuracy = 0.

batch_costs = []

with tf.Session() as session:
    session.run(init_var)
    batch_cnt = int(math.ceil(len(X_train) / batch_size))

    for epoch in range(epochs):
        for batch_i in range(batch_cnt):
            batch_start = batch_i * batch_size
            batch_x = X_train[batch_start:(batch_start + batch_size)]
            batch_y = y_train[batch_start:(batch_start + batch_size)]

            session.run(opt, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

        if epoch % 10 == 0:
            c, a = session.run([cost, accuracy], feed_dict=valid_feed_dict)
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.5f}".format(c),
                  "accuracy=", "{:.5f}".format(a))
            batch_costs.append(c)

    print("Finish Opt!")
