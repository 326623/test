import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, y_train, x_test, y_test = (tf.convert_to_tensor(x_train),
                                    tf.convert_to_tensor(y_train),
                                    tf.convert_to_tensor(x_test),
                                    tf.convert_to_tensor(y_test))

num_classes = 10
img_size_flat = 28 * 28

x = tf.placeholder(tf.float32, [None, 28, 28])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

weights = tf.Variable(tf.zeros([28, 28, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))
logits = tf.matmul(x, weight) + biases
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, axis=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                           labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradienDescentOptimizer(learning_rate=0.5).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.case(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

batch_size = 100
feed_dict_test = {x: x_test,
                  y_true: y_test,
                  y_true_cls: y_test_cls}

# def optimize(num_iterations):
#     for i in range(num_iterations):
#         x_batch, y_true_batch, _ = data.random
