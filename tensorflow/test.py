# import tensorflow as tf

# dataset = tf.data.TextLineDataset(['LSTM.py'])
# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next()
# with tf.Session() as sess:
#     sess.run(iterator.initializer)
#     sess.run(next_element)
#     sess.run(next_element)
#     sess.run(next_element)
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from gcn import BaseModel
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

train_data = tf.data.Dataset.from_tensor_slices(
    mnist.train.images.astype(np.float32))
val_data = tf.data.Dataset.from_tensor_slices(
    mnist.validation.images.astype(np.float32))
test_data = tf.data.Dataset.from_tensor_slices(
    mnist.test.images.astype(np.float32))
train_labels = tf.data.Dataset.from_tensor_slices(
    mnist.train.labels)
val_labels = tf.data.Dataset.from_tensor_slices(
    mnist.validation.labels)
test_labels = tf.data.Dataset.from_tensor_slices(
    mnist.test.labels)

tf.logging.set_verbosity(tf.logging.INFO)
class CNN(BaseModel):
    def __init__(self, shape_size, channel_size,
                 feature_size, n_classes):
        super().__init__()
        # input grid: K x K, channel: C,
        # output channel: F
        self._K = shape_size
        self._C = channel_size
        self._F = feature_size
        self._N = n_classes
        self.learning_rate = 0.0001
        self.dropout = 0.5
        self.regularization = 0.001

    def _inference(self, x, dropout):
        with tf.variable_scope('conv1'):
            W = self._weight_variable([self._K, self._K, 1, self._F])
            b = self._bias_variable([self._F])

            x_2d = tf.reshape(x, [-1, self._K, self._K, self._C])
            y_2d = tf.nn.conv2d(x_2d, W, strides=[1, 1, 1, 1], padding='SAME') + b
            y_2d = tf.nn.relu(y_2d)

        with tf.variable_scope('fc1'):
            y = tf.reshape(y_2d, [-1, self._K * self._K * self._F])
            W = self._weight_variable([self._K * self._K * self._F, self._N])
            b = self._bias_variable([self._N])
            y = tf.matmul(y, W) + b

        return y

cnn = CNN(28, 1, 2, 10)
train_data = train_data.batch(200).prefetch(6)
train_labels = train_labels.batch(200).prefetch(6)
val_data = val_data.batch(200).prefetch(6)
val_labels = val_labels.batch(200).prefetch(6)
data_format = (train_data.output_types, train_data.output_shapes)
labels_format = (train_labels.output_types, train_labels.output_shapes)
cnn.build_graph(data_format, labels_format, tf.get_default_graph())
cnn.fit(train_data, train_labels, val_data, val_labels)
# iterator = dataset.make_one_shot_iterator()
# next_batch = iterator.get_next()

# sess = tf.Session()
# for _ in range(1):
#     print(sess.run(next_batch).reshape(28, 28))
