import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")

# numpy array to tf.Dataset
assert x_train.shape[0] == y_train.shape[0]

time_steps = 28
num_units = 128
n_inputs = 28
learning_rate = 0.001
n_classes = 10
batch_size = 128

train_size = y_train.shape[0]
test_size = y_test.shape[0]
temp = np.zeros([train_size, n_classes])
temp[np.arange(train_size), y_train] = np.float32(1.0)
y_train = temp

temp = np.zeros([test_size, n_classes])
temp[np.arange(test_size), y_test] = np.float32(1.0)
y_test = temp

x_train, x_test = np.float32(x_train / 255.0), np.float32(x_test / 255.0)

features_placeholder = tf.placeholder(x_train.dtype, x_train.shape)
labels_placeholder = tf.placeholder(y_train.dtype, y_train.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
dataset = dataset.batch(batch_size)
iterator = dataset.make_initializable_iterator()
next_x, next_y = iterator.get_next()

out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))

input = tf.unstack(next_x, time_steps, 1)
lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)

outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")
prediction = tf.matmul(outputs[-1], out_weights) + out_bias

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=next_y))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(next_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run(iterator.initializer, feed_dict={features_placeholder: x_train,
                                              labels_placeholder: y_train})

    while True:
        try:
            batch_x, batch_y = sess.run([next_x, next_y])
            print(sess.run([opt, accuracy]))
        except tf.errors.OutOfRangeError:
            break
