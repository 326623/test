import tensorflow as tf

dataset = tf.data.TextLineDataset(['LSTM.py'])
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    sess.run(next_element)
    sess.run(next_element)
    sess.run(next_element)
