import tensorflow as tf

class base_model(object):
    def __init__(self):
        self._regularitizers = []

    def predict_with_labels(self, data, labels, sess=None):
        data_init_op = self.data_iterator.make_initailizer(data)
        labels_init_op = self.labels_iterator.make_initalizer(labels)
        feed_dict = {self.ph_dropout: 1.0}
        sess = self._get_session(sess)

        try:
            sess.run([data_init_op, labels_init_op])
            while True:
                batch_loss, batch_prediction = self.session.run(
                    [self.op_loss, self.op_prediction], feed_dict)
                # In the event of large dataset, we may want to write prediction
                # to save memory, the limitation is the lack of multi-thread support
                # with the keyword yield
                yield batch_loss, batch_prediction
        except tf.errors.OutOfRangeError:
            pass

    def predict(self, data, sess=None):
        data_init_op = self.data_iterator.make_initailizer(data)
        feed_dict = {self.ph_dropout: 1.0}
        sess = self._get_session(sess)

        try:
            sess.run([data_init_op])
            while True:
                batch_prediction = self.session.run(
                    [self.op_prediction], feed_dict)
                # In the event of large dataset, we may want to write prediction
                # to save memory
                yield batch_prediction
        except tf.errors.OutOfRangeError:
            pass

    def evaluate(self, data, labels, sess=None):
        t_process, t_wall = time.process_time(), time.time()
        predictions, loss = [], []
        for batch_loss, batch_prediction in self.predict(data, labels, sess):
            prediction.extend(batch_prediction)
            loss.extend(batch_loss)

        ncorrects = sum(predictions)

    def inference(self, data, dropout):
        """
        This method that build the main graph
        """
        logits = self._inference(data, dropout)
        return logits

    def fit(self, training_data, training_labels, val_data, val_labels):
        """Runs the entire epoch of the train_dataset
        """
        sess = tf.Session(graph=self.graph)
        # some model saving here
        sess.run(self.op_init)
        training_data_init_op = self.data_iterator.make_initializer(training_data)
        training_labels_init_op = self.labels_iterator.make_initializer(training_labels)
        # val_data_init_op = self.data_iterator.make_initailizer(val_data)
        # val_labels_init_op = self.labels_iterator.make_initializer(val_labels)

        for _ in range(10):
            sess.run([training_data_init_op, training_labels_init_op])
            try:
                while True:
                    feed_dict = {self.ph_dropout: self.dropout}
                    # The only guarantee tensorflow makes about order of execution is that
                    # all dependencies (either data or control) of an op are executed
                    # before that op gets executed.
                    loss_average, train_op = sess.run([self.op_loss_average, self.op_train, feed_dict])
            except tf.errors.OutOfRangeError:
                pass

            try:
                while True:
                    self.evaluate(val_data, val_labels, sess)
                    # self._get_session().run([self.op_prediction])

    def loss(self, logits, labels, regularization=None):
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                labels = tf.to_int64(labels)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                cross_entropy = tf.reduce_mean(cross_entropy)
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)

            loss = cross_entropy + regularization
            return loss

    def training(self, loss, learning_rate, decay_steps=None, decay_rate=None, momentum=None):
        with tf.name_scope('training'):
            # learning rate
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads, global_step)
            return train_op

    def build_graph(self, data_format, labels_format):
        """
        data_format, labels_format: output_types, output_shapes of data_dataset, labels_dataset
        distinguish data and labels because some operations don't depend on labels, e.g.,
        predict, inference etc.
        """
        self.graph = tf.Graph

        with self.graph.as_default():

            # Inputs
            with tf.name_scope('inputs'):
                self.data_iterator = tf.data.Iterator.from_structure(data_format)
                self.labels_iterator = tf.data.Iterator.from_structure(labels_format)
                self.pipe_data = self.data_iterator.get_next()
                self.pipe_labels = self.labels_iterator.get_next()
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')
                #self.pipe_data, self.pipe_labels, self.pipe_dropout \
                    #= input_iterator.get_next()

            op_logits = self.inference(self.pipe_data, self.ph_dropout)
            # self.op_loss, self.op_loss_average = self.loss(op_logits,
            #                                                self.pipe_labels,
            #                                                self.regularization)
            self.op_loss = self.loss(op_logits, self.pipe_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                                          self.decay_steps, self.decay_rate,
                                          self.momentum)
            self.op_prediction = self.prediction(op_logits)
            self.op_init = tf.global_variables_initializer()

            # self.op_summary = tf.summary.merge_all()
        #     self.op_saver = tf.train.Saver(max_to_keep=5)

        # self.graph.finalize()
