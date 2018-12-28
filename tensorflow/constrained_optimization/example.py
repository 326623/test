import math
import numpy as np
import tensorflow as tf

tfco = tf.contrib.constrained_optimization

class ExampleProblem(tfco.ConstrainedMinimizationProblem):

    def __init__(self, labels, predictions, recall_lower_bound):
        self._labels = labels
        self._predictions = predictions
        self._recall_lower_bound = recall_lower_bound
        # The number of positively-labeled examples.
        self._positive_count = tf.reduce_sum(self._labels)

    @property
    def objective(self):
        return tf.losses.hinge_loss(labels=self._labels, logits=self._predictions)

    @property
    def constraints(self):
        true_positives = self._labels * tf.to_float(self._predictions > 0)
        true_positive_count = tf.reduce_sum(true_positives)
        recall = true_positive_count / self._positive_count

        return self._recall_lower_bound - recall

    @property
    def proxy_constraints(self):
        true_positives = self._labels * tf.minimum(1.0, self._predictions)
        true_positive_count = tf.reduce_sum(true_positives)
        recall = true_positive_count / self._positive_count
        return self._recall_lower_bound - recall


def average_hinge_loss(labels, predictions):
    num_examples, = np.shape(labels)
    signed_labels = (labels * 2) - 1 # 0, 1 -> -1, 1
    total_hinge_loss = np.sum(np.maximum(0.0, 1.0 - signed_labels * predictions))
    return total_hinge_loss / num_examples

def recall(labels, predictions):
    positive_count = np.sum(labels)
    true_positives = labels * (predictions > 0)
    true_positive_count = np.sum(true_positives)
    return true_positive_count / positive_count

if __name__ == '__main__':
    num_examples = 1000
    num_mislabeled_examples = 200
    dimension = 10

    recall_lower_bound = 0.9

    ground_truth_weights = np.random.normal(size=dimension) / math.sqrt(dimension)
    ground_truth_threshold = 0

    features = np.random.normal(size=(num_examples, dimension)).astype(
        np.float32) / math.sqrt(dimension)

    labels = (np.matmul(features, ground_truth_weights) >
              ground_truth_threshold).astype(np.float32)

    mislabeled_indices = np.random.choice(
        num_examples, num_mislabeled_examples, replace=False)
    labels[mislabeled_indices] = 1 - labels[mislabeled_indices]

    weights = tf.Variable(tf.zeros(dimension), dtype=tf.float32, name='weights')
    threshold = tf.Variable(0.0, dtype=tf.float32, name='threshold')

    constant_labels = tf.constant(labels, dtype=tf.float32)
    constant_features = tf.constant(features, dtype=tf.float32)
    predictions = tf.tensordot(constant_features, weights, axes=(1, 0)) - threshold
    problem = ExampleProblem(
        labels=constant_labels,
        predictions=predictions,
        recall_lower_bound=recall_lower_bound
    )
    # print(problem.proxy_constraints)
    # print(problem.objective)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        optimizer = tfco.MultiplicativeSwapRegretOptimizer(
            optimizer=tf.train.AdagradOptimizer(learning_rate=1.0))
        # optimizer = tf.train.AdagradOptimizer(learning_rate=1.0)
        train_op = optimizer.minimize(problem)

        session.run(tf.global_variables_initializer())
        for _ in range(10000):
            session.run(train_op)

        trained_weights, trained_threshold = session.run((weights, threshold))

        trained_predictions = np.matmul(features, trained_weights) - trained_threshold
        print("Constrained average hinge loss = %f" % average_hinge_loss(
            labels, trained_predictions))
        print("Constrained recall = %f" % recall(labels, trained_predictions))
