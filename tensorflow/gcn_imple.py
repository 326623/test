import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from gcn import BaseModel
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.chebyshev import Chebyshev
import scipy.sparse
import graph
import time
import os
mnist = input_data.read_data_sets('data/mnist', one_hot=False)

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
flags = tf.app.flags
FLAGS = flags.FLAGS

# Graphs.
flags.DEFINE_integer('number_edges', 8, 'Graph: minimum number of edges per vertex.')
flags.DEFINE_string('metric', 'euclidean', 'Graph: similarity measure (between features).')
# TODO: change cgcnn for combinatorial Laplacians.
flags.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')
flags.DEFINE_integer('coarsening_levels', 4, 'Number of coarsened graphs.')

# Directories.
flags.DEFINE_string('dir_data', os.path.join('..', 'data', 'mnist'), 'Directory to store data.')


def grid_graph(m, corners=False):
    z = graph.grid(m)
    dist, idx = graph.distance_sklearn_metrics(z, k=FLAGS.number_edges, metric=FLAGS.metric)
    A = graph.adjacency(dist, idx)

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        import scipy.sparse
        A = A.toarray()
        A[A < A.max()/1.5] = 0
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    print("{} > {} edges".format(A.nnz//2, FLAGS.number_edges*m**2//2))
    return A

t_start = time.process_time()
A = grid_graph(28, corners=False)
A = graph.replace_random_edges(A, 0)
L = graph.laplacian(A, normalized=True)
print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
# graph.plot_spectrum(L)
del A

# return a list of lists [coef of power 0, coef of power 1, ..., coef of power k-1]
# for the chebyshev polynomial.
# len(weight_power) == K
def chebyshev_list(K):
    coef_list = []
    for k in range(K):
        coef = Polynomial.cast(Chebyshev.basis(k))
        coef_list.append(coef.coef)
    return coef_list

class GCNN(BaseModel):
    def __init__(self, L, F, K, num_classes):
        super().__init__()
        # number of polynomial, hops: K,
        # number of filters: F,
        # Normalized Laplacian,
        self.L = L
        self.F = F
        self.K = K
        self.NCls = num_classes
        self.learning_rate = 0.0001
        self.dropout = 0.5
        self.regularization = 0.001

    def chebyshev(self, x, L, Fout, K):
        # Fout: num of output features
        # N: number of signal, batch size
        # V: number of vertices, graph size
        # Fin: number of features per signal
        N, V, Fin = x.get_shape()
        L = scipy.sparse.csr_matrix(L)
        # convert to a list of chebyshev matrix
        base_L = graph.rescale_L(L, lmax=2)
        coef_list = chebyshev_list(K)
        chebyshev_Ls = []
        for coef in coef_list:
            L = 0
            for i in range(len(coef)):
                L += coef[i] * (base_L**i)
            chebyshev_Ls.append(L)

        # convert to sparseTensor
        def convert2Sparse(L):
            L = L.tocoo()
            indices = np.column_stack((L.row, L.col))
            L = tf.SparseTensor(indices, L.data, L.shape)
            return tf.sparse_reorder(L)
        chebyshev_Ls = map(lambda L: convert2Sparse(L), chebyshev_Ls)

        # chebyshev filtering
        # N x V x Fin -> N x Fin x V -> Fin*N x V -> V x Fin*N
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.reshape(x, [-1, V])
        x = tf.transpose(x)

        x_filtered = []
        for T in chebyshev_Ls:
            # T: V x V, x: V x Fin*N, output: V x Fin*N
            x_filtered.append(tf.sparse_tensor_dense_matmul(T, x))
            # T: V x V, x: N x V x Fin, output: N x V x Fin
            # x_filtered.append(tf.map_fn(lambda x: tf.sparse_tensor_dense_matmul(T, x), x))

        # K x N x V x Fin
        # x = tf.stack(x_filtered)
        # x = tf.parallel_stack(x_filtered)

        # K x V x Fin*N -> K x V x Fin x N -> N x V x Fin x K
        x = tf.stack(x_filtered)
        x = tf.reshape(x, [K, V, Fin, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])

        # K x N x V x Fin -> N x V x Fin x K
        # x = tf.transpose(x, perm=[1, 2, 3, 0])
        x = tf.reshape(x, [-1, Fin*K])
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W) # N*V x Fout
        x = tf.nn.relu(x)
        return tf.reshape(x, [-1, V, Fout])

    def fc(self, x, Mout, relu=True):
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def _inference(self, x, dropout):
        N, V, Fin = x.get_shape()
        Fout = self.F # number of filters
        K = self.K # support size of filter
        with tf.variable_scope('gconv1'):
            x = self.chebyshev(x, L, Fout, K)
            x = tf.nn.dropout(x, dropout)

        with tf.variable_scope('logits1'):
            x = tf.reshape(x, [-1, V * Fout])
            y = self.fc(x, self.NCls, relu=False)
        return y

gcnn = GCNN(L, 10, 2, 10)
batch_size = 100
epochs = 10
prefetch_size = 20

train_data_copy = train_data.map(lambda x: tf.expand_dims(x, 1)).batch(batch_size).prefetch(prefetch_size)
train_labels_copy = train_labels.batch(batch_size).prefetch(prefetch_size)
train_data = train_data.map(lambda x: tf.expand_dims(x, 1)).repeat(epochs).batch(batch_size).prefetch(prefetch_size)
train_labels = train_labels.repeat(epochs).batch(batch_size).prefetch(prefetch_size)
val_data = val_data.map(lambda x: tf.expand_dims(x, 1)).batch(batch_size).prefetch(prefetch_size)
val_labels = val_labels.batch(batch_size).prefetch(prefetch_size)
test_data = test_data.map(lambda x: tf.expand_dims(x, 1)).batch(batch_size).prefetch(prefetch_size)
test_labels = test_labels.batch(batch_size).prefetch(prefetch_size)

data_format = (train_data.output_types, train_data.output_shapes)
labels_format = (train_labels.output_types, train_labels.output_shapes)
gcnn.build_graph(data_format, labels_format, tf.get_default_graph())
sess = gcnn.fit(train_data, train_labels,
                val_data, val_labels,
                train_total=55000*epochs, val_total=5000,
                batch_size=batch_size)
print(gcnn.evaluate(test_data, test_labels, total=10000, sess=sess))
print(gcnn.evaluate(train_data_copy, train_labels_copy, total=55000, sess=sess))
