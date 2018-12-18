import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from gcn import BaseModel
import graph
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
A = graph.replace_random_edges
L = [graph.laplacian(A, normalized=True) for A in graphs]
print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
graph.plot_spectrum(L)
del A

class gcnn(BaseModel):
    def __init__(self, L, F, K):
        super().__init__()
        # number of polynomial, hops: K,
        # number of filters: F,
        # Normalized Laplacian,
        self.L = graph.reshape_L(L, lmax=2)
        self.F = F
        self.K = K
        self.learning_rate = 0.0001
        self.dropout = 0.5
        self.regularization = 0.001

    def __inference(self, x, dropout):
        with tf.variable_scope('gconv1'):
            N, M = x.get_shape()
        pass
