import numpy as np
import tensorflow as tf

tfco = tf.contrib.constrained_optimization

class TSP_Problem(tfco.ConstrainedMinimizationProblem):

    def __init__(self, map_weights, decision_vars, dummy_vars, epsilon=1e-4):
        # _map_weights[i, i] == 0
        # epsilon > 0
        decision_vars.shape.assert_has_rank(2)
        dummy_vars.shape.assert_has_rank(1)
        self._size = dummy_vars.shape.as_list()[0]
        assert(self._size ==
               decision_vars.shape.as_list()[0] ==
               decision_vars.shape.as_list()[1])
        self._map_weights = map_weights
        self._decision_vars = decision_vars
        self._dummy_vars = dummy_vars
        self._epsilon = epsilon

    @property
    def objective(self):
        return tf.reduce_sum(tf.multiply(self._map_weights, decision_vars))
        # return tf.constant(0.0)

    # @property
    # def constraints(self):
    #     # tf.rint()
    #     pass

    @property
    #def proxy_constraints(self):
    def proxy_constraints(self):
        # integer constraints
        positive_margin = 0.5 - tf.abs(self._decision_vars - 0.5) - self._epsilon
        negative_margin = -0.5 + tf.abs(self._decision_vars - 0.5) - self._epsilon
        # positive_margin = self._decision_vars - 1
        # negative_margin = -self._decision_vars
        positive_margin = tf.reshape(positive_margin, [-1])
        negative_margin = tf.reshape(negative_margin, [-1])

        # closed tour constraints, equalities
        lose_diagonal = self._decision_vars - tf.matrix_diag(tf.matrix_diag_part(self._decision_vars))
        predecessor = tf.reduce_sum(lose_diagonal, 0) - 1
        successor = tf.reduce_sum(lose_diagonal, 1) - 1
        n = self._size

        mtz = []
        # mtz, Miller-Tucker-Zemlin formulation
        # enforce single subtour
        for i in range(self._size):
            for j in range(self._size):
                if i != j:
                    mtz.append(dummy_vars[i] - dummy_vars[j] + n * lose_diagonal[i][j] - n + 1)

        mtz = tf.stack(mtz) # n x (n-1)
        return tf.concat([positive_margin, negative_margin, predecessor, successor, mtz], 0)
        # return tf.concat([predecessor, successor, mtz], 0)
        # return tf.concat([positive_margin, negative_margin], 0)

    @property
    def constraints(self):
        # don't how to do this, to keep in consistent, need to return this twice
        # decisions_at_negative = tf.less(self._decision_vars, 0.0 - self.epsilon)
        decisions_at_bound = 1 - tf.abs(
            tf.cast(tf.equal(self._decision_vars, 0.0), dtype=tf.float32) +
            tf.cast(tf.equal(self._decision_vars, 1.0), dtype=tf.float32))
        # decisions_at_bound *= 100
        # positive_margin = 0.5 - tf.abs(self._decision_vars - 0.5) - self._epsilon
        # negative_margin = -0.5 + tf.abs(self._decision_vars - 0.5) - self._epsilon
        # positive_margin = tf.reshape(positive_margin, [-1])
        # negative_margin = tf.reshape(negative_margin, [-1])

        # assume epsilon won't be too large s.t. two interval overlap
        # decisions_at_bound = tf.abs(
        #     tf.cast(tf.less(tf.abs(self._decision_vars - 0.0), self._epsilon), dtype=tf.float32) +
        #     tf.cast(tf.less(tf.abs(self._decision_vars - 1.0), self._epsilon), dtype=tf.float32))
        # decisions_at_negative = tf.less(self._decision_vars, 0.0, 'decisions_at_negative_side')
        decisions_at_bound = tf.reshape(decisions_at_bound, [-1])

        decision_vars = tf.rint(self._decision_vars)
        lose_diagonal = decision_vars - tf.matrix_diag(tf.matrix_diag_part(decision_vars))
        predecessor = tf.reduce_sum(lose_diagonal, 0) - 1
        successor = tf.reduce_sum(lose_diagonal, 1) - 1
        n = self._size

        mtz = []
        # mtz, Miller-Tucker-Zemlin formulation
        # enforce single subtour
        for i in range(n):
            for j in range(n):
                if i != j:
                    mtz.append(dummy_vars[i] - dummy_vars[j] + n * lose_diagonal[i][j] - n + 1)

        mtz = tf.stack(mtz) # n x (n-1)
        return tf.concat([decisions_at_bound, decisions_at_bound,
                          predecessor, successor, mtz], 0)
        # return tf.concat([positive_margin, negative_margin,
        #                   predecessor, successor, mtz], 0)
        # return tf.concat([predecessor, successor, mtz], 0)

if __name__ == '__main__':
    num_cities = 5
    map_weights = [[0.0,  3.0,  4.0,  2.0,  7.0],
                   [3.0,  0.0,  4.0,  6.0,  3.0],
                   [4.0,  4.0,  0.0,  5.0,  8.0],
                   [2.0,  6.0,  5.0,  0.0,  6.0],
                   [7.0,  3.0,  8.0,  6.0,  0.0]]
    map_weights = tf.constant(map_weights, dtype=tf.float32, shape=[num_cities, num_cities])
    decision_vars = tf.get_variable('decision_vars', shape=[num_cities, num_cities], initializer=tf.ones_initializer)
    dummy_vars = tf.get_variable('dummy_vars', shape=[num_cities])

    problem = TSP_Problem(map_weights, decision_vars, dummy_vars)
    # print(problem.constraints)
    # print(problem.proxy_constraints)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # print("number of constraints: %d" % problem.num_constraints)
    with tf.Session(config=config) as session:
        optimizer = tfco.MultiplicativeSwapRegretOptimizer(
        #optimizer = tfco.AdditiveSwapRegretOptimizer(
            optimizer=tf.train.AdagradOptimizer(learning_rate=1))
        train_op = optimizer.minimize(problem)

        session.run(tf.global_variables_initializer())
        for _ in range(100000):
            _, var, loss, proxy_loss, constraints_loss, dummy = \
                session.run([train_op, decision_vars,
                             problem.objective,
                             problem.proxy_constraints,
                             problem.constraints, dummy_vars])
            # print(proxy_loss)
            print(var)
            print(loss,
                  np.sum(proxy_loss < 0.0) / len(proxy_loss),
                  np.sum(constraints_loss < 0.0) / len(constraints_loss))
            # print(loss, proxy_loss)
            # print(loss, np.sum(constraints_loss < 0.0))

        decision_vars, dummy_vars = session.run([decision_vars, dummy_vars])

        print(decision_vars)
