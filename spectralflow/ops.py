import tensorflow as tf


def eigen_max(operator, shape, num_iters=10):
    """
    Compute maximum eigenvalue of the operator by the power method.
    shape: input shape for the probe operator, usually: [batch_size, d]
    num_iters: number of power method iterations to compute the maximum
        eigenval, at each call to the model, the eigenvector is cached.
    """
    def _l2normalize(v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2, axis=-1, keepdims=True) ** 0.5 + eps)

    def power_iteration(i, u_i):
        u_ip1 = _l2normalize(operator(u_i))
        return i + 1, u_ip1

    u = tf.get_variable("u", shape=shape,
                        initializer=tf.orthogonal_initializer(),
                        trainable=False)

    _, u_final = tf.while_loop(
        cond=lambda i, _1: i < num_iters,
        body=power_iteration,
        loop_vars=(tf.constant(0, dtype=tf.int32), u))

    with tf.control_dependencies([u.assign(u_final)]):
        result = tf.reduce_sum(u_final*operator(u_final), -1)

    return result
