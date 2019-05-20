import tensorflow as tf
import tensorflow_probability as tfp
import math


def ChebyshevCoefficients(func, a, b, n):
    """
    Returns Chebyshev coefficients up to order `n` for the function `func`
    defined in the interval [a, b]

    WARNING: Returns 2x the coefficient c0
    TODO: Figure out how to divide c0 by 2 :-/
    """
    bma = 0.5 * (b - a)
    bpa = 0.5 * (b + a)

    xk = tf.cos(math.pi * (tf.range(n, dtype=tf.float32) + 0.5)/(n + 1)) * bma + bpa
    f = func(xk)

    i = tf.range(n, dtype=tf.float32)
    xk, _ = tf.meshgrid(i, i)
    fac = 2.0 / (n+1)
    c = fac * tf.reduce_sum(f * tf.cos(
        tf.transpose(xk) * (math.pi * (xk + 0.5) / (n+1))),
                            axis=1)

    return c


def StochasticChebyshevTrace(operator, shape, coeffs, n_probe=100):
    """
    Computes the trace of the Chebyshev expansion of the function defined by
    `coeffs` and applied to `operator`, using the Hutchinson estimator

    operator: input operator
    shape: shape of input random vector to use, typically [batch_size, d]
    coeffs: Chebyshev coefficients of the function to evaluate
    n_probe: number of rademacher samples

    WARNING: Will divide coeffs[0] by two
    """
    def _chebyshev_recurrence(i, w1, w0, s):
        wi = 2*operator(w1) - w0
        s = s + coeffs[i]*wi
        return i+1, wi, w1, s

    # Sample a rademacher tensor with desired size
    v = tfp.math.random_rademacher([n_probe, ] + shape)

    # Initialize the iteration
    w0, w1 = v, operator(v)
    s = 0.5*coeffs[0]*w0 + coeffs[1]*w1

    _, _, _, final_s = tf.while_loop(
        cond=lambda i, _1, _2, _3: i < coeffs.shape[0],
        body=_chebyshev_recurrence,
        loop_vars=(tf.constant(2, dtype=tf.int32), w1, w0, s))

    r = tf.einsum('ijk,ijl->ij', v, final_s)

    return tf.reduce_mean(r, axis=0)
