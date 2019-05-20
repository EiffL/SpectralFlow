import tensorflow as tf

from .chebyshev import ChebyshevCoefficients, StochasticChebyshevTrace
from .ops import eigen_max


def chebyshev_spectral_sum(func, operator, a, b, shape, deg=20, m=100):
    """
    Computes the spectral sum tr(f(A)), where A is the operator,
    f is the func, [a, b] is the range of eigenvalues of A.

    shape: Shape of input vectors to the operator, typically [batch, d]
    deg: Degree of the Chebyshev approximation
    m: Number of random vectors to probe the trace

    This corresponds to Algorithm 1 in Han et al. 2017
    """

    # Compute the chebyshev coefficients of the operator
    c = ChebyshevCoefficients(func, a, b, deg)

    # Rescales the operator
    def scaled_op(x):
        return 2 * operator(x) / (b - a) - (b + a)/(b - a) * x

    Gamma = StochasticChebyshevTrace(scaled_op, shape=shape,
                                     coefficients=c, n_probes=m)

    return Gamma


def chebyshev_logdet(func, operator, shape, deg=20, m=100, num_iters=10,
                     g=1.1, eps=1e-4):
    """
    Estimates the log determinant of a positive definite matrix
    This corresponds to Algorithm 2 in Han et al. 2017
    """
    # Find the largest eigenvalue
    lmax = eigen_max(operator, shape, num_iters)
    a, b = eps, g*lmax

    # Rescales the operator
    def scaled_op(x):
        return operator(x) / (a+b)

    Gamma = chebyshev_spectral_sum(func, scaled_op, a / (a + b), b / (a+b),
                                   shape=shape, deg=deg, m=m)
    return Gamma + shape[-1]*tf.log(a+b)
